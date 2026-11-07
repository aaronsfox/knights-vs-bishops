"""Engine games from the study's starting positions, played with fastchess.

Two kinds of match, both starting every game from one of the position's
opening-book lines, each line played ``games_per_line`` times with colours
swapped in pairs (50 lines x 10 games = 500 games per position by default):

* **Self-play** (Step 3, the removal matrix): two copies of one player,
  ``<player>_a`` and ``<player>_b``. Swapping the copies does not move the
  material, so colour balance comes from each position's colour-flipped
  twin (positions.py); the analysis pools the two.
* **Player vs opponent** (Step 2): e.g. a manipulated Stockfish 11 build
  against the base build, from the standard position. Here swapping colours
  does balance the variant's colour.

A player is an engine folder under engines/ or a named player from
``[rungs]`` in config/selfplay.toml (an engine with a fixed node budget per
move, its own time control or extra UCI options; none are defined by default).

Subcommands
-----------
run      Play the games for a set of positions. One fastchess match per
         position, written to raw/<match>/<position>.pgn.gz once complete
         (<match> is the player, or "<player>__vs__<opponent>"). Finished
         positions are skipped, so rerunning the same command resumes.
         ``--batch`` adds a further set of games (e.g. topping up the key
         comparisons) in <position>.<batch>.pgn.gz.
collect  Parse every finished PGN into tables:
           tables/selfplay_games.parquet      one row per game
           tables/selfplay_positions.parquet  per match and position: W/D/L,
                                              White's expected score and Elo
           tables/selfplay_lines.parquet      the same per book line

Examples
--------
    python code/selfplay.py run --player stockfish --positions start,c1-b8 --concurrency 16
    python code/selfplay.py run --player sf11_b050 --opponent sf11 --positions start
    python code/selfplay.py run --player stockfish --positions key --batch b2 --games-per-line 20
    python code/selfplay.py collect
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import chess
import chess.pgn
import pandas as pd

from utils import (
    CONFIG_DIR, REPO_ROOT, EngineConfig, EngineSetupError, all_positions, elo_from_expected,
    get_position, load_engine_config, load_toml, parse_position_spec, platform_info,
    position_id_from_fen, setup_logging, sha256_file, utc_now, verify_engine_files,
    write_parquet_atomic,
)

log = logging.getLogger("kvb.selfplay")

DEFAULT_CONFIG = CONFIG_DIR / "selfplay.toml"
RESULT_WHITE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
VS = "__vs__"


# ---------------------------------------------------------------------------
# Settings, players and paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchSettings:
    tc: str
    games_per_line: int
    fastchess: Path
    draw: dict[str, Any]
    resign: dict[str, Any]
    maxmoves: int
    timemargin_ms: int
    tablebases: str
    tbpieces: int
    uci_overrides: dict[str, Any]

    @classmethod
    def from_config(cls, config: dict[str, Any], games_per_line: int | None = None
                    ) -> "MatchSettings":
        m = config["match"]
        games = int(games_per_line or m["games_per_line"])
        if games < 2 or games % 2:
            raise ValueError("games_per_line must be even and at least 2")
        path = Path(m["fastchess"])
        return cls(
            tc=str(m["tc"]), games_per_line=games,
            fastchess=path if path.is_absolute() else REPO_ROOT / path,
            draw=dict(m.get("draw", {})), resign=dict(m.get("resign", {})),
            maxmoves=int(m.get("maxmoves", 0)),
            timemargin_ms=int(m.get("timemargin_ms", 0)), tablebases=str(m.get("tablebases", "")),
            tbpieces=int(m.get("tbpieces", 5)),
            uci_overrides=dict(config.get("uci_overrides", {})),
        )


@dataclass(frozen=True)
class Player:
    """An engine as it plays: the engine's manifest plus, for a
    named player, an optional fixed node budget per move, its own
    time control and extra UCI options (e.g. UCI_LimitStrength/UCI_Elo)."""

    key: str                  # engine folder or named player; used in file names and PGNs
    cfg: EngineConfig
    nodes: int = 0            # 0 = no node limit
    tc: str | None = None     # None = the match time control
    options: tuple[tuple[str, Any], ...] = ()

    @property
    def display_name(self) -> str:
        extra = [f"{self.nodes:,} nodes"] if self.nodes else []
        extra += [f"{k}={v}" for k, v in self.options if k != "UCI_LimitStrength"]
        return f"{self.cfg.display_name} ({', '.join(extra)})" if extra else self.cfg.display_name


def load_player(config: dict[str, Any], key: str) -> Player:
    """A named player from [rungs] in the config, or an engine folder."""
    rung = config.get("rungs", {}).get(key)
    if rung is not None:
        return Player(key, load_engine_config(rung["engine"]), int(rung.get("nodes", 0)),
                      rung.get("tc"), tuple(dict(rung.get("options", {})).items()))
    return Player(key, load_engine_config(key))


def match_key(player: str, opponent: str | None) -> str:
    return f"{player}{VS}{opponent}" if opponent else player


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    path = Path(override) if override else REPO_ROOT / config["output"]["dir"]
    return path if path.is_absolute() else REPO_ROOT / path


def book_path(config: dict[str, Any]) -> Path:
    path = Path(config["book"]["lines"])
    return path if path.is_absolute() else REPO_ROOT / path


def raw_path(out_dir: Path, match: str, pos_id: str, batch: str = "") -> Path:
    name = f"{pos_id}.{batch}.pgn.gz" if batch else f"{pos_id}.pgn.gz"
    return out_dir / "raw" / match / name


def parse_raw_name(path: Path) -> tuple[str, str]:
    """(position id, batch) from a raw file name."""
    stem = path.name[: -len(".pgn.gz")]
    pos_id, _, batch = stem.partition(".")
    return pos_id, batch


def load_book(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"opening book not found at {path} "
                                "(run opening_book_generator.py collect)")
    cols = ["position_id", "line_index", "start_fen", "moves_uci", "moves_san"]
    return pd.read_parquet(path, columns=cols).sort_values(["position_id", "line_index"])


# ---------------------------------------------------------------------------
# Running fastchess
# ---------------------------------------------------------------------------

def write_openings(lines: pd.DataFrame, path: Path) -> None:
    """One position's book lines as a PGN, in line_index order: fastchess plays
    them in this order (round r uses line (r - 1) mod n)."""
    with open(path, "w") as fh:
        for row in lines.itertuples():
            board = chess.Board(row.start_fen)
            game = chess.pgn.Game()
            game.setup(board)
            game.headers["PositionId"] = str(row.position_id)
            game.headers["LineIndex"] = str(row.line_index)
            node = game
            for uci in row.moves_uci.split():
                node = node.add_variation(chess.Move.from_uci(uci))
            print(game, file=fh, end="\n\n")


def engine_args(p: Player, label: str, s: MatchSettings) -> list[str]:
    cfg = p.cfg
    opts = {**cfg.uci_options, **s.uci_overrides, **dict(p.options)}
    if cfg.network is not None and cfg.network_option:
        opts[cfg.network_option] = str(cfg.network.resolve())
    args = ["-engine", f"cmd={cfg.binary.resolve()}", f"name={label}",
            f"dir={cfg.binary.parent.resolve()}"]
    if p.nodes:
        args.append(f"nodes={p.nodes}")
    if p.tc:
        args.append(f"tc={p.tc}")
    for name, value in opts.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        args.append(f"option.{name}={value}")
    return args


def labels(player: Player, opponent: Player | None) -> tuple[str, str]:
    """PGN names: two copies for self-play, the keys otherwise."""
    if opponent is None:
        return f"{player.key}_a", f"{player.key}_b"
    return player.key, opponent.key


def fastchess_command(player: Player, opponent: Player | None, s: MatchSettings,
                      openings: Path, pgn: Path, n_lines: int, concurrency: int,
                      log_file: Path | None = None) -> list[str]:
    """Every opening is played twice per round with colours swapped (-repeat)."""
    rounds = n_lines * s.games_per_line // 2
    a, b = labels(player, opponent)
    cmd = [str(s.fastchess), "-variant", "standard"]
    cmd += engine_args(player, a, s) + engine_args(opponent or player, b, s)
    cmd += ["-each", f"tc={s.tc}", "proto=uci"]
    if s.timemargin_ms:
        cmd += [f"timemargin={s.timemargin_ms}"]
    cmd += [
            "-openings", f"file={openings}", "format=pgn", "order=sequential",
            "-rounds", str(rounds), "-repeat", "-concurrency", str(concurrency),
            "-recover", "-ratinginterval", "0", "-autosaveinterval", "0",
            "-pgnout", f"file={pgn}", "notation=san", "nodes=true", "seldepth=true"]
    if s.draw:
        cmd += ["-draw"] + [f"{k}={v}" for k, v in s.draw.items()]
    if s.resign:
        cmd += ["-resign"] + [f"{k}={str(v).lower()}" for k, v in s.resign.items()]
    if s.maxmoves:
        cmd += ["-maxmoves", str(s.maxmoves)]
    if s.tablebases:
        cmd += ["-tb", s.tablebases, "-tbpieces", str(s.tbpieces)]
    if log_file is not None:
        cmd += ["-log", f"file={log_file}", "level=info"]
    return cmd


def fastchess_version(binary: Path) -> str:
    try:
        out = subprocess.run([str(binary), "-version"], capture_output=True, text=True,
                             timeout=30)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unknown ({exc})"


def count_games(pgn: Path) -> int:
    if not pgn.exists():
        return 0
    with open(pgn, errors="replace") as fh:
        return sum(1 for line in fh if line.startswith("[Result "))


def play_position(player: Player, opponent: Player | None, s: MatchSettings,
                  lines: pd.DataFrame, dest: Path, concurrency: int, work_root: Path,
                  progress_seconds: float = 300) -> dict[str, Any]:
    """Play one position's match and write it to ``dest`` (gzip) atomically."""
    pos_id = str(lines.position_id.iloc[0])
    n_lines = len(lines)
    expected = n_lines * s.games_per_line
    work = Path(tempfile.mkdtemp(prefix=f"{pos_id}_", dir=work_root))
    try:
        openings, pgn = work / "openings.pgn", work / "games.pgn"
        write_openings(lines, openings)
        cmd = fastchess_command(player, opponent, s, openings, pgn, n_lines, concurrency,
                                log_file=work / "fastchess.log")
        started = time.monotonic()
        # fastchess writes config.json in its working directory: keep it in work/.
        out_file = work / "output.txt"
        with open(out_file, "w") as out:
            proc = subprocess.Popen(cmd, cwd=work, stdout=out, stderr=subprocess.STDOUT)
            # Report progress every few minutes so a long match is visible in
            # the job log (games finished so far, from the PGN).
            while True:
                try:
                    proc.wait(timeout=progress_seconds)
                    break
                except subprocess.TimeoutExpired:
                    done = count_games(pgn)
                    elapsed = time.monotonic() - started
                    eta = elapsed / done * (expected - done) if done else math.nan
                    log.info("  %s: %d/%d games after %.0f min (about %.0f min to go)",
                             pos_id, done, expected, elapsed / 60, eta / 60)
        seconds = time.monotonic() - started
        output = out_file.read_text(errors="replace")
        games = count_games(pgn)
        if proc.returncode != 0 or games != expected:
            tail = "\n".join(output.strip().splitlines()[-20:])
            status = (f"crashed (signal {-proc.returncode})" if proc.returncode < 0
                      else f"exit {proc.returncode}")
            # Keep fastchess's files (log, openings, config.json, any games)
            # for diagnosis; logs/ is not committed.
            keep = REPO_ROOT / "logs" / "selfplay_failed" / f"{dest.parent.name}_{work.name}"
            shutil.copytree(work, keep, dirs_exist_ok=True)
            (keep / "command.txt").write_text(" ".join(cmd) + "\n")
            raise RuntimeError(f"fastchess {status}, {games}/{expected} games "
                               f"(files kept in {keep})\n{tail}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        with open(pgn, "rb") as src, gzip.open(tmp, "wb", compresslevel=6) as out:
            shutil.copyfileobj(src, out)
        os.replace(tmp, dest)
        return {"games": games, "seconds": round(seconds, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def player_meta(p: Player, allow_unlocked: bool) -> dict[str, Any]:
    fingerprint = verify_engine_files(p.cfg, strict=not allow_unlocked)
    return {"key": p.key, "engine": p.cfg.key, "engine_name": p.cfg.name,
            "engine_version": p.cfg.version, "display_name": p.display_name,
            "nodes": p.nodes, "tc": p.tc, "options": dict(p.options), **fingerprint}


def cmd_run(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    s = MatchSettings.from_config(config, args.games_per_line)
    out_dir = output_dir(config, args.output_dir)
    if not s.fastchess.exists():
        log.error("fastchess not found at %s (run tools/fastchess/build_linux.sh)", s.fastchess)
        return 2
    try:
        player = load_player(config, args.player)
        opponent = load_player(config, args.opponent) if args.opponent else None
        metas = [player_meta(p, args.allow_unlocked) for p in (player, opponent) if p]
    except EngineSetupError as exc:
        log.error("%s", exc)
        return 2

    match = match_key(player.key, opponent.key if opponent else None)
    book = load_book(book_path(config))
    pos_ids = parse_position_spec(args.positions)
    todo = [p for p in pos_ids
            if args.overwrite or not raw_path(out_dir, match, p, args.batch).exists()]
    concurrency = args.concurrency or len(os.sched_getaffinity(0))
    log.info("%s | tc %s | %d games per line | concurrency %d | %d positions "
             "(%d already done)%s", " vs ".join(p.display_name for p in (player, opponent) if p),
             s.tc, s.games_per_line, concurrency, len(todo), len(pos_ids) - len(todo),
             f" | batch {args.batch}" if args.batch else "")
    if not todo:
        return 0

    run_id = uuid.uuid4().hex[:12]
    book_file = book_path(config)
    meta = {
        "run_id": run_id, "match": match, "batch": args.batch, "players": metas,
        "uci_overrides": s.uci_overrides,
        "fastchess_version": fastchess_version(s.fastchess),
        "fastchess_sha256": sha256_file(s.fastchess),
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in dataclasses.asdict(s).items()},
        "book": (str(book_file.relative_to(REPO_ROOT)) if book_file.is_relative_to(REPO_ROOT)
                 else str(book_file)),
        "book_sha256": sha256_file(book_file),
        "concurrency": concurrency, "positions": todo, "started": utc_now(),
        "command_example": fastchess_command(player, opponent, s, Path("openings.pgn"),
                                             Path("games.pgn"), 50, concurrency),
        **platform_info(),
    }
    meta_path = out_dir / "runs" / f"{run_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    work_root = Path(args.work_dir or os.environ.get("TMPDIR") or tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    finished: dict[str, dict[str, Any]] = {}
    for n, pos_id in enumerate(todo, 1):
        lines = book[book.position_id == pos_id]
        if lines.empty:
            log.error("%s: no book lines; skipping", pos_id)
            failed.append(pos_id)
            continue
        dest = raw_path(out_dir, match, pos_id, args.batch)
        try:
            info = play_position(player, opponent, s, lines, dest, concurrency, work_root,
                                 progress_seconds=60 * args.progress_minutes)
        except RuntimeError as exc:
            log.error("%s failed: %s", pos_id, exc)
            failed.append(pos_id)
            continue
        finished[pos_id] = info
        summary = summarise_pgn(dest)
        log.info("[%d/%d] %s %s: %d games in %.0fs | W %d D %d L %d | White E %.3f",
                 n, len(todo), pos_id, get_position(pos_id).design, info["games"],
                 info["seconds"], summary["white_wins"], summary["draws"],
                 summary["black_wins"], summary["expected_score"])

    meta.update({"finished": utc_now(), "position_results": finished, "failed": failed})
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    if failed:
        log.error("%d position(s) failed: %s (rerun to retry)", len(failed), ",".join(failed))
        return 1
    log.info("done")
    return 0


# ---------------------------------------------------------------------------
# Parsing games
# ---------------------------------------------------------------------------

# Engine move comments look like "+0.28/13 0.249s, n=110878, sd=21, nps=441745"
# (score in pawns from the mover's side, or +M5 / -M3 for mates, then depth).
SCORE_RE = re.compile(r"^(?P<score>[+-]?(?:M\d+|\d+(?:\.\d+)?))/(?P<depth>\d+)\s+(?P<time>[\d.]+)s")
KV_RE = re.compile(r"\b(n|sd|nps)=(\d+)")


def parse_move_comment(comment: str) -> dict[str, Any] | None:
    """Score (cp, mover's view), mate, depth, time and nodes from an engine
    move comment; None for book moves or comments without a score."""
    comment = comment.strip()
    m = SCORE_RE.match(comment)
    if not m:
        return None
    score = m.group("score")
    cp: int | None
    mate: int | None
    if "M" in score:
        n = int(score.lstrip("+-M"))
        mate, cp = (-n if score.startswith("-") else n), None
    else:
        mate, cp = None, int(round(float(score) * 100))
    kv = dict(KV_RE.findall(comment))
    return {"cp": cp, "mate": mate, "depth": int(m.group("depth")),
            "time": float(m.group("time")),
            "nodes": int(kv["n"]) if "n" in kv else None,
            "seldepth": int(kv["sd"]) if "sd" in kv else None}


OVERRUN_RE = re.compile(r"\s*\((\d+)ms overrun\)")


def end_reason(comment: str) -> str | None:
    """The reason fastchess writes into the last move's comment, e.g.
    'Draw by adjudication' or 'White wins by adjudication: SyzygyTB'. A loss
    on time's overrun ('White loses on time (12ms overrun)') is dropped here
    and kept separately by time_overrun_ms."""
    parts = [p.strip() for p in comment.split(",")]
    for part in reversed(parts):
        if part and not SCORE_RE.match(part) and not KV_RE.match(part) and part != "book":
            return OVERRUN_RE.sub("", part)
    return None


def time_overrun_ms(comment: str) -> float:
    m = OVERRUN_RE.search(comment)
    return float(m.group(1)) if m else math.nan


def iter_games(path: Path) -> Iterator[chess.pgn.Game]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            yield game


def player_engine(label: str) -> str:
    """The player key behind a PGN name ("stockfish_a" -> "stockfish")."""
    return label[:-2] if label.endswith(("_a", "_b")) else label


@dataclass(frozen=True)
class FileInfo:
    match: str
    position_id: str
    batch: str = ""

    @property
    def player(self) -> str:
        return self.match.split(VS)[0]

    @property
    def opponent(self) -> str:
        return self.match.split(VS)[1] if VS in self.match else ""


def game_row(game: chess.pgn.Game, info: FileInfo, book_moves: dict[int, str]
             ) -> dict[str, Any]:
    """One game as a row (full python-chess parse; used to check the fast
    parser). ``book_moves`` maps line_index -> moves_uci for the game's
    position (book lines are matched by round, then checked)."""
    h = game.headers
    played_book: list[str] = []
    played: list[str] = []
    stats = {chess.WHITE: [], chess.BLACK: []}
    last_comment = ""
    plies = 0
    first_eval_white: int | None = None
    for node in game.mainline():
        mover = not node.board().turn  # side that just moved
        comment = node.comment or ""
        if comment.strip().startswith("book"):
            played_book.append(node.move.uci())
        else:
            played.append(node.parent.board().san(node.move))
            parsed = parse_move_comment(comment)
            if parsed:
                stats[mover].append(parsed)
                if first_eval_white is None and parsed["cp"] is not None:
                    first_eval_white = parsed["cp"] if mover == chess.WHITE else -parsed["cp"]
        last_comment = comment
        plies += 1
    return _row(dict(h), info, book_moves, played_book, stats, first_eval_white,
                last_comment, plies, played)


# Fast parsing: fastchess's PGNs are machine-written and every move carries a
# comment, so games can be read from the text without replaying the moves
# (python-chess's full parser checks every move's legality and is ~50x slower).
HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$', re.M)
MOVE_COMMENT_RE = re.compile(r"([^\s{}]+)\s*\{([^}]*)\}")
MOVE_NUMBER_RE = re.compile(r"^\d+\.+$")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def split_games(text: str) -> Iterator[tuple[dict[str, str], str]]:
    """(headers, movetext) for each game in a PGN string."""
    for chunk in re.split(r"\n(?=\[Event )", text.strip()):
        head_end = chunk.find("\n\n")
        if head_end < 0:
            continue
        headers = dict(HEADER_RE.findall(chunk[:head_end]))
        yield headers, chunk[head_end:]


def fast_game_row(headers: dict[str, str], movetext: str, info: FileInfo,
                  book_moves_san: dict[int, str]) -> dict[str, Any]:
    """The same row as game_row, from the PGN text. Book lines are matched on
    SAN (``book_moves_san``: line_index -> moves_san)."""
    without_comments = re.sub(r"\{[^}]*\}", " ", movetext)
    sans = [tok for tok in without_comments.split()
            if not MOVE_NUMBER_RE.match(tok) and tok not in RESULTS]
    pairs = MOVE_COMMENT_RE.findall(movetext)
    played_book: list[str] = []
    played: list[str] = []
    stats: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    first_eval_white: int | None = None
    # Every starting position has White to move, so plies alternate W, B, ...
    for ply, (san, comment) in enumerate(pairs):
        white = ply % 2 == 0
        if comment.strip().startswith("book"):
            played_book.append(san)
            continue
        played.append(san)
        parsed = parse_move_comment(comment)
        if parsed:
            stats[white].append(parsed)
            if first_eval_white is None and parsed["cp"] is not None:
                first_eval_white = parsed["cp"] if white else -parsed["cp"]
    last_comment = pairs[-1][1] if pairs else ""
    return _row(headers, info, book_moves_san, played_book, stats, first_eval_white,
                last_comment, len(sans), played)


def _row(h: dict[str, str], info: FileInfo, book_moves: dict[int, str],
         played_book: list[str], stats: dict[bool, list[dict[str, Any]]],
         first_eval_white: int | None, last_comment: str, plies: int,
         played: list[str]) -> dict[str, Any]:
    rnd = int(h.get("Round", "0") or 0)
    n_lines = len(book_moves)
    line_index = (rnd - 1) % n_lines if n_lines and rnd > 0 else -1
    book_str = " ".join(played_book)
    line_match = book_moves.get(line_index) == book_str
    if not line_match:
        matches = [i for i, mv in book_moves.items() if mv == book_str]
        line_index = matches[0] if matches else -1
    result = h.get("Result", "*")
    fen_position = position_id_from_fen(h.get("FEN", ""))

    def mean(side: bool, key: str) -> float:
        vals = [x[key] for x in stats[side] if x[key] is not None]
        return float(sum(vals) / len(vals)) if vals else math.nan

    white, black = h.get("White", ""), h.get("Black", "")
    return {
        "match": info.match,
        "player": info.player,
        "opponent": info.opponent,
        "batch": info.batch,
        "position_id": info.position_id,
        "fen_matches_position": fen_position == info.position_id,
        "line_index": line_index,
        "line_matched_by_round": line_match,
        "round": rnd,
        "white_player": white,
        "black_player": black,
        "white_engine": player_engine(white),
        "black_engine": player_engine(black),
        "result": result,
        "score_white": RESULT_WHITE.get(result, math.nan),
        "termination": h.get("Termination", ""),
        "end_reason": end_reason(last_comment) or "",
        "time_overrun_ms": time_overrun_ms(last_comment),
        "plies": plies,
        "book_plies": len(played_book),
        # Identifies repeated games: a deterministic player (e.g. a fixed node
        # budget) replays the same game every time it gets the same line and colour.
        "moves_hash": zlib.crc32(" ".join(played).encode()),
        "first_eval_white_cp": first_eval_white,
        "mean_depth_white": mean(True, "depth"),
        "mean_depth_black": mean(False, "depth"),
        "mean_nodes_white": mean(True, "nodes"),
        "mean_nodes_black": mean(False, "nodes"),
        "time_used_white": sum(x["time"] for x in stats[True]),
        "time_used_black": sum(x["time"] for x in stats[False]),
        "duration": h.get("GameDuration", ""),
        "start_time": h.get("GameStartTime", ""),
        "time_control": h.get("TimeControl", ""),
        "engine_id_white": h.get("EngineWhiteName", ""),
    }


def parse_file(path: Path, book_moves_san: dict[int, str]) -> list[dict[str, Any]]:
    pos_id, batch = parse_raw_name(path)
    info = FileInfo(path.parent.name, pos_id, batch)
    with gzip.open(path, "rt", errors="replace") as fh:
        text = fh.read()
    return [fast_game_row(h, mt, info, book_moves_san) for h, mt in split_games(text)]


def _parse_job(job: tuple[Path, dict[int, str]]) -> list[dict[str, Any]]:
    return parse_file(*job)


def summarise_pgn(path: Path) -> dict[str, Any]:
    results = [g.headers.get("Result", "*") for g in iter_games(path)]
    w, d, l = results.count("1-0"), results.count("1/2-1/2"), results.count("0-1")
    n = w + d + l
    return {"white_wins": w, "draws": d, "black_wins": l,
            "expected_score": (w + d / 2) / n if n else math.nan}


def score_summary(games: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """W/D/L, White's expected score with a standard error, and Elo, by group.

    The standard error uses the per-game score variance; games are not quite
    independent (pairs share an opening line), so line-level summaries are
    also written for a cluster-robust analysis."""
    g = games.dropna(subset=["score_white"]).groupby(keys, sort=False)
    out = g.agg(games=("score_white", "size"),
                white_wins=("score_white", lambda x: int((x == 1).sum())),
                draws=("score_white", lambda x: int((x == 0.5).sum())),
                black_wins=("score_white", lambda x: int((x == 0).sum())),
                expected_score=("score_white", "mean"),
                score_sd=("score_white", "std"),
                mean_plies=("plies", "mean")).reset_index()
    out["draw_rate"] = out["draws"] / out["games"]
    out["se"] = out["score_sd"] / out["games"].pow(0.5)
    out["elo"] = out["expected_score"].map(elo_from_expected)
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    book = load_book(book_path(config))
    book_by_pos = {pid: dict(zip(grp.line_index, grp.moves_san))
                   for pid, grp in book.groupby("position_id")}
    files = sorted((out_dir / "raw").glob("*/*.pgn.gz"))
    if not files:
        log.error("no PGNs under %s", out_dir / "raw")
        return 1
    jobs = [(path, book_by_pos.get(parse_raw_name(path)[0], {})) for path in files]
    workers = args.workers or min(len(os.sched_getaffinity(0)), 16)
    log.info("parsing %d files with %d worker(s)", len(files), workers)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            for n, file_rows in enumerate(pool.map(_parse_job, jobs, chunksize=4), 1):
                rows.extend(file_rows)
                if n % 100 == 0:
                    log.info("parsed %d/%d files", n, len(files))
    else:
        for n, job in enumerate(jobs, 1):
            rows.extend(_parse_job(job))
            if n % 100 == 0:
                log.info("parsed %d/%d files", n, len(files))
    log.info("parsed %d games in %.0fs", len(rows), time.monotonic() - started)
    games = pd.DataFrame(rows)
    order = {pid: i for i, pid in enumerate(p.id for p in all_positions())}
    games = (games.assign(_order=games.position_id.map(order))
             .sort_values(["match", "_order", "batch", "round"], kind="stable")
             .drop(columns="_order").reset_index(drop=True))
    bad = games[(games.line_index < 0) | ~games.fen_matches_position]
    if len(bad):
        log.warning("%d games could not be matched to their position and book line", len(bad))
    unexpected = games[~games.termination.isin(["normal", "adjudication"])]
    if len(unexpected):
        log.warning("%d games ended abnormally: %s", len(unexpected),
                    unexpected.termination.value_counts().to_dict())

    tables = out_dir / "tables"
    write_parquet_atomic(games, tables / "selfplay_games.parquet")
    positions = score_summary(games, ["match", "position_id"])
    positions["design"] = positions.position_id.map(lambda i: get_position(i).design)
    write_parquet_atomic(positions, tables / "selfplay_positions.parquet")
    positions.to_csv(tables / "selfplay_positions.csv", index=False)
    lines = score_summary(games, ["match", "position_id", "batch", "line_index"])
    write_parquet_atomic(lines, tables / "selfplay_lines.parquet")

    for match, grp in positions.groupby("match"):
        log.info("%s: %d positions, %d games | draws %.1f%% | mean %.0f plies", match, len(grp),
                 int(grp.games.sum()), 100 * grp.draws.sum() / grp.games.sum(),
                 grp.mean_plies.mean())
    log.info("wrote %s", tables)
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--output-dir", help="override [output].dir")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # --config, --output-dir and -v also work after the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--output-dir", default=argparse.SUPPRESS, help="override [output].dir")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)

    r = sub.add_parser("run", parents=[common], help="play the games for some positions")
    r.add_argument("--player", required=True,
                   help="engine folder under engines/ or a named player in [rungs]")
    r.add_argument("--opponent", help="play against this player instead of self-play")
    r.add_argument("--positions", default="removal",
                   help="ids or groups, e.g. removal, key, start,c1-b8, or @file")
    r.add_argument("--games-per-line", type=int, help="override [match].games_per_line")
    r.add_argument("--batch", default="", help="label for an extra set of games, e.g. b2")
    r.add_argument("--concurrency", type=int, default=0,
                   help="games at once (default: the CPUs available to this process)")
    r.add_argument("--work-dir", help="scratch folder for fastchess (default $TMPDIR)")
    r.add_argument("--progress-minutes", type=float, default=5,
                   help="log games finished every this many minutes (default 5)")
    r.add_argument("--overwrite", action="store_true", help="replay finished positions")
    r.add_argument("--allow-unlocked", action="store_true",
                   help="allow an engine with no lock entry (testing only)")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("collect", parents=[common], help="parse finished games into tables")
    c.add_argument("--workers", type=int, default=0,
                   help="parallel processes (default: CPUs available, at most 16)")
    c.set_defaults(func=cmd_collect)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    if getattr(args, "batch", "") and "." in args.batch:
        log.error("--batch must not contain '.'")
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
