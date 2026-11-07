"""Lichess rated standard games: download, parse, filter (Step 1, human play).

Ported from the Chess960 project's lichess960.py. Streams the monthly Lichess
database exports of rated standard games
(https://database.lichess.org/standard/, CC0) and writes one row per kept game,
with its moves, to a Parquet file per month. Nothing is decompressed to disk:
each file is streamed from the web (or read from a local .pgn.zst) through a
zstd decompressor and parsed line by line. Headers are filtered as text, so
only kept games carry their movetext forward.

Needs internet access (compute nodes, or the login node): the month is
streamed and filtered as it arrives, holding only the kept games in memory,
so nothing large is written to disk. Parsing runs at roughly 300 MB of PGN a
second on one core, so a ~100-million-game month takes tens of minutes and
the download usually sets the pace; a dropped connection restarts that
month. jobs/lichess_fetch.slurm streams several months in parallel, one
array task each. --workers parses blocks of games in parallel processes,
which only helps when many games are kept. Files already downloaded with
wget can be read instead with --pgn-dir.

Subcommands
-----------
fetch      Download and parse months, e.g. --months 2026-10. Writes
           results/lichess/raw/lichess_YYYY-MM[_TAG].parquet (finished months
           are skipped, so rerunning resumes).
summarise  Counts for scoping: games by category, rating band, time control,
           termination; how many games have clocks and Lichess evaluations.
sample     Write a small PGN file of kept games (e.g. 2,000) for checking the
           event extractor by eye.

Kept while parsing: rated standard games with a known result, in the chosen
time-control categories (by Lichess's estimated duration, base + 40 x
increment: blitz 180-479 s, rapid 480-1499 s, classical 1500 s and over;
bullet and correspondence are never kept), optionally above a mean rating
and thinned by a hash of the game id (--sample-frac, reproducible). The
remaining inclusion rules (bots, abandoned games, berserk, provisional
ratings, minimum plies) are columns applied later, so a flow table can show
what each removes.

One row per game:
  month, game_id, utc_datetime, event, tournament, white_id, black_id
  (salted SHA-1 of the usernames, 12 hex characters), white_elo, black_elo,
  white_rating_diff, black_rating_diff, white_title, black_title, result,
  score_white (1, 0.5, 0), time_control, base_s, increment_s, estimated_s,
  category, termination, eco, opening, plies, has_clock, has_eval,
  white_berserk, black_berserk, movetext (with clock and eval comments).

Pilot (high-rated classical games from one month):

    pip install zstandard
    python code/lichess_games.py fetch --months 2026-10 --categories classical \\
        --min-mean-rating 2000 --tag pilot
    python code/lichess_games.py summarise --tag pilot
    python code/lichess_games.py sample --tag pilot --n 2000
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import re
import shutil
import subprocess
import sys
import urllib.request
import zlib
from pathlib import Path
from typing import Iterator, TextIO

import pandas as pd

from utils import REPO_ROOT, setup_logging, write_parquet_atomic

log = logging.getLogger("kvb.lichess")

URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"
OUT_DIR = REPO_ROOT / "results" / "lichess"
SALT = "knights-vs-bishops"          # fixed, so IDs are stable across months
RESULT_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
CATEGORIES = ("blitz", "rapid", "classical")

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def month_range(spec: str) -> list[str]:
    """'2026-07:2026-09' -> ['2026-07', '2026-08', '2026-09']; '2026-09' -> ['2026-09']."""
    first, _, last = spec.partition(":")
    last = last or first
    y, m = map(int, first.split("-"))
    y1, m1 = map(int, last.split("-"))
    out = []
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def category(estimated_s: float) -> str:
    """Lichess speed categories from estimated duration (base + 40 x inc)."""
    if estimated_s < 30:
        return "ultrabullet"
    if estimated_s < 180:
        return "bullet"
    if estimated_s < 480:
        return "blitz"
    if estimated_s < 1500:
        return "rapid"
    return "classical"


def parse_time_control(tc: str) -> tuple[float, float] | None:
    if "+" not in tc:
        return None            # "-" = correspondence / unlimited
    base, inc = tc.split("+", 1)
    try:
        return float(base), float(inc)
    except ValueError:
        return None


def player_id(name: str) -> str:
    return hashlib.sha1(f"{SALT}:{name.lower()}".encode()).hexdigest()[:12]


def hash_fraction(game_id: str) -> float:
    """A reproducible number in [0, 1) from the game id, for thinning."""
    return zlib.crc32(game_id.encode()) / 2 ** 32


def to_int(value: str | None) -> float:
    try:
        return float(int(value))          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def raw_path(out_dir: Path, month: str, tag: str | None) -> Path:
    return out_dir / "raw" / f"lichess_{month}{'_' + tag if tag else ''}.parquet"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def iter_games(stream: TextIO) -> Iterator[tuple[dict[str, str], str]]:
    """(headers, movetext) for each game in a PGN text stream."""
    headers: dict[str, str] = {}
    moves: list[str] = []
    in_moves = False
    for line in stream:
        line = line.rstrip("\n")
        if line.startswith("["):
            if in_moves:                       # a new game starts
                yield headers, " ".join(moves)
                headers, moves, in_moves = {}, [], False
            m = HEADER_RE.match(line)
            if m:
                headers[m.group(1)] = m.group(2)
        elif line.strip():
            in_moves = True
            moves.append(line)
    if headers:
        yield headers, " ".join(moves)


def header_filter(h: dict[str, str], categories: set[str], min_mean_rating: float,
                  sample_frac: float) -> tuple[float, float, float, str] | None:
    """Cheap checks on headers alone; (base, inc, estimated, category) if kept."""
    event = h.get("Event", "")
    if not event.lower().startswith("rated"):
        return None
    if h.get("Variant", "Standard").lower() not in ("standard", ""):
        return None
    if h.get("Result", "*") not in RESULT_SCORE:
        return None
    tc = parse_time_control(h.get("TimeControl", "-"))
    if tc is None:
        return None
    base, inc = tc
    est = base + 40 * inc
    cat = category(est)
    if cat not in categories:
        return None
    if min_mean_rating > 0:
        we, be = to_int(h.get("WhiteElo")), to_int(h.get("BlackElo"))
        if not (we + be) / 2 >= min_mean_rating:       # NaN fails too
            return None
    if sample_frac < 1:
        gid = h.get("Site", "").rsplit("/", 1)[-1]
        if hash_fraction(gid) >= sample_frac:
            return None
    return base, inc, est, cat


def game_row(h: dict[str, str], movetext: str, month: str, kept: tuple[float, float, float, str]) -> dict:
    """One kept game as a row."""
    base, inc, est, cat = kept
    event = h.get("Event", "")
    clocks = CLK_RE.findall(movetext)
    secs = [int(a) * 3600 + int(b) * 60 + float(c) for a, b, c in clocks]
    result = h["Result"]
    return {
        "month": month,
        "game_id": h.get("Site", "").rsplit("/", 1)[-1],
        "utc_datetime": f"{h.get('UTCDate', '')} {h.get('UTCTime', '')}".strip(),
        "event": event,
        "tournament": ("tournament" in event.lower()) or ("swiss" in event.lower()),
        "white_id": player_id(h.get("White", "")),
        "black_id": player_id(h.get("Black", "")),
        "white_elo": to_int(h.get("WhiteElo")),
        "black_elo": to_int(h.get("BlackElo")),
        "white_rating_diff": to_int(h.get("WhiteRatingDiff")),
        "black_rating_diff": to_int(h.get("BlackRatingDiff")),
        "white_title": h.get("WhiteTitle", ""),
        "black_title": h.get("BlackTitle", ""),
        "result": result,
        "score_white": RESULT_SCORE[result],
        "time_control": h.get("TimeControl", ""),
        "base_s": base,
        "increment_s": inc,
        "estimated_s": est,
        "category": cat,
        "termination": h.get("Termination", ""),
        "eco": h.get("ECO", ""),
        "opening": h.get("Opening", ""),
        "plies": len(secs),
        "has_clock": bool(secs),
        "has_eval": "[%eval" in movetext,
        "white_berserk": bool(secs) and base > 0 and secs[0] <= base / 2 + 1,
        "black_berserk": len(secs) > 1 and base > 0 and secs[1] <= base / 2 + 1,
        "movetext": movetext,
    }


def open_zst_binary(source: str) -> tuple[io.BufferedIOBase, object]:
    """Decompressed byte stream of a .pgn.zst from a URL or local path. Uses
    the `zstandard` Python package if installed, else the `zstd` command."""
    try:
        import zstandard  # type: ignore
        raw = urllib.request.urlopen(source) if source.startswith("http") else open(source, "rb")
        return zstandard.ZstdDecompressor().stream_reader(raw, read_size=1 << 20), raw
    except ImportError:
        if not shutil.which("zstd"):
            raise SystemExit("need the zstandard package (pip install zstandard) or the zstd command")
        cmd = f"curl -sSfL '{source}' | zstd -dc" if source.startswith("http") else f"zstd -dc '{source}'"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        return proc.stdout, proc  # type: ignore[return-value]


def open_zst_stream(source: str) -> tuple[TextIO, object]:
    """Text stream of a .pgn.zst (see open_zst_binary)."""
    binary, handle = open_zst_binary(source)
    return io.TextIOWrapper(binary, encoding="utf-8", errors="replace"), handle


def iter_chunks(binary: io.BufferedIOBase, size: int = 1 << 24) -> Iterator[bytes]:
    """Blocks of whole games (~size bytes each), split where a line starts
    with '[Event ', so each block can be parsed on its own."""
    buf = b""
    while True:
        data = binary.read(size)
        if not data:
            if buf.strip():
                yield buf
            return
        buf += data
        cut = buf.rfind(b"\n[Event ")
        if cut > 0:
            yield buf[:cut + 1]
            buf = buf[cut + 1:]


def parse_stream(stream: TextIO, month: str, categories: set[str], min_mean_rating: float,
                 sample_frac: float, max_games: int = 0) -> tuple[pd.DataFrame, int]:
    """Serial parse of a text stream."""
    rows, seen = [], 0
    for h, mt in iter_games(stream):
        seen += 1
        kept = header_filter(h, categories, min_mean_rating, sample_frac)
        if kept is not None:
            rows.append(game_row(h, mt, month, kept))
            if max_games and len(rows) >= max_games:
                break
        if seen % 1_000_000 == 0:
            log.info("  %s: %d games read, %d kept", month, seen, len(rows))
    return pd.DataFrame(rows), seen


def _parse_chunk(args: tuple[bytes, str, set[str], float, float]) -> tuple[list[dict], int]:
    chunk, month, categories, min_mean_rating, sample_frac = args
    rows, seen = [], 0
    for h, mt in iter_games(io.StringIO(chunk.decode("utf-8", errors="replace"))):
        seen += 1
        kept = header_filter(h, categories, min_mean_rating, sample_frac)
        if kept is not None:
            rows.append(game_row(h, mt, month, kept))
    return rows, seen


def parse_parallel(binary: io.BufferedIOBase, month: str, categories: set[str],
                   min_mean_rating: float, sample_frac: float, workers: int,
                   max_games: int = 0, chunk_size: int = 1 << 24) -> tuple[pd.DataFrame, int]:
    """The reader splits the stream into blocks of whole games and worker
    processes parse them. At most 2 x workers blocks are in flight, so memory
    stays flat however large the month is; results keep the file's order."""
    from collections import deque
    from multiprocessing import Pool
    rows: list[dict] = []
    seen, next_log = 0, 1_000_000
    with Pool(workers) as pool:
        pending: deque = deque()
        chunks = iter_chunks(binary, chunk_size)
        done = False
        while True:
            while not done and len(pending) < 2 * workers:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    done = True
                    break
                pending.append(pool.apply_async(_parse_chunk, ((chunk, month, categories,
                                                                min_mean_rating, sample_frac),)))
            if not pending:
                break
            r, n = pending.popleft().get()
            rows.extend(r)
            seen += n
            if seen >= next_log:
                log.info("  %s: %d games read, %d kept", month, seen, len(rows))
                next_log += 1_000_000
            if max_games and len(rows) >= max_games:
                rows = rows[:max_games]
                break
    return pd.DataFrame(rows), seen


def fetch_month(month: str, out: Path, categories: set[str], min_mean_rating: float,
                sample_frac: float, local: str | None = None, max_games: int = 0,
                workers: int = 1) -> Path:
    source = local or URL.format(month=month)
    log.info("%s: reading %s (%d worker%s)", month, source, workers, "s" if workers > 1 else "")
    if workers > 1:
        binary, handle = open_zst_binary(source)
        df, seen = parse_parallel(binary, month, categories, min_mean_rating, sample_frac,
                                  workers, max_games)
    else:
        stream, handle = open_zst_stream(source)
        df, seen = parse_stream(stream, month, categories, min_mean_rating, sample_frac, max_games)
    if isinstance(handle, subprocess.Popen):
        handle.kill() if max_games else handle.wait()
    write_parquet_atomic(df, out)
    log.info("%s: %d games read, %d kept -> %s", month, seen, len(df), out)
    return out


def cmd_fetch(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    cats = set(args.categories.split(","))
    bad = cats - set(CATEGORIES)
    if bad:
        raise SystemExit(f"unknown categories {sorted(bad)}; choose from {CATEGORIES}")
    for month in month_range(args.months):
        dest = raw_path(out_dir, month, args.tag)
        if dest.exists() and not args.overwrite:
            log.info("%s: already done (%s)", month, dest)
            continue
        local = None
        if args.pgn_dir:
            cand = Path(args.pgn_dir) / Path(URL.format(month=month)).name
            if cand.exists():
                local = str(cand)
            else:
                log.warning("%s: %s not found, downloading instead", month, cand)
        fetch_month(month, dest, cats, args.min_mean_rating, args.sample_frac, local, args.max_games,
                    args.workers)
    return 0


# ---------------------------------------------------------------------------
# Summary and sample
# ---------------------------------------------------------------------------

BANDS = [0, 1200, 1500, 1800, 2000, 2200, 2400, 4000]
BAND_LABELS = ["<1200", "1200-1499", "1500-1799", "1800-1999", "2000-2199", "2200-2399", "2400+"]


def load_raw(out_dir: Path, tag: str | None, columns: list[str] | None = None) -> pd.DataFrame:
    pattern = f"lichess_*_{tag}.parquet" if tag else "lichess_????-??.parquet"
    files = sorted((out_dir / "raw").glob(pattern))
    if not files:
        raise SystemExit(f"no parsed months matching {pattern} under {out_dir / 'raw'}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)


def summary_text(g: pd.DataFrame) -> str:
    mean_r = (g.white_elo + g.black_elo) / 2
    bands = pd.cut(mean_r, BANDS, labels=BAND_LABELS, right=False)
    bot = (g.white_title == "BOT") | (g.black_title == "BOT")
    L = ["Lichess rated standard games: what we've got", "=" * 44, ""]
    L.append(f"Games: {len(g):,} from {g.month.nunique()} month(s); players: "
             f"{pd.concat([g.white_id, g.black_id]).nunique():,}")
    L += ["", "By category and mean-rating band:",
          pd.crosstab(bands, g.category).to_string()]
    L += ["", "Most common time controls:", g.time_control.value_counts().head(10).to_string()]
    L += ["", "Terminations: " + ", ".join(f"{k} {v:,}" for k, v in g.termination.value_counts().items())]
    L.append("Results: White {:.1f}%, draw {:.1f}%, Black {:.1f}%".format(
        100 * (g.score_white == 1).mean(), 100 * (g.score_white == 0.5).mean(),
        100 * (g.score_white == 0).mean()))
    L.append(f"With clocks: {100 * g.has_clock.mean():.1f}%; with Lichess evaluations: "
             f"{100 * g.has_eval.mean():.1f}%; bot games: {bot.sum():,}; "
             f"berserk (either side): {(g.white_berserk | g.black_berserk).sum():,}")
    L.append("Plies: " + ", ".join(f"p{q}: {g.plies.quantile(q / 100):.0f}" for q in (5, 25, 50, 75, 95)))
    return "\n".join(L) + "\n"


def cmd_summarise(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    g = load_raw(out_dir, args.tag, columns=[c for c in (
        "month", "white_id", "black_id", "white_elo", "black_elo", "white_title", "black_title",
        "category", "time_control", "termination", "score_white", "has_clock", "has_eval",
        "white_berserk", "black_berserk", "plies")])
    text = summary_text(g)
    (out_dir / f"lichess_summary{'_' + args.tag if args.tag else ''}.txt").write_text(text)
    print(text)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """A small PGN of kept games (salted ids as player names) for checking by eye."""
    out_dir = Path(args.output_dir)
    g = load_raw(out_dir, args.tag)
    g = g.assign(h=g.game_id.map(hash_fraction)).sort_values("h").head(args.n)
    dest = out_dir / f"lichess_sample{'_' + args.tag if args.tag else ''}.pgn"
    with open(dest, "w") as f:
        for r in g.itertuples():
            f.write(f'[Event "{r.event}"]\n[Site "https://lichess.org/{r.game_id}"]\n'
                    f'[White "{r.white_id}"]\n[Black "{r.black_id}"]\n[Result "{r.result}"]\n'
                    f'[WhiteElo "{r.white_elo:.0f}"]\n[BlackElo "{r.black_elo:.0f}"]\n'
                    f'[TimeControl "{r.time_control}"]\n[Termination "{r.termination}"]\n\n'
                    f"{r.movetext}\n\n")
    log.info("wrote %d games to %s", len(g), dest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", default=str(OUT_DIR))
    common.add_argument("--tag", help="suffix for a separate run, e.g. pilot")
    common.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", parents=[common], help="download and parse months")
    f.add_argument("--months", required=True, help="YYYY-MM or YYYY-MM:YYYY-MM")
    f.add_argument("--categories", default="blitz,rapid,classical",
                   help="comma-separated: blitz, rapid, classical")
    f.add_argument("--min-mean-rating", type=float, default=0,
                   help="keep games whose mean rating is at least this (default: no limit)")
    f.add_argument("--sample-frac", type=float, default=1.0,
                   help="keep this fraction of games, chosen by a hash of the game id")
    f.add_argument("--max-games", type=int, default=0, help="stop after this many kept games (testing)")
    f.add_argument("--workers", type=int, default=1,
                   help="processes parsing blocks of games in parallel (the stream itself is read by one)")
    f.add_argument("--overwrite", action="store_true")
    f.add_argument("--pgn-dir", help="read lichess_db_standard_rated_YYYY-MM.pgn.zst from this "
                   "folder when present (e.g. downloaded with wget), instead of streaming")
    f.set_defaults(func=cmd_fetch)
    s = sub.add_parser("summarise", parents=[common], help="summary counts of the parsed games")
    s.set_defaults(func=cmd_summarise)
    a = sub.add_parser("sample", parents=[common], help="write a small PGN of kept games")
    a.add_argument("--n", type=int, default=2000)
    a.set_defaults(func=cmd_sample)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
