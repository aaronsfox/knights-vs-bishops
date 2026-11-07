"""Bishop and knight events in human games (Step 1).

Replays each game kept by lichess_games.py with python-chess and writes
tables of what happens to the minor pieces. Clock times and any Lichess
evaluations in the movetext are attached to each event.

Tables (results/lichess/events/<table>_<month>[_TAG].parquet):

pieces        One row per original minor piece (Nb1, Ng1, Bc1, Bf1 and Black's
              Nb8, Ng8, Bc8, Bf8): the ply it was captured (empty if it
              survived), what captured it, and the bishop's square colour.
exchanges     One row per level minor-for-minor exchange: a run of consecutive
              captures after which each side has lost exactly one minor piece
              and nothing else differs (pawns lost equal, no rook or queen
              difference, no promotion). Kind: "B for N" (one side gave a
              bishop for a knight; `bishop_giver` says who), or "B for B" /
              "N for N" (same-type controls). Records who started it, whether
              it broke a bishop pair, the minors and pawns before, game phase,
              clocks and evaluations before and after.
opportunities One row per ply on which the side to move could capture a
              defended knight with a bishop (BxN) or a defended bishop with a
              knight (NxB), i.e. offer that exchange, and what it played.
spells        Material spells starting on a quiet ply and lasting at least
              MIN_SPELL plies: "minor imbalance" (equal numbers of minors of
              different types, same rooks and queens; pawns may differ, see
              pawn_diff) and "pawn up" (identical pieces, one side a pawn up).
              Each records the design (bishop side first, e.g. "BB v BN"),
              which colour has more bishops (or the extra pawn), when it began
              and ended, and the context at its start.
endgames      The first quiet ply (held for 2 plies) at which the game is a
              minor-piece endgame: no queens, one minor each, and equal rooks
              (0 or 1 each). Type (B v N, B v B same or opposite colours,
              N v N), pawns, pawn features, clocks and evaluation.
games         One row per game: plies replayed and any parse error.

Results come from the games table of lichess_games.py, joined on game_id.

    python code/human_events.py run --tag pilot --workers 8
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator

import chess
import pandas as pd

from utils import REPO_ROOT, setup_logging, write_parquet_atomic

log = logging.getLogger("kvb.events")

OUT_DIR = REPO_ROOT / "results" / "lichess"
TABLES = ("pieces", "exchanges", "opportunities", "spells", "endgames", "games")
MIN_SPELL = 4                      # plies a spell must last (two moves each)
MATE_CP = 10_000                   # evaluations: mate scores as +/- this

ORIGINS = {
    chess.B1: (chess.WHITE, "N", ""), chess.G1: (chess.WHITE, "N", ""),
    chess.C1: (chess.WHITE, "B", "dark"), chess.F1: (chess.WHITE, "B", "light"),
    chess.B8: (chess.BLACK, "N", ""), chess.G8: (chess.BLACK, "N", ""),
    chess.C8: (chess.BLACK, "B", "light"), chess.F8: (chess.BLACK, "B", "dark"),
}
SYMBOL = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B", chess.ROOK: "R",
          chess.QUEEN: "Q", chess.KING: "K"}
COLOUR = {chess.WHITE: "white", chess.BLACK: "black"}

TOKEN_RE = re.compile(r"\{([^}]*)\}|(\S+)")
MOVE_NUMBER_RE = re.compile(r"^\d+\.(\.\.)?$")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
CLK_RE = re.compile(r"\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]")
EVAL_RE = re.compile(r"\[%eval (#?)(-?[\d.]+)\]")


# ---------------------------------------------------------------------------
# Movetext
# ---------------------------------------------------------------------------

def parse_movetext(movetext: str) -> list[tuple[str, float | None, float | None]]:
    """[(san, clock_s, eval_cp_white)] per ply, from Lichess movetext with
    { [%clk h:mm:ss] [%eval x] } comments. Mates become +/- MATE_CP."""
    out: list[list[Any]] = []
    for m in TOKEN_RE.finditer(movetext):
        comment, tok = m.group(1), m.group(2)
        if comment is not None:
            if out:
                c = CLK_RE.search(comment)
                if c:
                    out[-1][1] = int(c.group(1)) * 3600 + int(c.group(2)) * 60 + float(c.group(3))
                e = EVAL_RE.search(comment)
                if e:
                    v = float(e.group(2))
                    out[-1][2] = (MATE_CP if v > 0 or e.group(2) == "0" else -MATE_CP) if e.group(1) \
                        else 100 * v
            continue
        if MOVE_NUMBER_RE.match(tok) or tok in RESULTS or tok.startswith("$"):
            continue
        tok = re.sub(r"^\d+\.+", "", tok)          # "1.e4" written without a space
        tok = tok.rstrip("?!")
        if tok:
            out.append([tok, None, None])
    return [tuple(x) for x in out]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def counts(board: chess.Board) -> dict[str, int]:
    """Piece counts, e.g. wB, bN, wP."""
    out = {}
    for colour, c in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            out[c + SYMBOL[pt]] = chess.popcount(board.pieces_mask(pt, colour))
    return out


def phase(c: dict[str, int]) -> int:
    """Non-pawn material in the usual 3/3/5/9 units (62 at the start)."""
    return sum(3 * (c[s + "N"] + c[s + "B"]) + 5 * c[s + "R"] + 9 * c[s + "Q"] for s in "wb")


def side_minors(c: dict[str, int], s: str) -> str:
    return "B" * c[s + "B"] + "N" * c[s + "N"]


def minor_design(c: dict[str, int]) -> tuple[str, str] | None:
    """('BB v BN', 'white') when the minors differ in type but not number and
    rooks and queens match; the bishop-rich side first."""
    if c["wB"] + c["wN"] != c["bB"] + c["bN"] or c["wB"] == c["bB"]:
        return None
    if c["wR"] != c["bR"] or c["wQ"] != c["bQ"]:
        return None
    s = "w" if c["wB"] > c["bB"] else "b"
    o = "b" if s == "w" else "w"
    return f"{side_minors(c, s)} v {side_minors(c, o)}", COLOUR[s == "w"]


def pawn_up(c: dict[str, int]) -> str | None:
    """'white' or 'black' when all pieces match and one side is a pawn up."""
    if any(c["w" + p] != c["b" + p] for p in "NBRQ"):
        return None
    d = c["wP"] - c["bP"]
    return "white" if d == 1 else "black" if d == -1 else None


def pawn_features(board: chess.Board, side: chess.Color) -> dict[str, Any]:
    """Pawn structure from `side`'s view."""
    own = board.pieces(chess.PAWN, side)
    opp = board.pieces(chess.PAWN, not side)
    files = {chess.square_file(s) for s in own | opp}
    passed = 0
    for sq in own:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = [s for s in opp if abs(chess.square_file(s) - f) <= 1 and
                 (chess.square_rank(s) > r if side == chess.WHITE else chess.square_rank(s) < r)]
        passed += not ahead
    return {
        "pawns_both_wings": any(f <= 2 for f in files) and any(f >= 5 for f in files),
        "pawn_files": len(files),
        "passed_pawns": passed,
        "passed_pawns_opp": sum(
            not [s for s in own if abs(chess.square_file(s) - chess.square_file(q)) <= 1 and
                 (chess.square_rank(s) < chess.square_rank(q) if side == chess.WHITE
                  else chess.square_rank(s) > chess.square_rank(q))]
            for q in opp),
    }


def bishop_on_own_pawn_colour(board: chess.Board, side: chess.Color) -> float | None:
    """Share of `side`'s pawns on its bishop's square colour (one bishop only)."""
    bs = list(board.pieces(chess.BISHOP, side))
    pawns = list(board.pieces(chess.PAWN, side))
    if len(bs) != 1 or not pawns:
        return None
    colour = (chess.square_file(bs[0]) + chess.square_rank(bs[0])) % 2
    return sum((chess.square_file(p) + chess.square_rank(p)) % 2 == colour for p in pawns) / len(pawns)


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def game_events(game_id: str, movetext: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {t: [] for t in TABLES}
    plies = parse_movetext(movetext)
    board = chess.Board()
    ids = {sq: f"{'w' if c else 'b'}{p}{chess.square_name(sq)}" for sq, (c, p, _) in ORIGINS.items()}
    pieces = {ids[sq]: {"game_id": game_id, "piece_id": ids[sq], "colour": COLOUR[c], "piece": p,
                        "origin": chess.square_name(sq), "bishop_colour": bc,
                        "captured_ply": None, "captured_by": None}
              for sq, (c, p, bc) in ORIGINS.items()}
    where = {sq: ids[sq] for sq in ORIGINS}            # square -> original minor on it

    clock = {chess.WHITE: None, chess.BLACK: None}
    last_eval: float | None = None
    before = counts(board)
    run: dict[str, Any] | None = None                    # capture run in progress
    spell: dict[str, Any] | None = None
    endgame_candidate: dict[str, Any] | None = None
    endgame_done = False
    error = ""
    prev_counts = before

    def context(c: dict[str, int]) -> dict[str, Any]:
        return {"phase": phase(c), "pawns_white": c["wP"], "pawns_black": c["bP"],
                "queens_on": c["wQ"] + c["bQ"] > 0, "clock_white": clock[chess.WHITE],
                "clock_black": clock[chess.BLACK], "eval_cp": last_eval}

    def close_spell(end_ply: int, how: str) -> None:
        nonlocal spell
        if spell is not None:
            spell["end_ply"], spell["ended_by"] = end_ply, how
            spell["length"] = end_ply - spell["start_ply"] + 1
            if spell["length"] >= MIN_SPELL:
                out["spells"].append(spell)
        spell = None

    for i, (san, clk, ev) in enumerate(plies, start=1):
        mover = board.turn
        # Exchange opportunities before the move.
        bxn = nxb = False
        for mv in board.generate_legal_captures():
            pt, victim = board.piece_type_at(mv.from_square), board.piece_type_at(mv.to_square)
            if victim is None or not board.is_attacked_by(not mover, mv.to_square):
                continue
            bxn |= pt == chess.BISHOP and victim == chess.KNIGHT
            nxb |= pt == chess.KNIGHT and victim == chess.BISHOP
        try:
            move = board.parse_san(san)
        except ValueError as e:
            error = f"ply {i} {san}: {e}"
            break
        is_capture = board.is_capture(move)
        mover_pt = board.piece_type_at(move.from_square)
        victim_pt = board.piece_type_at(move.to_square) if is_capture and not board.is_en_passant(move) else None
        if bxn or nxb:
            took = ("BxN" if mover_pt == chess.BISHOP and victim_pt == chess.KNIGHT else
                    "NxB" if mover_pt == chess.KNIGHT and victim_pt == chess.BISHOP else
                    "other capture" if is_capture else "quiet")
            out["opportunities"].append({"game_id": game_id, "ply": i, "side": COLOUR[mover],
                                         "bxn_available": bxn, "nxb_available": nxb, "played": took,
                                         **{f"minors_{k}": v for k, v in (
                                             ("white", side_minors(prev_counts, "w")),
                                             ("black", side_minors(prev_counts, "b")))},
                                         **context(prev_counts)})
        # Capture runs.
        if is_capture and run is None:
            run = {"start": i, "initiator": mover, "before": prev_counts, "promotion": False,
                   "ctx": context(prev_counts)}
        if run is not None and move.promotion:
            run["promotion"] = True
        # Original minors: captured or moved.
        if is_capture and move.to_square in where and not board.is_en_passant(move):
            pid = where.pop(move.to_square)
            pieces[pid]["captured_ply"], pieces[pid]["captured_by"] = i, SYMBOL[mover_pt]
        if move.from_square in where:
            where[move.to_square] = where.pop(move.from_square)
        board.push(move)
        if clk is not None:
            clock[mover] = clk
        if ev is not None:
            last_eval = ev
        c = counts(board)
        # A capture run ends at the first non-capture after it, or here if the
        # next ply isn't a capture (decided on the next iteration / at the end).
        next_is_capture = False
        if i < len(plies) and run is not None:
            try:
                nxt = board.parse_san(plies[i][0])
                next_is_capture = board.is_capture(nxt)
            except ValueError:
                next_is_capture = False
        if run is not None and not next_is_capture:
            record_exchange(out, game_id, run, c, i, last_eval)
            run = None
        # Spells (start only on quiet plies).
        des = minor_design(c)
        pu = pawn_up(c)
        key = ("minor imbalance", des) if des else ("pawn up", pu) if pu else None
        if spell is not None and (key is None or (spell["kind"], spell["_key"]) != key):
            close_spell(i - 1, "capture" if is_capture else "other")
        if spell is None and key is not None and not is_capture and not board.is_check():
            kind, k = key
            design, focal = (k if kind == "minor imbalance" else ("pawn up", k))
            s = "w" if focal == "white" else "b"
            o = "b" if s == "w" else "w"
            spell = {"game_id": game_id, "kind": kind, "_key": k, "design": design,
                     "focal": focal, "start_ply": i, "pawn_diff": c[s + "P"] - c[o + "P"],
                     "rooks_each": c["wR"], **context(c),
                     **pawn_features(board, chess.WHITE if focal == "white" else chess.BLACK)}
        # Endgame reached (held for 2 plies).
        if not endgame_done:
            eg = endgame_type(board, c)
            if eg and not is_capture:
                if endgame_candidate and endgame_candidate["type"] == eg[0]:
                    out["endgames"].append(endgame_candidate)
                    endgame_done = True
                else:
                    endgame_candidate = {"game_id": game_id, "type": eg[0], "focal": eg[1],
                                         "ply": i, "rooks_each": c["wR"],
                                         "pawn_diff": (c["wP"] - c["bP"]) * (1 if eg[1] != "black" else -1),
                                         **context(c), **pawn_features(
                                             board, chess.BLACK if eg[1] == "black" else chess.WHITE),
                                         "bishop_pawn_colour_share": bishop_on_own_pawn_colour(
                                             board, chess.BLACK if eg[1] == "black" else chess.WHITE)
                                         if eg[0] == "B v N" else None}
            elif not eg:
                endgame_candidate = None
        prev_counts = c
    else:
        if run is not None:
            record_exchange(out, game_id, run, counts(board), len(plies), last_eval)
    close_spell(len(plies), "game end" if not error else "parse error")
    for s in out["spells"]:
        s.pop("_key", None)
    total = len(plies) if not error else int(error.split()[1])
    for p in pieces.values():
        p["plies"] = total
        p["survived"] = p["captured_ply"] is None
    out["pieces"] = list(pieces.values())
    out["games"] = [{"game_id": game_id, "plies_replayed": total if not error else total - 1,
                     "parse_error": error}]
    return out


def endgame_type(board: chess.Board, c: dict[str, int]) -> tuple[str, str] | None:
    """('B v N', 'white') etc. for a minor-piece endgame; focal side = the
    bishop side in B v N, White otherwise."""
    if c["wQ"] or c["bQ"] or c["wR"] != c["bR"] or c["wR"] > 1:
        return None
    if c["wB"] + c["wN"] != 1 or c["bB"] + c["bN"] != 1:
        return None
    if c["wB"] and c["bN"]:
        return "B v N", "white"
    if c["bB"] and c["wN"]:
        return "B v N", "black"
    if c["wN"]:
        return "N v N", "white"
    wb = next(iter(board.pieces(chess.BISHOP, chess.WHITE)))
    bb = next(iter(board.pieces(chess.BISHOP, chess.BLACK)))
    same = (chess.square_file(wb) + chess.square_rank(wb)) % 2 == (chess.square_file(bb) + chess.square_rank(bb)) % 2
    return ("B v B same colour" if same else "B v B opposite colour"), "white"


def record_exchange(out: dict[str, list[dict]], game_id: str, run: dict[str, Any],
                    after: dict[str, int], end_ply: int, eval_after: float | None) -> None:
    """Keep the run if it was a level minor-for-minor exchange."""
    b = run["before"]
    lost = {k: b[k] - after[k] for k in b}
    if run["promotion"]:
        return
    if lost["wB"] + lost["wN"] != 1 or lost["bB"] + lost["bN"] != 1:
        return
    if any(lost[s + p] for s in "wb" for p in "RQ") or lost["wP"] != lost["bP"]:
        return
    w, k = ("B" if lost["wB"] else "N"), ("B" if lost["bB"] else "N")
    if w == k:
        kind, giver = f"{w} for {k}", None
    else:
        kind, giver = "B for N", "white" if w == "B" else "black"
    gs = None if giver is None else ("w" if giver == "white" else "b")
    out["exchanges"].append({
        "game_id": game_id, "ply": run["start"], "end_ply": end_ply, "kind": kind,
        "bishop_giver": giver, "initiator": COLOUR[run["initiator"]],
        "initiator_gave_bishop": None if giver is None else COLOUR[run["initiator"]] == giver,
        "pair_broken": None if giver is None else b[gs + "B"] == 2,
        "minors_white": side_minors(b, "w"), "minors_black": side_minors(b, "b"),
        "pawns_lost_each": lost["wP"], **run["ctx"], "eval_cp_after": eval_after,
    })


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def _work(batch: list[tuple[str, str]]) -> dict[str, list[dict]]:
    acc: dict[str, list[dict]] = {t: [] for t in TABLES}
    for gid, mt in batch:
        try:
            ev = game_events(gid, mt)
        except Exception as e:  # noqa: BLE001 (one bad game must not stop a month)
            ev = {t: [] for t in TABLES}
            ev["games"] = [{"game_id": gid, "plies_replayed": 0, "parse_error": f"crash: {e}"}]
        for t in TABLES:
            acc[t].extend(ev[t])
    return acc


def batches(df: pd.DataFrame, size: int) -> Iterator[list[tuple[str, str]]]:
    pairs = list(zip(df.game_id, df.movetext))
    for i in range(0, len(pairs), size):
        yield pairs[i:i + size]


def process(df: pd.DataFrame, workers: int = 1, batch_size: int = 500) -> dict[str, pd.DataFrame]:
    acc: dict[str, list[dict]] = {t: [] for t in TABLES}
    if workers > 1:
        with Pool(workers) as pool:
            for res in pool.imap(_work, batches(df, batch_size)):
                for t in TABLES:
                    acc[t].extend(res[t])
    else:
        for b in batches(df, batch_size):
            res = _work(b)
            for t in TABLES:
                acc[t].extend(res[t])
    return {t: pd.DataFrame(acc[t]) for t in TABLES}


def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.output_dir)
    pattern = f"lichess_*_{args.tag}.parquet" if args.tag else "lichess_????-??.parquet"
    files = sorted((out_dir / "raw").glob(pattern))
    if not files:
        log.error("no parsed months matching %s under %s", pattern, out_dir / "raw")
        return 1
    for f in files:
        stem = f.stem.replace("lichess_", "")
        dest = {t: out_dir / "events" / f"{t}_{stem}.parquet" for t in TABLES}
        if all(p.exists() for p in dest.values()) and not args.overwrite:
            log.info("%s: already done", stem)
            continue
        df = pd.read_parquet(f, columns=["game_id", "movetext"])
        if args.max_games:
            df = df.head(args.max_games)
        log.info("%s: %d games, %d workers", stem, len(df), args.workers)
        tables = process(df, args.workers)
        for t, p in dest.items():
            write_parquet_atomic(tables[t], p)
        errs = (tables["games"].parse_error != "").sum()
        log.info("%s: %s; %d games with parse errors", stem,
                 ", ".join(f"{t} {len(v):,}" for t, v in tables.items()), errs)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="extract events from parsed months")
    r.add_argument("--output-dir", default=str(OUT_DIR))
    r.add_argument("--tag", help="the --tag used with lichess_games.py, e.g. pilot")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--max-games", type=int, default=0, help="first N games per month (testing)")
    r.add_argument("--overwrite", action="store_true")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
