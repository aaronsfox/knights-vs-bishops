"""Board diagrams of the study's starting positions (or any FEN) for figures.

Draws a grid of labelled boards (file and rank coordinates, one caption per
board) in a black-and-white diagram style like chess books (or --style
colour), optionally with the engine's best move as an arrow and its evaluation
in the caption, read from the evaluate_sp tables.

Examples
--------
    python code/plot_boards.py --positions b1-c1-c8-f8,b1-g1-b8-c8,b1-g1-c8-f8 --engine none \\
        --labels "BN v NN" "BB v BN" "BB v NN" --out results/figures/boards_key
    python code/plot_boards.py --positions start,c1-b8 --engine stockfish --cols 2
    python code/plot_boards.py --fen "8/8/8/8/8/P7/5k2/K1B5 w - - 0 1" --engine none \\
        --labels "Wrong bishop" --out results/figures/wrong_bishop

Writes <out>.pdf (for the manuscript) and <out>.png (for viewing).
"""

from __future__ import annotations

import argparse
import io
import string
import sys
from pathlib import Path

import chess
import chess.svg
import pandas as pd

from utils import REPO_ROOT, elo_from_expected, get_position, parse_position_spec

DEFAULT_TABLE = REPO_ROOT / "results" / "evaluate_sp" / "tables" / "evaluate_sp_final.parquet"

# Two styles:
#   diagram  black-and-white, as in chess books and training material: white
#            light squares, diagonally hatched dark squares, a thin frame, and
#            coordinates on the left and bottom (the default)
#   colour   muted tan squares with coordinates on all sides
STYLES = {
    "diagram": {"colors": {"square light": "#ffffff", "square dark": "url(#hatch)"},
                "arrow": "#222222b3"},
    "colour": {"colors": {"square light": "#ece6d8", "square dark": "#b4a58a",
                          "margin": "#ffffff", "coord": "#333333"},
               "arrow": "#1f6f5fcc"},
}

# Diagonal hatching for the dark squares in the diagram style (board drawn at
# 45 px per square by python-chess).
HATCH_DEFS = (
    '<defs><pattern id="hatch" patternUnits="userSpaceOnUse" width="5" height="5" '
    'patternTransform="rotate(45)"><rect width="5" height="5" fill="#ffffff"/>'
    '<line x1="0" y1="0" x2="0" y2="5" stroke="#5e5e5e" stroke-width="1.5"/>'
    '</pattern></defs>'
)


DEFAULT_BOOK = REPO_ROOT / "results" / "opening_book" / "book" / "opening_book_lines.parquet"


def book_line_items(spec: str, book_path: Path) -> list[dict]:
    """Boards after opening-book lines, from 'position:line_index,...'
    (e.g. 'start:3,c1-b8:0')."""
    book = pd.read_parquet(book_path, columns=["position_id", "line_index", "moves_uci"])
    items = []
    for token in spec.split(","):
        pos_s, line_s = token.strip().split(":")
        pos_id, line = get_position(pos_s).id, int(line_s)
        row = book[(book.position_id == pos_id) & (book.line_index == line)]
        if row.empty:
            raise SystemExit(f"no book line {line} for position {pos_id}")
        board = get_position(pos_id).board()
        start = board.copy()
        moves = [chess.Move.from_uci(u) for u in row.iloc[0].moves_uci.split()]
        for mv in moves:
            board.push(mv)
        items.append({"pos_id": pos_id, "board": board, "line": start.variation_san(moves)})
    return items


def line_caption(item: dict, label: str | None) -> str:
    """'BN v NN (b1-c1-c8-f8)' (or the label), then the moves, e.g. '1. e4 c5 2. Nc3 a6 3. g3 d6'."""
    head = label or get_position(item["pos_id"]).label
    return f"{head}\n{item['line']}"


def board_png(board: chess.Board, arrow: chess.Move | None, size: int, dpi_scale: float,
              style: str = "diagram") -> bytes:
    import cairosvg

    spec = STYLES[style]
    arrows = []
    if arrow is not None:
        to_sq = arrow.to_square
        # Chess960 castling is stored as king-takes-rook; draw it to the king's
        # destination square instead so the arrow reads as castling.
        if board.is_castling(arrow):
            kingside = board.is_kingside_castling(arrow)
            to_sq = chess.square(6 if kingside else 2, chess.square_rank(arrow.from_square))
        arrows.append(chess.svg.Arrow(arrow.from_square, to_sq, color=spec["arrow"]))
    # The diagram style draws its own frame and left/bottom coordinates in
    # matplotlib, so the board itself has none.
    svg = chess.svg.board(board, size=size, coordinates=(style == "colour"), arrows=arrows,
                          colors=spec["colors"])
    if style == "diagram":
        head_end = svg.index(">", svg.index("<svg")) + 1
        svg = svg[:head_end] + HATCH_DEFS + svg[head_end:]
    return cairosvg.svg2png(bytestring=svg.encode(), scale=dpi_scale)


def ordered_positions(spec: str) -> list[str]:
    """Position ids in the order given (parse_position_spec uses registry
    order). Accepts ids, group names (expanded in registry order) or "@file"."""
    if spec.startswith("@"):
        return parse_position_spec(spec)
    ids: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if token:
            ids.extend(i for i in parse_position_spec(token) if i not in ids)
    return ids


def evaluation_lookup(table: Path, engine: str) -> pd.DataFrame | None:
    if not table.exists():
        return None
    f = pd.read_parquet(table)
    f = f[(f.engine == engine) & (f.multipv_setting == 1) & (f.multipv_index == 1)]
    if f.empty:
        return None
    top = f.sort_values("budget_nodes").groupby("position_id").tail(1).set_index("position_id")
    top["rank"] = top["expected_score"].rank(ascending=False, method="min").astype(int)
    return top


def caption_for(pos_id: str | None, label: str | None, ev: pd.DataFrame | None,
                engine: str) -> str:
    if pos_id is None:          # a board from --fen
        return label or ""
    pos = get_position(pos_id)
    head = label or pos.label
    if ev is not None and pos_id in ev.index:
        r = ev.loc[pos_id]
        e = "" if pd.isna(r.expected_score) else f", E = {r.expected_score:.3f}"
        head += f"\n{engine.capitalize()}: {int(r.cp_white):+d} cp{e}"
    return head


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--positions", help="ids or groups, e.g. start,c1-b8 or @file")
    src.add_argument("--lines", help="opening-book lines as position:line_index, e.g. "
                     "start:14,c1-b8:0; each board shows the position after the line")
    src.add_argument("--fen", nargs="+", help="arbitrary positions as FENs (e.g. endgames)")
    p.add_argument("--labels", nargs="*", help="one short label per board (optional)")
    p.add_argument("--book", default=str(DEFAULT_BOOK), help="opening_book_lines.parquet")
    p.add_argument("--engine", default="stockfish", help="engine for arrow and caption ('none' to omit)")
    p.add_argument("--table", default=str(DEFAULT_TABLE), help="evaluate_sp_final.parquet")
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--out", default=str(REPO_ROOT / "results" / "figures" / "boards"))
    p.add_argument("--no-arrows", action="store_true")
    p.add_argument("--no-letters", action="store_true", help="omit the (a), (b), ... panel letters")
    p.add_argument("--style", choices=sorted(STYLES), default="diagram",
                   help="diagram: black-and-white book style (default); colour: tan squares")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import image as mpimg
    from matplotlib.patches import Rectangle

    if args.lines:
        items = book_line_items(args.lines, Path(args.book))
        ev = None
    elif args.fen:
        items = [{"pos_id": None, "board": chess.Board(f), "line": None} for f in args.fen]
        ev = None
    else:
        ids = ordered_positions(args.positions)
        ev = None if args.engine == "none" else evaluation_lookup(Path(args.table), args.engine)
        items = [{"pos_id": i, "board": get_position(i).board(), "line": None} for i in ids]
    labels = args.labels or [None] * len(items)
    if len(labels) != len(items):
        p.error("--labels needs one label per board")
    ids = [it["pos_id"] for it in items]

    cols = min(args.cols, len(ids))
    rows = -(-len(ids) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 3.25 * rows), squeeze=False)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= len(ids):
            continue
        pos_id = ids[i]
        board = items[i]["board"]
        arrow = None
        if ev is not None and not args.no_arrows and pos_id is not None and pos_id in ev.index:
            arrow = chess.Move.from_uci(ev.loc[pos_id, "bestmove"])
        png = board_png(board, arrow, size=400, dpi_scale=2.0, style=args.style)
        img = mpimg.imread(io.BytesIO(png), format="png")
        letter = string.ascii_lowercase[i]
        if args.style == "diagram":
            # Board in data units 0-8, a thin frame, coordinates left and bottom.
            ax.imshow(img, extent=(0, 8, 0, 8), interpolation="lanczos")
            ax.add_patch(Rectangle((0, 0), 8, 8, fill=False, lw=0.9, ec="black"))
            for k in range(8):
                ax.text(k + 0.5, -0.32, "abcdefgh"[k], ha="center", va="center", fontsize=7.5)
                ax.text(-0.32, k + 0.5, str(k + 1), ha="center", va="center", fontsize=7.5)
            ax.set_xlim(-0.65, 8.05)
            ax.set_ylim(-0.65, 8.05)
            ax.set_aspect("equal")
            if not args.no_letters:
                ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold", pad=2)
            ax.text(0.5, -0.04, (line_caption(items[i], labels[i]) if items[i]['line'] else caption_for(pos_id, labels[i], ev, args.engine)),
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.2, linespacing=1.35)
        else:
            ax.imshow(img)
            if not args.no_letters:
                ax.set_title(f"({letter})", loc="left", fontsize=9, fontweight="bold", pad=2)
            ax.text(0.5, -0.03, (line_caption(items[i], labels[i]) if items[i]['line'] else caption_for(pos_id, labels[i], ev, args.engine)),
                    transform=ax.transAxes, ha="center", va="top", fontsize=7.2, linespacing=1.35)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08, wspace=0.06, hspace=0.42)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out.with_suffix('.pdf')} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
