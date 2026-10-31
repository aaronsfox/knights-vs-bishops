"""Starting positions for the knights vs bishops study.

Every position is the standard starting array with some pieces removed,
White to move, castling rights intact. A position's id names the squares the
pieces were removed from, White's first, e.g.

    start          the standard position (nothing removed)
    c1-b8          White without the c1 bishop, Black without the b8 knight
    b1-c1-c8-f8    White without the b1 knight and c1 bishop (BN left),
                   Black without both bishops (NN left)
    f2             pawn odds: White without the f-pawn

Groups (usable in position specs, e.g. ``--positions 2v2``):

    control    the standard position
    1v1        one minor piece removed from each side (16 positions)
    2v2        two minor pieces removed from each side (36 positions)
    pawn_odds  one pawn removed from one side (f2, f7, b2, b7)
    key        the key comparisons and their controls, topped up (see KEY_DESIGNS)
    removal    1v1 + 2v2 (the 52-cell removal matrix)
    all        everything above

Colour balance
--------------
In self-play both sides are the same engine, so swapping the two engine
copies between games does not move the material: the side with the bishop is
always the same colour within a position. Colour balance comes from each
position's **colour-flipped twin** (every removed square mirrored to the other
side's back rank, e.g. ``c1-b8`` and ``b1-c8``). A position and its twin form
one **comparison**, scored from the **focal side**:

1. the side with more bishops (BN in BN v NN, BBN in BBN v BNN);
2. otherwise the side with more pawns (pawn odds);
3. otherwise (same material, different squares, e.g. ``b1-g8``) the side
   whose removed pieces stood on the earlier files, so ``b1-g8`` and its twin
   ``g1-b8`` are both scored for the side missing the b-file knight.

A position that is its own twin (``start``, ``c1-c8``, ...) has no focal side:
it is a mirrored control that measures White's first-move edge. Pooling a
comparison's two positions cancels White's edge from the focal side's score.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import chess

REPO_ROOT = Path(__file__).resolve().parent.parent
START_ID = "start"

# The removable pieces: the four minor pieces of each side, and the pawns
# used for pawn-odds calibration.
MINOR_SQUARES = {
    chess.WHITE: {"b1": "N", "g1": "N", "c1": "B", "f1": "B"},
    chess.BLACK: {"b8": "N", "g8": "N", "c8": "B", "f8": "B"},
}
PAWN_ODDS_SQUARES = ("f2", "f7", "b2", "b7")

# Designs of the removal matrix, named from the focal side: "<focal> v <other>"
# by the minor pieces each side keeps. The key designs isolate the single
# bishop vs knight (BN v NN), the pair over one bishop (BB v BN) and their
# combination (BB v NN); the matching equal-material designs are controls.
KEY_DESIGNS = ("BN v NN", "BB v BN", "BB v NN")
KEY_CONTROL_DESIGNS = ("NN v NN", "BB v BB")

GROUPS = ("control", "1v1", "2v2", "pawn_odds", "key", "removal", "all")


def flip_square(square: str) -> str:
    """The same file on the other side's back rank (rank r -> 9 - r)."""
    return f"{square[0]}{9 - int(square[1])}"


def square_colour(square: str) -> chess.Color:
    """chess.WHITE for squares on ranks 1-4 (White's pieces start there)."""
    return chess.WHITE if int(square[1]) <= 4 else chess.BLACK


def make_id(removed: tuple[str, ...] | list[str] | set[str]) -> str:
    """Canonical id: White's squares then Black's, each in file order."""
    if not removed:
        return START_ID
    white = sorted(s for s in removed if square_colour(s) == chess.WHITE)
    black = sorted(s for s in removed if square_colour(s) == chess.BLACK)
    return "-".join(white + black)


def parse_id(pos_id: str) -> tuple[str, ...]:
    """Removed squares of a position id (validated)."""
    if pos_id == START_ID:
        return ()
    squares = tuple(pos_id.split("-"))
    for s in squares:
        if s not in chess.SQUARE_NAMES:
            raise ValueError(f"bad square {s!r} in position id {pos_id!r}")
    if make_id(squares) != pos_id:
        raise ValueError(f"position id {pos_id!r} is not canonical; use {make_id(squares)!r}")
    return squares


@dataclass(frozen=True)
class Position:
    """One starting position (a cell of the design)."""

    id: str
    removed: tuple[str, ...]
    group: str           # control, 1v1, 2v2 or pawn_odds
    fen: str

    def board(self) -> chess.Board:
        """A fresh board for this position (White to move)."""
        return chess.Board(self.fen)

    # -- material ---------------------------------------------------------

    def minors(self, colour: chess.Color) -> str:
        """Minor pieces the side keeps, bishops first, e.g. "BN"."""
        gone = [MINOR_SQUARES[colour][s] for s in self.removed if s in MINOR_SQUARES[colour]]
        kept = list("BBNN")
        for piece in gone:
            kept.remove(piece)
        return "".join(kept)

    def pawns(self, colour: chess.Color) -> int:
        rank = "2" if colour == chess.WHITE else "7"
        return 8 - sum(1 for s in self.removed if s[1] == rank)

    @property
    def twin_id(self) -> str:
        """The colour-flipped twin (itself for a mirrored control)."""
        return make_id([flip_square(s) for s in self.removed])

    @property
    def is_mirrored_control(self) -> bool:
        return self.twin_id == self.id

    @property
    def focal(self) -> chess.Color | None:
        """The side a comparison is scored from (see the module docstring)."""
        if self.is_mirrored_control:
            return None
        bw, bb = self.minors(chess.WHITE).count("B"), self.minors(chess.BLACK).count("B")
        if bw != bb:
            return chess.WHITE if bw > bb else chess.BLACK
        pw, pb = self.pawns(chess.WHITE), self.pawns(chess.BLACK)
        if pw != pb:
            return chess.WHITE if pw > pb else chess.BLACK
        files_w = "".join(sorted(s[0] for s in self.removed if square_colour(s) == chess.WHITE))
        files_b = "".join(sorted(s[0] for s in self.removed if square_colour(s) == chess.BLACK))
        return chess.WHITE if files_w < files_b else chess.BLACK

    @property
    def design(self) -> str:
        """Material design from the focal side (White for controls),
        e.g. "BN v NN", "BBN v BNN", "pawn odds", "start"."""
        if self.group == "control":
            return "start"
        if self.group == "pawn_odds":
            return "pawn odds"
        side = self.focal if self.focal is not None else chess.WHITE
        return f"{self.minors(side)} v {self.minors(not side)}"

    @property
    def comparison_id(self) -> str:
        """The comparison this position belongs to, named by the member in
        which White is the focal side (or the position itself, for a
        mirrored control)."""
        if self.focal is None or self.focal == chess.WHITE:
            return self.id
        return self.twin_id

    @property
    def label(self) -> str:
        """Short description, e.g. 'BN v NN (c1-b8)'."""
        return f"{self.design} ({self.id})"


def _fen_without(removed: tuple[str, ...]) -> str:
    board = chess.Board()
    for s in removed:
        board.remove_piece_at(chess.parse_square(s))
    return board.fen()


@functools.lru_cache(maxsize=None)
def all_positions() -> tuple[Position, ...]:
    """Every position in the study, in a fixed order: the control, 1v1,
    2v2, then pawn odds."""
    out: list[Position] = [Position(START_ID, (), "control", chess.STARTING_FEN)]
    w_sq, b_sq = list(MINOR_SQUARES[chess.WHITE]), list(MINOR_SQUARES[chess.BLACK])
    for k, group in ((1, "1v1"), (2, "2v2")):
        from itertools import combinations
        for w in combinations(w_sq, k):
            for b in combinations(b_sq, k):
                removed = parse_id(make_id(w + b))
                out.append(Position(make_id(removed), removed, group, _fen_without(removed)))
    for s in PAWN_ODDS_SQUARES:
        out.append(Position(s, (s,), "pawn_odds", _fen_without((s,))))
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _by_id() -> dict[str, Position]:
    return {p.id: p for p in all_positions()}


@functools.lru_cache(maxsize=None)
def _by_board_fen() -> dict[str, str]:
    return {chess.Board(p.fen).board_fen(): p.id for p in all_positions()}


def get_position(pos_id: str) -> Position:
    """The position with this id (see the module docstring)."""
    try:
        return _by_id()[pos_id]
    except KeyError:
        raise ValueError(f"unknown position id {pos_id!r}") from None


def position_id_from_fen(fen: str) -> str | None:
    """Position id from a FEN (only the piece placement is compared), or
    None if it isn't one of the study's starting positions. An empty FEN is
    the standard starting position (PGNs omit it there)."""
    if not fen:
        return START_ID
    return _by_board_fen().get(fen.split()[0])


def in_group(p: Position, group: str) -> bool:
    if group == "all":
        return True
    if group == "removal":
        return p.group in ("1v1", "2v2")
    if group == "key":
        if p.group in ("control", "pawn_odds"):
            return True
        if p.design in KEY_DESIGNS or p.design in KEY_CONTROL_DESIGNS:
            return True
        # BN v BN controls that are their own twin (same squares on both sides)
        return p.design == "BN v BN" and p.is_mirrored_control
    return p.group == group


def parse_position_spec(spec: str) -> list[str]:
    """Position ids from a spec, in the registry order without duplicates.

    Accepts ids and group names separated by commas ("start,c1-b8",
    "2v2,pawn_odds", "all"), or a file of them ("@config/sample_positions.txt",
    one or more per line, '#' starts a comment).
    """
    spec = spec.strip()
    if spec.startswith("@"):
        path = Path(spec[1:])
        if not path.is_absolute():
            path = REPO_ROOT / path
        tokens: list[str] = []
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                tokens.extend(line.replace(",", " ").split())
    else:
        tokens = [t.strip() for t in spec.split(",") if t.strip()]
    wanted: set[str] = set()
    for token in tokens:
        if token in GROUPS:
            wanted.update(p.id for p in all_positions() if in_group(p, token))
        else:
            wanted.add(get_position(token).id)
    return [p.id for p in all_positions() if p.id in wanted]


def comparisons() -> list[dict]:
    """One row per comparison: its id, design, the member positions and
    which side is focal in each. Mirrored controls are comparisons of one."""
    rows: dict[str, dict] = {}
    for p in all_positions():
        row = rows.setdefault(p.comparison_id, {
            "comparison_id": p.comparison_id, "design": p.design, "group": p.group,
            "mirrored_control": p.is_mirrored_control, "positions": [], "focal": []})
        row["positions"].append(p.id)
        row["focal"].append(None if p.focal is None else ("white" if p.focal else "black"))
    return list(rows.values())
