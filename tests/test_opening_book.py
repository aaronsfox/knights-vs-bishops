"""Tests for the opening book's line filtering and balanced selection."""

import random

import chess

from opening_book_generator import BookSettings, Line, acceptable, select_lines


SETTINGS = BookSettings(
    plies=6, lines_per_position=4, seed=1, candidate_nodes=1, multipv=1, check_nodes=0,
    move_margin_cp=20, max_side_loss_cp=40, max_side_loss_gap_cp=20, max_shift_cp=25,
    pool_factor=2, max_walks=10,
)


def line(first: str, second: str, shift: float, wl: int = 0, bl: int = 0) -> Line:
    moves = (chess.Move.from_uci(first), chess.Move.from_uci(second))
    return Line(moves=moves, cp_end=0, white_loss=wl, black_loss=bl, shift_cp=shift, shift=shift)


def test_acceptable_limits():
    assert acceptable(line("e2e4", "e7e5", 10, wl=10, bl=20), SETTINGS)
    assert not acceptable(line("e2e4", "e7e5", 30), SETTINGS)            # drifted too far
    assert not acceptable(line("e2e4", "e7e5", 0, wl=30, bl=5), SETTINGS)  # sides unequal


def test_select_balances_shift_and_spreads_first_moves():
    pool = [line("e2e4", "e7e5", 10), line("e2e4", "c7c5", -10),
            line("d2d4", "d7d5", 8), line("d2d4", "g8f6", -8),
            line("g1f3", "d7d5", 20), line("g1f3", "g8f6", 15)]
    chosen = select_lines(pool, 4, random.Random(0))
    assert len(chosen) == 4
    assert len({ln.moves for ln in chosen}) == 4
    assert len({ln.first_move for ln in chosen}) >= 2
    assert abs(sum(ln.shift for ln in chosen) / 4) <= 5


def test_select_repeats_when_pool_is_small():
    pool = [line("e2e4", "e7e5", 3), line("d2d4", "d7d5", -3)]
    chosen = select_lines(pool, 5, random.Random(0))
    assert len(chosen) == 5
    assert len({ln.moves for ln in chosen}) == 2


def test_select_is_reproducible():
    def pool():
        return [line("e2e4", m, s) for m, s in [("e7e5", 4), ("c7c5", -6), ("e7e6", 1)]] + \
               [line("d2d4", m, s) for m, s in [("d7d5", 2), ("g8f6", -3)]]
    a = select_lines(pool(), 4, random.Random("1999:start"))
    b = select_lines(pool(), 4, random.Random("1999:start"))
    assert [ln.moves for ln in a] == [ln.moves for ln in b]


def test_swaps_fix_balance_without_dropping_first_moves():
    # One first move's lines all lean the same way, so rotation alone leaves
    # the mean off zero; swaps should balance it and keep all three first moves.
    pool = ([line("b2b3", m, -0.05) for m in ("e7e5", "c7c5", "e7e6", "d7d5")]
            + [line("e2e4", m, s) for m, s in (("e7e5", 0.02), ("c7c5", 0.03), ("e7e6", -0.01), ("d7d5", 0.04))]
            + [line("d2d4", m, s) for m, s in (("d7d5", 0.01), ("g8f6", 0.05), ("e7e6", -0.02), ("c7c5", 0.03))])
    chosen = select_lines(pool, 6, random.Random(1))
    assert len({ln.moves for ln in chosen}) == 6
    assert {ln.first_move.uci() for ln in chosen} == {"b2b3", "e2e4", "d2d4"}
    assert abs(sum(ln.shift for ln in chosen) / 6) <= 0.002
