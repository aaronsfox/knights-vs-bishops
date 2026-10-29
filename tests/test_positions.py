"""Tests for the position registry: the removal matrix, twins and comparisons."""

import collections

import chess
import pytest

import positions as P


def test_counts():
    groups = collections.Counter(p.group for p in P.all_positions())
    assert groups == {"control": 1, "1v1": 16, "2v2": 36, "pawn_odds": 4}
    assert len(P.parse_position_spec("removal")) == 52
    assert len(P.parse_position_spec("all")) == 57


def test_design_counts_match_the_research_plan():
    d = collections.Counter(p.design for p in P.all_positions() if p.group in ("1v1", "2v2"))
    assert d["BN v NN"] == 8 and d["BB v BN"] == 8 and d["BB v NN"] == 2
    assert d["BBN v BNN"] == 8
    assert d["NN v NN"] == 1 and d["BB v BB"] == 1 and d["BN v BN"] == 16


@pytest.mark.parametrize("p", P.all_positions(), ids=lambda p: p.id)
def test_every_position(p):
    board = p.board()
    assert board.is_valid() and board.turn == chess.WHITE
    assert board.castling_rights == chess.BB_A1 | chess.BB_H1 | chess.BB_A8 | chess.BB_H8
    assert P.make_id(p.removed) == p.id
    twin = P.get_position(p.twin_id)
    assert twin.twin_id == p.id                              # twins pair up
    assert twin.design == p.design                           # same comparison design
    assert twin.comparison_id == p.comparison_id
    if p.focal is not None:
        assert twin.focal == (not p.focal)                   # focal side swaps colour
        assert p.minors(p.focal).count("B") >= p.minors(not p.focal).count("B")
    assert P.position_id_from_fen(p.fen) == p.id


def test_focal_side_examples():
    p = P.get_position("c1-b8")         # White lost a bishop, Black a knight
    assert p.minors(chess.WHITE) == "BNN" and p.minors(chess.BLACK) == "BBN"
    assert p.focal == chess.BLACK and p.design == "BBN v BNN"
    assert p.twin_id == "b1-c8" and p.comparison_id == "b1-c8"
    assert P.get_position("f2").focal == chess.BLACK        # Black has the extra pawn
    assert P.get_position("c1-c8").focal is None            # mirrored control
    # Same material, different squares: scored for the side missing the b-knight.
    assert P.get_position("b1-g8").focal == chess.WHITE
    assert P.get_position("g1-b8").focal == chess.BLACK


def test_comparisons_partition_positions():
    rows = P.comparisons()
    members = [pid for r in rows for pid in r["positions"]]
    assert sorted(members) == sorted(p.id for p in P.all_positions())
    assert all(len(r["positions"]) == (1 if r["mirrored_control"] else 2) for r in rows)
    key = [r for r in rows if r["design"] in P.KEY_DESIGNS]
    assert collections.Counter(r["design"] for r in key) == {"BN v NN": 4, "BB v BN": 4,
                                                            "BB v NN": 1}


def test_parse_position_spec(tmp_path):
    assert P.parse_position_spec("c1-b8,start") == ["start", "c1-b8"]     # registry order
    assert P.parse_position_spec("control,f2") == ["start", "f2"]
    f = tmp_path / "ids.txt"
    f.write_text("# test\nstart  f7  # pawn odds\n")
    assert P.parse_position_spec(f"@{f}") == ["start", "f7"]
    with pytest.raises(ValueError):
        P.parse_position_spec("b8-c1")          # not canonical (White's squares first)
    with pytest.raises(ValueError):
        P.parse_position_spec("e4")             # not a study position


def test_key_group_has_every_twin():
    key = set(P.parse_position_spec("key"))
    assert len(key) == 29
    assert all(P.get_position(i).twin_id in key for i in key)


def test_pilot_file_pairs_twins():
    ids = set(P.parse_position_spec("@config/pilot_positions.txt"))
    assert all(P.get_position(i).twin_id in ids for i in ids)


def test_empty_fen_is_the_standard_position():
    assert P.position_id_from_fen("") == "start"
    assert P.position_id_from_fen("8/8/8/8/8/8/8/K6k w - - 0 1") is None
