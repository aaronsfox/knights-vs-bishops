"""Tests for the Lichess reader and the bishop/knight event extractor."""

import io

import chess
import pytest

import human_events as he
from lichess_games import category, game_row, hash_fraction, header_filter, iter_games, month_range

RUY = ("1. e4 { [%eval 0.2] [%clk 0:30:00] } 1... e5 { [%clk 0:30:00] } 2. Nf3 { [%clk 0:29:50] } "
       "2... Nc6 { [%clk 0:29:40] } 3. Bb5 { [%clk 0:29:30] } 3... a6 { [%clk 0:29:20] } "
       "4. Bxc6 { [%eval -0.1] [%clk 0:29:00] } 4... dxc6 { [%clk 0:29:10] } 5. O-O { [%clk 0:28:00] } "
       "5... f6 { [%clk 0:28:30] } 6. d4 { [%clk 0:27:00] } 6... Bg4 { [%clk 0:28:00] } "
       "7. c3 { [%clk 0:26:00] } 1-0")

PGN = f"""[Event "Rated Classical game"]
[Site "https://lichess.org/AbCdEfGh"]
[White "alice"]
[Black "bob"]
[Result "1-0"]
[WhiteElo "2210"]
[BlackElo "2150"]
[WhiteRatingDiff "+5"]
[BlackRatingDiff "-5"]
[TimeControl "1800+0"]
[Termination "Normal"]

{RUY}

[Event "Rated Bullet game"]
[Site "https://lichess.org/IjKlMnOp"]
[White "carol"]
[Black "dave"]
[Result "0-1"]
[WhiteElo "2500"]
[BlackElo "2400"]
[TimeControl "60+0"]

1. e4 {{ [%clk 0:01:00] }} 0-1

[Event "Casual Rapid game"]
[Site "https://lichess.org/QqQqQqQq"]
[White "erin"]
[Black "frank"]
[Result "1/2-1/2"]
[TimeControl "600+0"]

1. e4 1/2-1/2
"""


def test_reader_filters_and_row():
    games = list(iter_games(io.StringIO(PGN)))
    assert len(games) == 3
    cats = {"classical"}
    kept = [header_filter(h, cats, 2000, 1.0) for h, _ in games]
    assert kept[0] is not None and kept[1] is None and kept[2] is None
    row = game_row(games[0][0], games[0][1], "2026-10", kept[0])
    assert row["category"] == "classical" and row["plies"] == 13 and row["has_eval"]
    assert row["white_id"] != "alice" and len(row["white_id"]) == 12
    assert header_filter(games[0][0], cats, 2300, 1.0) is None          # mean 2180
    assert category(300 + 40 * 3) == "blitz" and category(1800) == "classical"
    assert month_range("2026-11:2027-01") == ["2026-11", "2026-12", "2027-01"]
    assert 0 <= hash_fraction("AbCdEfGh") < 1


def test_parse_movetext_clocks_and_evals():
    plies = he.parse_movetext(RUY)
    assert [p[0] for p in plies[:3]] == ["e4", "e5", "Nf3"]
    assert plies[0][1] == 1800 and plies[0][2] == pytest.approx(20)
    assert plies[6][0] == "Bxc6" and plies[6][2] == pytest.approx(-10)
    mates = he.parse_movetext("1. e4 { [%eval #-3] } 1... e5 { [%eval #2] } *")
    assert mates[0][2] == -he.MATE_CP and mates[1][2] == he.MATE_CP


def test_ruy_exchange_events():
    ev = he.game_events("g1", RUY)
    ex = ev["exchanges"]
    assert len(ex) == 1
    e = ex[0]
    assert e["kind"] == "B for N" and e["bishop_giver"] == "white" and e["initiator"] == "white"
    assert e["initiator_gave_bishop"] and e["pair_broken"] and e["ply"] == 7 and e["end_ply"] == 8
    assert e["clock_white"] == 1770 and e["eval_cp"] == pytest.approx(20)
    # The opportunity on ply 7 (Bxc6, c6 defended) was taken.
    opp = [o for o in ev["opportunities"] if o["ply"] == 7]
    assert opp and opp[0]["bxn_available"] and opp[0]["played"] == "BxN"
    # Pieces: White's f1 bishop and Black's b8 knight were captured on plies 8 and 7.
    pcs = {p["piece_id"]: p for p in ev["pieces"]}
    assert pcs["wBf1"]["captured_ply"] == 8 and pcs["wBf1"]["captured_by"] == "P"
    assert pcs["bNb8"]["captured_ply"] == 7 and pcs["bNb8"]["captured_by"] == "B"
    assert pcs["wNg1"]["survived"] and pcs["bBc8"]["survived"]
    # Black now has the pair: a BBN v BNN spell from the first quiet ply.
    sp = [s for s in ev["spells"] if s["kind"] == "minor imbalance"]
    assert len(sp) == 1 and sp[0]["design"] == "BBN v BNN" and sp[0]["focal"] == "black"
    assert sp[0]["start_ply"] == 9 and sp[0]["ended_by"] == "game end" and sp[0]["length"] == 5
    assert ev["games"][0]["parse_error"] == ""


def test_designs_and_endgames():
    c = he.counts(chess.Board("4k3/pp3n2/8/8/8/8/PP3B2/4K3 w - - 0 1"))
    assert he.minor_design(c) == ("B v N", "white")
    assert he.endgame_type(chess.Board("4k3/pp3n2/8/8/8/8/PP3B2/4K3 w - - 0 1"), c) == ("B v N", "white")
    c2 = he.counts(chess.Board("4k3/ppp5/8/8/8/8/PP6/4K3 w - - 0 1"))
    assert he.pawn_up(c2) == "black" and he.minor_design(c2) is None
    b = chess.Board("4k3/8/8/3b4/8/8/8/2B1K3 w - - 0 1")     # d5 light, c1 dark
    assert he.endgame_type(b, he.counts(b))[0] == "B v B opposite colour"
    f = he.pawn_features(chess.Board("4k3/p7/8/8/8/8/P6P/4K3 w - - 0 1"), chess.WHITE)
    assert f["pawns_both_wings"] and f["passed_pawns"] == 1


def test_bad_move_is_reported_not_raised():
    ev = he.game_events("bad", "1. e4 e5 2. Ke3 *")
    assert ev["games"][0]["parse_error"].startswith("ply 3")
