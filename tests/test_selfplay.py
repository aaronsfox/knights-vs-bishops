"""Tests for the game parsing and match settings."""

import io
from pathlib import Path

import chess.pgn
import pytest

from selfplay import (VS, FileInfo, MatchSettings, Player, end_reason, fast_game_row,
                      fastchess_command, game_row, load_player, match_key, parse_move_comment,
                      parse_raw_name, raw_path, score_summary, split_games)
from utils import load_engine_config

# A game from BN v NN (White without b1 and c1, Black without both bishops).
PGN = """[Event "Fastchess Tournament"]
[Round "2"]
[White "stockfish_b"]
[Black "stockfish_a"]
[Result "1-0"]
[EngineWhiteName "Stockfish 19"]
[SetUp "1"]
[FEN "rn1qk1nr/pppppppp/8/8/8/8/PPPPPPPP/R2QKBNR w KQkq - 0 1"]
[Termination "adjudication"]
[TimeControl "10+0.1"]

1. e4 {book} e5 {book} 2. Nf3 {book} Nc6 {book} 3. Bc4 {book} Nf6 {book}
4. d3 {+0.41/18 0.412s, n=210000, sd=25, nps=509000}
Nd4 {-M7/30 0.300s, n=100000, sd=31, nps=333333, White wins by adjudication} 1-0
"""

INFO = FileInfo("stockfish", "b1-c1-c8-f8")
BOOK = {0: "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6", 1: "e2e4 e7e5 g1f3 b8c6 f1c4 g8f6"}
BOOK_SAN = {0: "d4 d5 c4 c6 Nf3 Nf6", 1: "e4 e5 Nf3 Nc6 Bc4 Nf6"}


def test_parse_move_comment():
    c = parse_move_comment("+0.28/13 0.249s, n=110878, sd=21, nps=441745")
    assert c == {"cp": 28, "mate": None, "depth": 13, "time": 0.249,
                 "nodes": 110878, "seldepth": 21}
    m = parse_move_comment("-M18/23 0.006s, n=8242, sd=19, nps=1030250, White wins by adjudication")
    assert m["mate"] == -18 and m["cp"] is None
    assert parse_move_comment("book") is None


def test_end_reason():
    assert end_reason("-M6/74 0.007s, n=41376, sd=7, nps=4597333, White wins by adjudication") \
        == "White wins by adjudication"
    assert end_reason("+0.10/12 0.050s, n=1000, sd=12, nps=20000") is None


def test_game_row():
    row = game_row(chess.pgn.read_game(io.StringIO(PGN)), INFO, BOOK)
    assert row["position_id"] == "b1-c1-c8-f8" and row["fen_matches_position"]
    assert row["line_index"] == 1 and row["line_matched_by_round"]
    assert row["score_white"] == 1.0 and row["white_engine"] == "stockfish"
    assert row["book_plies"] == 6 and row["plies"] == 8
    assert row["first_eval_white_cp"] == 41
    assert row["end_reason"] == "White wins by adjudication"


def test_fast_parser_matches_full_parser():
    two = PGN + "\n" + PGN.replace('[Round "2"]', '[Round "3"]')
    fast = [fast_game_row(h, mt, INFO, BOOK_SAN) for h, mt in split_games(two)]
    slow = [game_row(chess.pgn.read_game(io.StringIO(g)), INFO, BOOK)
            for g in (PGN, PGN.replace('[Round "2"]', '[Round "3"]'))]
    assert fast == slow


def test_identical_games_share_a_hash():
    a = fast_game_row(*next(split_games(PGN)), INFO, BOOK_SAN)
    b = fast_game_row(*next(split_games(PGN.replace("Nd4", "Ne7"))), INFO, BOOK_SAN)
    assert a["moves_hash"] != b["moves_hash"]
    assert a["moves_hash"] == fast_game_row(*next(split_games(PGN)), INFO, BOOK_SAN)["moves_hash"]


def test_raw_names(tmp_path):
    assert match_key("sf11_b050", "sf11") == f"sf11_b050{VS}sf11"
    p = raw_path(tmp_path, "stockfish", "b1-c1-c8-f8", "b2")
    assert p.name == "b1-c1-c8-f8.b2.pgn.gz"
    assert parse_raw_name(p) == ("b1-c1-c8-f8", "b2")
    assert parse_raw_name(raw_path(tmp_path, "x", "start")) == ("start", "")
    info = FileInfo(f"sf11_b050{VS}sf11", "start")
    assert (info.player, info.opponent) == ("sf11_b050", "sf11")


CONFIG = {"match": {"tc": "10+0.1", "games_per_line": 10, "fastchess": "tools/fc",
                    "draw": {"movenumber": 40}, "resign": {"twosided": True}},
          "uci_overrides": {"Hash": 64},
          "rungs": {"sf_n1k": {"engine": "stockfish", "nodes": 1000},
                    "sf_elo": {"engine": "stockfish",
                               "options": {"UCI_LimitStrength": True, "UCI_Elo": 1500}}}}


def test_settings_and_self_play_command():
    s = MatchSettings.from_config(CONFIG)
    p = load_player(CONFIG, "stockfish")
    cmd = fastchess_command(p, None, s, Path("o.pgn"), Path("g.pgn"), n_lines=50, concurrency=8)
    assert cmd[cmd.index("-variant") + 1] == "standard"
    assert cmd[cmd.index("-rounds") + 1] == "250"
    assert "name=stockfish_a" in cmd and "name=stockfish_b" in cmd
    assert "-repeat" in cmd and "option.Hash=64" in cmd and "twosided=true" in cmd
    with pytest.raises(ValueError):
        MatchSettings.from_config(CONFIG, games_per_line=5)
    assert MatchSettings.from_config(CONFIG, games_per_line=40).games_per_line == 40


def test_opponent_and_rung_commands():
    s = MatchSettings.from_config(CONFIG)
    variant = Player("sf11_b050", load_engine_config("sf11_b050"))
    base = Player("sf11", load_engine_config("sf11"))
    cmd = fastchess_command(variant, base, s, Path("o"), Path("g"), 50, 8)
    assert "name=sf11_b050" in cmd and "name=sf11" in cmd
    rung = load_player(CONFIG, "sf_n1k")
    assert rung.nodes == 1000 and "nodes=1000" in fastchess_command(rung, None, s, Path("o"),
                                                                   Path("g"), 50, 8)
    elo = load_player(CONFIG, "sf_elo")
    cmd = fastchess_command(elo, None, s, Path("o"), Path("g"), 50, 8)
    assert "option.UCI_LimitStrength=true" in cmd and "option.UCI_Elo=1500" in cmd
    assert elo.display_name == "Stockfish (UCI_Elo=1500)"


def test_score_summary():
    import pandas as pd
    games = pd.DataFrame({"match": "x", "position_id": "start",
                          "score_white": [1, 0.5, 0.5, 0], "plies": 100})
    s = score_summary(games, ["match", "position_id"]).iloc[0]
    assert (s.white_wins, s.draws, s.black_wins) == (1, 2, 1)
    assert s.expected_score == 0.5 and abs(s.elo) < 1e-9


def test_parser_accepts_options_either_side():
    from selfplay import build_parser
    p = build_parser()
    assert p.parse_args(["--output-dir", "x", "run", "--player", "sf"]).output_dir == "x"
    assert p.parse_args(["run", "--player", "sf", "--output-dir", "y"]).output_dir == "y"
    assert p.parse_args(["collect"]).output_dir is None
