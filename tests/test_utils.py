"""Tests for code/utils.py. Run with `python -m pytest` from the repo root."""

import math

import pytest

import utils

ENGINES = {"stockfish", "caissa", "lc0", "sf11", "sf11_n020", "sf11_n050", "sf11_b020",
           "sf11_b050", "sf11_pair200"}


def test_expected_score_and_elo():
    assert utils.expected_score_from_wdl(100, 800, 100) == pytest.approx(0.5)
    assert utils.expected_score_from_wdl(200, 750, 50) == pytest.approx(0.575)
    assert utils.elo_from_expected(0.5) == pytest.approx(0.0)
    assert utils.elo_from_expected(0.64) == pytest.approx(100.0, abs=0.5)
    assert math.isnan(utils.elo_from_expected(math.nan))


def test_convergence_measures():
    values = [0.55, 0.53, 0.54, 0.54]
    assert utils.total_variation(values) == pytest.approx(0.03)
    assert utils.tail_drift(values, k=3) == pytest.approx(0.01)
    assert math.isnan(utils.tail_drift([0.5]))


def test_engine_manifests_load():
    keys = utils.list_engines()
    assert set(keys) == ENGINES
    for key in keys:
        cfg = utils.load_engine_config(key)
        managed = {name.lower() for name in cfg.uci_options} & utils.MANAGED_OPTIONS
        assert not managed, f"{key} sets options python-chess manages: {managed}"
        assert cfg.uci_options.get("Threads") == 1, f"{key} must run single-threaded"


def test_variant_display_names():
    names = {k: utils.load_engine_config(k).display_name for k in ENGINES if k.startswith("sf11")}
    assert names == {"sf11": "Stockfish 11", "sf11_n020": "Aron Knightzowitsch",
                     "sf11_n050": "Hikaru Knightamura", "sf11_b020": "Bishy Anand",
                     "sf11_b050": "Bobby Bischer", "sf11_pair200": "Magnus Pairlsen"}
    assert utils.load_engine_config("stockfish").display_name == "Stockfish"


def test_positions_are_re_exported():
    assert utils.get_position("start").fen.startswith("rnbqkbnr/pppppppp")
    assert utils.parse_position_spec("control") == ["start"]
