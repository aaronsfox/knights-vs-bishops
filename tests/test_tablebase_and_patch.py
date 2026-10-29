"""Tests for the tablebase sampler and the Stockfish 11 variant patcher."""

import importlib.util
import random
import sys
from pathlib import Path

import chess
import pytest

import tablebase as T

REPO = Path(__file__).resolve().parent.parent


def test_material_sets():
    ms = T.MaterialSet("KBPPvKN")
    assert ms.sides == ("KBPP", "KN") and ms.n_pieces == 6 and ms.minor == "B"
    assert T.MaterialSet("KNPvKB").minor == "N"


@pytest.mark.parametrize("name", ["KBPvKN", "KNPPvKB", "KRBPvKRN", "KBPvKNP"])
def test_samples_are_legal(name):
    ms = T.MaterialSet(name)
    rng = random.Random(0)
    for _ in range(300):
        b = T.sample_position(ms, rng)
        assert b.is_valid() and chess.popcount(b.occupied) == ms.n_pieces
        assert not b.pieces(chess.PAWN, chess.WHITE) & (chess.BB_RANK_1 | chess.BB_RANK_8)


def test_wrong_bishop():
    assert T.features(chess.Board("8/8/8/8/8/P7/5k2/K1B5 w - - 0 1"))["wrong_bishop"] is True
    assert T.features(chess.Board("8/8/8/8/8/P7/5k2/KB6 w - - 0 1"))["wrong_bishop"] is False


def test_passed_pawn_and_wings():
    f = T.features(chess.Board("8/8/3p4/8/4P3/8/5k2/K1B5 w - - 0 1"))
    assert f["strong_passed_pawns"] == 0 and f["pawns_one_wing"] is False
    f = T.features(chess.Board("8/8/8/8/4P3/8/5k2/K1B5 w - - 0 1"))
    assert f["strong_passed_pawns"] == 1 and f["pawns_one_wing"] is True


def test_jeffreys_interval():
    p, lo, hi = T.jeffreys(30, 100)
    assert p == 0.3 and lo < 0.3 < hi


def load_patcher():
    spec = importlib.util.spec_from_file_location("patch_sf11",
                                                  REPO / "engines" / "stockfish11" / "patch_sf11.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_source(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "types.h").write_text(
        "  PawnValueMg   = 128,   PawnValueEg   = 213,\n"
        "  KnightValueMg = 781,   KnightValueEg = 854,\n"
        "  BishopValueMg = 825,   BishopValueEg = 915,\n")
    (src / "material.cpp").write_text("    {1438                               }, // Bishop pair\n")
    (src / "misc.cpp").write_text('const string Version = "11";\n')
    return src


def test_patch_values(tmp_path):
    patcher = load_patcher()
    src = fake_source(tmp_path)
    summary = patcher.patch(src, "sf11_b050", 0.5, 2.0)
    v = summary["values"]
    # Half of 0.5 pawns each way: 32 (Mg) and 53 (Eg) internal units.
    assert (v["BishopValueMg"], v["KnightValueMg"]) == (857, 749)
    assert (v["BishopValueEg"], v["KnightValueEg"]) == (968, 801)
    assert v["BishopValueMg"] + v["KnightValueMg"] == 825 + 781     # total unchanged
    assert "BishopValueMg = 857" in (src / "types.h").read_text()
    assert "{2876" in (src / "material.cpp").read_text()
    assert '"11 sf11_b050"' in (src / "misc.cpp").read_text()
    with pytest.raises(SystemExit):           # already patched: refuses
        patcher.patch(src, "sf11_b050", 0.5, 1.0)


def test_patch_knight_direction(tmp_path):
    v = load_patcher().patch(fake_source(tmp_path), "sf11_n020", -0.2, 1.0)["values"]
    assert v["KnightValueMg"] > 781 and v["BishopValueMg"] < 825
    assert v["KnightValueMg"] != v["BishopValueMg"]


def test_variant_manifests_build_cleanly(tmp_path):
    import tomllib
    patcher = load_patcher()
    for manifest in sorted((REPO / "engines").glob("sf11_*/engine.toml")):
        variant = tomllib.loads(manifest.read_text())["variant"]
        sub = tmp_path / manifest.parent.name
        sub.mkdir()
        patcher.patch(fake_source(sub), manifest.parent.name,
                      variant["bishop_minus_knight_pawns"], variant["bishop_pair_scale"])
