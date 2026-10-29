"""Patch a Stockfish 11 source tree into a knight- or bishop-preferring variant.

Reads the ``[variant]`` table of an engine manifest (engines/<key>/engine.toml)
and rewrites the minor-piece values and the bishop-pair term in the source:

    [variant]
    bishop_minus_knight_pawns = 0.5   # shift of B - N, in pawns (+ likes bishops)
    bishop_pair_scale = 1.0           # multiplier on the bishop-pair term

The shift is split evenly: the bishop gains half and the knight loses half,
in both the middlegame and the endgame values, each half measured in that
phase's pawn value (PawnValueMg = 128, PawnValueEg = 213). Total minor-piece
material against pawns, rooks and queens therefore stays the same. Because
Stockfish uses these constants everywhere (evaluation, static exchange
evaluation in move ordering and pruning, endgame recognition, game phase),
the variant believes the new values throughout, not just in its evaluation.

The bishop-pair term is QuadraticOurs[0][0] in material.cpp (1438, divided
by 16 in Material::probe, so about 90 internal units, roughly 0.7 of a
middlegame pawn or 0.4 of an endgame pawn).

The original constants are checked before anything is changed; any mismatch
(another Stockfish version, an already patched tree) stops the patch.

    python engines/stockfish11/patch_sf11.py <stockfish-src-dir> engines/sf11_b050/engine.toml
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ORIGINAL = {
    "PawnValueMg": 128, "PawnValueEg": 213,
    "KnightValueMg": 781, "KnightValueEg": 854,
    "BishopValueMg": 825, "BishopValueEg": 915,
}
BISHOP_PAIR = 1438
VERSION = '"11"'


def new_values(shift_pawns: float) -> dict[str, int]:
    """Knight and bishop values after shifting B - N by ``shift_pawns``."""
    out = {}
    for phase in ("Mg", "Eg"):
        half = round(shift_pawns / 2 * ORIGINAL[f"PawnValue{phase}"])
        out[f"BishopValue{phase}"] = ORIGINAL[f"BishopValue{phase}"] + half
        out[f"KnightValue{phase}"] = ORIGINAL[f"KnightValue{phase}"] - half
        if out[f"BishopValue{phase}"] == out[f"KnightValue{phase}"]:
            # Stockfish tells pieces apart by material value in places
            # (e.g. the KBPsK recogniser), so the two must stay different.
            raise SystemExit(f"shift {shift_pawns} makes knight and bishop equal in {phase}")
    return out


def replace_once(text: str, pattern: str, repl: str, what: str) -> str:
    new, n = re.subn(pattern, repl, text, count=1)
    if n != 1:
        raise SystemExit(f"could not find {what} in the source (not Stockfish 11?)")
    return new


def patch(src: Path, key: str, shift: float, pair_scale: float) -> dict:
    types_h, material_cpp, misc_cpp = src / "types.h", src / "material.cpp", src / "misc.cpp"
    t = types_h.read_text()
    for name, value in ORIGINAL.items():
        if not re.search(rf"\b{name}\s*=\s*{value}\b", t):
            raise SystemExit(f"{types_h}: expected {name} = {value}")
    values = new_values(shift)
    for name, value in values.items():
        t = replace_once(t, rf"\b({name}\s*=\s*)\d+", rf"\g<1>{value}", name)

    m = material_cpp.read_text()
    pair_re = r"(\{\s*)" + str(BISHOP_PAIR) + r"(\s*\}\s*,\s*// Bishop pair)"
    if not re.search(pair_re, m):
        raise SystemExit(f"{material_cpp}: expected the bishop-pair term {BISHOP_PAIR}")
    pair = round(BISHOP_PAIR * pair_scale)
    m = replace_once(m, pair_re, rf"\g<1>{pair}\g<2>", "the bishop-pair term")

    v = misc_cpp.read_text()
    if f"const string Version = {VERSION};" not in v:
        raise SystemExit(f"{misc_cpp}: expected Version = {VERSION}")
    # The variant's key goes into its UCI id name, e.g. "Stockfish 11 sf11_b050 64 BMI2",
    # so every game record says which build played.
    v = v.replace(f"const string Version = {VERSION};", f'const string Version = "11 {key}";')

    types_h.write_text(t)
    material_cpp.write_text(m)
    misc_cpp.write_text(v)
    return {"key": key, "bishop_minus_knight_pawns": shift, "bishop_pair_scale": pair_scale,
            "values": {**{k: ORIGINAL[k] for k in ("PawnValueMg", "PawnValueEg")}, **values},
            "bishop_pair_term": pair,
            "original": {**ORIGINAL, "bishop_pair_term": BISHOP_PAIR}}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, manifest = Path(argv[0]), Path(argv[1])
    with open(manifest, "rb") as fh:
        data = tomllib.load(fh)
    variant = data.get("variant")
    if variant is None:
        raise SystemExit(f"{manifest} has no [variant] table")
    key = manifest.parent.name
    summary = patch(src, key, float(variant.get("bishop_minus_knight_pawns", 0.0)),
                    float(variant.get("bishop_pair_scale", 1.0)))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
