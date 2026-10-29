"""End-to-end tests of the analyses on synthetic games."""

import numpy as np
import pandas as pd

import analyse_bias
import analyse_removal as ar
import positions as P


def synthetic_games(match: str, focal_edge: float, white_edge: float, positions, seed=0,
                    lines=10, per_line=10) -> pd.DataFrame:
    """Self-play games in which the focal side scores 0.5 + focal_edge and
    White scores white_edge more (draws for half the rest)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pid in positions:
        p = P.get_position(pid)
        for line in range(lines):
            for k in range(per_line):
                white_is_focal = p.focal is None or p.focal
                e = 0.5 + white_edge + (focal_edge if white_is_focal else -focal_edge)
                if p.focal is None:
                    e = 0.5 + white_edge
                u = rng.random()
                score = 1.0 if u < e - 0.25 else 0.5 if u < e + 0.25 else 0.0
                rows.append({"match": match, "player": match, "opponent": "", "batch": "",
                             "position_id": pid, "line_index": line, "score_white": score,
                             "white_player": f"{match}_{'ab'[k % 2]}", "white_engine": match,
                             "termination": "normal", "plies": 100, "moves_hash": rng.integers(1e9),
                             "fen_matches_position": True})
    return pd.DataFrame(rows)


def test_removal_analysis_recovers_the_focal_edge(tmp_path):
    ids = P.parse_position_spec("key")
    games = ar.add_focal(synthetic_games("eng", focal_edge=0.08, white_edge=0.03, positions=ids))
    out = ar.run(games, None, tmp_path, None)
    comp = out["comparisons"].set_index("comparison_id")
    key = comp[comp.design.isin(ar.KEY_DESIGNS)]
    # Pooling twins cancels White's edge: focal E ~ 0.58, not 0.61 or 0.55.
    assert abs(key.escore.mean() - 0.58) < 0.02
    controls = comp[comp.mirrored_control]
    assert abs(controls.escore.mean() - 0.53) < 0.03            # White's edge in controls
    d = out["designs"].set_index("design")
    assert {"BN v NN", "BB v BN", "BB v NN"} <= set(d.index)
    assert "pawns" in d and d.loc["BN v NN", "pawns"] > 0
    assert len(out["additivity"]) == 1
    assert (tmp_path / "removal_summary.txt").exists()


def test_removal_quality_flags_repeated_games(tmp_path):
    games = synthetic_games("det", 0.0, 0.0, ["start", "f2", "f7"])
    games["moves_hash"] = games.line_index            # every line replays one game
    qc, _ = ar.quality_checks(ar.add_focal(games), None)
    assert qc.repeated_game_share.iloc[0] > 0.8


def test_bias_scores_the_variant(tmp_path):
    rows = []
    rng = np.random.default_rng(3)
    for line in range(20):
        for k in range(10):
            variant_white = k % 2 == 0
            win = rng.random() < 0.7
            score_variant = 1.0 if win else 0.0
            rows.append({"match": "sf11_b050__vs__sf11", "player": "sf11_b050", "opponent": "sf11",
                         "batch": "", "position_id": "start", "line_index": line,
                         "score_white": score_variant if variant_white else 1 - score_variant,
                         "white_player": "sf11_b050" if variant_white else "sf11",
                         "white_engine": "sf11_b050" if variant_white else "sf11",
                         "termination": "normal"})
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame(rows).to_parquet(tables / "selfplay_games.parquet")
    post, _ = analyse_bias.variant_posteriors(analyse_bias.load_games(tables))
    r = post.iloc[0]
    assert r.display_name == "Bobby Bischer"
    assert abs(r.escore - 0.7) < 0.05 and r.p_positive > 0.99
