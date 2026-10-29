"""Tests for how evaluate_sp picks the evaluation of each search."""

import pandas as pd

from evaluate_sp import evaluation_row, evaluation_table


def line(depth, cp, bound, final=False, pv=1):
    return {"multipv_index": pv, "depth": depth, "cp_white": cp, "bound": bound,
            "is_final": final}


def test_evaluation_row_skips_trailing_bound():
    rows = [line(10, 20, "exact"), line(11, 25, "exact"), line(12, 40, "lower", final=True)]
    assert evaluation_row(rows)["cp_white"] == 25


def test_evaluation_row_falls_back_to_last_line():
    rows = [line(1, 5, "upper"), line(1, 3, "lower", final=True)]
    assert evaluation_row(rows)["cp_white"] == 3


def raw_frame(lines_by_budget):
    records = []
    for budget, lines in lines_by_budget.items():
        for depth, cp, bound, final in lines:
            records.append({
                "engine": "x", "engine_version": "1", "position_id": "start",
                "design": "start", "fen": "", "multipv_setting": 1,
                "multipv_index": 1, "budget_nodes": budget, "depth": depth,
                "seldepth": depth, "nodes": depth * 100, "time_s": 0.1, "bound": bound,
                "cp_white": cp, "mate_white": None, "wdl_w": 100, "wdl_d": 800,
                "wdl_l": 100, "expected_score": 0.5 + cp / 1000, "pv": "e2e4",
                "bestmove": "e2e4", "search_wall_s": 0.1, "is_final": final,
            })
    return pd.DataFrame(records)


def test_evaluation_table_uses_last_exact_and_keeps_last_line():
    raw = raw_frame({
        1000: [(5, 10, "exact", False), (6, 12, "exact", False), (7, 30, "upper", True)],
        10000: [(8, 15, "exact", False), (9, 16, "exact", True)],
        100: [(1, 50, "lower", True)],
    })
    t = evaluation_table(raw).set_index("budget_nodes")
    assert len(t) == 3
    assert t.loc[1000, "cp_white"] == 12 and t.loc[1000, "depth"] == 6
    assert t.loc[1000, "last_cp_white"] == 30 and t.loc[1000, "last_bound"] == "upper"
    assert t.loc[1000, "eval_source"] == "last_exact"
    assert t.loc[10000, "cp_white"] == 16 and t.loc[10000, "last_bound"] == "exact"
    assert t.loc[100, "eval_source"] == "last_line_bound" and t.loc[100, "cp_white"] == 50
