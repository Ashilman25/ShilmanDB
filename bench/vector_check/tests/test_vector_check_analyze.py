"""Tests for bench/vector_check/analyze_vector_check.py on hand-made grids and size folders."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent


# Load a module by path under a vc_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, HERE.parent / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("vc_analyze", "analyze_vector_check.py")
vf = mod.vf
vc = mod.vc

SHARES = {
    "date_0.1": 0.001, "date_0.3": 0.003, "date_1": 0.01, "date_3": 0.03, "date_10": 0.1, "date_30": 0.3,
    "none": 1.0, "product_random": 0.002, "product_near": 0.001, "rating_1": 0.14, "rating_2": 0.06,
    "rating_le2": 0.2, "rating_5": 0.6, "category_cases": 0.4, "unverified": 0.05,
}
BAND_A = ("date_0.1", "date_0.3", "product_random", "product_near")


# The family a filter name belongs to.
def family(filt: str) -> str:
    head = filt.split("_")[0]
    return {"unverified": "verified"}.get(head, head)


# One parsed grid row; the plan columns follow the option.
def row(qid, filt, option, ms, recall, half="test", share=0.01, est=None) -> dict:
    kind = "setting" if option.startswith("set_") else "forced"
    scan = "seq" if option in ("exact_seq", "set_noindex") else "bitmap" if option == "exact_bitmap" else "hnsw"
    return {
        "qid": qid, "group": "review", "half": half, "filter": filt, "family": family(filt), "n_pass": 100,
        "share": share, "est_share": share if est is None else est, "option": option, "kind": kind,
        "intended": mod.INTENDED.get(option, ""), "scan": scan, "index": "", "recall": recall, "time_ms": ms,
        "timed_out": False, "deterministic": True,
    }


# A grid from cells given as (qid, filter, half, share, est, {option: (ms, recall)}).
def grid(*cells) -> pd.DataFrame:
    rows = [row(q, f, o, ms, r, h, s, e) for q, f, h, s, e, opts in cells for o, (ms, r) in opts.items()]
    return pd.DataFrame(rows)


# Band B cells: two train and two test, with the given train and test options.
def band_b(train: dict, test: dict) -> pd.DataFrame:
    return grid(*[(q, "date_1", "train", 0.01, 0.01, train) for q in ("q1", "q2")],
                *[(q, "date_1", "test", 0.01, 0.01, test) for q in ("q3", "q4")])


# The band B row of a part 1 result.
def band(result: dict, name: str) -> dict:
    return next(b for b in result["bands"] if b["band"] == name)


# Every bar constant equals the fixed value.
def test_constants_match_spec():
    assert mod.BANDS == (("A", 0.0, 0.005), ("B", 0.005, 0.05), ("C", 0.05, 0.5), ("D", 0.5, float("inf")))
    assert mod.RECALL_BAR == 0.95 and mod.ROOM_BAR == 2.0 and mod.KEEP_SHARE == 0.5
    assert mod.RULE_EDGES == (0.0, 0.001, 0.003, 0.01, 0.03, 0.10, 0.30, 1.0)
    assert mod.TARGETS == (0.90, 0.95, 0.99) and mod.WORK_BAR == 0.70
    assert mod.RECALL_TOL == 0.005 and mod.QUESTION_TOL == 0.01
    assert mod.E_RECALL == 0.995 and mod.E_CHOICES == (500, 1000, 2000)
    assert mod.GATE_SIZE == "500k" and mod.MIN_EXT == (0, 8, 0) and mod.MIN_GATE_QUERIES == 30
    assert mod.EXACT_OPTIONS == ("exact_seq", "exact_bitmap", "set_noindex")
    assert mod.ITERATIVE_COUNT == 10 == len(mod.ITERATIVE)
    assert mod.MEMCHECK_CELLS == 20 and mod.DETERMINISM_QUERIES == 20
    assert mod.TREES == 100 and mod.LEARNING_RATE == 0.1
    assert mod.CHARGE_DIVISOR == 25 and mod.CHARGE_FACTORS == (0, 1, 4)
    assert mod.EPS == 1e-12
    assert len(mod.OPTIONS) == 33 and len(mod.FORCED) == 17 and len(mod.SETTINGS) == 16


# My option copy must match the grid's names, kinds and intended scans.
def test_option_names_match_grid():
    pytest.importorskip("psycopg")
    grid_mod = _load("vc_analyze_grid", "pg_vector_grid.py")
    assert [o.name for o in grid_mod.ALL_OPTIONS] == list(mod.OPTIONS)
    assert [o.name for o in grid_mod.FORCED] == list(mod.FORCED)
    assert all(o.kind == ("forced" if o.name in mod.FORCED else "setting") for o in grid_mod.ALL_OPTIONS)
    assert {o.name: o.intended for o in grid_mod.FORCED} == mod.INTENDED
    assert all(o.intended is None for o in grid_mod.SETTINGS)
    assert [o.name for o in grid_mod.ITERATIVE] == list(mod.ITERATIVE)


# Lower band edges are inclusive; 1.0 and above go in the last bucket.
def test_band_and_bucket_edges():
    assert [mod.band_of(s) for s in (0.0, 0.004, 0.005, 0.05, 0.5, 1.0)] == ["A", "A", "B", "C", "D", "D"]
    assert mod.bucket_of(0.0) == 0 and mod.bucket_of(0.001) == 1 and mod.bucket_of(0.0029) == 1
    assert mod.bucket_of(0.3) == 6 and mod.bucket_of(1.0) == 6 and mod.bucket_of(5.0) == 6


# A NaN or negative estimated share is refused.
@pytest.mark.parametrize("bad", [math.nan, -0.1])
def test_bucket_refuses_nan_and_negative(bad):
    with pytest.raises(ValueError):
        mod.bucket_of(bad)


# A fast setting short of recall in band A loses to a slower one.
def test_baseline_must_meet_every_band():
    g = grid(("q1", "date_0.1", "test", 0.001, 0.001, {"set_default": (1, 0.8), "set_noindex": (10, 1.0)}),
             ("q1", "date_1", "test", 0.01, 0.01, {"set_default": (1, 1.0), "set_noindex": (10, 1.0)}))
    assert mod.best_single_setting(g) == "set_noindex"


# A mean recall of exactly 0.95 qualifies, and equal totals go to the first name.
def test_baseline_boundary_and_tie_by_name():
    opts = {"set_strict_ef40": (5, 0.9), "set_relaxed_ef40": (5, 0.9), "set_noindex": (10, 1.0)}
    g = grid(("q1", "date_1", "test", 0.01, 0.01, opts),
             ("q2", "date_1", "test", 0.01, 0.01, {**opts, "set_strict_ef40": (5, 1.0), "set_relaxed_ef40": (5, 1.0)}))
    assert mod.best_single_setting(g) == "set_relaxed_ef40"


# The ceiling skips the faster option below 0.95 and breaks the time tie by name.
def test_ceiling_picks_fastest_qualifying():
    g = grid(("q1", "date_1", "test", 0.01, 0.01, {
        "exact_seq": (10, 1.0), "hnsw_off_ef40": (1, 0.9),
        "hnsw_strict_ef40": (3, 1.0), "hnsw_relaxed_ef40": (3, 0.96)}))
    c = mod.ceiling_choice(g)
    assert len(c) == 1 and c.iloc[0]["option"] == "hnsw_relaxed_ef40" and c.iloc[0]["time_ms"] == 3


# Buckets 1 and 3 have picks; 2 ties and takes 1's.
# Bucket 4 has cells but nothing qualifies, so it borrows.
def test_rule_neighbour_fallback():
    a = {"hnsw_off_ef40": (1, 1.0), "exact_seq": (5, 1.0)}
    b = {"hnsw_off_ef40": (1, 0.5), "exact_seq": (5, 1.0)}
    none_ok = {"hnsw_off_ef40": (1, 0.5), "exact_seq": (5, 0.5)}
    g = grid(("q1", "date_0.3", "train", 0.002, 0.002, a), ("q1", "date_3", "train", 0.02, 0.02, b),
             ("q1", "date_10", "train", 0.05, 0.05, none_ok))
    rule = mod.fit_rule(g)
    assert rule == {**{i: "hnsw_off_ef40" for i in (0, 1, 2)}, **{i: "exact_seq" for i in (3, 4, 5, 6)}}
    applied = mod.apply_rule(rule, g)
    assert list(applied["option"]) == ["hnsw_off_ef40", "exact_seq", "exact_seq"]


# With no training cells there's no rule, and part 1 still runs.
def test_rule_with_no_training_cells_is_empty():
    g = grid(("q1", "date_1", "test", 0.01, 0.01, {"set_noindex": (10, 1.0), "exact_seq": (5, 1.0)}))
    assert mod.fit_rule(g[g["half"] == "train"]) == {}
    result = mod.part1(g)
    assert result["verdict"] == "NO-GO" and not band(result, "B")["rule_ok"] and band(result, "B")["rule_ms"] is None


# A none cell gets exact_seq's row as exact_bitmap for the rule only.
def test_none_cell_bitmap_takes_exact_seq_for_rule():
    opts = {"exact_seq": (7, 1.0), "hnsw_off_ef40": (2, 1.0), "set_noindex": (7, 1.0)}
    g = grid(("q1", "none", "test", 1.0, 1.0, opts))
    filled = mod.rule_rows(g)
    bitmap = filled[filled["option"] == "exact_bitmap"]
    assert len(bitmap) == 1 and bitmap.iloc[0]["time_ms"] == 7 and bitmap.iloc[0]["recall"] == 1.0
    applied = mod.apply_rule({i: "exact_bitmap" for i in range(7)}, filled)
    assert applied.iloc[0]["time_ms"] == 7
    assert "exact_bitmap" not in set(mod.ceiling_choice(g)["option"])



# Equal ceiling counts in a family go to the first option name.
def test_family_top_pick_breaks_ties_by_name():
    g = grid(("q1", "date_1", "test", 0.01, 0.01, {"hnsw_strict_ef40": (1, 1.0), "hnsw_off_ef40": (2, 1.0)}),
             ("q2", "date_3", "test", 0.03, 0.03, {"hnsw_off_ef40": (1, 1.0), "hnsw_strict_ef40": (2, 1.0)}))
    lines = mod.part1_lines(mod.part1(g), dev=False)
    assert next(line for line in lines if line.startswith("| date |")) == "| date | 2 | hnsw_off_ef40 | 50% |"


# A scan outside the four known ones counts as other; each row sums to 100%.
def test_planner_table_counts_other_scans():
    g = pd.DataFrame({"option": ["set_default"] * 3, "scan": ["hnsw", "bitmap", "tid_scan"]})
    lines = mod.planner_lines(g)
    assert "| setting | seq | bitmap | hnsw | btree_index | other |" in lines
    assert "| set_default | 0% | 33% | 33% | 0% | 33% |" in lines


GO_TRAIN = {"set_noindex": (12, 1.0), "set_default": (1, 0.5), "exact_seq": (12, 1.0),
            "hnsw_strict_ef40": (4, 0.9), "hnsw_relaxed_ef40": (7.2, 1.0)}
GO_TEST = {**GO_TRAIN, "hnsw_strict_ef40": (4, 1.0)}


# Band B: room about 2.1x, the rule keeps 60% of the saving: GO.
def test_part1_go():
    result = mod.part1(band_b(GO_TRAIN, GO_TEST))
    b = band(result, "B")
    assert result["verdict"] == "GO" and result["band"] == "B"
    assert b["baseline_ms"] == 48 and b["ceiling_ms"] == pytest.approx(22.4) and b["room"] == pytest.approx(48 / 22.4)
    assert b["rule_keep"] == pytest.approx(0.6) and b["room_ok"] and b["rule_ok"]
    assert result["baseline"] == "set_noindex" and result["rule_ratio"] == pytest.approx(24 / 14.4)
    assert band(result, "A")["n_cells"] == 0 and not band(result, "A")["room_ok"]


# A slower ceiling ratio of about 1.8x fails the room clause.
def test_part1_no_go_room():
    slow = {"set_noindex": (10, 1.0), "exact_seq": (10, 1.0)}
    result = mod.part1(band_b({**GO_TRAIN, **slow}, {**GO_TEST, **slow}))
    assert result["verdict"] == "NO-GO" and result["band"] is None
    assert band(result, "B")["room"] == pytest.approx(40 / 22.4) and not band(result, "B")["room_ok"]


# The rule's pick drops to recall 0.9 on the test half.
def test_part1_no_go_rule_recall():
    result = mod.part1(band_b(GO_TRAIN, {**GO_TEST, "hnsw_relaxed_ef40": (7.2, 0.9)}))
    b = band(result, "B")
    assert result["verdict"] == "NO-GO" and b["room_ok"] and not b["rule_ok"] and b["rule_recall"] == pytest.approx(0.9)


# The rule keeps only 40% of the saving.
def test_part1_no_go_keep_share():
    result = mod.part1(band_b(GO_TRAIN, {**GO_TEST, "hnsw_relaxed_ef40": (8.8, 1.0)}))
    b = band(result, "B")
    assert result["verdict"] == "NO-GO" and b["room_ok"] and b["rule_keep"] == pytest.approx(0.4) and not b["rule_ok"]


# Room of exactly 2.0 and a keep share of exactly 0.5 both pass.
def test_part1_exact_boundaries_pass():
    base = {"set_noindex": (10, 1.0), "set_default": (1, 0.5), "exact_seq": (10, 1.0)}
    train = {**base, "hnsw_strict_ef40": (5, 0.9), "hnsw_relaxed_ef40": (5, 1.0)}
    test = {**base, "hnsw_strict_ef40": (5, 1.0), "hnsw_relaxed_ef40": (7.5, 1.0)}
    result = mod.part1(band_b(train, test))
    b = band(result, "B")
    assert b["room"] == 2.0 and b["rule_keep"] == 0.5 and result["verdict"] == "GO"


# stopping.csv rows for the bars; learned work_1x = work_ratio x REM's 1000.
def stop_frame(learned_recall=None, work_ratio=0.5, q_learned=0.96) -> pd.DataFrame:
    rows = []
    for t in mod.TARGETS:
        lr = t if learned_recall is None else learned_recall[t]
        values = {
            ("test", "rem"): (t + 0.004, 1000, 0), ("test", "learned"): (lr, 1000 * work_ratio - 20, 5),
            ("test", "plain"): (0.999, 5000, 0), ("questions", "rem"): (0.96, 900, 0),
            ("questions", "learned"): (q_learned, 800, 6), ("questions", "plain"): (0.99, 4000, 0),
        }
        for (s, m), (rec, ndis, calls) in values.items():
            rows.append({"set": s, "target": t, "method": m, "param": "x",
                         "margin_ok": "true" if m == "learned" else "",
                         "mean_recall": rec, "mean_ndis": ndis, "mean_calls": calls, "work_1x": ndis + 4 * calls,
                         "below_target_share": 0.1, "n_queries": 100})
    return pd.DataFrame(rows)


# GO, then recall short by 0.006, work at 0.75x, questions 0.02 below REM.
def test_part2_go_and_each_clause_failing():
    go = mod.part2(stop_frame())
    assert go["verdict"] == "GO" and all(t["recall_ok"] and t["work_ok"] and t["questions_ok"] for t in go["targets"])
    first = go["targets"][0]
    assert first["test"]["learned"]["work"] == {0: 480, 1: 500, 4: 560}
    assert first["vs_rem"][1] == pytest.approx(0.5) and first["vs_plain"][1] == pytest.approx(0.1)

    short = mod.part2(stop_frame(learned_recall={t: round(t - 0.006, 3) for t in mod.TARGETS}))
    assert short["verdict"] == "NO-GO" and not short["targets"][1]["recall_ok"]
    heavy = mod.part2(stop_frame(work_ratio=0.75))
    assert heavy["verdict"] == "NO-GO" and not heavy["targets"][0]["work_ok"]
    unsafe = mod.part2(stop_frame(q_learned=0.94))
    assert unsafe["verdict"] == "NO-GO" and not unsafe["targets"][2]["questions_ok"]


# Recall exactly target - 0.005, work exactly 0.70x and questions exactly REM - 0.01 pass.
def test_part2_exact_boundaries_pass():
    edge = {0.90: 0.895, 0.95: 0.945, 0.99: 0.985}
    result = mod.part2(stop_frame(learned_recall=edge, work_ratio=0.70, q_learned=0.95))
    assert result["verdict"] == "GO"


# A missing learned row can't pass.
def test_part2_missing_row_is_no_go():
    s = stop_frame()
    s = s[~((s["set"] == "test") & (s["method"] == "learned") & (s["target"] == 0.95))]
    assert mod.part2(s)["verdict"] == "NO-GO"


# ---- size folders on disk ----

# The current cell set's queries at this size and N, as (qid, group, half).
def queries(size: str, n: int) -> list:
    q = vc.read_questions()
    tests = [f"r{i:03d}" for i in range(1, 6)]
    written = q.loc[q["group"] == "written", "qid"].tolist()
    ai53 = q.loc[q["group"] == "ai53", "qid"].tolist()
    return vf.part1_queries(tests, written, ai53, size, n_per_group=n)


# The plan a grid row records for an option on a filter.
def plan(size: str, option: str, filt: str) -> tuple[str, str]:
    if option == "exact_bitmap":
        return "bitmap", f"reviews_{size}_{mod.FAMILY_COLUMNS[family(filt)]}_idx"
    if option in ("exact_seq", "set_noindex"):
        return "seq", ""
    return "hnsw", f"reviews_{size}_hnsw_idx"


# Write a JSON file.
def put_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2))


# A complete, clean size folder under results, with the query split it needs.
def write_size(results: Path, size: str = "70k", n: int = 2, limits=None) -> Path:
    (results / "data").mkdir(parents=True, exist_ok=True)
    split = pd.DataFrame({"qid": [f"r{i:03d}" for i in range(1, 9)], "split": ["test"] * 5 + ["train"] * 3})
    split.to_csv(results / "data" / "query_reviews.csv", index=False)
    out = results / f"size{size}"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for qid, group, half in queries(size, n):
        for filt in vf.FILTER_NAMES:
            for i, option in enumerate(o for o in mod.OPTIONS if not (o == "exact_bitmap" and filt == "none")):
                scan, index = plan(size, option, filt)
                ms = 1.0 + i * 0.1 + (5.0 if option.startswith("exact") or option == "set_noindex" else 0.0)
                rows.append({
                    "size": size, "qid": qid, "group": group, "half": half, "filter": filt, "family": family(filt),
                    "n_pass": 100, "share": SHARES[filt], "est_rows": 90.0, "est_share": SHARES[filt], "option": option,
                    "kind": "setting" if option.startswith("set_") else "forced",
                    "intended": mod.INTENDED.get(option, ""),
                    "scan": scan, "index": index, "nodes": "Limit", "reused_from": "", "recall": 1.0, "time_ms": ms,
                    "t1": ms, "t2": ms, "t3": ms, "n_returned": 10, "timed_out": "False", "deterministic": "True",
                    "ids": "1 2", "filter_sql": "",
                })
    pd.DataFrame(rows).to_csv(out / "grid.csv", index=False)
    mem = [(q, f, o, "True") for q, _, _ in queries(size, n) for f in BAND_A for o in mod.ITERATIVE]
    pd.DataFrame(mem, columns=["qid", "filter", "option", "ids_equal"]).to_csv(out / "memcheck.csv", index=False)
    put_json(out / "pg_load.json", {"server_version": "17.11 (Homebrew)", "extversion": "0.8.7"})
    put_json(out / "grid_info.json", {"size": size, "n_per_group": n, "cells": n * 30,
                                      "server_version": "17.11 (Homebrew)", "extversion": "0.8.7"})
    rem = [(s, f"{s[0]}{i:02d}", ef, 0.9 if ef == 10 else 1.0, ef * 10)
           for s, k in (("validation", 20), ("test", 5)) for i in range(k) for ef in (10, 100)]
    pd.DataFrame(rem, columns=["set", "qid", "ef", "recall", "ndis"]).to_csv(out / "rem.csv", index=False)
    det = pd.DataFrame({"qid": [f"v{i:02d}" for i in range(20)], "same_ids": "True", "same_ndis": "True"})
    det.to_csv(out / "determinism.csv", index=False)
    lim = {} if limits is None else limits
    put_json(out / "plain_E.json", {"E": 1000, "val_recall": {"500": 0.99, "1000": 0.997}, "limits": lim})
    targets = {str(t): {"dists_t": 500.0, "ipi": 250, "mpi": 50} for t in mod.TARGETS}
    put_json(out / "train_info.json",
             {"E": 1000, "trees": 100, "learning_rate": 0.1, "limits": lim, "targets": targets})
    s = stop_frame()
    s = pd.concat([s.replace({"test": "validation"}).query("set == 'validation'"), s], ignore_index=True)
    for f in (0, 4):
        s[f"work_{f}x"] = s["mean_ndis"] + 4 * f * s["mean_calls"]
    s.to_csv(out / "stopping.csv", index=False)
    sq = [("test", f"t{i}", t, m, 1.0, 100 + i, 2 if m == "learned" else 0)
          for t in mod.TARGETS for m in mod.METHODS for i in range(5)]
    sq = pd.DataFrame(sq, columns=["set", "qid", "target", "method", "recall", "ndis", "calls"])
    sq.to_csv(out / "stopping_queries.csv", index=False)
    return out


# Read a CSV as strings, edit it, write it back.
def edit_csv(path: Path, fn) -> None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    fn(df).to_csv(path, index=False)


# Set column col to value on the first row matching every key=value.
def set_first(path: Path, col: str, value: str, **where) -> None:
    # Set the value on the first matching row.
    def fn(df):
        hit = pd.Series(True, index=df.index)
        for k, v in where.items():
            hit &= df[k] == v
        df.loc[hit[hit].index[0], col] = value
        return df
    edit_csv(path, fn)


# Rewrite one key of a JSON file.
def set_json(path: Path, key: str, value) -> None:
    data = json.loads(path.read_text())
    data[key] = value
    put_json(path, data)


# Drop the first row matching every key=value.
def drop_first(path: Path, **where) -> None:
    # Drop the first matching row.
    def fn(df):
        hit = pd.Series(True, index=df.index)
        for k, v in where.items():
            hit &= df[k] == v
        return df.drop(index=hit[hit].index[0])
    edit_csv(path, fn)


# The clean folder gives no warning and all 60 cells.
def test_clean_size_has_no_warnings(tmp_path):
    out = write_size(tmp_path, limits={"train": 200, "validation": 100, "test": 100})
    assert mod.validate(out, "70k", None) == []
    d = mod.load_size(out, "70k", None)
    assert len(d["grid"].drop_duplicates(["qid", "filter"])) == 60 and d["dropped"] == 0 and d["timeouts"] == 0


# Each mutation of a clean folder gives a warning containing its words.
CASES = {
    "ext_below": (lambda o: set_json(o / "pg_load.json", "extversion", "0.7.4"), "below 0.8.0"),
    "grid_info_ext": (lambda o: set_json(o / "grid_info.json", "extversion", "0.8.6"), "grid_info.json extversion"),
    "grid_info_size": (lambda o: set_json(o / "grid_info.json", "size", "500k"), "grid_info.json size"),
    "grid_info_n_text": (lambda o: set_json(o / "grid_info.json", "n_per_group", "two"), "n_per_group"),
    "grid_info_n_refused": (lambda o: set_json(o / "grid_info.json", "n_per_group", 0), "n_per_group"),
    "exact_seq_recall": (lambda o: set_first(o / "grid.csv", "recall", "0.9", option="exact_seq"), "exact option rows"),
    "exact_bitmap_recall": (lambda o: set_first(o / "grid.csv", "recall", "0.9", option="exact_bitmap"),
                            "exact option rows"),
    "set_noindex_recall": (lambda o: set_first(o / "grid.csv", "recall", "0.9", option="set_noindex"),
                           "exact option rows"),
    "hnsw_changed": (lambda o: set_first(o / "grid.csv", "deterministic", "False", option="hnsw_off_ef40"),
                     "changed between"),
    "setting_hnsw_changed": (lambda o: set_first(o / "grid.csv", "deterministic", "False", option="set_default"),
                             "changed between"),
    "forced_scan": (lambda o: set_first(o / "grid.csv", "scan", "seq", option="hnsw_strict_ef40"), "intended plan"),
    "forced_hnsw_index": (lambda o: set_first(o / "grid.csv", "index", "reviews_70k_rating_idx",
                                              option="hnsw_off_ef40"), "intended plan"),
    "bitmap_index": (lambda o: set_first(o / "grid.csv", "index", "reviews_70k_rating_idx",
                                         option="exact_bitmap", filter="date_1"), "intended plan"),
    "seq_index": (lambda o: set_first(o / "grid.csv", "index", "reviews_70k_hnsw_idx", option="exact_seq"),
                  "intended plan"),
    "kind_wrong": (lambda o: set_first(o / "grid.csv", "kind", "forced", option="set_default"), "kind or intended"),
    "missing_option": (lambda o: drop_first(o / "grid.csv", option="hnsw_off_ef100"), "miss an option"),
    "option_twice": (lambda o: edit_csv(o / "grid.csv", lambda df: pd.concat([df, df.iloc[[0]]])), "an option twice"),
    "half_differs": (lambda o: edit_csv(o / "grid.csv", lambda df: df.assign(
        half=["train" if h == "test" else "test" for h in df["half"]])), "half"),
    "not_a_number": (lambda o: set_first(o / "grid.csv", "time_ms", "fast", option="exact_seq"), "isn't a number"),
    "not_a_bool": (lambda o: set_first(o / "grid.csv", "timed_out", "yes", option="exact_seq"), "isn't True or False"),
    "no_ceiling": (lambda o: edit_csv(o / "grid.csv", lambda df: df.assign(
        timed_out=["True" if (q, f) == ("r001", "date_1") else t
                   for q, f, t in zip(df["qid"], df["filter"], df["timed_out"])],
        recall=["0" if (q, f) == ("r001", "date_1") else r
                for q, f, r in zip(df["qid"], df["filter"], df["recall"])])), "no option reaching"),
    "memcheck_missing": (lambda o: (o / "memcheck.csv").unlink(), "memcheck.csv is missing"),
    "memcheck_false": (lambda o: set_first(o / "memcheck.csv", "ids_equal", "False"), "different ids at 512MB"),
    "memcheck_option": (lambda o: drop_first(o / "memcheck.csv", option="hnsw_relaxed_ef40"), "iterative options"),
    "memcheck_cells": (lambda o: edit_csv(o / "memcheck.csv", lambda df: df[df["filter"] != "date_0.1"]),
                       "memcheck.csv covers"),
    "memcheck_outside_a": (lambda o: edit_csv(o / "memcheck.csv", lambda df: df.replace({"date_0.1": "date_1"})),
                           "memcheck.csv covers"),
    "determinism_false": (lambda o: set_first(o / "determinism.csv", "same_ndis", "False"), "determinism.csv"),
    "determinism_short": (lambda o: edit_csv(o / "determinism.csv", lambda df: df.head(10)), "determinism.csv"),
    "E_null": (lambda o: set_json(o / "plain_E.json", "E", None), "E is null"),
    "E_not_choice": (lambda o: set_json(o / "plain_E.json", "E", 700), "not one of"),
    "E_recall_low": (lambda o: set_json(o / "plain_E.json", "val_recall", {"1000": 0.99}), "below 0.995"),
    "trees": (lambda o: set_json(o / "train_info.json", "trees", 50), "trees"),
    "learning_rate": (lambda o: set_json(o / "train_info.json", "learning_rate", 0.2), "learning_rate"),
    "stopping_row_missing": (lambda o: drop_first(o / "stopping.csv", set="questions", method="plain"), "stopping.csv"),
    "stopping_row_twice": (lambda o: edit_csv(o / "stopping.csv", lambda df: pd.concat([df, df.tail(1)])),
                           "stopping.csv"),
    "stopping_work": (lambda o: set_first(o / "stopping.csv", "work_1x", "123", set="test", method="learned"),
                      "work_1x"),
    "rem_missing": (lambda o: (o / "rem.csv").unlink(), "rem.csv is missing"),
    "stopping_queries_missing": (lambda o: (o / "stopping_queries.csv").unlink(), "stopping_queries.csv is missing"),
    "query_reviews_missing": (lambda o: (o.parent / "data" / "query_reviews.csv").unlink(),
                              "query_reviews.csv is missing"),
    "unexpected_option": (lambda o: edit_csv(o / "grid.csv", lambda df: pd.concat(
        [df, df.head(1).assign(option="hnsw_magic")])), "an unexpected option"),
    "no_baseline": (lambda o: edit_csv(o / "grid.csv", lambda df: df.assign(recall=[
        "0.5" if opt.startswith("set_") and f == "date_1" else r
        for opt, f, r in zip(df["option"], df["filter"], df["recall"])])),
        "no setting reaches mean recall 0.95 in every band"),
}


# One mutation, one warning that names the size.
@pytest.mark.parametrize("case", list(CASES))
def test_validate_each_trigger(tmp_path, case):
    out = write_size(tmp_path)
    mutate, words = CASES[case]
    mutate(out)
    found = mod.validate(out, "70k", None)
    assert found and any(words in w for w in found), found
    assert all(w.startswith("70k: ") for w in found)


# A pgvector version that differs from the other size's warns.
def test_validate_other_size_extversion(tmp_path):
    out = write_size(tmp_path)
    assert any("differs from the other size" in w for w in mod.validate(out, "70k", "0.8.6"))


# Timed-out HNSW rows say deterministic False by design; they don't warn.
def test_timed_out_hnsw_row_does_not_warn(tmp_path):
    out = write_size(tmp_path)

    # Time out the first hnsw_off_ef40 row as the grid writes one.
    def fn(df):
        hit = df.index[df["option"] == "hnsw_off_ef40"][0]
        df.loc[hit, ["timed_out", "deterministic", "recall", "time_ms"]] = ["True", "False", "0", "10000"]
        return df
    edit_csv(out / "grid.csv", fn)
    assert mod.validate(out, "70k", None) == []
    assert mod.load_size(out, "70k", None)["timeouts"] == 1



# A setting whose plan isn't HNSW may differ between runs without a warning.
@pytest.mark.parametrize("option, scan", [("set_noindex", "seq"), ("set_default", "bitmap")])
def test_non_hnsw_setting_change_does_not_warn(tmp_path, option, scan):
    out = write_size(tmp_path)

    # Give the first row of this setting a non-HNSW plan that changed between runs.
    def fn(df):
        hit = df.index[df["option"] == option][0]
        df.loc[hit, ["scan", "index", "deterministic"]] = [scan, "", "False"]
        return df
    edit_csv(out / "grid.csv", fn)
    assert mod.validate(out, "70k", None) == []


# Rows outside the current cell set are dropped and counted, not warned about.
def test_rows_outside_cell_set_are_dropped(tmp_path):
    out = write_size(tmp_path)
    edit_csv(out / "grid.csv", lambda df: pd.concat([df, df.head(33).assign(qid="r999")]))
    assert mod.validate(out, "70k", None) == []
    assert mod.load_size(out, "70k", None)["dropped"] == 33


# Smoke limits are fine at 70K and warn twice at 500K.
def test_limits_warn_only_at_gate_size(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MIN_GATE_QUERIES", 1)
    lim = {"train": 200, "validation": 100, "test": 100}
    assert mod.validate(write_size(tmp_path, "70k", limits=lim), "70k", None) == []
    found = mod.validate(write_size(tmp_path, "500k", limits=lim), "500k", None)
    assert len(found) == 2 and all("limits" in w for w in found)


# At 500K fewer than 30 queries per group warns.
def test_gate_needs_30_queries_per_group(tmp_path):
    found = mod.validate(write_size(tmp_path, "500k", n=2), "500k", None)
    assert found == ["500k: grid_info.json n_per_group 2 is below the 30 queries per group the gate needs"]


# Only 70K: both parts pending, every output file written, exit 0.
def test_only_70k_reports_pending(tmp_path):
    write_size(tmp_path, limits={"train": 200, "validation": 100, "test": 100})
    rc = mod.main(["--results", str(tmp_path)])
    text = (tmp_path / "report.md").read_text()
    assert rc == 0 and "**Part 1: PENDING 500K**" in text and "**Part 2: PENDING 500K**" in text
    assert "development" in text
    assert "**WARNING**" not in text and text.splitlines()[0] == "# Vector-search check"
    cells = pd.read_csv(tmp_path / "cells.csv")
    assert len(cells) == 60 and set(cells["size"].astype(str)) == {"70k"} and cells["ceiling_option"].notna().all()
    stop = pd.read_csv(tmp_path / "stopping.csv", dtype=str)
    assert list(stop.columns[:2]) == ["size", "set"] and len(stop) == 27
    assert (tmp_path / "part1_70k.png").exists() and (tmp_path / "part2_70k.png").exists()


# A 70K-only warning still makes both headlines NO VERDICT and exits 1.
def test_warning_gives_no_verdict_and_exit_1(tmp_path):
    out = write_size(tmp_path)
    set_json(out / "plain_E.json", "E", None)
    rc = mod.main(["--results", str(tmp_path)])
    lines = (tmp_path / "report.md").read_text().splitlines()
    assert rc == 1 and lines[0] == "# Vector-search check" and lines[2].startswith("**WARNING** 70k: ")
    assert "**Part 1: NO VERDICT**" in lines and "**Part 2: NO VERDICT**" in lines
    assert (tmp_path / "cells.csv").exists()


# With a clean 500K, the headlines are 500K's own readings.
def test_clean_500k_gives_its_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MIN_GATE_QUERIES", 1)
    write_size(tmp_path, "70k")
    out = write_size(tmp_path, "500k")
    rc = mod.main(["--results", str(tmp_path)])
    text = (tmp_path / "report.md").read_text()
    d = mod.load_size(out, "500k", "0.8.7")
    p1, p2 = mod.part1(d["grid"])["verdict"], mod.part2(d["stopping"])["verdict"]
    assert rc == 0 and f"**Part 1: {p1}**" in text and f"**Part 2: {p2}**" in text and p2 == "GO"
    assert "## 70K (development, not a verdict)" in text and "## 500K\n" in text
    assert len(pd.read_csv(tmp_path / "cells.csv")) == 120


# A size folder with only truth or filter files is absent, not warned about.
def test_folder_without_inputs_is_absent(tmp_path):
    write_size(tmp_path, "70k")
    (tmp_path / "size500k").mkdir()
    (tmp_path / "size500k" / "filters.csv").write_text("qid\n")
    assert mod.main(["--results", str(tmp_path)]) == 0
    assert "**Part 1: PENDING 500K**" in (tmp_path / "report.md").read_text()


# A present size warns once per missing input.
def test_present_size_warns_per_missing_input(tmp_path):
    out = tmp_path / "size500k"
    out.mkdir()
    (out / "grid_info.json").write_text(json.dumps({"size": "500k", "n_per_group": 40}))
    found = mod.validate(out, "500k", None)
    for name in mod.INPUTS:
        if name != "grid_info.json":
            assert sum(f"{name} is missing" in w for w in found) == 1, (name, found)


# The report text from a heading to the next level-2 heading.
def section(text: str, heading: str) -> str:
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    return text[start:] if end < 0 else text[start:end]


# A warned 500K section carries no bold GO or NO-GO under its NO VERDICT headline.
def test_warned_500k_section_has_no_bold_verdict(tmp_path):
    write_size(tmp_path, "500k", n=2)
    rc = mod.main(["--results", str(tmp_path)])
    text = (tmp_path / "report.md").read_text()
    body = section(text, "## 500K (no verdict: see the warnings)")
    assert rc == 1 and "**Part 1: NO VERDICT**" in text and "**Part 2: NO VERDICT**" in text
    assert "**GO" not in body and "**NO-GO" not in body
    assert body.count("Reading (no verdict: see the warnings): ") == 2


# Clean 500K with a warned 70K gives no verdict; the warning names 70K.
def test_clean_500k_with_warned_70k_gives_no_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "MIN_GATE_QUERIES", 1)
    set_json(write_size(tmp_path, "70k") / "plain_E.json", "E", None)
    write_size(tmp_path, "500k")
    rc = mod.main(["--results", str(tmp_path)])
    text = (tmp_path / "report.md").read_text()
    lines = text.splitlines()
    assert rc == 1 and "**Part 1: NO VERDICT**" in lines and "**Part 2: NO VERDICT**" in lines
    assert any(line.startswith("**WARNING** 70k: ") for line in lines)
    assert not any(line.startswith("**WARNING** 500k") for line in lines)
    body = section(text, "## 500K (no verdict: see the warnings)")
    assert "**GO" not in body and "**NO-GO" not in body
