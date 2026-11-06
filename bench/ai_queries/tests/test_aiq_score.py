import json
import math

import numpy as np
import pandas as pd
import pytest

from aiq_world import make_world
from bench.ai_queries import run_check, score
from bench.ai_queries.approaches import Env, QueryRun
from bench.ai_queries.costs import Usage, Weights
from bench.ai_queries.labels import label_columns
from bench.ai_queries.queries import Truth
from bench.ai_queries.questions import QUESTIONS

W = Weights(small=0.1, proxy=1e-6, embed=0.01)
NAIVE = {q: 1000.0 for q in range(1, 11)}


def runs_for(approach: str, query: int, met_count: int, cost: float, seeds: int = 100) -> pd.DataFrame:
    """`seeds` runs of one approach on one query, the first `met_count` of them meeting the target."""
    return pd.DataFrame({
        "approach": approach, "query": query, "seed": range(1, seeds + 1), "cost": cost, "big": cost,
        "small": 0, "error": 0.0, "met": [i < met_count for i in range(seeds)]})


def row_of(summary: pd.DataFrame, approach: str, query: int) -> pd.Series:
    return summary[(summary["approach"] == approach) & (summary["query"] == query)].iloc[0]


# ---- summarize and the fallback ----


def test_met_needs_95_of_100():
    runs = pd.concat([runs_for("a", 1, 95, 10.0), runs_for("a", 2, 94, 10.0)])
    summary = score.summarize(runs, NAIVE, 100)
    assert row_of(summary, "a", 1)["met"] and row_of(summary, "a", 1)["met_runs"] == 95
    assert not row_of(summary, "a", 2)["met"]


def test_fallback_charges_naive():
    runs = pd.concat([runs_for("a", 1, 94, 10.0), runs_for("a", 2, 100, 10.0)])
    summary = score.summarize(runs, NAIVE, 100)
    assert row_of(summary, "a", 1)["median_cost"] == 10.0
    assert row_of(summary, "a", 1)["charged"] == 1000.0  # unmet: naive's cost, though cheaper
    assert row_of(summary, "a", 2)["charged"] == 10.0


def summary_of(snow, bq) -> pd.DataFrame:
    """One query; each variant is (met_runs, cost)."""
    runs = pd.concat([runs_for("incumbent_snowflake", 1, *snow), runs_for("incumbent_bigquery", 1, *bq)])
    return score.summarize(runs, NAIVE, 100)


def test_incumbent_is_cheaper_met_variant():
    best = score.incumbent_best(summary_of((100, 300.0), (100, 200.0))).iloc[0]
    assert best["cost"] == 200.0 and best["variant"] == "incumbent_bigquery"
    best = score.incumbent_best(summary_of((100, 300.0), (50, 200.0))).iloc[0]  # bigquery unmet
    assert best["cost"] == 300.0 and best["variant"] == "incumbent_snowflake"
    best = score.incumbent_best(summary_of((10, 300.0), (50, 200.0))).iloc[0]  # neither met
    assert best["cost"] == 1000.0 and best["variant"] == "naive"


# ---- bars ----


def made_summary(shilman: dict[int, float], inc: float = 240.0, met: bool = True) -> pd.DataFrame:
    """Both incumbent variants met at `inc`; shilmandb met at the given cost per query."""
    runs = [runs_for(a, q, 100, inc) for a in ("incumbent_snowflake", "incumbent_bigquery") for q in NAIVE]
    runs += [runs_for("shilmandb", q, 100 if met else 0, shilman[q]) for q in NAIVE]
    return score.summarize(pd.concat(runs), NAIVE, 100)


def outcome(shilman, agreement=None, inc=240.0):
    summary = made_summary(shilman, inc)
    return score.bars(summary, score.incumbent_best(summary), NAIVE, agreement)


ALL_GOOD = {q: 1.0 for q in QUESTIONS}


def plan(edge: int, cheap: int, rest: float = 250.0) -> dict[int, float]:
    """Queries 1..edge at 70 (under naive / 10 and incumbent / 3), up to `cheap` at 100 (under naive / 10 only).

    The incumbent is 240, so incumbent / 3 is 80 and 1.1 x the incumbent is 264.
    """
    return {q: 70.0 if q <= edge else (100.0 if q <= cheap else rest) for q in NAIVE}


def test_bars_outcomes():
    go = plan(5, 7)  # 7 under naive / 10, 5 under incumbent / 3
    b = outcome(go, ALL_GOOD)
    assert (b["bar1"], b["bar2"], b["bar3"], b["outcome"]) == (True, True, True, "go")

    low = {**ALL_GOOD, "F3": 0.89}
    assert outcome(go, low)["outcome"] == "go pending a paid run" and outcome(go, low)["bar3"] is False

    six = plan(5, 6)
    b = outcome(six, ALL_GOOD)
    assert b["bar1"] is False and b["outcome"] == "no-go"

    four = plan(4, 7)  # bar 1 ok, 4 of 10 at incumbent / 3
    b = outcome(four, ALL_GOOD)
    assert b["bar1"] is True and b["bar2"] is False and b["outcome"] == "no-go"

    slow = {**go, 10: 265.0}  # just over 1.1 x the incumbent on one query
    b = outcome(slow, ALL_GOOD)
    assert b["bar2"] is False and b["outcome"] == "no-go"
    ok = {**go, 10: 264.0}    # 1.1 x is allowed
    assert outcome(ok, ALL_GOOD)["bar2"] is True


def test_bar3_none_without_agreement():
    go = plan(5, 7)
    b = outcome(go, None)
    assert b["bar3"] is None and b["bar1"] and b["bar2"] and b["outcome"] == "go pending a paid run"
    bad = plan(0, 0)
    b = outcome(bad, None)
    assert b["bar3"] is None and b["outcome"] == "no-go"


def test_unmet_shilmandb_is_charged_naive():
    b = score.bars(*(lambda s: (s, score.incumbent_best(s)))(made_summary({q: 1.0 for q in NAIVE}, met=False)),
                   NAIVE, ALL_GOOD)
    assert b["bar1"] is False and b["outcome"] == "no-go"
    json.dumps(b)  # the result can be written as JSON


def test_incumbent_cost_is_capped_at_naive():
    # a met variant that costs more than naive is replaced by naive's cost
    summary = summary_of((100, 1500.0), (100, 1600.0))
    best = score.incumbent_best(summary, NAIVE).iloc[0]
    assert best["cost"] == 1000.0 and best["variant"] == "naive"
    best = score.incumbent_best(summary_of((100, 700.0), (100, 1600.0)), NAIVE).iloc[0]
    assert best["cost"] == 700.0


def test_unmet_queries_do_not_count_toward_bars():
    # queries 1-5: shilmandb unmet; snowflake met at 3,500 against naive 1,000
    runs = [runs_for("incumbent_snowflake", q, 100, 3500.0 if q <= 5 else 240.0) for q in NAIVE]
    runs += [runs_for("incumbent_bigquery", q, 0, 1.0) for q in NAIVE]
    runs += [runs_for("shilmandb", q, 0 if q <= 5 else 100, 50.0) for q in NAIVE]
    summary = score.summarize(pd.concat(runs), NAIVE, 100)
    best = score.incumbent_best(summary, NAIVE)
    assert list(best["cost"][:5]) == [1000.0] * 5
    b = score.bars(summary, best, NAIVE, ALL_GOOD)
    assert b["bar1_queries"] == 5 and b["bar2_queries"] == 5 and b["bar1"] is False
    assert b["outcome"] == "no-go"


def test_p95_big_in_summary():
    runs = runs_for("a", 1, 100, 10.0)
    assert score.summarize(runs, NAIVE, 100).iloc[0]["p95_big"] == 10.0


# ---- error_of and stated_held ----


def test_error_of():
    assert score.error_of(1, 210, Truth(200, None)) == pytest.approx(0.05)
    assert score.error_of(1, 0, Truth(0, None)) == 0.0
    truth = Truth(list("abcdefghij"), {k: 10 - i for i, k in enumerate("abcdefghijk")})
    assert score.error_of(3, list("abcdefghik"), truth) == 1  # 9 overlap
    assert score.error_of(3, list("abcdefghij"), truth) == 0
    assert score.error_of(10, None, Truth(None, None)) == 0.0
    assert score.error_of(10, 3.0, Truth(None, None)) == math.inf
    assert score.error_of(10, 3.5, Truth(3.4, None)) == pytest.approx(0.1)
    groups = Truth({"x": 100, "y": 10}, {"x": 100, "y": 10})
    assert score.error_of(2, {"x": 110, "y": 12}, groups) == pytest.approx(0.2)
    rows = Truth([1, 2], None, frozenset({1, 2, 3}))
    assert score.error_of(4, [1, 9, 3], rows) == 1


def estimate_run(**stated) -> QueryRun:
    return QueryRun(1, 200, Usage(), "estimate", {"method": "estimate", "confidence": 0.99, **stated}, "estimate")


def test_stated_held():
    run = estimate_run(est=200.0, lo=190.0, hi=210.0)
    assert score.stated_held(run, Truth(205, None))
    assert score.stated_held(run, Truth(210, None))
    assert not score.stated_held(run, Truth(211, None))  # one row off
    groups = QueryRun(2, {}, Usage(), "estimate", {"method": "estimate", "est": {"a": 5.0}, "lo": {"a": 4.0},
                                                   "hi": {"a": 6.0}}, "estimate")
    assert score.stated_held(groups, Truth({"a": 5}, {"a": 5}))
    assert not score.stated_held(groups, Truth({"a": 7}, {"a": 7}))
    full = QueryRun(1, 5, Usage(), "full", {"method": "full", "confidence": 1.0}, "big")
    assert score.stated_held(full, Truth(5, None))


def test_stated_held_cascade():
    class Big:
        answers = {"F1": {"a": "yes", "b": "yes", "c": "no", "d": "no"}}
    stated = {"method": "cascade", "targets": {"F1": {"precision": 0.9, "recall": 0.9}}}
    good = QueryRun(1, 2, Usage(), "exact", stated, "x", {"F1": {"a": "yes", "b": "yes", "c": "no", "d": "no"}})
    bad = QueryRun(1, 2, Usage(), "exact", stated, "x", {"F1": {"a": "yes", "b": "no", "c": "no", "d": "no"}})
    assert score.stated_held(good, Truth(2, None), Big())
    assert not score.stated_held(bad, Truth(2, None), Big())  # recall 0.5
    q4 = QueryRun(4, [1, 2], Usage(), "exact", {"method": "cascade", "targets": {}}, "big")
    assert score.stated_held(q4, Truth([1, 2], None, frozenset({1, 2, 3})))
    q4_bad = QueryRun(4, [1, 9], Usage(), "exact", {"method": "cascade", "targets": {}}, "big")
    assert not score.stated_held(q4_bad, Truth([1, 2], None, frozenset({1, 2, 3})))


# ---- run_all ----


@pytest.fixture(scope="module")
def world():
    return make_world(n=2500, seed=0)


def env_of(world) -> Env:
    return Env(world.reviews, world.products, world.emb, {k: i for i, k in enumerate(world.emb_keys)},
               world.question_emb, W, {})


def test_run_all_is_reproducible(world, tmp_path):
    outs = []
    for name in ("a", "b"):
        out = tmp_path / name
        bars = run_check.run_all(env_of(world), world.big, world.small, world.reviews, world.products, 3, out)
        outs.append(out)
        assert bars["outcome"] in ("go", "go pending a paid run", "no-go")
    for f in ("runs.csv", "results.csv", "ablation.csv", "bars.json", "report.md"):
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes(), f
    report = (outs[0] / "report.md").read_text().splitlines()
    assert report[0] in ("# Outcome: go", "# Outcome: go pending a paid run", "# Outcome: no-go")
    runs = pd.read_csv(outs[0] / "runs.csv", keep_default_na=False)
    assert list(runs.columns) == ["approach", "switches", "seed", "query", "mode", "variant", "big", "small",
                                  "proxy", "embed_rows", "cost", "error", "met", "stated_held", "answer_json",
                                  "stated_json"]
    assert len(runs) == 7 * 3 * 10
    key = runs.sort_values(["approach", "switches", "seed", "query"], kind="stable")
    assert key.index.is_monotonic_increasing
    results = pd.read_csv(outs[0] / "results.csv")
    assert set(results["approach"]) == {"naive", "incumbent_snowflake", "incumbent_bigquery", "shilmandb"}
    ablation = pd.read_csv(outs[0] / "ablation.csv")
    assert set(ablation["approach"]) == {"incumbent_bigquery", "shilmandb+T1", "shilmandb+T2", "shilmandb+T3",
                                         "shilmandb+T4", "shilmandb"}
    text = "\n".join(report)
    for heading in ("Fixed method choices", "T5", "Thin queries", "Ablation"):
        assert heading in text


# ---- the command line, end to end on a tiny results folder ----


def write_results_folder(root, world) -> None:
    data, emb = root / "data", root / "embeddings"
    for d in (data, emb, root / "reference"):
        d.mkdir(parents=True)
    world.reviews.to_csv(data / "reviews.csv", index=False, lineterminator="\n")
    world.products.to_csv(data / "products.csv", index=False, lineterminator="\n")
    (data / "design.json").write_text(json.dumps({"floor": 50, "cap": 300, "rows": len(world.reviews)}))
    for tier, labels, ms in (("big", world.big, 100.0), ("small", world.small, 10.0)):
        folder = root / "labels" / tier
        folder.mkdir(parents=True)
        for qid, q in QUESTIONS.items():
            rows = []
            for key, answer in labels.answers[qid].items():
                ps = list(labels.probs[qid].get(key, []))
                rows.append([key, answer, *ps, 200, 3, ms])
            pd.DataFrame(rows, columns=label_columns(q)).to_csv(folder / f"{qid}.csv", index=False,
                                                                lineterminator="\n")
        (folder / "run_info.json").write_text(json.dumps({"model": f"{tier}-model", "template_sha1": "abc",
                                                          "truncated_share": 0.01, "settings": {"greedy": True},
                                                          "date": "2026-11-05"}))
    np.save(emb / "test.npy", world.emb)
    pd.DataFrame({"row": range(len(world.emb_keys)), "text_key": world.emb_keys}).to_csv(
        emb / "test_keys.csv", index=False, lineterminator="\n")
    qids = list(world.question_emb)
    np.save(emb / "questions.npy", np.stack([world.question_emb[q] for q in qids]))
    pd.DataFrame({"row": range(len(qids)), "qid": qids}).to_csv(emb / "questions.csv", index=False,
                                                                 lineterminator="\n")
    pd.DataFrame([{"pass": "test", "rows": len(world.emb_keys), "seconds": 30.0, "rows_per_sec": 80.0}]).to_csv(
        emb / "timing.csv", index=False, lineterminator="\n")
    pd.DataFrame([
        {"tier": "big", "stand_in": "big-api", "usd_per_m_in": 3.0, "usd_per_m_out": 15.0, "source_url": "u",
         "date": "2026-11-05"},
        {"tier": "small", "stand_in": "small-api", "usd_per_m_in": 0.25, "usd_per_m_out": 1.25, "source_url": "u",
         "date": "2026-11-05"},
        {"tier": "embed", "stand_in": "embed-api", "usd_per_m_in": 0.02, "usd_per_m_out": 0.0, "source_url": "u",
         "date": "2026-11-05"}]).to_csv(root / "prices.csv", index=False, lineterminator="\n")
    pd.DataFrame({"qid": list(QUESTIONS), "agreement": 0.95, "rows": 100}).to_csv(
        root / "reference" / "agreement.csv", index=False, lineterminator="\n")


def test_cli_runs_on_a_tiny_results_folder(tmp_path):
    root = tmp_path / "ai"
    write_results_folder(root, make_world(n=1800, seed=2))
    assert run_check.main(["--seeds", "2", "--root", str(root)]) == 0
    runs_dir = root / "runs"
    for f in ("runs.csv", "results.csv", "ablation.csv", "bars.json", "report.md", "proxy_timing.json"):
        assert (runs_dir / f).exists(), f
    bars = json.loads((runs_dir / "bars.json").read_text())
    assert bars["bar3"] is True  # agreement was read from reference/agreement.csv
    report = (runs_dir / "report.md").read_text()
    assert "big-api" in report and "2026-11-05" in report
    timing = (runs_dir / "proxy_timing.json").read_bytes()
    first = (runs_dir / "runs.csv").read_bytes()
    assert run_check.main(["--seeds", "2", "--root", str(root)]) == 0
    assert (runs_dir / "proxy_timing.json").read_bytes() == timing  # measured once, then reused
    assert (runs_dir / "runs.csv").read_bytes() == first


def test_table_labels_are_distinct():
    from bench.ai_queries.approaches import ALL_ON, Switches
    labels = {run_check.table_label(a, sw) for a, sw in (*run_check.MAIN, *run_check.ABLATION)}
    assert len(labels) == len(run_check.MAIN) + len(run_check.ABLATION)
    assert run_check.table_label("shilmandb", Switches()) == "shilmandb+none" != run_check.table_label("shilmandb", ALL_ON)


def test_cli_refuses_missing_labels_or_embeddings(tmp_path, capsys):
    from bench.ai_queries import queries
    world = make_world(n=1800, seed=2)
    root = tmp_path / "ai"
    write_results_folder(root, world)
    f3 = root / "labels" / "small" / "F3.csv"
    df = pd.read_csv(f3, dtype=str, keep_default_na=False)
    scope3 = set(queries.question_scopes(world.reviews, world.products)["F3"])
    drop = [k for k in df["text_key"] if k in scope3][:7]
    df[~df["text_key"].isin(drop)].to_csv(f3, index=False, lineterminator="\n")
    (root / "labels" / "big" / "C2.csv").unlink()
    keys = pd.read_csv(root / "embeddings" / "test_keys.csv", dtype=str, keep_default_na=False)
    np.save(root / "embeddings" / "test.npy", world.emb[:-3])
    keys.iloc[:-3].to_csv(root / "embeddings" / "test_keys.csv", index=False, lineterminator="\n")

    assert run_check.main(["--seeds", "1", "--root", str(root)]) == 1
    err = capsys.readouterr().err
    n2 = len(queries.question_scopes(world.reviews, world.products)["C2"])
    assert f"small F3: 7 of {len(scope3)} scope keys have no label" in err
    assert f"big C2: {n2} of {n2} scope keys have no label" in err
    assert "embeddings: 3 of" in err
    assert not (root / "runs" / "runs.csv").exists()


def test_small_tier_needs_no_extract_labels():
    from bench.ai_queries import queries
    world = make_world(n=1800, seed=2)
    reviews, products = world.reviews, world.products
    no_e7 = type(world.small)("small", {q: a for q, a in world.small.answers.items() if q != "E7"}, world.small.probs)
    assert "E7" in world.big.answers and queries.question_scopes(reviews, products)["E7"]
    assert run_check.coverage_gaps(env_of(world), world.big, no_e7, reviews, products) == []
    no_big_e7 = type(world.big)("big", {q: a for q, a in world.big.answers.items() if q != "E7"}, world.big.probs)
    gaps = run_check.coverage_gaps(env_of(world), no_big_e7, no_e7, reviews, products)
    assert len(gaps) == 1 and gaps[0].startswith("big E7")
