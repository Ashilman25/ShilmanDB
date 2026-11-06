import json
import math

import numpy as np
import pandas as pd
import pytest

from bench.ai_queries import design_check, llm
from bench.ai_queries.common import TOKEN_CAP, text_key
from bench.ai_queries.labels import TierLabels, label_columns, read_label_file
from bench.ai_queries.questions import C2_CLASSES, QUESTIONS


class FakeBackend:
    """Tokens are whitespace words. Probabilities come from a function of the prompt."""

    def __init__(self, probs=None, crash_on_call=None, generated="iPhone 13.\nI love it"):
        self.probs = probs
        self.crash_on_call = crash_on_call
        self.generated = generated
        self.calls = 0
        self.prompts: list[str] = []

    def count_tokens(self, text):
        return len(text.split())

    def truncate(self, text, cap):
        words = text.split(" ")
        return text if len(words) <= cap else " ".join(words[:cap])

    last_option_mass = None

    def _tick(self, prompts):
        self.calls += 1
        self.prompts += prompts
        if self.crash_on_call == self.calls:
            raise RuntimeError("model crashed")

    def option_probs(self, prompts, options):
        self._tick(prompts)
        if self.probs is not None:
            return np.array([self.probs(p, options) for p in prompts])
        return np.array([[0.8] + [0.2 / (len(options) - 1)] * (len(options) - 1) for _ in prompts])

    def generate(self, prompts, max_tokens):
        self._tick(prompts)
        return [self.generated for _ in prompts]


def keys_and_inputs(n=10):
    keys = [f"k{i}" for i in range(n)]
    return keys, {k: f"Title: t\nReview: review {k}" for k in keys}


def test_prompt_contains_wording_and_options():
    f3 = llm.build_prompt(QUESTIONS["F3"], "Title: t\nReview: r")
    assert QUESTIONS["F3"].wording in f3 and "yes" in f3 and "no" in f3
    c2 = llm.build_prompt(QUESTIONS["C2"], "Title: t\nReview: r")
    for letter, name in zip("ABCDEF", C2_CLASSES):
        assert f"{letter}) {name}" in c2
    assert "A) shipping or delivery" in c2 and "F) other" in c2
    assert "none" in llm.build_prompt(QUESTIONS["E7"], "Title: t\nReview: r")


def test_truncates_review_to_cap(tmp_path):
    backend = FakeBackend()
    long = " ".join(f"w{i}" for i in range(600))
    short = " ".join(f"s{i}" for i in range(20))
    llm.label(QUESTIONS["F3"], ["long", "short"], {"long": long, "short": short}, backend, tmp_path / "F3.csv")
    assert f"w{TOKEN_CAP - 1}" in backend.prompts[0] and f"w{TOKEN_CAP}" not in backend.prompts[0]
    assert short in backend.prompts[1]


def test_option_token_ids_rejects_shared_first_tokens():
    table = {"a": [5, 9], "b": [5, 2]}
    with pytest.raises(ValueError):
        llm.option_token_ids(lambda s: table[s], ["a", "b"])
    with pytest.raises(ValueError):
        llm.option_token_ids(lambda s: [] if s == "z" else [1], ["a", "z"])
    assert llm.option_token_ids(lambda s: {"a": [5, 9], "b": [6]}[s], ["a", "b"]) == [5, 6]


def test_label_writes_answers_and_probabilities(tmp_path):
    keys, inputs = keys_and_inputs(3)
    path = tmp_path / "F3.csv"
    assert llm.label(QUESTIONS["F3"], keys, inputs, FakeBackend(), path, batch=2) == 3
    df = read_label_file(path)
    assert df["answer"].tolist() == ["yes"] * 3
    assert df["p0"].round(3).tolist() == [0.8] * 3 and df["p1"].round(3).tolist() == [0.2] * 3
    assert (df["out_tokens"] == 1).all() and (df["in_tokens"] > 5).all()
    c2 = tmp_path / "C2.csv"
    llm.label(QUESTIONS["C2"], keys, inputs, FakeBackend(), c2)
    assert read_label_file(c2)["answer"].tolist() == [C2_CLASSES[0]] * 3


def test_label_resumes_after_crash(tmp_path):
    keys, inputs = keys_and_inputs(10)
    path = tmp_path / "F3.csv"
    with pytest.raises(RuntimeError):
        llm.label(QUESTIONS["F3"], keys, inputs, FakeBackend(crash_on_call=2), path, batch=4)
    assert read_label_file(path)["text_key"].tolist() == keys[:4]
    assert llm.label(QUESTIONS["F3"], keys, inputs, FakeBackend(), path, batch=4) == 6
    assert sorted(read_label_file(path)["text_key"]) == sorted(keys)
    assert llm.label(QUESTIONS["F3"], keys, inputs, FakeBackend(), path, batch=4) == 0


def test_partial_last_line_is_redone(tmp_path):
    keys, inputs = keys_and_inputs(6)
    path = tmp_path / "F3.csv"
    llm.label(QUESTIONS["F3"], keys[:3], inputs, FakeBackend(), path, batch=4)
    with open(path, "a", encoding="utf-8") as f:
        f.write("k3,yes,0.8")  # a write that was cut off
    assert llm.label(QUESTIONS["F3"], keys, inputs, FakeBackend(), path, batch=4) == 3
    assert read_label_file(path)["text_key"].tolist() == keys


def test_nan_probs_raise(tmp_path):
    keys, inputs = keys_and_inputs(2)
    backend = FakeBackend(probs=lambda p, options: [math.nan, math.nan])
    with pytest.raises(ValueError, match="F3"):
        llm.label(QUESTIONS["F3"], keys, inputs, backend, tmp_path / "F3.csv")
    backend = FakeBackend(probs=lambda p, options: [0.9, 0.3])
    with pytest.raises(ValueError, match="F3"):
        llm.label(QUESTIONS["F3"], keys, inputs, backend, tmp_path / "F3b.csv")


def test_extract_cleanup(tmp_path):
    assert llm.clean_extract(" iPhone 13.\nI love it") == "iPhone 13"
    assert llm.clean_extract("") == ""
    keys, inputs = keys_and_inputs(2)
    path = tmp_path / "E7.csv"
    llm.label(QUESTIONS["E7"], keys, inputs, FakeBackend(), path)
    df = read_label_file(path)
    assert df["answer"].tolist() == ["iPhone 13"] * 2
    assert list(df.columns) == label_columns(QUESTIONS["E7"])


def test_refuses_other_model(tmp_path):
    keys, inputs = keys_and_inputs(4)
    tier = tmp_path / "big"
    info = llm.label_tier(tier, "model-a", FakeBackend(), ["F3"], {"F3": keys[:2]}, inputs)
    assert info["questions"] == ["F3"] and info["template_sha1"] == llm.template_sha1()
    with pytest.raises(SystemExit):
        llm.label_tier(tier, "model-b", FakeBackend(), ["F3"], {"F3": keys}, inputs)
    stale = json.loads((tier / "run_info.json").read_text())
    stale["template_sha1"] = "0" * 40
    (tier / "run_info.json").write_text(json.dumps(stale))
    with pytest.raises(SystemExit):
        llm.label_tier(tier, "model-a", FakeBackend(), ["F3"], {"F3": keys}, inputs)
    assert read_label_file(tier / "F3.csv")["text_key"].tolist() == keys[:2]


def test_run_info_records_truncation_and_resumes(tmp_path):
    keys, inputs = keys_and_inputs(4)
    inputs["k0"] = " ".join(["w"] * 600)
    tier = tmp_path / "scratch"
    llm.label_tier(tier, "model-a", FakeBackend(), ["F3"], {"F3": keys[:2]}, inputs)
    info = json.loads((tier / "run_info.json").read_text())
    assert info["truncated_share"] == 0.5 and info["path"] is None and info["quantization"] is None
    llm.label_tier(tier, "model-a", FakeBackend(), ["F3", "F8"], {"F3": keys, "F8": keys}, inputs)
    assert read_label_file(tier / "F3.csv")["text_key"].tolist() == keys
    assert json.loads((tier / "run_info.json").read_text())["questions"] == ["F3", "F8"]


def test_slug():
    assert llm.slug("mlx-community/Qwen3-32B_4bit/") == "qwen3-32b-4bit"


def test_latency_summary(tmp_path):
    (tmp_path / "F3.csv").write_text(
        "text_key,answer,p0,p1,in_tokens,out_tokens,latency_ms\n"
        "a,yes,0.9,0.1,100,1,10.0\nb,no,0.2,0.8,200,1,30.0\nc,no,0.2,0.8,300,1,20.0\n")
    (tmp_path / "E7.csv").write_text(
        "text_key,answer,in_tokens,out_tokens,latency_ms\na,x,50,4,40.0\nb,y,70,6,60.0\n")
    out = llm.latency_summary(tmp_path)
    assert out["qid"].tolist() == ["F3", "E7"]
    assert out.loc[0, "rows"] == 3 and out.loc[0, "median_ms"] == 20.0
    assert out.loc[0, "mean_in_tokens"] == 200.0 and out.loc[1, "mean_out_tokens"] == 5.0


# ---- design check ----

CASE_CATS = json.dumps(["Cell Phones", "Cases, Holsters & Sleeves"])


def toy_tables(per_product=6, n_products=12):
    rows = [(p, j) for p in range(n_products) for j in range(per_product)]
    reviews = pd.DataFrame({
        "review_id": range(len(rows)), "rating": 1,
        "title": [f"t{p}-{j}" for p, j in rows], "text": [f"x{p}-{j}" for p, j in rows],
        "parent_asin": [f"P{p}" for p, _ in rows], "timestamp": 1000, "helpful_vote": 0,
        "verified_purchase": True, "user_id": "u"})
    reviews["text_key"] = [text_key(t, x) for t, x in zip(reviews.title, reviews.text)]
    products = pd.DataFrame({
        "parent_asin": [f"P{p}" for p in range(n_products)], "title": [f"Case {p}" for p in range(n_products)],
        "store": [f"Store {p}" for p in range(n_products)], "price": 10.0, "main_category": "",
        "categories": CASE_CATS, "average_rating": 4.0, "rating_number": 10})
    return reviews, products, rows


def toy_small(reviews, rows, f6_yes):
    def answers(fn):
        return {k: fn(i, p, j) for i, (k, (p, j)) in enumerate(zip(reviews["text_key"], rows))}
    return TierLabels("small", {
        "F3": answers(lambda i, p, j: "yes"),
        "F4": answers(lambda i, p, j: "yes" if i < 50 else "no"),
        "F6": answers(lambda i, p, j: "yes" if j < f6_yes(p) else "no"),
        "E7": answers(lambda i, p, j: f"model {p}"),
        "F8": answers(lambda i, p, j: "yes" if i < 31 else "no"),
    }, {})


def test_design_check_flags_thin():
    reviews, products, rows = toy_tables()
    thin = toy_small(reviews, rows, lambda p: 6 if p < 9 else (3 if p == 9 else 0))
    table = design_check.check(reviews, products, thin, {}).set_index("query")
    assert table.loc["Q6", "value"] == 3 and not table.loc["Q6", "ok"]
    assert table.loc["Q3", "ok"] and table.loc["Q7", "ok"] and table.loc["Q9", "ok"]
    assert table.loc["Q4", "value"] == 50 and table.loc["Q4", "ok"]
    assert table.loc["Q8", "value"] == 31 and table.loc["Q8", "ok"]
    assert design_check.exit_code(table.reset_index()) == 1
    full = toy_small(reviews, rows, lambda p: 6)
    ok = design_check.check(reviews, products, full, {})
    assert design_check.exit_code(ok) == 0


def test_design_check_short_list_is_zero_and_q8_is_report_only():
    reviews, products, rows = toy_tables(n_products=8)
    small = toy_small(reviews, rows, lambda p: 6)
    table = design_check.check(reviews, products, small, {}).set_index("query")
    assert table.loc["Q3", "value"] == 0 and not table.loc["Q3", "ok"]
    only_q8 = pd.DataFrame([{"query": "Q8", "measure": "match count", "value": 1, "ok": False}])
    assert design_check.exit_code(only_q8) == 0


def test_repeated_key_is_written_once(tmp_path):
    keys, inputs = keys_and_inputs(3)
    path = tmp_path / "F3.csv"
    assert llm.label(QUESTIONS["F3"], [keys[0], keys[1], keys[0], keys[2], keys[1]], inputs,
                     FakeBackend(), path, batch=2) == 3
    assert read_label_file(path)["text_key"].tolist() == keys


def test_low_option_mass_stops_at_first_batch(tmp_path):
    keys, inputs = keys_and_inputs(8)
    backend = FakeBackend()
    backend.last_option_mass = 0.2
    with pytest.raises(ValueError, match=r"F3.*0\.200"):
        llm.label(QUESTIONS["F3"], keys, inputs, backend, tmp_path / "F3.csv", batch=4)
    assert backend.calls == 1 and not (tmp_path / "F3.csv").exists()
    backend.last_option_mass = 0.9
    assert llm.label(QUESTIONS["F3"], keys, inputs, backend, tmp_path / "F3.csv", batch=4) == 8


def test_open_thinking_block_is_rejected():
    with pytest.raises(ValueError, match="think"):
        llm.check_no_open_thinking("<|im_start|>assistant\n<think>\n")
    llm.check_no_open_thinking("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    llm.check_no_open_thinking("<start_of_turn>model\n")


def test_mlx_chat_checks_thinking_with_stub_tokenizer():
    backend = llm.MlxBackend.__new__(llm.MlxBackend)

    class Stub:
        def __init__(self, tail):
            self.tail = tail

        def apply_chat_template(self, messages, **kw):
            return "assistant\n" + self.tail

    backend.tokenizer = Stub("<think>\n")
    with pytest.raises(ValueError):
        backend._chat("hi")
    backend.tokenizer = Stub("")
    assert backend._chat("hi") == "assistant\n"


class MassBackend(FakeBackend):
    """Reports a raw option mass per row, like the real backend."""

    def __init__(self, masses, **kw):
        super().__init__(**kw)
        self.masses = list(masses)
        self.last_option_masses = None

    def option_probs(self, prompts, options):
        out = super().option_probs(prompts, options)
        self.last_option_masses = np.array(self.masses[:len(prompts)])
        self.masses = self.masses[len(prompts):]
        self.last_option_mass = float(self.last_option_masses.mean())
        return out


def test_option_mass_summary_merges_across_runs(tmp_path):
    keys, inputs = keys_and_inputs(10)
    tier = tmp_path / "big"
    first = [0.95, 0.9, 0.7, 0.99]
    with pytest.raises(RuntimeError):  # killed during the second batch
        llm.label_tier(tier, "m", MassBackend(first + [0.6] * 4, crash_on_call=2), ["F3"], {"F3": keys}, inputs, batch=4)
    info = json.loads((tier / "run_info.json").read_text())
    assert info["option_mass"]["F3"]["rows"] == 4  # the finished batch was saved
    rest = [0.45, 0.85, 0.98, 0.97, 0.96, 0.55]
    llm.label_tier(tier, "m", MassBackend(rest), ["F3"], {"F3": keys}, inputs, batch=4)
    info = json.loads((tier / "run_info.json").read_text())
    mass = info["option_mass"]["F3"]
    every = np.array(first + rest)
    assert mass["rows"] == 10 == len(read_label_file(tier / "F3.csv"))
    assert mass["mean"] == pytest.approx(every.mean())
    assert info["option_mass_mean"]["F3"] == pytest.approx(every.mean())  # not the second run's own mean
    assert mass["min"] == 0.45 and mass["under_0_5"] == 1 and mass["under_0_8"] == 3
    assert len(mass["hist"]) == llm.MASS_BINS and sum(mass["hist"]) == 10
    assert mass["hist"][9] == 1 and mass["hist"][19] == 4  # 0.45 in [0.45, 0.5), four rows at 0.95 or more
    assert mass["p05"] == 0.5  # the lowest 5% of rows sit at or below 0.5
    # an extract question has no option mass
    llm.label_tier(tier, "m", MassBackend([]), ["E7"], {"E7": keys[:2]}, inputs)
    assert "E7" not in json.loads((tier / "run_info.json").read_text())["option_mass"]


def test_hist_quantile():
    assert llm.hist_quantile([0] * 20, 0.05) is None
    assert llm.hist_quantile([1] + [0] * 18 + [19], 0.05) == 0.05
    assert llm.hist_quantile([0] * 19 + [5], 0.05) == 1.0


def test_run_info_is_written_atomically(tmp_path, monkeypatch):
    keys, inputs = keys_and_inputs(4)
    tier = tmp_path / "big"
    llm.label_tier(tier, "m", FakeBackend(), ["F3"], {"F3": keys}, inputs)
    before = (tier / "run_info.json").read_text()
    real = type(tier).write_text

    def cut_off(self, data, *a, **kw):  # the machine dies halfway through a write
        real(self, data[:len(data) // 2], *a, **kw)
        raise OSError("power lost")

    monkeypatch.setattr(type(tier), "write_text", cut_off)
    with pytest.raises(OSError):
        llm.label_tier(tier, "m", FakeBackend(), ["F8"], {"F8": keys}, inputs)
    monkeypatch.undo()
    assert (tier / "run_info.json").read_text() == before  # the old file is whole


def test_free_model_drops_the_weights():
    backend = FakeBackend()
    backend.model, backend.tokenizer = object(), object()
    llm.free_model(backend)
    assert not hasattr(backend, "model") and not hasattr(backend, "tokenizer")


def _tiny_table(n=40):
    rng = np.random.default_rng(0)
    reviews = pd.DataFrame({
        "review_id": range(n), "rating": [1, 2, 3, 4, 5] * (n // 5), "title": "t",
        "text": [f"text {i}" for i in range(n)], "parent_asin": ["P0", "P1"] * (n // 2),
        "timestamp": rng.permutation(n) + 1_600_000_000_000, "helpful_vote": 0,
        "verified_purchase": [i % 4 == 0 for i in range(n)], "user_id": "u"})
    reviews["text_key"] = [text_key(t, x) for t, x in zip(reviews["title"], reviews["text"])]
    products = pd.DataFrame({
        "parent_asin": ["P0", "P1"], "title": ["Case 0", "Case 1"], "store": ["A", "B"], "price": 10.0,
        "main_category": "", "categories": CASE_CATS, "average_rating": 4.0, "rating_number": 10})
    return reviews, products


def test_naive_bounds_from_the_tables_alone():
    reviews, products = _tiny_table()
    table = design_check.naive_bounds(reviews, products).set_index("query")
    exact = table["naive_low"] == table["naive_high"]
    assert exact.drop(["Q4", "Q5"]).all() and not exact["Q5"]
    assert table.loc["Q1", "naive_low"] == 16  # ratings 1 and 2
    assert table.loc["Q2", "naive_low"] == 24 and table.loc["Q7", "naive_low"] == 8
    assert tuple(table.loc["Q5", ["naive_low", "naive_high"]]) == (40, 80)
    assert tuple(table.loc["Q4", ["naive_low", "naive_high"]]) == (10, 10)  # fewer than 20 verified rows
    assert table.loc["Q1", "floor"] == 6000 and table.loc["Q5", "floor"] == 12000 and table.loc["Q4", "floor"] == 200
    assert set(table["reachable"]) == {"no"}  # far too small
    # with a tiny calibration budget most queries clear the floor
    cheap = design_check.naive_bounds(reviews, products, calibration=1).set_index("query")
    assert cheap.loc["Q1", "reachable"] == "yes" and cheap.loc["Q5", "reachable"] == "yes"
    assert cheap.loc["Q4", "reachable"] == "no"  # at most 10 calls, the floor is 200
    assert cheap.loc["Q7", "reachable"] == "no"  # 8 rows, floor 10
    cheap = design_check.naive_bounds(reviews, products, calibration=3).set_index("query")
    assert cheap.loc["Q5", "reachable"] == "unknown"  # 40 to 80 calls against a floor of 60


def test_naive_cli(tmp_path, capsys):
    reviews, products = _tiny_table()
    data = tmp_path / "ai" / "data"
    data.mkdir(parents=True)
    reviews.to_csv(data / "reviews.csv", index=False, lineterminator="\n")
    products.to_csv(data / "products.csv", index=False, lineterminator="\n")
    assert design_check.main(["naive", "--root", str(tmp_path / "ai")]) == 1  # bar 1 is out of reach
    out = capsys.readouterr().out
    assert "Q5" in out and "naive_high" in out
    assert "0 of 10 queries can reach bar 1, 0 unknown without labels (bar 1 needs 7)" in out
    assert design_check.main(["--root", str(tmp_path / "ai"), "naive"]) == 1  # --root on either side


def test_design_check_tolerates_a_partly_labeled_tier(capsys):
    reviews, products, rows = toy_tables()
    full = toy_small(reviews, rows, lambda p: 6)
    cut = {qid: dict(list(a.items())[::2]) for qid, a in full.answers.items()}  # every other key unlabeled
    part = TierLabels("small", cut, {})
    table = design_check.check(reviews, products, part, {})
    assert list(table.columns) == ["query", "measure", "value", "ok"] and len(table) == 6
    assert design_check.missing_labels(reviews, products, part) > 0
    assert design_check.missing_labels(reviews, products, full) == 0


def test_missing_labels_counts_distinct_keys_left_out_of_the_top_counts():
    reviews, products, rows = toy_tables()
    full = toy_small(reviews, rows, lambda p: 6)
    drop = set(list(reviews["text_key"])[:5])
    cut = {qid: {k: v for k, v in a.items() if k not in drop} for qid, a in full.answers.items()}
    assert design_check.missing_labels(reviews, products, TierLabels("small", cut, {})) == 5
    # a key missing from one question is counted once, even though several queries scope it
    only_f3 = {qid: (a if qid != "F3" else {k: v for k, v in a.items() if k not in drop}) for qid, a in full.answers.items()}
    assert design_check.missing_labels(reviews, products, TierLabels("small", only_f3, {})) <= 5
