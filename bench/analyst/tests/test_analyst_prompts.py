"""Tests for bench/analyst/prompts.py, prompt_examples.json and model_mlx.py."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sqlite3
import sys
import types
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ANALYST = HERE.parent


# Load a module by path under an an_ name, clear of other bench tests.
def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ANALYST / file_name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = _load("an_prompts", "prompts.py")
an = mod.an

SECTIONS = ["## Dataset", "## SQL you can use", "## Examples", "## Reply format", "## Question"]
DESCRIPTION = "# tpch\n\n## About\n\nSales data.\n\n## Tables\n\n### region (5 rows)\n\n- `r_name` VARCHAR, e.g. 'Brand#MN'\n"
RULES = "# SQL you can use\n\nUse only these.\n\n## Works\n\n- count_star\n\n## Doesn't work\n\n- having\n"
EXAMPLES = [
    {"question": "How many regions are there?",
     "reply": {"sql": "SELECT COUNT(*) AS region_count FROM region", "grain": "one row",
               "columns": [{"name": "region_count", "meaning": "number of regions", "entity": "regions"}]}},
    {"question": "What did shipping cost?", "reply": {"cannot_answer": "no shipping costs"}},
]
# Words for SQL the engine can't run, from the rules' "Doesn't work" list.
UNSUPPORTED = re.compile(r"\b(HAVING|OFFSET|LEFT|DISTINCT|WITH|UNION|NULL|COALESCE|EXTRACT|SUBSTRING|UPPER|CAST)\b", re.I)


# Every frozen question, read through code only; skips when the file isn't there.
def questions_if_present() -> list[dict]:
    path = an.paths().questions
    if not path.exists():
        pytest.skip("questions.jsonl hasn't been written")
    return an.read_jsonl(path)


def _examples() -> dict:
    return json.loads((ANALYST / "prompt_examples.json").read_text(encoding="utf-8"))


def _prompt() -> str:
    return mod.build_sql_prompt(DESCRIPTION, RULES, EXAMPLES, "Which region has the most nations?")


def test_parse_reply_tolerates_wrapping():
    raw = 'Sure!\n<think>hmm {"sql": "x"}</think>\n```json\n{"sql": "SELECT 1", "grain": "one row", "columns": []}\n```\nDone.'
    r = mod.parse_reply(raw)
    assert r.kind == "sql" and r.sql == "SELECT 1" and r.grain == "one row"
    assert r.raw == raw
    assert mod.parse_reply('{"cannot_answer": "no cost data"}').kind == "cannot"
    assert mod.parse_reply('{"sql": "SELECT 1"').kind == "malformed"          # never closed
    assert mod.parse_reply('<think>{"sql": "SELECT 1"}').kind == "malformed"  # thinking never closed
    assert mod.parse_reply('{"note": "x"} {"sql": "SELECT 2", "grain": "", "columns": []}').sql == "SELECT 2"


def test_braces_inside_strings():
    text = 'Here: {"sql": "SELECT \'}\' AS a, \'{\' AS b, \'say \\"hi\\"\' AS c FROM t", "grain": "one row per row of t", "columns": []}'
    r = mod.parse_reply(text)
    assert r.kind == "sql"
    assert r.sql == "SELECT '}' AS a, '{' AS b, 'say \"hi\"' AS c FROM t"
    assert r.grain == "one row per row of t"


def test_parse_reply_fields():
    cols = [{"name": "n", "meaning": "number of orders", "entity": "orders"}]
    r = mod.parse_reply(json.dumps({"sql": "SELECT COUNT(*) AS n FROM orders", "grain": "one row", "columns": cols}))
    assert (r.kind, r.sql, r.grain, r.columns, r.reason) == ("sql", "SELECT COUNT(*) AS n FROM orders", "one row", cols, None)

    no_extras = mod.parse_reply('{"sql": "SELECT 1"}')
    assert (no_extras.kind, no_extras.grain, no_extras.columns) == ("sql", None, [])

    cannot = mod.parse_reply('Answer: {"cannot_answer": "the data has no costs"}')
    assert (cannot.kind, cannot.sql, cannot.reason, cannot.columns) == ("cannot", None, "the data has no costs", [])

    for text in ['{"sql": ""}', '{"sql": "   "}', '{"sql": 5}', '["SELECT 1"]', "SELECT 1", "", '{"grain": "one row"}']:
        bad = mod.parse_reply(text)
        assert (bad.kind, bad.sql, bad.reason, bad.raw) == ("malformed", None, None, text), text


def test_parse_reply_reads_a_nested_reply():
    assert mod.parse_reply('{"answer": {"sql": "SELECT 3", "grain": "one row", "columns": []}}').sql == "SELECT 3"


def test_strip_thinking():
    assert mod.strip_thinking("<think>a</think>x<think>\nb {c}\n</think>y") == "xy"
    assert mod.strip_thinking("x <think>never closed") == ""
    assert mod.strip_thinking("no thinking here") == "no thinking here"


def test_prompt_sections_in_order():
    prompt = _prompt()
    lines = prompt.splitlines()
    assert lines[0].strip() and not lines[0].startswith("#")   # the one-line role
    assert [line for line in lines if line.startswith("## ")] == SECTIONS
    assert not [line for line in lines if line.startswith("# ")]

    at = {name: prompt.index(name + "\n") for name in SECTIONS}
    assert list(at.values()) == sorted(at.values())
    assert at["## Dataset"] < prompt.index("Sales data.") < at["## SQL you can use"]
    assert at["## SQL you can use"] < prompt.index("Use only these.") < at["## Examples"]
    assert at["## Examples"] < prompt.index("How many regions are there?") < prompt.index('{"cannot_answer": "no shipping costs"}') < at["## Reply format"]
    assert json.dumps(EXAMPLES[0]["reply"]) in prompt
    assert prompt.rstrip().endswith("## Question\n\nWhich region has the most nations?")

    reply_format = prompt[at["## Reply format"]:at["## Question"]]
    for word in ('"sql"', '"grain"', '"columns"', '"name"', '"meaning"', '"entity"', '"cannot_answer"', "exactly one JSON object"):
        assert word in reply_format


def test_inserted_markdown_sits_one_level_below():
    lines = _prompt().splitlines()
    for heading in ("### About", "### Tables", "#### region (5 rows)", "### Works", "### Doesn't work"):
        assert heading in lines
    assert "# tpch" not in lines and "# SQL you can use" not in lines
    assert "- `r_name` VARCHAR, e.g. 'Brand#MN'" in lines        # a # mid-line is left alone

    # Text without a title keeps its first line and still moves down.
    kept = mod.build_sql_prompt("## About\n\nSales.\n#hashtag", "Plain rules.", [], "Q?").splitlines()
    assert "### About" in kept and "#hashtag" in kept and "Plain rules." in kept


def test_real_description_and_rules_nest_under_their_sections():
    desc, rules = an.paths().dataset_dir("tpch") / "description.md", an.paths().rules
    if not desc.exists() or not rules.exists():
        pytest.skip("the description or rules haven't been written")

    prompt = mod.build_sql_prompt(desc.read_text(encoding="utf-8"), rules.read_text(encoding="utf-8"), _examples()["tpch"], "Q?")
    lines = prompt.splitlines()
    assert [line for line in lines if line.startswith("## ")] == SECTIONS
    assert not [line for line in lines if line.startswith("# ")]
    assert "### Glossary" in lines and "### Works" in lines


def test_retry_prompt_carries_error():
    base = _prompt()
    previous = '{"sql": "SELECT n_name FROM nation HAVING COUNT(*) > 1"}'
    problem = "parse error: HAVING is not supported"
    retry = mod.build_retry_prompt(base, previous, problem)
    assert retry.startswith(base)
    assert base.rstrip() != retry.rstrip()
    assert retry.index(previous) < retry.index(problem) < retry.index("Write a corrected reply in the same format.")
    assert retry.rstrip().endswith("Write a corrected reply in the same format.")


def test_retry_problems_say_what_went_wrong():
    assert mod.NOT_A_REPLY == "the reply wasn't one JSON object in the reply format"
    assert mod.TOO_SLOW == "the query took longer than 30 seconds"
    assert mod.CRASHED == "the engine crashed on that query"


def test_rephrase_prompt():
    prompt = mod.build_rephrase_prompt("How many orders were placed in 1995?")
    assert prompt.startswith("Rewrite this question in different words with exactly the same meaning. "
                             "Reply with the rewritten question only.")
    assert prompt.rstrip().endswith("How many orders were placed in 1995?")


def test_parse_rephrase():
    assert mod.parse_rephrase('"How many orders?"\n') == "How many orders?"
    assert mod.parse_rephrase("") is None
    assert mod.parse_rephrase("<think>hmm\nsome thought</think>\n\n  'What was 1995 revenue?'  \nextra line") == "What was 1995 revenue?"
    assert mod.parse_rephrase("“Which nation is largest?”") == "Which nation is largest?"
    assert mod.parse_rephrase("<think>ran out of tokens") is None
    assert mod.parse_rephrase('  \n""\n') is None
    assert mod.parse_rephrase("It's 'quoted' inside") == "It's 'quoted' inside"


def test_examples_shape_and_no_overlap():
    ex = _examples()
    assert set(ex) == {"tpch", "reviews"}
    assert {len(ex["tpch"]), len(ex["reviews"])} == {3}

    for ds in ("tpch", "reviews"):
        kinds = sorted(mod.parse_reply(json.dumps(e["reply"])).kind for e in ex[ds])
        assert kinds == ["cannot", "sql", "sql"], ds

    for e in ex["tpch"] + ex["reviews"]:
        assert isinstance(e["question"], str) and e["question"].strip()
        r = mod.parse_reply(json.dumps(e["reply"]))
        if r.kind == "cannot":
            assert set(e["reply"]) == {"cannot_answer"} and r.reason.strip()
            continue

        assert set(e["reply"]) == {"sql", "grain", "columns"}
        assert r.grain.strip() and r.columns
        for col in r.columns:
            assert set(col) == {"name", "meaning", "entity"}
            assert all(isinstance(v, str) and v.strip() for v in col.values())
        assert r.sql.upper().count("SELECT") == 1 and not UNSUPPORTED.search(r.sql), r.sql

    qs = questions_if_present()
    norm = lambda s: " ".join(s.casefold().split())
    overlap = {norm(e["question"]) for e in ex["tpch"] + ex["reviews"]} & {norm(q["question"]) for q in qs}
    assert not overlap, f"examples repeat {len(overlap)} question(s)"


def test_example_columns_match_their_sql():
    for ds, examples in _examples().items():
        db = an.paths().dataset_dir(ds) / f"{ds}.sqlite"
        if not db.exists():
            pytest.skip(f"{db.name} hasn't been written")

        with closing(sqlite3.connect(db)) as con:
            for e in examples:
                if "sql" not in e["reply"]:
                    continue
                cur = con.execute(an.to_sqlite(e["reply"]["sql"]))
                rows = cur.fetchall()
                assert rows, e["question"]
                assert [c[0] for c in cur.description] == [c["name"] for c in e["reply"]["columns"]], e["question"]
                if e["reply"]["grain"] == "one row":
                    assert len(rows) == 1, e["question"]


# model_mlx.py


class FakeTokenizer:
    """Renders a chat like Qwen's template does, and records each call's keywords."""

    def __init__(self, knows_thinking: bool = True, open_when: bool | None = True):
        self.knows_thinking = knows_thinking
        self.open_when = open_when  # the thinking value that ends the prompt in an open <think>
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(kwargs)
        if "enable_thinking" in kwargs and not self.knows_thinking:
            raise TypeError("unexpected keyword argument 'enable_thinking'")

        head = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        if self.open_when is None:
            return head
        thinking = kwargs.get("enable_thinking", True)
        return head + ("<think>\n" if thinking == self.open_when else "<think>\n\n</think>\n\n")


def _fake_mlx_lm(monkeypatch, reply: str) -> list:
    sent: list = []
    fake = types.ModuleType("mlx_lm")
    fake.load = lambda path: (f"model at {path}", FakeTokenizer())

    def generate(model, tokenizer, prompt, verbose=False, **kwargs):
        sent.append({"model": model, "prompt": prompt, **kwargs})
        return reply

    fake.generate = generate
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return sent


def test_model_configs():
    m = _load("an_model_mlx", "model_mlx.py")
    assert set(m.MODEL_CONFIGS) == {"gemma4-26b", "qwen36-35b", "qwen36-35b-think", "qwen38-27b", "qwen38-27b-think"}
    assert m.MODEL_CONFIGS["qwen36-35b-think"]["max_tokens"] == 8192 and m.MODEL_CONFIGS["gemma4-26b"]["max_tokens"] == 1024
    assert m.MODEL_CONFIGS["gemma4-26b"]["path"] == "mlx-community/gemma-4-26b-a4b-it-4bit"
    assert m.MODEL_CONFIGS["qwen36-35b"]["path"] == m.MODEL_CONFIGS["qwen36-35b-think"]["path"] == "mlx-community/Qwen3.6-35B-A3B-4bit"
    assert m.MODEL_CONFIGS["qwen38-27b"]["path"] == m.MODEL_CONFIGS["qwen38-27b-think"]["path"] == "mlx-community/Qwen3.8-27B-4bit"
    for cid, config in m.MODEL_CONFIGS.items():
        assert set(config) == {"path", "thinking", "max_tokens"}
        assert config["thinking"] is cid.endswith("-think")
        assert config["max_tokens"] == (8192 if config["thinking"] else 1024)


def test_model_configs_is_a_plain_literal():
    m = _load("an_model_mlx", "model_mlx.py")
    tree = ast.parse((ANALYST / "model_mlx.py").read_text(encoding="utf-8"))
    values = [node.value for node in tree.body
              if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == ["MODEL_CONFIGS"]]
    assert len(values) == 1
    assert ast.literal_eval(values[0]) == m.MODEL_CONFIGS


def test_render_chat_passes_thinking_and_falls_back():
    m = _load("an_model_mlx", "model_mlx.py")
    tok = FakeTokenizer()
    assert m.render_chat(tok, "Q?", thinking=False).endswith("</think>\n\n")
    assert m.render_chat(tok, "Q?", thinking=True).endswith("<think>\n")
    assert [c["enable_thinking"] for c in tok.calls] == [False, True]
    assert all(c["add_generation_prompt"] is True and c["tokenize"] is False for c in tok.calls)

    old = FakeTokenizer(knows_thinking=False, open_when=None)
    assert m.render_chat(old, "Q?", thinking=False).endswith("<|im_start|>assistant\n")
    assert "enable_thinking" not in old.calls[-1]


def test_reply_prefix():
    m = _load("an_model_mlx", "model_mlx.py")
    assert m.reply_prefix("...assistant\n<think>\n\n</think>\n\n", thinking=False) == ""
    assert m.reply_prefix("...assistant\n<think>\n\n</think>\n\n", thinking=True) == ""
    assert m.reply_prefix("...model\n<|channel>thought\n<channel|>", thinking=False) == ""
    assert m.reply_prefix("...assistant\n<think>\n", thinking=True) == "<think>"
    for tag in ("<think>", "<thinking>", "<thought>"):
        with pytest.raises(ValueError, match="thinking"):
            m.reply_prefix(f"...assistant\n{tag}\n", thinking=False)


def test_mlx_model_thinking_off(monkeypatch):
    sent = _fake_mlx_lm(monkeypatch, '{"sql": "SELECT 1", "grain": "one row", "columns": []}')
    m = _load("an_model_mlx", "model_mlx.py")
    model = m.MlxModel("qwen36-35b")
    assert model.config_id == "qwen36-35b" and model.config == m.MODEL_CONFIGS["qwen36-35b"]

    text, seconds = model.generate("How many orders?")
    assert text == '{"sql": "SELECT 1", "grain": "one row", "columns": []}'
    assert isinstance(seconds, float) and seconds >= 0
    assert sent[0]["model"] == "model at mlx-community/Qwen3.6-35B-A3B-4bit"
    assert sent[0]["max_tokens"] == 1024
    assert "How many orders?" in sent[0]["prompt"] and sent[0]["prompt"].endswith("</think>\n\n")
    assert set(sent[0]) == {"model", "prompt", "max_tokens"}   # default greedy sampler


def test_mlx_model_thinking_on_puts_the_tag_back(monkeypatch):
    sent = _fake_mlx_lm(monkeypatch, 'Let me count {orders}.\n</think>\n\n{"cannot_answer": "no costs"}')
    m = _load("an_model_mlx", "model_mlx.py")
    model = m.MlxModel("qwen38-27b-think")

    text, _ = model.generate("What did shipping cost?")
    assert text.startswith("<think>Let me count")
    assert sent[0]["max_tokens"] == 8192 and sent[0]["prompt"].endswith("<think>\n")
    assert mod.parse_reply(text).kind == "cannot"


def test_mlx_model_refuses_open_thinking_when_off(monkeypatch):
    _fake_mlx_lm(monkeypatch, "unused")
    m = _load("an_model_mlx", "model_mlx.py")
    model = m.MlxModel("qwen36-35b")
    model.tokenizer = FakeTokenizer(open_when=False)   # a template that ignores enable_thinking=False
    with pytest.raises(ValueError, match="thinking"):
        model.generate("Q?")


def test_mlx_model_unknown_config(monkeypatch):
    _fake_mlx_lm(monkeypatch, "unused")
    m = _load("an_model_mlx", "model_mlx.py")
    with pytest.raises(ValueError, match="qwen36-35b"):
        m.MlxModel("gpt-5")
