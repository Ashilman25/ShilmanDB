"""Tests for app/model/prompt_cache.py and app/model/model_cache.py, with stand-ins for MLX."""
from __future__ import annotations

import contextlib
import sys
import types

import pytest

from app.core import analyst_check
from app.core.datasets import MODEL_ID, load_dataset
from app.model import model_cache, prompt_cache
from app.model.prompt_cache import CHUNK, cache_len, dummy_prompts, encode, match, shared_prefix_len, starts_with

pr = analyst_check.pr
BOS = "<bos>"


def test_cache_len():
    assert CHUNK == 2048
    assert cache_len(5150) == 4096
    assert cache_len(2890) == 2048
    assert cache_len(2047) == 0
    assert cache_len(4096) == 4096
    assert cache_len(10, chunk=4) == 8


def test_shared_prefix_len():
    assert shared_prefix_len([1, 2, 3, 4], [1, 2, 9]) == 2
    assert shared_prefix_len([1, 2, 3], [1, 2, 3]) == 3
    assert shared_prefix_len([1, 2], [1, 2, 3]) == 2
    assert shared_prefix_len([], [1]) == 0
    assert shared_prefix_len([5], [1]) == 0


def test_dummy_prompts_share_everything_but_the_question(fixture_paths):
    ctx = load_dataset(fixture_paths, "tpch").context
    first, second = dummy_prompts(ctx)
    head = "## Question\n\n"
    cut = first.index(head) + len(head)

    assert first[:cut] == second[:cut]
    assert first[cut:] != second[cut:]
    assert first[cut] != second[cut]  # the questions differ from their first character
    assert first.startswith(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, "x")[:cut])
    assert first.count(head) == second.count(head) == 1


def test_starts_with():
    assert starts_with([1, 2, 3], [1, 2])
    assert starts_with([1, 2], [1, 2])
    assert starts_with([1, 2], [])
    assert not starts_with([1, 2], [1, 3])
    assert not starts_with([1], [1, 2])


# A tokenizer stand-in: one token per character, BOS as token 0 when special tokens are added.
class CharTokenizer:
    def __init__(self, bos_token: str | None = BOS):
        self.bos_token = bos_token
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls.append((text, add_special_tokens))
        return ([0] if add_special_tokens else []) + [ord(c) for c in text]

    # A chat template shaped like Gemma's: BOS, the user turn, then the model turn opened.
    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        assert add_generation_prompt and not tokenize and not enable_thinking
        return f"{BOS}<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n<start_of_turn>model\n"


def test_encode_follows_stream_generate():
    no_bos = CharTokenizer(bos_token=None)
    assert encode(no_bos, "ab") == [0, 97, 98]
    assert no_bos.calls == [("ab", True)]

    tok = CharTokenizer()
    assert encode(tok, "<bos>ab") == [ord(c) for c in "<bos>ab"]
    assert encode(tok, "ab") == [0, 97, 98]
    assert tok.calls == [("<bos>ab", False), ("ab", True)]
    assert isinstance(encode(tok, "ab"), list)


def test_match_needs_a_longer_prompt_and_an_equal_prefix():
    kept = {"tpch": [1, 2, 3, 4], "reviews": [7, 8]}
    assert match([1, 2, 3, 4, 5], kept) == ("tpch", 4)
    assert match([7, 8, 1], kept) == ("reviews", 2)
    assert match([1, 2, 3, 4], kept) is None  # no longer than the prefix
    assert match([1, 2, 3, 5, 6], kept) is None
    assert match([9, 9, 9], kept) is None
    assert match([1, 2, 3, 4, 5], {}) is None
    assert match([1, 2, 3], {"a": [1], "b": [1, 2]}) == ("a", 1)  # tried in order


# --- CachedModel, with a stand-in for the analyst check's MlxModel and for MLX itself ---

MODEL_CONFIG = {"path": "org/model", "revision": "abc", "thinking": False, "max_tokens": 1024}
PLAIN_SECONDS = 0.25


# Stands in for the analyst check's MlxModel: same attributes, a reply that names the prompt.
class FakeMlxModel:
    made: list = []

    def __init__(self, config_id: str):
        self.config_id = config_id
        self.config = dict(MODEL_CONFIG)
        self.model = object()
        self.tokenizer = CharTokenizer()
        self.prompts: list[str] = []
        FakeMlxModel.made.append(self)

    def generate(self, prompt: str) -> tuple[str, float]:
        self.prompts.append(prompt)
        return f"plain reply to {len(prompt)} chars", PLAIN_SECONDS


# Contexts whose dummy prompts share long_text plus the prompt's fixed parts.
def _context(long_text: str):
    return analyst_check.rp.Context(long_text, "rules", [])


# The tokens CachedModel works with for one prompt.
def _tokens(prompt: str) -> list[int]:
    tok = CharTokenizer()
    return encode(tok, tok.apply_chat_template([{"role": "user", "content": prompt}], True, False, False))


@pytest.fixture
def mlx_stub(monkeypatch):
    real = analyst_check.load_model_mlx()
    stub = types.SimpleNamespace(MlxModel=FakeMlxModel, render_chat=real.render_chat, reply_prefix=real.reply_prefix)
    monkeypatch.setattr(analyst_check, "load_model_mlx", lambda: stub)
    FakeMlxModel.made = []

    calls = {"fill": [], "copy": [], "generate": []}

    def fill(model, prefix):
        calls["fill"].append((model, list(prefix)))
        return [f"kept cache {len(calls['fill'])}"]

    def copy(kept):
        calls["copy"].append(kept)
        return [f"copy of {kept[0]}"]

    def generate(model, tokenizer, **kwargs):
        calls["generate"].append((model, tokenizer, kwargs))
        return "cached reply"

    monkeypatch.setattr(model_cache, "fill_cache", fill)
    monkeypatch.setattr(model_cache, "fresh_copy", copy)
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = generate
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    return calls


def test_cached_model_fills_and_hits(mlx_stub):
    ctx = _context("d" * 5000)
    model = model_cache.CachedModel(MODEL_ID, {"tpch": ctx, "reviews": _context("short")})
    plain = FakeMlxModel.made[0]
    first, second = (_tokens(p) for p in dummy_prompts(ctx))
    n = cache_len(shared_prefix_len(first, second))

    assert len(FakeMlxModel.made) == 1 and model.plain is plain
    assert (model.config_id, model.config) == (MODEL_ID, MODEL_CONFIG)
    assert n == 4096 and model.cached_tokens == {"tpch": 4096, "reviews": 0}
    assert mlx_stub["fill"] == [(plain.model, first[:n])]
    assert model.last_cached_tokens == 0

    prompt = pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, "How many orders?")
    text, seconds = model.generate(prompt)
    tokens = _tokens(prompt)
    assert text == "cached reply" and seconds >= 0
    assert mlx_stub["copy"] == [["kept cache 1"]]
    assert mlx_stub["generate"] == [(plain.model, plain.tokenizer,
                                     {"prompt": tokens[n:], "prompt_cache": ["copy of kept cache 1"], "max_tokens": 1024})]
    assert model.last_cached_tokens == 4096
    assert plain.prompts == []

    # Each hit gets its own copy of the kept cache.
    model.generate(prompt)
    assert mlx_stub["copy"] == [["kept cache 1"], ["kept cache 1"]]


def test_cached_model_misses_go_to_the_plain_model(mlx_stub):
    ctx = _context("d" * 5000)
    model = model_cache.CachedModel(MODEL_ID, {"tpch": ctx})
    model.generate(pr.build_sql_prompt(ctx.description, ctx.rules, ctx.examples, "How many orders?"))
    assert model.last_cached_tokens == 4096

    rephrase = pr.build_rephrase_prompt("How many orders?")
    assert model.generate(rephrase) == (f"plain reply to {len(rephrase)} chars", PLAIN_SECONDS)
    assert model.last_cached_tokens == 0
    assert model.plain.prompts == [rephrase]
    assert len(mlx_stub["generate"]) == 1

    other = pr.build_sql_prompt("e" * 5000, "rules", [], "How many orders?")
    assert model.generate(other)[1] == PLAIN_SECONDS
    assert model.plain.prompts == [rephrase, other]


def test_cached_model_keeps_nothing_for_a_short_prefix(mlx_stub):
    model = model_cache.CachedModel(MODEL_ID, {"tpch": _context("short")})
    assert model.cached_tokens == {"tpch": 0}
    assert mlx_stub["fill"] == []
    prompt = pr.build_sql_prompt("short", "rules", [], "How many?")
    assert model.generate(prompt) == (f"plain reply to {len(prompt)} chars", PLAIN_SECONDS)


def test_cached_model_keeps_the_reply_prefix_rule(mlx_stub, monkeypatch):
    monkeypatch.setattr(CharTokenizer, "apply_chat_template",
                        lambda self, messages, **kw: f"{BOS}{messages[0]['content']}<think>")
    model = model_cache.CachedModel(MODEL_ID, {})
    with pytest.raises(ValueError, match="thinking is not switched off"):
        model.generate("a question")
    assert model.plain.prompts == []


# --- the MLX calls themselves, on a stand-in mx that records every operation ---

# A 1-D token array: slicing gives a new one, [None] marks a batch of one.
class FakeArray:
    def __init__(self, data):
        self.data = list(data.data) if isinstance(data, FakeArray) else list(data)
        self.batched = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if key is None:
            out = FakeArray(self.data)
            out.batched = True
            return out
        return FakeArray(self.data[key])


# A cache shaped like mlx_lm's KVCache: state is keys, values, offset.
class FakeKVCache:
    def __init__(self):
        self.keys, self.values, self.offset = FakeArray([1, 2]), FakeArray([3, 4]), 2

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.state = state
        return obj


@pytest.fixture
def fake_mx(monkeypatch):
    events: list = []

    @contextlib.contextmanager
    def scope(name):
        events.append(f"enter {name}")
        yield
        events.append(f"exit {name}")

    mx = types.ModuleType("mlx.core")
    mx.array = FakeArray
    mx.eval = lambda states: events.append(("eval", len(states)))
    mx.clear_cache = lambda: events.append("clear")
    mx.stream = lambda s: scope(f"stream {s}")
    mlx = types.ModuleType("mlx")
    mlx.core = mx

    gen = types.ModuleType("mlx_lm.generate")
    gen.generation_stream = "gen"
    gen.wired_limit = lambda model, streams: scope(f"wired {streams}")
    cache_mod = types.ModuleType("mlx_lm.models.cache")
    cache_mod.make_prompt_cache = lambda model, max_kv_size=None: events.append(("make", max_kv_size)) or [FakeKVCache(), FakeKVCache()]
    mlx_lm = types.ModuleType("mlx_lm")
    models = types.ModuleType("mlx_lm.models")
    mlx_lm.generate, mlx_lm.models, models.cache = gen, models, cache_mod

    for name, module in {"mlx": mlx, "mlx.core": mx, "mlx_lm": mlx_lm, "mlx_lm.generate": gen,
                         "mlx_lm.models": models, "mlx_lm.models.cache": cache_mod}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return events


def test_fill_cache_prefills_like_generate_step(fake_mx):
    calls = []

    def model(tokens, cache):
        calls.append((tokens.batched, tokens.data, cache))

    prefix = list(range(2 * CHUNK))
    cache = model_cache.fill_cache(model, prefix)

    assert [c[1] for c in calls] == [prefix[:CHUNK], prefix[CHUNK:]]
    assert all(batched and c is cache for batched, _, c in calls)
    assert fake_mx == [("make", None), "enter wired ['gen']", "enter stream gen",
                       ("eval", 2), "clear", ("eval", 2), "clear",
                       "exit stream gen", "exit wired ['gen']"]


# A prefix that isn't whole chunks would end mid-piece, off generate_step's boundaries.
@pytest.mark.parametrize("length", [1, CHUNK - 1, CHUNK + 1, 2 * CHUNK + 7])
def test_fill_cache_refuses_a_partial_chunk(length):
    with pytest.raises(ValueError, match="whole number of"):
        model_cache.fill_cache(None, list(range(length)))
    assert "mlx" not in sys.modules and "mlx_lm" not in sys.modules  # refused before MLX is imported


def test_fresh_copy_makes_new_arrays(fake_mx):
    kept = [FakeKVCache(), FakeKVCache()]
    copies = model_cache.fresh_copy(kept)

    for old, new in zip(kept, copies):
        assert type(new) is FakeKVCache and new is not old
        assert new.keys is not old.keys and new.values is not old.values
        assert (new.keys.data, new.values.data, new.offset) == (old.keys.data, old.values.data, old.offset)
    assert fake_mx == [("eval", 2)]

    copies[0].keys.data[0] = 99
    assert kept[0].keys.data == [1, 2]


def test_fresh_copy_refuses_a_cache_with_fields_outside_its_state(fake_mx):
    kept = FakeKVCache()
    kept.extra = 1
    with pytest.raises(TypeError, match="FakeKVCache"):
        model_cache.fresh_copy([kept])


def test_prompt_cache_imports_no_mlx():
    assert "mlx" not in prompt_cache.__dict__ and "mx" not in prompt_cache.__dict__
    assert "mx" not in model_cache.__dict__
