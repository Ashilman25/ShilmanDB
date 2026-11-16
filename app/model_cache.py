from __future__ import annotations

import time

from app import sixd
from app.prompt_cache import CHUNK, cache_len, dummy_prompts, encode, match, shared_prefix_len

# MLX is imported inside the functions only, so importing this module never loads it.


# A new prompt cache holding the prefix, prefilled in CHUNK pieces exactly as mlx_lm's generate_step does.
# The prefix length must be a whole number of chunks, so every piece is CHUNK tokens.
def fill_cache(model, prefix: list[int]) -> list:
    if len(prefix) % CHUNK:
        raise ValueError(f"the prefix must be a whole number of {CHUNK}-token chunks, got {len(prefix)} tokens")

    import mlx.core as mx
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        prompt = mx.array(prefix)
        while len(prompt) > 0:
            processed = prompt[:CHUNK]
            model(processed[None], cache=cache)
            mx.eval([c.state for c in cache])
            prompt = prompt[CHUNK:]
            mx.clear_cache()
    return cache


# A copy of a kept cache list: same classes and fields, keys and values in new arrays.
# Generation writes into its cache in place, so a shared array would change the kept one.
def fresh_copy(kept: list) -> list:
    import mlx.core as mx

    copies = []
    for c in kept:
        new = type(c).from_state(tuple(mx.array(v) if isinstance(v, mx.array) else v for v in c.state))
        if set(vars(new)) != set(vars(c)):
            raise TypeError(f"can't copy a {type(c).__name__}: it has fields outside its state")
        copies.append(new)
        
    mx.eval([c.state for c in copies])
    return copies


# The analyst check's MlxModel with each dataset's shared prompt start prefilled once.
# A prompt that starts with a kept prefix skips those tokens; any other goes to the plain model.
class CachedModel:
    def __init__(self, config_id: str, contexts: dict[str, object]):
        self._mm = sixd.load_model_mlx()
        self.plain = self._mm.MlxModel(config_id)
        self.config_id = self.plain.config_id
        self.config = self.plain.config
        self.cached_tokens: dict[str, int] = {}
        self.last_cached_tokens = 0
        self._kept: dict[str, tuple[list[int], list]] = {}

        for name, context in contexts.items():
            first, second = (self._tokens(prompt) for prompt in dummy_prompts(context))
            n = cache_len(shared_prefix_len(first, second))
            self.cached_tokens[name] = n
            if n:
                self._kept[name] = (first[:n], fill_cache(self.plain.model, first[:n]))

    # The rendered prompt's tokens, as the plain model's generate would feed them.
    def _tokens(self, prompt: str) -> list[int]:
        rendered = self._mm.render_chat(self.plain.tokenizer, prompt, self.config["thinking"])
        return encode(self.plain.tokenizer, rendered)

    # The model's text and the seconds it took, like MlxModel.generate. The cached seconds also
    # count tokenizing, the prefix check and the copy.
    def generate(self, prompt: str) -> tuple[str, float]:
        from mlx_lm import generate

        thinking = self.config["thinking"]
        rendered = self._mm.render_chat(self.plain.tokenizer, prompt, thinking)
        prefix = self._mm.reply_prefix(rendered, thinking)

        start = time.perf_counter()
        tokens = encode(self.plain.tokenizer, rendered)
        hit = match(tokens, {name: kept[0] for name, kept in self._kept.items()})
        if hit is None:
            self.last_cached_tokens = 0
            return self.plain.generate(prompt)

        name, n = hit
        cache = fresh_copy(self._kept[name][1])
        text = generate(self.plain.model, self.plain.tokenizer, prompt=tokens[n:], prompt_cache=cache, max_tokens=self.config["max_tokens"])
        self.last_cached_tokens = n
        return prefix + text, time.perf_counter() - start
