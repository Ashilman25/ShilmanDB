from __future__ import annotations

from app import sixd

# mlx_lm 0.32.2's prefill_step_size: generate_step feeds the prompt in pieces this long.
CHUNK = 2048

# Two questions that differ from their first character, so the prompts share everything before them.
DUMMY_QUESTIONS = ("How many rows does the largest table hold?", "Which column has the most distinct values?")


# How many leading tokens two lists share.
def shared_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# The shared length rounded down to whole chunks, so a cached prefix ends where generate_step's pieces do.
def cache_len(shared: int, chunk: int = CHUNK) -> int:
    return (shared // chunk) * chunk


# Two base prompts for one dataset that differ only in the question.
def dummy_prompts(context) -> tuple[str, str]:
    first, second = (sixd.pr.build_sql_prompt(context.description, context.rules, context.examples, q) for q in DUMMY_QUESTIONS)
    return first, second


# Whether tokens begin with prefix.
def starts_with(tokens: list[int], prefix: list[int]) -> bool:
    return len(tokens) >= len(prefix) and tokens[:len(prefix)] == prefix


# Tokens for a rendered prompt, as mlx_lm's stream_generate encodes a string:
# special tokens only when there is no BOS token or the text doesn't start with it.
def encode(tokenizer, text: str) -> list[int]:
    add_special_tokens = tokenizer.bos_token is None or not text.startswith(tokenizer.bos_token)
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


# The first kept prefix the tokens start with and outgrow, as (name, length); None when none fits.
# Outgrowing it keeps generate_step's later pieces on the same chunk boundaries.
def match(tokens: list[int], kept: dict[str, list[int]]) -> tuple[str, int] | None:
    for name, prefix in kept.items():
        if len(tokens) > len(prefix) and starts_with(tokens, prefix):
            return name, len(prefix)
    return None
