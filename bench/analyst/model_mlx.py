from __future__ import annotations

import time

# A plain literal, so other scripts can read it with ast.literal_eval without mlx installed.
MODEL_CONFIGS = {
    "gemma4-26b": {"path": "mlx-community/gemma-4-26b-a4b-it-4bit", "thinking": False, "max_tokens": 1024},
    "qwen36-35b": {"path": "mlx-community/Qwen3.6-35B-A3B-4bit", "thinking": False, "max_tokens": 1024},
    "qwen36-35b-think": {"path": "mlx-community/Qwen3.6-35B-A3B-4bit", "thinking": True, "max_tokens": 8192},
    "qwen38-27b": {"path": "mlx-community/Qwen3.8-27B-4bit", "thinking": False, "max_tokens": 1024},
    "qwen38-27b-think": {"path": "mlx-community/Qwen3.8-27B-4bit", "thinking": True, "max_tokens": 8192},
}

OPEN_THINKING_TAGS = ("<think>", "<thinking>", "<thought>")


# The chat template on one user message, with thinking on or off.
def render_chat(tokenizer, prompt: str, thinking: bool) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=thinking)
    except TypeError:  # a template that doesn't know the keyword has no thinking mode
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


# The thinking tag the rendered prompt ends inside, or None.
def open_thinking_tag(rendered: str) -> str | None:
    tail = rendered.rstrip()
    return next((tag for tag in OPEN_THINKING_TAGS if tail.endswith(tag)), None)


# Text to put before the reply: the open tag when thinking is on, else "".
# Raises ValueError if thinking should be off but the prompt opens a thinking block.
def reply_prefix(rendered: str, thinking: bool) -> str:
    tag = open_thinking_tag(rendered)
    if tag is None:
        return ""
    if not thinking:
        raise ValueError(f"the chat template ends with an open {tag} block; thinking is not switched off")
    return tag


# One model config loaded with mlx_lm, answering one prompt at a time.
class MlxModel:
    def __init__(self, config_id: str):
        if config_id not in MODEL_CONFIGS:
            raise ValueError(f"model config must be one of {', '.join(MODEL_CONFIGS)}, got {config_id!r}")
        from mlx_lm import load

        self.config_id = config_id
        self.config = dict(MODEL_CONFIGS[config_id])
        self.model, self.tokenizer = load(self.config["path"])

    # The model's text for one prompt, and the seconds the generation alone took.
    def generate(self, prompt: str) -> tuple[str, float]:
        from mlx_lm import generate

        thinking = self.config["thinking"]
        rendered = render_chat(self.tokenizer, prompt, thinking)
        prefix = reply_prefix(rendered, thinking)

        start = time.perf_counter()
        text = generate(self.model, self.tokenizer, prompt=rendered, max_tokens=self.config["max_tokens"])
        return prefix + text, time.perf_counter() - start
