from __future__ import annotations

# Forbidden ranges, narrowest first, so a tick in the emoji block is named a tick.
FORBIDDEN = (
    (0x2713, 0x2718, "tick or cross"),
    (0x2014, 0x2014, "em dash"),
    (0x2015, 0x2015, "horizontal bar"),
    (0x2190, 0x21FF, "arrow"),
    (0x2300, 0x23FF, "technical symbol"),
    (0x25A0, 0x25FF, "geometric shape"),
    (0x2600, 0x27BF, "emoji"),
    (0x27F0, 0x27FF, "arrow"),
    (0x2900, 0x297F, "arrow"),
    (0x2B00, 0x2BFF, "symbol or arrow"),
    (0x2E3A, 0x2E3B, "long dash"),
    (0x1F000, 0x1FAFF, "emoji"),
)


# What a character breaks, or None when it's allowed.
def _kind(char: str) -> str | None:
    code = ord(char)
    return next((kind for low, high, kind in FORBIDDEN if low <= code <= high), None)


# One entry per forbidden character, with its code point and position: "U+2014 at 12 (em dash)".
def problems(text: str) -> list[str]:
    return [f"U+{ord(c):04X} at {i} ({kind})" for i, c in enumerate(text) if (kind := _kind(c)) is not None]
