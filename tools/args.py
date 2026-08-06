# tools/args.py
"""
Forgiving coercions for tool arguments supplied by a language model.

Schemas are a hint to a language model, not a contract it can be held to: a
field typed as an array of strings regularly arrives as a bare string, and a
number as "3" or 3.0. Bending where the intent is obvious beats failing a tool
call over a quoting detail — especially since a rejected call costs a whole
round trip to correct.

These live apart from any one tool because every tool takes arguments from the
same unreliable source.
"""

from __future__ import annotations

from typing import Any


def as_str_list(value: Any) -> list[str]:
    """Coerces a schema 'array of string' field the LLM may send as a string."""
    if isinstance(value, str):
        return [value] if value.strip() else []

    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]

    return []


def as_flag(kwargs: dict[str, Any], *names: str, default: bool = False) -> bool:
    """Reads a boolean field, accepting any of its accepted spellings."""
    for name in names:
        if name in kwargs and kwargs[name] is not None:
            return bool(kwargs[name])

    return default


def as_count(kwargs: dict[str, Any], *names: str) -> int | None:
    """Reads a positive integer field; ignores junk instead of failing."""
    for name in names:
        raw = kwargs.get(name)
        if raw is None:
            continue

        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue

        if value > 0:
            return value

    return None
