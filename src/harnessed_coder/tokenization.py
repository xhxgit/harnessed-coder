"""Token counting primitives shared across harnessed-coder domains."""

from __future__ import annotations

import json
from typing import Any

import tiktoken


TOKEN_ENCODING_NAME = "o200k_base"


class TokenCounter:
    """Local tiktoken-backed token counter."""

    def __init__(self, encoding_name: str = TOKEN_ENCODING_NAME) -> None:
        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def text_tokens(self, value: str) -> int:
        return len(self._encoding.encode(value, allowed_special="all"))

    def truncate_text(self, value: str, *, max_tokens: int) -> str:
        tokens = self._encoding.encode(value, allowed_special="all")
        if len(tokens) <= max_tokens:
            return value
        return self._encoding.decode(tokens[:max_tokens])

    def truncate_text_from_end(self, value: str, *, max_tokens: int) -> str:
        tokens = self._encoding.encode(value, allowed_special="all")
        if len(tokens) <= max_tokens:
            return value
        return self._encoding.decode(tokens[-max_tokens:])


def messages_tokens(
    messages: list[dict[str, Any]],
    *,
    token_counter: TokenCounter | None = None,
) -> int:
    """Count serialized message tokens, including tool calls and metadata."""
    counter = token_counter or TokenCounter()
    return sum(
        counter.text_tokens(json.dumps(message, ensure_ascii=False, default=str))
        for message in messages
    )
