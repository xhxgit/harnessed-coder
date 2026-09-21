"""Session ownership and accounting for all model calls in an execution scope."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Iterator, ParamSpec, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from .conversation_session import ConversationSession
    from ..llm import LLMResponse

_current: ContextVar[ConversationSession | None] = ContextVar("usage_session", default=None)
_receipt: ContextVar[dict[str, int] | None] = ContextVar("usage_receipt", default=None)


def report_usage(
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
) -> None:
    receipt = _receipt.get()
    if receipt is not None:
        receipt.update(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
        )


P = ParamSpec("P")
R = TypeVar("R")

@contextmanager
def session_usage(session: ConversationSession) -> Iterator[None]:
    token = _current.set(session)
    try:
        yield
    finally:
        _current.reset(token)


def agent_usage(method: Callable[P, R]) -> Callable[P, R]:
    """Nested agents charge the existing owner, retaining isolated histories."""
    @wraps(method)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        agent: Any = args[0]
        with session_usage(_current.get() or agent.session):
            return method(*args, **kwargs)
    return wrapped


def tracked_call(call: Callable[..., LLMResponse], purpose: str, *args: Any, **kwargs: Any) -> LLMResponse:
    session = _current.get()
    response = None
    receipt: dict[str, int] = {}
    token = _receipt.set(receipt)
    try:
        response = call(*args, **kwargs)
        return response
    finally:
        _receipt.reset(token)
        if session is not None:
            session.record_usage(
                model=kwargs["model"], purpose=purpose,
                prompt_tokens=response.prompt_tokens if response else receipt.get("prompt_tokens", 0),
                completion_tokens=response.completion_tokens if response else receipt.get("completion_tokens", 0),
                cached_prompt_tokens=(
                    response.cached_prompt_tokens or 0
                    if response
                    else receipt.get("cached_prompt_tokens", 0)
                ),
                unreported=not (response.usage_available if response else receipt),
                failed=response is None,
            )
