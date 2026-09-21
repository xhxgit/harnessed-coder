"""Interactive input support for the REPL."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion, PathCompleter
from prompt_toolkit.cursor_shapes import CursorShape, SimpleCursorShapeConfig
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


REPL_COMMANDS = (
    "/compact",
    "/exit",
    "/help",
    "/history",
    "/memory",
    "/model",
    "/refresh",
    "/session",
    "/skills",
    "/status",
    "/tokens",
    "/workspace",
)
_REPL_COMMAND_DESCRIPTIONS = {
    "/compact": "Compact the current model context",
    "/exit": "Exit the CLI",
    "/help": "Show available commands",
    "/history": "Search the current session",
    "/memory": "Manage long-term memory",
    "/model": "Show or switch the model",
    "/refresh": "Clear and redraw the terminal",
    "/session": "Show or switch the session",
    "/skills": "List or reload Skills",
    "/status": "Show workspace and context status",
    "/tokens": "Show context and API usage",
    "/workspace": "Show or switch the workspace",
}
REPL_HISTORY_FILE_NAME = "repl-history"


class CallableReplInput:
    """Adapt the legacy injectable input callable used by tests and embedders."""

    def __init__(self, input_func: Callable[[str], str]) -> None:
        self._input_func = input_func

    def read(self, prompt: str) -> str:
        return self._input_func(prompt)


class ReplCompleter(Completer):
    """Complete REPL commands and path-like command arguments."""

    def __init__(self) -> None:
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        text = document.text_before_cursor
        if text.startswith("/") and not any(character.isspace() for character in text):
            yield from _command_completions(text)
            return

        token = text.rsplit(maxsplit=1)[-1] if text.strip() else ""
        if not _looks_like_path_argument(text, token):
            return
        token_document = Document(token, cursor_position=len(token))
        yield from self._path_completer.get_completions(token_document, complete_event)


class PromptToolkitReplInput:
    """Read editable, multiline input from an interactive terminal."""

    def __init__(self, *, data_dir: str | Path | None = None) -> None:
        history = None
        if data_dir is not None:
            history_path = Path(data_dir).resolve() / REPL_HISTORY_FILE_NAME
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        self._session: PromptSession[str] | None
        try:
            self._session = PromptSession(
                history=history,
                completer=ReplCompleter(),
                key_bindings=_create_key_bindings(),
                multiline=True,
                complete_while_typing=True,
                prompt_continuation="  ",
                cursor=SimpleCursorShapeConfig(CursorShape.BLINKING_BEAM),
            )
        except Exception as exc:
            if exc.__class__.__name__ != "NoConsoleScreenBufferError":
                raise
            # PyCharm's non-terminal Run console has stdin/stdout, but no
            # Windows console screen buffer. Basic input remains usable there.
            self._session = None

    @property
    def enhanced_editing(self) -> bool:
        return self._session is not None

    def read(self, prompt: str) -> str:
        if self._session is None:
            return input(prompt)
        return self._session.prompt(prompt)


def _create_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add(Keys.BracketedPaste)
    def insert_pasted_text(event: object) -> None:
        pasted = event.data.replace("\r\n", "\n").replace("\r", "\n")  # type: ignore[attr-defined]
        event.current_buffer.insert_text(pasted)  # type: ignore[attr-defined]

    @bindings.add("enter")
    def accept_input(event: object) -> None:
        event.current_buffer.validate_and_handle()  # type: ignore[attr-defined]

    # Most terminals encode Alt+Enter as Escape followed by Enter.
    @bindings.add("escape", "enter")
    def insert_newline(event: object) -> None:
        event.current_buffer.insert_text("\n")  # type: ignore[attr-defined]

    return bindings


def _command_completions(text: str) -> Iterable[Completion]:
    for command in REPL_COMMANDS:
        if command.startswith(text):
            yield Completion(
                command[len(text) :],
                start_position=0,
                display=command,
                display_meta=_REPL_COMMAND_DESCRIPTIONS[command],
            )


def _looks_like_path_argument(text: str, token: str) -> bool:
    if text.startswith("/workspace "):
        return True
    return any(marker in token for marker in ("/", "\\", ".", "~"))
