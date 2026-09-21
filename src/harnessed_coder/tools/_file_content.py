"""Shared binary classification and encoding detection for text tools."""

from __future__ import annotations

import codecs
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from binaryornot.helpers import is_binary_string
from charset_normalizer import from_bytes, is_binary as is_charset_binary


_MINIMUM_COHERENCE = 0.1


@dataclass(frozen=True)
class TextFileContent:
    """Decoded text together with the codec needed to preserve its encoding."""

    text: str
    encoding: str
    raw_bytes: bytes


class UnsupportedTextFileError(ValueError):
    """Raised when a file is binary or its text encoding is ambiguous."""


def normalize_text_encoding(encoding: str) -> str:
    """Return the canonical name of a Python text codec."""
    try:
        canonical = codecs.lookup(encoding).name
        b"".decode(canonical)
        "".encode(canonical)
    except (LookupError, TypeError, ValueError) as exc:
        raise ValueError(
            f"unknown or unsupported text encoding: {encoding}"
        ) from exc
    return canonical


def normalize_text_encodings(encodings: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize and de-duplicate an ordered codec fallback list."""
    normalized: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        canonical = normalize_text_encoding(encoding)
        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return tuple(normalized)


def read_text_file(
    path: Path,
    *,
    encoding: str | None = None,
) -> TextFileContent:
    """Classify a file and decode it automatically or with one explicit codec."""
    encodings = None
    if encoding is not None:
        encodings = (normalize_text_encoding(encoding),)
    return _read_text_file(path, encodings=encodings)


def read_text_file_with_encodings(
    path: Path,
    encodings: Sequence[str],
) -> TextFileContent:
    """Classify a file and try an ordered list of explicit text codecs."""
    return _read_text_file(
        path,
        encodings=normalize_text_encodings(encodings),
    )


def _read_text_file(
    path: Path,
    *,
    encodings: tuple[str, ...] | None,
) -> TextFileContent:
    raw_bytes = read_non_binary_file_bytes(path)
    bom_encoding = _unicode_bom_encoding(raw_bytes)

    if encodings is None:
        detected_encoding = bom_encoding or _detect_encoding(raw_bytes)
        encodings = (detected_encoding,)

    for candidate in encodings:
        try:
            text = raw_bytes.decode(candidate)
        except UnicodeError:
            continue
        return TextFileContent(
            text=text,
            encoding=candidate,
            raw_bytes=raw_bytes,
        )

    requested = ", ".join(encodings)
    raise UnsupportedTextFileError(
        f"file cannot be decoded with requested encoding: {requested}"
    )


def read_non_binary_file_bytes(path: Path) -> bytes:
    """Read raw bytes after mature classifiers accept the file as text-like."""
    raw_bytes = path.read_bytes()
    bom_encoding = _unicode_bom_encoding(raw_bytes)
    if bom_encoding is not None:
        return raw_bytes

    # NUL bytes are a strong binary signal even when the remaining bytes are
    # technically valid UTF-8, so preserve that hard guard first.
    if b"\x00" in raw_bytes:
        raise UnsupportedTextFileError("file is binary")

    # A valid UTF-8 file is already unambiguously text.  Run this check before
    # binaryornot: its small-sample heuristic can classify UTF-8 source files
    # with non-ASCII text (especially Chinese comments/docstrings) as binary.
    # Non-UTF-8 text still goes through the mature binary classifiers below.
    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return raw_bytes

    if is_binary_string(raw_bytes[:512]) or is_charset_binary(raw_bytes):
        raise UnsupportedTextFileError("file is binary")
    return raw_bytes


def unsupported_text_file_error(path: str, reason: str) -> str:
    """Return a stable model-facing error for unsupported file content."""
    return f"Error: {reason}: {path}"


def _detect_encoding(raw_bytes: bytes) -> str:
    if not raw_bytes:
        return "utf-8"

    try:
        raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        return "utf-8"

    match = from_bytes(raw_bytes).best()
    if (
        match is None
        or match.encoding is None
        or match.coherence < _MINIMUM_COHERENCE
    ):
        raise UnsupportedTextFileError(
            "text encoding could not be determined reliably"
        )
    return match.encoding


def _unicode_bom_encoding(raw_bytes: bytes) -> str | None:
    """Return a BOM-aware codec before statistical binary classification."""
    if raw_bytes.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if raw_bytes.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    if raw_bytes.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    return None
