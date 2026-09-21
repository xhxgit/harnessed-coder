"""Shared model-facing response style rules."""

PLAIN_TEXT_RESPONSE_RULES = (
    "Output plain text only because the CLI does not render Markdown. "
    "Do not use Markdown syntax, including headings, bullet or numbered lists, "
    "emphasis, block quotes, fenced code blocks, tables, or Markdown links. "
    "Use short paragraphs and simple line breaks for structure."
)


__all__ = ["PLAIN_TEXT_RESPONSE_RULES"]
