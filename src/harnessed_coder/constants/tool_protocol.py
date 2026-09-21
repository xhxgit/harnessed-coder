"""Host-reserved fields used only in provider-facing tool history."""

HISTORICAL_COMPRESSION_ARGUMENT = "__historical_compressed__"
HISTORICAL_COMPRESSION_DESCRIPTION = (
    "Host-reserved historical projection marker. Never set this in a new tool "
    "call. If true in history, displayed summarized argument values replaced "
    "original full values only after execution; the tool received the original "
    "full values."
)
