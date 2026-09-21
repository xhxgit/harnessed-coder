"""Rolling context-summary budget calculations."""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import DEFAULT_CONTEXT_SUMMARY_FULL_GROWTH_UNITS


SUMMARY_TOKEN_GRANULARITY = 100


@dataclass(frozen=True)
class SummaryBudgetCalculator:
    """Map prior summary size and newly summarized tokens to a maximum."""

    initial_tokens: int
    cap_tokens: int
    growth_unit_tokens: int
    full_growth_units: float = DEFAULT_CONTEXT_SUMMARY_FULL_GROWTH_UNITS

    def __post_init__(self) -> None:
        if self.initial_tokens < 1:
            raise ValueError("initial_tokens must be at least 1")
        if self.cap_tokens < self.initial_tokens:
            raise ValueError("cap_tokens must be greater than or equal to initial_tokens")
        if self.growth_unit_tokens < 1:
            raise ValueError("growth_unit_tokens must be at least 1")
        if self.full_growth_units <= 1.0:
            raise ValueError("full_growth_units must be greater than 1")

    def maximum_for_update(
        self,
        *,
        previous_summary_tokens: int | None,
        added_tokens: int,
    ) -> int:
        """Return the maximum token budget for the next rolling summary."""
        if previous_summary_tokens is not None and previous_summary_tokens < 0:
            raise ValueError("previous_summary_tokens must be non-negative")
        if added_tokens < 0:
            raise ValueError("added_tokens must be non-negative")

        added_units = added_tokens / self.growth_unit_tokens
        if previous_summary_tokens is None:
            position = max(1.0, added_units)
        else:
            position = self._position_for_summary(previous_summary_tokens) + added_units
        return self._quantize_maximum(self._upper_tokens(position))

    @staticmethod
    def _quantize_maximum(maximum: int) -> int:
        if maximum < SUMMARY_TOKEN_GRANULARITY:
            return maximum
        return max(
            SUMMARY_TOKEN_GRANULARITY,
            maximum // SUMMARY_TOKEN_GRANULARITY * SUMMARY_TOKEN_GRANULARITY,
        )

    def _position_for_summary(self, summary_tokens: int) -> float:
        if self.cap_tokens == self.initial_tokens:
            return 1.0
        position = 1.0 + self._position_span * (
            summary_tokens - self.initial_tokens
        ) / (self.cap_tokens - self.initial_tokens)
        return min(self.full_growth_units, max(0.0, position))

    def _upper_tokens(self, position: float) -> int:
        if self.cap_tokens == self.initial_tokens:
            return self.initial_tokens
        return min(
            self.cap_tokens,
            max(self.initial_tokens, round(self._raw_tokens(position))),
        )

    def _raw_tokens(self, position: float) -> float:
        if self.cap_tokens == self.initial_tokens:
            return float(self.initial_tokens)
        bounded_position = min(
            self.full_growth_units,
            max(0.0, position),
        )
        return self.initial_tokens + (self.cap_tokens - self.initial_tokens) * (
            bounded_position - 1.0
        ) / self._position_span

    @property
    def _position_span(self) -> float:
        return self.full_growth_units - 1.0
