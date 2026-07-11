from __future__ import annotations

from src.logging import get_logger

logger = get_logger(__name__)

_PRICE_IN = 0.90
_PRICE_OUT = 0.90


class CostTracker:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.requests = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.requests += 1

    @property
    def est_cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * _PRICE_IN
            + self.completion_tokens / 1_000_000 * _PRICE_OUT
        )

    def log(self) -> None:
        logger.info(
            "LLM usage: %d requests, %d prompt + %d completion tokens, est. $%.2f",
            self.requests, self.prompt_tokens, self.completion_tokens, self.est_cost_usd,
        )
