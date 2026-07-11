from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

from src.config import get_llm_api_key, get_llm_base_url, get_llm_model
from src.data.cost_tracker import CostTracker
from src.logging import get_logger

logger = get_logger(__name__)

_client: OpenAI | None = None
_cost = CostTracker()


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=get_llm_api_key(),
            base_url=get_llm_base_url(),
        )
    return _client


def get_cost_tracker() -> CostTracker:
    return _cost


def complete(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
    retries: int = 4,
) -> str:
    import time as _time

    model = model or get_llm_model()

    last_err = None
    for attempt in range(retries):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            _cost.add(
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
    raise last_err


def complete_json(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict | list | None:
    raw = complete(messages, model, temperature, max_tokens)
    return parse_json_robust(raw)


def parse_json_robust(text: str) -> dict | list | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from LLM response")
    return None
