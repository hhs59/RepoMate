from __future__ import annotations

import time

from src.config import get_settings
from src.serve.prompt import build_prompt
from src.serve.vllm_server import chat, chat_llm_fallback
from src.serve.vllm_lora import adapter_exists, FT_MODEL_NAME
from src.logging import get_logger

logger = get_logger(__name__)


def answer(
    slug: str, query: str, mode: str = "hybrid",
) -> dict:
    t0 = time.perf_counter()

    prompt, citations = build_prompt(slug, query, mode=mode)

    model_name = None
    if mode == "hybrid" and adapter_exists(slug):
        model_name = FT_MODEL_NAME

    result = None
    try:
        result = chat(prompt, model=model_name)
    except Exception as e:
        logger.warning("vLLM chat failed: %s", e)

    if result is None:
        try:
            result = chat_llm_fallback(prompt)
        except Exception as e:
            logger.warning("LLM fallback also failed: %s", e)
            result = (
                "No LLM backend available. To use repomate, either:\n"
                "1. Run on a machine with a GPU (vLLM will auto-start), or\n"
                "2. Set REPOMATE_LLM_API_KEY in .env for cloud inference."
            )

    if not result or len(result.strip()) < 5:
        result = "No response generated. Check that vLLM is running or REPOMATE_LLM_API_KEY is set in .env."

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return {
        "answer": result,
        "citations": citations,
        "elapsed_ms": elapsed_ms,
        "mode": mode,
    }
