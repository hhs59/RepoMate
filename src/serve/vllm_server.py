from __future__ import annotations

import subprocess
import sys
import time
from typing import Iterator

import httpx
from openai import OpenAI

from src.config import get_settings, get_base_model_id
from src.logging import get_logger

logger = get_logger(__name__)

_vllm_proc: subprocess.Popen | None = None
_client: OpenAI | None = None


def start_vllm_server(
    model: str | None = None,
    port: int | None = None,
    hybrid: bool = False,
    adapter_dir: str | None = None,
) -> subprocess.Popen:
    global _vllm_proc
    settings = get_settings()
    model = model or get_base_model_id()
    port = port or settings.vllm_port

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--dtype", "bfloat16",
        "--max-model-len", "8192",
        "--port", str(port),
        "--gpu-memory-utilization", "0.85",
    ]
    if hybrid and adapter_dir:
        cmd.extend(["--enable-lora", "--lora-modules", f"repomate-ft={adapter_dir}"])

    logger.info("Starting vLLM server: %s", " ".join(cmd))
    _vllm_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return _vllm_proc


def is_ready(port: int | None = None) -> bool:
    settings = get_settings()
    port = port or settings.vllm_port
    try:
        resp = httpx.get(f"http://localhost:{port}/health", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_ready(port: int | None = None, timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_ready(port):
            return True
        time.sleep(3)
    return False


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        base_url = f"http://localhost:{settings.vllm_port}/v1"
        _client = OpenAI(api_key="repomate-local", base_url=base_url)
    return _client


def chat(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = False,
    model: str | None = None,
) -> str | Iterator:
    settings = get_settings()
    model = model or get_base_model_id()
    client = _get_client()

    if stream:
        return client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body={"add_special_tokens": False},
        )

    resp = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"add_special_tokens": False},
    )
    return resp.choices[0].text


def chat_llm_fallback(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str | Iterator:
    from src.config import get_llm_api_key, get_llm_base_url, get_llm_model
    client = OpenAI(api_key=get_llm_api_key(), base_url=get_llm_base_url())

    messages = [{"role": "user", "content": prompt}]

    if stream:
        return client.chat.completions.create(
            model=get_llm_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

    resp = client.chat.completions.create(
        model=get_llm_model(),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""
