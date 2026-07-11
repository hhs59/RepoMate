from __future__ import annotations

import tiktoken

from src.rag.models import RetrievedChunk

_tokenizer = tiktoken.get_encoding("cl100k_base")


def build_context(
    retrieved: list[RetrievedChunk], max_tokens: int = 3500,
) -> tuple[str, list[dict]]:
    if not retrieved:
        return "(No relevant code found in the repository.)", []

    context_parts: list[str] = []
    citations: list[dict] = []
    token_count = 0

    for chunk in retrieved:
        header_parts = [f"{chunk.file}:{chunk.start_line}-{chunk.end_line}"]
        if chunk.symbol:
            header_parts.append(f" — {chunk.symbol}")
        if chunk.relation != "target":
            header_parts.append(f" [{chunk.relation}]")
        header = f"### {''.join(header_parts)}"

        block = f"{header}\n```{chunk.language}\n{chunk.raw_code}\n```\n"
        block_tokens = len(_tokenizer.encode(block))

        if token_count + block_tokens > max_tokens:
            break

        context_parts.append(block)
        token_count += block_tokens
        citations.append({
            "file": chunk.file,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol": chunk.symbol,
            "relation": chunk.relation,
        })

    return "\n".join(context_parts), citations
