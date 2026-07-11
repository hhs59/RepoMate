from __future__ import annotations

import json
import re

from src.config import get_settings
from src.rag.store import retrieve_with_context, retrieve_with_qa, _get_or_create_collection
from src.rag.context import build_context
from src.rag.models import RetrievedChunk
from src.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_FRAMING = (
    "You are repomate, an expert assistant for this codebase. "
    "Use ONLY the context below to answer. If the answer isn't in the "
    "context, say you don't know. Cite file and line ranges. "
    "If the user asks to change/edit code, check the listed callers and "
    "callees and warn about anything that might break."
)

_VAGUE_PATTERNS = [
    r"tell me about",
    r"what is this",
    r"overview",
    r"summarize",
    r"what does this repo",
    r"what does this project",
    r"explain this codebase",
    r"describe this",
]


def _is_vague_query(query: str) -> bool:
    lower = query.lower()
    return any(re.search(pat, lower) for pat in _VAGUE_PATTERNS)


def _rewrite_query(query: str) -> str:
    lower = query.lower()
    if _is_vague_query(query):
        return (
            f"{query} README overview description purpose "
            f"main features architecture entry point"
        )
    return query


def _get_readme_chunk(slug: str) -> RetrievedChunk | None:
    settings = get_settings()
    chunks_path = settings.data_dir / slug / "chunks.jsonl"
    if not chunks_path.exists():
        return None

    best = None
    best_len = 0
    with open(chunks_path) as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            fname = c.get("file", "").lower()
            if fname.endswith("readme.md") or fname == "readme" or fname.endswith("readme"):
                code = c.get("raw_code", "")
                if len(code) > best_len:
                    best = c
                    best_len = len(code)

    if best:
        return RetrievedChunk(
            file=best["file"],
            start_line=int(best.get("start_line", 1)),
            end_line=int(best.get("end_line", 1)),
            symbol=best.get("symbol"),
            raw_code=best["raw_code"][:4000],
            score=1.0,
            citation=f"{best['file']}:{best.get('start_line', 1)}-{best.get('end_line', 1)}",
            language=best.get("language", "text"),
            kind="readme",
            relation="target",
            chunk_id=best.get("id", ""),
        )
    return None


def _qa_collection_exists(slug: str) -> bool:
    try:
        coll = _get_or_create_collection(slug, "qa")
        return coll.count() > 0
    except Exception:
        return False


def build_prompt(
    slug: str, user_query: str, k: int = 5, mode: str = "rag",
) -> tuple[str, list[dict]]:
    settings = get_settings()

    rewritten = _rewrite_query(user_query)
    if rewritten != user_query:
        logger.info("Rewrote query: '%s' → '%s'", user_query, rewritten)

    fetch_k = k + 3 if _is_vague_query(user_query) else k

    if _qa_collection_exists(slug):
        retrieved = retrieve_with_qa(slug, rewritten, k=fetch_k)
    else:
        retrieved = retrieve_with_context(slug, rewritten, k=fetch_k, expand=2)

    readme = _get_readme_chunk(slug)
    if readme:
        existing_ids = {r.chunk_id for r in retrieved}
        if readme.chunk_id not in existing_ids:
            retrieved.insert(0, readme)
        else:
            for r in retrieved:
                if r.chunk_id == readme.chunk_id:
                    r.score = 1.0
                    retrieved.remove(r)
                    retrieved.insert(0, r)
                    break

    retrieved = retrieved[:k + 3]

    max_ctx = 4000 if mode == "rag" else 3500
    context, citations = build_context(retrieved, max_tokens=max_ctx)

    system = _SYSTEM_FRAMING
    if mode == "hybrid":
        system = (
            "You are repomate, an expert assistant for this codebase. "
            "You have been fine-tuned on this repo's style and patterns. "
            "Use the context below and your training to answer. Cite file and "
            "line ranges. If the user asks to change/edit code, check the "
            "listed callers and callees and warn about anything that might break."
        )

    prompt = (
        f"<start_of_turn>user\n"
        f"{system}\n"
        f"Relevant code:\n{context}\n\n"
        f"Question: {user_query}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    return prompt, citations
