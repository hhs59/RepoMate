from __future__ import annotations

from src.rag.models import RetrievedChunk
from src.config import get_device
from src.logging import get_logger

logger = get_logger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        from src.config import get_settings

        settings = get_settings()
        device = get_device()
        logger.info("Loading reranker %s on %s", settings.reranker_model_id, device)
        _reranker = CrossEncoder(settings.reranker_model_id, device=device)
    return _reranker


def rerank(
    query: str, chunks: list[RetrievedChunk], top_k: int = 5,
) -> list[RetrievedChunk]:
    if not chunks:
        return chunks
    if len(chunks) == 1:
        return chunks

    model = _get_reranker()
    pairs = [[query, c.raw_code] for c in chunks]
    scores = model.predict(pairs)

    for c, s in zip(chunks, scores):
        c.score = float(s)

    chunks.sort(key=lambda x: -x.score)
    return chunks[:top_k]
