from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_settings, get_device
from src.logging import get_logger

logger = get_logger(__name__)

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        device = get_device()
        model_id = settings.bge_model_id
        logger.info("Loading embedding model %s on %s", model_id, device)
        _embedder = SentenceTransformer(model_id, device=device)
    return _embedder


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    model = _get_embedder()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
