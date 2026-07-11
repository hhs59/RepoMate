from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.rag.models import RetrievedChunk
from src.rag.store import retrieve, _get_or_create_collection
from src.logging import get_logger

logger = get_logger(__name__)

_BM25_CACHE: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_bm25(slug: str) -> tuple[BM25Okapi | None, list[dict]]:
    if slug in _BM25_CACHE:
        return _BM25_CACHE[slug]

    collection = _get_or_create_collection(slug, "code")
    if collection.count() == 0:
        return None, []

    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)

    entries = [
        {"id": eid, "meta": meta, "doc": doc}
        for eid, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"])
    ]
    _BM25_CACHE[slug] = (bm25, entries)
    return bm25, entries


def clear_bm25_cache(slug: str | None = None) -> None:
    if slug is not None:
        _BM25_CACHE.pop(slug, None)
    else:
        _BM25_CACHE.clear()


def retrieve_hybrid(
    slug: str,
    query: str,
    k: int = 5,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    overfetch: int = 4,
) -> list[RetrievedChunk]:
    fetch_n = k * overfetch
    dense_results = retrieve(slug, query, k=fetch_n)
    dense_map = {r.chunk_id: r for r in dense_results}

    bm25, entries = _get_bm25(slug)
    if bm25 is None:
        return dense_results[:k]

    tokenized_query = _tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    max_bm25 = float(bm25_scores.max()) if len(bm25_scores) > 0 else 0.0
    if max_bm25 > 0:
        bm25_norm = bm25_scores / max_bm25
    else:
        bm25_norm = bm25_scores

    bm25_lookup = {entries[i]["id"]: float(bm25_norm[i]) for i in range(len(entries))}

    bm25_top_indices = np.argsort(bm25_norm)[::-1][:fetch_n]
    for idx in bm25_top_indices:
        cid = entries[idx]["id"]
        if cid not in dense_map:
            meta = entries[idx]["meta"]
            dense_map[cid] = RetrievedChunk(
                file=meta.get("file", ""),
                start_line=int(meta.get("start_line", 1)),
                end_line=int(meta.get("end_line", 1)),
                symbol=meta.get("symbol") or None,
                raw_code=entries[idx]["doc"],
                score=0.0,
                citation=f"{meta.get('file', '')}:{meta.get('start_line', 1)}-{meta.get('end_line', 1)}",
                language=meta.get("language", "text"),
                kind=meta.get("kind", "chunk"),
                chunk_id=cid,
            )

    for r in dense_map.values():
        dense_score = r.score
        sparse_score = bm25_lookup.get(r.chunk_id, 0.0)
        r.score = dense_weight * dense_score + sparse_weight * sparse_score

    merged = list(dense_map.values())
    merged.sort(key=lambda x: -x.score)
    return merged[:k]
