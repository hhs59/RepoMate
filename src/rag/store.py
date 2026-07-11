from __future__ import annotations

import json
import time
from pathlib import Path

import chromadb

from src.config import get_settings
from src.rag.embedder import embed_texts
from src.rag.models import RetrievedChunk
from src.logging import get_logger

logger = get_logger(__name__)

_CLIENTS: dict[str, chromadb.PersistentClient] = {}


def _get_client(slug: str) -> chromadb.PersistentClient:
    if slug not in _CLIENTS:
        settings = get_settings()
        persist_dir = str(settings.data_dir / slug / "chroma")
        _CLIENTS[slug] = chromadb.PersistentClient(path=persist_dir)
    return _CLIENTS[slug]


def _collection_name(slug: str, collection_type: str = "code") -> str:
    if collection_type == "qa":
        return f"repomate_qa_{slug}"
    return f"repomate_{slug}"


def _get_or_create_collection(slug: str, collection_type: str = "code"):
    client = _get_client(slug)
    name = _collection_name(slug, collection_type)
    try:
        return client.get_collection(name)
    except Exception:
        return client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )


def build_index(slug: str) -> int:
    settings = get_settings()
    chunks_path = settings.data_dir / slug / "chunks.jsonl"

    if not chunks_path.exists():
        logger.error("chunks.jsonl not found for %s — run `repomate init` first", slug)
        raise FileNotFoundError(f"No chunks.jsonl for {slug}")

    chunks = []
    with open(chunks_path) as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    if not chunks:
        logger.warning("No chunks to index for %s", slug)
        return 0

    t0 = time.perf_counter()
    collection = _get_or_create_collection(slug, "code")

    batch_size = 512
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["raw_code"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [
            {
                "file": c["file"],
                "language": c.get("language", "text"),
                "symbol": c.get("symbol") or "",
                "kind": c.get("kind", "chunk"),
                "start_line": int(c.get("start_line", 1)),
                "end_line": int(c.get("end_line", 1)),
            }
            for c in batch
        ]

        logger.info("Embedding batch %d/%d (%d chunks)...", i // batch_size + 1, (total + batch_size - 1) // batch_size, len(batch))
        embeddings = embed_texts(texts)

        # Dedup within batch (chunker may produce non-unique ids for some repos)
        seen = set()
        deduped = {"ids": [], "embeddings": [], "documents": [], "metadatas": []}
        for idx in range(len(ids)):
            if ids[idx] not in seen:
                seen.add(ids[idx])
                deduped["ids"].append(ids[idx])
                deduped["embeddings"].append(embeddings[idx].tolist())
                deduped["documents"].append(texts[idx])
                deduped["metadatas"].append(metadatas[idx])

        collection.upsert(
            ids=deduped["ids"],
            embeddings=deduped["embeddings"],
            documents=deduped["documents"],
            metadatas=deduped["metadatas"],
        )

    elapsed = time.perf_counter() - t0
    logger.info("Indexed %d chunks in %.1fs", total, elapsed)

    try:
        from src.rag.hybrid import clear_bm25_cache
        clear_bm25_cache(slug)
    except Exception:
        pass

    return len(chunks)


def retrieve(
    slug: str, query: str, k: int = 5,
) -> list[RetrievedChunk]:
    from src.rag.embedder import embed_query
    query_vec = embed_query(query)

    collection = _get_or_create_collection(slug, "code")
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return chunks

    for i, doc_id in enumerate(results["ids"][0]):
        score = 1.0 - results["distances"][0][i] if results["distances"] else 0.0

        meta = results["metadatas"][0][i]
        doc = results["documents"][0][i]

        chunks.append(RetrievedChunk(
            file=meta.get("file", ""),
            start_line=int(meta.get("start_line", 1)),
            end_line=int(meta.get("end_line", 1)),
            symbol=meta.get("symbol") or None,
            raw_code=doc,
            score=score,
            citation=f"{meta.get('file', '')}:{meta.get('start_line', 1)}-{meta.get('end_line', 1)}",
            language=meta.get("language", "text"),
            kind=meta.get("kind", "chunk"),
            chunk_id=doc_id,
        ))

    return chunks


def retrieve_with_context(
    slug: str, query: str, k: int = 5, expand: int = 2,
    use_hybrid: bool = True, use_rerank: bool = True,
) -> list[RetrievedChunk]:
    pool_size = get_settings().rerank_pool_size if use_rerank else k
    fetch_k = max(pool_size, k)

    if use_hybrid:
        from src.rag.hybrid import retrieve_hybrid
        settings = get_settings()
        results = retrieve_hybrid(
            slug, query, k=fetch_k,
            dense_weight=settings.hybrid_dense_weight,
            sparse_weight=settings.hybrid_sparse_weight,
        )
    else:
        results = retrieve(slug, query, k=fetch_k)

    if use_rerank and len(results) > 1:
        from src.rag.reranker import rerank
        results = rerank(query, results, top_k=k)
    else:
        results = results[:k]

    call_graph_path = get_settings().data_dir / slug / "call_graph.json"
    if not call_graph_path.exists():
        return results

    try:
        graph = json.loads(call_graph_path.read_text())
    except Exception:
        return results

    target_symbols = set()
    for r in results:
        if r.symbol:
            target_symbols.add(r.symbol)

    neighbor_ids = set()
    neighbors: dict[str, str] = {}
    for symbol in target_symbols:
        for edge in graph.get("edges", []):
            if edge["callee"] == symbol:
                if len([n for n in neighbors.values() if n == "caller"]) < expand:
                    neighbors[edge["caller"]] = "caller"
            if edge["caller"] == symbol:
                if len([n for n in neighbors.values() if n == "callee"]) < expand:
                    neighbors[edge["callee"]] = "callee"

    seen_ids = {r.chunk_id for r in results}

    if neighbors:
        try:
            collection = _get_or_create_collection(slug, "code")
            existing_all = collection.get(include=["metadatas", "documents"])
            existing_map = {}
            for eid, meta, doc in zip(existing_all["ids"], existing_all["metadatas"], existing_all["documents"]):
                existing_map[meta.get("symbol", "")] = (eid, meta, doc)

            for neighbor_key in list(neighbors)[:expand * 2]:
                simple_name = neighbor_key.split(":")[-1] if ":" in neighbor_key else neighbor_key
                if neighbor_key in existing_map:
                    lookup_key = neighbor_key
                elif simple_name in existing_map:
                    lookup_key = simple_name
                else:
                    continue
                eid, meta, doc = existing_map[lookup_key]
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    results.append(RetrievedChunk(
                        file=meta.get("file", ""),
                        start_line=int(meta.get("start_line", 1)),
                        end_line=int(meta.get("end_line", 1)),
                        symbol=meta.get("symbol") or None,
                        raw_code=doc,
                        score=0.0,
                        citation=f"{meta.get('file', '')}:{meta.get('start_line', 1)}-{meta.get('end_line', 1)}",
                        language=meta.get("language", "text"),
                        kind=meta.get("kind", "chunk"),
                        relation=neighbors.get(neighbor_key, "target"),
                        chunk_id=eid,
                    ))
                else:
                    for r in results:
                        if r.chunk_id == eid:
                            r.relation = neighbors.get(neighbor_key, r.relation)
        except Exception as e:
            logger.warning("Failed to expand context: %s", e)

    return results


def populate_qa_index(slug: str, qa_questions_path: Path | str) -> int:
    qa_path = Path(qa_questions_path)
    if not qa_path.exists():
        logger.warning("QA questions file not found: %s", qa_path)
        return 0

    questions = []
    with open(qa_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if not questions:
        return 0

    texts = [q["question"] for q in questions]
    embeddings = embed_texts(texts)

    collection = _get_or_create_collection(slug, "qa")

    ids = [q.get("id", f"qa__{q.get('chunk_id', 'unk')}__{i}") for i, q in enumerate(questions)]
    metadatas = [
        {
            "chunk_id": q.get("chunk_id", ""),
            "source": q.get("source", "synth"),
            "type": "question",
        }
        for q in questions
    ]

    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("Populated QA index with %d questions for %s", len(questions), slug)
    return len(questions)


def retrieve_with_qa(
    slug: str, query: str, k: int = 5, boost_factor: float = 1.5,
) -> list[RetrievedChunk]:
    code_results = retrieve(slug, query, k=k)

    try:
        qa_collection = _get_or_create_collection(slug, "qa")
        if qa_collection.count() == 0:
            return code_results
    except Exception:
        return code_results

    from src.rag.embedder import embed_query
    query_vec = embed_query(query)

    qa_results = qa_collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        include=["metadatas", "distances"],
    )

    if not qa_results["ids"] or not qa_results["ids"][0]:
        return code_results

    qa_chunk_ids = set()
    for meta_list in qa_results["metadatas"]:
        for meta in meta_list:
            cid = meta.get("chunk_id", "")
            if cid:
                qa_chunk_ids.add(cid)

    for r in code_results:
        if r.chunk_id in qa_chunk_ids:
            r.score = min(r.score * boost_factor, 1.0)

    code_results.sort(key=lambda x: -x.score)
    return code_results
