from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.data.llm_client import complete_json
from src.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You create a training dataset for a codebase Q&A assistant. "
    "Use ONLY the code shown; cite real symbol names. Never invent APIs."
)


def _build_prompt(chunk: dict, callers: list[dict], callees: list[dict]) -> list[dict]:
    symbol = chunk.get("symbol") or "this function"
    file = chunk.get("file", "?")
    lang = chunk.get("language", "text")
    code = chunk["raw_code"]

    callers_str = "\n\n".join(
        f"  {c.get('symbol', '?')} ({c.get('file', '?')}):\n  ```{lang}\n  {c['raw_code'][:500]}\n  ```"
        for c in callers
    ) or "  (no callers found)"
    callees_str = "\n\n".join(
        f"  {c.get('symbol', '?')} ({c.get('file', '?')}):\n  ```{lang}\n  {c['raw_code'][:500]}\n  ```"
        for c in callees
    ) or "  (no callees found)"

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"Function `{symbol}` (from {file}):\n```{lang}\n{code}\n```\n\n"
            f"It is CALLED BY:\n{callers_str}\n\n"
            f"It CALLS:\n{callees_str}\n\n"
            f"Write 3 questions about HOW this function relates to its callers/callees "
            f"(e.g. \"How does a caller use {symbol}?\", \"What does {symbol} depend on?\", "
            f"\"Why might a caller break if {symbol} changes?\"). "
            f"Answer each from the code. "
            f'JSON: {{"pairs":[{{"question":"...","answer":"..."}}]}}'
        )},
    ]


def generate_contextual_pairs(
    chunks: list[dict],
    call_graph: dict,
    output_path: Path,
    max_chunks: int | None = None,
) -> list[dict]:
    chunk_by_symbol: dict[str, dict] = {}
    for c in chunks:
        if c.get("symbol"):
            chunk_by_symbol[c["symbol"]] = c

    callers_map: dict[str, list[str]] = {}
    callees_map: dict[str, list[str]] = {}
    for edge in call_graph.get("edges", []):
        callers_map.setdefault(edge["callee"], []).append(edge["caller"])
        callees_map.setdefault(edge["caller"], []).append(edge["callee"])

    pairs: list[dict] = []
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    pairs.append(p)
                    done_ids.add(p.get("chunk_id", ""))

    chunks_to_process = chunks
    if max_chunks:
        chunks_to_process = chunks[:max_chunks]

    with open(output_path, "a") as f:
        for i, chunk in enumerate(chunks_to_process):
            cid = chunk["id"]
            if cid in done_ids:
                continue

            symbol = chunk.get("symbol")
            if not symbol:
                continue

            caller_keys = callers_map.get(symbol, [])
            callee_keys = callees_map.get(symbol, [])
            if not caller_keys and not callee_keys:
                continue

            caller_chunks = []
            for key in caller_keys[:3]:
                simple = key.split(":")[-1] if ":" in key else key
                c = chunk_by_symbol.get(simple)
                if c:
                    caller_chunks.append(c)

            callee_chunks = []
            for key in callee_keys[:3]:
                simple = key.split(":")[-1] if ":" in key else key
                c = chunk_by_symbol.get(simple)
                if c:
                    callee_chunks.append(c)

            if not caller_chunks and not callee_chunks:
                continue

            result = complete_json(_build_prompt(chunk, caller_chunks, callee_chunks))
            if not result or "pairs" not in result:
                continue

            for p in result["pairs"]:
                q = p.get("question", "")
                a = p.get("answer", "")
                if len(q) >= 8 and len(a) >= 20:
                    pair = {
                        "instruction": q,
                        "response": a,
                        "source": "contextual",
                        "chunk_id": cid,
                    }
                    pairs.append(pair)
                    f.write(json.dumps(pair) + "\n")
                    f.flush()

            if (i + 1) % 50 == 0:
                logger.info("Contextual: processed %d/%d chunks", i + 1, len(chunks_to_process))

    logger.info("Generated %d contextual pairs", len(pairs))
    return pairs
