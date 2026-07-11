from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.data.llm_client import complete_json
from src.logging import get_logger

logger = get_logger(__name__)


def generate_impact_pairs(
    chunks: list[dict],
    call_graph: dict,
    output_path: Path,
    max_functions: int | None = None,
) -> list[dict]:
    chunk_by_symbol: dict[str, dict] = {}
    for c in chunks:
        if c.get("symbol"):
            chunk_by_symbol[c["symbol"]] = c

    callers_map: dict[str, list[str]] = {}
    for edge in call_graph.get("edges", []):
        callers_map.setdefault(edge["callee"], []).append(edge["caller"])

    targets = [(sym, callers) for sym, callers in callers_map.items() if len(callers) >= 1]
    if max_functions:
        targets = targets[:max_functions]

    pairs: list[dict] = []
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    pairs.append(p)
                    done_ids.add(p.get("chunk_id", ""))

    with open(output_path, "a") as f:
        for symbol, caller_keys in targets:
            if symbol in done_ids:
                continue

            chunk = chunk_by_symbol.get(symbol)
            file = chunk["file"] if chunk else "?"
            signature = chunk.get("signature") if chunk else "unknown"
            callers_str = ", ".join(caller_keys[:10])

            messages = [
                {"role": "user", "content": (
                    f"Function `{symbol}` ({file}) is called by: {callers_str}.\n"
                    f"Its current signature is: {signature}.\n\n"
                    f"Question: If I change {symbol}'s signature, what might break?\n"
                    f"Answer by listing the likely affected callers and why, "
                    f"using ONLY the names shown.\n"
                    f'JSON: {{"answer": "..."}}'
                )},
            ]

            result = complete_json(messages, max_tokens=512)
            if not result or "answer" not in result:
                continue

            answer = result["answer"].strip()
            if len(answer) < 20:
                continue

            question = f"If I change {symbol}'s signature, what might break?"
            pair = {
                "instruction": question,
                "response": answer,
                "source": "impact",
                "chunk_id": chunk["id"] if chunk else symbol,
            }
            pairs.append(pair)
            f.write(json.dumps(pair) + "\n")
            f.flush()

    logger.info("Generated %d impact pairs", len(pairs))
    return pairs
