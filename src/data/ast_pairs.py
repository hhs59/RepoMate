from __future__ import annotations

import json
from pathlib import Path

from src.logging import get_logger

logger = get_logger(__name__)


def generate_ast_pairs(chunks: list[dict], call_graph: dict, output_path: Path) -> list[dict]:
    pairs: list[dict] = []

    symbol_to_chunks: dict[str, dict] = {}
    for c in chunks:
        if c.get("symbol"):
            symbol_to_chunks[c["symbol"]] = c

    callers_map: dict[str, list[str]] = {}
    callees_map: dict[str, list[str]] = {}
    for edge in call_graph.get("edges", []):
        callers_map.setdefault(edge["callee"], []).append(edge["caller"])
        callees_map.setdefault(edge["caller"], []).append(edge["callee"])

    for chunk in chunks:
        symbol = chunk.get("symbol")
        if not symbol:
            continue

        docstring = chunk.get("docstring")
        if docstring and len(docstring) > 10:
            clean_doc = docstring.strip().strip('"').strip("'")
            pairs.append({
                "instruction": f"What does {symbol} do?",
                "response": f"{clean_doc}\n\n```{chunk.get('language', 'text')}\n{chunk['raw_code'][:500]}\n```",
                "source": "ast",
                "chunk_id": chunk["id"],
            })
            pairs.append({
                "instruction": f'Which {chunk.get("kind", "function")} has the docstring: "{clean_doc[:80]}"?',
                "response": f"{symbol} in {chunk['file']}:{chunk.get('start_line', '?')}",
                "source": "ast",
                "chunk_id": chunk["id"],
            })

        signature = chunk.get("signature")
        if signature and len(signature) > 5:
            pairs.append({
                "instruction": f"What are the parameters of {symbol}?",
                "response": f"The signature of {symbol} is: {signature}",
                "source": "ast",
                "chunk_id": chunk["id"],
            })

        pairs.append({
            "instruction": f"Where is {symbol} defined?",
            "response": f"{symbol} is defined in {chunk['file']}:{chunk.get('start_line', '?')}",
            "source": "ast",
            "chunk_id": chunk["id"],
        })

        callers = callers_map.get(symbol, [])
        if callers:
            caller_list = ", ".join(callers[:10])
            pairs.append({
                "instruction": f"Where is {symbol} called?",
                "response": f"{symbol} is called by: {caller_list}",
                "source": "ast",
                "chunk_id": chunk["id"],
            })

        callees = callees_map.get(symbol, [])
        if callees:
            callee_list = ", ".join(callees[:10])
            pairs.append({
                "instruction": f"What does {symbol} call?",
                "response": f"{symbol} calls: {callee_list}",
                "source": "ast",
                "chunk_id": chunk["id"],
            })

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info("Generated %d AST pairs", len(pairs))
    return pairs
