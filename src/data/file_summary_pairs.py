from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.data.llm_client import complete_json
from src.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = "You summarize code files for a codebase Q&A assistant. Use ONLY the symbols shown."


def _group_by_file(chunks: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for c in chunks:
        groups.setdefault(c["file"], []).append(c)
    return groups


def _build_symbol_list(file_chunks: list[dict]) -> str:
    lines = []
    for c in file_chunks:
        symbol = c.get("symbol") or "(anonymous)"
        kind = c.get("kind", "chunk")
        sig = c.get("signature") or ""
        lines.append(f"  - {kind} {symbol}" + (f" — {sig}" if sig else ""))
    return "\n".join(lines[:30])


def generate_file_summary_pairs(
    chunks: list[dict],
    output_path: Path,
    max_files: int | None = None,
) -> list[dict]:
    by_file = _group_by_file(chunks)
    files = list(by_file.items())
    if max_files:
        files = files[:max_files]

    pairs: list[dict] = []
    done_files: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    pairs.append(p)
                    done_files.add(p.get("file", ""))

    with open(output_path, "a") as f:
        for file_path, file_chunks in files:
            if file_path in done_files:
                continue

            lang = file_chunks[0].get("language", "text")
            symbol_list = _build_symbol_list(file_chunks)
            if len(symbol_list) < 10:
                continue

            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (
                    f"File `{file_path}` ({lang}) contains these symbols:\n"
                    f"{symbol_list}\n\n"
                    f"Write a 2-sentence summary of what this file/module is responsible for.\n"
                    f'Return JSON: {{"summary": "..."}}'
                )},
            ]

            result = complete_json(messages, max_tokens=256)
            if not result or "summary" not in result:
                continue

            summary = result["summary"].strip()
            if len(summary) < 20:
                continue

            for q_template in ["What does {file} do?", "What is {file} responsible for?"]:
                pair = {
                    "instruction": q_template.format(file=file_path),
                    "response": summary,
                    "source": "file-summary",
                    "file": file_path,
                }
                pairs.append(pair)
                f.write(json.dumps(pair) + "\n")
                f.flush()

    logger.info("Generated %d file-summary pairs", len(pairs))
    return pairs
