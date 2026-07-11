from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.data.llm_client import complete_json
from src.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You are a senior engineer creating a training dataset for a codebase "
    "Q&A assistant. Use ONLY the code shown. Do not invent APIs/symbols not "
    "present. Cite real function/variable/class names from the code."
)


def _build_prompt(chunk: dict, n: int) -> list[dict]:
    kind = chunk.get("kind", "function")
    symbol = chunk.get("symbol") or "this code"
    file = chunk.get("file", "?")
    lang = chunk.get("language", "text")
    code = chunk["raw_code"]
    start = chunk.get("start_line", "?")
    end = chunk.get("end_line", "?")

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (
            f"Below is a {kind} `{symbol}` from {file} (lines {start}-{end}), "
            f"written in {lang}:\n```{lang}\n{code}\n```\n"
            f"Write {n} specific questions a developer would ask about this code, "
            f"then answer each precisely using only the code shown. "
            f"Format STRICTLY as JSON:\n"
            f'{{"pairs":[{{"question":"...","answer":"..."}}]}}'
        )},
    ]


def _validate_pair(q: str, a: str) -> bool:
    if not q or not a:
        return False
    if len(q) < 8 or len(a) < 20:
        return False
    lower_a = a.lower()
    if "i don't" in lower_a or "cannot" in lower_a or "i can't" in lower_a:
        return False
    return True


def generate_synth_pairs(
    chunks: list[dict],
    output_path: Path,
    n_per_chunk: int = 3,
    max_chunks: int | None = None,
) -> list[dict]:
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    done_ids.add(json.loads(line).get("chunk_id", ""))
        logger.info("Resuming: %d chunks already done", len(done_ids))

    pairs: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))

    chunks_to_process = chunks
    if max_chunks:
        chunks_to_process = chunks[:max_chunks]

    with open(output_path, "a") as f:
        for i, chunk in enumerate(chunks_to_process):
            cid = chunk["id"]
            if cid in done_ids:
                continue

            result = complete_json(_build_prompt(chunk, n_per_chunk))
            if not result or "pairs" not in result:
                logger.warning("No pairs for chunk %s", cid)
                continue

            for p in result["pairs"]:
                q = p.get("question", "")
                a = p.get("answer", "")
                if _validate_pair(q, a):
                    pair = {
                        "instruction": q,
                        "response": a,
                        "source": "synth",
                        "chunk_id": cid,
                    }
                    pairs.append(pair)
                    f.write(json.dumps(pair) + "\n")
                    f.flush()

            if (i + 1) % 50 == 0:
                logger.info("Processed %d/%d chunks", i + 1, len(chunks_to_process))

    logger.info("Generated %d synth pairs total", len(pairs))
    return pairs
