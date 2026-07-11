from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)

_CHAT_SYS = "You are repomate, an assistant that answers about this codebase."

_DIFF_MAX_TOKENS = 2048


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _pair_hash(instruction: str, response: str) -> str:
    return hashlib.sha256(
        (_normalize(instruction) + "\n" + _normalize(response)).encode()
    ).hexdigest()[:16]


def _truncate_response(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"


def _format_chat(instruction: str, response: str) -> str:
    return (
        f"<start_of_turn>user\n{_CHAT_SYS}\n{instruction}<end_of_turn>\n"
        f"<start_of_turn>model\n{response}<end_of_turn>"
    )


def _load_pairs_file(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pairs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def _load_git_clean(slug: str) -> list[dict]:
    settings = get_settings()
    clean_path = settings.data_dir / slug / "git_pairs_clean.jsonl"
    pairs = _load_pairs_file(clean_path)
    for p in pairs:
        p["response"] = _truncate_response(p["response"])
        p["source"] = "git-clean"
        p["chunk_id"] = p.get("commit_hash", "")
    return pairs


def assemble(slug: str) -> dict:
    settings = get_settings()
    data_dir = settings.data_dir / slug

    all_pairs: list[dict] = []

    source_files = {
        "synth": data_dir / "synth_pairs.jsonl",
        "contextual": data_dir / "contextual_pairs.jsonl",
        "file-summary": data_dir / "file_summary_pairs.jsonl",
        "impact": data_dir / "impact_pairs.jsonl",
        "diff-relabel": data_dir / "relabel_pairs.jsonl",
        "ast": data_dir / "ast_pairs.jsonl",
    }

    for name, path in source_files.items():
        pairs = _load_pairs_file(path)
        logger.info("Loaded %d pairs from %s", len(pairs), name)
        all_pairs.extend(pairs)

    git_clean = _load_git_clean(slug)
    logger.info("Loaded %d git-clean pairs", len(git_clean))
    all_pairs.extend(git_clean)

    seen_hashes: set[str] = set()
    deduped: list[dict] = []
    for p in all_pairs:
        h = _pair_hash(p["instruction"], p["response"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        if len(p["instruction"]) < 8 or len(p["response"]) < 20:
            continue

        p["response"] = _truncate_response(p["response"])
        deduped.append(p)

    logger.info("After dedup + filter: %d pairs (from %d)", len(deduped), len(all_pairs))

    source_counts: dict[str, int] = {}
    for p in deduped:
        src = p.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    random.seed(42)
    random.shuffle(deduped)

    eval_count = max(1, len(deduped) // 20)
    eval_pairs = deduped[:eval_count]
    train_pairs = deduped[eval_count:]

    train_path = data_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for p in train_pairs:
            text = _format_chat(p["instruction"], p["response"])
            f.write(json.dumps({"text": text}) + "\n")

    eval_path = data_dir / "eval.jsonl"
    with open(eval_path, "w") as f:
        for p in eval_pairs:
            text = _format_chat(p["instruction"], p["response"])
            f.write(json.dumps({"text": text}) + "\n")

    qa_questions: list[dict] = []
    for p in train_pairs:
        if p.get("chunk_id") and p.get("source") in ("synth", "contextual", "ast"):
            qa_questions.append({
                "question": p["instruction"],
                "chunk_id": p["chunk_id"],
                "source": p["source"],
            })
    qa_path = data_dir / "qa_questions.jsonl"
    with open(qa_path, "w") as f:
        for q in qa_questions:
            f.write(json.dumps(q) + "\n")

    avg_instr_len = sum(len(p["instruction"]) for p in train_pairs) / max(1, len(train_pairs))
    avg_resp_len = sum(len(p["response"]) for p in train_pairs) / max(1, len(train_pairs))

    stats = {
        "total_pairs": len(deduped),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "qa_questions": len(qa_questions),
        "by_source": source_counts,
        "avg_instruction_len": round(avg_instr_len, 1),
        "avg_response_len": round(avg_resp_len, 1),
    }

    stats_path = data_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Dataset stats: %s", json.dumps(stats, indent=2))

    return stats
