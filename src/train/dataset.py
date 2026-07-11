from __future__ import annotations

import json
from pathlib import Path

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


def load_train_eval(slug: str) -> tuple[list[dict], list[dict]]:
    settings = get_settings()
    data_dir = settings.data_dir / slug

    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"No train.jsonl for '{slug}'. Run `repomate train` data generation first."
        )

    train_data = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                train_data.append(json.loads(line))

    eval_data = []
    if eval_path.exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    eval_data.append(json.loads(line))

    logger.info("Loaded %d train, %d eval examples for %s", len(train_data), len(eval_data), slug)
    return train_data, eval_data


def to_dataset(train_data: list[dict], eval_data: list[dict]):
    from datasets import Dataset

    train_ds = Dataset.from_list(train_data)
    eval_ds = Dataset.from_list(eval_data) if eval_data else None
    return train_ds, eval_ds


def load_raw_pairs(slug: str) -> tuple[list[dict], list[dict]]:
    settings = get_settings()
    data_dir = settings.data_dir / slug

    train_path = data_dir / "train.jsonl"
    eval_path = data_dir / "eval.jsonl"

    train_raw, eval_raw = [], []

    if train_path.exists():
        with open(train_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    instr, resp = _extract_turns(text)
                    if instr and resp:
                        train_raw.append({"instruction": instr, "response": resp, "text": text})

    if eval_path.exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    text = entry.get("text", "")
                    instr, resp = _extract_turns(text)
                    if instr and resp:
                        eval_raw.append({"instruction": instr, "response": resp, "text": text})

    return train_raw, eval_raw


def _extract_turns(text: str) -> tuple[str, str]:
    import re

    user_match = re.search(r"<start_of_turn>user\n(.*?)<end_of_turn>", text, re.DOTALL)
    model_match = re.search(r"<start_of_turn>model\n(.*?)<end_of_turn>", text, re.DOTALL)

    instruction = user_match.group(1).strip() if user_match else ""
    response = model_match.group(1).strip() if model_match else ""
    return instruction, response
