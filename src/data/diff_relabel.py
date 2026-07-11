from __future__ import annotations

import json
from pathlib import Path

from src.data.llm_client import complete
from src.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = "You write clear, concise commit messages from code diffs."


def generate_relabel_pairs(
    relabel_path: Path,
    output_path: Path,
    max_commits: int | None = None,
) -> list[dict]:
    if not relabel_path.exists():
        logger.info("No commits_to_relabel.jsonl found, skipping diff-relabel")
        return []

    commits = []
    with open(relabel_path) as f:
        for line in f:
            if line.strip():
                commits.append(json.loads(line))

    if max_commits:
        commits = commits[:max_commits]

    pairs: list[dict] = []
    done_hashes: set[str] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    pairs.append(p)
                    done_hashes.add(p.get("commit_hash", ""))

    with open(output_path, "a") as f:
        for i, commit in enumerate(commits):
            ch = commit.get("commit_hash", "")
            if ch in done_hashes:
                continue

            diff = commit["diff"][:1500]

            messages = [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": (
                    f"Below is a code diff. Write ONE clear sentence describing "
                    f"what this change does, in a developer's voice, as if explaining "
                    f"it to a teammate. Be specific about what was added/changed/fixed. "
                    f"Do not mention line numbers.\n```diff\n{diff}\n```\n"
                    f"Return ONLY the sentence."
                )},
            ]

            sentence = complete(messages, max_tokens=128, temperature=0.2).strip()
            if len(sentence) < 10:
                continue

            pair = {
                "instruction": sentence,
                "response": diff,
                "source": "diff-relabel",
                "commit_hash": ch,
            }
            pairs.append(pair)
            f.write(json.dumps(pair) + "\n")
            f.flush()

            if (i + 1) % 50 == 0:
                logger.info("Relabel: processed %d/%d commits", i + 1, len(commits))

    logger.info("Generated %d relabel pairs", len(pairs))
    return pairs
