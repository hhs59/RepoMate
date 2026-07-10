from __future__ import annotations

import json
import subprocess
from pathlib import Path

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


def extract_gh_pairs(repo_path: Path, slug: str) -> list[dict]:
    settings = get_settings()
    repo_path = Path(repo_path)

    gh_bin = subprocess.run(["which", "gh"], capture_output=True, text=True)
    if gh_bin.returncode != 0:
        logger.info("gh CLI not found, skipping PR extraction")
        return []

    pr_result = subprocess.run(
        ["gh", "pr", "list", "--state", "all", "--json", "title,number,body", "--limit", "100"],
        capture_output=True, text=True, cwd=str(repo_path),
    )
    if pr_result.returncode != 0:
        logger.info("gh PR list failed (probably not authenticated or not a GitHub repo)")
        return []

    try:
        prs = json.loads(pr_result.stdout)
    except json.JSONDecodeError:
        logger.warning("Failed to parse gh PR output")
        return []

    pairs = []
    for pr in prs:
        title = pr.get("title", "")
        body = pr.get("body", "")
        if not title:
            continue
        pairs.append({
            "instruction": title,
            "response": body or f"PR #{pr.get('number', '?')}",
            "source": "github-pr",
        })

    if pairs:
        output_path = settings.data_dir / slug / "gh_pairs.jsonl"
        with open(output_path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logger.info("Extracted %d GitHub PR pairs", len(pairs))

    return pairs
