from __future__ import annotations

import hashlib
import json
import re
import string
import subprocess
from pathlib import Path

import tiktoken

from ciyc.config import (
    KEEP_EXTENSIONS,
    DIFF_MAX_TOKENS,
    MAX_COMMITS,
    get_settings,
)
from ciyc.logging import get_logger

logger = get_logger(__name__)

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_STOPLIST: dict[str, bool] = {
    s: True for s in (
        "wip", "fix", "fixes", "fixed", "fixup", "update", "updated", "updates",
        "tmp", "temp", "test", "tests", "asdf", "lol", "no message", "commit",
        "changes", "stuff", "misc", "cleanup", "clean up", "revert", ".",
        "..", "...", "ok", "done", "final", "v1", "bump", "release", "merge",
    )
}

_VERSION_BUMP_RE = re.compile(r"^v?\d+\.\d+(\.\d+)?(-[\w.]+)?$")
_BUMP_VERSION_RE = re.compile(r"^bump\s+version", re.IGNORECASE)


def is_good_commit_message(msg: str) -> bool:
    msg = msg.strip()
    if len(msg) <= 3:
        return False
    lower = msg.lower()

    if lower in _STOPLIST:
        return False
    if _VERSION_BUMP_RE.match(lower):
        return False
    if _BUMP_VERSION_RE.match(lower):
        return False

    if re.fullmatch(r"[\s\d" + re.escape(string.punctuation) + r"]*", msg):
        return False

    for stopword in _STOPLIST:
        if len(stopword) >= 2 and re.search(r"\b" + re.escape(stopword) + r"\b", lower):
            return False

    alpha_tokens = [t for t in re.findall(r"[a-zA-Z]+", lower) if len(t) >= 4]
    if not alpha_tokens:
        return False

    return True


def _diff_has_code(diff_text: str) -> bool:
    for line in diff_text.splitlines():
        if not line.startswith("diff --git"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            path = parts[3].lstrip("b/")
            ext = Path(path).suffix.lower()
            if ext in KEEP_EXTENSIONS:
                return True
    return False


def _truncate_diff(diff_text: str, max_tokens: int = DIFF_MAX_TOKENS) -> str:
    tokens = _TOKENIZER.encode(diff_text)
    if len(tokens) <= max_tokens:
        return diff_text
    truncated = _TOKENIZER.decode(tokens[:max_tokens])
    return truncated + "\n... (truncated)"


def _diff_hash(diff_text: str) -> str:
    return hashlib.sha256(diff_text.encode()).hexdigest()[:16]


def extract_git_pairs(repo_path: Path, slug: str) -> tuple[list[dict], list[dict], int]:
    repo_path = Path(repo_path)
    settings = get_settings()

    cmd = ["git", "log", "--format=%H%x09%s%x09%b", "--no-merges", f"-n{MAX_COMMITS}"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_path))
    if result.returncode != 0:
        logger.warning("git log failed: %s", result.stderr)
        return [], [], 0

    clean_pairs = []
    relabel_pairs = []
    seen_diffs: set[str] = set()
    discarded = 0
    commits_data = result.stdout.strip().splitlines()

    for line in commits_data:
        if not line.strip():
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        commit_hash = parts[0]
        message = parts[1] if len(parts) > 1 else ""
        body = parts[2] if len(parts) > 2 else ""
        full_msg = f"{message}\n{body}".strip()

        diff_result = subprocess.run(
            ["git", "show", commit_hash, "--no-color"],
            capture_output=True, text=True, cwd=str(repo_path),
        )
        if diff_result.returncode != 0:
            continue

        diff_text = diff_result.stdout
        if not diff_text.strip() or not _diff_has_code(diff_text):
            discarded += 1
            continue

        dh = _diff_hash(diff_text)
        if dh in seen_diffs:
            discarded += 1
            continue
        seen_diffs.add(dh)

        truncated = _truncate_diff(diff_text)

        if is_good_commit_message(full_msg):
            clean_pairs.append({
                "instruction": full_msg,
                "response": truncated,
                "commit_hash": commit_hash,
                "source": "git-clean",
            })
        else:
            relabel_pairs.append({
                "commit_hash": commit_hash,
                "message": full_msg,
                "diff": truncated,
            })

    return clean_pairs, relabel_pairs, discarded


def write_git_pairs(slug: str, clean: list[dict], relabel: list[dict]) -> None:
    settings = get_settings()
    clean_path = settings.data_dir / slug / "git_pairs_clean.jsonl"
    relabel_path = settings.data_dir / slug / "commits_to_relabel.jsonl"

    with open(clean_path, "w") as f:
        for pair in clean:
            f.write(json.dumps(pair) + "\n")
    logger.info("Wrote %d clean git pairs to %s", len(clean), clean_path)

    with open(relabel_path, "w") as f:
        for pair in relabel:
            f.write(json.dumps(pair) + "\n")
    logger.info("Wrote %d commits to relabel to %s", len(relabel), relabel_path)
