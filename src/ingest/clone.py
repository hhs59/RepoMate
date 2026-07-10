from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from src.config import get_settings
from src.logging import get_logger

logger = get_logger(__name__)


def derive_slug(repo_url: str) -> str:
    match = re.search(r"([^/:]+)/([^/]+?)(?:\.git)?$", repo_url.rstrip("/"))
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return repo_url.rsplit("/", 1)[-1].replace(".git", "")


def clone_repo(repo_url: str, dest_dir: Path, force: bool = False) -> Path:
    dest_dir = Path(dest_dir)
    if dest_dir.exists() and any(dest_dir.iterdir()):
        if not force:
            logger.info("Repo already cloned at %s (use --force to re-clone)", dest_dir)
            return dest_dir
        logger.info("Force re-cloning %s", dest_dir)
        import shutil
        shutil.rmtree(dest_dir)

    dest_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", repo_url, str(dest_dir)]
    logger.info("Cloning %s -> %s", repo_url, dest_dir)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Clone failed: {result.stderr.strip()}")
    return dest_dir


def get_head_sha(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=str(repo_path),
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def get_remote_url(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True, text=True, cwd=str(repo_path),
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def write_meta(slug: str, repo_path: Path, stats: dict | None = None) -> None:
    settings = get_settings()
    meta_path = settings.data_dir / slug / "meta.json"
    meta = {
        "slug": slug,
        "head_sha": get_head_sha(repo_path),
        "remote_url": get_remote_url(repo_path),
        "cloned_at": datetime.now(timezone.utc).isoformat(),
    }
    if stats:
        meta["stats"] = stats
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Wrote meta to %s", meta_path)
