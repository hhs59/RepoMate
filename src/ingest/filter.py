from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pathspec

from src.config import (
    KEEP_EXTENSIONS,
    DROP_DIRS,
    BINARY_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILES,
)
from src.logging import get_logger

logger = get_logger(__name__)


def _is_binary(ext: str) -> bool:
    return ext.lower() in BINARY_EXTENSIONS


def _load_gitignore_specs(repo_root: Path) -> list[pathspec.PathSpec]:
    specs = []
    root_ignore = repo_root / ".gitignore"
    if root_ignore.exists():
        specs.append(pathspec.PathSpec.from_lines(
            "gitwildmatch", root_ignore.read_text().splitlines()
        ))
    for gitignore_path in repo_root.rglob(".gitignore"):
        if gitignore_path.parent != repo_root:
            specs.append(pathspec.PathSpec.from_lines(
                "gitwildmatch", gitignore_path.read_text().splitlines()
            ))
    return specs


def _should_skip(path: Path, specs: list[pathspec.PathSpec]) -> bool:
    for part in path.parts:
        if part in DROP_DIRS or part.startswith(".") and part not in (".github",):
            return True
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    for spec in specs:
        if spec.match_file(str(path)):
            return True
    return False


def iter_code_files(repo_root: Path) -> Iterator[Path]:
    repo_root = Path(repo_root)
    specs = _load_gitignore_specs(repo_root)
    count = 0

    for file_path in sorted(repo_root.rglob("*")):
        if not file_path.is_file():
            continue
        if count >= MAX_FILES:
            logger.warning("Reached max file cap (%d), truncating", MAX_FILES)
            break

        rel = file_path.relative_to(repo_root)

        if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
            logger.debug("Skipping large file: %s", rel)
            continue

        if file_path.suffix.lower() not in KEEP_EXTENSIONS:
            continue

        if _should_skip(rel, specs):
            continue

        count += 1
        yield file_path

    logger.info("Filtered %d code files from %s", count, repo_root.name)
