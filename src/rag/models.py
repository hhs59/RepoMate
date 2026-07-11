from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievedChunk:
    file: str
    start_line: int
    end_line: int
    symbol: Optional[str]
    raw_code: str
    score: float
    citation: str
    language: str = "text"
    kind: str = "chunk"
    relation: str = "target"
    chunk_id: str = ""
