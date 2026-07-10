from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodeChunk:
    id: str
    file: str
    language: str
    symbol: Optional[str]
    kind: str
    start_line: int
    end_line: int
    raw_code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class GraphNode:
    name: str
    file: str
    line: int
    kind: str
    signature: Optional[str] = None


@dataclass
class GraphEdge:
    caller: str
    callee: str


@dataclass
class CallGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
