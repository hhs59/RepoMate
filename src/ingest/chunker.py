from __future__ import annotations

import json
from pathlib import Path

import tiktoken
import tree_sitter_languages

from src.config import get_settings
from src.ingest.models import CodeChunk
from src.logging import get_logger

logger = get_logger(__name__)

_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".java": "java",
    ".kt": "kotlin",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".hxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".vue": "vue",
    ".svelte": "svelte",
    ".sql": "sql",
    ".css": "css",
    ".html": "html",
}

_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_SYMBOL_NODE_TYPES = frozenset({
    "function_definition", "function_declaration", "method_definition",
    "class_definition", "class_declaration", "decorated_definition",
    "constructor_declaration", "arrow_function",
})

_PARSER_CACHE: dict[str, object] = {}


def _get_parser(language: str) -> object | None:
    if language not in _PARSER_CACHE:
        try:
            _PARSER_CACHE[language] = tree_sitter_languages.get_parser(language)
        except Exception:
            _PARSER_CACHE[language] = None
    return _PARSER_CACHE[language]


def _language_for_file(file_path: Path) -> str | None:
    return _LANGUAGE_MAP.get(file_path.suffix.lower())


def _get_node_name(node) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode()
    decl = node.child_by_field_name("declarator")
    if decl:
        name_node = decl.child_by_field_name("name")
        if name_node:
            return name_node.text.decode()
    return None


def _get_node_text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode()


def _get_preceding_docstring(node, src: bytes) -> str | None:
    prev = node.prev_sibling
    if prev and prev.type in ("expression_statement", "comment"):
        text = _get_node_text(prev, src)
        if text.strip().startswith(("'''", '"""', "//", "/*", "*", "#")):
            return text.strip()
    return None


def _get_signature(node, src: bytes, language: str) -> str | None:
    if language == "python":
        def_node = node
        if node.type == "decorated_definition":
            def_node = node.child_by_field_name("definition")
        if def_node is None:
            return None
        body = def_node.child_by_field_name("body")
        if body:
            sig_end = body.start_byte
            return src[def_node.start_byte:sig_end].decode().strip().rstrip(":")
    return None


def _extract_symbol_nodes(node, src: bytes, language: str):
    results = []
    if node.type in _SYMBOL_NODE_TYPES:
        results.append(node)
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner and inner.type in _SYMBOL_NODE_TYPES:
            results.append(node)
            return results
    for child in node.children:
        results.extend(_extract_symbol_nodes(child, src, language))
    return results


def _build_qualified_name(node, file_path: Path, language: str) -> str | None:
    name = _get_node_name(node)
    if not name:
        return None
    parent = node.parent
    while parent:
        if parent.type in ("class_definition", "class_declaration"):
            parent_name = _get_node_name(parent)
            if parent_name:
                name = f"{parent_name}.{name}"
        parent = parent.parent
    return name


def chunk_file(
    file_path: Path, repo_root: Path, slug: str, sha8: str, idx_offset: int = 0
) -> list[CodeChunk]:
    rel_path = str(file_path.relative_to(repo_root))
    language = _language_for_file(file_path)
    chunks: list[CodeChunk] = []

    if language is None:
        return _chunk_sliding_window(file_path, rel_path, language or "text", slug, sha8, idx_offset)

    parser = _get_parser(language)
    if parser is None:
        return _chunk_sliding_window(file_path, rel_path, language, slug, sha8, idx_offset)

    try:
        src = file_path.read_bytes()
        tree = parser.parse(src)
        symbol_nodes = _extract_symbol_nodes(tree.root_node, src, language)
    except Exception as e:
        logger.warning("Parse error in %s: %s", rel_path, e)
        return _chunk_sliding_window(file_path, rel_path, language, slug, sha8, idx_offset)

    if not symbol_nodes:
        return _chunk_sliding_window(file_path, rel_path, language, slug, sha8, idx_offset)

    for i, node in enumerate(symbol_nodes):
        name = _get_node_name(node)
        if not name:
            continue
        node_type = node.type
        kind = node_type.replace("_definition", "").replace("_declaration", "")
        if node_type == "decorated_definition":
            kind = "function"
        symbol = _build_qualified_name(node, file_path, language)
        raw_code = _get_node_text(node, src)
        docstring = _get_preceding_docstring(node, src)
        signature = _get_signature(node, src, language)

        chunk = CodeChunk(
            id=f"{slug}__{sha8}__{idx_offset + i}",
            file=rel_path,
            language=language,
            symbol=symbol,
            kind=kind,
            start_line=node.start_point[0] + 1,
            end_line=node.end_point[0] + 1,
            raw_code=raw_code,
            docstring=docstring,
            signature=signature,
        )
        chunks.append(chunk)

    return chunks


def _chunk_sliding_window(
    file_path: Path, rel_path: str, language: str,
    slug: str, sha8: str, idx_offset: int,
) -> list[CodeChunk]:
    settings = get_settings()
    try:
        text = file_path.read_text()
    except Exception:
        return []

    tokens = _TOKENIZER.encode(text)
    chunk_size = settings.chunk_max_tokens
    overlap = settings.chunk_overlap
    chunks: list[CodeChunk] = []

    if len(tokens) <= chunk_size:
        decoded = _TOKENIZER.decode(tokens)
        return [CodeChunk(
            id=f"{slug}__{sha8}__{idx_offset}",
            file=rel_path, language=language, symbol=None, kind="chunk",
            start_line=1, end_line=text.count("\n") + 1, raw_code=decoded,
        )]

    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start:start + chunk_size]
        decoded = _TOKENIZER.decode(window)
        chunks.append(CodeChunk(
            id=f"{slug}__{sha8}__{idx_offset + i}",
            file=rel_path, language=language, symbol=None, kind="chunk",
            start_line=1, end_line=1, raw_code=decoded,
        ))
        if start + chunk_size >= len(tokens):
            break

    return chunks


def chunk_all(repo_root: Path, slug: str, sha8: str) -> tuple[list[CodeChunk], dict]:
    from src.ingest.filter import iter_code_files

    all_chunks = []
    stats_by_lang: dict[str, int] = {}
    file_count = 0
    idx = 0

    for file_path in iter_code_files(repo_root):
        chunks = chunk_file(file_path, repo_root, slug, sha8, idx)
        for c in chunks:
            stats_by_lang[c.language] = stats_by_lang.get(c.language, 0) + 1
        idx += len(chunks)
        all_chunks.extend(chunks)
        file_count += 1

    stats = {
        "total_files": file_count,
        "total_chunks": len(all_chunks),
        "by_language": stats_by_lang,
    }
    return all_chunks, stats


def write_chunks(slug: str, chunks: list[CodeChunk]) -> Path:
    from src.config import get_settings
    settings = get_settings()
    chunks_path = settings.data_dir / slug / "chunks.jsonl"
    with open(chunks_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps({
                "id": chunk.id,
                "file": chunk.file,
                "language": chunk.language,
                "symbol": chunk.symbol,
                "kind": chunk.kind,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "raw_code": chunk.raw_code,
                "docstring": chunk.docstring,
                "signature": chunk.signature,
            }) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), chunks_path)
    return chunks_path
