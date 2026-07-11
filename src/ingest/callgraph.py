from __future__ import annotations

import json
from pathlib import Path

import tree_sitter_languages

from src.config import get_settings
from src.ingest.models import CallGraph, GraphNode, GraphEdge
from src.ingest.chunker import _get_parser, _get_node_name, _SYMBOL_NODE_TYPES, _LANGUAGE_MAP
from src.logging import get_logger

logger = get_logger(__name__)

_CALL_QUERY_BY_LANG: dict[str, str] = {
    "python": """
        (call function: (identifier) @call_name)
        (call function: (attribute object: (_) attribute: (identifier) @call_name))
        (import_statement (dotted_name) @import_name)
    """,
    "javascript": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (member_expression property: (property_identifier) @call_name))
        (import_statement (import_clause (named_imports (import_specifier name: (identifier) @import_name))))
        (import_statement (import_clause (identifier) @import_name))
    """,
    "typescript": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (member_expression property: (property_identifier) @call_name))
        (import_statement (import_clause (named_imports (import_specifier name: (identifier) @import_name))))
        (import_statement (import_clause (identifier) @import_name))
    """,
    "tsx": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (member_expression property: (property_identifier) @call_name))
        (import_statement (import_clause (named_imports (import_specifier name: (identifier) @import_name))))
        (import_statement (import_clause (identifier) @import_name))
    """,
    "go": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (selector_expression field: (field_identifier) @call_name))
        (import_spec path: (_) @import_name)
    """,
    "rust": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (scoped_identifier path: (identifier) @call_name))
        (call_expression function: (field_expression field: (identifier) @call_name))
        (use_declaration argument: (_) @import_name)
    """,
    "java": """
        (method_invocation name: (identifier) @call_name)
        (method_invocation object: (_) name: (identifier) @call_method)
        (import_declaration (scoped_identifier) @import_name)
    """,
    "c": """
        (call_expression function: (identifier) @call_name)
        (field_expression field: (field_identifier) @call_method)
    """,
    "cpp": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (field_expression field: (field_identifier) @call_method))
        (call_expression function: (template_function) @call_template)
    """,
    "c_sharp": """
        (invocation_expression function: (identifier) @call_name)
        (invocation_expression function: (member_access_expression name: (identifier) @call_method))
    """,
    "ruby": """
        (call method: (identifier) @call_name)
        (call receiver: (_) method: (identifier) @call_method)
    """,
    "php": """
        (function_call_expression function: (name) @call_name)
        (method_call_expression method: (name) @call_method)
    """,
    "swift": """
        (call_expression function: (simple_identifier) @call_name)
        (call_expression function: (navigation_expression suffix: (simple_identifier) @call_method))
    """,
    "scala": """
        (call_expression function: (identifier) @call_name)
        (call_expression function: (field_expression field: (identifier) @call_method))
    """,
    "kotlin": """
        (call_expression (simple_identifier) @call_name)
        (call_expression (navigation_expression name: (simple_identifier) @call_method))
    """,
}


def _extract_definitions(node, src: bytes, file_path: str, nodes: list[GraphNode]):
    if node.type in _SYMBOL_NODE_TYPES:
        name = _get_node_name(node)
        if name:
            signature = None
            if node.type in ("function_definition", "method_definition", "function_declaration"):
                body = node.child_by_field_name("body")
                if body:
                    signature = src[node.start_byte:body.start_byte].decode().strip().rstrip(":")
            elif node.type == "decorated_definition":
                inner = node.child_by_field_name("definition")
                if inner:
                    inner_name = _get_node_name(inner)
                    if inner_name:
                        name = inner_name
                        body = inner.child_by_field_name("body")
                        if body:
                            signature = src[inner.start_byte:body.start_byte].decode().strip().rstrip(":")

            kind = node.type.replace("_definition", "").replace("_declaration", "")
            if node.type == "decorated_definition":
                kind = "function"

            nodes.append(GraphNode(
                name=name,
                file=file_path,
                line=node.start_point[0] + 1,
                kind=kind,
                signature=signature,
            ))

    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        if inner and inner.type in _SYMBOL_NODE_TYPES:
            return

    for child in node.children:
        _extract_definitions(child, src, file_path, nodes)


def _extract_call_sites(
    node, src: bytes, file_path: str, def_names: set[str], edges: list[GraphEdge],
):
    call_names: list[str] = []
    if node.type in ("call", "call_expression", "method_invocation",
                     "invocation_expression", "function_call_expression"):
        func = node.child_by_field_name("function")
        if func:
            call_names.extend(_collect_names(func, src))

    for name in call_names:
        clean = name.strip()
        if clean and clean in def_names:
            edges.append(GraphEdge(caller=f"{file_path}:{clean}", callee=clean))

    for child in node.children:
        _extract_call_sites(child, src, file_path, def_names, edges)


def _collect_names(node, src: bytes) -> list[str]:
    names = []
    if node.type == "identifier":
        names.append(node.text.decode())
    elif node.type == "attribute":
        name = node.child_by_field_name("attribute")
        if name:
            names.append(name.text.decode())
    elif node.type == "member_expression":
        prop = node.child_by_field_name("property")
        if prop:
            names.append(prop.text.decode())
    elif node.type == "navigation_expression":
        n = node.child_by_field_name("name")
        if n:
            names.append(n.text.decode())
    elif node.type == "selector_expression":
        field = node.child_by_field_name("field")
        if field:
            names.append(field.text.decode())
    elif node.type == "field_expression":
        field = node.child_by_field_name("field")
        if field:
            names.append(field.text.decode())
    for child in node.children:
        names.extend(_collect_names(child, src))
    return names


def extract_call_graph(slug: str, repo_path: Path) -> CallGraph:
    repo_path = Path(repo_path)
    graph = CallGraph()
    file_count = 0

    from src.ingest.filter import iter_code_files

    for file_path in iter_code_files(repo_path):
        lang = _LANGUAGE_MAP.get(file_path.suffix.lower())
        if lang is None:
            continue

        parser = _get_parser(lang)
        if parser is None:
            continue

        try:
            src = file_path.read_bytes()
            tree = parser.parse(src)
        except Exception:
            continue

        rel_path = str(file_path.relative_to(repo_path))

        _extract_definitions(tree.root_node, src, rel_path, graph.nodes)
        file_count += 1

    logger.info("Found %d symbol definitions across %d files", len(graph.nodes), file_count)

    def_names = {n.name for n in graph.nodes}

    for file_path in iter_code_files(repo_path):
        lang = _LANGUAGE_MAP.get(file_path.suffix.lower())
        if lang is None:
            continue

        parser = _get_parser(lang)
        if parser is None:
            continue

        try:
            src = file_path.read_bytes()
            tree = parser.parse(src)
        except Exception:
            continue

        rel_path = str(file_path.relative_to(repo_path))
        _extract_call_sites(tree.root_node, src, rel_path, def_names, graph.edges)

    logger.info("Found %d call edges", len(graph.edges))

    return graph


def write_call_graph(slug: str, graph: CallGraph) -> Path:
    settings = get_settings()
    output_path = settings.data_dir / slug / "call_graph.json"
    data = {
        "nodes": [{"name": n.name, "file": n.file, "line": n.line,
                    "kind": n.kind, "signature": n.signature} for n in graph.nodes],
        "edges": [{"caller": e.caller, "callee": e.callee} for e in graph.edges],
    }
    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote call graph (%d nodes, %d edges) to %s",
                 len(graph.nodes), len(graph.edges), output_path)
    return output_path
