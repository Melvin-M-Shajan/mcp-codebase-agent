"""§8.1 chunking -- function/class-level chunking (tree-sitter for Python, heading-level
for Markdown, regex fallback for anything else), not fixed-size windows.

Every chunk's `text` has a `# file: <path> lines: <start>-<end>` header prepended before
embedding, so the model sees provenance in-context even without metadata lookup.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tree_sitter import Language, Node, Parser
import tree_sitter_python as tspython

MAX_CHUNK_TOKENS = 1500
_PY_LANGUAGE = Language(tspython.language())


def _estimate_tokens(text: str) -> int:
    # No tokenizer dependency pulled in for this -- 1 token ~= 4 chars is the standard
    # rule-of-thumb estimate, good enough for a chunk-size guard.
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    text: str  # header + source, ready to embed
    chunk_type: Literal["code", "doc"]

    def vector_id(self) -> str:
        import hashlib

        raw = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _make_chunk(file_path: str, start_line: int, end_line: int, body: str, chunk_type: str) -> Chunk:
    header = f"# file: {file_path} lines: {start_line}-{end_line}\n"
    return Chunk(
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        text=header + body,
        chunk_type=chunk_type,  # type: ignore[arg-type]
    )


def _node_line_span(node: Node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _top_level_definitions(root: Node) -> list[Node]:
    defs = []
    for child in root.children:
        node = child
        if node.type == "decorated_definition":
            inner = node.child_by_field_name("definition")
            if inner is not None and inner.type in ("function_definition", "class_definition"):
                defs.append(node)
        elif node.type in ("function_definition", "class_definition"):
            defs.append(node)
    return defs


def _class_methods(class_node: Node) -> list[Node]:
    body = class_node.child_by_field_name("body")
    if body is None:
        return []
    return _top_level_definitions(body)


def chunk_python_file(file_path: str, source: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    parser = Parser(_PY_LANGUAGE)
    tree = parser.parse(source.encode("utf-8"))
    src_bytes = source.encode("utf-8")

    def node_text(node: Node) -> str:
        return src_bytes[node.start_byte : node.end_byte].decode("utf-8")

    chunks: list[Chunk] = []
    for def_node in _top_level_definitions(tree.root_node):
        start, end = _node_line_span(def_node)
        text = node_text(def_node)

        actual_def = def_node
        if def_node.type == "decorated_definition":
            actual_def = def_node.child_by_field_name("definition")

        if _estimate_tokens(text) <= max_tokens or actual_def.type != "class_definition":
            chunks.append(_make_chunk(file_path, start, end, text, "code"))
            continue

        # Class too large as one chunk -- split into per-method chunks instead.
        class_name_node = actual_def.child_by_field_name("name")
        class_name = node_text(class_name_node) if class_name_node else "?"
        methods = _class_methods(actual_def)
        if not methods:
            chunks.append(_make_chunk(file_path, start, end, text, "code"))
            continue
        for method_node in methods:
            m_start, m_end = _node_line_span(method_node)
            m_text = f"class {class_name}:\n" + node_text(method_node)
            chunks.append(_make_chunk(file_path, m_start, m_end, m_text, "code"))

    if not chunks and source.strip():
        # No top-level def/class found (e.g. a script or __init__.py of constants) --
        # fall back to chunking the whole file as one unit.
        chunks.append(_make_chunk(file_path, 1, len(source.splitlines()) or 1, source, "code"))

    return chunks


_GENERIC_DEF_RE = re.compile(
    r"^\s*(def |class |function |const \w+\s*=\s*(\(|\{|function|async)|export (function|class|const)\b)",
)


def chunk_generic_source_file(file_path: str, source: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    """Regex fallback for languages tree-sitter isn't wired up for here: split on
    top-level def/class/function boundaries."""
    lines = source.splitlines()
    if not lines:
        return []

    boundaries = [i for i, line in enumerate(lines) if _GENERIC_DEF_RE.match(line)]
    if not boundaries:
        return [_make_chunk(file_path, 1, len(lines), source, "code")]

    chunks: list[Chunk] = []
    for idx, start_idx in enumerate(boundaries):
        end_idx = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start_idx:end_idx])
        chunks.append(_make_chunk(file_path, start_idx + 1, end_idx, body, "code"))
    return chunks


def chunk_markdown_file(file_path: str, source: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    lines = source.splitlines()
    if not lines:
        return []

    heading_re = re.compile(r"^#{1,6}\s+")
    boundaries = [i for i, line in enumerate(lines) if heading_re.match(line)]
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + boundaries

    chunks: list[Chunk] = []
    for idx, start_idx in enumerate(boundaries):
        end_idx = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start_idx:end_idx]).strip("\n")
        if not body.strip():
            continue
        if _estimate_tokens(body) <= max_tokens:
            chunks.append(_make_chunk(file_path, start_idx + 1, end_idx, body, "doc"))
            continue
        # Section too big even for one heading -- split further by paragraph.
        para_start = start_idx
        para_lines: list[str] = []
        cursor = start_idx
        for line_no in range(start_idx, end_idx):
            para_lines.append(lines[line_no])
            if lines[line_no].strip() == "" and para_lines:
                chunk_body = "\n".join(para_lines).strip("\n")
                if chunk_body.strip():
                    chunks.append(_make_chunk(file_path, para_start + 1, line_no + 1, chunk_body, "doc"))
                para_lines = []
                para_start = line_no + 1
            cursor = line_no
        if para_lines:
            chunk_body = "\n".join(para_lines).strip("\n")
            if chunk_body.strip():
                chunks.append(_make_chunk(file_path, para_start + 1, cursor + 1, chunk_body, "doc"))
    return chunks


def chunk_file(file_path: str, source: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list[Chunk]:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return chunk_python_file(file_path, source, max_tokens)
    if suffix == ".md":
        return chunk_markdown_file(file_path, source, max_tokens)
    if suffix in (".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h"):
        return chunk_generic_source_file(file_path, source, max_tokens)
    return []
