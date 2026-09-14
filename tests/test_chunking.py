"""§14: chunk boundaries land on function/class edges, and the file-path header is
present in every chunk."""

from src.rag.chunking import chunk_file, chunk_markdown_file, chunk_python_file

_FIXTURE_SOURCE = '''"""Module docstring."""

import os


def top_level_function(x):
    return x + 1


class Widget:
    """A widget."""

    def __init__(self, name):
        self.name = name

    def render(self):
        return f"<widget {self.name}>"


@some_decorator
def decorated_function():
    return 42
'''


class TestChunkPythonFile:
    def test_every_chunk_has_file_header(self):
        chunks = chunk_python_file("pkg/widget.py", _FIXTURE_SOURCE)
        assert chunks
        for chunk in chunks:
            assert chunk.text.startswith(f"# file: pkg/widget.py lines: {chunk.start_line}-{chunk.end_line}\n")

    def test_boundaries_land_on_def_and_class(self):
        chunks = chunk_python_file("pkg/widget.py", _FIXTURE_SOURCE)
        func_chunk = next(c for c in chunks if "top_level_function" in c.text)
        body_after_header = func_chunk.text.split("\n", 1)[1]
        assert body_after_header.startswith("def top_level_function(x):")
        assert body_after_header.strip().endswith("return x + 1")

        # The decorated function's chunk must include the decorator line, not start mid-def.
        decorated = next(c for c in chunks if "decorated_function" in c.text)
        assert "@some_decorator" in decorated.text

    def test_small_class_kept_as_one_chunk(self):
        chunks = chunk_python_file("pkg/widget.py", _FIXTURE_SOURCE)
        widget_chunks = [c for c in chunks if "class Widget" in c.text]
        assert len(widget_chunks) == 1
        assert "def render" in widget_chunks[0].text

    def test_large_class_splits_into_methods(self):
        methods = "\n".join(
            f"    def method_{i}(self):\n" + "        x = 1\n" * 40 + "        return x\n"
            for i in range(10)
        )
        source = f"class Big:\n{methods}"
        chunks = chunk_python_file("pkg/big.py", source)
        assert len(chunks) == 10
        for chunk in chunks:
            assert chunk.text.startswith("# file: pkg/big.py lines:")
            assert "class Big:" in chunk.text

    def test_chunk_ids_are_deterministic(self):
        chunks_a = chunk_python_file("pkg/widget.py", _FIXTURE_SOURCE)
        chunks_b = chunk_python_file("pkg/widget.py", _FIXTURE_SOURCE)
        assert [c.vector_id() for c in chunks_a] == [c.vector_id() for c in chunks_b]


class TestChunkMarkdownFile:
    def test_splits_by_heading_and_has_header(self):
        md = "# Title\n\nIntro text.\n\n## Section A\n\nBody A.\n\n## Section B\n\nBody B.\n"
        chunks = chunk_markdown_file("README.md", md)
        assert len(chunks) == 3
        for chunk in chunks:
            assert chunk.text.startswith(f"# file: README.md lines: {chunk.start_line}-{chunk.end_line}\n")
            assert chunk.chunk_type == "doc"
        assert "Section A" in chunks[1].text
        assert "Section B" in chunks[2].text


class TestChunkFileDispatch:
    def test_dispatches_python(self):
        chunks = chunk_file("a.py", "def f():\n    return 1\n")
        assert chunks and chunks[0].chunk_type == "code"

    def test_dispatches_markdown(self):
        chunks = chunk_file("a.md", "# H\n\nbody\n")
        assert chunks and chunks[0].chunk_type == "doc"

    def test_unknown_extension_returns_empty(self):
        assert chunk_file("a.bin", "whatever") == []
