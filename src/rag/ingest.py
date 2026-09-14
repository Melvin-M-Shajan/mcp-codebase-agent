"""§8.4 ingestion -- walks repo_root, chunks per chunking.py, embeds per embeddings.py,
upserts to Pinecone per pinecone_client.py. Idempotent: vector IDs are a deterministic
hash of file_path+start_line+end_line, so re-running on the same repo/commit upserts the
same IDs instead of duplicating.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from src.rag.chunking import Chunk, chunk_file
from src.rag.embeddings import get_embeddings_client
from src.rag.pinecone_client import INDEX_NAME, ensure_index, get_pinecone_client

INDEXABLE_SUFFIXES = {
    ".py", ".md", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".rb", ".rs", ".c", ".cpp", ".h",
}
SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
# Test files are deliberately excluded from RAG indexing: get_recent_commits/run_tests/
# read_file already cover test-related questions directly, and skipping them keeps
# ingestion comfortably under Gemini's free-tier daily embedding quota (see README
# "Deviations from PRD").
TEST_DIR_NAMES = {"tests", "test", "__tests__"}

EMBED_BATCH_SIZE = 25
UPSERT_BATCH_SIZE = 100
MAX_RETRIES = 6
# Gemini's free tier caps gemini-embedding-2 at 100 embed requests/minute (each text in
# a batch counts as one request). Pacing batches below that avoids ever hitting the cap.
INTER_BATCH_SLEEP_SECONDS = 20
RATE_LIMIT_BACKOFF_SECONDS = 65


def _tracked_files(repo_root: Path) -> list[Path]:
    """`git ls-files` gives us every tracked, non-gitignored file for free -- no need
    to hand-roll .gitignore parsing."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [repo_root / line for line in result.stdout.splitlines() if line.strip()]


def _iter_chunks(repo_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in _tracked_files(repo_root):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if any(part in TEST_DIR_NAMES for part in path.relative_to(repo_root).parts[:-1]):
            continue
        if path.suffix.lower() not in INDEXABLE_SUFFIXES:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary or unreadable -- skip
        rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
        chunks.extend(chunk_file(rel_path, source))
    return chunks


def _with_retry(fn, *args, **kwargs):
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # Gemini/Pinecone rate limits, transient network errors
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RATE_LIMIT_BACKOFF_SECONDS if "RESOURCE_EXHAUSTED" in str(exc) else delay
            print(f"  retrying after error ({attempt + 1}/{MAX_RETRIES}), sleeping {wait}s: {exc}", file=sys.stderr)
            time.sleep(wait)
            delay *= 2


def ingest_repo(repo_root: str, index_name: str, namespace: str) -> int:
    root = Path(repo_root).resolve()
    chunks = _iter_chunks(root)
    if not chunks:
        return 0

    pc = get_pinecone_client()
    index = ensure_index(pc, index_name)
    embedder = get_embeddings_client(task_type="RETRIEVAL_DOCUMENT")

    total_upserted = 0
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]
        texts = [c.text for c in batch]
        vectors = _with_retry(embedder.embed_documents, texts)

        pinecone_vectors = [
            {
                "id": chunk.vector_id(),
                "values": vector,
                "metadata": {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text[:4000],  # keep metadata payload bounded
                },
            }
            for chunk, vector in zip(batch, vectors)
        ]

        for up_start in range(0, len(pinecone_vectors), UPSERT_BATCH_SIZE):
            up_batch = pinecone_vectors[up_start : up_start + UPSERT_BATCH_SIZE]
            _with_retry(index.upsert, vectors=up_batch, namespace=namespace)
            total_upserted += len(up_batch)

        print(f"  ingested {min(batch_start + EMBED_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks", file=sys.stderr)
        if batch_start + EMBED_BATCH_SIZE < len(chunks):
            time.sleep(INTER_BATCH_SLEEP_SECONDS)

    return total_upserted


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--index-name", default=INDEX_NAME)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()

    count = ingest_repo(args.repo_root, args.index_name, args.namespace)
    print(f"Ingested {count} chunks into index={args.index_name!r} namespace={args.namespace!r}")


if __name__ == "__main__":
    main()
