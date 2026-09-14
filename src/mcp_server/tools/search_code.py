"""§6.1 search_code -- semantic search against the Pinecone index built by src/rag/ingest.py."""

from pydantic import BaseModel

from src.rag.embeddings import get_embeddings_client
from src.rag.pinecone_client import INDEX_NAME, ensure_index, get_pinecone_client


class CodeMatch(BaseModel):
    file_path: str
    start_line: int
    end_line: int
    snippet: str
    score: float


def search_code(namespace: str, query: str, top_k: int = 8) -> list[CodeMatch]:
    top_k = min(max(top_k, 1), 20)

    embedder = get_embeddings_client(task_type="RETRIEVAL_QUERY")
    query_vector = embedder.embed_query(query)

    pc = get_pinecone_client()
    index = ensure_index(pc, INDEX_NAME)
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )

    matches: list[CodeMatch] = []
    for match in response.matches:
        metadata = match.metadata or {}
        matches.append(
            CodeMatch(
                file_path=metadata.get("file_path", ""),
                start_line=int(metadata.get("start_line", 0)),
                end_line=int(metadata.get("end_line", 0)),
                snippet=metadata.get("text", ""),
                score=float(match.score),
            )
        )
    return matches
