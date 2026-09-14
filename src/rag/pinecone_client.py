"""§8.3 vector store -- one Pinecone index, one namespace per indexed repo."""

import os

from pinecone import Index, Pinecone, ServerlessSpec

from src.rag.embeddings import EMBEDDING_DIM

INDEX_NAME = "mcp-codebase-agent"


def get_pinecone_client() -> Pinecone:
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def ensure_index(pc: Pinecone, index_name: str = INDEX_NAME, dimension: int = EMBEDDING_DIM) -> Index:
    existing = {idx["name"] for idx in pc.list_indexes()}
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index(index_name)
