"""§8.2 embedding model -- `sentence-transformers/all-MiniLM-L6-v2`, the PRD's explicit
open-source fallback ("if avoiding API cost"), run locally with no API key or rate limit.
384 dimensions (see README "Deviations from PRD" for why this replaced Gemini
embeddings specifically -- httpx's ~800 non-test chunks alone exceed Gemini's free-tier
daily embedding quota, and that quota is shared with every later re-ingestion and every
agent search_code call).
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_client: HuggingFaceEmbeddings | None = None


def get_embeddings_client() -> HuggingFaceEmbeddings:
    global _client
    if _client is None:
        _client = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _client
