"""§8.2 embedding model -- Gemini `gemini-embedding-2` in place of the PRD's
`text-embedding-3-small` default (see README "Deviations from PRD"). 1536 dimensions
kept to match the PRD's own OpenAI-dimension precedent.
"""

import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

EMBEDDING_MODEL = "models/gemini-embedding-2"
EMBEDDING_DIM = 1536


def get_embeddings_client(task_type: str) -> GoogleGenerativeAIEmbeddings:
    """`task_type` is "RETRIEVAL_DOCUMENT" for chunks being indexed, "RETRIEVAL_QUERY"
    for a search query -- Gemini embeddings are asymmetric, using the matching task type
    on each side measurably improves retrieval quality."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        output_dimensionality=EMBEDDING_DIM,
        task_type=task_type,
    )
