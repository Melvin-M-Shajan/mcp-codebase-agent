"""Gemini chat model used by plan_node and answer_node (see README "Deviations from
PRD" for why Gemini replaces the PRD's OpenAI/Anthropic default)."""

import os

from langchain_google_genai import ChatGoogleGenerativeAI

CHAT_MODEL = "gemini-2.5-flash"


def get_llm(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=temperature,
    )
