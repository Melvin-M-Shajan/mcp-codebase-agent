"""Chat models used by the agent (see README "Deviations from PRD" for why Groq
replaces the PRD's OpenAI/Anthropic default, why Gemini -- tried first -- was replaced
too, and why two different Groq models are used for two different roles here).

Two roles, two models:
- `get_planner_llm()` -- plan_node's tool-choosing decisions (bind_tools). Needs a model
  that reliably calls tools; `openai/gpt-oss-120b` does this well.
- `get_answer_llm()` -- answer_node's final-answer synthesis, and RAGAS's judge. Needs a
  model that reliably writes plain English instead of tool-call-shaped output.
  `openai/gpt-oss-120b` does NOT: given context describing prior tool calls (exactly
  what answer_node's prompt contains), it reflexively emits a fake tool-call payload
  (sometimes even inventing tool names from its own pretraining, e.g. `repo_browser.
  open_file`, that were never offered) instead of an answer -- verified live, repeatedly,
  including after removing every function-call-shaped string from the prompt.
  `groq/compound-mini` does not have this failure mode and has a large (128k) context
  window -- `allam-2-7b` was tried first and also avoided the failure mode, but its 4096-
  token context window is too small once a `read_file` result is in the prompt.

Groq's free tier also caps `openai/gpt-oss-120b` at 200,000 tokens/day (a rolling
window, not a hard midnight reset) -- a real limit for a multi-step agent loop across a
20+ question eval. `_summarize_context` in graph.py truncates snippet/result previews
sent to the LLM (they're resent on every loop iteration) to stay well under it, and
`_invoke_with_retry` fails fast on a daily-cap error instead of burning more of a budget
that won't refill soon (it still retries per-minute rate limits, which are transient).
"""

import os

from langchain_groq import ChatGroq

PLANNER_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "groq/compound-mini"


def _get_groq_llm(model: str, temperature: float) -> ChatGroq:
    return ChatGroq(model=model, api_key=os.environ["GROQ_API_KEY"], temperature=temperature)


def get_planner_llm(temperature: float = 0.0) -> ChatGroq:
    return _get_groq_llm(PLANNER_MODEL, temperature)


def get_answer_llm(temperature: float = 0.0) -> ChatGroq:
    return _get_groq_llm(ANSWER_MODEL, temperature)


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Used where only one general-purpose model is needed (RAGAS's judge)."""
    return get_answer_llm(temperature)
