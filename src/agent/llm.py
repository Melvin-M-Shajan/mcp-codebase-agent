"""Chat models used by the agent (see README "Deviations from PRD" for why Groq
replaces the PRD's OpenAI/Anthropic default, why Gemini -- tried first -- was replaced
too, and why two different Groq models are used for two different roles here).

Two roles, two models:
- `get_planner_llms()` -- plan_node's tool-choosing decisions (bind_tools). Needs a model
  that reliably calls tools; `openai/gpt-oss-120b` does this well.
- `get_answer_llms()` -- answer_node's final-answer synthesis, and RAGAS's judge. Needs a
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
20+ question eval; ~40k tokens/question observed even after trimming context in
`_summarize_context`, so a full run needs more than one key's daily budget. Each function
here returns a *list* of clients, one per `GROQ_API_KEY*` env var found (the user
supplied a backup key for exactly this); `_invoke_with_retry` in graph.py falls through
to the next client in the list on a daily-cap error instead of just failing.
"""

import os

from langchain_groq import ChatGroq

PLANNER_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "groq/compound-mini"


def _groq_api_keys() -> list[str]:
    keys = [os.environ["GROQ_API_KEY"]]
    backup = os.environ.get("GROQ_API_KEY_BACKUP")
    if backup:
        keys.append(backup)
    return keys


def _get_groq_llms(model: str, temperature: float) -> list[ChatGroq]:
    return [ChatGroq(model=model, api_key=key, temperature=temperature) for key in _groq_api_keys()]


def get_planner_llms(temperature: float = 0.0) -> list[ChatGroq]:
    return _get_groq_llms(PLANNER_MODEL, temperature)


def get_answer_llms(temperature: float = 0.0) -> list[ChatGroq]:
    return _get_groq_llms(ANSWER_MODEL, temperature)


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Used where only one general-purpose model/key is needed (RAGAS's judge)."""
    return get_answer_llms(temperature)[0]
