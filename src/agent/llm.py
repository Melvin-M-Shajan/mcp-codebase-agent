"""Chat models used by the agent (see README "Deviations from PRD" for why Groq
replaces the PRD's OpenAI/Anthropic default, why Gemini -- tried first -- was replaced
too, and why two different Groq models are used for two different roles here).

Two roles, two models:
- `get_planner_llms()` -- plan_node's tool-choosing decisions (bind_tools). Needs a model
  that reliably calls tools; `openai/gpt-oss-120b` does this well.
- `get_answer_llms()` -- answer_node's final-answer synthesis, and RAGAS's judge. Needs a
  model that reliably writes plain English instead of tool-call-shaped output.
  Every OpenAI `gpt-oss` model (120b and 20b both tested) fails this: given context
  describing prior tool calls (exactly what answer_node's prompt contains), it
  reflexively emits a fake tool-call payload -- sometimes even inventing tool names
  from its own pretraining (e.g. `repo_browser.open_file`) -- instead of an answer.
  Verified live, repeatedly, including after removing every function-call-shaped
  string from the prompt; it's a `gpt-oss`/"harmony" family trait, not fixable by
  prompting. `qwen/qwen3.8-27b` (with `reasoning_effort="low"`, which keeps its
  reasoning trace out of the final answer) does not have this failure mode, has a 131k
  context window, and -- as a different model family entirely -- draws from a separate
  daily quota than the planner, which matters given the budget pressure below.
  (`groq/compound-mini` and `allam-2-7b` were tried first: compound-mini avoided the
  tool-hallucination bug but turned out to have its own tight 250-requests/day cap,
  separate from and much stricter than any token-based limit; allam-2-7b avoided the
  bug too but its 4096-token context window is too small once a `read_file` result is
  in the prompt.)

Groq's free tier also caps `openai/gpt-oss-120b` at 200,000 tokens/day (a rolling
window, not a hard midnight reset) -- a real limit for a multi-step agent loop across a
20+ question eval; ~40k tokens/question observed even after trimming context in
`_summarize_context`, so a full run needs more than one key's daily budget. Each function
here returns a *list* of clients, one per `GROQ_API_KEY*` env var found (the user
supplied backup keys for exactly this); `_invoke_with_retry` in graph.py falls through
to the next client in the list on a daily-cap error instead of just failing.
"""

import os

from langchain_groq import ChatGroq

PLANNER_MODEL = "openai/gpt-oss-120b"
ANSWER_MODEL = "qwen/qwen3.8-27b"


def _groq_api_keys() -> list[str]:
    """GROQ_API_KEY, then GROQ_API_KEY_BACKUP, GROQ_API_KEY_BACKUP_2, ... -- as many as
    are set. Each is a separate Groq account/project with its own independent daily
    token budget, so more keys means more total headroom for a run."""
    keys = [os.environ["GROQ_API_KEY"]]
    backup = os.environ.get("GROQ_API_KEY_BACKUP")
    if backup:
        keys.append(backup)
    i = 2
    while True:
        extra = os.environ.get(f"GROQ_API_KEY_BACKUP_{i}")
        if not extra:
            break
        keys.append(extra)
        i += 1
    return keys


def get_planner_llms(temperature: float = 0.0) -> list[ChatGroq]:
    return [ChatGroq(model=PLANNER_MODEL, api_key=key, temperature=temperature) for key in _groq_api_keys()]


def get_answer_llms(temperature: float = 0.0) -> list[ChatGroq]:
    return [
        ChatGroq(model=ANSWER_MODEL, api_key=key, temperature=temperature, reasoning_effort="low")
        for key in _groq_api_keys()
    ]


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Used where only one client is needed and multi-key fallback isn't wired up
    (RAGAS's judge -- LangchainLLMWrapper sets attributes directly on the wrapped
    object, which breaks with a LangChain `.with_fallbacks()` Runnable since that isn't
    a BaseChatModel, so a single plain client is used instead). Picks the *last*
    available key rather than the first: by the time RAGAS scoring runs, the agent loop
    has already spent most of the earlier keys' daily budget (they're tried first in
    the fallback order), so the last key has the most headroom left."""
    return get_answer_llms(temperature)[-1]
