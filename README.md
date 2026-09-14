# MCP Codebase Agent

A read-only LangGraph agent that investigates a real open-source codebase through an MCP
server -- searching, reading files, running tests, and checking commit history -- and
answers developer questions with file/line citations, grounded by RAG over the repo's
code and docs. Built from [`projects/mcp-codebase-agent/PRD.md`](../PRD.md) in the parent
job-search repo.

## Target Repo

- **Repo:** [encode/httpx](https://github.com/encode/httpx)
- **Commit indexed:** `b5addb64f0161ff6bfe94c124ef76f6a1fba5254`
- **Why this repo (§5):** ~8.8k LOC in `httpx/` + ~8.9k LOC of tests -- comfortably inside
  the 5k-40k LOC range, not a monorepo. Real, actively maintained, standard `pytest`
  suite, and a git history with substantive commits (retries, auth, HTTP/2, proxies) that
  supports real multi-step "why did X change" questions. Pure Python, so `tree-sitter`
  chunking (§8.1) applies cleanly. The builder did not have httpx's internals memorized
  going in, so correct multi-step answers are real evidence of the agent investigating,
  not the builder already knowing the answer.
- **Test command:** `pytest -q` (run from the repo root; `run_tests`'s `target` argument
  restricts to a path/pattern, e.g. `tests/test_auth.py`).

## Architecture

```
questions from chat.py / run_eval.py
        |
        v
  LangGraph agent (src/agent/graph.py)
    retrieve_node --(search_code via RAG)--> plan_node <-> tool_node
        |                                        |
        v                                        v
   Pinecone (src/rag)              MCP client (stdio) -> MCP server (src/mcp_server)
                                                            |
                                                       target-repo/ (read-only)
```

- **MCP server** (`src/mcp_server/server.py`): 4 read-only tools over stdio --
  `search_code`, `read_file`, `run_tests`, `get_recent_commits`. No write/delete tool
  exists, so the agent cannot modify the target repo even if prompted to.
- **RAG** (`src/rag/`): tree-sitter-based function/class chunking for Python,
  heading-based chunking for Markdown, embedded locally with
  `sentence-transformers/all-MiniLM-L6-v2`, stored in Pinecone (namespace = repo name).
- **Agent** (`src/agent/`): `retrieve_node -> plan_node -> (tool_node -> plan_node)* ->
  answer_node`, capped at `max_steps=6`.
- **Eval** (`src/eval/run_eval.py`): runs the real agent against `eval/questions.jsonl`,
  scores with RAGAS, traces every run with Langfuse.

## Setup

```bash
cp .env.example .env   # fill in GROQ_API_KEY, PINECONE_API_KEY, LANGFUSE_*
pip install -r requirements.txt
git clone https://github.com/encode/httpx.git target-repo
git -C target-repo checkout b5addb64f0161ff6bfe94c124ef76f6a1fba5254

python -m src.rag.ingest --repo-root ./target-repo --index-name mcp-codebase-agent --namespace httpx
python -m src.demo.chat
```

`run_tests` must only ever execute inside the Docker container (`docker build -t
mcp-codebase-agent . && docker run --env-file .env mcp-codebase-agent`), never on the
host -- it shells out to the target repo's own `pytest` command, which is arbitrary code
from a third party. See §16 / "Deviations" below for how this is enforced locally.

## Running the eval suite

```bash
docker run --env-file .env mcp-codebase-agent python -m src.eval.run_eval \
  --questions eval/questions.jsonl --out eval/EVAL_REPORT.md
```

## Deviations from the PRD

- **LLM: Groq (`openai/gpt-oss-120b`) instead of OpenAI/Anthropic.** The PRD's primary
  default is OpenAI or Anthropic. Gemini was tried first (a working key was available),
  but `gemini-2.5-flash`'s free tier caps out at **20 generateContent requests/day** --
  discovered mid-build when a single demo question exhausted enough of that budget to
  start failing. That's far too low for an agent loop (2+ LLM calls per question) plus a
  24-question eval plus RAGAS's own per-sample judge calls. Groq's free tier has no such
  wall for this volume, so `plan_node`/`answer_node`/RAGAS's judge model now run on
  `openai/gpt-oss-120b` via `langchain-groq`.
- **`plan_node` uses native tool-calling instead of a custom structured-output schema.**
  `openai/gpt-oss-120b` has strong agentic post-training: the moment a prompt describes
  tools like `search_code` as available, it tries to genuinely call one, regardless of
  which `with_structured_output` method (`function_calling`, `json_schema`) was used for
  a separate custom "decision" schema -- Groq's API then rejects the call since that
  tool isn't in the actual bound-tools list. `plan_node` (`src/agent/graph.py`) instead
  binds the 4 real MCP tools plus one pseudo-tool, `finish_investigation`, directly via
  `llm.bind_tools(...)`, and reads `response.tool_calls[0]` -- working with the model's
  instinct instead of fighting it. This changes only plan_node's internal LLM-calling
  mechanism; the state schema, node signatures, and routing logic in §7 are unchanged.
- **Two different models for two different roles (`src/agent/llm.py`).** The same
  `openai/gpt-oss-120b` instinct that makes it good at `plan_node` makes it unreliable
  for `answer_node`: given a prompt describing prior tool calls (exactly what
  answer_node's context contains), it would reflexively emit a fake tool-call payload --
  sometimes even inventing tool names from its own pretraining that were never offered
  (e.g. `repo_browser.open_file`) -- instead of writing an answer, both as outright API
  errors ("Tool choice is none, but model called a tool") and, worse, as silently useless
  JSON-shaped "answers" that don't error at all. This reproduced reliably even after
  rewording the context to avoid any `func(args)`-shaped text and adding an explicit
  "never output JSON" instruction. `get_planner_llm()` (gpt-oss-120b) is used for
  `plan_node`'s tool-choosing; `get_answer_llm()` (`groq/compound-mini`, 128k context, no
  such failure mode observed) is used for `answer_node`'s final-answer synthesis and as
  RAGAS's judge model -- `allam-2-7b` was tried first and also avoided the failure mode,
  but its 4096-token context window turned out too small once a `read_file` result
  landed in the prompt (`context_length_exceeded`). `build_graph(session, planner_llm,
  answer_llm)` takes both models explicitly.
- **Embeddings: local `sentence-transformers/all-MiniLM-L6-v2` instead of an API
  model.** §8.2 explicitly offers this as the fallback "if avoiding API cost" -- it
  became necessary, not just preferred: `gemini-embedding-2`'s free tier caps out at
  1000 embed requests/day/model, and a first ingestion attempt exhausted that quota
  partway through (900/1426 chunks embedded) before any eval or demo questions had even
  run, since every `search_code` call also spends one embed request. Switching to a
  local model (`src/rag/embeddings.py`, 384 dimensions, no API key, no rate limit)
  removes this bottleneck entirely rather than working around it. The Pinecone index was
  deleted and recreated at 384 dimensions to match. (Groq has no embeddings endpoint, so
  this couldn't be solved by the Groq switch above -- it needed its own fix.)
- **RAG index excludes `tests/`.** §8.4 says "walks repo_root" without carving out test
  files. `get_recent_commits`, `run_tests`, and `read_file` already cover test-related
  questions directly, so `tests/` (826 non-test chunks vs. 1426 for the whole repo) is
  skipped as a deliberate scope choice -- implemented as `TEST_DIR_NAMES` in
  `src/rag/ingest.py`, not a prompt instruction.
- **`PINECONE_NAMESPACE` env var added.** §12's env list doesn't include it, but §8.3
  says "use the repo name as the namespace" and both `ingest.py` and the `search_code`
  tool need to agree on that namespace at runtime, so it's read from the environment
  (`.env.example` documents it).
- **CI workflow: `GROQ_API_KEY` instead of `OPENAI_API_KEY`, plus an explicit clone
  step.** §11.2's literal YAML references `./target-repo` as if it already exists in the
  checkout, but the PRD also says `target-repo/` is gitignored (§13) -- on a fresh
  `ubuntu-latest` runner there's nothing to ingest without cloning it first, so
  `.github/workflows/eval.yml` adds a "Clone target repo at pinned commit" step before
  the ingest step. Everything else in the workflow (trigger, job name, step order,
  artifact upload) matches §11.2 exactly.
- **Local Docker Engine could not be verified on the builder's machine.** Docker Desktop
  failed to start (`starting services: initializing Inference manager: ... The file
  cannot be accessed by the system`), a stuck runtime socket file under
  `%LOCALAPPDATA%\Docker\run\` that survived killing every Docker process and `wsl
  --shutdown`, and most likely needs a reboot to clear. The Dockerfile and
  `run_tests`-must-run-in-Docker rule are implemented as specified either way; the
  authoritative `EVAL_REPORT.md` comes from the GitHub Actions run in §11.2, which runs
  directly on the `ubuntu-latest` runner (an isolated, ephemeral CI VM, not the "host" the
  rule is about) rather than through this repo's own Dockerfile, so it doesn't depend on
  Docker working locally.
