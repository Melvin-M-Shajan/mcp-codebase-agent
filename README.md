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
  heading-based chunking for Markdown, embedded with Gemini, stored in Pinecone
  (namespace = repo name).
- **Agent** (`src/agent/`): `retrieve_node -> plan_node -> (tool_node -> plan_node)* ->
  answer_node`, capped at `max_steps=6`.
- **Eval** (`src/eval/run_eval.py`): runs the real agent against `eval/questions.jsonl`,
  scores with RAGAS, traces every run with Langfuse.

## Setup

```bash
cp .env.example .env   # fill in GOOGLE_API_KEY, PINECONE_API_KEY, LANGFUSE_*
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

- **LLM + embeddings: Gemini instead of OpenAI/Anthropic.** The PRD's primary default is
  OpenAI (`text-embedding-3-small` + an OpenAI/Anthropic chat model). The user supplied a
  working Gemini API key instead, so `plan_node`/`answer_node`/RAGAS's judge model use
  `gemini-2.5-flash` (`src/agent/llm.py`) and embeddings use `gemini-embedding-2` at 1536
  dimensions (`src/rag/embeddings.py`) -- chosen to match the PRD's own OpenAI-dimension
  precedent so the rest of §8.3 (Pinecone index config) applies unchanged.
- **RAG index excludes `tests/`.** §8.4 says "walks repo_root" without carving out test
  files. `get_recent_commits`, `run_tests`, and `read_file` already cover
  test-related questions directly, and excluding `tests/` (826 chunks vs. 1426 for the
  whole repo) keeps ingestion comfortably under Gemini's free-tier daily embedding quota
  (1000 embed requests/day/model -- see below). This is implemented as
  `TEST_DIR_NAMES` in `src/rag/ingest.py`, not a prompt instruction.
- **`PINECONE_NAMESPACE` env var added.** §12's env list doesn't include it, but §8.3
  says "use the repo name as the namespace" and both `ingest.py` and the `search_code`
  tool need to agree on that namespace at runtime, so it's read from the environment
  (`.env.example` documents it).
- **CI workflow: `GOOGLE_API_KEY` instead of `OPENAI_API_KEY`, plus an explicit clone
  step.** §11.2's literal YAML references `./target-repo` as if it already exists in the
  checkout, but the PRD also says `target-repo/` is gitignored (§13) -- on a fresh
  `ubuntu-latest` runner there's nothing to ingest without cloning it first, so
  `.github/workflows/eval.yml` adds a "Clone target repo at pinned commit" step before
  the ingest step. Everything else in the workflow (trigger, job name, step order,
  artifact upload) matches §11.2 exactly.
- **Gemini free-tier embedding quota is tight for this repo's size.** `gemini-embedding-2`
  free tier caps out at 1000 embed requests/day/model (and 100/minute -- `ingest.py`
  paces batches to stay under the per-minute cap). A full non-test ingestion of httpx
  (826 chunks) uses most of a day's quota on its own; running ingestion twice in one day
  (e.g. once locally, once in CI) on the same key can exhaust it. If `EVAL_REPORT.md`
  shows fewer than the full question set scored, or CI's ingest step fails with
  `RESOURCE_EXHAUSTED`, that's why -- either wait for the daily reset or enable billing
  on the Gemini key to remove the cap.
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
