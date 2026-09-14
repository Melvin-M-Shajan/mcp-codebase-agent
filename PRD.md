# PRD — MCP Codebase Agent (Agentic AI + RAG + MCP)

Status: Spec — not yet built. This document is written so a coding agent can implement the
project end-to-end without further clarification. Every interface, schema, and config below
is a concrete default — deviate only when blocked (see §14 Risks), and record any deviation
in the repo README.

## 1. Purpose
Replace the GraphRAG Enterprise resume project (private-source, unverifiable) with a public,
demoable project that closes the biggest recurring gap in target job postings: Model
Context Protocol (MCP). Reinforces Agentic AI + RAG evidence with a real, working system.

## 2. Problem
Swark (existing resume project) produces a static, one-shot architecture summary of an
unfamiliar repo. An engineer still has to read that output and manually investigate further.
An agent that takes follow-up questions and actively investigates the codebase — searching,
reading files, checking test results and commit history as needed — is the natural next
step, and matches how target job postings describe the work: tool-using agents that reason,
plan, and call APIs, not single-shot RAG lookups.

## 3. Goal
Build an MCP server exposing codebase-inspection tools, and a LangGraph agent that plans
multi-step investigations, calls those tools, and grounds its answers with RAG over the
repo's code and docs — able to correctly answer real developer questions about one specific,
chosen open-source repo.

## 4. Scope

**In scope**
- MCP server exposing exactly 4 tools (§6).
- LangGraph agent with an explicit plan → act → observe loop (§7), not single-shot RAG.
- RAG layer over the target repo's code + docs using Pinecone (§8).
- One target repo, chosen per §5 criteria — tools should be written repo-agnostic
  (parameterized by a repo root path/URL), but only one repo needs to actually be indexed
  and demoed.
- Evaluation: 20-30 question benchmark, scored with RAGAS (§9), traced with Langfuse.
- Dockerized deployment + GitHub Actions CI running the eval suite (§11).

**Out of scope (do not build these)**
- IDE/editor plugin (VS Code extension, etc.) — CLI or simple chat script only (§10).
- Multi-repo or cross-repo reasoning.
- Write access — the agent only reads/inspects/runs tests; it must never modify files,
  commit, or open PRs. Enforce this in the MCP server itself (no write/delete tool exists),
  not just by prompt instruction.
- Fine-tuning any model (that's the separate invoice-finetune project).
- Support for arbitrary/unknown repos supplied at demo time — index one repo ahead of time.

## 5. Target repo selection
Pick a real, actively maintained open-source repo, roughly 5,000-40,000 lines of source
code (large enough to be non-trivial, small enough that RAG chunking stays manageable —
avoid a massive monorepo). Prefer a repo the builder does NOT already know well, so the
demo Q&A is credible evidence of the agent doing real work rather than the builder already
knowing the answers. Record the exact repo name, URL, and commit hash indexed in
`README.md` under "Target Repo" — the eval benchmark (§9) is tied to that exact commit.

## 6. MCP Server Spec

Implementation: Python MCP SDK (`mcp` package), stdio transport (simplest for local
dev/CI; do not build an SSE/HTTP transport unless stdio proves insufficient).

Exactly 4 tools, each defined with a strict JSON input schema and a strict output schema.
An implementing agent must implement all 4 exactly as specified — no additional tools in
v1 (extra tools dilute the "primary skill depth" story the PRD is built around).

### 6.1 `search_code`
```json
{
  "name": "search_code",
  "description": "Semantic + keyword search across the indexed repository's source files. Returns the most relevant code chunks for a natural-language or keyword query.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Natural language or keyword search query."},
      "top_k": {"type": "integer", "default": 8, "minimum": 1, "maximum": 20}
    },
    "required": ["query"]
  },
  "output_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "file_path": {"type": "string"},
        "start_line": {"type": "integer"},
        "end_line": {"type": "integer"},
        "snippet": {"type": "string"},
        "score": {"type": "number"}
      }
    }
  }
}
```
Implementation: queries the Pinecone index (§8), returns top_k matches re-ranked by score.

### 6.2 `read_file`
```json
{
  "name": "read_file",
  "description": "Read the contents of a specific file in the repository, optionally restricted to a line range.",
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string", "description": "Repo-relative file path."},
      "start_line": {"type": "integer"},
      "end_line": {"type": "integer"}
    },
    "required": ["path"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "content": {"type": "string"},
      "total_lines": {"type": "integer"}
    }
  }
}
```
Implementation: reads directly from the checked-out repo on disk (read-only filesystem
access, path-traversal-safe — resolve and validate `path` stays within the repo root before
reading, reject `..` segments).

### 6.3 `run_tests`
```json
{
  "name": "run_tests",
  "description": "Run the repository's test suite, or a subset matching a target path/pattern, and return pass/fail results.",
  "input_schema": {
    "type": "object",
    "properties": {
      "target": {"type": "string", "description": "Optional test file/path/pattern to restrict the run."}
    }
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "passed": {"type": "integer"},
      "failed": {"type": "integer"},
      "output": {"type": "string", "description": "Truncated to last 4000 chars of raw test-runner output."}
    }
  }
}
```
Implementation: shells out to the target repo's actual test command (e.g., `pytest`,
`npm test`) via subprocess with a hard timeout (120s) and no network access if avoidable
(run in the Docker container built in §11, which already sandboxes this). Record which
test command the target repo uses in `README.md`.

### 6.4 `get_recent_commits`
```json
{
  "name": "get_recent_commits",
  "description": "Get recent commit history for the whole repo or a specific file/path.",
  "input_schema": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "limit": {"type": "integer", "default": 10, "maximum": 50}
    }
  },
  "output_schema": {
    "type": "array",
    "items": {
      "type": "object",
      "properties": {
        "hash": {"type": "string"},
        "author": {"type": "string"},
        "date": {"type": "string"},
        "message": {"type": "string"},
        "files_changed": {"type": "array", "items": {"type": "string"}}
      }
    }
  }
}
```
Implementation: shells out to `git log` against the local clone (`git log -n <limit>
--pretty=... -- <path>`), parse into the schema above.

### 6.5 Server entry point
`src/mcp_server/server.py` — registers all 4 tools, reads `REPO_ROOT` from environment/config
at startup, refuses to start if `REPO_ROOT` is not a valid git repo.

## 7. Agent Spec (LangGraph)

### 7.1 State schema — `src/agent/state.py`
```python
from typing import TypedDict, Literal

class ToolCall(TypedDict):
    tool: str
    args: dict
    result: dict | list

class AgentState(TypedDict):
    question: str
    plan: str | None
    tool_calls: list[ToolCall]
    retrieved_context: list[dict]   # RAG hits from §8, if used
    answer: str | None
    step_count: int
    max_steps: int   # default 6 — hard cap to prevent infinite loops
```

### 7.2 Graph nodes — `src/agent/graph.py`
```python
def plan_node(state: AgentState) -> AgentState:
    """LLM call: given question + tool_calls so far + retrieved_context, decide next
    action. Returns updated state with either a new tool call queued in state["plan"]
    (as a structured {"tool": ..., "args": ...} the router reads) or a final answer if
    enough information has been gathered."""

def retrieve_node(state: AgentState) -> AgentState:
    """Always runs once at the start: calls search_code via RAG (§8) with the raw
    question to seed retrieved_context before planning begins."""

def tool_node(state: AgentState) -> AgentState:
    """Executes the tool call the plan_node decided on (via the MCP client), appends
    the result to state["tool_calls"], increments step_count."""

def answer_node(state: AgentState) -> AgentState:
    """LLM call: given the full question + tool_calls + retrieved_context, produce the
    final answer, citing specific files/lines it used."""
```

### 7.3 Routing logic
`retrieve_node` → `plan_node` → conditional edge:
- if `state["step_count"] >= state["max_steps"]` → force `answer_node` (log a warning
  that the step budget was exhausted).
- elif `plan_node` decided it has enough information → `answer_node`.
- else → `tool_node` → back to `plan_node`.

This is the multi-step loop — at least some of the 20-30 eval questions (§9) must exercise
2+ tool calls (e.g., search then read, or search then check commit history) to prove this
isn't single-shot RAG with extra steps.

### 7.4 System prompt (plan_node)
```
You are a codebase investigation agent. You have access to these tools: search_code,
read_file, run_tests, get_recent_commits. Given a developer's question about this
repository, decide what you still need to find out. If you already have enough
information from the context gathered so far, say so and stop calling tools. Prefer the
minimum number of tool calls that gets a correct, well-grounded answer. Always cite
specific file paths and line numbers you relied on.
```

## 8. RAG Layer Spec

### 8.1 Chunking
Function/class-level chunking, not fixed-size windows: parse the target repo's source
files with a language-aware splitter (use `tree-sitter` if the target repo is in a language
`tree-sitter` supports well, otherwise fall back to a simple heuristic — split on top-level
`def`/`class`/`function` boundaries via regex). Each chunk: max 1500 tokens, include a
header comment with `# file: <path> lines: <start>-<end>` prepended before embedding so the
model sees provenance in-context. Also chunk `README.md` and any `docs/` markdown files by
heading section.

### 8.2 Embedding model
`text-embedding-3-small` (OpenAI) or an equivalent open-source model
(`sentence-transformers/all-MiniLM-L6-v2` if avoiding API cost) — record which was used in
`README.md`.

### 8.3 Vector store — Pinecone
- One index, e.g. `mcp-codebase-agent`, dimension matching the embedding model (1536 for
  `text-embedding-3-small`, 384 for `all-MiniLM-L6-v2`), metric `cosine`.
- Metadata per vector: `file_path`, `start_line`, `end_line`, `chunk_type`
  (`"code" | "doc"`).
- Namespace per indexed repo (so re-indexing a different repo later doesn't require a new
  index) — use the repo name as the namespace.

### 8.4 Ingestion script — `src/rag/ingest.py`
```python
def ingest_repo(repo_root: str, index_name: str, namespace: str) -> int:
    """Walks repo_root, chunks per §8.1, embeds per §8.2, upserts to Pinecone per §8.3.
    Skips binary files, .git/, node_modules/, and anything matching .gitignore. Returns
    the number of chunks ingested. Must be idempotent — re-running on the same repo/commit
    should not create duplicates (use a deterministic vector ID: hash of file_path +
    start_line + end_line)."""
```

## 9. Evaluation

### 9.1 Benchmark format — `eval/questions.jsonl`
One JSON object per line:
```json
{
  "id": "q01",
  "question": "Where is rate limiting implemented for the API client?",
  "requires_multistep": true,
  "reference_files": ["src/client/http.py"],
  "reference_answer": "Free-text description of what a correct answer must cover, used as the RAGAS reference/ground truth."
}
```
20-30 questions covering: "where is X handled" (search-heavy), "why does Y fail under Z
condition" (search + read, often multi-step), "what changed recently in module A" (commit
history), "does test suite T currently pass" (run_tests), and at least 3 questions whose
correct answer requires chaining 2+ tools (e.g., find the function via search, then check
its recent commit history because the question is about a regression).

### 9.2 Scoring — `src/eval/run_eval.py`
For each question: run the full agent graph (§7), capture the final answer + the full
`tool_calls` trace, then score with RAGAS:
- `faithfulness` — is the answer grounded in the retrieved/tool-returned context (not
  hallucinated)?
- `answer_relevancy` — does the answer address the actual question?
- `context_precision` — of the context retrieved/used, how much was actually relevant?

Also compute a simple non-RAGAS check: for questions with `reference_files` set, did the
agent's tool calls (`read_file` or `search_code` hits) touch at least one of the reference
files? Report this as `file_grounding_rate` alongside the RAGAS scores.

### 9.3 Tracing — Langfuse
Wrap the agent graph invocation with the Langfuse callback/handler so every eval run
produces a full trace (each node's input/output, each tool call and its latency) visible
in the Langfuse UI/dashboard. Use one Langfuse "session" per eval run so all 20-30 question
traces for a given run are grouped together.

### 9.4 Report — `eval/EVAL_REPORT.md`
Must contain: a table of per-question scores (faithfulness, answer_relevancy,
context_precision, file_grounding_rate, num_tool_calls), aggregate means, and 2-3 annotated
failure cases with the actual tool-call trace shown, not hidden.

## 10. Demo
`src/demo/chat.py` — simple REPL: reads a question from stdin, runs the agent graph,
prints the final answer plus a compact log of which tools were called in what order. A
short screen recording (or asciinema cast) showing 3-4 real exchanges, including at least
one multi-step case, goes in `demo/` as the deliverable artifact (a Gradio/web chat UI is
optional/stretch, not required).

## 11. Deployment

### 11.1 Docker — `Dockerfile`
Base image `python:3.11-slim`. Installs the target repo's runtime (if `run_tests`
requires, e.g., Node — install it too) plus this project's `requirements.txt`. Clones or
copies in the target repo at a pinned commit (§5) at build time so the demo is
reproducible. Entrypoint runs `src/demo/chat.py` by default; eval is run via
`docker run <image> python -m src.eval.run_eval`.

### 11.2 CI — `.github/workflows/eval.yml`
Outline (implement exactly this trigger/job structure):
```yaml
name: eval
on:
  pull_request:
  workflow_dispatch:
jobs:
  run-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: python -m src.rag.ingest --repo-root ./target-repo --index-name mcp-codebase-agent --namespace ci
        env:
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - run: python -m src.eval.run_eval --questions eval/questions.jsonl --out eval/EVAL_REPORT.md
        env:
          PINECONE_API_KEY: ${{ secrets.PINECONE_API_KEY }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          LANGFUSE_PUBLIC_KEY: ${{ secrets.LANGFUSE_PUBLIC_KEY }}
          LANGFUSE_SECRET_KEY: ${{ secrets.LANGFUSE_SECRET_KEY }}
      - uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: eval/EVAL_REPORT.md
```
This is the "real CI/CD" proof point — it must actually run the eval suite, not just lint.

## 12. Environment & Dependencies
`requirements.txt` (minimum versions to start from):
```
mcp>=1.0
langgraph>=0.2
langchain-core>=0.2
pinecone-client>=5.0
openai>=1.30            # embeddings + LLM calls, or swap for anthropic>=0.30
ragas>=0.1.9
langfuse>=2.36
tree-sitter>=0.22        # optional, only if used for chunking
gitpython>=3.1
```
`.env.example` (document all, never commit real values):
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`
- `PINECONE_API_KEY`
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`
- `REPO_ROOT` — local path to the cloned target repo

## 13. Repository Structure
```
mcp-codebase-agent/
  README.md                    # setup, target repo + commit, architecture diagram, results
  requirements.txt
  .env.example
  Dockerfile
  .github/workflows/eval.yml
  target-repo/                  # gitignored, or a git submodule pinned to the commit from §5
  src/
    mcp_server/
      server.py
      tools/
        search_code.py
        read_file.py
        run_tests.py
        get_recent_commits.py
    rag/
      ingest.py
      chunking.py
    agent/
      state.py
      graph.py
      prompts.py
    eval/
      run_eval.py
    demo/
      chat.py
  eval/
    questions.jsonl
    EVAL_REPORT.md
  demo/
    recording.gif              # or asciinema cast link in README
  tests/
    test_tools.py
    test_chunking.py
    test_state_transitions.py
```

## 14. Testing Plan
- Unit tests for each MCP tool (§6) against a small fixture repo checked into `tests/`
  (not the full target repo) — verify schema compliance, path-traversal rejection in
  `read_file`, and correct truncation behavior in `run_tests`.
- Unit test for `chunking.py` — verify chunk boundaries land on function/class edges for a
  fixture source file, and that the file-path header is present in every chunk.
- Integration test: run the full agent graph (§7) against one fixture question with a
  mocked MCP client, assert it terminates within `max_steps` and produces a non-empty
  answer.

## 15. Milestones (Definition of Done per milestone)
1. **MCP server** — all 4 tools implemented and pass `tests/test_tools.py` standalone
   (no agent yet). DoD: tools importable and callable directly, correct schemas.
2. **RAG ingestion** — `ingest_repo` runs against the chosen target repo, chunks land in
   Pinecone, manual spot-check of 5 queries returns sensible results. DoD:
   `tests/test_chunking.py` passes, Pinecone index has >0 vectors in the target namespace.
3. **Agent graph** — full plan→tool→observe→answer loop wired, `max_steps` cap enforced,
   passes the integration test. DoD: can answer a hand-picked easy question correctly
   end-to-end via `src/demo/chat.py`.
4. **Eval suite** — `eval/questions.jsonl` has 20-30 real questions per §9.1, full run
   produces `eval/EVAL_REPORT.md` with real RAGAS scores (not placeholders) and a Langfuse
   trace per question. DoD: report shows at least 3 questions with `num_tool_calls >= 2`.
5. **Packaging** — Dockerfile builds and runs the demo; `.github/workflows/eval.yml`
   actually executes on a PR and uploads the report artifact. DoD: a fresh PR to the repo
   triggers the workflow and it completes (pass or documented fail) without manual steps.

## 16. Risks & Mitigations
- **MCP SDK rough edges** (protocol is young) — budget extra time for the server
  implementation specifically; isolate MCP protocol issues from agent-logic issues by
  testing tools standalone first (Milestone 1) before wiring the agent.
- **Target repo too large** → chunking/retrieval quality becomes the bottleneck; stick to
  the 5k-40k LOC guidance in §5, don't pick a monorepo.
- **`run_tests` sandboxing** — always execute inside the Docker container (§11.1), never on
  the host, since it runs arbitrary repo test commands.
- **RAGAS/Langfuse API cost** — both call an LLM for scoring/embedding; keep the eval set
  at 20-30 questions (not larger) to keep this cheap and fast to rerun in CI.
