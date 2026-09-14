# Demo session transcript

Real output from `python -m src.demo.chat`, captured verbatim (Docker recording/asciinema
wasn't possible on this machine -- see README "Deviations from PRD" for why; this is a
literal transcript of the actual REPL exchanges instead). Model: `openai/gpt-oss-120b`
(Groq). Repo: [encode/httpx](https://github.com/encode/httpx) @
`b5addb64f0161ff6bfe94c124ef76f6a1fba5254`.

---

## Exchange 1 -- single-step (RAG-only)

**Q:** Where is Basic Auth implemented in httpx, and what does it do to the request?

**A:**

`BasicAuth` is primarily implemented in the `httpx._auth.py` file, within the `BasicAuth` class.

Specifically:
- The `BasicAuth` class is defined in `httpx/_auth.py` starting at line 126.
- The `__init__` method (lines 131-132) takes a `username` and `password` and calls `_build_auth_header` to create the `Authorization` header.
- The `auth_flow` method (lines 134-136) is responsible for adding the pre-built `Authorization` header to the outgoing `Request`.
- The `_build_auth_header` method (lines 138-142) handles the encoding of the username and password into a base64 token and formats it as a "Basic" authentication header.

Additionally, `httpx/_client.py` contains logic to automatically create `BasicAuth` instances:
- If the `auth` argument passed to a client or request is a `tuple` (e.g., `("user", "pass")`), the `_build_auth` method in `httpx/_client.py` (lines 445-455) converts it into a `BasicAuth` object.
- If a URL contains a username and password (e.g., `https://user:pass@example.com`), the `_build_request_auth` method in `httpx/_client.py` (lines 457-473) will automatically create a `BasicAuth` object from them.

**Tools called:** none -- answered entirely from the RAG context gathered by `retrieve_node`.

---

## Exchange 2 -- multi-step (search -> git history)

**Q:** Find where DigestAuth is implemented, then tell me the commit hash and author of the most recent commit to that file.

**A:**

`DigestAuth` is implemented in the file `httpx/_auth.py`.

The most recent commit to `httpx/_auth.py` is:
- **Commit Hash:** `ae1b9f66238f75ced3ced5e4485408435de1076`
- **Author:** Josh Cannon

**Tools called (in order):**
1. `get_recent_commits({'path': 'httpx/_auth.py', 'limit': 1})`

---

## Exchange 3 -- multi-step (read_file + git history, two rounds)

**Q:** How many total lines are in httpx/_multipart.py, and what is the hash of the most recent commit to that file?

**A:**

The file `httpx/_multipart.py` has 304 lines.
The hash of the most recent commit to `httpx/_multipart.py` is `6212e8fa3bf90154b891c7721f6d29737caac861` (from `get_recent_commits` tool call).

**Tools called (in order):**
1. `read_file({'path': 'httpx/_multipart.py'})`
2. `read_file({'path': 'httpx/_multipart.py'})`
3. `get_recent_commits({'path': 'httpx/_multipart.py', 'limit': 1})`

**Note (unfabricated -- included on purpose):** the file actually has **300** lines (verified
directly: `read_file(...).total_lines == 300`, matching `wc -l`). The agent's own tool
call got the correct number back, but its final-answer generation misstated it as 304 --
a real, observed faithfulness slip, not a tool bug. This is exactly the kind of gap
`EVAL_REPORT.md`'s faithfulness metric is meant to catch; see that report for the scored
version of this failure mode across the full question set.

---

## Exchange 4 -- multi-step (search + read_file x2, deep investigation)

**Q:** Why would DigestAuth fail with a ProtocolError when handling a server's 401 response?

**A:**

`httpx.DigestAuth` only raises a `ProtocolError` while it is trying to turn the 401
`WWW-Authenticate` header the server sent into a usable challenge. Two distinct places in
the implementation can produce that exception:

| Location | What is checked | When `ProtocolError` is raised |
|---|---|---|
| `DigestAuth._parse_challenge` (lines 224-253) | Header must contain mandatory `realm` and `nonce` fields | Missing either key -> `KeyError` caught and re-raised as `ProtocolError("Malformed Digest WWW-Authenticate header")` |
| `DigestAuth._resolve_qop` (lines 329-340) | The optional `qop` directive is validated | `qop` present but not `auth` and not exactly `[b"auth-int"]` -> `ProtocolError(f'Unexpected qop value "{qop!r}" in digest auth')` |

So a `DigestAuth` failure with `ProtocolError` on a 401 means either the server's Digest
challenge is malformed (missing `realm`/`nonce`) or specifies an unsupported `qop` value.

**Tools called (in order):**
1. `read_file({'path': 'httpx/_auth.py', 'start_line': 320, 'end_line': 350})`
2. `search_code({'query': 'ProtocolError DigestAuth', 'top_k': 10})`
3. `read_file({'path': 'httpx/_auth.py', 'start_line': 224, 'end_line': 253})`
