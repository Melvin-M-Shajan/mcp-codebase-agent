"""§6.3 run_tests -- shells out to the target repo's test command with a hard timeout.

Risk mitigation (§16): this MUST only ever be invoked from inside the project's Docker
container (see Dockerfile / README), never on the host, since it executes the target
repo's arbitrary test command via subprocess.
"""

import re
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from src.mcp_server.repo_root import PathTraversalError, resolve_safe_path

TIMEOUT_SECONDS = 120
OUTPUT_TRUNCATE_CHARS = 4000

# Regex forms pytest itself prints, e.g. "3 passed, 1 failed in 0.42s" or "5 passed in 0.10s"
_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")
_ERROR_RE = re.compile(r"(\d+) error")


class RunTestsResult(BaseModel):
    passed: int
    failed: int
    output: str


def run_tests(repo_root: Path, target: str | None = None) -> RunTestsResult:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    if target:
        try:
            safe_target = resolve_safe_path(repo_root, target)
        except PathTraversalError:
            return RunTestsResult(passed=0, failed=0, output=f"rejected: unsafe target path {target!r}")
        cmd.append(str(safe_target.relative_to(repo_root)))

    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
        raw_output = proc.stdout + "\n" + proc.stderr
    except subprocess.TimeoutExpired as exc:
        raw_output = (exc.stdout or "") + (exc.stderr or "") + f"\n[run_tests] timed out after {TIMEOUT_SECONDS}s"

    passed = sum(int(m) for m in _PASSED_RE.findall(raw_output))
    failed = sum(int(m) for m in _FAILED_RE.findall(raw_output)) + sum(
        int(m) for m in _ERROR_RE.findall(raw_output)
    )

    truncated = raw_output[-OUTPUT_TRUNCATE_CHARS:]
    return RunTestsResult(passed=passed, failed=failed, output=truncated)
