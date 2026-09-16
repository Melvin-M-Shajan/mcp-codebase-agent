FROM python:3.11-slim

# git is needed for get_recent_commits (git log) and for cloning the target repo.
# build-essential covers the compiled deps ragas/tree-sitter pull in.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Target repo pinned to the commit recorded in README.md ("Target Repo"). Cloning at
# build time (rather than relying on a bind mount) is what makes the image reproducible.
ARG TARGET_REPO_URL=https://github.com/encode/httpx.git
ARG TARGET_REPO_COMMIT=b5addb64f0161ff6bfe94c124ef76f6a1fba5254
RUN git clone ${TARGET_REPO_URL} target-repo \
    && cd target-repo \
    && git checkout ${TARGET_REPO_COMMIT}
# Matches target-repo/requirements.txt's test-relevant subset (skips docs/packaging
# tools httpx's own CI needs but run_tests doesn't).
RUN pip install --no-cache-dir -e "./target-repo[brotli,cli,http2,socks,zstd]" \
    chardet cryptography pytest trio trustme uvicorn

COPY src/ ./src/
COPY eval/ ./eval/

ENV REPO_ROOT=/app/target-repo
ENV PYTHONPATH=/app

# CMD, not ENTRYPOINT: §11.1 runs eval as `docker run <image> python -m src.eval.run_eval
# ...`, which needs to replace the default command outright, not append to it.
CMD ["python", "-m", "src.demo.chat"]
