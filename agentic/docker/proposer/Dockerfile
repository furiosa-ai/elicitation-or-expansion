# Proposer sandbox image: Claude Code CLI + the SAME Python stack the host evaluates candidates with
# (uv.lock), so the proposer can import-check and prototype candidates. Always started unprivileged by
# core/claude_wrapper.wrap_in_container: --user <uid:gid> --cap-drop=ALL --security-opt no-new-privileges,
# mounts only this run's checkout + per-run HOME + train/val data views. No docker.sock, ever.
#
# Build (from repo root; CLAUDE_CODE_VERSION must equal the host `claude --version`):
#   docker build --build-arg CLAUDE_CODE_VERSION=$(claude --version | awk '{print $1}') -t agentic-proposer:v1 -f docker/proposer/Dockerfile .
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates ripgrep procps bash curl jq \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.260
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && claude --version

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy
WORKDIR /opt/build
COPY pyproject.toml uv.lock ./
RUN uv python install 3.12 \
    && uv sync --frozen --no-dev \
    && /opt/venv/bin/appworld install \
    && /opt/venv/bin/python -c "import appworld.apps.admin.models, bm25s, openai, jinja2, yaml" \
    && chmod -R a+rX /opt/uv-python /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/proposer \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace
WORKDIR /workspace

# login shells (bash -l) reset PATH via /etc/profile; keep the venv first there too
RUN printf 'export PATH=/opt/venv/bin:$PATH\nexport PYTHONPATH=/workspace\n' > /etc/profile.d/agentic.sh
