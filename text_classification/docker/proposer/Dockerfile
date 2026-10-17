# Proposer sandbox image (base stack): Claude Code CLI plus the text_classification Python stack.
#
# The proposer writes candidate memory systems and import-checks them, so the image must hold the
# same Python stack as the host. claude_wrapper.wrap_in_container always starts the container
# unprivileged:
#   --user <host uid:gid> --cap-drop=ALL --security-opt no-new-privileges
#   mounts: this run's checkout -> /workspace, per-run home -> /proposer-home
#   docker.sock is never mounted.
#
# Build:
#   docker build -t text-proposer:v1 docker/proposer
FROM node:22-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep procps bash \
    && rm -rf /var/lib/apt/lists/*

ARG CLAUDE_CODE_VERSION=2.1.263
RUN npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} \
    && claude --version

# The Python stack lives in world-readable locations so that it runs under any --user.
# Versions follow text_classification/pyproject.toml.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
# The proposer prototypes candidates here that are later evaluated on the host, so the versions below
# match uv.lock.
RUN uv python install 3.13.14 \
    && uv venv /opt/venv --python 3.13.14 \
    && uv pip install --python /opt/venv/bin/python \
        litellm==1.84.10 \
        openai-harmony==0.0.8 \
        datasets==5.0.1 \
        tenacity==9.1.4 \
        tqdm==4.70.0 \
        pyyaml==6.0.3 \
        numpy==2.5.2 \
        python-dotenv==1.2.3 \
    && chmod -R a+rX /opt/uv-python /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/proposer-home \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace
