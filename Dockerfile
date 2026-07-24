# The application image: FastAPI server + LangGraph orchestration.
#
# Agent shell commands do NOT run here — they run in the separate, minimal runner image
# (Dockerfile.runner), launched per command via the Docker socket. This image therefore
# needs only the Docker CLI, not a full toolchain.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# docker-cli only: the client that talks to the mounted socket. No daemon here.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/appbuilder

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY benchmark ./benchmark

# The workspace is a volume shared with the runner containers via --volumes-from.
ENV PM_WORKSPACE=/srv/appbuilder/workspace \
    RUNS_DIR=/srv/appbuilder/runs
RUN mkdir -p "$PM_WORKSPACE" "$RUNS_DIR"

EXPOSE 8020

CMD ["python", "-m", "app.server"]
