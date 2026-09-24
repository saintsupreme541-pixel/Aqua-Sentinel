# AQUA-SENTINEL backend — build from the repository root:
#   docker build -f backend/Dockerfile -t aqua-backend .
#
# The container layout mirrors the repo so default config paths resolve:
# code at /workspace/backend, registry at /workspace/models, samples at
# /workspace/data/samples.
FROM python:3.12-slim

WORKDIR /workspace

COPY backend/pyproject.toml backend/requirements-docker.txt* backend/
COPY backend/app backend/app
COPY backend/scripts backend/scripts
COPY models models
COPY data/samples data/samples

RUN cd backend && pip install --no-cache-dir -e ".[dev]"

ENV AQUA_STORAGE_DIR=/data \
    AQUA_SAMPLE_DATA=/workspace/data/samples \
    AQUA_MODEL_REGISTRY=/workspace/models/registry.json \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
