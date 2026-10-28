# One image for trainer, API and dashboard (different commands in docker-compose).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    SUPPLYSIGHT_HOME=/app PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 make && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

COPY . .
RUN useradd -m appuser && mkdir -p data artifacts && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "supplysight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
