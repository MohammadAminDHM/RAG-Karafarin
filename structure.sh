#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./create_rag_scaffold.sh [project_name]
# Example:
#   ./create_rag_scaffold.sh rag-service

PROJECT_ROOT="${1:-rag-service}"

echo "Creating project scaffold: ${PROJECT_ROOT}"

# -----------------------------
# Directories
# -----------------------------
mkdir -p "${PROJECT_ROOT}"/app/api/routes
mkdir -p "${PROJECT_ROOT}"/app/core
mkdir -p "${PROJECT_ROOT}"/app/schemas
mkdir -p "${PROJECT_ROOT}"/app/rag
mkdir -p "${PROJECT_ROOT}"/app/providers
mkdir -p "${PROJECT_ROOT}"/app/services
mkdir -p "${PROJECT_ROOT}"/app/storage/vectorstore
mkdir -p "${PROJECT_ROOT}"/app/storage/embeddings
mkdir -p "${PROJECT_ROOT}"/app/storage/documents
mkdir -p "${PROJECT_ROOT}"/app/middleware
mkdir -p "${PROJECT_ROOT}"/app/utils

mkdir -p "${PROJECT_ROOT}"/data/input
mkdir -p "${PROJECT_ROOT}"/data/processed
mkdir -p "${PROJECT_ROOT}"/data/indexes

mkdir -p "${PROJECT_ROOT}"/tests/unit
mkdir -p "${PROJECT_ROOT}"/tests/integration
mkdir -p "${PROJECT_ROOT}"/tests/fixtures

mkdir -p "${PROJECT_ROOT}"/scripts
mkdir -p "${PROJECT_ROOT}"/configs
mkdir -p "${PROJECT_ROOT}"/docker

# -----------------------------
# Files (app root)
# -----------------------------
touch "${PROJECT_ROOT}"/app/main.py

# API
touch "${PROJECT_ROOT}"/app/api/deps.py
touch "${PROJECT_ROOT}"/app/api/routes/query.py
touch "${PROJECT_ROOT}"/app/api/routes/health.py
touch "${PROJECT_ROOT}"/app/api/routes/admin.py

# Core
touch "${PROJECT_ROOT}"/app/core/config.py
touch "${PROJECT_ROOT}"/app/core/logging.py
touch "${PROJECT_ROOT}"/app/core/exceptions.py
touch "${PROJECT_ROOT}"/app/core/constants.py

# Schemas
touch "${PROJECT_ROOT}"/app/schemas/query.py
touch "${PROJECT_ROOT}"/app/schemas/retrieval.py
touch "${PROJECT_ROOT}"/app/schemas/provider.py

# RAG
touch "${PROJECT_ROOT}"/app/rag/pipeline.py
touch "${PROJECT_ROOT}"/app/rag/retriever.py
touch "${PROJECT_ROOT}"/app/rag/prompt_builder.py
touch "${PROJECT_ROOT}"/app/rag/context_formatter.py
touch "${PROJECT_ROOT}"/app/rag/citations.py

# Providers
touch "${PROJECT_ROOT}"/app/providers/base.py
touch "${PROJECT_ROOT}"/app/providers/local_provider.py
touch "${PROJECT_ROOT}"/app/providers/api_provider.py
touch "${PROJECT_ROOT}"/app/providers/router.py
touch "${PROJECT_ROOT}"/app/providers/policies.py

# Services
touch "${PROJECT_ROOT}"/app/services/inference_service.py
touch "${PROJECT_ROOT}"/app/services/ingestion_service.py
touch "${PROJECT_ROOT}"/app/services/busy_detector.py
touch "${PROJECT_ROOT}"/app/services/metrics_service.py

# Storage - vectorstore
touch "${PROJECT_ROOT}"/app/storage/vectorstore/base.py
touch "${PROJECT_ROOT}"/app/storage/vectorstore/qdrant_store.py
touch "${PROJECT_ROOT}"/app/storage/vectorstore/faiss_store.py

# Storage - embeddings
touch "${PROJECT_ROOT}"/app/storage/embeddings/embedder.py
touch "${PROJECT_ROOT}"/app/storage/embeddings/cache.py

# Storage - documents
touch "${PROJECT_ROOT}"/app/storage/documents/loader.py
touch "${PROJECT_ROOT}"/app/storage/documents/cleaner.py
touch "${PROJECT_ROOT}"/app/storage/documents/chunker.py

# Middleware
touch "${PROJECT_ROOT}"/app/middleware/request_id.py
touch "${PROJECT_ROOT}"/app/middleware/timing.py
touch "${PROJECT_ROOT}"/app/middleware/auth.py

# Utils
touch "${PROJECT_ROOT}"/app/utils/retry.py
touch "${PROJECT_ROOT}"/app/utils/timeouts.py
touch "${PROJECT_ROOT}"/app/utils/hashing.py

# -----------------------------
# Data files
# -----------------------------
touch "${PROJECT_ROOT}"/data/input/static_knowledge.txt
touch "${PROJECT_ROOT}"/data/processed/chunks.jsonl
touch "${PROJECT_ROOT}"/data/processed/metadata.json

# -----------------------------
# Tests
# -----------------------------
touch "${PROJECT_ROOT}"/tests/unit/test_router.py
touch "${PROJECT_ROOT}"/tests/unit/test_busy_detector.py
touch "${PROJECT_ROOT}"/tests/unit/test_prompt_builder.py
touch "${PROJECT_ROOT}"/tests/unit/test_retriever.py

touch "${PROJECT_ROOT}"/tests/integration/test_query_local.py
touch "${PROJECT_ROOT}"/tests/integration/test_query_fallback_api.py
touch "${PROJECT_ROOT}"/tests/integration/test_startup_ingestion.py

touch "${PROJECT_ROOT}"/tests/fixtures/sample_docs.txt

# -----------------------------
# Scripts
# -----------------------------
touch "${PROJECT_ROOT}"/scripts/ingest_static.py
touch "${PROJECT_ROOT}"/scripts/smoke_test_query.py
touch "${PROJECT_ROOT}"/scripts/load_test.py

# -----------------------------
# Configs
# -----------------------------
touch "${PROJECT_ROOT}"/configs/app.example.env
touch "${PROJECT_ROOT}"/configs/logging.yaml
touch "${PROJECT_ROOT}"/configs/provider_map.yaml

# -----------------------------
# Docker
# -----------------------------
touch "${PROJECT_ROOT}"/docker/Dockerfile
touch "${PROJECT_ROOT}"/docker/docker-compose.yml
touch "${PROJECT_ROOT}"/docker/.dockerignore

# -----------------------------
# Project root files
# -----------------------------
touch "${PROJECT_ROOT}"/.env
touch "${PROJECT_ROOT}"/requirements.txt
touch "${PROJECT_ROOT}"/README.md
touch "${PROJECT_ROOT}"/TODO.md

# Optional: make Python packages importable (recommended)
# You can comment this block if you do not want __init__.py files.
touch "${PROJECT_ROOT}"/app/__init__.py
touch "${PROJECT_ROOT}"/app/api/__init__.py
touch "${PROJECT_ROOT}"/app/api/routes/__init__.py
touch "${PROJECT_ROOT}"/app/core/__init__.py
touch "${PROJECT_ROOT}"/app/schemas/__init__.py
touch "${PROJECT_ROOT}"/app/rag/__init__.py
touch "${PROJECT_ROOT}"/app/providers/__init__.py
touch "${PROJECT_ROOT}"/app/services/__init__.py
touch "${PROJECT_ROOT}"/app/storage/__init__.py
touch "${PROJECT_ROOT}"/app/storage/vectorstore/__init__.py
touch "${PROJECT_ROOT}"/app/storage/embeddings/__init__.py
touch "${PROJECT_ROOT}"/app/storage/documents/__init__.py
touch "${PROJECT_ROOT}"/app/middleware/__init__.py
touch "${PROJECT_ROOT}"/app/utils/__init__.py
touch "${PROJECT_ROOT}"/tests/__init__.py
touch "${PROJECT_ROOT}"/tests/unit/__init__.py
touch "${PROJECT_ROOT}"/tests/integration/__init__.py

echo "✅ Scaffold created successfully at: ${PROJECT_ROOT}"

# Print structure (tree if available, otherwise fallback to find)
if command -v tree >/dev/null 2>&1; then
  tree -a "${PROJECT_ROOT}"
else
  echo "tree command not found. Showing files with find:"
  find "${PROJECT_ROOT}" | sort
fi