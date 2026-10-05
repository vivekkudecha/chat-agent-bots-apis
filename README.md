# Multi-Bot AI Platform API (`chat-agent-bots-apis`)

A production-grade, enterprise-ready multi-bot AI backend platform built with **FastAPI**, **PostgreSQL**, **Qdrant**, **Celery**, **Redis**, and **vLLM / Ollama**.

This service provides end-to-end capabilities for creating customizable AI bots, versioning configurations, managing vector-indexed knowledge bases, automated document processing (PDF, DOCX, TXT, OCR), multi-stage guardrails (PII, toxicity, prompt injection, secrets), and conversation management with Retrieval-Augmented Generation (RAG).

---

## Table of Contents

- [System Architecture](#system-architecture)
  - [Architecture Diagram](#architecture-diagram)
  - [Core Components & Data Flow](#core-components--data-flow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Quick Start Guide](#quick-start-guide)
  - [Setup for Linux / macOS](#setup-for-linux--macos)
  - [Setup for Windows](#setup-for-windows)
- [Database Migrations (Alembic)](#database-migrations-alembic)
  - [Applying Migrations](#applying-migrations)
  - [Creating New Migrations](#creating-new-migrations)
  - [Rolling Back Migrations](#rolling-back-migrations)
- [Running the Services](#running-the-services)
  - [1. FastAPI Server](#1-fastapi-server)
  - [2. Celery Worker (Background Tasks)](#2-celery-worker-background-tasks)
- [Interactive API Documentation](#interactive-api-documentation)
- [Testing & Postman Collection](#testing--postman-collection)
- [Troubleshooting & Gotchas](#troubleshooting--gotchas)

---

## System Architecture

### Architecture Diagram

```mermaid
flowchart TD
    subgraph Clients["Client Layer"]
        UI["Web UI / Frontend (React / Vite)"]
        PM["Postman / API Consumers"]
    end

    subgraph Gateway["API & Security Layer (FastAPI)"]
        CORS["CORS Middleware"]
        Auth["JWT Authentication & RBAC"]
        Router["API v1 Router"]
        Handlers["Exception Handlers"]
    end

    subgraph Services["Core Business Logic Services"]
        ChatSvc["Chat Service (RAG)"]
        BotSvc["Bot & Versioning Service"]
        KBSvc["Knowledge Base Service"]
        DocSvc["Document Service"]
        GuardSvc["Guardrails Engine (Input/Output)"]
        ModelSvc["AI Model Service"]
    end

    subgraph AsyncWorker["Background Workers (Celery & Redis)"]
        Redis[("Redis Broker & Result Backend")]
        CeleryWorker["Celery Worker"]
        DocTasks["Document Ingestion Pipeline
        - PyMuPDF / Tesseract OCR
        - Semantic Chunking
        - Vector Embeddings"]
    end

    subgraph DataStorage["Persistence & Vector Stores"]
        Postgres[("PostgreSQL
        (Users, Bots, KBs, Docs, Chats, Audit)")]
        Qdrant[("Qdrant Vector Database
        (Knowledge Chunks & Embeddings)")]
        FileStore[("File Storage
        (Local ./storage or S3)")]
    end

    subgraph AIInference["AI Inference & Embedding Models"]
        LLM["LLM Providers (vLLM / Ollama / OpenAI)"]
        Embedder["Embedding Providers (Local BGE / Ollama)"]
    end

    %% Client Interactions
    UI --> CORS
    PM --> CORS
    CORS --> Auth
    Auth --> Router
    Router --> Handlers

    %% Router to Services
    Router --> ChatSvc
    Router --> BotSvc
    Router --> KBSvc
    Router --> DocSvc
    Router --> GuardSvc
    Router --> ModelSvc

    %% Service Operations
    BotSvc --> Postgres
    ModelSvc --> Postgres
    GuardSvc --> Postgres
    ChatSvc --> GuardSvc
    ChatSvc --> Qdrant
    ChatSvc --> LLM
    ChatSvc --> Postgres

    %% Document Ingestion Flow
    DocSvc --> FileStore
    DocSvc --> Redis
    Redis --> CeleryWorker
    CeleryWorker --> DocTasks
    DocTasks --> FileStore
    DocTasks --> Embedder
    DocTasks --> Qdrant
    DocTasks --> Postgres
```

### Core Components & Data Flow

1. **FastAPI Application Gateway (`app/main.py`)**:
   - Manages application lifecycle via asynchronous lifespan context.
   - Automatically initializes Qdrant collections and vector indexes on startup.
   - Enforces CORS policies and routes all incoming HTTP traffic through `/api/v1`.

2. **Relational Database (`PostgreSQL` via `SQLAlchemy 2.x` + `asyncpg`)**:
   - Stores users, bots, bot versions, attached knowledge bases, document metadata, guardrail configurations, chat conversations, messages, and token usage analytics.
   - Managed via **Alembic** asynchronous migrations.

3. **Vector Database (`Qdrant`)**:
   - Stores dense high-dimensional vector embeddings generated from chunked knowledge base documents.
   - Performs cosine similarity searches to retrieve context during the RAG chat pipeline.

4. **Async Task Queue (`Celery` + `Redis`)**:
   - Offloads compute-heavy document processing from web threads.
   - Executes text extraction (via PyMuPDF), OCR fallback for scanned pages (via Tesseract), text chunking, and embedding generation asynchronously.

5. **AI Inference & Embeddings (`vLLM` / `Ollama` / `OpenAI`)**:
   - Modular integration supporting OpenAI-compatible LLM endpoints (e.g., local vLLM instances or cloud APIs) and embedding models (e.g., Ollama `qwen3-embedding`, BAAI `bge-base-en-v1.5`, or HuggingFace transformers).

6. **Multi-Stage Guardrails Engine (`app/guardrails/`)**:
   - Evaluates input prompts and generated bot responses against pluggable guardrails (secrets detection, PII masking, toxicity analysis, prompt injection defense).

---

## Project Structure

```
chat-agent-bots-apis/
├── alembic.ini                   # Alembic database migration configuration
├── pyproject.toml                # Project metadata and dependencies (uv / pip)
├── requirements.txt              # Pinned Python package dependencies
├── .env.example                  # Template for environment configuration
├── .env                          # Local environment variables (do not commit)
├── README.md                     # Project documentation
│
├── app/                          # Core application package
│   ├── alembic/                  # Database migration scripts & history
│   │   ├── env.py                # Alembic runtime environment (Async SQLAlchemy)
│   │   ├── script.py.mako        # Migration template
│   │   └── versions/             # Versioned schema migration files
│   │
│   ├── api/                      # REST API Layer (API endpoints & routes only)
│   │   └── v1/                   # API Version 1 endpoints
│   │       ├── auth.py           # User registration, login, token refresh
│   │       ├── users.py          # User profile management & password update
│   │       ├── bots.py           # Bot CRUD, version publishing, KB association
│   │       ├── models.py         # AI Model CRUD & bot model configurations
│   │       ├── knowledge_bases.py# Knowledge base management
│   │       ├── documents.py      # Document upload, listing, status & retry
│   │       ├── guardrails.py     # Guardrail definitions, bot attachment & testing
│   │       ├── conversations.py  # Conversation sessions & message history
│   │       ├── chat.py           # Chat completion & response generation
│   │       └── router.py         # Consolidated API v1 router
│   │
│   ├── ai/                       # AI Layer (LangGraph, RAG, LLM, Guardrails)
│   │   ├── agent/                # LangGraph StateGraph, router, state & tools
│   │   │   ├── graph.py          # Compiled StateGraph chat workflow
│   │   │   ├── router.py         # Adaptive intent router (Direct vs RAG vs Tool)
│   │   │   ├── state.py          # Agent state definitions & RouteType
│   │   │   └── tools.py          # Bot tool registry & schema serializer
│   │   ├── rag/                  # RAG Subsystem
│   │   │   ├── retrieval.py      # Semantic vector retrieval & distinct source grouping
│   │   │   ├── vector_store.py   # Qdrant vector store management
│   │   │   ├── chunking.py       # Document chunking service
│   │   │   └── text_extraction.py# Text extraction & OCR (PyMuPDF / Tesseract)
│   │   ├── llm/                  # LLM Subsystem
│   │   │   ├── provider.py       # LLM provider interface & OpenAI-compatible client
│   │   │   ├── embeddings.py     # Embedding providers (Ollama / SentenceTransformers)
│   │   │   └── prompt_builder.py # Context prompt builder & security policy formatter
│   │   └── guardrails/           # Multi-Stage Guardrails Engine
│   │       ├── base.py           # Base guardrail interfaces & stages (Input/Output/Retrieval)
│   │       ├── registry.py       # Guardrail registry & runner
│   │       ├── service.py        # Guardrail execution service
│   │       └── implementations/  # PII, secrets, toxicity, prompt injection guardrails
│   │
│   ├── core/                     # Application core utilities
│   │   ├── dependencies.py       # FastAPI dependency injection (auth, db session)
│   │   ├── exception_handlers.py # Global HTTP & validation exception handlers
│   │   ├── exceptions.py         # Domain-specific custom exceptions
│   │   └── security.py           # Password hashing (Argon2), JWT token handling
│   │
│   ├── database.py               # Async SQLAlchemy engine & session factory
│   ├── config.py                 # Pydantic BaseSettings application configuration
│   │
│   ├── integrations/             # External service clients & infrastructure adapters
│   │   ├── qdrant.py             # Qdrant client connection & lifecycle management
│   │   └── storage.py            # Local & S3 file storage providers
│   │
│   ├── models/                   # SQLAlchemy declarative ORM models
│   ├── repositories/             # Data access repository layer
│   ├── schemas/                  # Pydantic request/response validation schemas
│   │
│   ├── services/                 # Application Business Logic Services
│   │   ├── auth_service.py       # Authentication & credential verification
│   │   ├── bot_service.py        # Bot lifecycle, versioning & orchestration
│   │   ├── chat_service.py       # Chat orchestration (invokes AI LangGraph workflow)
│   │   ├── document_service.py   # Document upload & ingestion pipeline triggering
│   │   ├── knowledge_base_service.py # Knowledge base collection logic
│   │   └── user_service.py       # User profile operations
│   │
│   ├── workers/                  # Celery background workers & asynchronous tasks
│   │   ├── celery_app.py         # Celery instance configuration & Redis broker
│   │   └── document_tasks.py     # Async document processing, extraction & embedding
│   │
│   └── main.py                   # FastAPI application initialization & lifespan
│
├── data/                         # OCR training data (tessdata/eng.traineddata)
├── postman/                      # Postman collection & environment for API testing
├── scripts/                      # Utility scripts (e.g. Postman collection generator)
└── storage/                      # Local file storage repository for uploaded docs
```

---

## Prerequisites

Before starting, ensure you have the following installed on your machine:

1. **Python 3.11+** (Python 3.11, 3.12, or 3.14 compatible).
2. **PostgreSQL 15+** (Relational Database).
3. **Redis 7+** (Celery Broker & Result Store).
4. **Qdrant** (Vector Database).
5. **Docker** *(Optional, recommended for quickly spinning up external dependencies)*.
6. **Ollama or vLLM** *(Optional for local embeddings / LLM inference)*.

### Quick Start with Docker (External Services)

If you have Docker installed, you can spin up PostgreSQL, Redis, and Qdrant with the following commands:

#### Linux / macOS:
```bash
# PostgreSQL
docker run -d --name ai_bot_postgres \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=admin \
  -e POSTGRES_DB=ai_chat_bot \
  -p 5432:5432 \
  postgres:16-alpine

# Redis
docker run -d --name ai_bot_redis \
  -p 6379:6379 \
  redis:7-alpine

# Qdrant Vector Store
docker run -d --name ai_bot_qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/storage/qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

#### Windows (PowerShell):
```powershell
# PostgreSQL
docker run -d --name ai_bot_postgres `
  -e POSTGRES_USER=admin `
  -e POSTGRES_PASSWORD=admin `
  -e POSTGRES_DB=ai_chat_bot `
  -p 5432:5432 `
  postgres:16-alpine

# Redis
docker run -d --name ai_bot_redis `
  -p 6379:6379 `
  redis:7-alpine

# Qdrant Vector Store
docker run -d --name ai_bot_qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v ${PWD}/storage/qdrant_data:/qdrant/storage `
  qdrant/qdrant:latest
```

---

## Environment Configuration

Create a `.env` file in the project root by copying `.env.example`:

### Linux / macOS:
```bash
cp .env.example .env
```

### Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

### Key Environment Variables

Review and adjust `.env` according to your environment:

```ini
APP_NAME="AI Bot Platform"
APP_VERSION="1.0.0"
DEBUG=true

# PostgreSQL Connection
DATABASE_URL=postgresql+asyncpg://admin:admin@localhost:5432/ai_chat_bot
DB_ECHO=false

# Qdrant Vector Store
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=knowledge_chunks

# Embedding Configuration
EMBEDDING_PROVIDER=ollama               # options: 'ollama' | 'local'
EMBEDDING_MODEL=qwen3-embedding:0.6b    # e.g., 'BAAI/bge-base-en-v1.5' or 'qwen3-embedding:0.6b'
EMBEDDING_DIMENSION=1024                # Must match model dimensions (768 for BGE, 1024 for Qwen)
OLLAMA_BASE_URL=http://localhost:11434

# LLM / vLLM Provider
LLM_PROVIDER=vllm                       # options: 'vllm' | 'openai'
VLLM_BASE_URL=http://localhost:11434/v1
VLLM_API_KEY=local-vllm
VLLM_DEFAULT_MODEL=llama3.2:latest
VLLM_TIMEOUT=120

# Redis & Celery
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# File Storage & OCR
STORAGE_PROVIDER=local
LOCAL_STORAGE_PATH=./storage
MAX_UPLOAD_SIZE_MB=200
TESSDATA_PREFIX=./data/tessdata
OCR_LANGUAGE=eng
OCR_DPI=150

# Authentication (Change in production!)
JWT_SECRET_KEY=CHANGE_THIS_TO_A_LONG_RANDOM_SECRET
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# RAG & Guardrails Defaults
DEFAULT_CHUNK_SIZE=800
DEFAULT_CHUNK_OVERLAP=100
DEFAULT_TOP_K=5
ENABLE_INPUT_GUARDRAILS=true
ENABLE_OUTPUT_GUARDRAILS=true
```

---

## Quick Start Guide

### Setup for Linux / macOS

```bash
# 1. Clone repository & enter directory
cd /path/to/chat-agent-bots-apis

# 2. Create Python virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Or if you use 'uv' (ultrafast package manager):
# uv venv
# source .venv/bin/activate
# uv pip install -r requirements.txt

# 5. Setup environment configuration
cp .env.example .env
# Edit .env with your PostgreSQL, Redis, and Qdrant credentials

# 6. Apply database migrations
alembic upgrade head

# 7. Start the FastAPI development server
uvicorn app.main:app --reload --port 8000
```

---

### Setup for Windows

Open **PowerShell** (Run as Administrator if setting execution policy for scripts is required):

```powershell
# 1. Enter project directory
cd C:\path\to\chat-agent-bots-apis

# 2. Allow local script execution (if not previously configured)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Create Python virtual environment
python -m venv .venv

# 4. Activate virtual environment
.venv\Scripts\Activate.ps1

# (For classic Command Prompt cmd.exe instead of PowerShell, use:
#  .venv\Scripts\activate.bat)

# 5. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 6. Create environment file
Copy-Item .env.example .env
# Open .env in your text editor (e.g. notepad .env) and verify credentials

# 7. Apply database migrations
alembic upgrade head

# 8. Start the FastAPI development server
uvicorn app.main:app --reload --port 8000
```

---

## Database Migrations (Alembic)

The application uses **Alembic** with asynchronous SQLAlchemy (`asyncpg`) to handle database schema revisions.

Configuration details:
- Alembic config file: [`alembic.ini`](file:///Users/vivek/tatatel/chat-agent-bots-apis/alembic.ini)
- Migration environment: [`app/alembic/env.py`](file:///Users/vivek/tatatel/chat-agent-bots-apis/app/alembic/env.py)
- Versions folder: [`app/alembic/versions/`](file:///Users/vivek/tatatel/chat-agent-bots-apis/app/alembic/versions)

### Applying Migrations

Make sure your virtual environment is active and PostgreSQL is running:

#### Linux / macOS:
```bash
alembic upgrade head
```

#### Windows:
```powershell
alembic upgrade head
```

### Creating New Migrations

Whenever you add or modify SQLAlchemy models in [`app/models/`](file:///Users/vivek/tatatel/chat-agent-bots-apis/app/models), generate a new auto-detected migration:

#### Linux / macOS / Windows:
```bash
alembic revision --autogenerate -m "describe_your_changes"
```
Review the generated file in `app/alembic/versions/` and then run `alembic upgrade head`.

### Rolling Back Migrations

To downgrade by one revision:
```bash
alembic downgrade -1
```

To check current migration revision:
```bash
alembic current
```

---

## Running the Services

To run the full stack locally, you need two terminal processes:

### 1. FastAPI Server

Runs the web API, OpenAPI docs, and synchronous endpoints.

#### Linux / macOS:
```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Celery Worker (Background Tasks)

Processes asynchronous tasks: document parsing, OCR with PyMuPDF/Tesseract, semantic chunking, and Qdrant vector embedding generation.

#### Linux / macOS:
```bash
source .venv/bin/activate
celery -A app.workers.celery_app.celery_app worker --loglevel=INFO
```

#### Windows (PowerShell):
> [!IMPORTANT]
> **Windows Celery Note**: Celery's default `prefork` execution pool does **not** work on Windows. You **must** specify `-P solo` (or `-P threads` / `-P gevent`) on Windows:

```powershell
.venv\Scripts\Activate.ps1
celery -A app.workers.celery_app.celery_app worker --loglevel=INFO -P solo
```

---

## Interactive API Documentation

Once the server is running, you can explore and test the endpoints directly in your browser:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc UI**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI JSON**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## Testing & Postman Collection

A complete, production-grade Postman collection covering all 52 API endpoints with automated variable extraction is provided in the [`postman/`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman) directory:

- [`chat-agent-bots-api.postman_collection.json`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman/chat-agent-bots-api.postman_collection.json)
- [`chat-agent-bots-local.postman_environment.json`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman/chat-agent-bots-local.postman_environment.json)

For detailed testing workflow and automated Newman CLI instructions, refer to [`postman/README.md`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman/README.md).

---

## Troubleshooting & Gotchas

1. **`asyncpg.exceptions.InvalidPasswordError` / Connection Refused**:
   - Check that PostgreSQL is running and matches `DATABASE_URL` in `.env`.
   - Ensure the database specified in `.env` exists (`createdb ai_chat_bot` or via SQL: `CREATE DATABASE ai_chat_bot;`).

2. **Celery Fails on Windows (`ValueError: not enough values to unpack` / permission error)**:
   - On Windows, always pass `-P solo` to the Celery worker command:
     ```powershell
     celery -A app.workers.celery_app.celery_app worker --loglevel=INFO -P solo
     ```

3. **Qdrant Vector Dimension Mismatch (`Vector dimension error`)**:
   - `EMBEDDING_DIMENSION` in `.env` must match the model's actual vector dimension:
     - `BAAI/bge-base-en-v1.5`: `768`
     - `qwen3-embedding:0.6b`: `1024`
     - `text-embedding-3-small`: `1536`

4. **CORS Issues from Frontend**:
   - CORS is configured in [`app/main.py`](file:///Users/vivek/tatatel/chat-agent-bots-apis/app/main.py#L98-L107) with `allow_origins=["*"]` and `allow_credentials=True`, allowing connections from local Vite dev servers (`http://localhost:5173`) or production domains.
