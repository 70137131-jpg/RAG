# RAG Project

Retrieval-augmented question answering over the SQuAD dataset with a Flask web UI, ChromaDB vector store, and a pluggable LLM backend — **Google Gemini** (default), any **OpenAI-compatible** provider (OpenAI, OpenRouter, DeepSeek), **or a fully-local offline extractive mode that needs no API key at all.**

## Features
- **Runs with zero API keys** — an offline extractive answerer draws grounded answers straight from retrieved contexts, so the app works end-to-end on localhost out of the box
- Drop in a Gemini or OpenAI-compatible key to enable full LLM generation; automatic graceful fallback to offline mode if a live call fails
- End-to-end RAG pipeline (retrieve + generate) with optional grounding checks
- Context compression to keep prompts within a target context size
- Token-aware chunking for better document splits
- Optional reranking via a cross-encoder (if available)
- Persisted ChromaDB vector store for fast startup
- Flask chat UI with history, stats modal, and typing indicator
- Operational endpoints: health check and Prometheus-style metrics

## Fastest Start
```bash
./run.sh          # installs deps, ingests 100 SQuAD samples, serves on :5000
```
Then open `http://localhost:5000`. No key required (offline mode). Add a key to `.env` for LLM answers.

## How It Works
1. Load SQuAD data and extract unique contexts
2. Embed and store contexts in ChromaDB
3. Retrieve top-k contexts for a question
4. Generate a grounded answer with citations

## Quick Start (manual)
1) Install dependencies
```bash
python3 -m pip install -r requirements.txt
```

2) (Optional) Configure a key for LLM answers — skip this to run offline
```bash
cp .env.example .env
# then edit .env and set ONE of:
#   GOOGLE_API_KEY=...        (Gemini — free at https://aistudio.google.com/apikey)
#   OPENAI_API_KEY=sk-...
#   OPENROUTER_API_KEY=sk-or-v1-...
#   DEEPSEEK_API_KEY=sk-...
```

3) Run the demo
```bash
python3 quickstart.py
```

## Ingest Data (Required for the Web App)
Build the vector index before starting the web server. `--yes` runs non-interactively.

```bash
python3 ingest.py --config balanced --samples 100 --yes
```

Other options:
```bash
python3 ingest.py --config fast --yes
python3 ingest.py --config accurate --yes
python3 ingest.py --config gemini --yes
python3 ingest.py --samples 50 --yes
python3 ingest.py --collection my_collection --yes
python3 ingest.py --reset            # rebuild an existing collection without prompting
```

## Run the Web App
```bash
python3 app.py
```
Then open `http://localhost:5000`.

## Evaluate
Run a small evaluation over SQuAD questions (uses your LLM key if set, else offline mode):
```bash
python3 evaluate.py --config balanced --max-samples 100
```

## API Endpoints
- `POST /api/query` - body: `{ "question": "...", "top_k": 3 }`
- `GET /api/stats` - model + collection stats
- `GET /api/history` - session chat history
- `POST /api/clear` - clear session history
- `GET /health` - basic health check
- `GET /metrics` - Prometheus-style metrics

## Configuration
You can control defaults via environment variables (see `.env.example`):
- `LLM_MODEL` (default: `gemini-2.5-flash`)
- `OPENAI_BASE_URL` (optional; for compatible providers)
- `TEMPERATURE`, `MAX_TOKENS`, `TOP_K`
- `COLLECTION_NAME`, `PERSIST_DIRECTORY`
- `EMBEDDING_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RERANKER_MODEL`
- `DATASET_NAME`, `DATASET_SPLIT`, `MAX_SAMPLES`
- `REQUEST_MAX_BYTES`, `MAX_QUESTION_LENGTH`
- `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW_SECONDS`
- `ADMIN_TOKEN` (protects `/api/stats` and `/metrics` when set)
- `SESSION_COOKIE_SECURE`, `SESSION_COOKIE_SAMESITE`

Runtime configuration is validated at startup. Invalid numeric values, empty
required strings, and invalid chunk settings fail fast with clear errors.

Prebuilt profiles are available via `ingest.py --config`:
- `fast`, `balanced`, `accurate`, `gemini`

## Project Structure
- `app.py` - Flask web server
- `rag_pipeline.py` - retrieval + generation logic
- `vector_store.py` - ChromaDB integration and chunking
- `data_loader.py` - SQuAD dataset loader
- `ingest.py` - build the vector store
- `evaluate.py` - retrieval + generation evaluation
- `config.py` / `config_utils.py` - configuration and helpers
- `templates/` and `static/` - UI assets

## Notes
- The web app will fail fast if the vector store is empty; run `ingest.py` first.
- With **no** LLM key set, the app runs in **offline extractive mode** (answers pulled from retrieved contexts). Add a key to enable full LLM generation.
- Gemini requires `google-generativeai` and `GOOGLE_API_KEY`. Default model is `gemini-2.5-flash` (the older `gemini-1.5-*` models have been retired).
- Dependencies are pinned in `requirements.txt` for reproducible installs.
- The vector store persists under `./chroma_db` by default.
