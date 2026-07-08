# How to Run the RAG Project

This project can run end-to-end with no LLM API key. Without a key, answers use
the local offline extractive fallback. Add one provider key to `.env` only if you
want LLM-generated answers.

## Prerequisites

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Optional: create `.env` from the example file:

```bash
cp .env.example .env
```

Then set one provider key if you want live LLM generation:

```bash
GOOGLE_API_KEY=your-google-key-here
# or OPENAI_API_KEY=sk-...
# or OPENROUTER_API_KEY=sk-or-v1-...
# or DEEPSEEK_API_KEY=sk-...
```

Default LLM model: `gemini-2.5-flash`.

## Fastest Start

```bash
./run.sh
```

This installs dependencies, ingests 100 SQuAD samples, and starts the web app on
`http://localhost:5000`.

Useful overrides:

```bash
SAMPLES=50 ./run.sh
CONFIG=fast ./run.sh
PORT=5050 ./run.sh
SKIP_INGEST=1 ./run.sh
PYTHON=/path/to/python3 ./run.sh
```

## Manual Workflow

Ingest data before starting the web app:

```bash
python3 ingest.py --config balanced --samples 100 --yes
```

Start the Flask app:

```bash
python3 app.py
```

Then open `http://localhost:5000`.

## Quick Demo

Run a small local demo that builds a temporary collection and asks example
questions:

```bash
python3 quickstart.py
```

## Configuration Profiles

- `fast`: smaller chunks, lower `top_k`, fastest local demo
- `balanced`: default profile
- `accurate`: larger embedding model and higher `top_k`
- `gemini`: Gemini-oriented defaults using `gemini-2.5-flash`

## Runtime Configuration

The app validates environment values at startup. Invalid values such as
`TOP_K=many`, `TEMPERATURE=3`, or `CHUNK_OVERLAP >= CHUNK_SIZE` fail fast with a
clear config error.

Important env vars:

- `LLM_MODEL`
- `TEMPERATURE`
- `MAX_TOKENS`
- `TOP_K`
- `COLLECTION_NAME`
- `PERSIST_DIRECTORY`
- `EMBEDDING_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `DATASET_NAME`
- `DATASET_SPLIT`
- `MAX_SAMPLES`
- `SECRET_KEY`
- `REQUEST_MAX_BYTES`
- `MAX_QUESTION_LENGTH`
- `RATE_LIMIT_REQUESTS`
- `RATE_LIMIT_WINDOW_SECONDS`
- `ADMIN_TOKEN`
- `SESSION_COOKIE_SECURE`
- `SESSION_COOKIE_SAMESITE`

## Troubleshooting

- Missing packages: run `python3 -m pip install -r requirements.txt`.
- Empty vector store: run `python3 ingest.py --config balanced --samples 100 --yes`.
- No API key: expected; the app uses offline extractive answers.
- LLM call failure: the pipeline falls back to offline extractive answers.
