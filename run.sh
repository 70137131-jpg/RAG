#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# One-command launcher for the RAG Chatbot.
#   ./run.sh              # install deps (if needed), ingest, serve
#   SKIP_INGEST=1 ./run.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
SAMPLES="${SAMPLES:-100}"
CONFIG="${CONFIG:-balanced}"
PORT="${PORT:-5000}"

echo "▶ Using interpreter: $($PYTHON --version)"

if [ ! -f .env ] && [ -f .env.example ]; then
  echo "▶ No .env found — copying .env.example (offline mode until you add a key)"
  cp .env.example .env
fi

echo "▶ Installing dependencies…"
$PYTHON -m pip install -q -r requirements.txt

if [ "${SKIP_INGEST:-0}" != "1" ]; then
  echo "▶ Ingesting SQuAD data (config=$CONFIG, samples=$SAMPLES)…"
  $PYTHON ingest.py --config "$CONFIG" --samples "$SAMPLES" --yes
fi

echo "▶ Starting server on http://localhost:$PORT"
PORT="$PORT" $PYTHON app.py
