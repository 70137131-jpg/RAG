"""
Environment bootstrap — MUST be imported before transformers / sentence-transformers.

Some environments have a stray TensorFlow install. When `transformers` sees it, it
tries to import TensorFlow, whose native `preload_check` can deadlock on macOS
(surfacing as a hung process and an "[mutex.cc] RAW: Lock blocking" message).

Forcing the PyTorch backend and disabling the TF/Flax backends avoids that import
entirely and cuts cold-start time dramatically. Importing this module first sets the
relevant environment variables before any transformers code runs.
"""

import os

# Prefer PyTorch, never import TensorFlow or Flax from within transformers.
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# Quiet down native logging / telemetry.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB telemetry off
