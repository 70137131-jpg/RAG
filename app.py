import _env_setup  # noqa: F401  (must be first: forces torch backend, disables TensorFlow)
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from dotenv import load_dotenv

# Load environment variables BEFORE importing other modules
load_dotenv()

# NOTE: The heavy RAG imports (rag_pipeline / vector_store / config_utils) pull in
# torch + chromadb, which are imported lazily inside initialize_rag() instead of at
# module load. This keeps `import app` fast and lets the web UI start even when the
# RAG backend is unavailable (e.g. UI-preview mode).
import hmac
import os
import json
import logging
import time
from datetime import datetime
import uuid

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _int_env(name, default, min_value=None, max_value=None):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw_value, default)
        return default

    if min_value is not None and value < min_value:
        logger.warning("%s=%s is below minimum %s; using default %s", name, value, min_value, default)
        return default
    if max_value is not None and value > max_value:
        logger.warning("%s=%s is above maximum %s; using default %s", name, value, max_value, default)
        return default
    return value


def _bool_env(name, default=False):
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


REQUEST_MAX_BYTES = _int_env("REQUEST_MAX_BYTES", 16 * 1024, min_value=1024)
MAX_QUESTION_LENGTH = _int_env("MAX_QUESTION_LENGTH", 2000, min_value=1)
RATE_LIMIT_REQUESTS = _int_env("RATE_LIMIT_REQUESTS", 60, min_value=0)
RATE_LIMIT_WINDOW_SECONDS = _int_env("RATE_LIMIT_WINDOW_SECONDS", 60, min_value=1)

app = Flask(__name__, static_folder='static', static_url_path='/static', template_folder='templates')
app.config.update(
    MAX_CONTENT_LENGTH=REQUEST_MAX_BYTES,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=os.getenv("SESSION_COOKIE_SAMESITE", "Lax"),
    SESSION_COOKIE_SECURE=_bool_env("SESSION_COOKIE_SECURE", False),
)

# Session secret. Prefer an explicit SECRET_KEY in production so sessions survive
# restarts and multiple workers share the same key. Fall back to a random key for
# local development (sessions reset on restart, which is fine for a demo).
_secret_key = os.getenv('SECRET_KEY')
if not _secret_key:
    _secret_key = os.urandom(32).hex()
    logger.warning(
        "SECRET_KEY not set — generated an ephemeral key for this run. "
        "Set SECRET_KEY in the environment for production/multi-worker deployments."
    )
app.secret_key = _secret_key

# Global RAG instance (initialized on first request)
rag_system = None
config = None
chat_history = {}
rate_limit_state = {}
metrics = {
    'total_queries': 0,
    'total_sessions': 0,
    'avg_response_time': 0,
    'response_times': []
}


def error_response(message, status_code):
    return jsonify({
        'success': False,
        'error': message
    }), status_code


def admin_authorized():
    """Require an admin token only when ADMIN_TOKEN is configured."""
    expected_token = os.getenv("ADMIN_TOKEN")
    if not expected_token:
        return True

    provided_token = request.headers.get("X-Admin-Token", "")
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        provided_token = auth_header[7:].strip()

    return bool(provided_token) and hmac.compare_digest(provided_token, expected_token)


def require_admin():
    if admin_authorized():
        return None
    return error_response("Unauthorized", 401)


def rate_limit_key():
    session_id = session.get('session_id')
    if session_id:
        return f"session:{session_id}"

    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        client_ip = forwarded_for.split(",", 1)[0].strip()
    else:
        client_ip = request.remote_addr or "unknown"
    return f"ip:{client_ip}"


def check_rate_limit():
    if RATE_LIMIT_REQUESTS <= 0:
        return None

    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS
    key = rate_limit_key()
    timestamps = [ts for ts in rate_limit_state.get(key, []) if ts > window_start]

    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])))
        rate_limit_state[key] = timestamps
        return retry_after

    timestamps.append(now)
    rate_limit_state[key] = timestamps
    return None


@app.errorhandler(413)
def request_too_large(_error):
    return error_response("Request payload too large", 413)


def initialize_rag():
    """Initialize the RAG system (loads prebuilt index)"""
    global rag_system, config

    if rag_system is not None:
        return rag_system

    # UI-preview mode: serve the interface without booting the (heavy) RAG backend.
    if os.getenv('UI_PREVIEW') == '1':
        raise RuntimeError(
            "RAG backend is disabled (UI preview mode). "
            "Restart without UI_PREVIEW=1 — and after running 'python ingest.py' and "
            "setting an LLM API key in .env — to enable live queries."
        )

    # Deferred heavy imports (torch + chromadb) — see note at top of file.
    from config import RAGConfig
    from config_utils import create_vector_store_from_config, create_rag_pipeline_from_config

    logger.info("Initializing RAG system...")
    start_time = time.time()

    # Load configuration
    config = RAGConfig.from_env()
    logger.info(f"Configuration loaded: {config.vector_store.collection_name}")

    # Initialize vector store (load-only mode)
    vs = create_vector_store_from_config(config.vector_store)

    # Check if index exists
    doc_count = vs.collection.count()
    if doc_count == 0:
        logger.error("No documents found in vector store!")
        logger.error(f"Please run: python ingest.py")
        raise RuntimeError(
            f"Vector store '{config.vector_store.collection_name}' is empty. "
            f"Run 'python ingest.py' to build the index first."
        )

    logger.info(f"Vector store loaded with {doc_count} documents")

    # Initialize RAG pipeline
    rag_system = create_rag_pipeline_from_config(config, vs)

    elapsed = time.time() - start_time
    logger.info(f"RAG system ready in {elapsed:.2f}s")
    return rag_system


@app.route('/')
def index():
    """Render the chat interface"""
    # Generate a session ID if not exists
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        chat_history[session['session_id']] = []

    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle chat queries with logging and metrics"""
    start_time = time.time()

    try:
        data = request.get_json(silent=True) or {}
        question = str(data.get('question', '')).strip()
        try:
            top_k = int(data.get('top_k', config.top_k if config else 3))
        except (TypeError, ValueError):
            top_k = config.top_k if config else 3
        top_k = max(1, min(top_k, 10))  # clamp to a sane range

        if not question:
            logger.warning("Empty question received")
            return error_response('Question cannot be empty', 400)

        if len(question) > MAX_QUESTION_LENGTH:
            logger.warning("Question too long: %s characters", len(question))
            return error_response(f"Question exceeds {MAX_QUESTION_LENGTH} characters", 400)

        retry_after = check_rate_limit()
        if retry_after is not None:
            logger.warning("Rate limit exceeded for %s", rate_limit_key())
            response, status_code = error_response("Rate limit exceeded", 429)
            response.headers["Retry-After"] = str(retry_after)
            return response, status_code

        # Initialize RAG if needed
        rag = initialize_rag()

        # Get session ID
        session_id = session.get('session_id', 'default')

        logger.info(f"Query from session {session_id[:8]}: {question[:100]}")

        # Query the RAG system
        result = rag.query(question, top_k=top_k, return_metadata=True)

        # Calculate response time
        response_time = time.time() - start_time

        # Update metrics
        metrics['total_queries'] += 1
        metrics['response_times'].append(response_time)
        metrics['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])

        # Format response
        response = {
            'success': True,
            'question': question,
            'answer': result['answer'],
            'sources': [
                {
                    'id': doc['id'],
                    'text': doc['text'][:300] + ('...' if len(doc['text']) > 300 else ''),
                    'similarity': round((1 - doc['distance']) * 100, 1),
                    'rerank_score': doc.get('rerank_score', None)
                }
                for doc in result['metadata']['retrieved_docs']
            ],
            'timestamp': datetime.now().isoformat(),
            'response_time_ms': round(response_time * 1000, 2)
        }

        # Store in chat history
        if session_id not in chat_history:
            chat_history[session_id] = []

        chat_history[session_id].append({
            'question': question,
            'answer': result['answer'],
            'timestamp': response['timestamp'],
            'response_time_ms': response['response_time_ms']
        })

        logger.info(f"Query processed in {response_time:.2f}s, retrieved {len(result['metadata']['retrieved_docs'])} docs")

        return jsonify(response)

    except Exception as e:
        response_time = time.time() - start_time
        logger.error(f"Error processing query (took {response_time:.2f}s): {str(e)}", exc_info=True)
        return error_response("Failed to process query", 500)


@app.route('/api/history', methods=['GET'])
def get_history():
    """Get chat history for current session"""
    session_id = session.get('session_id', 'default')
    history = chat_history.get(session_id, [])

    return jsonify({
        'success': True,
        'history': history
    })


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Clear chat history for current session"""
    session_id = session.get('session_id', 'default')

    if session_id in chat_history:
        chat_history[session_id] = []

    return jsonify({
        'success': True,
        'message': 'Chat history cleared'
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    admin_error = require_admin()
    if admin_error:
        return admin_error

    try:
        rag = initialize_rag()
        stats = rag.vector_store.get_stats()

        backend = getattr(rag, 'backend', 'unknown')
        if not isinstance(backend, str):
            backend = 'unknown'
        llm_model = config.llm.model if (config and backend != 'offline') else 'offline-extractive'

        return jsonify({
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'embedding_model': stats['embedding_model'],
                'llm_model': llm_model,
                'backend': backend,
                'total_sessions': len(chat_history),
                'total_queries': metrics['total_queries'],
                'avg_response_time_ms': round(metrics['avg_response_time'] * 1000, 2),
                'collection_name': stats['collection_name']
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return error_response("Failed to load stats", 500)


@app.route('/metrics')
def get_metrics():
    """Prometheus-style metrics endpoint"""
    admin_error = require_admin()
    if admin_error:
        return admin_error

    try:
        rag = initialize_rag()
        doc_count = rag.vector_store.collection.count()

        metrics_text = f"""# HELP rag_total_queries Total number of queries processed
# TYPE rag_total_queries counter
rag_total_queries {metrics['total_queries']}

# HELP rag_total_sessions Total number of user sessions
# TYPE rag_total_sessions gauge
rag_total_sessions {len(chat_history)}

# HELP rag_avg_response_time_seconds Average response time in seconds
# TYPE rag_avg_response_time_seconds gauge
rag_avg_response_time_seconds {metrics['avg_response_time']:.4f}

# HELP rag_documents_indexed Total number of documents in the index
# TYPE rag_documents_indexed gauge
rag_documents_indexed {doc_count}
"""
        return Response(metrics_text, mimetype='text/plain')
    except Exception as e:
        logger.error(f"Error generating metrics: {str(e)}", exc_info=True)
        return Response("# Error: metrics unavailable\n", mimetype='text/plain'), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        rag = initialize_rag()
        return jsonify({
            'status': 'healthy',
            'documents_indexed': rag.vector_store.collection.count()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'unhealthy',
            'error': 'RAG backend unavailable'
        }), 500


if __name__ == '__main__':
    # Initialize RAG system at startup (only for local development)
    # On Vercel, this happens on first request to reduce cold start time
    if not os.getenv('VERCEL'):
        try:
            initialize_rag()
        except Exception as e:
            logger.warning(f"Failed to initialize RAG on startup: {e}")
            logger.info("RAG will be initialized on first request")

    # Run the Flask app
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV') == 'development'

    print(f"\n{'='*60}")
    print(f"RAG Chatbot Server Starting")
    print(f"{'='*60}")
    print(f"Access the chatbot at: http://localhost:{port}")
    print(f"{'='*60}\n")

    app.run(host='0.0.0.0', port=port, debug=debug)
