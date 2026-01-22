from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from dotenv import load_dotenv

# Load environment variables BEFORE importing other modules
load_dotenv()

from rag_pipeline import RAGPipeline
from vector_store import VectorStore
from data_loader import SQuADLoader
from config import RAGConfig
from config_utils import create_vector_store_from_config, create_rag_pipeline_from_config
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

app = Flask(__name__, static_folder='static', static_url_path='/static', template_folder='templates')
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')

# Global RAG instance (initialized on first request)
rag_system = None
config = None
chat_history = {}
metrics = {
    'total_queries': 0,
    'total_sessions': 0,
    'avg_response_time': 0,
    'response_times': []
}


def initialize_rag():
    """Initialize the RAG system (loads prebuilt index)"""
    global rag_system, config

    if rag_system is not None:
        return rag_system

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
        data = request.json
        question = data.get('question', '').strip()
        top_k = data.get('top_k', config.top_k if config else 3)

        if not question:
            logger.warning("Empty question received")
            return jsonify({
                'success': False,
                'error': 'Question cannot be empty'
            }), 400

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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
    try:
        rag = initialize_rag()
        stats = rag.vector_store.get_stats()

        return jsonify({
            'success': True,
            'stats': {
                'total_documents': stats['total_documents'],
                'embedding_model': stats['embedding_model'],
                'llm_model': config.llm.model if config else 'unknown',
                'total_sessions': len(chat_history),
                'total_queries': metrics['total_queries'],
                'avg_response_time_ms': round(metrics['avg_response_time'] * 1000, 2),
                'collection_name': stats['collection_name']
            }
        })
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/metrics')
def get_metrics():
    """Prometheus-style metrics endpoint"""
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
        return Response(f"# Error: {str(e)}", mimetype='text/plain'), 500


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
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
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
