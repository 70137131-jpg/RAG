"""
Quick Start Script
Fastest way to get the RAG system running
"""

import _env_setup  # noqa: F401  (must be first: forces torch backend, disables TensorFlow)
import os
import sys


def check_requirements():
    """Check if all requirements are installed"""
    print("Checking requirements...")

    required_packages = [
        'sentence_transformers',
        'chromadb',
        'datasets',
        'openai',
        'google.generativeai',
        'dotenv'
    ]

    missing = []

    for package in required_packages:
        try:
            __import__(package if package != 'dotenv' else 'dotenv')
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print("  python3 -m pip install -r requirements.txt")
        return False

    print("\n✓ All requirements satisfied!\n")
    return True


def check_api_key():
    """Check if an LLM API key is configured (optional — offline mode works without one)."""
    from dotenv import load_dotenv
    load_dotenv()

    google_key = os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    if google_key:
        print("✓ Gemini API key found (LLM generation enabled)\n")
    elif openai_key:
        print("✓ OpenAI-compatible API key found (LLM generation enabled)\n")
    else:
        print("ℹ️  No LLM API key found — running in OFFLINE extractive mode.")
        print("   Answers are drawn directly from retrieved contexts (no external calls).")
        print("   To enable LLM-generated answers, copy .env.example to .env and add a key.\n")

    # Always return True: the pipeline degrades gracefully to offline mode.
    return True


def run_simple_demo():
    """Run a simple demonstration"""
    print("="*70)
    print(" " * 20 + "RAG QUICKSTART")
    print("="*70 + "\n")

    if not check_requirements():
        sys.exit(1)

    if not check_api_key():
        sys.exit(1)

    print("Starting RAG demo...\n")
    print("This will:")
    print("  1. Load 50 samples from SQuAD dataset")
    print("  2. Build a vector database")
    print("  3. Run 3 example queries")
    print("\n" + "-"*70 + "\n")

    try:
        from data_loader import SQuADLoader
        from vector_store import VectorStore
        from rag_pipeline import RAGPipeline

        # Load dataset
        print("[1/4] Loading SQuAD dataset...")
        loader = SQuADLoader(dataset_name="squad_v2", split="validation")
        loader.load(max_samples=50)
        contexts = loader.get_contexts()
        qa_pairs = loader.get_qa_pairs()
        print(f"      Loaded {len(contexts)} contexts\n")

        # Initialize vector store
        print("[2/4] Building vector database...")
        vector_store = VectorStore(collection_name="quickstart_demo")

        if vector_store.collection.count() == 0:
            vector_store.add_documents(contexts)

        print(f"      Indexed {vector_store.collection.count()} documents\n")

        # Initialize RAG
        print("[3/4] Initializing RAG pipeline...\n")
        rag = RAGPipeline(vector_store=vector_store)

        # Run example queries
        print("[4/4] Running example queries...")
        print("\n" + "="*70)

        example_questions = [
            qa_pairs[0]['question'],
            qa_pairs[1]['question'] if len(qa_pairs) > 1 else "What is AI?",
            qa_pairs[2]['question'] if len(qa_pairs) > 2 else "What is machine learning?"
        ]

        for i, question in enumerate(example_questions, 1):
            print(f"\nQuery {i}: {question}")
            print("-" * 70)

            result = rag.query(question, top_k=2, return_metadata=True)

            print(f"\n💡 Answer:\n{result['answer']}\n")

            # Show retrieval info
            retrieved = result['metadata']['retrieved_docs']
            print(f"📚 Retrieved {len(retrieved)} contexts:")
            for j, doc in enumerate(retrieved, 1):
                similarity = 1 - doc['distance']
                print(f"  [{j}] Similarity: {similarity:.2%}")

            print("\n" + "="*70)

        print("\n✅ Quickstart demo completed successfully!")
        print("\nNext steps:")
        print("  - Run 'python3 ingest.py --config balanced --samples 100 --yes'")
        print("  - Run 'python3 app.py' for the web interface")
        print("  - Run 'python3 evaluate.py' to evaluate on more samples")
        print("  - Check README.md for full documentation")
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Make sure you have internet connection")
        print("  - Add an LLM key only if you want generated answers instead of offline extractive answers")
        print("  - Try: python3 -m pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    run_simple_demo()
