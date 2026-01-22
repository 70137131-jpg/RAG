"""
Quick Start Script
Fastest way to get the RAG system running
"""

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
        print("  pip install -r requirements.txt")
        return False

    print("\n✓ All requirements satisfied!\n")
    return True


def check_api_key():
    """Check if API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("❌ API key not found!")
        print("\nPlease set your API key:")
        print("  1. Copy .env.example to .env")
        print("  2. Edit .env and add OPENROUTER_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY")
        print("\nOr set it as an environment variable:")
        print("  export OPENROUTER_API_KEY='your-key-here'")
        return False

    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key format looks incorrect")
        print("   OpenAI keys usually start with 'sk-'")

    print("✓ API key found\n")
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
        print("  - Run 'python demo.py' for interactive mode")
        print("  - Run 'python evaluate.py' to evaluate on more samples")
        print("  - Check README.md for full documentation")
        print()

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  - Make sure you have internet connection")
        print("  - Check your API key (OPENROUTER_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY)")
        print("  - Try: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    run_simple_demo()
