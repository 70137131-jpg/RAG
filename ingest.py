"""
Data Ingestion Script
Load SQuAD data and populate the vector store
"""

from data_loader import SQuADLoader
from vector_store import VectorStore
from config import RAGConfig
from config_utils import create_vector_store_from_config, create_data_loader_from_config
import argparse


def main():
    parser = argparse.ArgumentParser(description="Ingest data into vector store")
    parser.add_argument("--config", type=str, choices=["default", "fast", "balanced", "accurate", "gemini"],
                        default="balanced", help="Configuration profile")
    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to ingest (default: from config)")
    parser.add_argument("--collection", type=str, default=None,
                        help="Collection name (overrides config)")
    args = parser.parse_args()

    print("="*60)
    print("RAG DATA INGESTION")
    print("="*60)

    if args.config == "default":
        config = RAGConfig.default()
    elif args.config == "fast":
        from config import FAST_CONFIG
        config = FAST_CONFIG
    elif args.config == "accurate":
        from config import ACCURATE_CONFIG
        config = ACCURATE_CONFIG
    elif args.config == "gemini":
        from config import GEMINI_CONFIG
        config = GEMINI_CONFIG
    else:
        from config import BALANCED_CONFIG
        config = BALANCED_CONFIG

    if args.samples:
        config.data.max_samples = args.samples

    if args.collection:
        config.vector_store.collection_name = args.collection

    print(f"\nConfiguration: {args.config}")
    print(f"Collection: {config.vector_store.collection_name}")
    print(f"Embedding Model: {config.vector_store.embedding_model}")

    loader = create_data_loader_from_config(config)
    contexts = loader.get_contexts()

    print(f"\nLoaded {len(contexts)} unique contexts")

    vector_store = create_vector_store_from_config(config.vector_store)

    doc_count = vector_store.collection.count()
    if doc_count > 0:
        print(f"Collection already has {doc_count} documents")
        response = input("Reset collection? (y/n): ")
        if response.lower() == 'y':
            vector_store.reset()
        else:
            print("Adding documents to existing collection...")
            vector_store.add_documents(contexts)
            print(f"Total documents: {vector_store.collection.count()}")
            return

    print("\nAdding documents to vector store...")
    vector_store.add_documents(contexts)

    print(f"\n✓ Successfully ingested {len(contexts)} documents")
    print(f"  Collection: {config.vector_store.collection_name}")
    print(f"  Total documents: {vector_store.collection.count()}")


if __name__ == "__main__":
    main()
