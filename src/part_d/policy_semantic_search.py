import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Input files
EMBEDDING_FILE = (
    BASE_DIR / "data" / "processed" /
    "refund_policy_embeddings.npy"
)

METADATA_FILE = (
    BASE_DIR / "data" / "processed" /
    "refund_policy_embedding_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"


def load_knowledge_base():

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError("Policy embeddings file not found.")

    if not METADATA_FILE.exists():
        raise FileNotFoundError("Policy metadata file not found.")

    embeddings = np.load(EMBEDDING_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata count mismatch.")

    return embeddings, metadata


def search_policy(question, model, embeddings, metadata, top_k=3):

    if not question.strip():
        print("Please enter a valid question.")
        return

    # Convert user question into an embedding
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Calculate cosine similarity
    scores = np.dot(embeddings, query_embedding)

    # Get highest matching results
    top_indices = np.argsort(scores)[::-1][:top_k]

    print("\n" + "=" * 60)
    print("RELEVANT REFUND POLICIES")
    print("=" * 60)

    for rank, index in enumerate(top_indices, start=1):

        policy = metadata[index]
        similarity = float(scores[index])

        print(f"\nResult {rank}")
        print("-" * 60)
        print(f"Policy ID  : {policy['policy_id']}")
        print(f"Title      : {policy['title']}")
        print(f"Source     : {policy['source_file']}")
        print(f"Similarity : {similarity:.4f}")
        print(f"Policy Text:\n{policy['text']}")

    print("\n" + "=" * 60)


def main():

    print("=" * 60)
    print("INSTAMART REFUND POLICY SEMANTIC SEARCH")
    print("=" * 60)

    print("\nLoading knowledge base...")

    embeddings, metadata = load_knowledge_base()

    print(f"Loaded policy chunks: {len(metadata)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully!")

    print("\nYou can now ask refund-related questions.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nEnter your question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nSemantic search stopped.")
            break

        search_policy(
            question,
            model,
            embeddings,
            metadata,
            top_k=3
        )


if __name__ == "__main__":
    main()