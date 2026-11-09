import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_FILE = BASE_DIR / "data" / "processed" / "refund_policy_chunks.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

EMBEDDING_FILE = OUTPUT_DIR / "refund_policy_embeddings.npy"
METADATA_FILE = OUTPUT_DIR / "refund_policy_embedding_metadata.json"

# Embedding model
MODEL_NAME = "all-MiniLM-L6-v2"


def generate_embeddings():

    print("=" * 60)
    print("REFUND POLICY EMBEDDING GENERATOR")
    print("=" * 60)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load policy chunks
    df = pd.read_csv(INPUT_FILE)

    if df.empty or "text" not in df.columns:
        raise ValueError("No policy chunks found in the input CSV.")

    texts = df["text"].fillna("").astype(str).tolist()

    if not any(text.strip() for text in texts):
        raise ValueError("Policy text is empty.")

    print(f"\nTotal policy chunks: {len(texts)}")
    print(f"Loading model: {MODEL_NAME}")

    # Load pretrained sentence transformer
    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully!")

    # Convert text into numerical vectors
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    # Save embedding vectors
    np.save(EMBEDDING_FILE, embeddings)

    # Save metadata separately
    metadata = df.to_dict(orient="records")

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 60)

    print(f"Chunks processed : {len(texts)}")
    print(f"Embedding shape  : {embeddings.shape}")
    print(f"Vector dimensions: {embeddings.shape[1]}")

    print("\nSaved files:")
    print(f"Embeddings: {EMBEDDING_FILE}")
    print(f"Metadata  : {METADATA_FILE}")

    print("\nPOLICY EMBEDDINGS GENERATED SUCCESSFULLY!")


if __name__ == "__main__":
    generate_embeddings()