import json
from pathlib import Path

import pandas as pd


# Project folder
BASE_DIR = Path(__file__).resolve().parents[2]

# Input and output folders
POLICY_DIR = BASE_DIR / "data" / "raw" / "refund_policies"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Chunk settings
CHUNK_SIZE = 100
CHUNK_OVERLAP = 20


def split_into_chunks(text, chunk_size=100, overlap=20):
    """Split policy text into overlapping word chunks."""

    words = text.split()

    if not words:
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks


def process_documents():

    print("=" * 60)
    print("REFUND POLICY DOCUMENT PROCESSOR")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy folder not found: {POLICY_DIR}"
        )

    policy_files = sorted(POLICY_DIR.glob("*.txt"))

    if not policy_files:
        raise FileNotFoundError(
            "No .txt policy documents found in the refund_policies folder."
        )

    all_chunks = []
    total_documents = 0

    for file_path in policy_files:

        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty document: {file_path.name}")
            continue

        total_documents += 1

        # First line is treated as the document title
        title = text.splitlines()[0].strip()

        # Normalize whitespace
        clean_text = " ".join(text.split())

        chunks = split_into_chunks(
            clean_text,
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        policy_id = file_path.stem

        for index, chunk in enumerate(chunks, start=1):

            record = {
                "chunk_id": f"{policy_id}_CH{index:03d}",
                "policy_id": policy_id,
                "source_file": file_path.name,
                "title": title,
                "chunk_index": index,
                "text": chunk
            }

            all_chunks.append(record)

        print(
            f"{file_path.name}: "
            f"{len(chunks)} chunks created"
        )

    if not all_chunks:
        raise ValueError("No text chunks were generated.")

    # Save as CSV
    csv_path = OUTPUT_DIR / "refund_policy_chunks.csv"

    dataframe = pd.DataFrame(all_chunks)
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")

    # Save as JSON
    json_path = OUTPUT_DIR / "refund_policy_chunks.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            all_chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("DOCUMENT PROCESSING SUMMARY")
    print("=" * 60)

    print(f"Documents processed : {total_documents}")
    print(f"Total chunks        : {len(all_chunks)}")
    print(f"Chunk size          : {CHUNK_SIZE} words")
    print(f"Chunk overlap       : {CHUNK_OVERLAP} words")

    print("\nOutput files:")
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")

    print("\nFirst chunk preview:")
    print("-" * 60)
    print(all_chunks[0]["text"])

    print("\nPOLICY DOCUMENT PROCESSING COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    process_documents()