import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Existing embedding and metadata files
EMBEDDING_FILE = (
    BASE_DIR / "data" / "processed" /
    "refund_policy_embeddings.npy"
)

METADATA_FILE = (
    BASE_DIR / "data" / "processed" /
    "refund_policy_embedding_metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum similarity threshold (demo setting)
SIMILARITY_THRESHOLD = 0.25


def load_knowledge_base():

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError("Policy embeddings not found.")

    if not METADATA_FILE.exists():
        raise FileNotFoundError("Policy metadata not found.")

    embeddings = np.load(EMBEDDING_FILE)

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    if len(embeddings) != len(metadata):
        raise ValueError("Embedding and metadata count mismatch.")

    return embeddings, metadata


def retrieve_policy(question, model, embeddings, metadata):

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Calculate cosine similarity
    scores = np.dot(embeddings, query_embedding)

    best_index = int(np.argmax(scores))
    best_score = float(scores[best_index])

    if best_score < SIMILARITY_THRESHOLD:
        return None, best_score

    return metadata[best_index], best_score


def generate_response(question, policy, similarity):

    question_lower = question.lower()

    if policy is None:

        return (
            "I could not find a sufficiently relevant refund policy "
            "for your question. Please contact human customer support "
            "for further assistance."
        )

    policy_id = policy["policy_id"]
    policy_text = policy["text"]

    # Rule-based response generation using retrieved policy
    if policy_id == "REF002":

        response = (
            "We understand that you received a damaged product. "
            "According to the demonstration refund policy, "
            "your issue requires verification before a refund decision. "
            "Please provide your order ID and supporting details. "
            "Our support team can review your request."
        )

    elif policy_id == "REF003":

        response = (
            "We understand that an item is missing from your delivery. "
            "Please provide your order ID and details of the missing item. "
            "Our support team can verify the order and review your request."
        )

    elif policy_id == "REF004":

        response = (
            "If money was deducted but your order was not created, "
            "the transaction must be verified first. "
            "Please provide your order ID or transaction reference. "
            "Refund approval depends on the verified payment status."
        )

    elif policy_id == "REF005":

        response = (
            "Your refund request can be escalated to human support "
            "if it cannot be verified. High-priority complaints should "
            "be reviewed quickly. Human support has final authority "
            "over unresolved refund cases. You should receive an "
            "explanation of the escalation status."
        )

    else:

        response = (
            "According to the general refund policy, refund eligibility "
            "depends on order status and issue type. Please provide a "
            "valid order ID. Refund approval is subject to verification "
            "by the support team."
        )

    # Special handling for unrelated questions
    refund_keywords = [
        "refund", "return", "payment", "deducted",
        "damaged", "missing", "item", "product",
        "order", "escalate", "money", "transaction"
    ]

    if not any(word in question_lower for word in refund_keywords):
        response = (
            "I can currently assist with refund-related questions. "
            "Please ask me about a refund, damaged product, missing item, "
            "or payment failure."
        )

    return response


def main():

    print("=" * 65)
    print("INSTAMART AI REFUND RESPONSE GENERATOR")
    print("=" * 65)

    print("\nLoading policy knowledge base...")

    embeddings, metadata = load_knowledge_base()

    print(f"Knowledge base loaded: {len(metadata)} policy chunks")

    print("\nLoading sentence transformer model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Model loaded successfully!")

    print("\nAI Refund Agent is ready!")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nCustomer Question: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("\nAI Refund Agent stopped.")
            break

        if not question:
            print("Please enter a question.")
            continue

        policy, similarity = retrieve_policy(
            question,
            model,
            embeddings,
            metadata
        )

        response = generate_response(
            question,
            policy,
            similarity
        )

        print("\n" + "=" * 65)
        print("AI REFUND AGENT RESPONSE")
        print("=" * 65)

        print(f"\nCustomer Question: {question}")

        if policy is not None:
            print(f"\nRetrieved Policy: {policy['policy_id']}")
            print(f"Policy Title: {policy['title']}")
            print(f"Similarity Score: {similarity:.4f}")
        else:
            print("\nRetrieved Policy: No sufficiently relevant policy")

        print("\nGenerated Response:")
        print("-" * 65)
        print(response)

        print("\n" + "=" * 65)
        print("Note: This is a demonstration system. "
              "Final refund approval requires human verification.")
        print("=" * 65)


if __name__ == "__main__":
    main()