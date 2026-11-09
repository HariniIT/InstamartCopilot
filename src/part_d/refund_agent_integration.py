
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "outputs" / "part_c"
DATA_DIR = BASE_DIR / "data" / "processed"

# Part C ML models
CATEGORY_MODEL = MODEL_DIR / "ticket_category_model.pkl"
SENTIMENT_MODEL = MODEL_DIR / "ticket_sentiment_model.pkl"
URGENCY_MODEL = MODEL_DIR / "ticket_urgency_model.pkl"
PRIORITY_MODEL = MODEL_DIR / "ticket_priority_model.pkl"

# Part D RAG files
EMBEDDING_FILE = DATA_DIR / "refund_policy_embeddings.npy"
METADATA_FILE = DATA_DIR / "refund_policy_embedding_metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

SIMILARITY_THRESHOLD = 0.25


# ==================================================
# 2. LOAD MACHINE LEARNING MODELS
# ==================================================

print("=" * 70)
print("INSTAMART CUSTOMER SUPPORT + AI REFUND AGENT")
print("=" * 70)

print("\nLoading Part C ML models...")

model_paths = {
    "category": CATEGORY_MODEL,
    "sentiment": SENTIMENT_MODEL,
    "urgency": URGENCY_MODEL,
    "priority": PRIORITY_MODEL
}

for name, path in model_paths.items():

    if not path.exists():
        raise FileNotFoundError(
            f"{name} model not found: {path}"
        )

category_model = joblib.load(CATEGORY_MODEL)
sentiment_model = joblib.load(SENTIMENT_MODEL)
urgency_model = joblib.load(URGENCY_MODEL)
priority_model = joblib.load(PRIORITY_MODEL)

print("All 4 ML models loaded successfully!")


# ==================================================
# 3. LOAD RAG KNOWLEDGE BASE
# ==================================================

print("\nLoading Part D policy knowledge base...")

if not EMBEDDING_FILE.exists():
    raise FileNotFoundError(
        f"Embedding file not found: {EMBEDDING_FILE}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found: {METADATA_FILE}"
    )

embeddings = np.load(EMBEDDING_FILE)

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

if len(embeddings) != len(metadata):
    raise ValueError(
        "Embedding and metadata count mismatch."
    )

print(f"Policy chunks loaded: {len(metadata)}")


# ==================================================
# 4. LOAD SENTENCE TRANSFORMER
# ==================================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded successfully!")


# ==================================================
# 5. TEXT PREPROCESSING
# ==================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    stop_words = {
        "the", "is", "are", "was", "were",
        "a", "an", "to", "of", "in", "on",
        "for", "my", "i", "it", "this", "that",
        "and", "with"
    }

    words = [
        word for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


# ==================================================
# 6. CUSTOMER TICKET INTELLIGENCE
# ==================================================

def analyze_ticket(customer_message):

    cleaned_message = clean_text(customer_message)

    if not cleaned_message.strip():
        raise ValueError(
            "Please enter a valid customer complaint."
        )

    category = category_model.predict(
        [cleaned_message]
    )[0]

    sentiment = sentiment_model.predict(
        [cleaned_message]
    )[0]

    urgency = urgency_model.predict(
        [cleaned_message]
    )[0]

    priority = priority_model.predict(
        [cleaned_message]
    )[0]

    return {
        "customer_message": customer_message,
        "category": str(category),
        "sentiment": str(sentiment),
        "urgency": str(urgency),
        "priority": str(priority)
    }


# ==================================================
# 7. SEMANTIC POLICY RETRIEVAL
# ==================================================

def retrieve_policy(question):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = np.dot(
        embeddings,
        query_embedding
    )

    best_index = int(np.argmax(scores))

    best_score = float(scores[best_index])

    if best_score < SIMILARITY_THRESHOLD:
        return None, best_score

    return metadata[best_index], best_score


# ==================================================
# 8. REFUND RESPONSE GENERATION
# ==================================================

def generate_response(question, policy):

    question_lower = question.lower()

    if policy is None:

        return (
            "I could not find a sufficiently relevant refund policy. "
            "Please contact human customer support for assistance."
        )

    policy_id = policy["policy_id"]

    # Damaged product
    if policy_id == "REF002":

        response = (
            "We understand that you received a damaged product. "
            "Please provide your order ID and supporting details. "
            "Our support team can verify the issue and review "
            "your refund request."
        )

    # Missing item
    elif policy_id == "REF003":

        response = (
            "We understand that an item is missing from your delivery. "
            "Please provide your order ID and details of the missing item. "
            "Our support team can verify the order and review your request."
        )

    # Payment failure
    elif policy_id == "REF004":

        if (
            "repeated" in question_lower
            or "multiple times" in question_lower
            or "again and again" in question_lower
            or "payment failed" in question_lower
            or "immediate assistance" in question_lower
):

            response = (
                "We understand that your payment is failing repeatedly. "
                "Please check your payment method and transaction status. "
                "Avoid making repeated payment attempts until the status "
                "is confirmed. Our support team should review this issue."
            )

        elif (
            "deducted" in question_lower
            or "money was taken" in question_lower
        ):

            response = (
                "If money was deducted but your order was not created, "
                "the transaction must be verified first. Please provide "
                "your transaction reference. Refund approval depends "
                "on the verified payment status."
            )

        else:

            response = (
                "We understand that you are facing a payment issue. "
                "Please check your payment status and contact support "
                "for assistance."
            )

    # Refund escalation
    elif policy_id == "REF005":

        response = (
            "Your refund request can be escalated to human support "
            "if it cannot be verified. High-priority complaints should "
            "be reviewed quickly. Human support has final authority "
            "over unresolved refund cases."
        )

    # General refund policy
    else:

        response = (
            "According to the general refund policy, refund eligibility "
            "depends on order status and issue type. Please provide "
            "a valid order ID. Refund approval is subject to verification."
        )

    # Refund-related query validation
    refund_keywords = [
        "refund", "return", "payment", "deducted",
        "damaged", "missing", "item", "product",
        "order", "escalate", "money", "transaction",
        "failed", "failing", "repeatedly"
    ]

    if not any(
        word in question_lower
        for word in refund_keywords
    ):

        response = (
            "I can currently assist with refund-related questions. "
            "Please ask about a refund, damaged product, missing item, "
            "or payment failure."
        )

    return response


# ==================================================
# 9. COMPLETE CUSTOMER SUPPORT INTEGRATION
# ==================================================

def process_customer_complaint(message):

    # Part C: Machine Learning predictions
    ticket_result = analyze_ticket(message)

    # Part D: Semantic policy retrieval
    policy, similarity = retrieve_policy(message)

    # Part D: Customer response generation
    response = generate_response(message, policy)

    # Human escalation flag
    escalation_required = (
        ticket_result["urgency"].lower() == "high"
        or ticket_result["priority"].lower() == "critical"
    )

    return {
        "ticket": ticket_result,
        "policy": policy,
        "similarity": similarity,
        "response": response,
        "escalation_required": escalation_required
    }


# ==================================================
# 10. DISPLAY COMPLETE CUSTOMER REPORT
# ==================================================

def display_report(result):

    ticket = result["ticket"]
    policy = result["policy"]

    print("\n" + "=" * 70)
    print("CUSTOMER SUPPORT INTELLIGENCE REPORT")
    print("=" * 70)

    print("\nCUSTOMER MESSAGE:")
    print(ticket["customer_message"])

    print("\nPART C - MACHINE LEARNING PREDICTIONS")
    print("-" * 70)

    print("Ticket Category :", ticket["category"])
    print("Sentiment       :", ticket["sentiment"])
    print("Urgency         :", ticket["urgency"])
    print("Priority        :", ticket["priority"])

    print("\nPART D - SEMANTIC POLICY RETRIEVAL")
    print("-" * 70)

    if policy:

        print("Policy ID       :", policy["policy_id"])
        print("Policy Title    :", policy["title"])
        print(
            "Similarity      :",
            f"{result['similarity']:.4f}"
        )

    else:

        print("No sufficiently relevant policy found.")

    print("\nAI REFUND AGENT RESPONSE")
    print("-" * 70)

    print(result["response"])

    print("\nHUMAN SUPPORT ESCALATION")
    print("-" * 70)

    if result["escalation_required"]:

        print("ESCALATION FLAG: YES")
        print("This complaint should be reviewed by human support.")

    else:

        print("ESCALATION FLAG: NO")
        print("No automatic escalation flag was triggered.")

    print("\n" + "=" * 70)
    print("CUSTOMER SUPPORT PROCESSING COMPLETED")
    print("=" * 70)

    print(
        "Note: This is a demonstration system. "
        "No actual refund or support escalation is performed."
    )


# ==================================================
# 11. MAIN APPLICATION
# ==================================================

def main():

    print("\nINTEGRATED CUSTOMER SUPPORT SYSTEM READY!")

    print("\nComponents:")
    print("1. Ticket Category Prediction")
    print("2. Sentiment Analysis")
    print("3. Urgency Prediction")
    print("4. Priority Prediction")
    print("5. Refund Policy Retrieval")
    print("6. Customer Response Generation")
    print("7. Human Support Escalation Flag")

    print("\nType 'exit' to stop.")

    while True:

        message = input(
            "\nEnter Customer Complaint: "
        ).strip()

        if message.lower() in ["exit", "quit"]:

            print("\nApplication closed successfully.")
            break

        try:

            result = process_customer_complaint(message)

            display_report(result)

        except ValueError as error:

            print("\nInput Error:", error)


# ==================================================
# 12. APPLICATION ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
