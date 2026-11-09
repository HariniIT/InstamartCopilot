
import os
import re
import joblib

# --------------------------------------------------
# 1. MODEL PATHS
# --------------------------------------------------

MODEL_DIR = "outputs/part_c"

category_model_path = os.path.join(
    MODEL_DIR, "ticket_category_model.pkl"
)

sentiment_model_path = os.path.join(
    MODEL_DIR, "ticket_sentiment_model.pkl"
)

urgency_model_path = os.path.join(
    MODEL_DIR, "ticket_urgency_model.pkl"
)

priority_model_path = os.path.join(
    MODEL_DIR, "ticket_priority_model.pkl"
)

# --------------------------------------------------
# 2. CHECK MODEL FILES
# --------------------------------------------------

model_paths = [
    category_model_path,
    sentiment_model_path,
    urgency_model_path,
    priority_model_path
]

for path in model_paths:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required model not found: {path}"
        )

# --------------------------------------------------
# 3. LOAD ALL MODELS
# --------------------------------------------------

print("=" * 65)
print("INSTAMART CUSTOMER SUPPORT INTELLIGENCE SYSTEM")
print("=" * 65)

print("\nLoading ML models...")

category_model = joblib.load(category_model_path)
sentiment_model = joblib.load(sentiment_model_path)
urgency_model = joblib.load(urgency_model_path)
priority_model = joblib.load(priority_model_path)

print("All 4 models loaded successfully!")

# --------------------------------------------------
# 4. TEXT PREPROCESSING
# --------------------------------------------------

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

# --------------------------------------------------
# 5. TICKET INTELLIGENCE FUNCTION
# --------------------------------------------------

def analyze_ticket(customer_message):

    cleaned_message = clean_text(customer_message)

    if not cleaned_message.strip():
        raise ValueError(
            "Please enter a valid customer complaint."
        )

    # Category prediction
    category = category_model.predict(
        [cleaned_message]
    )[0]

    # Sentiment prediction
    sentiment = sentiment_model.predict(
        [cleaned_message]
    )[0]

    # Urgency prediction
    urgency = urgency_model.predict(
        [cleaned_message]
    )[0]

    # Priority prediction
    priority = priority_model.predict(
        [cleaned_message]
    )[0]

    return {
        "customer_message": customer_message,
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "priority": priority
    }

# --------------------------------------------------
# 6. DISPLAY CUSTOMER REPORT
# --------------------------------------------------

def display_report(result):

    print("\n" + "=" * 65)
    print("CUSTOMER SUPPORT TICKET ANALYSIS REPORT")
    print("=" * 65)

    print("\nCUSTOMER MESSAGE:")
    print(result["customer_message"])

    print("\nPREDICTION RESULTS")
    print("-" * 65)

    print("Ticket Category :", result["category"])
    print("Sentiment       :", result["sentiment"])
    print("Urgency         :", result["urgency"])
    print("Priority        :", result["priority"])

    print("\n" + "=" * 65)
    print("TICKET ANALYSIS COMPLETED!")
    print("=" * 65)

# --------------------------------------------------
# 7. INTERACTIVE CUSTOMER INPUT
# --------------------------------------------------

if __name__ == "__main__":

    print("\nCUSTOMER SUPPORT SYSTEM READY!")

    while True:

        print("\nEnter customer complaint.")
        print("Type 'exit' to stop the application.")

        message = input("\nCustomer: ").strip()

        if message.lower() == "exit":
            print("\nThank you! Application closed.")
            break

        try:
            result = analyze_ticket(message)
            display_report(result)

        except ValueError as error:
            print("\nError:", error)
