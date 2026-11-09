
import pandas as pd
import numpy as np

from pathlib import Path

# --------------------------------------------------
# 1. RANDOM GENERATOR
# --------------------------------------------------

rng = np.random.default_rng(42)

# --------------------------------------------------
# 2. CUSTOMER SUPPORT TICKET TEMPLATES
# --------------------------------------------------

ticket_templates = {

    "Delivery": {
        "Positive": [
            "My order arrived on time and delivery was smooth.",
            "The delivery service was good and the package arrived safely.",
            "I received my order without any delivery problems."
        ],
        "Negative": [
            "My order is delayed and I still have not received it.",
            "The delivery person has not delivered my order.",
            "My order shows delivered but I have not received it."
        ],
        "Neutral": [
            "I want to know the current status of my delivery.",
            "Please provide an update about my order delivery.",
            "Can you check the expected delivery time?"
        ]
    },

    "Payment": {
        "Positive": [
            "My payment was successful and the transaction was smooth.",
            "The payment process worked perfectly.",
            "I completed my payment without any issue."
        ],
        "Negative": [
            "Money was deducted from my account but the order was not placed.",
            "My payment failed but the amount was deducted.",
            "I was charged twice for the same order."
        ],
        "Neutral": [
            "I want to check my payment transaction status.",
            "Please provide the payment receipt.",
            "Can you verify my payment details?"
        ]
    },

    "Product Quality": {
        "Positive": [
            "The product quality is excellent and I am satisfied.",
            "The items were fresh and properly packed.",
            "I received a good quality product."
        ],
        "Negative": [
            "The product was damaged when I received it.",
            "The food item was expired and unsafe to consume.",
            "I received a poor quality and defective product."
        ],
        "Neutral": [
            "I want more information about the product quality.",
            "Please check the product details.",
            "Can you provide information about this item?"
        ]
    },

    "Refund": {
        "Positive": [
            "My refund was processed successfully.",
            "I received my refund quickly. Thank you.",
            "The refund process was completed without problems."
        ],
        "Negative": [
            "I have not received my refund even after several days.",
            "My refund request was rejected incorrectly.",
            "The refund amount is missing from my bank account."
        ],
        "Neutral": [
            "Please check the status of my refund request.",
            "I want to know when my refund will be processed.",
            "Can you provide an update on my refund?"
        ]
    },

    "Account": {
        "Positive": [
            "I can log in successfully and my account works properly.",
            "The account registration process was easy.",
            "My account is working fine now."
        ],
        "Negative": [
            "I cannot log in to my account even with the correct password.",
            "My account has been locked unexpectedly.",
            "I am unable to access my registered account."
        ],
        "Neutral": [
            "I want to update my account information.",
            "Please help me change my registered phone number.",
            "Can you explain how to update my profile?"
        ]
    }
}

# --------------------------------------------------
# 3. URGENCY KEYWORDS
# --------------------------------------------------

urgent_phrases = [
    "This is urgent and needs immediate attention.",
    "Please resolve this issue immediately.",
    "I need help as soon as possible."
]

normal_phrases = [
    "Please look into this issue.",
    "I would appreciate your assistance.",
    "Please help me resolve this."
]

# --------------------------------------------------
# 4. GENERATE TICKETS
# --------------------------------------------------

total_tickets = 1200

categories = list(ticket_templates.keys())
sentiments = ["Positive", "Negative", "Neutral"]

category_weights = [0.25, 0.20, 0.20, 0.20, 0.15]
sentiment_weights = [0.15, 0.65, 0.20]

tickets = []

for i in range(1, total_tickets + 1):

    category = rng.choice(
        categories,
        p=category_weights
    )

    sentiment = rng.choice(
        sentiments,
        p=sentiment_weights
    )

    message = rng.choice(
        ticket_templates[category][sentiment]
    )

    # Urgency is more likely for negative complaints
    if sentiment == "Negative":
        urgency = rng.choice(
            ["High", "Medium", "Low"],
            p=[0.45, 0.40, 0.15]
        )
    else:
        urgency = rng.choice(
            ["High", "Medium", "Low"],
            p=[0.05, 0.30, 0.65]
        )

    # Add extra message detail
    if urgency == "High":
        message += " " + rng.choice(urgent_phrases)
    else:
        message += " " + rng.choice(normal_phrases)

    # Assign ticket priority
    if urgency == "High":
        priority = "Critical"
    elif urgency == "Medium":
        priority = "Normal"
    else:
        priority = "Low"

    tickets.append({
        "ticket_id": i,
        "customer_id": int(rng.integers(1, 1001)),
        "ticket_text": message,
        "category": category,
        "sentiment": sentiment,
        "urgency": urgency,
        "priority": priority,
        "created_at": (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(
                days=int(rng.integers(0, 300))
            )
        )
    })

# --------------------------------------------------
# 5. CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame(tickets)

# --------------------------------------------------
# 6. SAVE DATASET
# --------------------------------------------------

output_path = Path(
    "data/processed/support_tickets.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(output_path, index=False)

# --------------------------------------------------
# 7. DISPLAY DATASET SUMMARY
# --------------------------------------------------

print("=" * 60)
print("CUSTOMER SUPPORT TICKET DATASET GENERATED!")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nCategory distribution:")
print(df["category"].value_counts())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nUrgency distribution:")
print(df["urgency"].value_counts())

print("\nPriority distribution:")
print(df["priority"].value_counts())

print("\nFirst 5 tickets:")
print(
    df[[
        "ticket_id",
        "ticket_text",
        "category",
        "sentiment",
        "urgency",
        "priority"
    ]].head().to_string(index=False)
)

print("\nSaved:", output_path)

print("\nTICKET DATASET GENERATION COMPLETED!")
