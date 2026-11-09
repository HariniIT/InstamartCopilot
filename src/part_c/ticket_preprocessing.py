
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# 1. LOAD SUPPORT TICKETS
# --------------------------------------------------

input_path = Path("data/processed/support_tickets.csv")

df = pd.read_csv(input_path)

print("=" * 60)
print("CUSTOMER SUPPORT TICKET TEXT PREPROCESSING")
print("=" * 60)

print("\nOriginal records:", len(df))

# --------------------------------------------------
# 2. DEFINE STOPWORDS
# --------------------------------------------------

stopwords = {
    "i", "me", "my", "we", "our", "you", "your",
    "is", "am", "are", "was", "were", "the", "a",
    "an", "and", "or", "but", "to", "of", "in",
    "on", "for", "with", "this", "that", "it",
    "have", "has", "had", "be", "been", "can",
    "could", "would", "should", "please", "kindly"
}

# --------------------------------------------------
# 3. TEXT CLEANING FUNCTION
# --------------------------------------------------

def clean_text(text):

    # Convert text into lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        "",
        text
    )

    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    words = text.split()

    # Remove stopwords
    words = [
        word for word in words
        if word not in stopwords
    ]

    return " ".join(words)

# --------------------------------------------------
# 4. APPLY TEXT PREPROCESSING
# --------------------------------------------------

df["cleaned_text"] = df["ticket_text"].apply(clean_text)

# --------------------------------------------------
# 5. REMOVE EMPTY RECORDS
# --------------------------------------------------

df = df[
    df["cleaned_text"].str.strip() != ""
].copy()

# --------------------------------------------------
# 6. DISPLAY BEFORE AND AFTER
# --------------------------------------------------

print("\nTEXT CLEANING EXAMPLES")
print("-" * 60)

for i in range(min(5, len(df))):

    print("\nOriginal:")
    print(df.iloc[i]["ticket_text"])

    print("Cleaned:")
    print(df.iloc[i]["cleaned_text"])

# --------------------------------------------------
# 7. SAVE CLEANED DATASET
# --------------------------------------------------

output_path = Path(
    "data/processed/support_tickets_cleaned.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(output_path, index=False)

# --------------------------------------------------
# 8. SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("TEXT PREPROCESSING COMPLETED!")
print("=" * 60)

print("Total cleaned records:", len(df))
print("Original columns:", len(df.columns) - 1)
print("New column added: cleaned_text")

print("\nSaved:", output_path)
