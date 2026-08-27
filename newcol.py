import pandas as pd
import torch
from transformers import pipeline


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = "my_data.csv"
OUTPUT_FILE = "my_data_with_emotions.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Loaded {len(df):,} complaints.")


# ============================================================
# CREATE TEXT FOR EMOTION ANALYSIS
# ============================================================

# Combine complaint_type and descriptor
# because they provide more information about the complaint.

df["Complaint Text"] = (
    df["complaint_type"].fillna("").astype(str)
    + ". "
    + df["descriptor"].fillna("").astype(str)
)


# ============================================================
# LOAD EMOTION MODEL
# ============================================================

print("Loading emotion model...")

device = 0 if torch.cuda.is_available() else -1

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)


# ============================================================
# FUNCTION TO PREDICT EMOTION
# ============================================================

def predict_emotion(text):

    if not text or text.strip() == "":
        return "Neutral"

    try:

        result = emotion_classifier(
            text,
            truncation=True
        )[0]

        return result["label"].capitalize()

    except Exception as e:

        print("Error:", e)

        return "Unknown"


# ============================================================
# PREDICT EMOTIONS
# ============================================================

print("Analyzing emotions...")
print("This may take some time for a large dataset.")


df["Emotion"] = df["Complaint Text"].apply(
    predict_emotion
)


# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("DONE!")
print("=" * 60)

print(
    f"Saved dataset to: {OUTPUT_FILE}"
)

print()
print("Emotion distribution:")
print(
    df["Emotion"].value_counts()
)