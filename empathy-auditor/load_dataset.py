from datasets import load_dataset
import pandas as pd

print("Loading dataset...")

# Load the EmpatheticDialogues dataset
# Note: Added trust_remote_code=True to suppress the warning you saw earlier
dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)

# Convert to pandas dataframe
data = dataset["train"].to_pandas()

print("Dataset loaded!")

# Distress emotions more suitable for empathy auditing
target_emotions = [
    "sad", "afraid", "angry", "disappointed", 
    "embarrassed", "lonely", "anxious", "devastated"
]

balanced_samples = []

for emotion in target_emotions:
    # Use "context" as the column name
    subset = data[data["context"] == emotion]

    # Sample up to 10 prompts per emotion
    if not subset.empty:
        sample = subset.sample(min(10, len(subset)), random_state=42)
        balanced_samples.append(sample)

# Combine samples
balanced_df = pd.concat(balanced_samples)

# FIX: Use "context" here instead of "emotion"
df = balanced_df[["context", "utterance"]]

# Rename for clarity
df = df.rename(columns={"context": "emotion", "utterance": "prompt"})

# Save prompts
df.to_csv("prompts.csv", index=False)

print("Prompts saved to prompts.csv")
print("Total prompts:", len(df))