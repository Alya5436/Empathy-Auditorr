from datasets import load_dataset
import pandas as pd

print("Loading dataset...")
dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
data = dataset["train"].to_pandas()

# 1. Filter for utterance_idx == 1 
# This is the "Opening Statement" from the person feeling the emotion.
# It is ALWAYS the prompt/situation.
user_prompts_only = data[data["utterance_idx"] == 1]

target_emotions = [
    "sad", "afraid", "angry", "disappointed", 
    "embarrassed", "lonely", "anxious", "devastated"
]

balanced_samples = []

for emotion in target_emotions:
    # Filter by the emotion context
    subset = user_prompts_only[user_prompts_only["context"] == emotion]

    if not subset.empty:
        # Sample 10 unique starting prompts for this emotion
        sample = subset.sample(min(10, len(subset)), random_state=42)
        balanced_samples.append(sample)

balanced_df = pd.concat(balanced_samples)

# Select and rename
df = balanced_df[["context", "utterance"]].rename(columns={"context": "emotion", "utterance": "prompt"})

# Save
df.to_csv("prompts.csv", index=False)

print(f"Success! Saved {len(df)} user-only opening prompts.")