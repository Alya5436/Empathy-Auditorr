from datasets import load_dataset
import pandas as pd

# 1. Load the dataset
dataset = load_dataset("empathetic_dialogues", trust_remote_code=True)
data = dataset["train"].to_pandas()

# 2. Re-establish your exact 80 prompts (using your specific random_state)
target_emotions = ["sad", "afraid", "angry", "disappointed", "embarrassed", "lonely", "anxious", "devastated"]
user_prompts_only = data[data["utterance_idx"] == 1]

balanced_samples_list = []
for emotion in target_emotions:
    subset = user_prompts_only[user_prompts_only["context"] == emotion]
    if not subset.empty:
        sample = subset.sample(min(10, len(subset)), random_state=42)
        balanced_samples_list.append(sample)

selected_prompts_df = pd.concat(balanced_samples_list)
selected_conv_ids = selected_prompts_df["conv_id"].tolist()

# 3. CRITICAL FIX: Only grab UTTERANCE_IDX 2, 4, 6...
# This strictly ignores turns 1, 3, 5 where the user talks back.
pure_human_responses = data[
    (data["conv_id"].isin(selected_conv_ids)) & 
    (data["utterance_idx"] % 2 == 0)
]

# 4. Group by conversation and join ONLY the human listener's words
combined_responses = pure_human_responses.groupby("conv_id")["utterance"].apply(" ".join).reset_index()

# 5. Final Merge
final_audit_df = pd.merge(
    selected_prompts_df[["conv_id", "context", "utterance"]].rename(columns={"context": "emotion", "utterance": "prompt"}),
    combined_responses.rename(columns={"utterance": "ground_truth"}),
    on="conv_id"
)

# Clean dataset artifacts
final_audit_df['ground_truth'] = final_audit_df['ground_truth'].str.replace('_comma_', ',')

# 6. Save as the Final Gold Standard
final_audit_df[["emotion", "prompt", "ground_truth"]].to_csv("gold_standard_empathy.csv", index=False)

print("Check complete. The 'It really doesnt' (Speaker 1) has been removed.")