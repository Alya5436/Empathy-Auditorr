import pandas as pd
from sentence_transformers import SentenceTransformer, util

# ==============================
# LOAD MODEL
# ==============================
print("Loading SBERT model...")

model = SentenceTransformer('all-mpnet-base-v2')

# ==============================
# LOAD DATA
# ==============================
model_df = pd.read_csv("responses.csv")
human_df = pd.read_csv("human_responses.csv")
print("Human columns:", human_df.columns)

# ==============================
# CHECK LENGTH MATCH
# ==============================
assert len(model_df) == len(human_df), "Datasets must be same length!"

# ==============================
# COMPUTE SIMILARITY
# ==============================
similarities = []

for i in range(len(model_df)):
    print(f"Comparing {i+1}/{len(model_df)}")

    model_response = str(model_df.loc[i, "response"])
    human_response = str(human_df.loc[i, "response"])

    emb1 = model.encode(model_response, convert_to_tensor=True)
    emb2 = model.encode(human_response, convert_to_tensor=True)

    score = util.cos_sim(emb1, emb2).item()
    similarities.append(round(score, 3))

# ==============================
# SAVE RESULTS
# ==============================
model_df["human_similarity_score"] = similarities

# Optional: binary threshold
THRESHOLD = 0.5
model_df["human_alignment"] = [1 if s >= THRESHOLD else 0 for s in similarities]

model_df.to_csv("comparison_results.csv", index=False)

print("✅ Done! Saved to comparison_results.csv")