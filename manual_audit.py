import pandas as pd

# =========================
# CONFIG
# =========================
INPUT_FILE = "llm_responses.csv"
OUTPUT_FILE = "manual_annotation_set.csv"

N_SAMPLES = 40

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)

# clean column names
df.columns = df.columns.str.strip()

# =========================
# SELECT FIRST N SAMPLES
# =========================
sample_df = df.head(N_SAMPLES).copy()

# =========================
# ADD UNIQUE IDS
# =========================
sample_df.insert(
    0,
    "id",
    [f"P{str(i+1).zfill(3)}" for i in range(len(sample_df))]
)

# =========================
# ADD MANUAL ANNOTATION COLUMNS
# =========================
annotation_columns = [
    "A_h",   # Emotional Acknowledgment
    "G_h",   # Contextual Grounding
    "V_h",   # Emotional Validation
    "I_h",   # Supportive Intent
    "GE_h",  # Generic Empathy
    "EO_h",  # Emotional Overclaim
    "notes"
]

for col in annotation_columns:
    sample_df[col] = ""

# =========================
# KEEP ONLY NECESSARY COLUMNS
# =========================
final_columns = [
    "id",
    "prompt",
    "response",
    "A_h",
    "G_h",
    "V_h",
    "I_h",
    "GE_h",
    "EO_h",
    "notes"
]

sample_df = sample_df[final_columns]

# =========================
# EXPORT
# =========================
sample_df.to_csv(OUTPUT_FILE, index=False)

print(f"Saved annotation dataset to: {OUTPUT_FILE}")
print(f"Total samples: {len(sample_df)}")