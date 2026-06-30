import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    cohen_kappa_score
)

# ==========================================
# CONFIGURATION & PARAMETERS
# ==========================================
ANNOTATION_FILE = "manual_annotation.csv"
AUDITOR_FILE = "empathy_scoredd.csv"
THRESHOLD = 0.8
SAMPLE_SIZE = 30  # Keep at 30 for evaluation testing

# ==========================================
# DATA LOADING & CLEANING
# ==========================================
try:
    annotations = pd.read_csv(ANNOTATION_FILE)
    auditor = pd.read_csv(AUDITOR_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find data files. Details: {e}")
    exit(1)

# Strip hidden whitespaces from column headers
annotations.columns = annotations.columns.str.strip()
auditor.columns = auditor.columns.str.strip()

# ==========================================
# DATA ALIGNMENT STRATEGY
# ==========================================
# Look for a common identifier column present in both datasets
common_cols = list(set(annotations.columns) & set(auditor.columns))

# Filter down to potential ID columns if they exist
id_candidates = [col for col in common_cols if col.lower() in ["id", "text_id", "uuid", "row_num", "index"]]

if id_candidates:
    merge_col = id_candidates[0]
    print(f"--> Found matching identifier column: '{merge_col}'. Performing Inner Join...")
    df = pd.merge(annotations, auditor, on=merge_col)
else:
    print("--> No common ID column found. Falling back to safe Row-by-Row Index alignment...")
    
    # Trim to the smallest common denominator to prevent mismatched shapes
    min_len = min(len(annotations), len(auditor))
    df_ann = annotations.head(min_len).reset_index(drop=True)
    df_aud = auditor.head(min_len).reset_index(drop=True)
    
    # Concatenate columns sideways
    df = pd.concat([df_ann, df_aud], axis=1)

# Extract your target subset evaluated for testing
df = df.head(SAMPLE_SIZE)

# ==========================================
# DIMENSION MAPPING
# ==========================================
# Key: Human column prefix (_h will be appended) -> Value: AI continuous score column
dimensions = {
    "A": "emotional_acknowledgment",
    "G": "contextual_grounding",
    "V": "emotional_validation",
    "I": "supportive_intent",
    "GE": "generic_empathy",
    "EO": "emotional_overclaim"
}

# ==========================================
# EVALUATION LOOP
# ==========================================
print(f"\nEvaluating performance on {len(df)} samples (Threshold: {THRESHOLD})")

for human_col, ai_col in dimensions.items():
    human_target = f"{human_col}_h"
    
    # Validation checks to verify columns actually exist in the aligned dataframe
    if human_target not in df.columns:
        print(f"\nSkipping: Human column '{human_target}' not found in data.")
        continue
    if ai_col not in df.columns:
        print(f"\nSkipping: AI column '{ai_col}' not found in data.")
        continue

    print("\n" + "=" * 60)
    print(f"DIMENSION: {ai_col.replace('_', ' ').title()} ({human_target} vs {ai_col})")
    print("=" * 60)

    # Cast datasets to explicit numerical arrays
    y_true = df[human_target].astype(int)
    y_pred = (df[ai_col] >= THRESHOLD).astype(int)

    # Force confusion matrix to evaluate both labels [0, 1] to avoid unpack crashes 
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Handle calculations safely with explicit labels to prevent warnings
    precision = precision_score(y_true, y_pred, labels=[0, 1], zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[0, 1], zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Pass explicit labels to kappa to stop user warnings
    kappa = cohen_kappa_score(y_true, y_pred, labels=[0, 1])

    # Raw Volume Breakdown
    print(f"True Positives (TP):  {tp:<4} | False Positives (FP): {fp}")
    print(f"True Negatives (TN):  {tn:<4} | False Negatives (FN): {fn}")
    print(f"Data Distribution:    {y_true.sum()} positive classes out of {len(y_true)} items")
    print("-" * 60)
    
    # Statistical Metrics
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1-Score:    {f1:.3f}")
    print(f"Accuracy:    {accuracy:.3f}")
    
    # Catch and print the edge case when kappa returns NaN due to zero variance
    if pd.isna(kappa):
        print("Cohen's K:   NaN (Undefined: Data contains only a single class label)")
    else:
        print(f"Cohen's K:   {kappa:.3f} (Agreement adjusted for baseline chance)")