import pandas as pd
import numpy as np
import json
import os
from groq import Groq
import os



# =========================
# CONFIG
# =========================
API_KEY = "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"  # set this in your environment
INPUT_FILE = "llm_responses.csv"
OUTPUT_FILE = "empathy_scoredd.csv"

MODEL = "llama-3.1-8b-instant"
N_RUNS = 2  # stability: run each response multiple times

client = Groq(api_key=API_KEY)
if os.path.exists(OUTPUT_FILE):
    done_df = pd.read_csv(OUTPUT_FILE)
    done_indices = set(done_df.index.tolist())
else:
    done_df = pd.DataFrame()
    done_indices = set()

# =========================
# LLM SCORING (JSON ONLY)
# =========================
def score_empathy(response):

    prompt = f"""
You are an expert annotator of empathy in conversational responses.

Rate each dimension from 0 to 1.

Return ONLY valid JSON. No explanation.

Schema:
{{
  "emotional_acknowledgment": float,
  "contextual_grounding": float,
  "emotional_validation": float,
  "supportive_intent": float,
  "generic_empathy": float,
  "emotional_overclaim": float
}}

Response:
{response}
"""

    res = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    text = res.choices[0].message.content

    # robust JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            raise ValueError(f"Invalid JSON output:\n{text}")


# =========================
# FUNCTIONAL EMPATHY
# =========================
def functional_empathy(scores):

    return (
        scores["emotional_acknowledgment"] +
        scores["contextual_grounding"] +
        scores["emotional_validation"] +
        scores["supportive_intent"]
    ) / 4


# =========================
# ELIZA RISK SCORE
# =========================
def eliza_risk(scores):

    surface = (
        scores["generic_empathy"] +
        scores["emotional_overclaim"]
    ) / 2

    grounded = (
        scores["emotional_acknowledgment"] +
        scores["contextual_grounding"] +
        scores["emotional_validation"] +
        scores["supportive_intent"]
    ) / 4

    return np.tanh(surface - grounded)


# =========================
# MULTI-RUN AVERAGING
# =========================
def run_scoring_multiple_times(response, n_runs=3):

    all_scores = []

    for _ in range(n_runs):
        scores = score_empathy(response)
        all_scores.append(scores)

    # average each dimension
    avg_scores = {
        key: np.mean([s[key] for s in all_scores])
        for key in all_scores[0].keys()
    }

    return avg_scores


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip()

results = []

# =========================
# MAIN LOOP
# =========================
for i, row in df.iterrows():

    response = row["response"]

    try:
        # step 1: stable scoring
        scores = run_scoring_multiple_times(response, N_RUNS)

        # step 2: derived metrics
        func = functional_empathy(scores)
        risk = eliza_risk(scores)

        # step 3: optional combined score (ranking only)
        final_score = func - risk

    except Exception as e:
        print(f"Error on row {i}: {e}")
        continue

    results.append({
        "response": response,

        # raw dimensions
        **scores,

        # derived metrics
        "functional_empathy": func,
        "eliza_risk": risk,
        "final_score": final_score
    })

    print(f"Processed {i+1}/{len(df)}")

# =========================
# SAVE OUTPUT
# =========================
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)

print("Done. Saved to:", OUTPUT_FILE)