import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
human = pd.read_csv("human_empathy_scoredd.csv")
gemini = pd.read_csv("empathy_scoredd.csv")

# ensure alignment
human = human.reset_index(drop=True)
gemini = gemini.reset_index(drop=True)

assert len(human) == len(gemini), "Human and Gemini datasets must have same length"

# =========================
# FEATURES
# =========================
dims = [
    "emotional_acknowledgment",
    "contextual_grounding",
    "emotional_validation",
    "supportive_intent",
    "generic_empathy",
    "emotional_overclaim",
    "functional_empathy",
    "eliza_risk",
    "final_score"
]

# =========================
# =========================
# 1. THESIS TABLE EXPORT
# =========================
# =========================

table = pd.DataFrame({
    "Human Mean": human[dims].mean(),
    "Gemini Mean": gemini[dims].mean(),
    "Difference (Gemini - Human)": gemini[dims].mean() - human[dims].mean()
})

table.to_csv("empathy_table.csv")
table.to_excel("empathy_table.xlsx")

print("\n=== THESIS TABLE ===")
print(table)

# =========================
# 2. BAR CHART COMPARISON
# =========================
x = np.arange(len(dims))
width = 0.35

plt.figure(figsize=(14,6))

plt.bar(x - width/2, human[dims].mean(), width, label="Human")
plt.bar(x + width/2, gemini[dims].mean(), width, label="Gemini")

plt.xticks(x, dims, rotation=45, ha="right")
plt.ylabel("Mean Score")
plt.title("Empathy Dimension Comparison (Human vs Gemini)")
plt.legend()

plt.tight_layout()
plt.savefig("bar_comparison.png", dpi=300)
plt.show()

# =========================
# 3. RADAR / SPIDER CHART
# =========================
labels = dims
num_vars = len(labels)

human_vals = human[dims].mean().values.tolist()
gemini_vals = gemini[dims].mean().values.tolist()

# close loop
human_vals += human_vals[:1]
gemini_vals += gemini_vals[:1]
labels += labels[:1]

angles = np.linspace(0, 2*np.pi, num_vars+1, endpoint=True)

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

ax.plot(angles, human_vals, linewidth=2, label="Human")
ax.fill(angles, human_vals, alpha=0.2)

ax.plot(angles, gemini_vals, linewidth=2, label="Gemini")
ax.fill(angles, gemini_vals, alpha=0.2)

ax.set_thetagrids(angles * 180/np.pi, labels)

plt.title("Empathy Profile Radar (Human vs Gemini)")
plt.legend(loc="upper right")

plt.savefig("radar_chart.png", dpi=300)
plt.show()

# =========================
# 4. ELIZA RISK GAP
# =========================
gap = gemini["eliza_risk"] - human["eliza_risk"]

plt.figure(figsize=(10,5))
plt.plot(gap.values, linewidth=1)
plt.axhline(0, linestyle="--", color="black")

plt.title("ELIZA Risk Gap (Gemini - Human)")
plt.xlabel("Sample Index")
plt.ylabel("Difference")

plt.savefig("eliza_gap.png", dpi=300)
plt.show()

# =========================
# 5. FINAL SCORE DISTRIBUTION
# =========================
plt.figure(figsize=(10,6))

plt.hist(human["final_score"], bins=25, alpha=0.5, label="Human")
plt.hist(gemini["final_score"], bins=25, alpha=0.5, label="Gemini")

plt.title("Final Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()

plt.savefig("final_score_distribution.png", dpi=300)
plt.show()

# =========================
# 6. QUICK SUMMARY STATS (for thesis text)
# =========================
print("\n=== SUMMARY STATS ===")
print("Human ELIZA mean:", human["eliza_risk"].mean())
print("Gemini ELIZA mean:", gemini["eliza_risk"].mean())
print("ELIZA gap mean:", gap.mean())
print("Human final score mean:", human["final_score"].mean())
print("Gemini final score mean:", gemini["final_score"].mean())