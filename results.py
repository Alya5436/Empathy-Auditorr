import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("empathy_scoredd.csv")

# =========================
# STYLE (makes plots more “thesis ready”)
# =========================
sns.set(style="whitegrid")

# =========================================================
# FIGURE 1: CORE ELIZA EFFECT RELATIONSHIP (MOST IMPORTANT)
# Functional Empathy vs ELIZA Risk
# =========================================================
plt.figure(figsize=(7,5))
plt.scatter(
    df["functional_empathy"],
    df["eliza_risk"],
    alpha=0.6
)

plt.title("Functional Empathy vs ELIZA Risk")
plt.xlabel("Functional Empathy (Grounded Support)")
plt.ylabel("ELIZA Risk (Surface Empathy Illusion)")
plt.tight_layout()
plt.show()


# =========================================================
# FIGURE 2: CORRELATION HEATMAP (STRUCTURE OF YOUR MODEL)
# =========================================================
cols = [
    "emotional_acknowledgment",
    "contextual_grounding",
    "emotional_validation",
    "supportive_intent",
    "generic_empathy",
    "emotional_overclaim",
    "functional_empathy",
    "eliza_risk"
]

plt.figure(figsize=(9,6))
sns.heatmap(df[cols].corr(), cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Empathy Dimensions")
plt.tight_layout()
plt.show()


# =========================================================
# FIGURE 3: DISTRIBUTION OF ELIZA RISK (CORE METRIC)
# =========================================================
plt.figure(figsize=(7,5))
plt.hist(df["eliza_risk"], bins=20)
plt.title("Distribution of ELIZA Risk")
plt.xlabel("ELIZA Risk Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# =========================================================
# FIGURE 4: MEAN EMPATHY PROFILE (INTERPRETABLE SUMMARY)
# =========================================================
dims = [
    "emotional_acknowledgment",
    "contextual_grounding",
    "emotional_validation",
    "supportive_intent",
    "generic_empathy",
    "emotional_overclaim"
]

means = df[dims].mean()

plt.figure(figsize=(8,5))
plt.bar(dims, means)
plt.xticks(rotation=45, ha="right")
plt.title("Average Empathy Signal Profile")
plt.ylabel("Mean Score")
plt.tight_layout()
plt.show()