import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


human = pd.read_csv("human_empathy_scoredd.csv")
gemini = pd.read_csv("empathy_scoredd.csv")

# ensure alignment
human = human.reset_index(drop=True)
gemini = gemini.reset_index(drop=True)

assert len(human) == len(gemini), "Human and Gemini datasets must have same length"

# =========================
# ONLY RAW DIMENSIONS
# =========================
dims_6 = [
    "emotional_acknowledgment",
    "contextual_grounding",
    "emotional_validation",
    "supportive_intent",
    "generic_empathy",
    "emotional_overclaim"
]

labels = dims_6
num_vars = len(labels)

# =========================
# MEAN VALUES
# =========================
human_vals = human[dims_6].mean().values.tolist()
gemini_vals = gemini[dims_6].mean().values.tolist()

# close the loop (IMPORTANT for radar chart)
human_vals += human_vals[:1]
gemini_vals += gemini_vals[:1]
labels += labels[:1]

angles = np.linspace(0, 2 * np.pi, num_vars + 1, endpoint=True)

# =========================
# PLOT
# =========================
plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)

ax.plot(angles, human_vals, linewidth=2, label="Human")
ax.fill(angles, human_vals, alpha=0.2)

ax.plot(angles, gemini_vals, linewidth=2, label="Gemini")
ax.fill(angles, gemini_vals, alpha=0.2)

ax.set_thetagrids(angles * 180/np.pi, labels)

plt.title("Empathy Dimensions Radar Comparison")
plt.legend(loc="upper right")

plt.savefig("radar_dimensions_only.png", dpi=300)
plt.show()