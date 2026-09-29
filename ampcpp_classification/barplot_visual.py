#!/usr/bin/env python3
"""Bar plot of predicted AMP vs CPP counts from a predictions CSV."""

import pandas as pd
import matplotlib.pyplot as plt

# --- config ---
CSV_PATH = "BBBPPs_AMPCPP_predictions.csv"      # <- change to your file path
CLASS_COL = "predicted_class"     # 1 = AMP, 0 = CPP
OUT_PATH = "amp_vs_cpp_counts.png"

# --- load & count ---
df = pd.read_csv(CSV_PATH)

# count each class, guaranteeing both 0 and 1 appear even if one is empty
counts = df[CLASS_COL].value_counts().reindex([0, 1], fill_value=0)
labels = ["CPP (0)", "AMP (1)"]
values = [int(counts[0]), int(counts[1])]
total = sum(values)

# --- plot ---
fig, ax = plt.subplots(figsize=(5, 5))
colors = ["#4C78A8", "#E45756"]   # blue = CPP, red = AMP (colorblind-safe)
bars = ax.bar(labels, values, color=colors, width=0.6)

# annotate each bar with count + percentage
for bar, v in zip(bars, values):
    pct = f"{v/total*100:.1f}%" if total else "0%"
    ax.text(bar.get_x() + bar.get_width()/2, v,
            f"{v}\n({pct})", ha="center", va="bottom", fontsize=11)

ax.set_ylabel("Number of peptides")
ax.set_title(f"Predicted class distribution (n = {total})")
ax.set_ylim(0, max(values) * 1.15 if total else 1)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_PATH}  |  CPP={values[0]}, AMP={values[1]}, total={total}")
plt.show()