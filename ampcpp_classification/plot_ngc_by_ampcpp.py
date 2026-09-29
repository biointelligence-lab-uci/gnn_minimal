"""
Predicted AMP/CPP class distribution for the b3ppsexp peptides, with each bar
split by the NGC checkpoint's pore-forming call.

Styled to match the existing predicted-class barplot: same title, same class
order (CPP first), same per-class colors. Each bar keeps its class hue and the
pore-forming share sits on top in a lighter step of that same hue, so the split
reads without changing what the bar colors mean.

Input:  BBBPPs_AMPCPP_NGC_predictions.csv  (written by add_ngc_predictions.py)
Output: ngc_generators_by_ampcpp.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))

# class hues as in the existing barplot; the lighter step of each is that
# class's pore-forming segment
CPP_COLOR, CPP_LIGHT = "#4C72B0", "#A5B8D7"
AMP_COLOR, AMP_LIGHT = "#C44E52", "#E1A6A8"
INK = "#0b0b0b"

df = pd.read_csv(os.path.join(HERE, "BBBPPs_AMPCPP_NGC_predictions.csv"))
ct = (pd.crosstab(df.AMP_CPP_call, df.NGC_call)
        .reindex(index=["CPP", "AMP"], columns=["non-pore", "pore-forming"],
                 fill_value=0))
totals = ct.sum(axis=1)
n = int(totals.sum())

labels = ["CPP (0)", "AMP (1)"]
base = [CPP_COLOR, AMP_COLOR]
light = [CPP_LIGHT, AMP_LIGHT]

fig, ax = plt.subplots(figsize=(7.5, 7.5))
x = range(len(ct))
width = 0.62
gap = totals.max() * 0.006          # thin surface gap between the two segments
bottom_pore = ct["non-pore"] + gap

ax.bar(x, ct["non-pore"], width, color=base, zorder=3)
ax.bar(x, ct["pore-forming"], width, bottom=bottom_pore, color=light, zorder=3)

min_inside = totals.max() * 0.045    # shorter segments get labelled beside the bar

for i, cls in enumerate(ct.index):
    n_non, n_pore, n_tot = ct.loc[cls, "non-pore"], ct.loc[cls, "pore-forming"], totals[cls]
    ax.text(i, n_tot + gap + totals.max() * 0.02,
            f"{n_tot}\n({n_tot / n:.1%})", ha="center", va="bottom",
            fontsize=17, linespacing=1.3, color=INK)
    for value, mid, ink in [(n_non, n_non / 2, "#ffffff"),
                            (n_pore, bottom_pore[cls] + n_pore / 2, INK)]:
        if value >= min_inside:
            ax.text(i, mid, str(value), ha="center", va="center",
                    fontsize=16, color=ink, zorder=4)
        else:
            ax.text(i + width / 2 + 0.05, mid, str(value), ha="left",
                    va="center", fontsize=16, color=INK, zorder=4)

ax.set_title(f"Predicted class distribution (n = {n})", fontsize=19, pad=16)
ax.set_ylabel("Number of peptides", fontsize=16)
ax.set_xticks(list(x), labels, fontsize=15)
ax.set_xlim(-0.6, len(ct) - 0.4)
ax.set_ylim(0, totals.max() * 1.16)
ax.tick_params(axis="y", labelsize=14)
ax.spines[["top", "right"]].set_visible(False)

ax.legend(handles=[Patch(facecolor="#9a9a9a", label="non-pore"),
                   Patch(facecolor="#d6d6d6", label="NGC pore-forming")],
          frameon=False, fontsize=13, loc="upper right", ncol=1,
          bbox_to_anchor=(1.0, 0.93), handlelength=1.3, handletextpad=0.6,
          labelspacing=0.5)

plt.tight_layout()
out = os.path.join(HERE, "ngc_generators_by_ampcpp.png")
plt.savefig(out, dpi=200)

print(ct)
print(f"\npore-forming share: CPP {ct.loc['CPP','pore-forming'] / totals['CPP']:.1%}, "
      f"AMP {ct.loc['AMP','pore-forming'] / totals['AMP']:.1%}")
print(f"Wrote {out}")
