#!/usr/bin/env python3
"""
The NGC generators as a readable table: name, sequence, and what is known about
each one. No probabilities, and no AMP/CPP column -- the two tables are already
split by class.

Rows stay in sigma order (strongest NGC call first) even though the score is not
shown, so the ranking from sigma_score_ranking.py carries over.

Writes a Markdown copy as well as printing. Markdown because the annotation
fields are long: in a terminal they have to be truncated to fit, while the .md
renders every field in full and previews directly in the editor.

Inputs:
    ../tables/ngc_poreformers_{amp,cpp}_annotated.csv
Output:
    ../tables/ngc_generators.md

Usage:
    python scripts/ngc_generators_table.py
"""
from __future__ import annotations

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NGC_DIR = os.path.dirname(HERE)                      # ngc/
TABLES = os.path.join(NGC_DIR, "tables")
OUT_MD = os.path.join(TABLES, "ngc_generators.md")

# the GNN's raw score, out[:,1] - out[:,0], read straight off the model by
# ../../get_logits_scoring.py. Call boundary is +1.3395, not 0.
SCORES = os.path.join(NGC_DIR, "predictions", "BBBPPs_NGC_scores.csv")
BOUNDARY = 1.3395

# reference field -> column header
FIELDS = ["Transport", "Target", "Chemistry", "Evidence", "Notes"]

# terminal view drops the two longest fields; the .md keeps everything
TERM = [("Transport", "transport", 11),
        ("Target", "target", 26),
        ("Chemistry", "chemistry", 24)]


def clean(v):
    return "" if pd.isna(v) else str(v).strip()


def trim(v, n):
    v = clean(v)
    return v if len(v) <= n else v[: n - 1] + "…"


md = ["# NGC generators", ""]

for cls, stem in [("AMP", "amp"), ("CPP", "cpp")]:
    src = os.path.join(TABLES, f"ngc_poreformers_{stem}_annotated.csv")
    df = pd.read_csv(src)   # already ranked by Sigma, md column set

    # ---- terminal ----
    head = f"{'#':>3}  {'sigma':>7}  {'name':<26}{'sequence':<52}"
    for _, label, w in TERM:
        head += f"{label:<{w + 2}}"
    print(f"\n{'=' * len(head)}\n{cls} NGC generators  (n = {len(df)})\n{'=' * len(head)}")
    print(head)
    print("-" * len(head))
    for i, r in df.iterrows():
        line = (f"{i + 1:>3}  {r['Sigma']:>7.3f}  {trim(r['Name'], 25):<26}"
                f"{clean(r['Sequence']):<52}")
        for col, _, w in TERM:
            line += f"{trim(r[col], w):<{w + 2}}"
        print(line.rstrip())
    print("-" * len(head))
    print(f"sigma {df.Sigma.min():.3f} .. {df.Sigma.max():.3f}   median {df.Sigma.median():.3f}"
          f"   (call boundary +{BOUNDARY})")

    # ---- markdown ----
    md += [f"## {cls} ({len(df)})", "",
           "| # | Sigma | Name | Sequence | " + " | ".join(FIELDS) + " |",
           "|---:|---:|---|---|" + "---|" * len(FIELDS)]
    for i, r in df.iterrows():
        cells = [clean(r[col]).replace("|", "\\|") or "—" for col in FIELDS]
        seq = clean(r["Sequence"])
        md.append(f"| {i + 1} | {r['Sigma']:.3f} | {clean(r['Name'])} | `{seq}` | "
                  + " | ".join(cells) + " |")
    md.append("")

with open(OUT_MD, "w") as fh:
    fh.write("\n".join(md))
print(f"\nwrote {OUT_MD}")
