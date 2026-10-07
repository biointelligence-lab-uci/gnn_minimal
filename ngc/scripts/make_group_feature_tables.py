"""
Split the BBBPPs into the four AMP/CPP x NGC groups and show the 12 QSAR
features for every peptide in each one.

Four tables, printed and written as CSVs:
    NGC AMP      AMP_CPP_call == AMP,  NGC_call == pore-forming   (n = 20)
    NGC CPP      AMP_CPP_call == CPP,  NGC_call == pore-forming   (n = 38)
    non-NGC AMP  AMP_CPP_call == AMP,  NGC_call == non-pore       (n = 58)
    non-NGC CPP  AMP_CPP_call == CPP,  NGC_call == non-pore       (n = 202)

Rows are sorted by prob_NGC_pore, descending, so each table reads from the
model's most confident member down.

The 12 features are taken as every column after prob_NGC_pore, so this keeps
working if the feature set in the input CSV changes.

Input:  BBBPPs_AMPCPP_qsar12.csv
Output: ngc/qsar/{ngc_amp,ngc_cpp,nonngc_amp,nonngc_cpp}_qsar12.csv

Usage:
    python make_group_feature_tables.py            # print tables, write CSVs
    python make_group_feature_tables.py --seq      # include the sequence column
    python make_group_feature_tables.py --max 25   # cap printed rows per table
    python make_group_feature_tables.py --quiet    # write CSVs only
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NGC_DIR = os.path.dirname(HERE)                      # ngc/
IN_CSV = os.path.join(NGC_DIR, "qsar", "BBBPPs_AMPCPP_qsar12.csv")
OUT_DIR = os.path.join(NGC_DIR, "qsar")

# (label, AMP_CPP_call, NGC_call, output stem)
GROUPS = [
    ("NGC AMP",     "AMP", "pore-forming", "ngc_amp"),
    ("NGC CPP",     "CPP", "pore-forming", "ngc_cpp"),
    ("non-NGC AMP", "AMP", "non-pore",     "nonngc_amp"),
    ("non-NGC CPP", "CPP", "non-pore",     "nonngc_cpp"),
]

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--seq", action="store_true",
                    help="include the sequence column when printing "
                         "(always present in the CSVs)")
parser.add_argument("--max", type=int, default=None, metavar="N",
                    help="print at most N rows per table (default: all)")
parser.add_argument("--quiet", action="store_true",
                    help="write the CSVs without printing the tables")
args = parser.parse_args()

df = pd.read_csv(IN_CSV)

# the 12 features are everything after the id/sequence/call/probability block
FEATURES = list(df.columns[df.columns.get_loc("prob_NGC_pore") + 1:])

os.makedirs(OUT_DIR, exist_ok=True)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

for label, amp_cpp, ngc, stem in GROUPS:
    sub = (df[(df.AMP_CPP_call == amp_cpp) & (df.NGC_call == ngc)]
             .sort_values("prob_NGC_pore", ascending=False)
             .reset_index(drop=True))
    sub.insert(2, "len", sub.sequence.str.len())

    out = os.path.join(OUT_DIR, f"{stem}_qsar12.csv")
    sub.to_csv(out, index=False)

    if args.quiet:
        print(f"{label:<12} n = {len(sub):>3}   ->  {out}")
        continue

    cols = ["id"] + (["sequence"] if args.seq else []) + ["len", "prob_NGC_pore"] + FEATURES
    shown = sub.head(args.max) if args.max else sub

    print(f"\n{'=' * 118}")
    print(f"{label}   (n = {len(sub)})")
    print("=" * 118)
    # %.4g keeps both the tau columns (~1e4) and the QSO columns (~1e-3) legible
    print(shown[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if args.max and len(sub) > args.max:
        print(f"... {len(sub) - args.max} more rows; full table in {out}")
    print(f"\nwrote {out}")

print(f"\n{len(FEATURES)} features: {', '.join(FEATURES)}")
