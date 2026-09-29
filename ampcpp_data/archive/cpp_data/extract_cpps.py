#!/usr/bin/env python3
"""
Extract known cell-penetrating peptides (CPPs) from the CPPpred-En data files.

IMPORTANT: the CPP924 files in that repo have INVERTED labels. The upstream
source (StackCPPred) uses 0 = CPP, but the repo author converted assuming
1 = positive, so every CPP924 sequence tagged 'CPP_positive' is actually a
NON-CPP, and the 'CPP_negative' ones are the real CPPs. The MLCPP files are
labelled correctly. This script accounts for that.

Usage:
    python extract_cpps.py /path/to/CPPpred-En/data
    python extract_cpps.py /path/to/data --out mycpps 
"""
import argparse
import sys
from pathlib import Path

# filename -> which FASTA header tag marks the real CPPs
CPP_TAG = {
    "CPP924_train": "CPP_negative",   # inverted
    "CPP924_test":  "CPP_negative",   # inverted
    "MLCPP_train":  "CPP_positive",   # correct
    "MLCPP_test":   "CPP_positive",   # correct
}

# sequences MLCPP explicitly lists as non-CPPs; drop if they sneak in
KNOWN_NON_CPP = {
    "EKRPRTAFSSEQLARLKREFNENRYLTTERRRQQLSSELGLNEAQIKIWFQNKRAKIKKST",
}

# canonical CPPs used to verify the extraction pulled the right side
SANITY = {
    "Transportan": "GWTLNSAGYLLGKINLKALAALAKKIL",
    "Pep-1":       "KETWWETWWTEWSQPKKKRKV",
    "R7-analogue": "RRRRRRRGGIYLATALAKWALKQGF",
}

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path):
    """Yield (header, sequence) pairs."""
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir", help="the CPPpred-En 'data' folder")
    ap.add_argument("--out", default="cpps", help="output prefix (default: cpps)")
    ap.add_argument("--label", default="0",
                    help="label written to the CSV (default: 0, the non-AMP class)")
    ap.add_argument("--train-only", action="store_true",
                    help="use only the *_train files, skip the test splits")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit(f"error: not a directory: {data_dir}")

    wanted = {k: v for k, v in CPP_TAG.items()
              if not (args.train_only and k.endswith("_test"))}

    cpps = {}        # sequence -> set of source file stems
    missing = []
    print("Extracting:")
    for stem, tag in wanted.items():
        path = data_dir / f"{stem}.txt"
        if not path.exists():
            missing.append(path.name)
            continue
        n = 0
        for header, seq in read_fasta(path):
            if header.startswith(tag):
                cpps.setdefault(seq, set()).add(stem)
                n += 1
        flag = "  [labels inverted -> used CPP_negative]" if "924" in stem else ""
        print(f"  {path.name:20s} {n:5d} CPPs from '{tag}'{flag}")

    if missing:
        print(f"\nwarning: file(s) not found: {', '.join(missing)}")
    if not cpps:
        sys.exit("error: no sequences extracted - check the data directory path")

    # drop known non-CPPs
    for seq in list(cpps):
        if seq in KNOWN_NON_CPP:
            del cpps[seq]
            print(f"\ndropped 1 sequence that MLCPP lists as a non-CPP")

    seqs = sorted(cpps, key=lambda s: (len(s), s))

    # ---- verification -------------------------------------------------
    print(f"\nUnique CPPs: {len(seqs)}")
    print("\nSanity checks (all should be PRESENT):")
    ok = True
    for name, seq in SANITY.items():
        present = seq in cpps
        ok &= present
        print(f"  {name:14s} {'PRESENT' if present else 'MISSING'}")

    rk = sorted((s.count("R") + s.count("K")) / len(s) for s in seqs)
    median_rk = rk[len(rk) // 2]
    print(f"\nMedian R+K fraction: {median_rk:.3f}")
    if median_rk >= 0.20:
        print("  OK - cationic, consistent with real CPPs")
    else:
        ok = False
        print("  WARNING - too low. You may have pulled the non-CPP side.")

    odd = [s for s in seqs if set(s) - STANDARD_AA]
    if odd:
        print(f"\nwarning: {len(odd)} sequence(s) contain non-standard residues")

    # ---- write ---------------------------------------------------------
    csv_path = Path(f"{args.out}.csv")
    fasta_path = Path(f"{args.out}.fasta")
    with open(csv_path, "w", newline="") as fh:
        fh.write("id,label,sequence,length,source\n")
        for i, s in enumerate(seqs, 1):
            fh.write(f"cpp{i},{args.label},{s},{len(s)},{';'.join(sorted(cpps[s]))}\n")
    with open(fasta_path, "w") as fh:
        for i, s in enumerate(seqs, 1):
            fh.write(f">cpp{i}\n{s}\n")

    print(f"\nWrote {csv_path}  ({len(seqs)} rows, label={args.label})")
    print(f"Wrote {fasta_path}")
    if not ok:
        print("\nSome checks failed - inspect the output before using it.")
        sys.exit(1)


if __name__ == "__main__":
    main()