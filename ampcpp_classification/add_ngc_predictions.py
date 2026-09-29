"""
Run the second pretrained GNN -- the NGC pore-forming checkpoint -- over the same
b3ppsexp peptides the AMP/CPP checkpoint scored, and join the two calls.

The NGC checkpoint (GNN_Checkpoints_ngc_generator/) is a QSAR-feature GAT with
esm2_raw_dim=0, so it reuses gnn_minimal/generated/ as-is -- no ESMFold or ESM2
rerun. inference.py always writes its positive-class column as `prob_AMP`
whatever the checkpoint predicts; for this one that value is P(pore-forming), so
it is renamed here to keep the two models' columns unambiguous.

Inference step (already run; re-run it if generated/ changes):
    conda run -n gnn python inference.py \
        --model_path GNN_Checkpoints_ngc_generator \
        --input generated \
        --save_predictions <ngc_preds.csv>

Inputs:
    BBBPPs_AMPCPP_predictions.csv   id, predicted_AMP_CPP, prob_AMP  (1 = AMP)
    <ngc_preds.csv>                 id, predicted_class, prob_AMP    (1 = pore-forming)
    ../b3ppsexp_sequences.csv       id -> sequence
Output:
    BBBPPs_AMPCPP_NGC_predictions.csv
"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GNN_MINIMAL = os.path.dirname(HERE)

ngc_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ngc_preds_raw.csv")

ampcpp = pd.read_csv(os.path.join(HERE, "BBBPPs_AMPCPP_predictions.csv"))
ngc = pd.read_csv(ngc_path)
seqs = pd.read_csv(os.path.join(GNN_MINIMAL, "b3ppsexp_sequences.csv"))

ngc = ngc.rename(columns={"predicted_class": "predicted_NGC_pore",
                          "prob_AMP": "prob_NGC_pore"})

# both models scored the same generated/ folder in the same order, so the ids
# line up exactly -- assert it rather than trusting a merge to be lossless
assert ampcpp.id.tolist() == ngc.id.tolist(), "id order differs between the two runs"

df = ampcpp.merge(ngc[["id", "predicted_NGC_pore", "prob_NGC_pore"]], on="id")
df = df.merge(seqs, on="id", how="left")
df["AMP_CPP_call"] = df.predicted_AMP_CPP.map({1: "AMP", 0: "CPP"})
df["NGC_call"] = df.predicted_NGC_pore.map({1: "pore-forming", 0: "non-pore"})

df = df[["id", "sequence", "predicted_AMP_CPP", "AMP_CPP_call", "prob_AMP",
         "predicted_NGC_pore", "NGC_call", "prob_NGC_pore"]]
out = os.path.join(HERE, "BBBPPs_AMPCPP_NGC_predictions.csv")
df.to_csv(out, index=False)

print(f"n = {len(df)}   missing sequence: {df.sequence.isna().sum()}")
print(f"NGC: {df.NGC_call.value_counts().to_dict()}")
print()
print(pd.crosstab(df.AMP_CPP_call, df.NGC_call))
print(f"\nWrote {out}")
