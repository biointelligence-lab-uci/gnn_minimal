# Pretrained AMP vs. CPP classifier

A trained GAT checkpoint that scores a peptide as antimicrobial (AMP) or
cell-penetrating (CPP).

```
checkpoint/
├── gnn_model.pt          # weights (788 KB)
├── gnn_platt_scaling     # logistic calibration on the logit margin
└── model_summary.json    # architecture + feature layout + training metrics
```

## Label direction

**`label 1 = AMP`, `label 0 = CPP`.** `inference.py` writes `prob_AMP`, which is
P(AMP) after Platt scaling, so a score near 1 means AMP and near 0 means CPP.
Read the column as AMP-ness, not as a generic "positive" score.

## Quick check that the checkpoint loads

Needs only `torch`, `torch_geometric`, `biopython` and `scikit-learn` -- no
ESMFold, no ESM2, no GPU, runs in seconds:

```bash
python amp_cpp/sanity_check.py
```

It rebuilds the architecture from `model_summary.json`, loads the weights
strictly, runs a forward pass on synthetic chain graphs, and confirms Platt
scaling returns P(AMP) in [0, 1] rising with the logit margin. It verifies the
checkpoint is intact and matches this repo's model code. It does **not** score
real peptides.

## Scoring real peptides

```bash
python inference.py --model_path amp_cpp/checkpoint --input your_peptides.csv
```

The input CSV needs `id,sequence` columns with unique ids; a `label` column is
optional and only adds `true_label` to the output. Predictions land in
`predictions/<input>_<model>.csv` with `id,predicted_class,prob_AMP`.

**This is the expensive part.** The checkpoint uses per-residue ESM2 node
features (`esm2_residue: true`), so if the input has no adjacent `generated/`
artifacts, `process_data.py` first runs the full pipeline: ESMFold structure
prediction, ESM2 embeddings, and QSAR descriptors. That means multi-gigabyte
model downloads on first run, and it is slow without a GPU. The 800 KB
checkpoint is the cheap part; the feature pipeline is not.

## Architecture

From `model_summary.json`:

| | |
|---|---|
| architecture | GAT, 3 layers, 64 hidden channels, dropout 0.2 |
| pooling | mean + max |
| node features | 20-d one-hot, plus ESM2 1280-d projected to 64-d |
| tabular features | none (`geo_feature_dim: 0`) |
| parameters | 196,674 |
| classes | 2 |

No `tabular_scaler.joblib` ships with this checkpoint, and none is needed --
`geo_feature_dim` is 0, so `resolve_tabular_contract` in `inference.py` returns
an empty contract.

## Metrics

As recorded by the training run that produced this checkpoint (2026-10-06);
Platt calibration was fit on 425 examples.

| metric | value |
|---|---|
| accuracy | 0.9247 |
| precision | 0.9054 |
| recall | 0.9481 |
| F1 | 0.9263 |
| MCC | 0.8504 |
| AUC-ROC | 0.9783 |
| AUC-PR | 0.9774 |

These are the numbers reported at training time, not an independent evaluation.

## Training data

Trained on 2,127 peptides: AMPs with APD3 ids (`AP00022`) and CPPs with CPPsite
ids (`cpp317`). The dataset CSV is not included here -- those databases carry
their own redistribution terms. Check them before republishing the sequences.
