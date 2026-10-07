# ngc/ — NGC pore-forming classifier

Everything produced by the **NGC checkpoint**: the model, its predictions on the
318 BBBPPs, and every table and figure derived from them.

> **Naming:** "NGC generator" is the *peptide class this model classifies* —
> negative Gaussian curvature generators, i.e. pore-formers. It is **not** a
> generative model and it generated no sequences or structures. The checkpoint is
> a discriminative GAT; its positive-class output is P(pore-forming).

## Layout

```
ngc/
├── checkpoint/      the model (was GNN_Checkpoints_ngc_generator/)
├── predictions/     raw inference output on the 318 BBBPPs
│   └── archive/     older duplicate runs, kept for provenance
├── qsar/            12 QSAR descriptors, split by NGC x AMP/CPP group
├── tables/          ranked and reference-annotated tables
├── figures/
└── scripts/
```

`inference.py` writes its positive class as `prob_AMP` whatever the checkpoint
predicts; for this one that value is P(pore-forming), renamed `prob_NGC_pore`
downstream.

### predictions/

| File | Contents |
|---|---|
| `BBBPPs_NGC_scores.csv` | **canonical** — adds `logit_margin` (sigma) to the call |
| `ngc_preds_raw.csv` | strict subset of the above; kept only as the documented `inference.py` output |
| `BBBPPs_AMPCPP_NGC_predictions.csv` | NGC joined to the GNN AMP/CPP calls |
| `BBBPPs_NGC_vs_AMPCPP_SVM.csv` | NGC joined to **both** the GNN and ESM2-SVM AMP/CPP calls |
| `archive/predictions.csv`, `archive/Sabrina_predictions.csv` | earlier NGC runs, same checkpoint |

### The 318 BBBPPs, NGC x AMP/CPP (GNN)

|  | non-pore | pore-forming | total |
|---|---|---|---|
| **AMP** | 58 | 20 | 78 |
| **CPP** | 202 | 38 | 240 |
| **total** | 260 | 58 | 318 |

The ESM2-SVM is more conservative about AMP (54 vs 78) and agrees with the GNN on
86.8% of calls. Both models put pore-formers at a higher AMP rate than non-pore
(GNN 34.5% vs 22.3%; SVM 27.6% vs 14.6%).

## Regenerating

Run from `gnn_minimal/`. Scripts resolve paths from their own location, so the
working directory does not matter.

```bash
# 1. score the 318 structures in generated/ (reuses them; no ESMFold/ESM2 rerun)
conda run -n gnn python get_logits_scoring.py \
    --model_path ngc/checkpoint \
    --input generated \
    --output ngc/predictions/BBBPPs_NGC_scores.csv

# 2. join onto the AMP/CPP calls, then build every table and figure
python ngc/scripts/add_ngc_predictions.py
python ngc/scripts/ngc_poreformers_annotated.py    # -> tables/*_annotated.csv
python ngc/scripts/ngc_generators_table.py         # -> tables/ngc_generators.md
python ngc/scripts/make_excel.py                   # -> tables/ngc_generators.xlsx
python ngc/scripts/md_to_pdf.py                    # -> tables/ngc_generators.pdf
python ngc/scripts/sigma_score_ranking.py          # -> tables/ngc_generators_ranked.csv
python ngc/scripts/make_group_feature_tables.py    # -> qsar/{ngc,nonngc}_{amp,cpp}_qsar12.csv
python ngc/scripts/plot_ngc_by_ampcpp.py           # -> figures/
```

## What deliberately stays outside this folder

- `../generated/` — ESMFold structures and descriptors, **input** to the model and
  shared with the AMP/CPP checkpoint, so it belongs to neither.
- `../ampcpp_classification/` — the AMP/CPP-only outputs
  (`BBBPPs_AMPCPP_predictions.csv`, `amp_vs_cpp_counts.png`), from a different
  checkpoint under `../results/gnn_models/AMPCPP_run_*/`.

Note `qsar/BBBPPs_AMPCPP_qsar12.csv` lives here despite its name: it carries
`NGC_call`/`prob_NGC_pore` and is the parent of the four group tables.

## Known data issues

- `../b3ppsexp_sequences.csv` keeps trailing whitespace on some ids that the GNN
  pipeline strips. `add_ngc_predictions.py` normalises both sides before merging;
  without that one peptide silently lost its sequence.
- The same file has one row with an **empty id** (sequence `SLKP`), which carries
  through into the prediction tables.
- `../generated/esm2_per_residue/` is empty — correct, since this checkpoint is
  `esm2_raw_dim=0` — but `process_data.py` tests it with `.is_dir()` rather than
  checking for contents, so a checkpoint that *does* need ESM2 would read it as
  satisfied and get nothing.
