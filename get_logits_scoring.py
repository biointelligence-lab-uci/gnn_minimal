#!/usr/bin/env python3
"""
Save per-peptide GNN scores (logit margin = "sigma") to CSV.

Usage -- BBBPPs against the NGC checkpoint, run from gnn_minimal/:

  conda run -n gnn python get_logits_scoring.py \
      --model_path ngc/checkpoint \
      --input generated \
      --output ngc/predictions/BBBPPs_NGC_scores.csv

For AMP-vs-CPP scores, use the AMP/CPP checkpoint and set ``--task amp_cpp``:

  python get_logits_scoring.py \
      --model_path results/gnn_models/amp_cpp_model/gat_Graph-only \
      --input generated \
      --output amp_cpp_logits_scores.csv \
      --task amp_cpp

`--input generated` reuses the 318 already-processed structures and descriptors
in generated/; passing b3ppsexp_sequences.csv instead re-runs the structure and
descriptor pipeline for no gain. Add ``--force-process`` if processed artifacts
need to be regenerated (for example, missing per-residue ESM2 tensors).

The default task is ``ngc`` to preserve the existing NGC prediction format.
For ``amp_cpp``, the checkpoint must have been trained with class 1 = AMP.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.artifacts import model_checkpoint_path, platt_scaling_path, resolve_model_dir
from core.checkpoint_meta import load_peptide_gnn_meta, resolve_node_layout_for_checkpoint
from core.data_utils import PeptideGraphDataset
from core.models import PeptideGNN, esm2_raw_dim_from_state_dict, esm2_hidden_dim_from_state_dict
from core.platt import load_platt_json, platt_prob_amp
from process_data import ensure_processed_data, input_has_labels, resolve_input_and_generated
from inference import resolve_tabular_contract, load_processed_features   # reuse inference's helpers


def main():
    ap = argparse.ArgumentParser(description="Dump GNN logit-margin scores to CSV")
    ap.add_argument("--model_path", required=True, help="Model dir or gnn_model.pt")
    ap.add_argument("--input", required=True, help="CSV (id,sequence[,label]) or generated/ dir")
    ap.add_argument("--output", default="scores.csv")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--task",
        choices=("ngc", "amp_cpp"),
        default="ngc",
        help="Score semantics: NGC (class 1 is pore-forming) or AMP/CPP (class 1 is AMP)",
    )
    ap.add_argument(
        "--force-process",
        action="store_true",
        help="Regenerate processed data before scoring",
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    process_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = resolve_model_dir(args.model_path)
    ckpt_path = model_checkpoint_path(model_dir)
    if not ckpt_path.is_file():
        print(f"Error: missing checkpoint {ckpt_path}"); sys.exit(1)

    csv_path, gen_dir = resolve_input_and_generated(args.input)
    has_labels = bool(csv_path) and input_has_labels(csv_path)

    sd0 = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(sd0)
    esm2_h = esm2_hidden_dim_from_state_dict(sd0)

    gen_dir = ensure_processed_data(
        csv_path, gen_dir, need_esm2=esm2_raw > 0,
        require_labels=False, device=process_device, force=args.force_process,
    )
    geo_csv = gen_dir / "geometric_features.csv"
    qsar_csv = gen_dir / "qsar12_descriptors.csv"
    pdb_dir = gen_dir / "structures"
    esm2_residue_dir = gen_dir / "esm2_per_residue"

    meta = load_peptide_gnn_meta(model_dir) or {}
    architecture = meta.get("architecture", "gat")
    hidden_channels = meta.get("hidden_channels", 64)
    num_layers = meta.get("num_layers", 3)
    pooling = meta.get("pooling", "mean_max")
    num_classes = int(meta.get("num_classes", 2))
    if num_classes != 2:
        raise ValueError(f"Expected a binary checkpoint, got num_classes={num_classes}")

    ng_infer, in_base_ckpt, _ = resolve_node_layout_for_checkpoint(
        model_dir, sd0, architecture, user_node_groups=None)
    scaler_path, feature_cols = resolve_tabular_contract(model_dir, meta)
    processed_df = load_processed_features(geo_csv, qsar_csv, feature_cols)

    dataset = PeptideGraphDataset(
        csv_path=str(geo_csv), pdb_dir=str(pdb_dir),
        use_geometric_features=bool(feature_cols),
        geometric_feature_cols=feature_cols,
        tabular_scaler_path=str(scaler_path) if scaler_path else None,
        esm2_residue_dir=str(esm2_residue_dir) if esm2_raw > 0 else None,
        node_feature_groups=ng_infer, dataframe=processed_df,
    )

    geo_dim = 0
    if len(dataset) > 0 and hasattr(dataset[0], "geo_features"):
        geo_dim = int(dataset[0].geo_features.shape[1])

    model = PeptideGNN(
        architecture=architecture, in_channels=in_base_ckpt,
        hidden_channels=hidden_channels, num_layers=num_layers,
        num_classes=num_classes, pooling=pooling,
        geo_feature_dim=geo_dim, esm2_raw_dim=esm2_raw, esm2_hidden_dim=esm2_h,
    )
    model.load_state_dict(sd0)
    model = model.to(device).eval()

    platt = load_platt_json(platt_scaling_path(model_dir))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    margins, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            m = (out[:, 1] - out[:, 0]).cpu().numpy()          # logit margin = sigma
            p = (platt_prob_amp(m, platt["coef"], platt["intercept"])
                 if platt else F.softmax(out, dim=1)[:, 1].cpu().numpy())
            margins.extend(m); probs.extend(p)

    margins, probs = np.array(margins), np.array(probs)
    df = dataset.df
    predicted_class = (probs >= 0.5).astype(int)
    res = {
        "id": df["peptide_id"].astype(str).tolist(),
        "predicted_class": predicted_class,
        "prob_AMP": probs,
        "logit_margin": margins,
    }
    if args.task == "amp_cpp":
        if "sequence" not in df.columns:
            raise ValueError(
                "AMP/CPP scoring requires sequence values in the processed input data"
            )
        res.update({
            "sequence": df["sequence"].astype(str).tolist(),
            "predicted_AMP_CPP": predicted_class,
            "AMP_CPP_call": np.where(predicted_class == 1, "AMP", "CPP"),
            "prob_CPP": 1.0 - probs,
        })
    if has_labels and "label" in df.columns:
        res["true_label"] = df["label"].values
    result_df = pd.DataFrame(res)
    if args.task == "amp_cpp":
        columns = [
            "id",
            "sequence",
            "predicted_AMP_CPP",
            "AMP_CPP_call",
            "prob_AMP",
            "prob_CPP",
            "logit_margin",
            "predicted_class",
        ]
        if "true_label" in result_df.columns:
            columns.append("true_label")
        result_df = result_df[columns]
    result_df.to_csv(args.output, index=False)
    print(f"Saved {len(margins)} scores to {args.output}")


if __name__ == "__main__":
    main()