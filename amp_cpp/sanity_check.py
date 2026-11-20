#!/usr/bin/env python3
"""
Sanity check for the pretrained AMP/CPP checkpoint.

Verifies that the checkpoint in ``amp_cpp/checkpoint/`` is intact and loadable
without running ESMFold or ESM2: it rebuilds the architecture from
``model_summary.json``, loads the weights strictly, runs a forward pass on a
synthetic graph batch, and checks that Platt scaling returns usable P(AMP).

This proves the checkpoint matches the code in this repo. It does NOT validate
predictions on real peptides -- for that, use inference.py, which needs the
full feature pipeline.

Usage:
  python amp_cpp/sanity_check.py
  python amp_cpp/sanity_check.py --model_path amp_cpp/checkpoint
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.artifacts import model_checkpoint_path, platt_scaling_path, resolve_model_dir
from core.checkpoint_meta import load_peptide_gnn_meta, resolve_node_layout_for_checkpoint
from core.models import (
    PeptideGNN,
    esm2_hidden_dim_from_state_dict,
    esm2_raw_dim_from_state_dict,
)
from core.platt import load_platt_json, platt_prob_amp

PASS = "PASS"
FAIL = "FAIL"


def _fake_peptide_graph(n_residues: int, node_dim: int, esm2_raw_dim: int) -> Data:
    """A chain graph standing in for one peptide: backbone edges only."""
    generator = torch.Generator().manual_seed(n_residues)
    x = torch.zeros(n_residues, node_dim)
    # One-hot-like rows: exactly one residue identity set per node.
    picks = torch.randint(0, node_dim, (n_residues,), generator=generator)
    x[torch.arange(n_residues), picks] = 1.0

    src = torch.arange(n_residues - 1)
    dst = src + 1
    edge_index = torch.cat(
        [torch.stack([src, dst]), torch.stack([dst, src])], dim=1
    ).long()

    data = Data(x=x, edge_index=edge_index)
    if esm2_raw_dim > 0:
        data.esm2_node = torch.randn(n_residues, esm2_raw_dim, generator=generator)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path",
        default=str(Path(__file__).resolve().parent / "checkpoint"),
        help="Model directory or path to gnn_model.pt (default: amp_cpp/checkpoint)",
    )
    args = parser.parse_args()

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        status = PASS if ok else FAIL
        print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    print("AMP/CPP pretrained checkpoint sanity check")
    print("=" * 60)

    # ---- 1. Artifacts present -------------------------------------------
    print("\n1. Artifacts")
    try:
        model_dir = resolve_model_dir(args.model_path)
    except FileNotFoundError as exc:
        print(f"  [{FAIL}] model directory -- {exc}")
        return 1
    print(f"  model_dir: {model_dir}")

    ckpt_path = model_checkpoint_path(model_dir)
    check(ckpt_path.is_file(), "gnn_model.pt present", str(ckpt_path.name))
    if not ckpt_path.is_file():
        return 1

    meta = load_peptide_gnn_meta(model_dir)
    check(meta is not None, "model_summary.json readable")
    if meta is None:
        return 1

    # ---- 2. Metadata self-consistency -----------------------------------
    print("\n2. Metadata")
    architecture = meta.get("architecture", "gat")
    num_classes = int(meta.get("num_classes", 2))
    groups = meta.get("node_feature_groups", {})
    print(f"  architecture={architecture} feature_set={meta.get('feature_set')}")
    print(f"  node_feature_groups={groups}")
    check(num_classes == 2, "binary classifier", f"num_classes={num_classes}")
    check(
        meta.get("kind") == "peptide_gnn",
        "checkpoint kind is peptide_gnn",
        str(meta.get("kind")),
    )

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    esm2_raw = esm2_raw_dim_from_state_dict(sd)
    esm2_hidden = esm2_hidden_dim_from_state_dict(sd)
    check(
        esm2_raw == int(meta.get("esm2_raw_dim", 0)),
        "esm2_raw_dim matches weights",
        f"weights={esm2_raw} summary={meta.get('esm2_raw_dim')}",
    )
    check(
        esm2_hidden == int(meta.get("esm2_hidden_dim", 64)),
        "esm2_hidden_dim matches weights",
        f"weights={esm2_hidden} summary={meta.get('esm2_hidden_dim')}",
    )

    node_groups, node_dim, notes = resolve_node_layout_for_checkpoint(
        model_dir, sd, architecture, user_node_groups=None
    )
    for note in notes:
        print(f"  note: {note}")
    print(f"  base node width: {node_dim}")

    geo_dim = int(meta.get("geo_feature_dim", 0))
    tabular_cols = list(meta.get("tabular_feature_cols") or [])
    check(
        geo_dim == len(tabular_cols),
        "tabular contract consistent",
        f"geo_feature_dim={geo_dim} cols={len(tabular_cols)}",
    )

    # ---- 3. Weights load strictly ---------------------------------------
    print("\n3. Weight loading")
    model = PeptideGNN(
        architecture=architecture,
        in_channels=node_dim,
        hidden_channels=int(meta.get("hidden_channels", 64)),
        num_layers=int(meta.get("num_layers", 3)),
        dropout=float(meta.get("dropout", 0.2)),
        num_classes=num_classes,
        pooling=meta.get("pooling", "mean_max"),
        geo_feature_dim=geo_dim,
        esm2_raw_dim=esm2_raw,
        esm2_hidden_dim=esm2_hidden,
    )
    try:
        model.load_state_dict(sd)
        loaded = True
        detail = f"{sum(p.numel() for p in model.parameters()):,} parameters"
    except (RuntimeError, ValueError) as exc:
        loaded = False
        detail = str(exc)[:200]
    check(loaded, "state_dict loads strictly", detail)
    if not loaded:
        return 1

    finite = all(torch.isfinite(t).all().item() for t in sd.values() if t.is_floating_point())
    check(finite, "no NaN or Inf in weights")

    # ---- 4. Forward pass on synthetic graphs ----------------------------
    print("\n4. Forward pass (synthetic graphs, no ESMFold/ESM2)")
    if geo_dim:
        print(f"  [{FAIL}] this checkpoint needs {geo_dim} tabular features; skipping forward")
        failures.append("forward pass")
        return 1

    model.eval()
    lengths = [8, 17, 23, 42]
    graphs = [_fake_peptide_graph(n, node_dim, esm2_raw) for n in lengths]
    loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
    batch = next(iter(loader))
    with torch.no_grad():
        logits = model(batch)
    check(
        tuple(logits.shape) == (len(lengths), num_classes),
        "logit shape",
        f"{tuple(logits.shape)} for {len(lengths)} graphs",
    )
    check(bool(torch.isfinite(logits).all()), "logits finite")

    # Batched and one-at-a-time must agree, or pooling/batching is wrong.
    with torch.no_grad():
        single = torch.cat(
            [model(next(iter(DataLoader([g], batch_size=1)))) for g in graphs], dim=0
        )
    max_delta = float((logits - single).abs().max())
    check(max_delta < 1e-4, "batched == per-graph", f"max|delta|={max_delta:.2e}")

    # ---- 5. Platt calibration -------------------------------------------
    print("\n5. Platt calibration")
    platt = load_platt_json(platt_scaling_path(model_dir))
    check(platt is not None, "gnn_platt_scaling readable")
    if platt is None:
        return 1
    print(
        f"  coef={platt['coef']:.4f} intercept={platt['intercept']:.4f} "
        f"n_calib={platt.get('n_calib')}"
    )

    margins = (logits[:, 1] - logits[:, 0]).numpy()
    probs = platt_prob_amp(margins, platt["coef"], platt["intercept"])
    check(
        bool(np.all((probs >= 0.0) & (probs <= 1.0))),
        "P(AMP) within [0, 1]",
        f"range=[{probs.min():.4f}, {probs.max():.4f}]",
    )

    # P(AMP) must rise with the logit margin, or the label direction is flipped.
    sweep = np.linspace(-10.0, 10.0, 41)
    swept = platt_prob_amp(sweep, platt["coef"], platt["intercept"])
    check(
        bool(np.all(np.diff(swept) > 0)),
        "P(AMP) increases with margin (label 1 = AMP)",
        f"P(-10)={swept[0]:.4f} -> P(+10)={swept[-1]:.4f}",
    )

    # ---- 6. Reported training metrics -----------------------------------
    print("\n6. Reported metrics (from training, not recomputed here)")
    metrics = meta.get("metrics") or {}
    if metrics:
        for key in ("accuracy", "precision", "recall", "f1", "mcc", "auc_roc", "auc_pr"):
            if key in metrics:
                print(f"  {key:>10}: {float(metrics[key]):.4f}")
    else:
        print("  (none recorded)")

    print("\n" + "=" * 60)
    if failures:
        print(f"{FAIL}: {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"{PASS}: checkpoint is intact and matches this repo's model code.")
    print("Note: this does not score real peptides. Use inference.py for that.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
