# -*- coding: utf-8 -*-
"""
Family recovery for baseline encoders.

Each encoder's own 125x125 script distance matrix goes through the same
family-recovery evaluation as the main encoder
(``family_recovery.score_matrix``), with the same genealogical labels.

The distance matrices (one per backbone x method x variant) are computed in
the companion representation-learning project and stored in
``data/baseline_distances/``.

Writes to RESULTS_DIR / "baseline_family_recovery".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from glyphworld.analysis.family_recovery import load_families, score_matrix
from glyphworld.core.paths import DATA_DIR, RESULTS_DIR

BASELINE_DIR = DATA_DIR / "baseline_distances"
OUTPUT_DIR = RESULTS_DIR / "baseline_family_recovery"

# Script-name reconciliation (companion vs worldwide spelling).
NAME_FIXES = {"Ge_ez": "Ge`ez"}

FNAME_PREFIX = "alphabets_charrep_"
FNAME_SUFFIX = "_script-siamese_point-cosine_distances.csv"
BACKBONES = {"contrastive", "koch", "resnet18", "resnet34", "resnet50"}

# Human-readable method names for the paper table.
METHOD_LABELS = {
    "student_teacher": "Teacher-Student (ours)",
    "teacher_only": "Teacher only (SupCon)",
    "barlow_twins": "Barlow Twins",
    "byol_simple": "BYOL",
    "byol_stage2_only": "BYOL (stage-2 only)",
    "dino_frozen": "DINOv2 (frozen)",
    "dino_finetune": "DINOv2 (fine-tuned)",
    "dino_full_finetune": "DINOv2 (full fine-tune)",
}


def _parse(fname: str):
    core = fname[len(FNAME_PREFIX):-len(FNAME_SUFFIX)]
    parts = core.split("_")
    backbone = parts[0]
    assert backbone in BACKBONES, f"unexpected backbone in {fname!r}"
    variant = parts[-1]          # student | teacher
    method = "_".join(parts[1:-1])
    return backbone, method, variant


def _load_matrix(path, script_order) -> np.ndarray:
    df = pd.read_csv(path, index_col=0)
    df = df.rename(index=NAME_FIXES, columns=NAME_FIXES)
    missing = set(script_order) - set(df.index)
    if missing:
        raise ValueError(f"{path.name}: missing scripts {sorted(missing)}")
    df = df.loc[script_order, script_order]
    D = df.values.astype(float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fams = load_families()
    scripts = list(fams.index)

    macro_names = sorted(fams["macro_family"].unique())
    fine_names = sorted(fams["family"].unique())
    y_macro = fams["macro_family"].map({n: i for i, n in enumerate(macro_names)}).values
    y_fine = fams["family"].map({n: i for i, n in enumerate(fine_names)}).values

    rows = []
    for path in sorted(BASELINE_DIR.glob("*.csv")):
        backbone, method, variant = _parse(path.name)
        D = _load_matrix(path, scripts)
        m = score_matrix(D, y_macro, len(macro_names))
        f = score_matrix(D, y_fine, len(fine_names))
        rows.append({
            "method": METHOD_LABELS.get(method, method),
            "method_key": method,
            "backbone": backbone,
            "variant": variant,
            "ARI_macro": m["ARI"], "NMI_macro": m["NMI"],
            "1NN_macro": m["one_nn_purity"], "mAP_macro": m["mAP"],
            "1NN_fine": f["one_nn_purity"], "mAP_fine": f["mAP"],
        })

    df = pd.DataFrame(rows).sort_values("mAP_macro", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)

    # Backbone-controlled view (resnet50) + our model's native backbone.
    view = df[(df["backbone"] == "resnet50") |
              (df["method_key"] == "student_teacher")].copy()
    view.to_csv(OUTPUT_DIR / "baseline_comparison_curated.csv", index=False)

    cols = ["method", "backbone", "variant", "ARI_macro", "NMI_macro",
            "1NN_macro", "mAP_macro", "1NN_fine"]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print("\n=== Family recovery by encoder (all, ranked by macro mAP) ===")
        print(df[cols].round(3).to_string(index=False))
    print(f"\n[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
