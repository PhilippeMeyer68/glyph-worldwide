# -*- coding: utf-8 -*-
"""
Recovery of expert script families by the learned script distances.

Genealogical family labels following Daniels & Bright
(``data/info/scripts_families.csv``) are compared with the 125x125 script
distance matrix, as a sanity check rather than a ground truth.

For both a broad (macro_family) and a fine (family) labeling:
  * clustering agreement  : ARI, NMI, homogeneity/completeness/V-measure, B-cubed
  * cut-free retrieval    : 1-NN family purity, mAP, precision@k (leave-one-out)
  * per-family breakdown  : size, 1-NN purity, mAP
  * nearest-neighbour dump : to inspect the deviations

Writes to RESULTS_DIR / "family_recovery".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)

from glyphworld.core.paths import RESULTS_DIR, INFO_DIR

DISTANCE_MATRIX = RESULTS_DIR / "worldwide_clustering" / "distance_matrix.csv"
FAMILIES_CSV = INFO_DIR / "scripts_families.csv"
OUTPUT_DIR = RESULTS_DIR / "family_recovery"

LABELINGS = ["macro_family", "family"]  # broad, then fine
EXTRA_CUTS = [34]                        # k used for the qualitative figure


def _load():
    dm = pd.read_csv(DISTANCE_MATRIX, index_col=0)
    scripts = list(dm.index)
    D = dm.values.astype(float)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    fams = pd.read_csv(FAMILIES_CSV).set_index("script")
    missing = [s for s in scripts if s not in fams.index]
    if missing:
        raise ValueError(f"scripts without a family label: {missing}")
    fams = fams.loc[scripts]
    return scripts, D, fams


def _bcubed(y: np.ndarray, pred: np.ndarray):
    n = len(y)
    P = np.empty(n)
    R = np.empty(n)
    for i in range(n):
        same_c = pred == pred[i]
        same_l = y == y[i]
        both = np.count_nonzero(same_c & same_l)
        P[i] = both / same_c.sum()
        R[i] = both / same_l.sum()
    p, r = P.mean(), R.mean()
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def _retrieval(D: np.ndarray, y: np.ndarray):
    """Leave-one-out, cut-free retrieval metrics."""
    n = len(y)
    nn1 = 0
    aps, p1, p3, p5 = [], [], [], []
    for i in range(n):
        order = np.argsort(D[i])
        order = order[order != i]
        same = (y[order] == y[i]).astype(int)
        nn1 += int(same[0])
        p1.append(same[:1].mean())
        p3.append(same[:3].mean())
        p5.append(same[:5].mean())
        R = int(same.sum())
        if R > 0:
            csum = np.cumsum(same)
            prec = csum / np.arange(1, len(same) + 1)
            aps.append((prec * same).sum() / R)
    return {
        "one_nn_purity": nn1 / n,
        "mAP": float(np.mean(aps)),
        "precision_at_1": float(np.mean(p1)),
        "precision_at_3": float(np.mean(p3)),
        "precision_at_5": float(np.mean(p5)),
    }


def load_families() -> pd.DataFrame:
    """Genealogical family labels (Daniels & Bright), indexed by script."""
    return pd.read_csv(FAMILIES_CSV).set_index("script")


def score_matrix(D: np.ndarray, y: np.ndarray, n_clusters: int) -> dict:
    """Compact recovery score for one distance matrix and one labeling:
    Ward-clustering agreement (ARI, NMI) plus cut-free retrieval.
    Reused by the baseline comparison (baseline_family_recovery)."""
    Z = linkage(squareform(D, checks=False), method="ward")
    pred = fcluster(Z, n_clusters, criterion="maxclust")
    retr = _retrieval(D, y)
    return {
        "ARI": adjusted_rand_score(y, pred),
        "NMI": normalized_mutual_info_score(y, pred),
        "one_nn_purity": retr["one_nn_purity"],
        "mAP": retr["mAP"],
        "precision_at_3": retr["precision_at_3"],
    }


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scripts, D, fams = _load()
    Z = linkage(squareform(D, checks=False), method="ward")

    summary_rows, perfam_rows, retr_rows = [], [], []

    for col in LABELINGS:
        names = sorted(fams[col].unique())
        y = fams[col].map({nm: i for i, nm in enumerate(names)}).values
        nfam = len(names)

        # clustering agreement at k = #families and at extra cuts
        for k in [nfam, *EXTRA_CUTS]:
            pred = fcluster(Z, k, criterion="maxclust")
            h, c, v = homogeneity_completeness_v_measure(y, pred)
            bp, br, bf = _bcubed(y, pred)
            summary_rows.append({
                "labeling": col, "n_families": nfam, "k": k,
                "ARI": adjusted_rand_score(y, pred),
                "NMI": normalized_mutual_info_score(y, pred),
                "homogeneity": h, "completeness": c, "v_measure": v,
                "bcubed_precision": bp, "bcubed_recall": br, "bcubed_f1": bf,
            })

        # cut-free retrieval
        retr = _retrieval(D, y)
        retr_rows.append({"labeling": col, "n_families": nfam, **retr})

        # per-family retrieval breakdown
        for nm in names:
            idx = np.where(fams[col].values == nm)[0]
            nn = 0
            aps = []
            for i in idx:
                order = np.argsort(D[i])
                order = order[order != i]
                same = (y[order] == y[i]).astype(int)
                nn += int(same[0])
                R = int(same.sum())
                if R > 0:
                    csum = np.cumsum(same)
                    prec = csum / np.arange(1, len(same) + 1)
                    aps.append((prec * same).sum() / R)
            perfam_rows.append({
                "labeling": col, "family": nm, "size": len(idx),
                "one_nn_purity": nn / len(idx),
                "mAP": float(np.mean(aps)) if aps else 0.0,
            })

    # nearest-neighbour dump (error analysis) using the fine labeling
    nn_rows = []
    yf = fams["family"].values
    ymf = fams["macro_family"].values
    for i, s in enumerate(scripts):
        order = np.argsort(D[i])
        order = order[order != i]
        j = order[0]
        nn_rows.append({
            "script": s, "family": yf[i], "macro_family": ymf[i],
            "nearest": scripts[j], "nearest_family": yf[j],
            "nearest_macro_family": ymf[j],
            "same_family": bool(yf[i] == yf[j]),
            "same_macro_family": bool(ymf[i] == ymf[j]),
            "distance": float(D[i, j]),
        })

    summary = pd.DataFrame(summary_rows)
    retr = pd.DataFrame(retr_rows)
    perfam = pd.DataFrame(perfam_rows)
    nn = pd.DataFrame(nn_rows)

    summary.to_csv(OUTPUT_DIR / "family_recovery_summary.csv", index=False)
    retr.to_csv(OUTPUT_DIR / "family_recovery_retrieval.csv", index=False)
    perfam.to_csv(OUTPUT_DIR / "family_recovery_per_family.csv", index=False)
    nn.to_csv(OUTPUT_DIR / "family_recovery_nearest_neighbours.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_rows", None):
        print("\n=== Clustering agreement (Ward partition vs family labels) ===")
        print(summary.round(3).to_string(index=False))
        print("\n=== Cut-free retrieval (leave-one-out) ===")
        print(retr.round(3).to_string(index=False))
        print("\n=== Per-family retrieval ===")
        print(perfam.round(3).to_string(index=False))
        misses = nn[~nn["same_macro_family"]]
        print(f"\n=== 1-NN crosses macro-family for {len(misses)}/{len(nn)} scripts "
              f"(visual != genealogy) ===")
        print(misses[["script", "macro_family", "nearest", "nearest_macro_family"]]
              .to_string(index=False))
    print(f"\n[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
