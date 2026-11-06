# -*- coding: utf-8 -*-
"""
Knee scan of the worldwide clustering curves.

For each internal clustering curve of the worldwide hierarchy (inertia,
log-inertia, Ward merge heights, silhouette, Davies--Bouldin, Dunn), the knee
is the k farthest from the chord joining the points k_min and k_max, both axes
being scaled to [0, 1]. Repeating this over a grid of (k_min, k_max) windows
shows which resolutions the curves agree on, independently of the bounds.

Reads (output of the worldwide clustering script):
    RESULTS_DIR / "worldwide_clustering" / "clustering_metrics.csv"
    RESULTS_DIR / "worldwide_clustering" / "linkage_matrix.npy"

Writes to RESULTS_DIR / "worldwide_clustering":
    - knee_scan.csv          knee per (curve, k_min, k_max)
    - knee_scan_summary.csv  median knee and share in K_RANGE, by k_min regime
    - knee_scan.png          heatmaps of the knee location
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from glyphworld.core.paths import RESULTS_DIR

OUTPUT_DIR = RESULTS_DIR / "worldwide_clustering"

CURVES = ["inertia", "log_inertia", "merge_height", "silhouette", "davies_bouldin", "dunn"]
K_MIN = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
K_MAX = [40, 50, 60, 70, 80, 90, 100, 110, 124]
K_RANGE = (27, 38)          # intermediate range reported in the paper
COARSE_K_MIN = 4            # windows with k_min <= 4 include the first splits


def load_curves() -> pd.DataFrame:
    d = pd.read_csv(OUTPUT_DIR / "clustering_metrics.csv")
    d = d[d["n_clusters_observed"] == d["k"]].copy()
    Z = np.load(OUTPUT_DIR / "linkage_matrix.npy")
    n = Z.shape[0] + 1
    d["merge_height"] = d["k"].map({k: Z[n - k, 2] for k in range(2, n + 1)})  # merge k -> k-1
    d["log_inertia"] = np.log(d["inertia"].where(d["inertia"] > 0))
    return d


def knee(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.ptp(y) == 0:
        return np.nan
    xs = (x - x.min()) / np.ptp(x)
    ys = (y - y.min()) / np.ptp(y)
    p0, p1 = np.array([xs[0], ys[0]]), np.array([xs[-1], ys[-1]])
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    v = np.stack([xs, ys], 1) - p0
    return float(x[np.argmax(np.abs(v[:, 0] * u[1] - v[:, 1] * u[0]))])


def main() -> None:
    d = load_curves()
    rows = []
    for c in CURVES:
        for a in K_MIN:
            for b in K_MAX:
                w = d[(d["k"] >= a) & (d["k"] <= b)]
                rows.append({"curve": c, "k_min": a, "k_max": b,
                             "knee": knee(w["k"].to_numpy(float), w[c].to_numpy(float))})
    scan = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scan.to_csv(OUTPUT_DIR / "knee_scan.csv", index=False)

    lo, hi = K_RANGE
    summary = []
    for regime, sub in [("k_min<=4", scan[scan.k_min <= COARSE_K_MIN]),
                        ("k_min>=8", scan[scan.k_min >= 8])]:
        for c in CURVES:
            k = sub.loc[sub.curve == c, "knee"].dropna()
            summary.append({"regime": regime, "curve": c, "median_knee": k.median(),
                            f"share_in_{lo}_{hi}": ((k >= lo) & (k <= hi)).mean()})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUTPUT_DIR / "knee_scan_summary.csv", index=False)
    print(summary.round(2).to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, c in zip(axes.flat, CURVES):
        s = scan[scan.curve == c].pivot(index="k_min", columns="k_max", values="knee")
        im = ax.imshow(s.values, cmap="viridis", aspect="auto", vmin=2, vmax=60)
        ax.set_xticks(range(len(s.columns)), s.columns)
        ax.set_yticks(range(len(s.index)), s.index)
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                v = s.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, int(v), ha="center", va="center", fontsize=7,
                            color="white" if v < 35 else "black")
        ax.set_title(c.replace("_", " "))
        ax.set_xlabel("$k_{max}$")
        ax.set_ylabel("$k_{min}$")
    fig.colorbar(im, ax=axes, shrink=0.6, label="knee $k$")
    fig.savefig(OUTPUT_DIR / "knee_scan.png", dpi=200)
    plt.close(fig)
    print(f"[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
