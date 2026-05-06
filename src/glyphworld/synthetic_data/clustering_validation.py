# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Worldwide script clustering pipeline.

This script performs the final hierarchical clustering of worldwide scripts
from the learned script distance matrix.

It exports all paper/reproducibility outputs:

RESULTS_DIR / "worldwide_clustering"
    - radial_dendrogram.pdf
    - rectangular_dendrogram.png
    - mds_2d.png
    - mds_2d_coordinates.csv
    - clusters.csv
    - clusters.xlsx
    - cluster_members.csv
    - medoids.csv
    - distance_matrix.npy
    - distance_matrix.csv
    - condensed_distances.npy
    - linkage_matrix.npy
    - linkage_matrix.csv
    - dendrogram_text.txt
    - dendrogram_leaf_order.csv
    - clustering_metrics.csv
    - k_selection_summary.json
    - tree_statistics.json
    - run_summary.txt

Use:
    python clustering.py
"""

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import (
    dendrogram,
    linkage,
    fcluster,
    to_tree,
    cophenet,
)
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS

from glyphworld.core.paths import PROCESSED_DIR, RESULTS_DIR
from glyphworld.visualization import radial_tree


# =====================================================================
# Config
# =====================================================================

IN_PATH = PROCESSED_DIR / "global distances" / "alphabet_similarity_matrix_cosine.pkl"

OUTPUT_DIR = RESULTS_DIR / "worldwide_clustering"

K = 34
LINKAGE_METHOD = "ward"

SHOW_PLOTS = False
SAVE_RADIAL = True
SAVE_RECTANGULAR = True

RADIAL_OUT = OUTPUT_DIR / "radial_dendrogram.pdf"
RECTANGULAR_OUT = OUTPUT_DIR / "rectangular_dendrogram.png"
MDS_OUT = OUTPUT_DIR / "mds_2d.png"

K_MIN = 2
K_MAX = None  # None = number of scripts - 1

FIG_DPI = 300

MDS_RANDOM_STATE = 42
MDS_MAX_ITER = 3000
MDS_EPS = 1e-9


# =====================================================================
# General helpers
# =====================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


# =====================================================================
# Loading distances
# =====================================================================

def load_similarity_pickle(path: Path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    alphabets = list(data["alphabets"])
    alphabets = ["Ge`ez" if a == "Ge_ez" else a for a in alphabets]

    alphabet_distances = data["alphabet_distances"]

    sorted_items = sorted(alphabet_distances.items())
    sorted_keys = [k for k, _ in sorted_items]
    condensed = np.asarray([v for _, v in sorted_items], dtype=float)

    return alphabets, alphabet_distances, sorted_keys, condensed


def condensed_to_square(condensed: np.ndarray) -> np.ndarray:
    m = len(condensed)
    n = int((1 + np.sqrt(1 + 8 * m)) / 2)

    if n * (n - 1) // 2 != m:
        raise ValueError(
            f"Invalid condensed distance vector length: {m}. "
            "Expected n*(n-1)/2."
        )

    return squareform(condensed)


def dict_to_square(alphabet_distances: dict, n: int) -> np.ndarray:
    D = np.zeros((n, n), dtype=float)

    for (i, j), d in alphabet_distances.items():
        D[i, j] = float(d)
        D[j, i] = float(d)

    return D


# =====================================================================
# Dendrograms
# =====================================================================

def plot_rectangular_dendrogram(
    Z: np.ndarray,
    labels: list[str],
    k: int,
    save_path: Path | None = None,
    show: bool = False,
):
    plt.rcParams["figure.figsize"] = [10, 15]
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots(figsize=(10, 15))

    color_threshold = Z[-(k - 1), 2]

    dend = dendrogram(
        Z,
        ax=ax,
        truncate_mode="lastp",
        p=len(labels),
        show_contracted=True,
        labels=labels,
        color_threshold=color_threshold,
        orientation="right",
    )

    ax.tick_params(axis="y", which="major", labelsize=10)
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.set_xlabel("Ward distance")
    ax.set_title(f"Worldwide script clustering, k={k}")

    if save_path is not None:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return dend


def tree_to_text(node, labels: list[str], indent: int = 0) -> str:
    if node.is_leaf():
        return "  " * indent + f"- {labels[node.id]}\n"

    s = "  " * indent + f"+ cluster (dist={node.dist:.4f}, n={node.count})\n"
    s += tree_to_text(node.left, labels, indent + 1)
    s += tree_to_text(node.right, labels, indent + 1)

    return s


# =====================================================================
# Clustering quality metrics
# =====================================================================

def within_cluster_sum_squares_from_square(
    D: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """
    Inertia-like criterion based on pairwise distances.

    For each cluster, distances inside the cluster are centered around their
    cluster mean and squared.
    """
    total = 0.0

    for c in np.unique(cluster_labels):
        idx = np.where(cluster_labels == c)[0]

        if len(idx) <= 1:
            continue

        subD = D[np.ix_(idx, idx)]
        vals = subD[np.triu_indices_from(subD, k=1)]

        if len(vals) == 0:
            continue

        total += float(np.sum((vals - vals.mean()) ** 2))

    return total


def dunn_index(D: np.ndarray, cluster_labels: np.ndarray) -> float:
    clusters = np.unique(cluster_labels)

    max_diam = 0.0
    for c in clusters:
        idx = np.where(cluster_labels == c)[0]

        if len(idx) <= 1:
            continue

        subD = D[np.ix_(idx, idx)]
        diam = float(np.max(subD))
        max_diam = max(max_diam, diam)

    if max_diam <= 0:
        return float("inf")

    min_inter = float("inf")

    for i, ci in enumerate(clusters):
        idx_i = np.where(cluster_labels == ci)[0]

        for cj in clusters[i + 1:]:
            idx_j = np.where(cluster_labels == cj)[0]
            inter = float(np.min(D[np.ix_(idx_i, idx_j)]))
            min_inter = min(min_inter, inter)

    return float(min_inter / max_diam)


def davies_bouldin_from_distance(
    D: np.ndarray,
    cluster_labels: np.ndarray,
) -> float:
    """
    Distance-only Davies--Bouldin style score.

    Scatter = mean distance to cluster medoid.
    Separation = distance between cluster medoids.
    Lower is better.
    """
    clusters = np.unique(cluster_labels)

    medoid_idx = {}
    scatter = {}

    for c in clusters:
        idx = np.where(cluster_labels == c)[0]

        if len(idx) == 1:
            medoid_idx[c] = idx[0]
            scatter[c] = 0.0
            continue

        subD = D[np.ix_(idx, idx)]
        medoid_local = int(np.argmin(subD.sum(axis=1)))
        medoid_global = int(idx[medoid_local])

        medoid_idx[c] = medoid_global
        scatter[c] = float(np.mean(D[idx, medoid_global]))

    values = []

    for ci in clusters:
        worst = -np.inf

        for cj in clusters:
            if ci == cj:
                continue

            sep = float(D[medoid_idx[ci], medoid_idx[cj]])
            if sep <= 0:
                continue

            score = (scatter[ci] + scatter[cj]) / sep
            worst = max(worst, score)

        values.append(worst)

    return float(np.mean(values))


def safe_silhouette_score(D: np.ndarray, cluster_labels: np.ndarray) -> float:
    n_clusters = len(np.unique(cluster_labels))

    if n_clusters <= 1 or n_clusters >= len(cluster_labels):
        return float("nan")

    try:
        return float(silhouette_score(D, cluster_labels, metric="precomputed"))
    except Exception:
        return float("nan")


def evaluate_k_range(
    Z: np.ndarray,
    D: np.ndarray,
    k_values: list[int],
) -> pd.DataFrame:
    rows = []

    for k in k_values:
        labels_k = fcluster(Z, t=k, criterion="maxclust")

        rows.append(
            {
                "k": int(k),
                "n_clusters_observed": int(len(np.unique(labels_k))),
                "inertia": within_cluster_sum_squares_from_square(D, labels_k),
                "davies_bouldin": davies_bouldin_from_distance(D, labels_k),
                "dunn": dunn_index(D, labels_k),
                "silhouette": safe_silhouette_score(D, labels_k),
            }
        )

    df = pd.DataFrame(rows)

    inertia = df["inertia"].to_numpy(dtype=float)
    if len(inertia) >= 3:
        diff2 = np.full(len(inertia), np.nan)
        diff2[1:-1] = np.diff(np.diff(inertia))
        df["inertia_second_difference"] = diff2
    else:
        df["inertia_second_difference"] = np.nan

    return df


def select_best_k(metrics_df: pd.DataFrame) -> dict[str, Any]:
    summary = {}

    valid_elbow = metrics_df.dropna(subset=["inertia_second_difference"])
    if len(valid_elbow) > 0:
        row = valid_elbow.sort_values("inertia_second_difference", ascending=True).iloc[0]
        summary["best_elbow"] = int(row["k"])
        summary["best_elbow_value"] = float(row["inertia_second_difference"])
    else:
        summary["best_elbow"] = None
        summary["best_elbow_value"] = None

    row = metrics_df.sort_values("davies_bouldin", ascending=True).iloc[0]
    summary["best_davies_bouldin"] = int(row["k"])
    summary["best_davies_bouldin_value"] = float(row["davies_bouldin"])

    row = metrics_df.sort_values("dunn", ascending=False).iloc[0]
    summary["best_dunn"] = int(row["k"])
    summary["best_dunn_value"] = float(row["dunn"])

    valid_sil = metrics_df.dropna(subset=["silhouette"])
    if len(valid_sil) > 0:
        row = valid_sil.sort_values("silhouette", ascending=False).iloc[0]
        summary["best_silhouette"] = int(row["k"])
        summary["best_silhouette_value"] = float(row["silhouette"])
    else:
        summary["best_silhouette"] = None
        summary["best_silhouette_value"] = None

    return summary


# =====================================================================
# Tree diagnostics
# =====================================================================

def ultrametric_deviation(condensed: np.ndarray, tau: float = 1e-6):
    D = squareform(condensed)
    n = D.shape[0]

    dev_sum = 0.0
    viol = 0
    total = 0

    for i in range(n - 2):
        Di = D[i]

        for j in range(i + 1, n - 1):
            dij = Di[j]
            Dj = D[j]

            for k in range(j + 1, n):
                dik = Di[k]
                djk = Dj[k]

                a, b, c = np.sort([dij, dik, djk])
                delta = 0.0 if c == 0 else (c - b) / c

                dev_sum += float(delta)
                total += 1

                if delta > tau:
                    viol += 1

    mean_dev = dev_sum / total
    U = 1.0 - mean_dev
    viol_rate = viol / total

    return float(mean_dev), float(U), float(viol_rate)


def sackin_index(root_node) -> int:
    total = 0
    stack = [(root_node, 0)]

    while stack:
        node, depth = stack.pop()

        if node.is_leaf():
            total += depth
        else:
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))

    return int(total)


def colless_index(root_node) -> int:
    total = 0

    def recurse(node) -> int:
        nonlocal total

        if node.is_leaf():
            return 1

        n_left = recurse(node.left)
        n_right = recurse(node.right)

        total += abs(n_left - n_right)

        return n_left + n_right

    recurse(root_node)

    return int(total)


# =====================================================================
# Cluster exports
# =====================================================================

def compute_cluster_assignments(
    labels: list[str],
    clusters: np.ndarray,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Script": labels,
            "Cluster": clusters.astype(int),
        }
    )

    return df.sort_values(["Cluster", "Script"]).reset_index(drop=True)


def compute_cluster_members(cluster_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster, sub in cluster_df.groupby("Cluster"):
        members = sub["Script"].tolist()

        rows.append(
            {
                "Cluster": int(cluster),
                "Size": int(len(members)),
                "Members": "; ".join(members),
            }
        )

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


def compute_medoids(
    labels: list[str],
    D: np.ndarray,
    clusters: np.ndarray,
) -> pd.DataFrame:
    rows = []

    for c in sorted(np.unique(clusters)):
        idx = np.where(clusters == c)[0]
        subD = D[np.ix_(idx, idx)]

        medoid_local = int(np.argmin(subD.sum(axis=1)))
        medoid_global = int(idx[medoid_local])

        rows.append(
            {
                "Cluster": int(c),
                "Size": int(len(idx)),
                "Medoid": labels[medoid_global],
                "MedoidIndex": medoid_global,
                "MeanWithinDistance": float(subD.mean()) if len(idx) > 1 else 0.0,
                "MaxWithinDistance": float(subD.max()) if len(idx) > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("Cluster").reset_index(drop=True)


def compute_leaf_order(dend: dict, labels: list[str], clusters: np.ndarray) -> pd.DataFrame:
    rows = []

    for rank, leaf_idx in enumerate(dend["leaves"], start=1):
        rows.append(
            {
                "LeafOrder": rank,
                "Script": labels[leaf_idx],
                "Cluster": int(clusters[leaf_idx]),
                "DendrogramColor": dend["leaves_color_list"][rank - 1],
            }
        )

    return pd.DataFrame(rows)


# =====================================================================
# 2D MDS projection
# =====================================================================

def compute_mds_2d(
    labels: list[str],
    D: np.ndarray,
    clusters: np.ndarray,
    random_state: int = MDS_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Compute a two-dimensional metric MDS embedding from a precomputed
    script distance matrix.
    """
    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        random_state=random_state,
        max_iter=MDS_MAX_ITER,
        eps=MDS_EPS,
        n_init=8,
    )

    coords = mds.fit_transform(D)

    return pd.DataFrame(
        {
            "Script": labels,
            "Cluster": clusters.astype(int),
            "MDS1": coords[:, 0],
            "MDS2": coords[:, 1],
            "MDSStress": float(mds.stress_),
        }
    )


def plot_mds_2d(
    mds_df: pd.DataFrame,
    save_path: Path,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        mds_df["MDS1"],
        mds_df["MDS2"],
        c=mds_df["Cluster"],
        cmap="tab20",
        s=70,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.3,
    )

    for _, row in mds_df.iterrows():
        ax.annotate(
            row["Script"],
            xy=(row["MDS1"], row["MDS2"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            alpha=0.9,
        )

    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.set_title("2D MDS projection of script distances")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.75)
    cbar.set_label("Cluster")

    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =====================================================================
# Plot metrics
# =====================================================================

def plot_metric_curve(
    metrics_df: pd.DataFrame,
    y: str,
    ylabel: str,
    title: str,
    save_path: Path,
    higher_is_better: bool = True,
):
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(metrics_df["k"], metrics_df[y], marker="o", linewidth=1)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if metrics_df[y].notna().any():
        if higher_is_better:
            best_idx = metrics_df[y].idxmax()
        else:
            best_idx = metrics_df[y].idxmin()

        best_k = metrics_df.loc[best_idx, "k"]
        best_v = metrics_df.loc[best_idx, y]
        ax.scatter([best_k], [best_v], s=80, zorder=5)
        ax.annotate(
            f"k={int(best_k)}",
            xy=(best_k, best_v),
            xytext=(5, 5),
            textcoords="offset points",
        )

    fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def export_metric_plots(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plot_metric_curve(
        metrics_df,
        y="inertia",
        ylabel="Within-cluster inertia",
        title="Elbow method",
        save_path=output_dir / "metric_elbow_inertia.png",
        higher_is_better=False,
    )

    plot_metric_curve(
        metrics_df,
        y="davies_bouldin",
        ylabel="Davies--Bouldin index",
        title="Davies--Bouldin index",
        save_path=output_dir / "metric_davies_bouldin.png",
        higher_is_better=False,
    )

    plot_metric_curve(
        metrics_df,
        y="dunn",
        ylabel="Dunn index",
        title="Dunn index",
        save_path=output_dir / "metric_dunn.png",
        higher_is_better=True,
    )

    plot_metric_curve(
        metrics_df,
        y="silhouette",
        ylabel="Silhouette score",
        title="Silhouette score",
        save_path=output_dir / "metric_silhouette.png",
        higher_is_better=True,
    )


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    alphabets, alphabet_distances, sorted_keys, condensed = load_similarity_pickle(IN_PATH)
    n_scripts = len(alphabets)

    print(f"Number of scripts: {n_scripts}")
    print(f"Input distance file: {IN_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")

    D_full = condensed_to_square(condensed)
    D_dict = dict_to_square(alphabet_distances, n=n_scripts)

    if not np.allclose(D_full, D_dict):
        print("Warning: squareform distance matrix differs from dictionary matrix.")
        D_export = D_dict
    else:
        D_export = D_full

    # -----------------------------------------------------------------
    # Linkage
    # -----------------------------------------------------------------
    Z = linkage(condensed, method=LINKAGE_METHOD)

    # -----------------------------------------------------------------
    # Final cluster assignment
    # -----------------------------------------------------------------
    clusters = fcluster(Z, t=K, criterion="maxclust")

    cluster_df = compute_cluster_assignments(alphabets, clusters)
    cluster_members_df = compute_cluster_members(cluster_df)
    medoids_df = compute_medoids(alphabets, D_export, clusters)

    # -----------------------------------------------------------------
    # Dendrograms
    # -----------------------------------------------------------------
    dend = plot_rectangular_dendrogram(
        Z=Z,
        labels=alphabets,
        k=K,
        save_path=RECTANGULAR_OUT if SAVE_RECTANGULAR else None,
        show=SHOW_PLOTS,
    )

    if SAVE_RADIAL:
        radial_tree.plot_radial_dendrogram(
            dend,
            fontsize=20,
            show=SHOW_PLOTS,
            show_labels=True,
            save_path=str(RADIAL_OUT),
        )

    leaf_order_df = compute_leaf_order(dend, alphabets, clusters)

    # -----------------------------------------------------------------
    # 2D MDS projection
    # -----------------------------------------------------------------
    mds_df = compute_mds_2d(alphabets, D_export, clusters)
    plot_mds_2d(mds_df, save_path=MDS_OUT, show=SHOW_PLOTS)

    root, _nodes = to_tree(Z, rd=True)
    dendrogram_text = tree_to_text(root, alphabets)

    # -----------------------------------------------------------------
    # Metrics over k
    # -----------------------------------------------------------------
    k_max = K_MAX if K_MAX is not None else n_scripts - 1
    k_values = list(range(K_MIN, k_max + 1))

    metrics_df = evaluate_k_range(Z, D_export, k_values)
    k_summary = select_best_k(metrics_df)
    k_summary["chosen_k"] = int(K)
    k_summary["k_min"] = int(K_MIN)
    k_summary["k_max"] = int(k_max)
    k_summary["linkage_method"] = LINKAGE_METHOD

    export_metric_plots(metrics_df, OUTPUT_DIR)

    # -----------------------------------------------------------------
    # Tree diagnostics
    # -----------------------------------------------------------------
    ccc, coph_dists = cophenet(Z, condensed)

    mean_dev, U, viol_rate = ultrametric_deviation(condensed, tau=1e-6)
    mean_dev_coph, U_coph, viol_rate_coph = ultrametric_deviation(coph_dists, tau=1e-6)

    S = sackin_index(root)
    C = colless_index(root)

    C_max = (n_scripts - 1) * (n_scripts - 2) / 2
    S_norm = S / (n_scripts * np.log2(n_scripts))
    C_norm = C / C_max if C_max > 0 else 0.0

    tree_stats = {
        "n_scripts": int(n_scripts),
        "linkage_method": LINKAGE_METHOD,
        "chosen_k": int(K),
        "cophenetic_correlation_coefficient": float(ccc),
        "input_distance_ultrametric_mean_deviation": float(mean_dev),
        "input_distance_ultrametricity_coefficient": float(U),
        "input_distance_ultrametric_violation_rate": float(viol_rate),
        "cophenetic_ultrametric_mean_deviation": float(mean_dev_coph),
        "cophenetic_ultrametricity_coefficient": float(U_coph),
        "cophenetic_ultrametric_violation_rate": float(viol_rate_coph),
        "sackin_index": int(S),
        "sackin_normalized": float(S_norm),
        "colless_index": int(C),
        "colless_normalized": float(C_norm),
    }

    # -----------------------------------------------------------------
    # Exports
    # -----------------------------------------------------------------

    # Cluster outputs
    cluster_df.to_csv(OUTPUT_DIR / "clusters.csv", index=False)
    cluster_df.to_excel(OUTPUT_DIR / "clusters.xlsx", index=False)

    cluster_members_df.to_csv(OUTPUT_DIR / "cluster_members.csv", index=False)
    medoids_df.to_csv(OUTPUT_DIR / "medoids.csv", index=False)
    leaf_order_df.to_csv(OUTPUT_DIR / "dendrogram_leaf_order.csv", index=False)
    mds_df.to_csv(OUTPUT_DIR / "mds_2d_coordinates.csv", index=False)

    # Distance / linkage outputs
    np.save(OUTPUT_DIR / "condensed_distances.npy", condensed)
    np.save(OUTPUT_DIR / "distance_matrix.npy", D_export)
    np.save(OUTPUT_DIR / "linkage_matrix.npy", Z)

    pd.DataFrame(
        D_export,
        index=alphabets,
        columns=alphabets,
    ).to_csv(OUTPUT_DIR / "distance_matrix.csv")

    pd.DataFrame(
        Z,
        columns=["cluster_1", "cluster_2", "distance", "n_leaves"],
    ).to_csv(OUTPUT_DIR / "linkage_matrix.csv", index=False)

    # Text / JSON outputs
    with open(OUTPUT_DIR / "dendrogram_text.txt", "w", encoding="utf-8") as f:
        f.write(dendrogram_text)

    with open(OUTPUT_DIR / "k_selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(k_summary), f, indent=2)

    with open(OUTPUT_DIR / "tree_statistics.json", "w", encoding="utf-8") as f:
        json.dump(to_jsonable(tree_stats), f, indent=2)

    metrics_df.to_csv(OUTPUT_DIR / "clustering_metrics.csv", index=False)

    # Run summary
    run_summary = [
        "Worldwide clustering summary",
        "=" * 50,
        "",
        f"Input file: {IN_PATH}",
        f"Output directory: {OUTPUT_DIR}",
        f"Number of scripts: {n_scripts}",
        f"Linkage method: {LINKAGE_METHOD}",
        f"Chosen k: {K}",
        "",
        "Cluster sizes:",
        cluster_members_df[["Cluster", "Size"]].to_string(index=False),
        "",
        "Medoids:",
        medoids_df[["Cluster", "Size", "Medoid"]].to_string(index=False),
        "",
        "k-selection summary:",
        json.dumps(to_jsonable(k_summary), indent=2),
        "",
        "Tree statistics:",
        json.dumps(to_jsonable(tree_stats), indent=2),
        "",
        "Exported files:",
        "  - radial_dendrogram.pdf",
        "  - rectangular_dendrogram.png",
        "  - mds_2d.png",
        "  - mds_2d_coordinates.csv",
        "  - clusters.csv",
        "  - clusters.xlsx",
        "  - cluster_members.csv",
        "  - medoids.csv",
        "  - distance_matrix.npy",
        "  - distance_matrix.csv",
        "  - condensed_distances.npy",
        "  - linkage_matrix.npy",
        "  - linkage_matrix.csv",
        "  - dendrogram_text.txt",
        "  - dendrogram_leaf_order.csv",
        "  - clustering_metrics.csv",
        "  - metric_elbow_inertia.png",
        "  - metric_davies_bouldin.png",
        "  - metric_dunn.png",
        "  - metric_silhouette.png",
        "  - k_selection_summary.json",
        "  - tree_statistics.json",
        "  - run_summary.txt",
    ]

    with open(OUTPUT_DIR / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(run_summary))

    # Console summary
    print("\n--- Clustering complete ---")
    print(f"Chosen k: {K}")
    print(f"CCC: {ccc:.4f}")
    print(f"MDS stress: {mds_df['MDSStress'].iloc[0]:.4f}")
    print(f"Sackin normalized: {S_norm:.4f}")
    print(f"Colless normalized: {C_norm:.4f}")
    print(f"All results exported to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()