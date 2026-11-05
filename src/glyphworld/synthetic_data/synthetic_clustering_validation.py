"""
Select the clustering method on the synthetic clustering-validation dataset.

Each synthetic script is generated from a single source alphabet, which gives
a ground-truth ancestor partition (63 clusters). Script distances are computed
with the selected script metric (symmetric nearest-neighbour mean,
sym_min_mean), and each clustering method is asked for as many clusters as
there are ancestors.

Evaluation:
    - ARI / NMI against the ancestor labels
    - Silhouette score on the precomputed distance matrix
    - Hierarchical methods only: cophenetic correlation, and Spearman rho /
      pairwise AUCs between the cophenetic distance of each labelled pair and
      its close / distant / unrelated label

Outputs (RESULTS_DIR / "validation" / "clustering_validation_evaluation"):
    - clustering_method_comparison.csv
    - hierarchical_linkage_comparison.csv
    - flat_clustering_comparison.csv
    - clustering_validation_summary.txt
    - synthetic_script_distance_matrix.npy
    - synthetic_script_ids.json
    - synthetic_script_ancestors.csv
    - linkage_{method}.npy, dendrogram_{method}.png
    - ranking_{score}.png

Use:
    python synthetic_clustering_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.manifold import MDS
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn_extra.cluster import KMedoids

from glyphworld.core.paths import RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary
from glyphworld.synthetic_data.metric_validation import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MODEL_KIND,
    DEFAULT_USE_TEACHER,
    DEVICE,
    compute_all_script_embeddings,
    load_metric_model,
    pairwise_auc,
)


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_DATASET_PATH = (
    RESULTS_DIR / "validation" / "synthetic_pairs_clustering_validation.pkl"
)

DEFAULT_OUTPUT_DIR = (
    RESULTS_DIR / "validation" / "clustering_validation_evaluation"
)

HIERARCHICAL_LINKAGES = ["ward", "complete", "average", "weighted", "single"]

RANDOM_STATE = 42
MDS_COMPONENTS = 20
KMEANS_N_INIT = 10

RANKING_SCORES = [
    "ari",
    "nmi",
    "silhouette",
    "cophenetic_corr",
    "pair_spearman_rho",
    "pair_mean_auc",
]

DISTANCE_METRIC_LABEL = (
    "similarity_between_languages(dist_dict_from_embeddings_cosine(z1, z2))"
)


# ---------------------------------------------------------------------
# Script distance matrix
# ---------------------------------------------------------------------
def _l2_normalize(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)


def sym_min_mean_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    """
    Vectorised equivalent of
    similarity_between_languages(dist_dict_from_embeddings_cosine(z1, z2)):
    same float32 cosine distances, same symmetric nearest-neighbour mean.
    """
    D = (1.0 - z1 @ z2.T).astype(np.float32)
    row_mean = np.mean(D.min(axis=1).astype(np.float64))
    col_mean = np.mean(D.min(axis=0).astype(np.float64))
    return float((row_mean + col_mean) / 2.0)


def build_script_distance_matrix(
    script_ids: list[str],
    script_embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    Z = [_l2_normalize(script_embeddings[sid]) for sid in script_ids]
    n = len(Z)
    D = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = sym_min_mean_distance(Z[i], Z[j])

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"[INFO] distance matrix rows: {i + 1}/{n}")

    return D


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def partition_scores(
    D: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "silhouette": float(silhouette_score(D, y_pred, metric="precomputed")),
    }


def pair_scores_from_cophenetic(
    coph_square: np.ndarray,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_labels: np.ndarray,
) -> dict[str, Any]:
    d = coph_square[pair_left, pair_right]
    rho, p_value = spearmanr(-d, pair_labels)

    auc_1_vs_05 = pairwise_auc(pair_labels, d, 1.0, 0.5)
    auc_1_vs_0 = pairwise_auc(pair_labels, d, 1.0, 0.0)
    auc_05_vs_0 = pairwise_auc(pair_labels, d, 0.5, 0.0)

    return {
        "pair_spearman_rho": float(rho),
        "pair_spearman_p": float(p_value),
        "pair_mean_auc": float(np.mean([auc_1_vs_05, auc_1_vs_0, auc_05_vs_0])),
        "pair_auc_close_vs_distant": auc_1_vs_05,
        "pair_auc_close_vs_unrelated": auc_1_vs_0,
        "pair_auc_distant_vs_unrelated": auc_05_vs_0,
        "n_pairs_used": int(len(d)),
    }


def evaluate_hierarchical(
    D: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
    pair_left: np.ndarray,
    pair_right: np.ndarray,
    pair_labels: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    condensed = squareform(D, checks=False)
    rows = []
    linkages = {}

    for method in HIERARCHICAL_LINKAGES:
        Z = linkage(condensed, method=method)
        y_pred = fcluster(Z, t=n_clusters, criterion="maxclust")
        coph_corr, coph_condensed = cophenet(Z, condensed)

        rows.append(
            {
                "method": f"hierarchical_{method}",
                "family": "hierarchical",
                "n_clusters": int(len(np.unique(y_pred))),
                **partition_scores(D, y_true, y_pred),
                "linkage": method,
                "cophenetic_corr": float(coph_corr),
                **pair_scores_from_cophenetic(
                    squareform(coph_condensed), pair_left, pair_right, pair_labels
                ),
            }
        )
        linkages[method] = Z

    return rows, linkages


def evaluate_flat(
    D: np.ndarray,
    y_true: np.ndarray,
    n_clusters: int,
) -> list[dict[str, Any]]:
    D64 = D.astype(np.float64)

    # Euclidean embedding of the dissimilarity, used by the MDS baselines.
    X_mds = MDS(
        n_components=MDS_COMPONENTS,
        dissimilarity="precomputed",
        random_state=RANDOM_STATE,
    ).fit_transform(D64)

    # Gaussian affinity with the median non-zero distance as bandwidth.
    sigma = np.median(D64[D64 > 0])
    affinity = np.exp(-(D64 ** 2) / (2.0 * sigma ** 2))

    predictions = {
        "kmedoids": KMedoids(
            n_clusters=n_clusters,
            metric="precomputed",
            method="alternate",
            init="k-medoids++",
            random_state=RANDOM_STATE,
        ).fit_predict(D64),
        "agglomerative_ward_mds": AgglomerativeClustering(
            n_clusters=n_clusters, linkage="ward"
        ).fit_predict(X_mds),
        "kmeans_mds": KMeans(
            n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT
        ).fit_predict(X_mds),
        "spectral": SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=RANDOM_STATE
        ).fit_predict(affinity),
        "agglomerative_complete": AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="complete"
        ).fit_predict(D64),
        "agglomerative_average": AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        ).fit_predict(D64),
    }

    rows = []
    for method, y_pred in predictions.items():
        rows.append(
            {
                "method": method,
                "family": "flat",
                "n_clusters": int(len(np.unique(y_pred))),
                **partition_scores(D64, y_true, y_pred),
                "n_pairs_used": 0,
            }
        )

    return rows


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_dendrogram(Z: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(18, 7))
    dendrogram(Z, no_labels=True, color_threshold=None)
    plt.title(title)
    plt.ylabel("Merge height")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ranking(comparison: pd.DataFrame, score: str, out_path: Path) -> None:
    sub = comparison.dropna(subset=[score]).sort_values(score, ascending=True)

    plt.figure(figsize=(8, max(3.0, 0.4 * len(sub) + 1.0)))
    plt.barh(sub["method"], sub[score])
    plt.xlabel(score)
    plt.title(f"Clustering methods ranked by {score}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def write_text_summary(path: Path, summary: dict[str, Any], comparison: pd.DataFrame) -> None:
    cfg = summary["dataset_config"]
    best = comparison.iloc[0]

    lines = [
        "Synthetic clustering validation",
        "=" * 50,
        "",
        f"Dataset: {summary['dataset_path']}",
        f"Output directory: {summary['output_dir']}",
        "",
        "Dataset configuration:",
        f"  dataset_name: {cfg.get('dataset_name')}",
        f"  seed: {cfg.get('seed')}",
        f"  source alphabets: {summary['n_source_alphabets']}",
        f"  close pairs: {cfg.get('n_close_pairs')}",
        f"  distant pairs: {cfg.get('n_distant_pairs')}",
        f"  negative pairs: {cfg.get('n_negative_pairs')}",
        "",
        "Distance metric:",
        f"  {DISTANCE_METRIC_LABEL}",
        "",
        f"Model kind: {summary['model_kind']}",
        f"Use teacher: {summary['use_teacher']}",
        f"Checkpoint: {summary['checkpoint_path']}",
        f"Device: {summary['device']}",
        "",
        f"Number of synthetic scripts: {summary['n_scripts']}",
        f"Number of synthetic labelled pairs: {summary['n_pairs']}",
        f"Number of ground-truth ancestor clusters: {summary['n_clusters']}",
        "",
        "Best clustering method:",
        f"  method: {best['method']}",
        f"  family: {best['family']}",
        f"  ARI: {best['ari']:.6f}",
        f"  NMI: {best['nmi']:.6f}",
        f"  silhouette: {best['silhouette']:.6f}",
        f"  cophenetic correlation: {best['cophenetic_corr']:.6f}",
        f"  pair Spearman rho: {best['pair_spearman_rho']:.6f}",
        f"  pair mean AUC: {best['pair_mean_auc']:.6f}",
        "",
        "Ancestor clusters:",
        ", ".join(summary["ancestors"]),
        "",
        "Full ranking:",
        comparison.to_string(index=False),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--model-kind",
        type=str,
        default=DEFAULT_MODEL_KIND,
        choices=["student_teacher", "teacher_only"],
    )

    parser.add_argument("--use-teacher", action="store_true", default=DEFAULT_USE_TEACHER)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)

    parser.add_argument(
        "--distance-matrix",
        type=Path,
        default=None,
        help="Reuse a precomputed synthetic_script_distance_matrix.npy "
             "(rows ordered by sorted script id) instead of embedding the scripts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_glyph_dictionary(args.dataset_path)
    pairs = dataset["pairs"]
    synthetic_scripts = dataset["synthetic_scripts"]
    dataset_config = dataset.get("config", {})

    script_ids = sorted(synthetic_scripts)
    ancestors_of = [synthetic_scripts[sid]["ancestor_alphabet"] for sid in script_ids]
    ancestors = sorted(set(ancestors_of))
    ancestor_index = {name: i for i, name in enumerate(ancestors)}
    y_true = np.array([ancestor_index[a] for a in ancestors_of])
    n_clusters = len(ancestors)

    print(f"[INFO] scripts={len(script_ids)}, pairs={len(pairs)}, ancestors={n_clusters}")

    model, ckpt_path = None, None
    if args.distance_matrix is None:
        model, ckpt_path = load_metric_model(args.model_kind)
        script_embeddings = compute_all_script_embeddings(
            synthetic_scripts=synthetic_scripts,
            model=model,
            model_kind=args.model_kind,
            use_teacher=args.use_teacher,
            batch_size=args.batch_size,
            image_size=args.image_size,
        )
        D = build_script_distance_matrix(script_ids, script_embeddings)
    else:
        D = np.load(args.distance_matrix).astype(np.float32)
        if D.shape != (len(script_ids), len(script_ids)):
            raise ValueError(f"Distance matrix shape {D.shape} does not match {len(script_ids)} scripts.")

    position = {sid: i for i, sid in enumerate(script_ids)}
    pair_left = np.array([position[p["left_id"]] for p in pairs])
    pair_right = np.array([position[p["right_id"]] for p in pairs])
    pair_labels = np.array([float(p["label"]) for p in pairs])

    hier_rows, linkages = evaluate_hierarchical(
        D, y_true, n_clusters, pair_left, pair_right, pair_labels
    )
    flat_rows = evaluate_flat(D, y_true, n_clusters)

    columns = [
        "method", "family", "n_clusters", "ari", "nmi", "silhouette",
        "linkage", "cophenetic_corr", "pair_spearman_rho", "pair_spearman_p",
        "pair_mean_auc", "pair_auc_close_vs_distant", "pair_auc_close_vs_unrelated",
        "pair_auc_distant_vs_unrelated", "n_pairs_used",
    ]

    def ranked(rows: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows).reindex(columns=columns)
        return df.sort_values(by=["ari", "nmi"], ascending=False).reset_index(drop=True)

    comparison = ranked(hier_rows + flat_rows)
    comparison.to_csv(args.output_dir / "clustering_method_comparison.csv", index=False)
    ranked(hier_rows).to_csv(args.output_dir / "hierarchical_linkage_comparison.csv", index=False)
    ranked(flat_rows).to_csv(args.output_dir / "flat_clustering_comparison.csv", index=False)

    np.save(args.output_dir / "synthetic_script_distance_matrix.npy", D)
    with open(args.output_dir / "synthetic_script_ids.json", "w", encoding="utf-8") as f:
        json.dump(script_ids, f, indent=2)
    pd.DataFrame(
        {"script_id": script_ids, "ancestor_alphabet": ancestors_of, "ancestor_label": y_true}
    ).to_csv(args.output_dir / "synthetic_script_ancestors.csv", index=False)

    for method, Z in linkages.items():
        np.save(args.output_dir / f"linkage_{method}.npy", Z)
        plot_dendrogram(
            Z,
            args.output_dir / f"dendrogram_{method}.png",
            title=f"Synthetic clustering validation: {method} linkage",
        )

    for score in RANKING_SCORES:
        plot_ranking(comparison, score, args.output_dir / f"ranking_{score}.png")

    summary = {
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "dataset_config": dataset_config,
        "n_source_alphabets": n_clusters,
        "model_kind": args.model_kind,
        "use_teacher": bool(args.use_teacher),
        "checkpoint_path": str(ckpt_path) if ckpt_path else f"(distance matrix: {args.distance_matrix})",
        "device": str(DEVICE),
        "n_scripts": len(script_ids),
        "n_pairs": len(pairs),
        "n_clusters": n_clusters,
        "ancestors": ancestors,
    }
    write_text_summary(args.output_dir / "clustering_validation_summary.txt", summary, comparison)

    print("\n=== CLUSTERING METHOD COMPARISON ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
