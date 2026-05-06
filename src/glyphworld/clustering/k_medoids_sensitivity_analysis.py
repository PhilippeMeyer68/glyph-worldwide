# -*- coding: utf-8 -*-
"""
Sensitivity analysis for large-script k-medoids representative glyphs.

Spyder-friendly version.
No predefined number of clusters.
Only evaluates stability at the script-distance level.

The script:
1. loads the full glyph embedding dictionary,
2. builds reduced dictionaries using k-medoids for scripts ending with " Full",
3. recomputes all pairwise script distances,
4. compares every run to a reference run,
5. saves summary tables, distance matrices, and diagnostic plots.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr

from glyphworld.core.paths import ALPHABETS_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary, save_glyph_dictionary
from glyphworld.distances.distance_functions import similarity_between_languages


# =============================================================================
# Low-level distances and k-medoids
# =============================================================================

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = l2_normalize_rows(A)
    B = l2_normalize_rows(B)
    return (1.0 - A @ B.T).astype(np.float32, copy=False)


def assign_labels(Z: np.ndarray, medoids: np.ndarray) -> tuple[np.ndarray, float]:
    D = cosine_distance_matrix(Z[medoids], Z)
    labels = np.argmin(D, axis=0)
    cost = D[labels, np.arange(Z.shape[0])].sum()
    return labels, float(cost)


def update_medoids(
    Z: np.ndarray,
    labels: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = Z.shape[0]
    new_medoids = np.empty(k, dtype=int)

    for c in range(k):
        members = np.where(labels == c)[0]

        if len(members) == 0:
            new_medoids[c] = int(rng.integers(n))
            continue

        Zm = Z[members]
        D = cosine_distance_matrix(Zm, Zm)
        costs = D.sum(axis=1)
        new_medoids[c] = int(members[np.argmin(costs)])

    return new_medoids


def kmedoids_cosine(
    Z: np.ndarray,
    k: int,
    seed: int = 0,
    max_iter: int = 20,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]

    if n == 0:
        raise ValueError("Empty embedding matrix.")

    if k > n:
        k = n

    rng = np.random.default_rng(seed)
    medoids = rng.choice(n, size=k, replace=False)
    prev_cost = None

    for it in range(max_iter):
        labels, cost = assign_labels(Z, medoids)
        new_medoids = update_medoids(Z, labels, k, rng)

        # Ensure unique medoids.
        unique_medoids = []
        used = set()

        for m in new_medoids:
            m = int(m)
            if m not in used:
                unique_medoids.append(m)
                used.add(m)

        while len(unique_medoids) < k:
            candidate = int(rng.integers(n))
            if candidate not in used:
                unique_medoids.append(candidate)
                used.add(candidate)

        new_medoids = np.asarray(unique_medoids, dtype=int)

        if verbose:
            print(f"      iter={it:02d} cost={cost:.6f}")

        if np.array_equal(new_medoids, medoids):
            return medoids, labels, cost

        if prev_cost is not None and abs(prev_cost - cost) < 1e-12:
            return new_medoids, labels, cost

        medoids = new_medoids
        prev_cost = cost

    labels, cost = assign_labels(Z, medoids)
    return medoids, labels, cost


# =============================================================================
# Script-level distance computation
# =============================================================================

def distance_dict_from_dense_cosine(
    Z1: np.ndarray,
    Z2: np.ndarray,
) -> dict[tuple[int, int], float]:
    D = cosine_distance_matrix(Z1, Z2)

    out = {}
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            out[(i, j)] = float(D[i, j])

    return out


def script_distance(Z1: np.ndarray, Z2: np.ndarray) -> float:
    dist_dict = distance_dict_from_dense_cosine(Z1, Z2)
    return float(similarity_between_languages(dist_dict))


def embeddings_to_matrix(embeddings_for_script: dict) -> np.ndarray:
    return np.stack(
        list(embeddings_for_script.values()),
        axis=0,
    ).astype(np.float32, copy=False)


def build_reduced_embeddings(
    embeddings_dic: dict,
    k: int,
    seed: int,
    max_iter: int,
    verbose: bool = False,
) -> tuple[dict, pd.DataFrame]:
    """
    Keep normal scripts unchanged.
    Replace each 'X Full' script by a reduced 'X' script using k-medoids.
    """
    out = dict(embeddings_dic)
    rows = []

    full_names = sorted([name for name in embeddings_dic if name.endswith(" Full")])

    for name in full_names:
        base_name = name[:-5]

        glyph_embs = embeddings_dic[name]
        keys = list(glyph_embs.keys())

        Z = np.stack(
            [glyph_embs[g] for g in keys],
            axis=0,
        ).astype(np.float32, copy=False)

        effective_k = min(k, len(keys))

        if verbose:
            print(
                f"    {name} -> {base_name}: "
                f"n={len(keys)}, k={effective_k}, seed={seed}"
            )

        medoids, labels, cost = kmedoids_cosine(
            Z,
            k=effective_k,
            seed=seed,
            max_iter=max_iter,
            verbose=False,
        )

        medoid_keys = [keys[i] for i in medoids]
        out[base_name] = {g: glyph_embs[g] for g in medoid_keys}

        rows.append({
            "source_script": name,
            "reduced_script": base_name,
            "n_full_glyphs": len(keys),
            "k_requested": k,
            "k_effective": effective_k,
            "seed": seed,
            "kmedoids_cost": cost,
            "medoid_keys_json": json.dumps([str(x) for x in medoid_keys]),
        })

    return out, pd.DataFrame(rows)


def compute_all_script_distances(
    embeddings_dic: dict,
    exclude_full: bool = True,
) -> tuple[list[str], dict[tuple[int, int], float]]:
    alphabets = sorted([
        a for a in embeddings_dic
        if not (exclude_full and a.endswith(" Full"))
    ])

    matrices = {
        a: embeddings_to_matrix(embeddings_dic[a])
        for a in alphabets
    }

    distances = {}

    for i in range(len(alphabets) - 1):
        a1 = alphabets[i]
        Z1 = matrices[a1]

        print(f"      pair block {i + 1}/{len(alphabets) - 1}: {a1} {Z1.shape}")

        for j in range(i + 1, len(alphabets)):
            a2 = alphabets[j]
            distances[(i, j)] = script_distance(Z1, matrices[a2])

    return alphabets, distances


def distances_to_vector(
    distances: dict[tuple[int, int], float],
    n: int,
) -> np.ndarray:
    return np.asarray(
        [distances[(i, j)] for i in range(n - 1) for j in range(i + 1, n)],
        dtype=np.float64,
    )


def save_distance_outputs(
    out_dir: Path,
    run_id: str,
    alphabets: list[str],
    distances: dict[tuple[int, int], float],
) -> np.ndarray:
    n = len(alphabets)
    v = distances_to_vector(distances, n)

    rows = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            rows.append({
                "script_i": alphabets[i],
                "script_j": alphabets[j],
                "distance": distances[(i, j)],
            })

    pd.DataFrame(rows).to_csv(
        out_dir / "distance_vectors" / f"{run_id}.csv",
        index=False,
    )

    save_glyph_dictionary(
        {
            "alphabets": alphabets,
            "alphabet_distances": distances,
        },
        out_dir / "distance_pickles" / f"{run_id}.pkl",
    )

    return v


# =============================================================================
# Stability metrics
# =============================================================================

@dataclass
class RunResult:
    run_id: str
    k: int
    seed: int
    n_scripts: int
    n_pairs: int
    pearson_vs_reference: float
    spearman_vs_reference: float
    mae_vs_reference: float
    rmse_vs_reference: float
    max_abs_diff_vs_reference: float


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float:
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(fn(x, y).statistic)


def compare_to_reference(
    run_id: str,
    k: int,
    seed: int,
    v: np.ndarray,
    ref_v: np.ndarray,
    n_scripts: int,
) -> RunResult:
    diff = v - ref_v

    return RunResult(
        run_id=run_id,
        k=k,
        seed=seed,
        n_scripts=n_scripts,
        n_pairs=len(v),
        pearson_vs_reference=safe_corr(pearsonr, v, ref_v),
        spearman_vs_reference=safe_corr(spearmanr, v, ref_v),
        mae_vs_reference=float(np.mean(np.abs(diff))),
        rmse_vs_reference=float(np.sqrt(np.mean(diff ** 2))),
        max_abs_diff_vs_reference=float(np.max(np.abs(diff))),
    )


def pairwise_run_comparisons(vectors: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    run_ids = sorted(vectors)

    for idx, r1 in enumerate(run_ids[:-1]):
        for r2 in run_ids[idx + 1:]:
            x = vectors[r1]
            y = vectors[r2]
            diff = x - y

            rows.append({
                "run_1": r1,
                "run_2": r2,
                "pearson": safe_corr(pearsonr, x, y),
                "spearman": safe_corr(spearmanr, x, y),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "max_abs_diff": float(np.max(np.abs(diff))),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    plots_dir = out_dir / "plots"

    grouped = summary.groupby("k", as_index=False).agg(
        pearson_mean=("pearson_vs_reference", "mean"),
        pearson_std=("pearson_vs_reference", "std"),
        spearman_mean=("spearman_vs_reference", "mean"),
        spearman_std=("spearman_vs_reference", "std"),
        mae_mean=("mae_vs_reference", "mean"),
        mae_std=("mae_vs_reference", "std"),
        rmse_mean=("rmse_vs_reference", "mean"),
        rmse_std=("rmse_vs_reference", "std"),
        max_abs_mean=("max_abs_diff_vs_reference", "mean"),
        max_abs_std=("max_abs_diff_vs_reference", "std"),
    )

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grouped["k"],
        grouped["spearman_mean"],
        yerr=grouped["spearman_std"],
        marker="o",
    )
    plt.xlabel("Number of medoids per large script (k)")
    plt.ylabel("Spearman correlation vs reference")
    plt.title("Script-distance rank stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "spearman_vs_reference_by_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grouped["k"],
        grouped["pearson_mean"],
        yerr=grouped["pearson_std"],
        marker="o",
    )
    plt.xlabel("Number of medoids per large script (k)")
    plt.ylabel("Pearson correlation vs reference")
    plt.title("Script-distance linear stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "pearson_vs_reference_by_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grouped["k"],
        grouped["mae_mean"],
        yerr=grouped["mae_std"],
        marker="o",
    )
    plt.xlabel("Number of medoids per large script (k)")
    plt.ylabel("Mean absolute distance change")
    plt.title("Absolute script-distance perturbation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "mae_vs_reference_by_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grouped["k"],
        grouped["rmse_mean"],
        yerr=grouped["rmse_std"],
        marker="o",
    )
    plt.xlabel("Number of medoids per large script (k)")
    plt.ylabel("RMSE vs reference")
    plt.title("Script-distance RMSE perturbation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "rmse_vs_reference_by_k.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.errorbar(
        grouped["k"],
        grouped["max_abs_mean"],
        yerr=grouped["max_abs_std"],
        marker="o",
    )
    plt.xlabel("Number of medoids per large script (k)")
    plt.ylabel("Maximum absolute distance change")
    plt.title("Worst-case script-distance perturbation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / "max_abs_diff_vs_reference_by_k.png", dpi=200)
    plt.close()


def plot_distance_scatter(
    ref_v: np.ndarray,
    vectors: dict[str, np.ndarray],
    out_dir: Path,
    max_plots: int = 20,
) -> None:
    plots_dir = out_dir / "plots"

    for idx, (run_id, v) in enumerate(sorted(vectors.items())):
        if idx >= max_plots:
            break

        plt.figure(figsize=(5, 5))
        plt.scatter(ref_v, v, s=8, alpha=0.5)

        lo = float(min(ref_v.min(), v.min()))
        hi = float(max(ref_v.max(), v.max()))

        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlabel("Reference script distances")
        plt.ylabel(f"{run_id} script distances")
        plt.title(f"Script-distance stability: {run_id}")
        plt.tight_layout()
        plt.savefig(plots_dir / f"scatter_reference_vs_{run_id}.png", dpi=200)
        plt.close()


# =============================================================================
# CONFIG — edit here in Spyder
# =============================================================================

INPUT_EMBEDDINGS = ALPHABETS_DIR / "alphabets_unicode_embeddings_css.pkl"

K_VALUES = [50, 100, 200]
SEEDS = [0, 1, 2, 3, 4]

REFERENCE_K = 100
REFERENCE_SEED = 0

MAX_ITER = 20

OUT_DIR = RESULTS_DIR / "kmedoids_sensitivity_no_clusters"

VERBOSE = False


# =============================================================================
# Main
# =============================================================================

def ensure_dirs(out_dir: Path) -> None:
    for sub in ["distance_vectors", "distance_pickles", "medoid_logs", "plots"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs(OUT_DIR)

    print(f"[LOAD] {INPUT_EMBEDDINGS}")
    embeddings_dic = load_glyph_dictionary(INPUT_EMBEDDINGS)

    configs = [(k, seed) for k in K_VALUES for seed in SEEDS]
    ref_config = (REFERENCE_K, REFERENCE_SEED)

    if ref_config not in configs:
        configs.insert(0, ref_config)

    vectors = {}
    alphabets_by_run = {}
    medoid_logs = []

    for k, seed in configs:
        run_id = f"k{k}_seed{seed}"
        print(f"\n[RUN] {run_id}")

        reduced_dic, medoid_log = build_reduced_embeddings(
            embeddings_dic,
            k=k,
            seed=seed,
            max_iter=MAX_ITER,
            verbose=VERBOSE,
        )

        medoid_log.insert(0, "run_id", run_id)
        medoid_logs.append(medoid_log)

        medoid_log.to_csv(
            OUT_DIR / "medoid_logs" / f"{run_id}_medoids.csv",
            index=False,
        )

        alphabets, distances = compute_all_script_distances(
            reduced_dic,
            exclude_full=True,
        )

        v = save_distance_outputs(
            OUT_DIR,
            run_id,
            alphabets,
            distances,
        )

        vectors[run_id] = v
        alphabets_by_run[run_id] = alphabets

    reference_id = f"k{REFERENCE_K}_seed{REFERENCE_SEED}"

    if reference_id not in vectors:
        raise RuntimeError(f"Reference run {reference_id} was not computed.")

    ref_alphabets = alphabets_by_run[reference_id]
    ref_v = vectors[reference_id]

    # Sanity check: same script order in all runs.
    for run_id, alphabets in alphabets_by_run.items():
        if alphabets != ref_alphabets:
            raise RuntimeError(
                f"Alphabet order mismatch for {run_id}. "
                "This would invalidate vector comparisons."
            )

    summary_rows = []

    for k, seed in configs:
        run_id = f"k{k}_seed{seed}"

        result = compare_to_reference(
            run_id=run_id,
            k=k,
            seed=seed,
            v=vectors[run_id],
            ref_v=ref_v,
            n_scripts=len(ref_alphabets),
        )

        summary_rows.append(asdict(result))

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["k", "seed"])
        .reset_index(drop=True)
    )

    pairwise = pairwise_run_comparisons(vectors)

    if medoid_logs:
        all_medoid_logs = pd.concat(medoid_logs, ignore_index=True)
    else:
        all_medoid_logs = pd.DataFrame()

    summary.to_csv(OUT_DIR / "sensitivity_summary.csv", index=False)
    pairwise.to_csv(OUT_DIR / "sensitivity_pairwise_runs.csv", index=False)
    all_medoid_logs.to_csv(OUT_DIR / "all_medoid_logs.csv", index=False)

    plot_summary(summary, OUT_DIR)
    plot_distance_scatter(ref_v, vectors, OUT_DIR)

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "input_embeddings": str(INPUT_EMBEDDINGS),
            "k_values": K_VALUES,
            "seeds": SEEDS,
            "reference_k": REFERENCE_K,
            "reference_seed": REFERENCE_SEED,
            "max_iter": MAX_ITER,
            "n_scripts": len(ref_alphabets),
            "n_pairs": len(ref_v),
            "analysis_level": "script_distance_matrix_only",
            "clustering_used": False,
        }, f, indent=2)

    print("\n[DONE]")
    print(f"Saved outputs to: {OUT_DIR}")
    print("\nSummary:")
    print(summary.to_string(index=False))


# In Spyder: edit CONFIG, then press Run.
main()