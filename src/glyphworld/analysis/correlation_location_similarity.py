# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Correlate geographic / temporal distances with script visual distance.

This script evaluates several geographic and temporal distance variants as
predictors of learned visual script distance/similarity, using in-sample and
K-fold cross-validated correlation.

It exports results in the same style as metric_validation.py and
clustering_validation.py:

Outputs
-------
RESULTS_DIR / "validation" / "location_similarity_correlation"
    - pairwise_location_similarity_data.csv
    - baseline_single_predictor_results.csv
    - mixed_geo_time_results.csv
    - all_location_similarity_results.csv
    - heatmap_mixed_cv_<metric>.csv
    - heatmap_mixed_cv_<metric>.png
    - baseline_barplot_<metric>.png
    - ranking_cv_mean.png
    - location_similarity_correlation_summary.txt
    - location_similarity_correlation_summary.json

Use
---
    python correlation_location_similarity.py

Optional CLI:
    python correlation_location_similarity.py --metric spearman
    python correlation_location_similarity.py --metric pearson --k-folds 5
"""

import argparse
import json
import math
import pickle
import warnings
from itertools import product as iterproduct
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.stats import pearsonr, spearmanr

from glyphworld.core.paths import DATA_DIR, PROCESSED_DIR, RESULTS_DIR
from glyphworld.core.utils import short_name_from_filename

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "location_similarity_correlation"

DEFAULT_SCRIPTS_LOCATION = (
    DATA_DIR / "info" / "scripts_location_with_latlon_and_timerange.xlsx"
)

DEFAULT_SCRIPTS_SIMILARITIES = (
    PROCESSED_DIR / "global distances" / "alphabet_similarity_matrix_cosine.pkl"
)

DEFAULT_K_FOLDS = 5
DEFAULT_SEED = 42
DEFAULT_METRIC = "spearman"


# =====================================================================
# IO helpers
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
# Core helpers
# =====================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * np.arcsin(np.sqrt(a))


def _parse_float_list_cell(x):
    if pd.isna(x):
        return np.array([], dtype=float)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([float(x)], dtype=float)

    s = str(x).strip()
    if not s:
        return np.array([], dtype=float)

    for sep in [";", "|", "/", ","]:
        s = s.replace(sep, " ")

    out = []
    for token in s.split():
        try:
            out.append(float(token))
        except ValueError:
            pass

    return np.asarray(out, dtype=float)


def normalize_0_1(v):
    v = np.asarray(v, dtype=np.float64)
    m = np.nanmax(v)
    if not np.isfinite(m) or m == 0:
        return v
    return v / m


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


# =====================================================================
# Geographic distance variants
# =====================================================================

def _precompute_geo_data(df):
    scripts = df["Script"].astype(str).to_list()
    lats = df["Latitudes"].apply(_parse_float_list_cell).to_list()
    lons = df["Longitudes"].apply(_parse_float_list_cell).to_list()
    idx_of = {s: i for i, s in enumerate(scripts)}
    return scripts, lats, lons, idx_of


def _pairwise_dists(lat_i, lon_i, lat_j, lon_j):
    """Full pairwise Haversine matrix between two sets of locations."""
    return haversine_km(
        lat_i[:, None],
        lon_i[:, None],
        lat_j[None, :],
        lon_j[None, :],
    )


def compute_geo_min(df, pairs_scripts):
    """Minimum pairwise Haversine distance."""
    _, lats, lons, idx_of = _precompute_geo_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        if lats[i].size == 0 or lats[j].size == 0:
            out.append(np.nan)
            continue
        out.append(float(np.min(_pairwise_dists(lats[i], lons[i], lats[j], lons[j]))))
    return np.asarray(out, dtype=np.float64)


def compute_geo_mean(df, pairs_scripts):
    """Mean of all pairwise Haversine distances."""
    _, lats, lons, idx_of = _precompute_geo_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        if lats[i].size == 0 or lats[j].size == 0:
            out.append(np.nan)
            continue
        out.append(float(np.mean(_pairwise_dists(lats[i], lons[i], lats[j], lons[j]))))
    return np.asarray(out, dtype=np.float64)


def compute_geo_median(df, pairs_scripts):
    """Median of all pairwise Haversine distances."""
    _, lats, lons, idx_of = _precompute_geo_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        if lats[i].size == 0 or lats[j].size == 0:
            out.append(np.nan)
            continue
        out.append(float(np.median(_pairwise_dists(lats[i], lons[i], lats[j], lons[j]))))
    return np.asarray(out, dtype=np.float64)


def compute_geo_hausdorff(df, pairs_scripts):
    """Symmetric Hausdorff geographic distance."""
    _, lats, lons, idx_of = _precompute_geo_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        if lats[i].size == 0 or lats[j].size == 0:
            out.append(np.nan)
            continue
        D = _pairwise_dists(lats[i], lons[i], lats[j], lons[j])
        h = max(float(np.max(np.min(D, axis=1))), float(np.max(np.min(D, axis=0))))
        out.append(h)
    return np.asarray(out, dtype=np.float64)


def compute_geo_centroid(df, pairs_scripts):
    """Haversine distance between geographic centroids."""
    _, lats, lons, idx_of = _precompute_geo_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        if lats[i].size == 0 or lats[j].size == 0:
            out.append(np.nan)
            continue
        d = haversine_km(
            np.mean(lats[i]),
            np.mean(lons[i]),
            np.mean(lats[j]),
            np.mean(lons[j]),
        )
        out.append(float(d))
    return np.asarray(out, dtype=np.float64)


# =====================================================================
# Temporal distance variants
# =====================================================================

def _precompute_time_data(df):
    scripts = df["Script"].astype(str).to_list()
    begins = df["begin"].astype(float).to_list()
    ends = df["end"].astype(float).to_list()
    idx_of = {s: i for i, s in enumerate(scripts)}
    return scripts, begins, ends, idx_of


def compute_time_L1(df, pairs_scripts):
    """|b1-b2| + |e1-e2|."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        out.append(abs(begins[i] - begins[j]) + abs(ends[i] - ends[j]))
    return np.asarray(out, dtype=np.float64)


def compute_time_L2(df, pairs_scripts):
    """sqrt((b1-b2)^2 + (e1-e2)^2)."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        out.append(math.sqrt((begins[i] - begins[j]) ** 2 + (ends[i] - ends[j]) ** 2))
    return np.asarray(out, dtype=np.float64)


def compute_time_Linf(df, pairs_scripts):
    """max(|b1-b2|, |e1-e2|)."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        out.append(max(abs(begins[i] - begins[j]), abs(ends[i] - ends[j])))
    return np.asarray(out, dtype=np.float64)


def compute_time_midpoint(df, pairs_scripts):
    """Absolute midpoint distance."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        m1 = (begins[i] + ends[i]) / 2.0
        m2 = (begins[j] + ends[j]) / 2.0
        out.append(abs(m1 - m2))
    return np.asarray(out, dtype=np.float64)


def compute_time_gap(df, pairs_scripts):
    """Gap between intervals; 0 if overlapping."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        a1, b1 = min(begins[i], ends[i]), max(begins[i], ends[i])
        a2, b2 = min(begins[j], ends[j]), max(begins[j], ends[j])
        out.append(max(0, max(a1, a2) - min(b1, b2)))
    return np.asarray(out, dtype=np.float64)


def compute_time_jaccard(df, pairs_scripts):
    """1 - temporal interval intersection/union."""
    _, begins, ends, idx_of = _precompute_time_data(df)
    out = []
    for s1, s2 in pairs_scripts:
        i, j = idx_of[s1], idx_of[s2]
        a1, b1 = min(begins[i], ends[i]), max(begins[i], ends[i])
        a2, b2 = min(begins[j], ends[j]), max(begins[j], ends[j])
        inter = max(0, min(b1, b2) - max(a1, a2))
        union = max(b1, b2) - min(a1, a2)
        out.append(0.0 if union == 0 else 1.0 - inter / union)
    return np.asarray(out, dtype=np.float64)


# =====================================================================
# Model functions
# =====================================================================

def compute_v_single(v, cs):
    cs = np.asarray(cs, dtype=np.float64)
    return cs[0] * np.power(v, cs[1]) + cs[2]


def compute_v_mix(v_geo, v_time, cs):
    cs = np.asarray(cs, dtype=np.float64)
    out = cs[0] * np.power(v_geo, cs[1]) + cs[2] * np.power(v_time, cs[3])
    out = np.power(out, cs[4])
    return out + cs[5]


def _corr(x, y, method="pearson"):
    if len(x) < 3:
        return np.nan
    try:
        if method == "pearson":
            return pearsonr(x, y)[0]
        return spearmanr(x, y)[0]
    except Exception:
        return np.nan


def optimise_single(v_train, v_sim_train, metric="pearson", seed=0):
    bounds = [(0.0, 10.0), (0.01, 10.0), (0.0, 10.0)]

    def obj(x):
        return -_corr(compute_v_single(v_train, x), v_sim_train, method=metric)

    res = differential_evolution(
        obj,
        bounds=bounds,
        seed=seed,
        polish=True,
        maxiter=300,
        tol=1e-6,
    )
    return res.x, -res.fun


def optimise_mix(v_geo_train, v_time_train, v_sim_train, metric="pearson", seed=0):
    bounds = [(0.0, 10.0)] * 6

    def obj(x):
        return -_corr(compute_v_mix(v_geo_train, v_time_train, x), v_sim_train, method=metric)

    res = differential_evolution(
        obj,
        bounds=bounds,
        seed=seed,
        polish=True,
        maxiter=300,
        tol=1e-6,
    )
    return res.x, -res.fun


def kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    current = 0
    folds = []
    for fs in fold_sizes:
        folds.append(perm[current : current + fs])
        current += fs

    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def cv_single(v_all, v_sim_all, n_pairs, k=5, seed=42, metric="pearson"):
    """Return mean CV correlation for a single-predictor model."""
    rs = []
    params = []
    train_scores = []

    for fold_i, (tr, te) in enumerate(kfold_indices(n_pairs, k=k, seed=seed)):
        c, train_corr = optimise_single(
            v_all[tr],
            v_sim_all[tr],
            metric=metric,
            seed=seed + fold_i,
        )
        test_corr = _corr(compute_v_single(v_all[te], c), v_sim_all[te], metric)
        rs.append(test_corr)
        params.append(c)
        train_scores.append(train_corr)

    return np.nanmean(rs), np.nanstd(rs), rs, train_scores, params


def cv_mix(v_geo, v_time, v_sim_all, n_pairs, k=5, seed=42, metric="pearson"):
    """Return mean CV correlation for the mixed model."""
    rs = []
    params = []
    train_scores = []

    for fold_i, (tr, te) in enumerate(kfold_indices(n_pairs, k=k, seed=seed)):
        c, train_corr = optimise_mix(
            v_geo[tr],
            v_time[tr],
            v_sim_all[tr],
            metric=metric,
            seed=seed + fold_i,
        )
        test_corr = _corr(compute_v_mix(v_geo[te], v_time[te], c), v_sim_all[te], metric)
        rs.append(test_corr)
        params.append(c)
        train_scores.append(train_corr)

    return np.nanmean(rs), np.nanstd(rs), rs, train_scores, params


# =====================================================================
# Result builders
# =====================================================================

def make_pairwise_dataframe(
    pairs_scripts: list[tuple[str, str]],
    v_sim: np.ndarray,
    valid: np.ndarray,
    geo_raw: dict[str, np.ndarray],
    time_raw: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for idx, ((s1, s2), is_valid) in enumerate(zip(pairs_scripts, valid)):
        row = {
            "pair_index": idx,
            "script_1": s1,
            "script_2": s2,
            "visual_distance": float(v_sim[idx]),
            "valid": bool(is_valid),
        }
        for name, values in geo_raw.items():
            row[name] = float(values[idx]) if np.isfinite(values[idx]) else np.nan
        for name, values in time_raw.items():
            row[name] = float(values[idx]) if np.isfinite(values[idx]) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_baseline_results(
    geo_norm: dict[str, np.ndarray],
    time_norm: dict[str, np.ndarray],
    v_sim_all: np.ndarray,
    n_pairs: int,
    k_folds: int,
    seed: int,
    metric: str,
) -> pd.DataFrame:
    rows = []

    for family, vdict in [("geographic", geo_norm), ("temporal", time_norm)]:
        for name, v in vdict.items():
            c_is, r_is = optimise_single(v, v_sim_all, metric=metric, seed=seed)
            cv_mean, cv_std, cv_folds, cv_train_scores, cv_params = cv_single(
                v,
                v_sim_all,
                n_pairs,
                k=k_folds,
                seed=seed,
                metric=metric,
            )

            rows.append(
                {
                    "model": "single_predictor",
                    "family": family,
                    "geo_metric": name if family == "geographic" else "",
                    "time_metric": name if family == "temporal" else "",
                    "predictor": name,
                    "n_parameters": 3,
                    "in_sample_corr": float(r_is),
                    "cv_mean": float(cv_mean),
                    "cv_std": float(cv_std),
                    "cv_folds": json.dumps([float(x) for x in cv_folds]),
                    "in_sample_params": json.dumps([float(x) for x in c_is]),
                    "cv_train_scores": json.dumps([float(x) for x in cv_train_scores]),
                    "cv_params": json.dumps([[float(y) for y in x] for x in cv_params]),
                    "n_pairs": int(n_pairs),
                }
            )

    out = pd.DataFrame(rows)
    out = out.sort_values("cv_mean", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_mixed_results(
    geo_norm: dict[str, np.ndarray],
    time_norm: dict[str, np.ndarray],
    v_sim_all: np.ndarray,
    n_pairs: int,
    k_folds: int,
    seed: int,
    metric: str,
) -> pd.DataFrame:
    rows = []
    geo_keys = sorted(geo_norm.keys())
    time_keys = sorted(time_norm.keys())

    for gk, tk in iterproduct(geo_keys, time_keys):
        vg = geo_norm[gk]
        vt = time_norm[tk]

        c_is, _ = optimise_mix(vg, vt, v_sim_all, metric=metric, seed=seed)
        r_is = _corr(compute_v_mix(vg, vt, c_is), v_sim_all, metric)

        cv_mean, cv_std, cv_folds, cv_train_scores, cv_params = cv_mix(
            vg,
            vt,
            v_sim_all,
            n_pairs,
            k=k_folds,
            seed=seed,
            metric=metric,
        )

        rows.append(
            {
                "model": "mixed_geo_time",
                "family": "mixed",
                "geo_metric": gk,
                "time_metric": tk,
                "predictor": f"{gk}+{tk}",
                "n_parameters": 6,
                "in_sample_corr": float(r_is),
                "cv_mean": float(cv_mean),
                "cv_std": float(cv_std),
                "cv_folds": json.dumps([float(x) for x in cv_folds]),
                "in_sample_params": json.dumps([float(x) for x in c_is]),
                "cv_train_scores": json.dumps([float(x) for x in cv_train_scores]),
                "cv_params": json.dumps([[float(y) for y in x] for x in cv_params]),
                "n_pairs": int(n_pairs),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("cv_mean", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def build_all_results(baseline_df: pd.DataFrame, mixed_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([baseline_df, mixed_df], ignore_index=True)
    out = out.sort_values("cv_mean", ascending=False, na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def build_heatmap_table(mixed_df: pd.DataFrame) -> pd.DataFrame:
    return mixed_df.pivot(index="geo_metric", columns="time_metric", values="cv_mean")


# =====================================================================
# Plot helpers
# =====================================================================

def plot_heatmap(heatmap_df: pd.DataFrame, corr_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    mat = heatmap_df.to_numpy(dtype=float)

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(
        [k.replace("time_", "") for k in heatmap_df.columns],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels([k.replace("geo_", "") for k in heatmap_df.index])
    ax.set_xlabel("Temporal distance metric")
    ax.set_ylabel("Geographic distance metric")
    ax.set_title(f"5-fold CV {corr_label}: mixed geo × time model")

    for gi in range(mat.shape[0]):
        for ti in range(mat.shape[1]):
            value = mat[gi, ti]
            if np.isfinite(value):
                ax.text(
                    ti,
                    gi,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black" if value < 0.3 else "white",
                )

    fig.colorbar(im, ax=ax, label=f"CV {corr_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_barplot(baseline_df: pd.DataFrame, corr_label: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    geo = baseline_df[baseline_df["family"] == "geographic"].copy()
    geo = geo.sort_values("cv_mean", ascending=True)
    ax1.barh(
        geo["predictor"].str.replace("geo_", "", regex=False),
        geo["cv_mean"],
        xerr=geo["cv_std"],
        capsize=3,
    )
    ax1.set_xlabel(f"CV {corr_label}")
    ax1.set_title("Geographic distance baselines")

    time = baseline_df[baseline_df["family"] == "temporal"].copy()
    time = time.sort_values("cv_mean", ascending=True)
    ax2.barh(
        time["predictor"].str.replace("time_", "", regex=False),
        time["cv_mean"],
        xerr=time["cv_std"],
        capsize=3,
    )
    ax2.set_xlabel(f"CV {corr_label}")
    ax2.set_title("Temporal distance baselines")

    fig.suptitle(f"Single-predictor baselines: {corr_label}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_ranking(results: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    sub = results[np.isfinite(results["cv_mean"])].head(top_n).copy()
    if len(sub) == 0:
        return

    sub = sub.sort_values("cv_mean", ascending=True)
    labels = []
    for _, row in sub.iterrows():
        if row["model"] == "mixed_geo_time":
            labels.append(
                f"{row['geo_metric'].replace('geo_', '')} + "
                f"{row['time_metric'].replace('time_', '')}"
            )
        else:
            labels.append(str(row["predictor"]).replace("geo_", "").replace("time_", ""))

    plt.figure(figsize=(10, max(5, 0.35 * len(sub))))
    plt.barh(labels, sub["cv_mean"])
    plt.xlabel("CV mean correlation")
    plt.ylabel("Model")
    plt.title(f"Top {len(sub)} location/time predictors")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


# =====================================================================
# Reporting
# =====================================================================

def write_text_summary(path: Path, summary: dict[str, Any], all_results: pd.DataFrame) -> None:
    best = all_results.iloc[0]

    lines = [
        "Location/time similarity correlation",
        "=" * 50,
        "",
        f"Location file: {summary['scripts_location_path']}",
        f"Similarity file: {summary['scripts_similarity_path']}",
        f"Output directory: {summary['output_dir']}",
        "",
        "Configuration:",
        f"  metric: {summary['metric']}",
        f"  k_folds: {summary['k_folds']}",
        f"  seed: {summary['seed']}",
        "",
        "Data:",
        f"  total pairs: {summary['n_total_pairs']}",
        f"  valid pairs: {summary['n_valid_pairs']}",
        f"  invalid pairs: {summary['n_invalid_pairs']}",
        f"  geographic metrics: {', '.join(summary['geo_metrics'])}",
        f"  temporal metrics: {', '.join(summary['time_metrics'])}",
        "",
        "Best model:",
        f"  model: {best['model']}",
        f"  family: {best['family']}",
        f"  predictor: {best['predictor']}",
        f"  geo_metric: {best['geo_metric']}",
        f"  time_metric: {best['time_metric']}",
        f"  in_sample_corr: {best['in_sample_corr']:.6f}",
        f"  cv_mean: {best['cv_mean']:.6f}",
        f"  cv_std: {best['cv_std']:.6f}",
        "",
        "Top 10 results:",
        all_results.head(10).to_string(index=False),
        "",
        "Full ranking:",
        all_results.to_string(index=False),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json_summary(path: Path, summary: dict[str, Any], all_results: pd.DataFrame) -> None:
    payload = {
        **summary,
        "best": all_results.iloc[0].to_dict(),
        "results": all_results.to_dict(orient="records"),
    }
    path.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


# =====================================================================
# Main
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--scripts-location", type=Path, default=DEFAULT_SCRIPTS_LOCATION)
    parser.add_argument("--scripts-similarities", type=Path, default=DEFAULT_SCRIPTS_SIMILARITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        choices=["pearson", "spearman"],
    )
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    corr_label = "Pearson r" if args.metric == "pearson" else "Spearman ρ"

    print(f"[load] locations: {args.scripts_location}")
    df = pd.read_excel(args.scripts_location)

    print(f"[load] similarities: {args.scripts_similarities}")
    with open(args.scripts_similarities, "rb") as f:
        sim = pickle.load(f)

    alphabets_files = sim["alphabets"]
    dist_dict = sim["alphabet_distances"]

    v_sim = np.asarray([dist_dict[k] for k in dist_dict], dtype=np.float64)
    pairs_scripts = [
        (
            short_name_from_filename(alphabets_files[i]),
            short_name_from_filename(alphabets_files[j]),
        )
        for (i, j) in dist_dict
    ]

    print("[compute] geographic distances")
    geo_funcs = {
        "geo_min": compute_geo_min,
        "geo_mean": compute_geo_mean,
        "geo_median": compute_geo_median,
        "geo_hausdorff": compute_geo_hausdorff,
        "geo_centroid": compute_geo_centroid,
    }
    geo_raw = {name: fn(df, pairs_scripts) for name, fn in geo_funcs.items()}

    print("[compute] temporal distances")
    time_funcs = {
        "time_L1": compute_time_L1,
        "time_L2": compute_time_L2,
        "time_Linf": compute_time_Linf,
        "time_midpoint": compute_time_midpoint,
        "time_gap": compute_time_gap,
        "time_jaccard": compute_time_jaccard,
    }
    time_raw = {name: fn(df, pairs_scripts) for name, fn in time_funcs.items()}

    valid = finite_mask(v_sim, *geo_raw.values(), *time_raw.values())
    n_total_pairs = int(len(v_sim))
    n_valid_pairs = int(valid.sum())
    n_invalid_pairs = int(n_total_pairs - n_valid_pairs)

    if n_valid_pairs == 0:
        raise ValueError("No valid pairs after filtering finite geographic/time/similarity values.")

    print(f"[data] valid pairs: {n_valid_pairs}/{n_total_pairs}")

    v_sim_all = v_sim[valid]
    geo_norm = {name: normalize_0_1(values[valid]) for name, values in geo_raw.items()}
    time_norm = {name: normalize_0_1(values[valid]) for name, values in time_raw.items()}

    # -----------------------------------------------------------------
    # Save pairwise data
    # -----------------------------------------------------------------
    pairwise_df = make_pairwise_dataframe(
        pairs_scripts=pairs_scripts,
        v_sim=v_sim,
        valid=valid,
        geo_raw=geo_raw,
        time_raw=time_raw,
    )
    pairwise_path = args.output_dir / "pairwise_location_similarity_data.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"[saved] {pairwise_path}")

    # -----------------------------------------------------------------
    # Single-predictor baselines
    # -----------------------------------------------------------------
    print("[evaluate] single-predictor baselines")
    baseline_df = build_baseline_results(
        geo_norm=geo_norm,
        time_norm=time_norm,
        v_sim_all=v_sim_all,
        n_pairs=n_valid_pairs,
        k_folds=args.k_folds,
        seed=args.seed,
        metric=args.metric,
    )
    baseline_path = args.output_dir / "baseline_single_predictor_results.csv"
    baseline_df.to_csv(baseline_path, index=False)
    print(f"[saved] {baseline_path}")

    # -----------------------------------------------------------------
    # Mixed geo-time models
    # -----------------------------------------------------------------
    print("[evaluate] mixed geo-time models")
    mixed_df = build_mixed_results(
        geo_norm=geo_norm,
        time_norm=time_norm,
        v_sim_all=v_sim_all,
        n_pairs=n_valid_pairs,
        k_folds=args.k_folds,
        seed=args.seed,
        metric=args.metric,
    )
    mixed_path = args.output_dir / "mixed_geo_time_results.csv"
    mixed_df.to_csv(mixed_path, index=False)
    print(f"[saved] {mixed_path}")

    # -----------------------------------------------------------------
    # Combined ranking
    # -----------------------------------------------------------------
    all_results = build_all_results(baseline_df, mixed_df)
    all_results_path = args.output_dir / "all_location_similarity_results.csv"
    all_results.to_csv(all_results_path, index=False)
    print(f"[saved] {all_results_path}")

    print("\n=== LOCATION/TIME SIMILARITY CORRELATION RESULTS ===")
    print(all_results.head(20).to_string(index=False))

    # -----------------------------------------------------------------
    # Heatmap table and plots
    # -----------------------------------------------------------------
    heatmap_df = build_heatmap_table(mixed_df)
    heatmap_csv_path = args.output_dir / f"heatmap_mixed_cv_{args.metric}.csv"
    heatmap_df.to_csv(heatmap_csv_path)
    print(f"[saved] {heatmap_csv_path}")

    heatmap_png_path = args.output_dir / f"heatmap_mixed_cv_{args.metric}.png"
    plot_heatmap(heatmap_df, corr_label=corr_label, out_path=heatmap_png_path)
    print(f"[saved] {heatmap_png_path}")

    baseline_png_path = args.output_dir / f"baseline_barplot_{args.metric}.png"
    plot_baseline_barplot(baseline_df, corr_label=corr_label, out_path=baseline_png_path)
    print(f"[saved] {baseline_png_path}")

    ranking_png_path = args.output_dir / "ranking_cv_mean.png"
    plot_ranking(all_results, out_path=ranking_png_path, top_n=20)
    print(f"[saved] {ranking_png_path}")

    # -----------------------------------------------------------------
    # Summaries
    # -----------------------------------------------------------------
    summary = {
        "scripts_location_path": str(args.scripts_location),
        "scripts_similarity_path": str(args.scripts_similarities),
        "output_dir": str(args.output_dir),
        "metric": args.metric,
        "correlation_label": corr_label,
        "k_folds": int(args.k_folds),
        "seed": int(args.seed),
        "n_total_pairs": n_total_pairs,
        "n_valid_pairs": n_valid_pairs,
        "n_invalid_pairs": n_invalid_pairs,
        "geo_metrics": list(geo_funcs.keys()),
        "time_metrics": list(time_funcs.keys()),
        "n_single_predictor_models": int(len(baseline_df)),
        "n_mixed_models": int(len(mixed_df)),
        "n_total_models": int(len(all_results)),
    }

    summary_txt_path = args.output_dir / "location_similarity_correlation_summary.txt"
    write_text_summary(summary_txt_path, summary, all_results)
    print(f"[saved] {summary_txt_path}")

    summary_json_path = args.output_dir / "location_similarity_correlation_summary.json"
    write_json_summary(summary_json_path, summary, all_results)
    print(f"[saved] {summary_json_path}")

    best = all_results.iloc[0]
    print("\nBest model:")
    print(f"  model = {best['model']}")
    print(f"  predictor = {best['predictor']}")
    print(f"  in_sample_corr = {best['in_sample_corr']:.4f}")
    print(f"  cv_mean = {best['cv_mean']:.4f}")
    print(f"  cv_std = {best['cv_std']:.4f}")


if __name__ == "__main__":
    main()
