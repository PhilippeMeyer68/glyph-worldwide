"""
Evaluate glyph-script distance metrics on a 3-tier synthetic validation dataset.

Labels:
    1.0 = closely related (same ancestor, high overlap, light distortion)
    0.5 = distantly related (same ancestor, low overlap, heavy distortion)
    0.0 = unrelated (different ancestors)

Ordinal evaluation:
    - Spearman rho between -distance and label
    - Mean pairwise AUC across the 3 label pairs
    - Per-tier mean distances

Use:
    python evaluate_synthetic_metric_dataset.py --metric-set light
    python evaluate_synthetic_metric_dataset.py --metric-set full
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from glyphworld.core.paths import RESULTS_DIR, MODELS_DIR
from glyphworld.core.utils import (
    load_glyph_dictionary,
    dist_dict_from_embeddings_cosine,
)
from glyphworld.distances.distance_functions import similarity_between_languages
from glyphworld.models.model import load_student_teacher_from_pt, load_teacher_from_pt


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------
DEFAULT_DATASET_PATH = (
    RESULTS_DIR / "validation" / "synthetic_pairs_metric_selection.pkl"
)

DEFAULT_OUTPUT_DIR = (
    RESULTS_DIR / "validation" / "metric_selection_evaluation"
)

DEFAULT_MODEL_KIND = "student_teacher"
DEFAULT_USE_TEACHER = False
DEFAULT_BATCH_SIZE = 256
DEFAULT_IMAGE_SIZE = 105
DEFAULT_METRIC_SET = "light"

MEAN = [0.5]
STD = [0.5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Metric names
# ---------------------------------------------------------------------
FULL_METRIC_NAMES = [
    "sym_min_mean",
    "directed_min_mean_12",
    "directed_min_mean_21",
    "sym_min_median",
    "sym_min_max",
    "hausdorff",
    "mean_all",
    "median_all",
    "min_all",
    "max_all",
    "trimmed_mean_all",
    "mean_of_row_means",
    "mean_of_col_means",
    "knn_mean_k3",
    "knn_mean_k5",
    "softmin_tau_001",
    "softmin_tau_01",
    "hungarian",
    "quantile_10",
    "quantile_20",
    "coverage_002",
]

LIGHT_METRIC_NAMES = [
    "sym_min_mean",
    "directed_min_mean_12",
    "directed_min_mean_21",
    "hausdorff",
    "mean_all",
    "median_all",
    "min_all",
    "max_all",
    "trimmed_mean_all",
    "hungarian",
]

METRIC_SETS = {
    "light": LIGHT_METRIC_NAMES,
    "full": FULL_METRIC_NAMES,
}


# ---------------------------------------------------------------------
# Model / embeddings helpers
# ---------------------------------------------------------------------
def build_transform(image_size: int = 105) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def imgs_to_batch(images: list[np.ndarray], tf, device) -> torch.Tensor:
    xs = []
    for img in images:
        x = tf(img)
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.mean(dim=0, keepdim=True)
        xs.append(x)
    return torch.stack(xs, dim=0).to(device, non_blocking=True)


@torch.no_grad()
def compute_embeddings(
    model,
    images: list[np.ndarray],
    tf,
    batch_size: int = 256,
    use_teacher: bool = True,
    model_kind: str = "student_teacher",
) -> np.ndarray:
    all_z = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch = imgs_to_batch(batch_imgs, tf=tf, device=DEVICE)

        if model_kind == "student_teacher":
            z = model.teacher(batch) if use_teacher else model.student(batch)
        elif model_kind == "teacher_only":
            z = model(batch)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        z = F.normalize(z, dim=1, p=2)
        all_z.append(z.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(all_z, axis=0)


def load_metric_model(model_kind: str):
    if model_kind == "student_teacher":
        ckpt_path = MODELS_DIR / "css" / "student_teacher_final.pt"
        model = load_student_teacher_from_pt(
            ckpt_path=ckpt_path,
            device=DEVICE,
            normalize=True,
            strict=True,
        )
        model.eval()
        return model, ckpt_path

    if model_kind == "teacher_only":
        ckpt_path = MODELS_DIR / "css" / "teacher_pretrained.pt"
        model = load_teacher_from_pt(
            ckpt_path=ckpt_path,
            device=DEVICE,
            normalize=True,
            strict=True,
        )
        model.eval()
        return model, ckpt_path

    raise ValueError(f"Unsupported model_kind: {model_kind}")


# ---------------------------------------------------------------------
# Synthetic dataset parsing helpers
# ---------------------------------------------------------------------
def ensure_uint8_bw(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)

    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected 3D image shape: {arr.shape}")

    if arr.ndim != 2:
        raise ValueError(f"Expected a single 2D glyph image, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    return arr


def _is_single_image(x) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    if x.ndim == 2:
        return True
    if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
        return True
    return False


def _collect_images_recursive(obj, out: list[np.ndarray]) -> None:
    if obj is None:
        return

    if _is_single_image(obj):
        out.append(ensure_uint8_bw(obj))
        return

    if isinstance(obj, dict):
        preferred_keys = [
            "glyph_images",
            "glyphs",
            "images",
            "imgs",
            "variants",
            "augmented_glyphs",
            "samples",
        ]

        found = False
        for key in preferred_keys:
            if key in obj:
                _collect_images_recursive(obj[key], out)
                found = True

        if not found:
            for value in obj.values():
                _collect_images_recursive(value, out)

        return

    if isinstance(obj, (list, tuple)):
        for value in obj:
            _collect_images_recursive(value, out)
        return


def extract_script_glyphs(script_obj) -> list[np.ndarray]:
    imgs: list[np.ndarray] = []
    _collect_images_recursive(script_obj, imgs)

    if len(imgs) == 0:
        raise ValueError(
            "Unsupported synthetic script format or no glyph images found. "
            f"Top-level type: {type(script_obj)}"
        )

    return imgs


def get_pair_ids(pair: dict[str, Any]) -> tuple[str, str]:
    left_keys = ["left_script_id", "script_id_left", "left_id", "script1_id", "a_id"]
    right_keys = ["right_script_id", "script_id_right", "right_id", "script2_id", "b_id"]

    left = next((pair[key] for key in left_keys if key in pair), None)
    right = next((pair[key] for key in right_keys if key in pair), None)

    if left is None or right is None:
        raise KeyError("Could not find pair script identifiers.")

    return str(left), str(right)


# ---------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------
def cosine_distance_matrix_dense(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    z1 = np.asarray(z1, dtype=np.float32)
    z2 = np.asarray(z2, dtype=np.float32)

    z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-12)
    z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-12)

    D = 1.0 - z1 @ z2.T
    D = np.clip(D, 0.0, 2.0)

    return D.astype(np.float32, copy=False)


def trimmed_mean(x: np.ndarray, proportion_to_cut: float = 0.1) -> float:
    x = np.sort(np.asarray(x, dtype=float).ravel())
    n = len(x)

    if n == 0:
        return float("nan")

    k = int(np.floor(proportion_to_cut * n))

    if 2 * k >= n:
        return float(np.mean(x))

    return float(np.mean(x[k : n - k]))


def stable_softmin_mean(D: np.ndarray, tau: float) -> float:
    """
    Computes the symmetric soft-min aggregation:

        -tau * log(mean(exp(-D / tau)))

    using logsumexp for numerical stability.
    """
    D = np.asarray(D, dtype=np.float64)

    row_soft = -tau * (logsumexp(-D / tau, axis=1) - np.log(D.shape[1]))
    col_soft = -tau * (logsumexp(-D / tau, axis=0) - np.log(D.shape[0]))

    return float(0.5 * (np.mean(row_soft) + np.mean(col_soft)))


def aggregate_distance_metric(z1: np.ndarray, z2: np.ndarray, metric_name: str) -> float:
    D = cosine_distance_matrix_dense(z1, z2)
    row_mins = D.min(axis=1)
    col_mins = D.min(axis=0)

    if metric_name == "sym_min_mean":
        dist_dict = dist_dict_from_embeddings_cosine(z1, z2)
        return float(similarity_between_languages(dist_dict))

    if metric_name == "directed_min_mean_12":
        return float(np.mean(row_mins))

    if metric_name == "directed_min_mean_21":
        return float(np.mean(col_mins))

    if metric_name == "sym_min_median":
        return float((np.median(row_mins) + np.median(col_mins)) / 2.0)

    if metric_name == "sym_min_max":
        return float((np.max(row_mins) + np.max(col_mins)) / 2.0)

    if metric_name == "hausdorff":
        return float(max(np.max(row_mins), np.max(col_mins)))

    if metric_name == "mean_all":
        return float(np.mean(D))

    if metric_name == "median_all":
        return float(np.median(D))

    if metric_name == "min_all":
        return float(np.min(D))

    if metric_name == "max_all":
        return float(np.max(D))

    if metric_name == "trimmed_mean_all":
        return float(trimmed_mean(D, proportion_to_cut=0.1))

    if metric_name == "mean_of_row_means":
        return float(np.mean(D.mean(axis=1)))

    if metric_name == "mean_of_col_means":
        return float(np.mean(D.mean(axis=0)))

    if metric_name == "knn_mean_k3":
        k = min(3, D.shape[1], D.shape[0])
        row_knn = np.sort(D, axis=1)[:, :k].mean()
        col_knn = np.sort(D, axis=0)[:k, :].mean()
        return float(0.5 * (row_knn + col_knn))

    if metric_name == "knn_mean_k5":
        k = min(5, D.shape[1], D.shape[0])
        row_knn = np.sort(D, axis=1)[:, :k].mean()
        col_knn = np.sort(D, axis=0)[:k, :].mean()
        return float(0.5 * (row_knn + col_knn))

    if metric_name == "softmin_tau_001":
        return stable_softmin_mean(D, tau=0.01)

    if metric_name == "softmin_tau_01":
        return stable_softmin_mean(D, tau=0.1)

    if metric_name == "hungarian":
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(D)
        return float(D[row_ind, col_ind].mean())

    if metric_name == "quantile_10":
        return float(np.quantile(D, 0.1))

    if metric_name == "quantile_20":
        return float(np.quantile(D, 0.2))

    if metric_name == "coverage_002":
        threshold = 0.02
        coverage = 0.5 * ((row_mins < threshold).mean() + (col_mins < threshold).mean())
        return float(1.0 - coverage)

    raise ValueError(f"Unknown metric_name: {metric_name}")


def compute_all_script_embeddings(
    synthetic_scripts: dict[str, Any],
    model,
    model_kind: str,
    use_teacher: bool,
    batch_size: int,
    image_size: int,
) -> dict[str, np.ndarray]:
    tf = build_transform(image_size=image_size)
    out = {}

    for script_id, script_obj in synthetic_scripts.items():
        imgs = extract_script_glyphs(script_obj)

        if len(imgs) == 0:
            raise ValueError(f"Synthetic script {script_id!r} contains no glyphs.")

        z = compute_embeddings(
            model=model,
            images=imgs,
            tf=tf,
            batch_size=batch_size,
            use_teacher=use_teacher,
            model_kind=model_kind,
        )

        if len(out) < 5:
            print(
                f"[INFO] {script_id}: n_imgs={len(imgs)}, "
                f"emb_shape={z.shape}, emb_std={float(np.std(z)):.6f}"
            )

        out[str(script_id)] = z

    return out


def compute_pair_metric_scores(
    pairs: list[dict[str, Any]],
    script_embeddings: dict[str, np.ndarray],
    metric_names: list[str],
    max_pairs: int | None = None,
) -> pd.DataFrame:
    rows = []
    iterable_pairs = pairs if max_pairs is None else pairs[:max_pairs]

    for idx, pair in enumerate(iterable_pairs):
        left_id, right_id = get_pair_ids(pair)

        if left_id not in script_embeddings:
            raise KeyError(f"Missing embeddings for left script: {left_id}")
        if right_id not in script_embeddings:
            raise KeyError(f"Missing embeddings for right script: {right_id}")

        z1 = script_embeddings[left_id]
        z2 = script_embeddings[right_id]

        base_row = {
            "pair_id": pair.get("pair_id", idx),
            "left_id": left_id,
            "right_id": right_id,
            "label": float(pair["label"]),
            "split": pair.get("split", "unknown"),
        }

        for metric_name in metric_names:
            row = dict(base_row)
            distance = aggregate_distance_metric(z1, z2, metric_name)

            if not np.isfinite(distance):
                print(
                    f"[WARNING] Non-finite distance detected: "
                    f"metric={metric_name}, pair_id={base_row['pair_id']}, "
                    f"left={left_id}, right={right_id}, distance={distance}"
                )

            row["metric"] = metric_name
            row["distance"] = distance
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Ordinal evaluation helpers
# ---------------------------------------------------------------------
def finite_filter(labels: np.ndarray, distances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=float)
    distances = np.asarray(distances, dtype=float)

    mask = np.isfinite(labels) & np.isfinite(distances)

    return labels[mask], distances[mask]


def pairwise_auc(labels: np.ndarray, distances: np.ndarray, la: float, lb: float) -> float:
    """
    Binary AUC between two label classes.

    la is the higher / positive class.
    lb is the lower / negative class.

    Lower distance means stronger predicted relatedness,
    so the AUC score uses -distance.
    """
    labels, distances = finite_filter(labels, distances)

    mask = np.isin(labels, [la, lb])
    y = (labels[mask] == la).astype(int)
    d = distances[mask]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")

    return float(roc_auc_score(y, -d))


def mean_pairwise_auc(labels: np.ndarray, distances: np.ndarray) -> float:
    tier_pairs = [(1.0, 0.5), (1.0, 0.0), (0.5, 0.0)]
    aucs = [pairwise_auc(labels, distances, hi, lo) for hi, lo in tier_pairs]
    valid = [auc for auc in aucs if np.isfinite(auc)]

    return float(np.mean(valid)) if valid else float("nan")


def summarize_one_metric_ordinal(
    labels: np.ndarray,
    distances: np.ndarray,
) -> dict[str, Any]:
    labels, distances = finite_filter(labels, distances)

    if len(labels) == 0:
        return {
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "mean_pairwise_auc": float("nan"),
            "auc_close_vs_distant": float("nan"),
            "auc_close_vs_unrelated": float("nan"),
            "auc_distant_vs_unrelated": float("nan"),
            "distance_mean_close": float("nan"),
            "distance_mean_distant": float("nan"),
            "distance_mean_unrelated": float("nan"),
            "distance_std_close": float("nan"),
            "distance_std_distant": float("nan"),
            "distance_std_unrelated": float("nan"),
            "n_valid": 0,
            "n_invalid": int(len(distances)),
        }

    rho, p_value = spearmanr(-distances, labels)

    auc_1_vs_05 = pairwise_auc(labels, distances, 1.0, 0.5)
    auc_1_vs_0 = pairwise_auc(labels, distances, 1.0, 0.0)
    auc_05_vs_0 = pairwise_auc(labels, distances, 0.5, 0.0)
    mp_auc = mean_pairwise_auc(labels, distances)

    d_close = distances[labels == 1.0]
    d_distant = distances[(labels > 0) & (labels < 1.0)]
    d_unrelated = distances[labels == 0.0]

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_value),
        "mean_pairwise_auc": mp_auc,
        "auc_close_vs_distant": auc_1_vs_05,
        "auc_close_vs_unrelated": auc_1_vs_0,
        "auc_distant_vs_unrelated": auc_05_vs_0,
        "distance_mean_close": float(np.mean(d_close)) if len(d_close) > 0 else float("nan"),
        "distance_mean_distant": float(np.mean(d_distant)) if len(d_distant) > 0 else float("nan"),
        "distance_mean_unrelated": float(np.mean(d_unrelated)) if len(d_unrelated) > 0 else float("nan"),
        "distance_std_close": float(np.std(d_close)) if len(d_close) > 0 else float("nan"),
        "distance_std_distant": float(np.std(d_distant)) if len(d_distant) > 0 else float("nan"),
        "distance_std_unrelated": float(np.std(d_unrelated)) if len(d_unrelated) > 0 else float("nan"),
        "n_valid": int(len(labels)),
    }


def build_ordinal_comparison_table(pair_scores_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric_name, sub in pair_scores_df.groupby("metric"):
        labels = sub["label"].to_numpy()
        distances = sub["distance"].to_numpy()
        n_total = len(distances)
        n_invalid = int(np.sum(~np.isfinite(distances)))

        s = summarize_one_metric_ordinal(labels, distances)
        rows.append(
            {
                "metric": metric_name,
                **s,
                "n_total": n_total,
                "n_invalid": n_invalid,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["spearman_rho", "mean_pairwise_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return out


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_distance_histograms_3class(
    distances: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    labels, distances = finite_filter(labels, distances)

    close = distances[labels == 1.0]
    distant = distances[(labels > 0) & (labels < 1.0)]
    unrelated = distances[labels == 0.0]

    plt.figure(figsize=(10, 7))

    if len(close) > 0:
        plt.hist(
            close,
            bins=40,
            alpha=0.5,
            density=True,
            label="close (label=1.0)",
            color="tab:blue",
        )

    if len(distant) > 0:
        plt.hist(
            distant,
            bins=40,
            alpha=0.5,
            density=True,
            label="distant (label=0.5)",
            color="tab:orange",
        )

    if len(unrelated) > 0:
        plt.hist(
            unrelated,
            bins=40,
            alpha=0.5,
            density=True,
            label="unrelated (label=0.0)",
            color="tab:red",
        )

    plt.xlabel("Script distance")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def write_text_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "Synthetic metric selection (3-tier ordinal)",
        "=" * 48,
        "",
        f"Dataset: {summary['dataset_path']}",
        f"Output dir: {summary['output_dir']}",
        f"Total pairs: {summary['n_pairs']}",
        f"  close (1.0): {summary['n_close']}",
        f"  distant (0.5): {summary['n_distant']}",
        f"  unrelated (0.0): {summary['n_unrelated']}",
        f"Model kind: {summary['model_kind']}",
        f"Use teacher: {summary['use_teacher']}",
        f"Checkpoint: {summary['checkpoint_path']}",
        f"Device: {summary['device']}",
        "",
        f"Metric set: {summary['metric_set']}",
        f"Metric names: {', '.join(summary['metric_names'])}",
        "",
        f"Best metric: {summary['best_metric']}",
        f"  Spearman rho: {summary['best_spearman_rho']:.6f}",
        f"  Mean pairwise AUC: {summary['best_mean_pairwise_auc']:.6f}",
        f"  AUC close vs distant: {summary['best_auc_close_vs_distant']:.6f}",
        f"  AUC close vs unrelated: {summary['best_auc_close_vs_unrelated']:.6f}",
        f"  AUC distant vs unrelated: {summary['best_auc_distant_vs_unrelated']:.6f}",
        f"  Mean distance (close): {summary['best_distance_mean_close']:.6f}",
        f"  Mean distance (distant): {summary['best_distance_mean_distant']:.6f}",
        f"  Mean distance (unrelated): {summary['best_distance_mean_unrelated']:.6f}",
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
    parser.add_argument("--max-pairs", type=int, default=None)

    parser.add_argument(
        "--metric-set",
        type=str,
        default=DEFAULT_METRIC_SET,
        choices=["light", "full"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_names = METRIC_SETS[args.metric_set]

    dataset = load_glyph_dictionary(args.dataset_path)
    pairs = dataset["pairs"]
    synthetic_scripts = dataset["synthetic_scripts"]
    dataset_config = dataset.get("config", {})

    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    if len(pairs) == 0:
        raise ValueError("No pairs found in dataset.")

    labels = np.array([float(pair["label"]) for pair in pairs])
    n_close = int(np.sum(labels == 1.0))
    n_distant = int(np.sum((labels > 0) & (labels < 1.0)))
    n_unrelated = int(np.sum(labels == 0.0))

    print(f"[INFO] pairs: close={n_close}, distant={n_distant}, unrelated={n_unrelated}")

    model, ckpt_path = load_metric_model(args.model_kind)

    script_embeddings = compute_all_script_embeddings(
        synthetic_scripts=synthetic_scripts,
        model=model,
        model_kind=args.model_kind,
        use_teacher=args.use_teacher,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    print(f"\n[INFO] metric_set={args.metric_set}")
    print(f"[INFO] metrics={metric_names}")

    pair_scores_df = compute_pair_metric_scores(
        pairs=pairs,
        script_embeddings=script_embeddings,
        metric_names=metric_names,
        max_pairs=None,
    )

    pair_scores_path = args.output_dir / f"pair_scores_{args.metric_set}_metrics.csv"
    pair_scores_df.to_csv(pair_scores_path, index=False)

    comparison_all = build_ordinal_comparison_table(pair_scores_df)
    comparison_all_path = args.output_dir / f"metric_comparison_all_{args.metric_set}.csv"
    comparison_all.to_csv(comparison_all_path, index=False)

    print("\n=== METRIC COMPARISON (ALL) ===")
    print(comparison_all.to_string(index=False))

    has_test = np.any(pair_scores_df["split"] == "test")
    if has_test:
        comparison_test = build_ordinal_comparison_table(
            pair_scores_df[pair_scores_df["split"] == "test"].copy()
        )
        comparison_test_path = args.output_dir / f"metric_comparison_test_{args.metric_set}.csv"
        comparison_test.to_csv(comparison_test_path, index=False)

        print("\n=== METRIC COMPARISON (TEST) ===")
        print(comparison_test.to_string(index=False))

    if len(comparison_all) == 0:
        raise ValueError("No metric comparison results were produced.")

    best_metric = str(comparison_all.iloc[0]["metric"])
    best_row = comparison_all.iloc[0]

    best_sub = pair_scores_df[pair_scores_df["metric"] == best_metric]

    plot_distance_histograms_3class(
        distances=best_sub["distance"].to_numpy(),
        labels=best_sub["label"].to_numpy(),
        out_path=args.output_dir / f"distance_histograms_3class_{args.metric_set}.png",
        title=f"3-tier distance distributions ({best_metric})",
    )

    for metric_name in metric_names:
        sub = pair_scores_df[pair_scores_df["metric"] == metric_name]

        plot_distance_histograms_3class(
            distances=sub["distance"].to_numpy(),
            labels=sub["label"].to_numpy(),
            out_path=args.output_dir / f"distance_histograms_3class_{metric_name}_{args.metric_set}.png",
            title=f"3-tier distance distributions ({metric_name})",
        )

    summary = {
        "dataset_path": str(args.dataset_path),
        "output_dir": str(args.output_dir),
        "n_pairs": int(len(pairs)),
        "n_close": n_close,
        "n_distant": n_distant,
        "n_unrelated": n_unrelated,
        "model_kind": args.model_kind,
        "use_teacher": bool(args.use_teacher),
        "checkpoint_path": str(ckpt_path),
        "device": str(DEVICE),
        "metric_set": args.metric_set,
        "metric_names": metric_names,
        "best_metric": best_metric,
        "best_spearman_rho": float(best_row["spearman_rho"]),
        "best_mean_pairwise_auc": float(best_row["mean_pairwise_auc"]),
        "best_auc_close_vs_distant": float(best_row["auc_close_vs_distant"]),
        "best_auc_close_vs_unrelated": float(best_row["auc_close_vs_unrelated"]),
        "best_auc_distant_vs_unrelated": float(best_row["auc_distant_vs_unrelated"]),
        "best_distance_mean_close": float(best_row["distance_mean_close"]),
        "best_distance_mean_distant": float(best_row["distance_mean_distant"]),
        "best_distance_mean_unrelated": float(best_row["distance_mean_unrelated"]),
        "dataset_config": dataset_config,
    }

    summary_txt_path = args.output_dir / f"summary_{args.metric_set}.txt"

    write_text_summary(summary_txt_path, summary)

    print(f"\nBest metric: {best_metric}")
    print(f"  Spearman rho = {best_row['spearman_rho']:.4f}")
    print(f"  Mean pairwise AUC = {best_row['mean_pairwise_auc']:.4f}")
    print(f"  AUC close vs distant = {best_row['auc_close_vs_distant']:.4f}")
    print(f"  AUC close vs unrelated = {best_row['auc_close_vs_unrelated']:.4f}")
    print(f"  AUC distant vs unrelated = {best_row['auc_distant_vs_unrelated']:.4f}")

    print(f"\nSaved pair scores to: {pair_scores_path}")
    print(f"Saved comparison to: {comparison_all_path}")


if __name__ == "__main__":
    main()