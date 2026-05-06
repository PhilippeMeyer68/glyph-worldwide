from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from scipy.stats import pearsonr, spearmanr

from glyphworld.core.paths import PROCESSED_DIR, RESULTS_DIR, MODELS_DIR
from glyphworld.core.utils import dist_dict_from_embeddings_cosine
from glyphworld.distances.distance_functions import similarity_between_languages
from glyphworld.models.model import load_student_teacher_from_pt, load_teacher_from_pt


# =============================================================================
# Config
# =============================================================================

IN_PATH = PROCESSED_DIR / "font_sensitivity" / "alphabets_font_variants.pkl"

OUT_DIR = RESULTS_DIR / "font_sensitivity"

OUT_SCRIPT_PAIRWISE_CSV = OUT_DIR / "script_pair_distances_by_font.csv"
OUT_SCRIPT_STABILITY_CSV = OUT_DIR / "script_font_stability_summary.csv"

OUT_GLYPH_PAIRWISE_CSV = OUT_DIR / "glyph_font_sensitivity.csv"
OUT_GLYPH_SCRIPT_SUMMARY_CSV = OUT_DIR / "glyph_font_sensitivity_by_script.csv"
OUT_GLYPH_FONT_PAIR_SUMMARY_CSV = OUT_DIR / "glyph_font_sensitivity_by_font_pair.csv"

OUT_SCRIPT_PAIRWISE_PNG = OUT_DIR / "script_pair_distances_by_font.png"
OUT_SCRIPT_CORR_PNG = OUT_DIR / "script_font_rank_correlation.png"
OUT_GLYPH_SCRIPT_PNG = OUT_DIR / "glyph_font_sensitivity_by_script.png"
OUT_GLYPH_FONT_PAIR_PNG = OUT_DIR / "glyph_font_sensitivity_by_font_pair.png"

MODEL_KIND = "student_teacher"  # "student_teacher" or "teacher_only"
USE_TEACHER = False

BATCH_SIZE = 256
IMAGE_SIZE = 105
MEAN = [0.5]
STD = [0.5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model helpers
# =============================================================================

def build_transform(image_size: int = 105):
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def imgs_to_batch(images: list[np.ndarray], tf, device: torch.device) -> torch.Tensor:
    xs = []

    for img in images:
        x = tf(img)

        if x.ndim == 3 and x.shape[0] == 3:
            x = x.mean(dim=0, keepdim=True)

        xs.append(x)

    return torch.stack(xs, dim=0).to(device)


@torch.no_grad()
def compute_embeddings(model, images: list[np.ndarray], tf) -> np.ndarray:
    all_z = []

    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i : i + BATCH_SIZE]
        batch = imgs_to_batch(batch_imgs, tf, DEVICE)

        if MODEL_KIND == "student_teacher":
            z = model.teacher(batch) if USE_TEACHER else model.student(batch)
        elif MODEL_KIND == "teacher_only":
            z = model(batch)
        else:
            raise ValueError(f"Unknown MODEL_KIND: {MODEL_KIND}")

        z = F.normalize(z, dim=1, p=2)
        all_z.append(z.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(all_z, axis=0)


def load_model():
    if MODEL_KIND == "student_teacher":
        ckpt_path = MODELS_DIR / "css" / "student_teacher_final.pt"
        model = load_student_teacher_from_pt(
            ckpt_path=ckpt_path,
            device=DEVICE,
            normalize=True,
            strict=True,
        )

    elif MODEL_KIND == "teacher_only":
        ckpt_path = MODELS_DIR / "css" / "teacher_pretrained.pt"
        model = load_teacher_from_pt(
            ckpt_path=ckpt_path,
            device=DEVICE,
            normalize=True,
            strict=True,
        )

    else:
        raise ValueError(f"Unknown MODEL_KIND: {MODEL_KIND}")

    model.eval()
    return model


# =============================================================================
# Distance helpers
# =============================================================================

def script_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    dist_dict = dist_dict_from_embeddings_cosine(z1, z2)
    return float(similarity_between_languages(dist_dict))


def cosine_distance(z1: np.ndarray, z2: np.ndarray) -> float:
    return float(1.0 - np.dot(z1, z2))


def safe_corr(x, y, kind: str = "spearman") -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    if kind == "spearman":
        r, p = spearmanr(x, y)
    elif kind == "pearson":
        r, p = pearsonr(x, y)
    else:
        raise ValueError(kind)

    return float(r), float(p)


# =============================================================================
# Plot helpers
# =============================================================================

def save_or_show(path: Path | None = None) -> None:
    plt.tight_layout()

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[PLOT SAVED] {path}")

    plt.show()
    plt.close()


def plot_script_pairwise_distances(df_wide: pd.DataFrame, out_path: Path | None = None) -> None:
    if len(df_wide) > 30:
        print("[SKIP PLOT] Too many script pairs for readable bar plot.")
        return

    font_cols = [c for c in df_wide.columns if c not in ["script_a", "script_b", "pair"]]

    if len(font_cols) < 2:
        print("[SKIP PLOT] Fewer than two fonts available.")
        return

    x = np.arange(len(df_wide))
    width = 0.8 / len(font_cols)

    plt.figure(figsize=(8, 4))

    for i, font in enumerate(font_cols):
        plt.bar(
            x + i * width,
            df_wide[font].values,
            width=width,
            label=font,
        )

    plt.xticks(
        x + width * (len(font_cols) - 1) / 2,
        df_wide["pair"],
        rotation=45,
        ha="right",
    )
    plt.ylabel("Script distance")
    plt.title("Inter-script distances under different fonts")
    plt.legend()
    save_or_show(out_path)


def plot_script_font_correlation(df_wide: pd.DataFrame, out_path: Path | None = None) -> None:
    font_cols = [c for c in df_wide.columns if c not in ["script_a", "script_b", "pair"]]

    if len(font_cols) < 2:
        print("[SKIP PLOT] Fewer than two fonts available.")
        return

    corr = df_wide[font_cols].corr(method="spearman", min_periods=2)

    plt.figure(figsize=(4.5, 4))
    plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.xticks(range(len(font_cols)), font_cols, rotation=45, ha="right")
    plt.yticks(range(len(font_cols)), font_cols)
    plt.colorbar(label="Spearman correlation")
    plt.title("Stability of script-distance rankings")

    for i in range(len(font_cols)):
        for j in range(len(font_cols)):
            value = corr.values[i, j]
            if np.isfinite(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    save_or_show(out_path)


def plot_script_font_pair_scatter(
    df_wide: pd.DataFrame,
    font_a: str,
    font_b: str,
    out_path: Path | None = None,
) -> None:
    sub = df_wide[[font_a, font_b]].dropna()

    if len(sub) < 2:
        print(f"[SKIP PLOT] {font_a} vs {font_b}: fewer than two shared script pairs.")
        return

    x = sub[font_a].values
    y = sub[font_b].values

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y)

    min_v = min(x.min(), y.min())
    max_v = max(x.max(), y.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel(font_a)
    plt.ylabel(font_b)
    plt.title(f"Script distances: {font_a} vs {font_b}")
    save_or_show(out_path)


def plot_glyph_sensitivity_by_script(
    df_summary: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    if df_summary.empty:
        print("[SKIP PLOT] Empty glyph summary by script.")
        return

    df_plot = df_summary.sort_values("mean")

    plt.figure(figsize=(9, 4))
    plt.bar(df_plot["script"], df_plot["mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean same-glyph cosine distance")
    plt.title("Glyph-level font sensitivity by script")
    save_or_show(out_path)


def plot_glyph_sensitivity_by_font_pair(
    df_summary: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    if df_summary.empty:
        print("[SKIP PLOT] Empty glyph summary by font pair.")
        return

    df_plot = df_summary.copy()
    df_plot["font_pair"] = df_plot["font_a"] + " - " + df_plot["font_b"]
    df_plot = df_plot.sort_values("mean")

    plt.figure(figsize=(9, 4))
    plt.bar(df_plot["font_pair"], df_plot["mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean same-glyph cosine distance")
    plt.title("Glyph-level font sensitivity by font pair")
    save_or_show(out_path)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(IN_PATH, "rb") as f:
        data = pickle.load(f)

    alphabets = data["alphabets"]

    scripts = list(alphabets.keys())
    all_font_names = sorted(
        {font_name for font_dict in alphabets.values() for font_name in font_dict.keys()}
    )

    print(f"[LOADED] {IN_PATH}")
    print("Scripts:", scripts)
    print("Fonts:", all_font_names)
    print("Device:", DEVICE)

    model = load_model()
    tf = build_transform(IMAGE_SIZE)

    # -------------------------------------------------------------------------
    # 1. Compute embeddings for each available script/font combination
    # -------------------------------------------------------------------------
    embeddings: dict[str, dict[str, dict[str, object]]] = {}

    for script, font_dict in alphabets.items():
        embeddings[script] = {}

        for font_name, glyph_dict in font_dict.items():
            keys = list(glyph_dict.keys())
            images = [glyph_dict[k] for k in keys]

            print(f"[EMBED] {script} | {font_name} | n={len(images)}")

            z = compute_embeddings(model, images, tf)

            embeddings[script][font_name] = {
                "keys": keys,
                "z": z,
            }

    # -------------------------------------------------------------------------
    # 2. Script-level font sensitivity
    # -------------------------------------------------------------------------
    script_rows = []

    for font_name in all_font_names:
        print(f"\n[SCRIPT FONT] {font_name}")

        available_scripts = [
            script for script in scripts
            if font_name in embeddings[script]
        ]

        if len(available_scripts) < 2:
            print(f"  [SKIP] font={font_name}: fewer than two scripts available")
            continue

        for script_a, script_b in combinations(available_scripts, 2):
            z_a = embeddings[script_a][font_name]["z"]
            z_b = embeddings[script_b][font_name]["z"]

            d = script_distance(z_a, z_b)

            script_rows.append(
                {
                    "font": font_name,
                    "script_a": script_a,
                    "script_b": script_b,
                    "pair": f"{script_a} - {script_b}",
                    "distance": d,
                }
            )

            print(f"{script_a:25s} vs {script_b:25s} | d={d:.4f}")

    df_script_long = pd.DataFrame(script_rows)
    df_script_long.to_csv(OUT_SCRIPT_PAIRWISE_CSV, index=False)

    if df_script_long.empty:
        raise RuntimeError("No script-level distances were computed.")

    df_script_wide = df_script_long.pivot_table(
        index=["script_a", "script_b", "pair"],
        columns="font",
        values="distance",
    ).reset_index()

    df_script_wide.columns.name = None

    print("\n=== Script distances by font ===")
    print(df_script_wide)

    script_summary_rows = []

    font_cols = [
        c for c in df_script_wide.columns
        if c not in ["script_a", "script_b", "pair"]
    ]

    for font_a, font_b in combinations(font_cols, 2):
        sub = df_script_wide[["pair", font_a, font_b]].dropna()

        if len(sub) < 2:
            print(f"[SKIP] {font_a} vs {font_b}: fewer than two shared script pairs")
            continue

        x = sub[font_a].values
        y = sub[font_b].values

        spearman_r, spearman_p = safe_corr(x, y, kind="spearman")
        pearson_r, pearson_p = safe_corr(x, y, kind="pearson")

        abs_diff = np.abs(x - y)

        script_summary_rows.append(
            {
                "font_a": font_a,
                "font_b": font_b,
                "n_script_pairs": len(sub),
                "spearman_r": spearman_r,
                "spearman_p": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "mean_abs_distance_change": float(np.mean(abs_diff)),
                "median_abs_distance_change": float(np.median(abs_diff)),
                "max_abs_distance_change": float(np.max(abs_diff)),
            }
        )

    df_script_summary = pd.DataFrame(script_summary_rows)
    df_script_summary.to_csv(OUT_SCRIPT_STABILITY_CSV, index=False)

    print("\n=== Script-level font stability summary ===")
    print(df_script_summary)

    # -------------------------------------------------------------------------
    # 3. Glyph-level font sensitivity
    # -------------------------------------------------------------------------
    glyph_rows = []

    for script in scripts:
        print(f"\n[GLYPH FONT SENSITIVITY] {script}")

        available_fonts = list(embeddings[script].keys())

        if len(available_fonts) < 2:
            print("  [SKIP] fewer than two fonts available")
            continue

        for font_a, font_b in combinations(available_fonts, 2):
            keys_a = embeddings[script][font_a]["keys"]
            keys_b = embeddings[script][font_b]["keys"]

            z_a = embeddings[script][font_a]["z"]
            z_b = embeddings[script][font_b]["z"]

            idx_a = {k: i for i, k in enumerate(keys_a)}
            idx_b = {k: i for i, k in enumerate(keys_b)}

            common_keys = sorted(set(keys_a) & set(keys_b))

            if len(common_keys) == 0:
                print(f"  {font_a} vs {font_b}: no shared glyphs")
                continue

            distances = []

            for glyph in common_keys:
                va = z_a[idx_a[glyph]]
                vb = z_b[idx_b[glyph]]

                d = cosine_distance(va, vb)
                distances.append(d)

                glyph_rows.append(
                    {
                        "script": script,
                        "glyph": glyph,
                        "font_a": font_a,
                        "font_b": font_b,
                        "same_glyph_cosine_distance": d,
                    }
                )

            print(
                f"  {font_a:20s} vs {font_b:20s} | "
                f"n={len(common_keys):4d} | "
                f"mean={np.mean(distances):.4f} | "
                f"max={np.max(distances):.4f}"
            )

    df_glyph = pd.DataFrame(glyph_rows)
    df_glyph.to_csv(OUT_GLYPH_PAIRWISE_CSV, index=False)

    if df_glyph.empty:
        print("[WARNING] No glyph-level font sensitivity rows were computed.")
        df_glyph_script_summary = pd.DataFrame()
        df_glyph_font_pair_summary = pd.DataFrame()

    else:
        df_glyph_script_summary = (
            df_glyph
            .groupby("script")["same_glyph_cosine_distance"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )

        df_glyph_font_pair_summary = (
            df_glyph
            .groupby(["font_a", "font_b"])["same_glyph_cosine_distance"]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .reset_index()
        )

    df_glyph_script_summary.to_csv(OUT_GLYPH_SCRIPT_SUMMARY_CSV, index=False)
    df_glyph_font_pair_summary.to_csv(OUT_GLYPH_FONT_PAIR_SUMMARY_CSV, index=False)

    print("\n=== Glyph-level font sensitivity by script ===")
    print(df_glyph_script_summary)

    print("\n=== Glyph-level font sensitivity by font pair ===")
    print(df_glyph_font_pair_summary)

    # -------------------------------------------------------------------------
    # 4. Plots
    # -------------------------------------------------------------------------
    plot_script_pairwise_distances(df_script_wide, OUT_SCRIPT_PAIRWISE_PNG)
    plot_script_font_correlation(df_script_wide, OUT_SCRIPT_CORR_PNG)

    for _, row in df_script_summary.iterrows():
        font_a = row["font_a"]
        font_b = row["font_b"]
        out_path = OUT_DIR / f"script_distance_scatter_{font_a}_vs_{font_b}.png"
        plot_script_font_pair_scatter(df_script_wide, font_a, font_b, out_path)

    plot_glyph_sensitivity_by_script(df_glyph_script_summary, OUT_GLYPH_SCRIPT_PNG)
    plot_glyph_sensitivity_by_font_pair(df_glyph_font_pair_summary, OUT_GLYPH_FONT_PAIR_PNG)

    # -------------------------------------------------------------------------
    # 5. Saved files
    # -------------------------------------------------------------------------
    print("\n[SAVED]")
    print(OUT_SCRIPT_PAIRWISE_CSV)
    print(OUT_SCRIPT_STABILITY_CSV)
    print(OUT_GLYPH_PAIRWISE_CSV)
    print(OUT_GLYPH_SCRIPT_SUMMARY_CSV)
    print(OUT_GLYPH_FONT_PAIR_SUMMARY_CSV)