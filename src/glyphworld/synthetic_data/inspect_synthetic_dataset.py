"""
Summary statistics and plots of the synthetic benchmarks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glyphworld.core.paths import RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary


OUTPUT_DIR = RESULTS_DIR / "validation"
INSPECTION_DIR = OUTPUT_DIR / "inspection"

DATASETS = {
    "metric_selection": OUTPUT_DIR / "synthetic_pairs_metric_selection.pkl",
    "clustering_validation": OUTPUT_DIR / "synthetic_pairs_clustering_validation.pkl",
}


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def label_name(label: float) -> str:
    if label == 1.0:
        return "close"
    if label == 0.5:
        return "distant"
    if label == 0.0:
        return "unrelated"
    return str(label)


def summarize_numeric(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)

    if len(arr) == 0:
        return {}

    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def get_script_images(script: dict) -> list[np.ndarray]:
    """
    Try to recover image arrays from a synthetic script dictionary.

    This is intentionally flexible because the exact image key can differ
    depending on how the synthetic dataset was generated.
    """
    candidate_keys = [
        "images",
        "glyph_images",
        "tokens",
        "glyphs",
        "draws",
        "draw_images",
    ]

    for key in candidate_keys:
        if key in script:
            return script[key]

    raise KeyError(
        "No image-like field found in synthetic script. "
        f"Available keys: {list(script.keys())}"
    )


def normalize_image(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)

    if img.ndim == 3:
        img = img[..., 0]

    img = img.astype(float)

    if img.max() > 1.0:
        img = img / 255.0

    return img


def plot_glyph_grid(
    ax,
    images: list[np.ndarray],
    title: str,
    max_glyphs: int = 20,
) -> None:
    images = list(images)[:max_glyphs]

    if len(images) == 0:
        ax.set_title(title)
        ax.axis("off")
        return

    images = [normalize_image(img) for img in images]

    n = len(images)
    cols = min(10, n)
    rows = int(np.ceil(n / cols))

    h, w = images[0].shape[:2]

    canvas = np.ones((rows * h, cols * w), dtype=float)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)

        if img.shape[:2] != (h, w):
            raise ValueError(
                "All glyph images must have the same shape for plotting. "
                f"Expected {(h, w)}, got {img.shape[:2]}."
            )

        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img

    ax.imshow(canvas, cmap="gray")
    ax.set_title(title)
    ax.axis("off")


def plot_pair_example(
    pair: dict,
    scripts: dict,
    dataset_dir: Path,
    tier: str,
    max_glyphs: int = 20,
) -> None:
    left = scripts[pair["left_id"]]
    right = scripts[pair["right_id"]]

    left_images = get_script_images(left)
    right_images = get_script_images(right)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_glyph_grid(
        axes[0],
        left_images,
        title=(
            f"Left synthetic script\n"
            f"ancestor: {pair['left_ancestor_alphabet']}"
        ),
        max_glyphs=max_glyphs,
    )

    plot_glyph_grid(
        axes[1],
        right_images,
        title=(
            f"Right synthetic script\n"
            f"ancestor: {pair['right_ancestor_alphabet']}"
        ),
        max_glyphs=max_glyphs,
    )

    fig.suptitle(
        (
            f"{tier.capitalize()} synthetic pair "
            f"| label={pair['label']} "
            f"| pair_id={pair['pair_id']}"
        ),
        fontsize=13,
    )

    plt.tight_layout()
    plt.savefig(
        dataset_dir / f"{tier}_example_pair_with_ancestors.png",
        dpi=300,
    )
    plt.close()


def save_example_pair_plots(
    pairs: list[dict],
    scripts: dict,
    pairs_df: pd.DataFrame,
    dataset_dir: Path,
    max_glyphs: int = 20,
) -> None:
    for tier in ["close", "distant", "unrelated"]:
        tier_df = pairs_df[pairs_df["tier"] == tier]

        if len(tier_df) == 0:
            print(f"[warning] no pair found for tier: {tier}")
            continue

        example_pair_id = tier_df.iloc[0]["pair_id"]

        example_pair = next(
            pair for pair in pairs if pair["pair_id"] == example_pair_id
        )

        plot_pair_example(
            pair=example_pair,
            scripts=scripts,
            dataset_dir=dataset_dir,
            tier=tier,
            max_glyphs=max_glyphs,
        )


# =============================================================================
# Dataset inspection
# =============================================================================

def inspect_dataset(name: str, path: Path) -> dict:
    print("\n" + "=" * 80)
    print(f"[inspect] {name}")
    print(f"[load] {path}")

    dataset = load_glyph_dictionary(path)

    config = dataset["config"]
    pairs = dataset["pairs"]
    scripts = dataset["synthetic_scripts"]

    pair_rows = []
    script_rows = []

    # -------------------------------------------------------------------------
    # Script-level table
    # -------------------------------------------------------------------------
    for script_id, script in scripts.items():
        source_keys = script["source_keys"]
        unique_source_keys = set(source_keys)

        script_rows.append(
            {
                "dataset": name,
                "script_id": script_id,
                "ancestor_alphabet": script["ancestor_alphabet"],
                "draw_count": script["draw_count"],
                "n_unique_source_glyphs": len(unique_source_keys),
                "mean_tokens_per_source_glyph": (
                    script["draw_count"] / len(unique_source_keys)
                    if unique_source_keys
                    else np.nan
                ),
            }
        )

    scripts_df = pd.DataFrame(script_rows)

    # -------------------------------------------------------------------------
    # Pair-level table
    # -------------------------------------------------------------------------
    for pair in pairs:
        left = scripts[pair["left_id"]]
        right = scripts[pair["right_id"]]

        left_sources = set(left["source_keys"])
        right_sources = set(right["source_keys"])

        inter = len(left_sources & right_sources)
        union = len(left_sources | right_sources)

        pair_rows.append(
            {
                "dataset": name,
                "pair_id": pair["pair_id"],
                "label": pair["label"],
                "tier": label_name(float(pair["label"])),
                "split": pair["split"],
                "left_id": pair["left_id"],
                "right_id": pair["right_id"],
                "left_ancestor": pair["left_ancestor_alphabet"],
                "right_ancestor": pair["right_ancestor_alphabet"],
                "same_ancestor": (
                    pair["left_ancestor_alphabet"]
                    == pair["right_ancestor_alphabet"]
                ),
                "left_draw_count": pair["left_draw_count"],
                "right_draw_count": pair["right_draw_count"],
                "mean_draw_count": 0.5
                * (pair["left_draw_count"] + pair["right_draw_count"]),
                "left_unique_source_glyphs": len(left_sources),
                "right_unique_source_glyphs": len(right_sources),
                "shared_source_glyphs": inter,
                "source_glyph_union": union,
                "source_jaccard": inter / union if union else np.nan,
            }
        )

    pairs_df = pd.DataFrame(pair_rows)

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------
    dataset_dir = INSPECTION_DIR / name
    ensure_dir(dataset_dir)

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------
    pairs_csv = dataset_dir / f"{name}_pairs_summary.csv"
    scripts_csv = dataset_dir / f"{name}_scripts_summary.csv"
    summary_json = dataset_dir / f"{name}_inspection_summary.json"

    pairs_df.to_csv(pairs_csv, index=False)
    scripts_df.to_csv(scripts_csv, index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    summary = {
        "dataset_name": name,
        "path": str(path),
        "n_pairs": int(len(pairs_df)),
        "n_synthetic_scripts": int(len(scripts_df)),
        "n_source_alphabets": int(config["n_source_alphabets"]),
        "source_alphabets": config["source_alphabets"],
        "config": config,
        "pair_counts_by_tier": pairs_df["tier"].value_counts().to_dict(),
        "pair_counts_by_split": pairs_df["split"].value_counts().to_dict(),
        "pair_counts_by_tier_and_split": (
            pairs_df.groupby(["tier", "split"])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        ),
        "script_draw_count_stats": summarize_numeric(
            scripts_df["draw_count"].tolist()
        ),
        "script_unique_source_glyph_stats": summarize_numeric(
            scripts_df["n_unique_source_glyphs"].tolist()
        ),
        "pair_mean_draw_count_stats": summarize_numeric(
            pairs_df["mean_draw_count"].tolist()
        ),
        "source_jaccard_by_tier": {
            tier: summarize_numeric(group["source_jaccard"].dropna().tolist())
            for tier, group in pairs_df.groupby("tier")
        },
        "top_ancestor_alphabets_scripts": (
            scripts_df["ancestor_alphabet"].value_counts().head(20).to_dict()
        ),
        "top_ancestor_alphabets_pairs_left": (
            pairs_df["left_ancestor"].value_counts().head(20).to_dict()
        ),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------------------
    # Diagnostic plots
    # -------------------------------------------------------------------------
    plt.figure(figsize=(7, 4))
    pairs_df["tier"].value_counts().reindex(
        ["close", "distant", "unrelated"]
    ).plot(kind="bar")
    plt.ylabel("Number of pairs")
    plt.title(f"{name}: pairs by tier")
    plt.tight_layout()
    plt.savefig(dataset_dir / f"{name}_pairs_by_tier.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    pairs_df["split"].value_counts().reindex(
        ["train", "valid", "test"]
    ).plot(kind="bar")
    plt.ylabel("Number of pairs")
    plt.title(f"{name}: pairs by split")
    plt.tight_layout()
    plt.savefig(dataset_dir / f"{name}_pairs_by_split.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    scripts_df["draw_count"].hist(bins=30)
    plt.xlabel("Glyph tokens per synthetic script")
    plt.ylabel("Number of scripts")
    plt.title(f"{name}: synthetic script sizes")
    plt.tight_layout()
    plt.savefig(dataset_dir / f"{name}_script_size_distribution.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    pairs_df.boxplot(column="source_jaccard", by="tier")
    plt.suptitle("")
    plt.title(f"{name}: source-glyph overlap by tier")
    plt.xlabel("Tier")
    plt.ylabel("Jaccard overlap of source glyphs")
    plt.tight_layout()
    plt.savefig(dataset_dir / f"{name}_source_jaccard_by_tier.png", dpi=300)
    plt.close()

    # -------------------------------------------------------------------------
    # Example visualizations: one pair per tier
    # -------------------------------------------------------------------------
    try:
        save_example_pair_plots(
            pairs=pairs,
            scripts=scripts,
            pairs_df=pairs_df,
            dataset_dir=dataset_dir,
            max_glyphs=20,
        )
    except Exception as exc:
        print(f"[warning] could not save example pair plots: {exc}")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print(f"[saved] {pairs_csv}")
    print(f"[saved] {scripts_csv}")
    print(f"[saved] {summary_json}")

    print("\n[summary]")
    print(f"pairs: {summary['n_pairs']}")
    print(f"synthetic scripts: {summary['n_synthetic_scripts']}")
    print(f"source alphabets: {summary['n_source_alphabets']}")
    print(f"tiers: {summary['pair_counts_by_tier']}")
    print(f"splits: {summary['pair_counts_by_split']}")
    print(f"script size stats: {summary['script_draw_count_stats']}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ensure_dir(INSPECTION_DIR)

    all_summaries = {}

    for name, path in DATASETS.items():
        if not path.exists():
            print(f"[warning] missing dataset: {path}")
            continue

        all_summaries[name] = inspect_dataset(name, path)

    combined_json = INSPECTION_DIR / "all_synthetic_datasets_inspection_summary.json"

    with open(combined_json, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"[saved] combined summary: {combined_json}")


if __name__ == "__main__":
    main()