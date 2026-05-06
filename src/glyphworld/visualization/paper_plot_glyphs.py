# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Paper figure: real glyphs + synthetic script pairs.

Left panel:
    Six randomly sampled real scripts from alphabets_unicode.pkl,
    one title per script row.

Right panel:
    One close, one distant, and one unrelated synthetic pair.
    Each pair is shown with two rows.

Each alphabet row is framed by a black box.

Outputs:
    RESULTS_DIR / "validation" / "real_and_synthetic_glyph_examples.png"
    RESULTS_DIR / "validation" / "real_and_synthetic_glyph_examples.pdf"
"""

import argparse
import pickle
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from glyphworld.core.paths import ALPHABETS_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary


DEFAULT_REAL_PATH = ALPHABETS_DIR / "alphabets_unicode.pkl"
DEFAULT_SYNTHETIC_PATH = RESULTS_DIR / "validation" / "synthetic_pairs_metric_selection.pkl"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "plot_glyphs" / "real_and_synthetic_glyph_examples.png"

LABEL_TO_NAME = {
    1.0: "Close",
    0.5: "Distant",
    0.0: "Unrelated",
}


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_name(name: str) -> str:
    return name.replace("Ge_ez", "Ge`ez")


def sample_pair_by_label(pairs: list[dict], label: float, rng: random.Random) -> dict:
    candidates = [p for p in pairs if float(p["label"]) == label]
    if not candidates:
        raise ValueError(f"No pair found with label={label}")
    return rng.choice(candidates)


def sample_glyphs(
    glyphs: list[np.ndarray],
    n: int,
    rng: random.Random,
) -> list[np.ndarray]:
    if len(glyphs) <= n:
        return glyphs
    idx = rng.sample(range(len(glyphs)), n)
    return [glyphs[i] for i in idx]


def get_real_script_glyphs(
    real_alphabets: dict[str, dict[str, np.ndarray]],
    script_name: str,
) -> list[np.ndarray]:
    return [np.asarray(img) for img in real_alphabets[script_name].values()]


def sample_real_scripts(
    real_alphabets: dict[str, dict[str, np.ndarray]],
    n: int,
    rng: random.Random,
    min_glyphs: int = 5,
) -> list[str]:
    candidates = [
        name for name, glyphs in real_alphabets.items()
        if len(glyphs) >= min_glyphs
    ]
    if len(candidates) < n:
        raise ValueError(f"Not enough real scripts with at least {min_glyphs} glyphs.")
    return rng.sample(sorted(candidates), n)


def plot_glyph_row(
    fig,
    gs,
    row: int,
    col_start: int,
    n_cols: int,
    glyphs: list[np.ndarray],
    max_glyphs: int,
) -> list:
    """Plot glyphs in a row. Returns list of axes created."""
    axes = []
    for j in range(n_cols):
        ax = fig.add_subplot(gs[row, col_start + j])
        ax.axis("off")
        if j < min(len(glyphs), max_glyphs):
            ax.imshow(glyphs[j], cmap="gray", vmin=0, vmax=255)
        axes.append(ax)
    return axes


def draw_box_around_axes(fig, axes_row, pad=3.0, linewidth=1.2):
    """
    Draw a box in figure pixel coordinates around a row of axes.
    Must be called AFTER fig.canvas.draw() so positions are finalised.
    """
    renderer = fig.canvas.get_renderer()

    # Collect display-coordinate bounding boxes of all axes in the row
    bboxes = []
    for ax in axes_row:
        bb = ax.get_tightbbox(renderer)
        if bb is not None:
            bboxes.append(bb)
    if not bboxes:
        return

    # Union of all bounding boxes (in display / pixel coords)
    x0 = min(bb.x0 for bb in bboxes) - pad
    y0 = min(bb.y0 for bb in bboxes) - pad
    x1 = max(bb.x1 for bb in bboxes) + pad
    y1 = max(bb.y1 for bb in bboxes) + pad

    # Convert to figure fraction coordinates
    inv = fig.transFigure.inverted()
    (fx0, fy0) = inv.transform((x0, y0))
    (fx1, fy1) = inv.transform((x1, y1))

    rect = Rectangle(
        (fx0, fy0),
        fx1 - fx0,
        fy1 - fy0,
        transform=fig.transFigure,
        fill=False,
        linewidth=linewidth,
        edgecolor="black",
        clip_on=False,
        zorder=10,
    )
    fig.patches.append(rect)


def plot_real_and_synthetic_examples(
    real_alphabets: dict[str, dict[str, np.ndarray]],
    synthetic_dataset: dict,
    output_path: Path,
    seed: int = 42,
    max_glyphs: int = 10,
    dpi: int = 300,
    show: bool = False,
) -> None:
    rng = random.Random(seed)

    pairs = synthetic_dataset["pairs"]
    synthetic_scripts = synthetic_dataset["synthetic_scripts"]

    labels = [1.0, 0.5, 0.0]
    selected_pairs = [sample_pair_by_label(pairs, label, rng) for label in labels]
    selected_real_scripts = sample_real_scripts(
        real_alphabets,
        n=2 * len(labels),
        rng=rng,
    )

    n_blocks = len(labels)
    n_cols = max_glyphs

    # Grid layout per block: title_a, glyph_a, title_b, glyph_b
    rows_per_block = 4
    n_rows = n_blocks * rows_per_block
    total_cols = 2 * n_cols

    fig = plt.figure(figsize=(1.4 * n_cols, 1.6 * n_blocks * rows_per_block / 3))

    # Height ratios: for each block [title, glyphs, title, glyphs]
    height_ratios = sum(([0.4, 1.0, 0.4, 1.0] for _ in range(n_blocks)), [])

    gs = fig.add_gridspec(
        n_rows,
        total_cols,
        height_ratios=height_ratios,
        hspace=0.30,
        wspace=0.01,
    )

    # Panel headers ("Real scripts" / "Synthetic script pairs") will be
    # placed after fig.canvas.draw(), centred over actual column extents.

    # Collect axes for boxes:
    # - left_rows_to_box: individual row boxes (one per real script)
    # - right_pairs_to_box: paired boxes (two rows grouped per synthetic pair)
    left_rows_to_box = []
    right_pairs_to_box = []  # list of (axes_row_a, axes_row_b)

    for i, pair in enumerate(selected_pairs):
        base = rows_per_block * i
        title_row_a = base + 0
        glyph_row_a = base + 1
        title_row_b = base + 2
        glyph_row_b = base + 3

        # =====================================================
        # Left panel: two real scripts with proper title rows
        # =====================================================
        real_name_a = selected_real_scripts[2 * i]
        real_name_b = selected_real_scripts[2 * i + 1]

        real_glyphs_a = sample_glyphs(
            get_real_script_glyphs(real_alphabets, real_name_a),
            max_glyphs,
            rng,
        )
        real_glyphs_b = sample_glyphs(
            get_real_script_glyphs(real_alphabets, real_name_b),
            max_glyphs,
            rng,
        )

        # Title for real script A (left half of title row A)
        ax_title = fig.add_subplot(gs[title_row_a, 0:n_cols])
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5,
            clean_name(real_name_a),
            ha="center", va="center",
            fontsize=11, fontweight="normal",
            transform=ax_title.transAxes,
        )

        # Glyph row A (left)
        axes_a_left = plot_glyph_row(
            fig=fig, gs=gs, row=glyph_row_a, col_start=0,
            n_cols=n_cols, glyphs=real_glyphs_a, max_glyphs=max_glyphs,
        )
        left_rows_to_box.append(axes_a_left)

        # Title for real script B (left half of title row B)
        ax_title_b = fig.add_subplot(gs[title_row_b, 0:n_cols])
        ax_title_b.axis("off")
        ax_title_b.text(
            0.5, 0.5,
            clean_name(real_name_b),
            ha="center", va="center",
            fontsize=11, fontweight="normal",
            transform=ax_title_b.transAxes,
        )

        # Glyph row B (left)
        axes_b_left = plot_glyph_row(
            fig=fig, gs=gs, row=glyph_row_b, col_start=0,
            n_cols=n_cols, glyphs=real_glyphs_b, max_glyphs=max_glyphs,
        )
        left_rows_to_box.append(axes_b_left)

        # =====================================================
        # Right panel: synthetic pair
        # =====================================================
        label = float(pair["label"])
        label_name = LABEL_TO_NAME[label]

        left_script = synthetic_scripts[pair["left_id"]]
        right_script = synthetic_scripts[pair["right_id"]]

        left_glyphs = sample_glyphs(left_script["glyph_images"], max_glyphs, rng)
        right_glyphs = sample_glyphs(right_script["glyph_images"], max_glyphs, rng)

        same_ancestor = pair["left_ancestor_alphabet"] == pair["right_ancestor_alphabet"]
        ancestor_text = (
            f"ancestor: {clean_name(pair['left_ancestor_alphabet'])}"
            if same_ancestor
            else (
                f"ancestors: {clean_name(pair['left_ancestor_alphabet'])} "
                f"vs {clean_name(pair['right_ancestor_alphabet'])}"
            )
        )

        # Synthetic pair title (right half of title row A)
        ax_synth_title = fig.add_subplot(gs[title_row_a, n_cols:2 * n_cols])
        ax_synth_title.axis("off")
        ax_synth_title.text(
            0.5, 0.5,
            f"{label_name} pair ({ancestor_text})",
            ha="center", va="center",
            fontsize=11, fontweight="normal",
            transform=ax_synth_title.transAxes,
        )

        # Leave right half of title_row_b empty
        ax_empty = fig.add_subplot(gs[title_row_b, n_cols:2 * n_cols])
        ax_empty.axis("off")

        # Glyph row A (right) — synthetic left script
        axes_a_right = plot_glyph_row(
            fig=fig, gs=gs, row=glyph_row_a, col_start=n_cols,
            n_cols=n_cols, glyphs=left_glyphs, max_glyphs=max_glyphs,
        )

        # Glyph row B (right) — synthetic right script
        axes_b_right = plot_glyph_row(
            fig=fig, gs=gs, row=glyph_row_b, col_start=n_cols,
            n_cols=n_cols, glyphs=right_glyphs, max_glyphs=max_glyphs,
        )

        # Group both right rows as a pair for a single box
        right_pairs_to_box.append((axes_a_right, axes_b_right))

    # --- Draw boxes and panel headers AFTER layout is computed ---
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Left panel: individual box per row
    for axes_row in left_rows_to_box:
        draw_box_around_axes(fig, axes_row, pad=3.0, linewidth=1.2)

    # Right panel: one box spanning both rows per pair, with a thin separator
    inv = fig.transFigure.inverted()
    for axes_row_a, axes_row_b in right_pairs_to_box:
        all_axes = axes_row_a + axes_row_b
        draw_box_around_axes(fig, all_axes, pad=3.0, linewidth=1.2)

        # Draw a thin dashed horizontal separator between the two rows
        bboxes_a = [ax.get_tightbbox(renderer) for ax in axes_row_a if ax.get_tightbbox(renderer) is not None]
        bboxes_b = [ax.get_tightbbox(renderer) for ax in axes_row_b if ax.get_tightbbox(renderer) is not None]
        if bboxes_a and bboxes_b:
            # In display coords, y increases upward.
            # Row A is visually above row B, so row A has higher y.
            # We want the midpoint between the bottom of row A and top of row B.
            bottom_of_a = min(bb.y0 for bb in bboxes_a)   # lowest y in row A
            top_of_b = max(bb.y1 for bb in bboxes_b)       # highest y in row B
            y_mid = (bottom_of_a + top_of_b) / 2

            x_left = min(bb.x0 for bb in bboxes_a) - 3.0
            x_right = max(bb.x1 for bb in bboxes_a) + 3.0

            fx_left, fy_mid = inv.transform((x_left, y_mid))
            fx_right, _ = inv.transform((x_right, y_mid))

            line = plt.Line2D(
                [fx_left, fx_right], [fy_mid, fy_mid],
                transform=fig.transFigure,
                color="gray", linewidth=0.8, linestyle=(0, (4, 3)),
                clip_on=False, zorder=15,
            )
            fig.lines.append(line)

    # Compute centered x-positions for panel headers from actual axes positions
    # Left panel: use first row of left-side axes
    left_bboxes = [ax.get_tightbbox(renderer) for ax in left_rows_to_box[0] if ax.get_tightbbox(renderer) is not None]
    left_x0 = inv.transform((min(bb.x0 for bb in left_bboxes), 0))[0]
    left_x1 = inv.transform((max(bb.x1 for bb in left_bboxes), 0))[0]
    left_center_x = (left_x0 + left_x1) / 2

    # Right panel: use first pair's row A axes
    right_bboxes = [ax.get_tightbbox(renderer) for ax in right_pairs_to_box[0][0] if ax.get_tightbbox(renderer) is not None]
    right_x0 = inv.transform((min(bb.x0 for bb in right_bboxes), 0))[0]
    right_x1 = inv.transform((max(bb.x1 for bb in right_bboxes), 0))[0]
    right_center_x = (right_x0 + right_x1) / 2

    # y-position: just above the suptitle area
    suptitle_y = 0.98
    header_y = suptitle_y - 0.07

    fig.text(
        left_center_x, header_y,
        "Real scripts",
        ha="center", va="center",
        fontsize=14, fontweight="bold",
    )
    fig.text(
        right_center_x, header_y,
        "Synthetic scripts",
        ha="center", va="center",
        fontsize=14, fontweight="bold",
    )

    ensure_dir(output_path.parent)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved PNG to: {output_path}")
    print(f"Saved PDF to: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--real", type=Path, default=DEFAULT_REAL_PATH)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-glyphs", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    real_alphabets = load_glyph_dictionary(args.real)
    synthetic_dataset = load_pickle(args.synthetic)

    plot_real_and_synthetic_examples(
        real_alphabets=real_alphabets,
        synthetic_dataset=synthetic_dataset,
        output_path=args.output,
        seed=args.seed,
        max_glyphs=args.max_glyphs,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()