# -*- coding: utf-8 -*-
"""
Plot and save all glyphs of one selected script.

Change SELECTED_SCRIPT below to choose the script.
"""

import math
import re

import matplotlib.pyplot as plt
import numpy as np

from glyphworld.core.paths import ALPHABETS_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary


# ============================================================
# User parameters
# ============================================================

# Any script name of the dataset, e.g. "Latin", "Greek", "Linear A Full".
SELECTED_SCRIPT = "Linear B Full"

N_COLS = 10
DPI = 300
SHOW_LABELS = True

ALPHABETS_PATH = ALPHABETS_DIR / "alphabets_unicode.pkl"
OUTPUT_DIR = RESULTS_DIR / "glyph_plots"


# ============================================================
# Functions
# ============================================================

def safe_filename(name):
    """Create a safe filename from a script name."""
    name = name.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "", name)


def plot_script_glyphs(glyphs, script_name, output_path, n_cols=10, dpi=300, show_labels=True):
    """Plot all glyphs of one script and save the figure."""

    items = list(glyphs.items())
    n_glyphs = len(items)

    if n_glyphs == 0:
        raise ValueError(f"No glyphs found for script: {script_name}")

    n_cols = min(n_cols, n_glyphs)
    n_rows = math.ceil(n_glyphs / n_cols)

    fig_w = n_cols * 1.2
    fig_h = n_rows * 1.35 if show_labels else n_rows * 1.1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (ax, (_, img)) in enumerate(zip(axes.ravel(), items)):
        img = np.asarray(img)

        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

        if show_labels:
            ax.set_title(str(idx), fontsize=6, pad=2)

    fig.suptitle(f"{script_name} — {n_glyphs} glyphs", fontsize=14)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVED] {output_path}")


# ============================================================
# Main script
# ============================================================

if __name__ == "__main__":
    print(f"[LOAD] {ALPHABETS_PATH}")
    alphabets = load_glyph_dictionary(ALPHABETS_PATH)

    if SELECTED_SCRIPT not in alphabets:
        print(f"\n[ERROR] Script not found: {SELECTED_SCRIPT}")
        print("\nAvailable scripts are:\n")

        for script_name in sorted(alphabets.keys()):
            print(f"  - {script_name}")

        raise KeyError(f"Script not found: {SELECTED_SCRIPT}")

    glyphs = alphabets[SELECTED_SCRIPT]

    output_name = f"{safe_filename(SELECTED_SCRIPT)}_glyphs.png"
    output_path = OUTPUT_DIR / output_name

    plot_script_glyphs(
        glyphs=glyphs,
        script_name=SELECTED_SCRIPT,
        output_path=output_path,
        n_cols=N_COLS,
        dpi=DPI,
        show_labels=SHOW_LABELS,
    )