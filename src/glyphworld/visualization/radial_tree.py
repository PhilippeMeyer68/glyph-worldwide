# -*- coding: utf-8 -*-
"""
Create circular dendrogram.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from typing import Dict, Iterable, Mapping, Sequence

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
COLORMAPS = [
    "nipy_spectral",
    "terrain",
    "gist_rainbow",
    "CMRmap",
    "coolwarm",
    "gnuplot",
    "gist_stern",
    "brg",
    "rainbow",
]

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _unique_preserve_order(seq: Iterable):
    """Return unique elements in first-appearance order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _polar_to_cart(r: np.ndarray, theta: np.ndarray):
    """Vectorized polar→Cartesian."""
    return np.cos(theta) * r, np.sin(theta) * r


def _unwrap_pair(a1: float, a2: float) -> tuple[float, float]:
    """Return a1, a2 unwrapped to be contiguous (no 2π jump between them)."""
    t = np.array([a1, a2], dtype=float)
    t = np.unwrap(t)
    return float(t[0]), float(t[1])


def _arc_thetas(a1: float, a2: float, n: int = 256) -> np.ndarray:
    """Return the *shortest* modular arc from a1 to a2 (robust to wrap-around).

    We compute the signed minimal difference on the circle and interpolate
    from a1 to a1 + d. This guarantees an arc with angular span <= π.
    """
    d = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi  # in (-π, π]
    return np.linspace(a1, a1 + d, n)


def _arc_through(a1: float, a2: float, a_through: float, n: int = 256) -> np.ndarray:
    """
    Return the circular arc from a1 to a2 (CW or CCW) that *passes through* a_through.
    We build short/long candidates and pick the one whose unwrapped interval contains a_through.
    """
    two_pi = 2 * np.pi
    start = float(a1)

    # Short arc (|d| <= π)
    d_short = (a2 - start + np.pi) % (two_pi) - np.pi
    end_short = start + d_short

    # Long arc = complementary direction (|d| > π)
    d_long = d_short - two_pi if d_short > 0 else d_short + two_pi
    end_long = start + d_long

    def contains(end_val: float) -> bool:
        # Place a_through near the interval center, then test numeric inclusion
        mid = 0.5 * (start + end_val)
        k = round((mid - a_through) / two_pi)
        t = a_through + k * two_pi
        lo, hi = (start, end_val) if start <= end_val else (end_val, start)
        return lo - 1e-12 <= t <= hi + 1e-12

    end = end_short if contains(end_short) else end_long
    return np.linspace(start, end, n)


# -----------------------------------------------------------------------------
# Public helpers
# -----------------------------------------------------------------------------

def compute_label_positions(labels: Sequence[str], max_x: float, offset: float):
    """Compute (x, y, angle_deg) positions for leaf labels.

    Parameters
    ----------
    labels : list of str
        Leaf labels in plotting order.
    max_x : float
        Dendrogram's horizontal span in Matplotlib units (usually 10 * n_leaves).
    offset : float
        Radial offset beyond unit circle to place text (e.g., to clear rings).
    """
    n = len(labels)
    base = 5 + np.arange(n) * 10  # Matplotlib's dendrogram leaf centers
    angles = (base / max_x) * 2 * np.pi

    r = 1.05 + offset
    xs, ys = _polar_to_cart(r, angles)
    ang_deg = np.degrees(angles)
    return np.column_stack([xs, ys, ang_deg])


def draw_text_labels(ax: Axes, coords: np.ndarray, labels: Sequence[str], fontsize: int):
    """Draw tangentially-oriented labels at given polar coords."""
    for (x, y, angle), label in zip(coords, labels):
        angle = float(angle)
        rot = angle + 180 if 90 < angle < 270 else angle
        ha = "right" if 90 < angle < 270 else "left"
        ax.text(
            float(x),
            float(y),
            label,
            va="center",
            ha=ha,
            rotation=rot,
            rotation_mode="anchor",
            fontsize=fontsize,
        )


def draw_dendrogram_branches(
    ax: Axes,
    Z2: Mapping[str, object],
    max_x: float,
    *,
    radius: float = 1.0,
    linewidth: float = 2.0,
    palette: str | None = None,
):
    """Draw radial dendrogram branches (radial child segments + parent arc).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    Z2 : dict
        Output from `scipy.cluster.hierarchy.dendrogram(..., no_plot=True)`.
    max_x : float
        Horizontal extent used by `dendrogram` (typically 10 * n_leaves).
    radius : float, default=1.0
        Base radius for leaves. Internal nodes are drawn inwards from this.
    linewidth : float, default=2.0
        Line width for branches.
    palette : str | None, default=None
        Optional Matplotlib colormap name to recolor dendrogram colors.
    """
    max_y = float(np.max(Z2["dcoord"]))

    # Stable color order by first appearance (for recoloring)
    color_order = _unique_preserve_order(Z2["color_list"])  # type: ignore[index]

    if palette:
        base_cmap = cm.get_cmap(palette, len(color_order))
        palette_colors = base_cmap(np.linspace(0, 1, len(color_order)))

        def map_color(c):
            return palette_colors[color_order.index(c)]
    else:
        def map_color(c):  # passthrough: use Matplotlib's assigned colors
            return c

    style = dict(lw=linewidth, solid_capstyle="round", solid_joinstyle="round")

    for icoord, dcoord, color in zip(
        Z2["icoord"], Z2["dcoord"], Z2["color_list"]  # type: ignore[index]
    ):
        col = map_color(color)

        dcoord = np.asarray(dcoord, dtype=float)
        r = radius * (1.0 - dcoord / max_y)
        theta = 2.0 * np.pi * (np.asarray(icoord, dtype=float) / max_x)

        x, y = _polar_to_cart(r, theta)

        # child radial segments: left (0→1) and right (3→2)
        ax.plot(x[0:2], y[0:2], c=col, **style)
        ax.plot(x[3:1:-1], y[3:1:-1], c=col, **style)

        # parent arc at the *node* height (use the single parent level from dcoord[1])
        r_parent = radius * (1.0 - float(dcoord[1]) / max_y)
        if r_parent >= 1e-9:
            # angle of the upward radial (parent branch) – it's at the midpoint between inner x's
            theta_center = 2.0 * np.pi * (((icoord[1] + icoord[2]) / 2.0) / max_x)

            # choose CW/CCW so the arc *passes through* that upward radial
            th = _arc_through(theta[1], theta[2], theta_center, 256)
            ax.plot(*_polar_to_cart(r_parent, th), c=col, **style)


def draw_annotation_rings(
    ax: Axes,
    coords: np.ndarray,
    labels_dict: Mapping[str, Sequence],
    leaf_order: Sequence[int],
    *,
    is_class: bool = True,
    ring_radius: float = 1.05,
    ring_width: float = 0.1,
    ring_spacing: float = 0.05,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Draw one pie ring per label set around the dendrogram.

    Parameters
    ----------
    labels_dict : mapping
        {ring_name -> values_per_leaf}. If `is_class=True`, values are class labels
        and a categorical colormap is built. Otherwise, values are assumed to be
        precomputed colors per leaf.
    leaf_order : sequence of int
        Permutation produced by `dendrogram_dict["leaves"]`.
    Returns
    -------
    legend : dict
        {ring_name: {"labels": array, "colors": array}}
    """
    n = len(coords)
    arc_lengths = np.ones(n)  # uniform wedges

    legend: Dict[str, Dict[str, np.ndarray]] = {}

    for i, (ring_name, values) in enumerate(labels_dict.items()):
        r = ring_radius + i * (ring_width + ring_spacing)

        if is_class:
            classes_order = _unique_preserve_order(values)
            cmap = cm.get_cmap(COLORMAPS[i % len(COLORMAPS)], len(classes_order))
            color_seq = [cmap(classes_order.index(v)) for v in values]
            color_seq = np.asarray(color_seq)[leaf_order]
            legend[ring_name] = {
                "labels": np.array(classes_order),
                "colors": cmap(np.linspace(0, 1, len(classes_order))),
            }
        else:
            color_seq = np.asarray(values)[leaf_order]
            legend[ring_name] = {
                "labels": np.unique(values),
                "colors": np.unique(color_seq, axis=0),
            }

        startangle = float(coords[0, 2])  # degrees
        ax.pie(
            arc_lengths,
            colors=color_seq,
            radius=r,
            counterclock=True,
            startangle=startangle,
            wedgeprops=dict(width=ring_width),
        )

    return legend


# -----------------------------------------------------------------------------
# Main plotting entry point
# -----------------------------------------------------------------------------

def plot_radial_dendrogram(
    Z2: Mapping[str, object],
    *,
    fontsize: int = 8,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    palette: str | None = None,
    show_labels: bool = True,
    show: bool = True,
    sample_classes: Mapping[str, Sequence] | None = None,
    color_annotations: Mapping[str, Sequence] | None = None,
    custom_legend: Mapping[str, Mapping[str, Sequence]] | None = None,
    ring_radius: float = 1.05,
    ring_width: float = 0.1,
    ring_spacing: float = 0.05,
    save_path: str | None = None,   # ← renamed
):
    """Plot a clean radial dendrogram with optional outer annotation rings.

    Parameters
    ----------
    Z2 : dict
        Output from `scipy.cluster.hierarchy.dendrogram(..., no_plot=True)`.
    palette : str | None
        Optional Matplotlib colormap name to recolor branch colors coherently.
    sample_classes, color_annotations : mapping
        Per-leaf metadata for drawing rings. Only one of them should be used.
        - `sample_classes`: treat values as categorical labels and build a legend.
        - `color_annotations`: treat values as precomputed colors.
    custom_legend : mapping or None
        Override the automatically built legend.
    """
    created_fig = False
    if ax is None:
        if figsize is None:
            figsize = (15, 15)
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # How far to push labels (accounting for how many rings there will be)
    label_data = color_annotations or sample_classes
    n_rings = len(label_data) if label_data else 0
    offset = 0.0
    if n_rings:
        offset = (ring_radius - 1.05) + n_rings * ring_width + max(n_rings - 1, 0) * ring_spacing

    max_x = 10.0 * len(Z2["ivl"])  # Matplotlib's leaf spacing convention

    draw_dendrogram_branches(ax, Z2, max_x, radius=1.0, linewidth=2.0, palette=palette)
    coords = compute_label_positions(Z2["ivl"], max_x, offset)

    if show_labels:
        draw_text_labels(ax, coords, Z2["ivl"], fontsize)

    legend: Dict[str, Dict[str, np.ndarray]] = {}
    if color_annotations:
        legend = draw_annotation_rings(
            ax,
            coords,
            color_annotations,
            Z2["leaves"],  # type: ignore[index]
            is_class=False,
            ring_radius=ring_radius,
            ring_width=ring_width,
            ring_spacing=ring_spacing,
        )
    elif sample_classes:
        legend = draw_annotation_rings(
            ax,
            coords,
            sample_classes,
            Z2["leaves"],  # type: ignore[index]
            is_class=True,
            ring_radius=ring_radius,
            ring_width=ring_width,
            ring_spacing=ring_spacing,
        )

    # Legend (auto or custom)
    legend_src = custom_legend or legend
    if legend_src:
        for i, (label, items) in enumerate(legend_src.items()):
            handles = [Line2D([0], [0], color=c, lw=4) for c in items["colors"]]
            ax.legend(
                handles,
                items["labels"],
                bbox_to_anchor=(1.5 + 0.3 * i, 1.0),
                title=label,
                frameon=False,
            )

    # Clean look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    pad = 1.05 + n_rings * (ring_width + ring_spacing) + 0.1
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if show and created_fig:
        plt.show()
    if save_path and created_fig:
        ax.figure.savefig(save_path, bbox_inches="tight")

    return ax, legend_src


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Random demo
    np.random.seed(42)
    X = np.random.rand(100, 4)
    Z = linkage(X, method="ward")

    dend = dendrogram(
        Z,
        truncate_mode="lastp",
        p=150,
        show_leaf_counts=True,
        no_plot=True,
    )

    plot_radial_dendrogram(
        dend,
        fontsize=12,
        show=True,
        show_labels=True,
        palette="viridis",
    )
