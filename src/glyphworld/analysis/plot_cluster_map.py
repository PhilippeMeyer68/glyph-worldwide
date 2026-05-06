# -*- coding: utf-8 -*-
"""
Plot script cities on a world map with country boundaries,
colored by visual-similarity cluster.

Reads:  DATA_DIR / "info" / "scripts_location_with_latlon_clusters.xlsx"
Saves:  RESULTS_DIR / "cluster_map.png"
        RESULTS_DIR / "cluster_map_script_medoids.png"
        RESULTS_DIR / "cluster_map_cluster_medoids.png"

Requirements: geopandas for country boundaries.
  pip install geopandas
Optional:  pip install adjustText   (for label placement; a built-in
           fallback is included if not installed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from glyphworld.core.paths import DATA_DIR, RESULTS_DIR

# =====================================================================
# Config
# =====================================================================
IN_PATH = DATA_DIR / "info" / "scripts_location_with_latlon_and_timerange_clusters.xlsx"
OUT_PATH = RESULTS_DIR / "cluster_map.png"
OUT_MEDOID_SCRIPTS = RESULTS_DIR / "cluster_map_script_medoids.png"
OUT_MEDOID_CLUSTERS = RESULTS_DIR / "cluster_map_cluster_medoids.png"

POINT_SIZE = 30
POINT_ALPHA = 0.8
FIG_WIDTH = 24
FIG_HEIGHT = 13


# =====================================================================
# Helpers
# =====================================================================
def parse_float_list(x):
    if pd.isna(x):
        return []
    parts = str(x).replace(",", ";").split(";")
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            pass
    return out


def build_colormap(n_clusters):
    colors_a = [plt.cm.tab20(i / 20) for i in range(20)]
    colors_b = [plt.cm.tab20b(i / 20) for i in range(20)]
    all_colors = colors_a + colors_b
    return {i: all_colors[i % len(all_colors)] for i in range(n_clusters)}


def draw_world_background(ax):
    try:
        import geopandas as gpd
        try:
            path = gpd.datasets.get_path("naturalearth_lowres")
            world = gpd.read_file(path)
        except Exception:
            world = gpd.read_file(
                "https://naciscdn.org/naturalearth/110m/cultural/"
                "ne_110m_admin_0_countries.zip"
            )
        world.plot(ax=ax, color="#f2f2f2", edgecolor="#bbbbbb", linewidth=0.4)
        return True
    except Exception as e:
        print(f"[MAP] geopandas not available ({e})")
        return False


# =====================================================================
# Label repulsion (built-in fallback if adjustText not installed)
# =====================================================================
def _repel_labels(ax, texts, x_points, y_points,
                  n_iter=500, force_text=0.5, force_point=0.8,
                  lim_pad=2.0):
    """
    Iteratively move text annotations so they don't overlap each other
    or sit on top of data points. Then draw thin connector lines.

    Uses display coordinates for overlap detection, data coordinates
    for final placement.
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    n = len(texts)
    if n == 0:
        return

    # Current label positions in data coords
    tx = np.array([t.get_position()[0] for t in texts], dtype=float)
    ty = np.array([t.get_position()[1] for t in texts], dtype=float)
    px = np.array(x_points, dtype=float)
    py = np.array(y_points, dtype=float)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for iteration in range(n_iter):
        # Get bounding boxes in display coords
        bboxes = []
        for t in texts:
            bb = t.get_window_extent(renderer=renderer)
            bboxes.append(bb)

        moved = False

        for i in range(n):
            dx, dy = 0.0, 0.0

            # Repel from other labels
            for j in range(n):
                if i == j:
                    continue
                bi, bj = bboxes[i], bboxes[j]
                # Check overlap
                ox = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                oy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if ox > 0 and oy > 0:
                    # Overlap — push apart
                    cx_i = (bi.x0 + bi.x1) / 2
                    cy_i = (bi.y0 + bi.y1) / 2
                    cx_j = (bj.x0 + bj.x1) / 2
                    cy_j = (bj.y0 + bj.y1) / 2
                    ddx = cx_i - cx_j
                    ddy = cy_i - cy_j
                    dist = max(np.sqrt(ddx**2 + ddy**2), 1e-6)
                    dx += force_text * ddx / dist * (ox + oy) * 0.01
                    dy += force_text * ddy / dist * (ox + oy) * 0.01

            # Repel from own data point (avoid sitting on top)
            ddx = tx[i] - px[i]
            ddy = ty[i] - py[i]
            dist = max(np.sqrt(ddx**2 + ddy**2), 1e-6)
            if dist < 2.0:
                dx += force_point * ddx / dist * 0.3
                dy += force_point * ddy / dist * 0.3

            if abs(dx) > 0.01 or abs(dy) > 0.01:
                tx[i] += dx
                ty[i] += dy
                # Clamp to axes limits
                tx[i] = np.clip(tx[i], xlim[0] + lim_pad, xlim[1] - lim_pad)
                ty[i] = np.clip(ty[i], ylim[0] + lim_pad, ylim[1] - lim_pad)
                texts[i].set_position((tx[i], ty[i]))
                moved = True

        if not moved:
            break

    # Draw connector lines
    for i in range(n):
        dist = np.sqrt((tx[i] - px[i])**2 + (ty[i] - py[i])**2)
        if dist > 1.5:
            ax.plot([px[i], tx[i]], [py[i], ty[i]],
                    color="grey", linewidth=0.3, alpha=0.5, zorder=4)


def add_labels_no_overlap(ax, names, x_points, y_points, fontsize=5.5):
    """
    Add text labels with overlap avoidance.
    Uses adjustText if installed, otherwise the built-in repeller.
    """
    texts = []
    for name, xp, yp in zip(names, x_points, y_points):
        t = ax.text(
            xp + 1.5, yp + 1.0, name,
            fontsize=fontsize, color="black", alpha=0.85,
            zorder=6, clip_on=True,
        )
        texts.append(t)

    try:
        from adjustText import adjust_text
        adjust_text(
            texts, x=list(x_points), y=list(y_points), ax=ax,
            arrowprops=dict(arrowstyle="-", color="grey", lw=0.3, alpha=0.5),
            force_text=(0.5, 0.8),
            force_points=(0.8, 1.0),
            expand=(1.2, 1.4),
            max_move=50,
        )
        print("[LABELS] Placed with adjustText")
    except ImportError:
        print("[LABELS] adjustText not found, using built-in repulsion")
        _repel_labels(ax, texts, x_points, y_points)

    return texts


# =====================================================================
# Medoid computation
# =====================================================================
def compute_script_medoids(pts):
    """For each script, pick the city closest to its geographic centroid."""
    rows = []
    for script, grp in pts.groupby("Script"):
        cluster = grp["Cluster"].iloc[0]
        clat, clon = grp["lat"].mean(), grp["lon"].mean()
        dists = np.sqrt((grp["lat"] - clat)**2 + (grp["lon"] - clon)**2)
        best = dists.idxmin()
        rows.append({
            "Script": script, "Cluster": cluster,
            "lat": grp.loc[best, "lat"], "lon": grp.loc[best, "lon"],
        })
    return pd.DataFrame(rows)


def compute_cluster_medoids(script_medoids):
    """For each cluster, pick the script medoid closest to the cluster centroid."""
    rows = []
    for cid, grp in script_medoids.groupby("Cluster"):
        clat, clon = grp["lat"].mean(), grp["lon"].mean()
        dists = np.sqrt((grp["lat"] - clat)**2 + (grp["lon"] - clon)**2)
        best = dists.idxmin()
        scripts_list = ", ".join(sorted(grp["Script"].tolist()))
        rows.append({
            "Cluster": cid,
            "lat": grp.loc[best, "lat"], "lon": grp.loc[best, "lon"],
            "Scripts": scripts_list,
            "RepScript": grp.loc[best, "Script"],
        })
    return pd.DataFrame(rows)


# =====================================================================
# Plot functions
# =====================================================================
def make_legend(ax, cluster_ids, cmap, cid_to_idx, cluster_scripts,
                marker_edge="white", marker_ew=0.3, ms=7):
    handles = []
    for cid in cluster_ids:
        color = cmap[cid_to_idx[cid]]
        scripts_str = cluster_scripts.get(cid, "")
        if len(scripts_str) > 65:
            scripts_str = scripts_str[:62] + "…"
        handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markeredgecolor=marker_edge, markeredgewidth=marker_ew,
                   markersize=ms, label=f"C{cid}: {scripts_str}")
        )
    ax.legend(
        handles=handles, loc="center left",
        bbox_to_anchor=(1.01, 0.5), fontsize=6.5,
        frameon=True, framealpha=0.95, ncol=1,
        title="Cluster", title_fontsize=9,
    )


def plot_all_cities(pts, cluster_ids, cmap, cid_to_idx):
    """Plot 1: all cities, no labels."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_facecolor("#dceefb")
    draw_world_background(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")

    for cid in cluster_ids:
        subset = pts[pts["Cluster"] == cid]
        color = cmap[cid_to_idx[cid]]
        ax.scatter(subset["lon"], subset["lat"],
                   s=POINT_SIZE, alpha=POINT_ALPHA, color=color,
                   edgecolors="white", linewidths=0.3, zorder=5)

    ax.set_title("Writing systems colored by visual-similarity cluster",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    cs = pts.groupby("Cluster")["Script"].apply(
        lambda x: ", ".join(sorted(x.unique()))).to_dict()
    make_legend(ax, cluster_ids, cmap, cid_to_idx, cs)

    fig.tight_layout()
    fig.savefig(str(OUT_PATH), dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.show()


def plot_script_medoids(script_medoids, cluster_ids, cmap, cid_to_idx):
    """Plot 2: one labelled point per script."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_facecolor("#dceefb")
    draw_world_background(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")

    for cid in cluster_ids:
        subset = script_medoids[script_medoids["Cluster"] == cid]
        color = cmap[cid_to_idx[cid]]
        ax.scatter(subset["lon"], subset["lat"],
                   s=60, alpha=0.85, color=color,
                   edgecolors="black", linewidths=0.5, zorder=5)

    # Labels with repulsion
    add_labels_no_overlap(
        ax,
        script_medoids["Script"].tolist(),
        script_medoids["lon"].values,
        script_medoids["lat"].values,
        fontsize=5.5,
    )

    ax.set_title("Script geographic medoids, colored by visual-similarity cluster",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    cs = script_medoids.groupby("Cluster")["Script"].apply(
        lambda x: ", ".join(sorted(x))).to_dict()
    make_legend(ax, cluster_ids, cmap, cid_to_idx, cs,
                marker_edge="black", marker_ew=0.5)

    fig.tight_layout()
    fig.savefig(str(OUT_MEDOID_SCRIPTS), dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_MEDOID_SCRIPTS}")
    plt.show()


def plot_cluster_medoids(cluster_medoids, cluster_ids, cmap, cid_to_idx):
    """Plot 3: one labelled point per cluster."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_facecolor("#dceefb")
    draw_world_background(ax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")

    for _, row in cluster_medoids.iterrows():
        cid = row["Cluster"]
        color = cmap[cid_to_idx[cid]]
        ax.scatter(row["lon"], row["lat"],
                   s=140, alpha=0.9, color=color,
                   edgecolors="black", linewidths=0.8, zorder=5)

    # Labels: "C{id}: RepScript"
    labels = [f"C{int(r['Cluster'])}: {r['RepScript']}"
              for _, r in cluster_medoids.iterrows()]
    add_labels_no_overlap(
        ax, labels,
        cluster_medoids["lon"].values,
        cluster_medoids["lat"].values,
        fontsize=7,
    )

    ax.set_title("Cluster geographic medoids (one point per cluster)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)

    cs = {int(r["Cluster"]): r["Scripts"] for _, r in cluster_medoids.iterrows()}
    make_legend(ax, cluster_ids, cmap, cid_to_idx, cs,
                marker_edge="black", marker_ew=0.8, ms=9)

    fig.tight_layout()
    fig.savefig(str(OUT_MEDOID_CLUSTERS), dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_MEDOID_CLUSTERS}")
    plt.show()


# =====================================================================
# Main
# =====================================================================
def main():
    df = pd.read_excel(IN_PATH)
    print(f"Loaded {len(df)} scripts")

    # Explode scripts into city-level rows
    rows = []
    for _, r in df.iterrows():
        lats = parse_float_list(r["Latitudes"])
        lons = parse_float_list(r["Longitudes"])
        cluster = int(r["Cluster"])
        script = str(r["Script"])
        for lat, lon in zip(lats, lons):
            rows.append({"Script": script, "Cluster": cluster,
                         "lat": lat, "lon": lon})

    pts = pd.DataFrame(rows)
    print(f"Total city points: {len(pts)}")

    cluster_ids = sorted(pts["Cluster"].unique())
    n_clusters = len(cluster_ids)
    cmap = build_colormap(n_clusters)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}

    # --- Plot 1: all cities ---
    plot_all_cities(pts, cluster_ids, cmap, cid_to_idx)

    # --- Plot 2: script medoids ---
    script_meds = compute_script_medoids(pts)
    print(f"\nScript medoids: {len(script_meds)} points")
    plot_script_medoids(script_meds, cluster_ids, cmap, cid_to_idx)

    # --- Plot 3: cluster medoids ---
    cluster_meds = compute_cluster_medoids(script_meds)
    print(f"\nCluster medoids: {len(cluster_meds)} points")
    plot_cluster_medoids(cluster_meds, cluster_ids, cmap, cid_to_idx)


if __name__ == "__main__":
    main()