# -*- coding: utf-8 -*-
"""
Sensitivity of the spatio-temporal correlation to the script metadata.

Locations and time ranges are approximate, so the metadata are perturbed and
the correlation between visual distance and a fixed geographic + temporal
predictor (min geographic + temporal L2, the components of the best mixed
model, without fitted parameters) is recomputed.

Perturbations (repeated trials):
  jitter_geo    Gaussian noise on every latitude/longitude (~1-2 deg)
  jitter_time   shift each begin/end century endpoint by +/- a few centuries
  drop_centers  keep a single random representative location per script

Writes to RESULTS_DIR / "spatiotemporal_sensitivity".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from glyphworld.core.paths import RESULTS_DIR, INFO_DIR
from glyphworld.analysis.correlation_location_similarity import (
    compute_geo_min, compute_time_L2, _parse_float_list_cell,
)

META = INFO_DIR / "scripts_location_with_latlon_and_timerange.xlsx"
DM = RESULTS_DIR / "worldwide_clustering" / "distance_matrix.csv"
OUTPUT_DIR = RESULTS_DIR / "spatiotemporal_sensitivity"

N_TRIALS = 30
GEO_SIGMA_DEG = 1.5     # ~150 km
TIME_JITTER = 2         # +/- 2 centuries


def _norm(s: str) -> str:
    return s.replace("`", "_")


def _serialize(arr) -> str:
    return "; ".join(f"{v}" for v in np.asarray(arr).ravel())


def _load():
    dm = pd.read_csv(DM, index_col=0)
    meta = pd.read_excel(META)
    meta_names = set(meta["Script"])
    idx = {n: i for i, n in enumerate(dm.index)}
    D = (dm.values + dm.values.T) / 2.0

    keep = []
    for n in dm.index:
        if n in meta_names:
            keep.append((n, n))
        elif _norm(n) in meta_names:
            keep.append((n, _norm(n)))
    pairs, vis = [], []
    for a in range(len(keep)):
        for b in range(a + 1, len(keep)):
            pairs.append((keep[a][1], keep[b][1]))
            vis.append(D[idx[keep[a][0]], idx[keep[b][0]]])
    print(f"[data] {len(keep)}/{len(dm.index)} scripts matched metadata, "
          f"{len(pairs)} pairs")
    return meta, pairs, np.asarray(vis, float)


def _combined_rho(df, pairs, vis) -> float:
    g = np.asarray(compute_geo_min(df, pairs), float)
    t = np.asarray(compute_time_L2(df, pairs), float)
    m = np.isfinite(g) & np.isfinite(t) & np.isfinite(vis)

    def nz(x):
        x = x[m]
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    combo = nz(g) + nz(t)
    return float(spearmanr(combo, vis[m])[0])


def _perturb_geo(df, rng):
    d = df.copy()
    la, lo = [], []
    for a, o in zip(df["Latitudes"], df["Longitudes"]):
        aa = _parse_float_list_cell(a) + rng.normal(0, GEO_SIGMA_DEG, np.size(_parse_float_list_cell(a)))
        oo = _parse_float_list_cell(o) + rng.normal(0, GEO_SIGMA_DEG, np.size(_parse_float_list_cell(o)))
        la.append(_serialize(aa)); lo.append(_serialize(oo))
    d["Latitudes"] = la; d["Longitudes"] = lo
    return d


def _perturb_time(df, rng):
    d = df.copy()
    b = df["begin"].to_numpy() + rng.integers(-TIME_JITTER, TIME_JITTER + 1, len(df))
    e = df["end"].to_numpy() + rng.integers(-TIME_JITTER, TIME_JITTER + 1, len(df))
    lo, hi = np.minimum(b, e), np.maximum(b, e)
    hi = np.where(hi == lo, hi + 1, hi)
    d["begin"] = lo; d["end"] = hi
    return d


def _drop_centers(df, rng):
    d = df.copy()
    la, lo = [], []
    for a, o in zip(df["Latitudes"], df["Longitudes"]):
        aa = _parse_float_list_cell(a); oo = _parse_float_list_cell(o)
        if aa.size > 1:
            i = int(rng.integers(aa.size)); aa = aa[i:i + 1]; oo = oo[i:i + 1]
        la.append(_serialize(aa)); lo.append(_serialize(oo))
    d["Latitudes"] = la; d["Longitudes"] = lo
    return d


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    meta, pairs, vis = _load()

    base = _combined_rho(meta, pairs, vis)
    print(f"[baseline] combined Spearman rho = {base:.3f}")

    perturbations = {
        "jitter_geo": _perturb_geo,
        "jitter_time": _perturb_time,
        "drop_centers": _drop_centers,
    }
    rows = [{"perturbation": "baseline", "rho_mean": round(base, 3), "rho_std": 0.0,
             "rho_min": round(base, 3), "rho_max": round(base, 3)}]
    for name, fn in perturbations.items():
        rng = np.random.default_rng(0)
        rhos = [_combined_rho(fn(meta, rng), pairs, vis) for _ in range(N_TRIALS)]
        rhos = np.asarray(rhos)
        rows.append({"perturbation": name, "rho_mean": round(rhos.mean(), 3),
                     "rho_std": round(rhos.std(), 3), "rho_min": round(rhos.min(), 3),
                     "rho_max": round(rhos.max(), 3)})
        print(f"[{name:12s}] rho = {rhos.mean():.3f} +/- {rhos.std():.3f}  "
              f"[{rhos.min():.3f}, {rhos.max():.3f}]  (n={N_TRIALS})")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "spatiotemporal_sensitivity.csv", index=False)
    drop = base - df.loc[df.perturbation != "baseline", "rho_mean"].min()
    print(f"\nMax mean drop from baseline across perturbations: {drop:.3f}")
    print(f"[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
