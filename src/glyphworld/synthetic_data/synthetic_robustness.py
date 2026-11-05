# -*- coding: utf-8 -*-
"""
Robustness of the metric and clustering selection to the synthetic generator.

The selection benchmarks use affine distortions; here they are regenerated
under other regimes to check whether sym_min_mean and Ward remain preferred:

  baseline           original affine construction
  strong_affine      much larger affine distortions on every tier
  low_overlap        smaller shared-core overlap (harder to grade)
  more_allographic   more distorted duplicates per source glyph
  noise_glyphs       spurious foreign glyphs injected into every script
  elastic            non-affine elastic warping after the affine step
  stroke_reduction   ink erosion after the affine step (thinner strokes)
  stroke_thickening  ink dilation after the affine step (heavier strokes)

For each regime, the 10 script metrics are ranked by close-vs-distant AUC and
the five linkages by ARI against the ancestor labels. Each regime is repeated
over several generator seeds (``--seeds``, default 0-4).

Writes to RESULTS_DIR / "validation" / "synthetic_robustness":
    - synthetic_robustness.csv          one row per (regime, seed)
    - synthetic_robustness_summary.csv  how often each selection survives
"""
from __future__ import annotations

import random
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import glyphworld.synthetic_data.generate_synthetic_dataset as G
from glyphworld.core.paths import ALPHABETS_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary
from glyphworld.synthetic_data.metric_validation import (
    load_metric_model, build_transform, compute_embeddings, ensure_uint8_bw,
    aggregate_distance_metric, summarize_one_metric_ordinal, METRIC_SETS,
)

OUTPUT_DIR = RESULTS_DIR / "validation" / "synthetic_robustness"
METRICS = list(METRIC_SETS["light"])
LINKAGES = ["ward", "complete", "average", "weighted", "single"]
N_PAIRS = 100         # per tier (close/distant/negative), matching the main benchmark
N_ALPHABETS = 55      # source alphabets per regime
CLUSTER_CAP = 220     # subsample scripts for the O(n^2) clustering matrix

_ORIG_DISTORT = G.distort_glyph
_model = _tf = _alphabets = None


def _elastic(img, alpha=10.0, sigma=4.0):
    h, w = img.shape
    dx = gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    dy = gaussian_filter(np.random.rand(h, w) * 2 - 1, sigma) * alpha
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    warped = map_coordinates(img, [np.clip(yy + dy, 0, h - 1), np.clip(xx + dx, 0, w - 1)],
                             order=0, mode="constant", cval=255)
    return warped.astype(np.uint8)


def _reset_globals():
    G.distort_glyph = _ORIG_DISTORT
    G.CLOSE_CORE_OVERLAP_RANGE = (0.6, 0.9); G.DISTANT_CORE_OVERLAP_RANGE = (0.3, 0.5)
    G.CLOSE_ROTATION_RANGE = (-10, 10); G.CLOSE_SHEAR_RANGE = (-0.3, 0.3)
    G.CLOSE_ZOOM_RANGE = (0.8, 1.2); G.CLOSE_SHIFT_RANGE = (-2, 2)
    G.DISTANT_ROTATION_RANGE = (-30, 30); G.DISTANT_SHEAR_RANGE = (-0.6, 0.6)
    G.DISTANT_ZOOM_RANGE = (0.6, 1.4); G.DISTANT_SHIFT_RANGE = (-5, 5)


def _apply_regime(regime: str, tokens: int) -> int:
    """Mutate generator globals for the regime; return max_tokens_per_type."""
    _reset_globals()
    if regime == "strong_affine":
        G.CLOSE_ROTATION_RANGE = (-30, 30); G.CLOSE_SHEAR_RANGE = (-0.6, 0.6)
        G.CLOSE_ZOOM_RANGE = (0.6, 1.4); G.CLOSE_SHIFT_RANGE = (-5, 5)
        G.DISTANT_ROTATION_RANGE = (-60, 60); G.DISTANT_SHEAR_RANGE = (-1.0, 1.0)
        G.DISTANT_ZOOM_RANGE = (0.5, 1.6); G.DISTANT_SHIFT_RANGE = (-9, 9)
    elif regime == "low_overlap":
        G.CLOSE_CORE_OVERLAP_RANGE = (0.35, 0.6); G.DISTANT_CORE_OVERLAP_RANGE = (0.1, 0.3)
    elif regime == "more_allographic":
        tokens = 6
    elif regime == "elastic":
        def wrapped(img, rotation_range, shear_range, zoom_range, shift_range):
            return _elastic(_ORIG_DISTORT(img, rotation_range, shear_range, zoom_range, shift_range))
        G.distort_glyph = wrapped
    elif regime in ("stroke_reduction", "stroke_thickening"):
        # non-affine morphology: erode the ink (thinner strokes) or dilate it
        # (thicker strokes) -- the "stroke reduction" the meta-review names.
        op = cv2.erode if regime == "stroke_reduction" else cv2.dilate
        def wrapped(img, rotation_range, shear_range, zoom_range, shift_range, _op=op):
            out = _ORIG_DISTORT(img, rotation_range, shear_range, zoom_range, shift_range)
            inv = 255 - out                                   # ink -> foreground
            k = np.ones((3, 3), np.uint8)
            return (255 - _op(inv, k)).astype(np.uint8)
        G.distort_glyph = wrapped
    return tokens


def _cfg(name, seed, tokens):
    return {"dataset_name": name, "seed": seed,
            "n_close_pairs": N_PAIRS, "n_distant_pairs": N_PAIRS, "n_negative_pairs": N_PAIRS,
            "min_glyph_draws": 5, "max_glyph_draws": 80, "max_tokens_per_type": tokens}


def _inject_noise(store, rng):
    """Append ~25% spurious glyphs from a different alphabet to each script."""
    names = sorted(_alphabets.keys())
    for sid, rec in store.items():
        k = max(1, int(round(0.25 * len(rec["glyph_images"]))))
        other = rng.choice([n for n in names if n != rec["ancestor_alphabet"]])
        keys = list(_alphabets[other].keys())
        for _ in range(k):
            g = _alphabets[other][rng.choice(keys)]
            rec["glyph_images"].append(G.normalize_glyph(np.asarray(g, np.uint8)))


def _embed(imgs):
    return compute_embeddings(_model, [ensure_uint8_bw(g) for g in imgs], _tf,
                              use_teacher=False, model_kind="student_teacher")


def _run_regime(regime: str, seed: int = 0) -> dict:
    tokens = _apply_regime(regime, tokens=3)
    # Fixed seed: identical source alphabets across regimes and seeds, so
    # differences are due to the regime (or the generator seed) alone.
    rng = random.Random(12345)
    all = [a for a in _alphabets if not a.endswith(" Full")]
    src = {a: _alphabets[a] for a in rng.sample(all, N_ALPHABETS)}

    # The glyph distortions draw from the global `random` / `np.random`
    # generators (affine and elastic steps): seed them so that each
    # (regime, seed) run is reproducible.
    G.set_seed(1000 + seed)
    cfg = _cfg(regime, seed=1000 + seed, tokens=tokens)
    close, sc = G.build_close_pairs(src, cfg, 0)
    distant, sd = G.build_distant_pairs(src, cfg, 2 * N_PAIRS)
    neg, sn = G.build_negative_pairs(src, cfg, 4 * N_PAIRS)
    store = {**sc, **sd, **sn}
    pairs = close + distant + neg
    if regime == "noise_glyphs":
        _inject_noise(store, rng)

    Z = {sid: _embed(store[sid]["glyph_images"]) for sid in store}

    # ---- metric ranking (Spearman vs ordinal label) ----
    labels = np.array([p.label for p in pairs])
    mrows = []
    for m in METRICS:
        d = np.array([aggregate_distance_metric(Z[p.left_id], Z[p.right_id], m) for p in pairs])
        s = summarize_one_metric_ordinal(labels, d)
        mrows.append((m, s["spearman_rho"], s["auc_close_vs_distant"]))
    # Rank by the PRE-REGISTERED selection criterion (Appendix, metric selection):
    # close-vs-distant AUC -- the graded-relatedness sub-task. Raw Spearman is
    # inflated by the easy "unrelated" tier and is NOT the criterion used to pick
    # the metric (e.g. min_all scores high on Spearman but collapses close/distant).
    mdf = pd.DataFrame(mrows, columns=["metric", "rho", "auc_c_d"]).sort_values("auc_c_d", ascending=False)
    mdf["rank"] = range(1, len(mdf) + 1)
    sym_rank = int(mdf.loc[mdf.metric == "sym_min_mean", "rank"].iloc[0])
    sym_cd = float(mdf.loc[mdf.metric == "sym_min_mean", "auc_c_d"].iloc[0])

    # ---- clustering ranking (ARI vs ancestor) ----
    sids = list(store)
    if len(sids) > CLUSTER_CAP:
        sids = random.Random(0).sample(sids, CLUSTER_CAP)
    y = pd.factorize([store[s]["ancestor_alphabet"] for s in sids])[0]
    k = len(set(y))
    n = len(sids)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = aggregate_distance_metric(Z[sids[i]], Z[sids[j]], "sym_min_mean")
    cond = squareform(M, checks=False)
    crows = []
    for lk in LINKAGES:
        pred = fcluster(linkage(cond, method=lk), k, criterion="maxclust")
        crows.append((lk, adjusted_rand_score(y, pred), normalized_mutual_info_score(y, pred)))
    cdf = pd.DataFrame(crows, columns=["linkage", "ari", "nmi"]).sort_values("ari", ascending=False)
    cdf["rank"] = range(1, len(cdf) + 1)
    ward_rank = int(cdf.loc[cdf.linkage == "ward", "rank"].iloc[0])

    print(f"\n### {regime} (seed {seed}) ###")
    print("  metrics by close-vs-distant AUC (top-3):",
          ", ".join(f"{r.metric}={r.auc_c_d:.3f}" for r in mdf.head(3).itertuples()))
    print(f"  -> sym_min_mean rank = {sym_rank}/{len(METRICS)}  (c/d AUC={sym_cd:.3f})")
    print("  linkage (top-3):", ", ".join(f"{r.linkage}={r.ari:.3f}" for r in cdf.head(3).itertuples()))
    print(f"  -> Ward rank = {ward_rank}/{len(LINKAGES)}  (ARI={cdf.loc[cdf.linkage=='ward','ari'].iloc[0]:.3f})")
    return {
        "regime": regime, "seed": seed,
        "best_metric_cd": mdf.iloc[0].metric, "sym_min_mean_rank": sym_rank,
        "sym_min_mean_auc_cd": round(sym_cd, 3),
        "best_linkage": cdf.iloc[0].linkage, "ward_rank": ward_rank,
        "ward_ari": round(float(cdf.loc[cdf.linkage == "ward", "ari"].iloc[0]), 3),
    }


REGIMES = ["baseline", "strong_affine", "low_overlap",
           "more_allographic", "noise_glyphs", "elastic",
           "stroke_reduction", "stroke_thickening"]
SEEDS = [0, 1, 2, 3, 4]


def summarize_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per regime: how often each selection survives across generator seeds."""
    g = df.groupby("regime", sort=False)
    out = pd.DataFrame({
        "n_seeds": g.size(),
        "sym_min_mean_best": g["sym_min_mean_rank"].apply(lambda r: int((r == 1).sum())),
        "sym_min_mean_top3": g["sym_min_mean_rank"].apply(lambda r: int((r <= 3).sum())),
        "sym_min_mean_auc_cd_mean": g["sym_min_mean_auc_cd"].mean().round(3),
        "sym_min_mean_auc_cd_std": g["sym_min_mean_auc_cd"].std().round(3),
        "ward_best": g["ward_rank"].apply(lambda r: int((r == 1).sum())),
        "ward_ari_mean": g["ward_ari"].mean().round(3),
        "ward_ari_std": g["ward_ari"].std().round(3),
        "ward_ari_min": g["ward_ari"].min(),
        "ward_ari_max": g["ward_ari"].max(),
    }).reset_index()
    return out


def run(seeds: list[int] = SEEDS) -> None:
    global _model, _tf, _alphabets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _model, _ = load_metric_model("student_teacher")
    _tf = build_transform(image_size=105)
    _alphabets = load_glyph_dictionary(ALPHABETS_DIR / "alphabets_unicode.pkl")

    rows = []
    for seed in seeds:
        for r in REGIMES:
            rows.append(_run_regime(r, seed=seed))
            # Write as we go: a full run takes ~2 h on CPU.
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "synthetic_robustness.csv", index=False)
    _reset_globals()

    df = pd.DataFrame(rows)
    summary = summarize_over_seeds(df)
    summary.to_csv(OUTPUT_DIR / "synthetic_robustness_summary.csv", index=False)

    n = len(df)
    print("\n=== Summary over seeds: does the selection survive? ===")
    print(summary.to_string(index=False))
    print(f"\nWard #1: {int((df['ward_rank'] == 1).sum())}/{n} runs | "
          f"sym_min_mean #1: {int((df['sym_min_mean_rank'] == 1).sum())}/{n} | "
          f"sym_min_mean top-3: {int((df['sym_min_mean_rank'] <= 3).sum())}/{n}")
    print(f"[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    run(parser.parse_args().seeds)
