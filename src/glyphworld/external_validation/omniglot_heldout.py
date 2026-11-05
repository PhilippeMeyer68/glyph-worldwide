# -*- coding: utf-8 -*-
"""
Validation on held-out handwritten Omniglot alphabets.

The 10 Omniglot evaluation alphabets are never used to train the encoder.
Each handwritten alphabet is embedded with the frozen student encoder and
compared, with the sym_min_mean script distance, to the 125 font-rendered
Noto scripts: does it land nearest its own Noto counterpart?

The default draw uses the first drawer of each character; ``--draws N``
(default 5) repeats the retrieval with a random drawer per character.

Writes to RESULTS_DIR / "external_validation" / "omniglot_heldout".
"""
from __future__ import annotations

import pickle
import random

import numpy as np
import pandas as pd

from glyphworld.core.paths import ALPHABETS_DIR, DATA_DIR, INFO_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary, dist_dict_from_embeddings_cosine
from glyphworld.distances.distance_functions import similarity_between_languages
from glyphworld.synthetic_data.metric_validation import (
    load_metric_model,
    build_transform,
    compute_embeddings,
    ensure_uint8_bw,
)

# Held-out handwriting: Omniglot evaluation split, as prepared in the companion project.
HELDOUT_PKL = DATA_DIR / "external" / "omniglot" / "glyphts_evaluation.pkl"
NOTO_EMB = ALPHABETS_DIR / "alphabets_unicode_embeddings_css_kmedoids.pkl"
OUTPUT_DIR = RESULTS_DIR / "external_validation" / "omniglot_heldout"

# Held-out handwritten alphabet -> its counterpart among the 125 Noto scripts.
HOMONYM = {
    "Mongolian": "Mongolian",
    "Oriya": "Oriya",
    "Sanskrit": "Devanagari",              # Omniglot "Sanskrit" is Devanagari
    "Sylheti": "Syloti Nagri",
    "Syriac_(Estrangelo)": "Syriac",
    "Syriac_(Serto)": "Syriac",
    "Tagalog": "Tagalog",
    "Tibetan": "Tibetan",
    "Tifinagh": "Tifinagh",
    "Old_Church_Slavonic_(Cyrillic)": "Cyrillic",  # OCS ~ early Cyrillic
}


def _embed_heldout(model, tf, heldout, rng: random.Random | None) -> dict[str, np.ndarray]:
    """One instance per character: the first drawer (rng=None) or a random one."""
    Zh = {}
    for alph, chars in heldout.items():
        imgs = [ensure_uint8_bw(inst[0] if rng is None else inst[rng.randrange(len(inst))])
                for inst in chars.values()]
        Zh[alph] = compute_embeddings(
            model, imgs, tf, use_teacher=False, model_kind="student_teacher"
        )
    return Zh


def _counterpart_ranks(Zh, noto, noto_scripts) -> dict[str, int]:
    ranks = {}
    for h, Z in Zh.items():
        dists = {n: similarity_between_languages(dist_dict_from_embeddings_cosine(Z, noto[n]))
                 for n in noto_scripts}
        order = sorted(noto_scripts, key=lambda n: dists[n])
        ranks[h] = order.index(HOMONYM[h]) + 1
    return ranks


def run(n_draws: int = 5) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- frozen student encoder (same as the paper's Noto embeddings) ---
    model, ckpt = load_metric_model("student_teacher")
    tf = build_transform(image_size=105)
    print(f"[model] {ckpt.name}")

    # --- embed the held-out handwritten alphabets (one instance per character) ---
    heldout = pickle.load(open(HELDOUT_PKL, "rb"))
    Zh = _embed_heldout(model, tf, heldout, rng=None)
    for alph, Z in Zh.items():
        print(f"[heldout] {alph:<32} {len(Z):>3} glyphs")

    # --- Noto script embeddings (125 scripts; drop the ' Full' large-script duplicates) ---
    emb = load_glyph_dictionary(NOTO_EMB)
    noto = {a: np.stack(list(emb[a].values())).astype(np.float32)
            for a in emb if not a.endswith(" Full")}
    noto_scripts = sorted(noto)
    fams = pd.read_csv(INFO_DIR / "scripts_families.csv").set_index("script")
    print(f"[noto] {len(noto_scripts)} scripts")

    # --- cross-domain retrieval: handwritten script vs all Noto scripts ---
    rows = []
    for h, Z in Zh.items():
        dists = {n: similarity_between_languages(dist_dict_from_embeddings_cosine(Z, noto[n]))
                 for n in noto_scripts}
        order = sorted(noto_scripts, key=lambda n: dists[n])
        homo = HOMONYM[h]
        rank = order.index(homo) + 1
        top1 = order[0]
        homo_fam = fams.loc[homo, "macro_family"]
        top1_fam = fams.loc[top1, "macro_family"]
        rows.append({
            "handwritten": h,
            "noto_counterpart": homo,
            "counterpart_rank": rank,          # 1 = nearest of all 125
            "counterpart_dist": round(dists[homo], 4),
            "nearest_noto": top1,
            "nearest_dist": round(dists[top1], 4),
            "nearest_same_family": bool(top1_fam == homo_fam),
            "counterpart_in_top5": rank <= 5,
            "top5": "; ".join(order[:5]),
        })

    df = pd.DataFrame(rows).sort_values("counterpart_rank")
    df.to_csv(OUTPUT_DIR / "omniglot_heldout_retrieval.csv", index=False)

    n = len(df)
    top1_id = (df["counterpart_rank"] == 1).mean()
    top5_id = df["counterpart_in_top5"].mean()
    mrr = (1.0 / df["counterpart_rank"]).mean()
    fam_hit = df["nearest_same_family"].mean()

    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print("\n=== Cross-domain retrieval: held-out handwriting -> Noto ===")
        print(df[["handwritten", "noto_counterpart", "counterpart_rank",
                  "nearest_noto", "nearest_same_family"]].to_string(index=False))
        print(f"\nIdentity  top-1 = {top1_id:.2f}   top-5 = {top5_id:.2f}   MRR = {mrr:.3f}")
        print(f"Family    nearest-Noto in same macro-family = {fam_hit:.2f}   (n={n})")

    # --- random draws: a different drawer for each character ---
    draw_rows = []
    for seed in range(n_draws):
        ranks = _counterpart_ranks(_embed_heldout(model, tf, heldout, random.Random(seed)),
                                   noto, noto_scripts)
        draw_rows += [{"draw": seed, "handwritten": h, "noto_counterpart": HOMONYM[h],
                       "counterpart_rank": r} for h, r in ranks.items()]
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(OUTPUT_DIR / "omniglot_heldout_retrieval_draws.csv", index=False)
    print(f"\n=== Counterpart rank over {n_draws} random draws (drawer per character) ===")
    print(draws.pivot(index="handwritten", columns="draw", values="counterpart_rank").to_string())
    print(f"rank 1 in {int((draws['counterpart_rank'] == 1).sum())}/{len(draws)} "
          f"(alphabet, draw) pairs; max rank {int(draws['counterpart_rank'].max())}")
    print(f"\n[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=5)
    run(parser.parse_args().draws)
