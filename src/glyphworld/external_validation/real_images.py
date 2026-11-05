# -*- coding: utf-8 -*-
"""
Validation on real glyph images from outside the Noto fonts.

  * Egyptian hieroglyphs -- photographs from the Pyramid of Unas
    (Franken & van Gemert GlyphDataset), one image per Gardiner sign.
  * Oracle-bone characters -- Oracle-MNIST (archaic Chinese).
  * Devanagari -- handwritten characters (DHCD).
  * Kuzushiji -- cursive historical Japanese (Kuzushiji-MNIST).

Each image set is embedded with the frozen student encoder and compared, with
the sym_min_mean script distance, to the 125 Noto scripts: does it land on or
near its Noto counterpart?

The default draw uses the first images of each class; ``--draws N`` (default
5) repeats the retrieval with N random draws of other images of the same
classes and summarises how stable the counterpart rank is.

The datasets are public downloads, documented in ``data/external/README.md``.
Writes to RESULTS_DIR / "external_validation" / "real_images".
"""
from __future__ import annotations

import gzip
import random
import re
import zipfile
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from glyphworld.core.paths import ALPHABETS_DIR, DATA_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary, dist_dict_from_embeddings_cosine
from glyphworld.distances.distance_functions import similarity_between_languages
from glyphworld.synthetic_data.metric_validation import (
    load_metric_model, build_transform, compute_embeddings,
)
from glyphworld.data.build_alphabet_data import estimate_stroke_width, get_nonwhite_bounds

# Canonical glyph standardization targets (see build_alphabet_data.create_glyph):
# binary glyphs sized to a ~8 px stroke width, centred in a 105x105 white frame.
TARGET_STROKE = 8.0
FRAME = 105
MAXDIM = 101  # leave a 2 px margin so the glyph never touches the border

# External public datasets, stored under data/external/ (see its README).
EXT = DATA_DIR / "external"
FRANKEN_ZIP = EXT / "GlyphDataset.zip"          # Franken & van Gemert, iamai.nl GlyphDataset
ORACLE_DIR = EXT / "oracle_mnist"               # github.com/wm-bupt/oracle-mnist
DHCD_ZIP = EXT / "DevanagariHandwritten.zip"    # UCI Devanagari Handwritten Char Dataset
KMNIST_DIR = EXT / "kmnist"                     # ROIS-CODH Kuzushiji-MNIST (cursive Hiragana)

NOTO_EMB = ALPHABETS_DIR / "alphabets_unicode_embeddings_css_kmedoids.pkl"
OUTPUT_DIR = RESULTS_DIR / "external_validation" / "real_images"

TARGET = {
    "Egyptian hieroglyphs (real photos)": "Egyptian Hieroglyphs",
    "Oracle-bone (archaic Chinese)": "CJK",
    "Devanagari (real handwriting)": "Devanagari",
    "Kuzushiji cursive (historical Japanese)": "Hiragana",
}
# High-level "dense/pictorial" regime (Sec. 5 top split) for context on Oracle.
DENSE = {"CJK", "Tangut", "Egyptian Hieroglyphs", "Cuneiform",
         "Anatolian Hieroglyphs", "Phaistos Disc"}


def _standardize(img: np.ndarray, invert: bool) -> np.ndarray:
    """Bring a raw raster glyph into the model's expected form: binary, black
    ink on white, rescaled to a ~8 px stroke width (fit-priority), centred in
    105x105 -- the same normalization the Noto/Omniglot glyphs went through."""
    if invert:
        img = 255 - img
    bw = np.where(img > 128, 255, 0).astype(np.uint8)        # binarize (ink = 0)
    xs, ys = get_nonwhite_bounds(bw)
    if xs.size == 0 or ys.size == 0:
        return np.full((FRAME, FRAME), 255, np.uint8)
    crop = bw[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    sw = estimate_stroke_width(crop) or 1.0
    k = min(TARGET_STROKE / sw, MAXDIM / max(crop.shape))     # stroke=8 unless it would clip
    nh, nw = max(1, round(crop.shape[0] * k)), max(1, round(crop.shape[1] * k))
    r = cv2.resize(crop, (nw, nh),
                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    r = np.where(r > 128, 255, 0).astype(np.uint8)
    canvas = np.full((FRAME, FRAME), 255, np.uint8)
    y0, x0 = (FRAME - nh) // 2, (FRAME - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def _pick(items: list, k: int, rng: random.Random | None) -> list:
    """First k items (rng=None, the default draw) or k items sampled with rng."""
    items = list(items)
    if rng is None:
        return items[:k]
    return rng.sample(items, min(k, len(items)))


def _load_franken(max_per_label: int = 1, rng: random.Random | None = None) -> list[np.ndarray]:
    z = zipfile.ZipFile(FRANKEN_ZIP)
    by_label: dict[str, list[str]] = defaultdict(list)
    for n in z.namelist():
        if n.lower().endswith(".png") and "/Preprocessed/" in n:
            m = re.search(r"_([A-Za-z0-9]+)\.png$", n)
            if m and m.group(1) != "UNKNOWN":
                by_label[m.group(1)].append(n)
    imgs = []
    for label, files in sorted(by_label.items()):
        for f in _pick(files, max_per_label, rng):
            im = cv2.imdecode(np.frombuffer(z.read(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            imgs.append(_standardize(im, invert=False))
    return imgs


def _load_oracle(per_class: int = 5, rng: random.Random | None = None) -> list[np.ndarray]:
    with gzip.open(ORACLE_DIR / "train-images-idx3-ubyte.gz") as f:
        f.read(16); X = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28)
    with gzip.open(ORACLE_DIR / "train-labels-idx1-ubyte.gz") as f:
        f.read(8); y = np.frombuffer(f.read(), np.uint8)
    imgs = []
    for c in np.unique(y):
        for i in _pick(np.where(y == c)[0], per_class, rng):
            imgs.append(_standardize(X[i], invert=True))     # white-on-black -> black-on-white
    return imgs


def _load_dhcd(per_class: int = 1, rng: random.Random | None = None) -> list[np.ndarray]:
    z = zipfile.ZipFile(DHCD_ZIP)
    by_class: dict[str, list[str]] = defaultdict(list)
    for n in z.namelist():
        if n.lower().endswith(".png") and "/Train/" in n:
            by_class[n.split("/")[-2]].append(n)
    imgs = []
    for cls, files in sorted(by_class.items()):
        for f in _pick(files, per_class, rng):
            im = cv2.imdecode(np.frombuffer(z.read(f), np.uint8), cv2.IMREAD_GRAYSCALE)
            imgs.append(_standardize(im, invert=True))          # white-on-black -> black-on-white
    return imgs


def _load_kmnist(per_class: int = 3, rng: random.Random | None = None) -> list[np.ndarray]:
    X = np.load(KMNIST_DIR / "kmnist-train-imgs.npz")["arr_0"]
    y = np.load(KMNIST_DIR / "kmnist-train-labels.npz")["arr_0"]
    imgs = []
    for c in np.unique(y):
        for i in _pick(np.where(y == c)[0], per_class, rng):
            imgs.append(_standardize(X[i], invert=True))         # white-on-black -> black-on-white
    return imgs


def _load_all(rng: random.Random | None) -> dict[str, list[np.ndarray]]:
    return {
        "Egyptian hieroglyphs (real photos)": _load_franken(max_per_label=1, rng=rng),
        "Oracle-bone (archaic Chinese)": _load_oracle(per_class=5, rng=rng),
        "Devanagari (real handwriting)": _load_dhcd(per_class=1, rng=rng),
        "Kuzushiji cursive (historical Japanese)": _load_kmnist(per_class=3, rng=rng),
    }


def _retrieve(model, tf, noto: dict[str, np.ndarray], scripts: dict[str, list[np.ndarray]]) -> list[dict]:
    noto_scripts = sorted(noto)
    rows = []
    for name, imgs in scripts.items():
        Z = compute_embeddings(model, imgs, tf, use_teacher=False, model_kind="student_teacher")
        dists = {n: similarity_between_languages(dist_dict_from_embeddings_cosine(Z, noto[n]))
                 for n in noto_scripts}
        order = sorted(noto_scripts, key=lambda n: dists[n])
        tgt = TARGET[name]
        rank = order.index(tgt) + 1
        top1 = order[0]
        rows.append({
            "external_script": name,
            "n_glyphs": len(imgs),
            "noto_counterpart": tgt,
            "counterpart_rank": rank,
            "counterpart_dist": round(dists[tgt], 4),
            "nearest_noto": top1,
            "nearest_dist": round(dists[top1], 4),
            "nearest_in_dense_regime": top1 in DENSE,
            "counterpart_in_top3": rank <= 3,
            "top5": "; ".join(order[:5]),
        })
    return rows


def run(n_draws: int = 5) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_metric_model("student_teacher")
    tf = build_transform(image_size=105)
    print(f"[model] {ckpt.name}")

    emb = load_glyph_dictionary(NOTO_EMB)
    noto = {a: np.stack(list(emb[a].values())).astype(np.float32)
            for a in emb if not a.endswith(" Full")}

    # Default draw: the first images of each class (deterministic).
    df = pd.DataFrame(_retrieve(model, tf, noto, _load_all(rng=None)))
    df.to_csv(OUTPUT_DIR / "real_images_retrieval.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print("\n=== Cross-domain retrieval: real images -> Noto (default draw) ===")
        for _, r in df.iterrows():
            print(f"\n{r['external_script']}  (counterpart: {r['noto_counterpart']}, {r['n_glyphs']} glyphs)")
            print(f"  counterpart rank = {r['counterpart_rank']}/125   "
                  f"nearest = {r['nearest_noto']}   dense-regime = {r['nearest_in_dense_regime']}")
            print(f"  top-5: {r['top5']}")

    # Random draws: other images of the same classes, to check that the
    # counterpart rank does not depend on which instances were picked.
    draw_rows = []
    for seed in range(n_draws):
        for row in _retrieve(model, tf, noto, _load_all(rng=random.Random(seed))):
            draw_rows.append({"draw": seed, **row})
    draws = pd.DataFrame(draw_rows)
    draws.to_csv(OUTPUT_DIR / "real_images_retrieval_draws.csv", index=False)

    summary = draws.groupby(["external_script", "noto_counterpart"], sort=False).agg(
        n_draws=("draw", "size"),
        rank_1=("counterpart_rank", lambda r: int((r == 1).sum())),
        top_3=("counterpart_rank", lambda r: int((r <= 3).sum())),
        top_5=("counterpart_rank", lambda r: int((r <= 5).sum())),
        median_rank=("counterpart_rank", "median"),
        max_rank=("counterpart_rank", "max"),
        nearest_in_dense_regime=("nearest_in_dense_regime", "sum"),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / "real_images_retrieval_draws_summary.csv", index=False)
    print(f"\n=== Counterpart rank over {n_draws} random draws ===")
    print(summary.to_string(index=False))
    print(f"\n[OK] results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=5)
    run(parser.parse_args().draws)
