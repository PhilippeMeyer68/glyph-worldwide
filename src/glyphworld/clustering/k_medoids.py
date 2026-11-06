# -*- coding: utf-8 -*-
"""
Batch-free k-medoids on glyph embeddings using cosine distance.
"""

from __future__ import annotations

import numpy as np

from glyphworld.core.paths import ALPHABETS_DIR
from glyphworld.core.utils import load_glyph_dictionary, save_glyph_dictionary


# ============================================================
# CONFIG
# ============================================================

K_MEDOIDS = 100
MAX_ITER = 20
SEED = 0

# pickle names (without .pkl if your utils adds it automatically;
IN_NAME = ALPHABETS_DIR / "alphabets_unicode_embeddings_css.pkl"
OUT_NAME = ALPHABETS_DIR / "alphabets_unicode_embeddings_css_kmedoids.pkl"

# IMAGES
IN_IMAGES_NAME = ALPHABETS_DIR / "alphabets_unicode.pkl"
OUT_IMAGES_NAME = ALPHABETS_DIR / "alphabets_unicode_medoids.pkl"


# ============================================================
# DISTANCE HELPERS
# ============================================================

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)


def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Returns dense (nA, nB) float32 matrix with:
        d(a, b) = 1 - cosine_similarity(a, b)

    A and B are L2-normalized row-wise inside this function for safety.
    """
    A = l2_normalize_rows(A)
    B = l2_normalize_rows(B)
    sims = A @ B.T
    return (1.0 - sims).astype(np.float32, copy=False)


# ============================================================
# K-MEDOIDS
# ============================================================

def assign_labels(Z: np.ndarray, medoids: np.ndarray):
    """Assign each point to nearest medoid."""
    D = cosine_distance_matrix(Z[medoids], Z)   # (k, n)
    labels = np.argmin(D, axis=0)
    cost = D[labels, np.arange(Z.shape[0])].sum()
    return labels, float(cost)


def update_medoids(Z: np.ndarray, labels: np.ndarray, k: int, rng):
    """Exact medoid update inside each cluster."""
    n = Z.shape[0]
    new_medoids = np.empty(k, dtype=int)

    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            new_medoids[c] = int(rng.integers(n))
            continue

        Zm = np.asarray(Z[members], dtype=np.float32)
        D = cosine_distance_matrix(Zm, Zm)
        costs = D.sum(axis=1)
        new_medoids[c] = int(members[np.argmin(costs)])

    return new_medoids


def kmedoids(Z: np.ndarray, k: int, seed: int = SEED, max_iter: int = MAX_ITER):
    """Plain PAM-style k-medoids with cosine distance."""
    Z = np.asarray(Z, dtype=np.float32)
    n = Z.shape[0]

    if n == 0:
        raise ValueError("Empty Z.")
    if k > n:
        k = n

    rng = np.random.default_rng(seed)
    medoids = rng.choice(n, size=k, replace=False)

    prev_cost = None
    for it in range(max_iter):
        labels, cost = assign_labels(Z, medoids)
        new_medoids = update_medoids(Z, labels, k, rng)

        # ensure uniqueness
        uniq, used = [], set()
        for m in new_medoids:
            m = int(m)
            if m not in used:
                uniq.append(m)
                used.add(m)

        while len(uniq) < k:
            cand = int(rng.integers(n))
            if cand not in used:
                uniq.append(cand)
                used.add(cand)

        new_medoids = np.array(uniq, dtype=int)

        if np.array_equal(new_medoids, medoids):
            print(f"[STOP] converged at iter {it}")
            return medoids, labels, cost

        if prev_cost is not None and abs(prev_cost - cost) < 1e-12:
            print(f"[STOP] cost stabilized at iter {it}")
            return new_medoids, labels, cost

        medoids = new_medoids
        prev_cost = cost
        print(f"[ITER {it:02d}] cost={cost:.6f}")

    labels, cost = assign_labels(Z, medoids)
    print("[STOP] max_iter reached")
    return medoids, labels, cost


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print(f"[LOAD] embeddings dictionary: {IN_NAME}")
    embeddings_dic = load_glyph_dictionary(IN_NAME)

    print(f"[LOAD] images dictionary: {IN_IMAGES_NAME}")
    images_dic = load_glyph_dictionary(IN_IMAGES_NAME)

    # Keep all original entries + add reduced versions for names ending with " Full"
    out_emb_dic = dict(embeddings_dic)
    out_img_dic = dict(images_dic)

    full_names = sorted([name for name in embeddings_dic.keys() if name.endswith(" Full")])
    print("Number of *Full alphabets:", len(full_names))

    for name in full_names:
        base_name = name[:-5]  # remove trailing " Full"
        print("\n", name, "->", base_name)

        glyph_embs = embeddings_dic[name]   # dict[glyph_key] -> embedding
        keys = list(glyph_embs.keys())
        Z = np.stack([glyph_embs[k] for k in keys], axis=0).astype(np.float32, copy=False)

        print(f"Glyphs: {len(keys)} | Z: {Z.shape}")

        print("[RUN] k-medoids")
        medoids, labels, cost = kmedoids(Z, K_MEDOIDS)
        print("Total cost:", cost)

        medoid_keys = [keys[i] for i in medoids]

        # save medoid embeddings
        out_emb_dic[base_name] = {k: glyph_embs[k] for k in medoid_keys}

        # save corresponding medoid images
        glyph_imgs = images_dic[name]
        out_img_dic[base_name] = {k: glyph_imgs[k] for k in medoid_keys}

        print(f"[OK] {base_name}: kept {len(medoid_keys)} medoids")

    save_glyph_dictionary(out_emb_dic, OUT_NAME)
    print(f"\n[SAVED] {OUT_NAME}")

    save_glyph_dictionary(out_img_dic, OUT_IMAGES_NAME)
    print(f"[SAVED] {OUT_IMAGES_NAME}")
