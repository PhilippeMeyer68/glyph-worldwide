# -*- coding: utf-8 -*-
"""
Compute alphabet-pair similarities using precomputed teacher/student embeddings
and cosine distance.
"""

from __future__ import annotations

import numpy as np

from glyphworld.distances.distance_functions import similarity_between_languages
from glyphworld.core.paths import ALPHABETS_DIR, PROCESSED_DIR
from glyphworld.core.utils import (
    dist_dict_from_embeddings_cosine,
    short_name_from_filename,
    save_glyph_dictionary,
    load_glyph_dictionary,
)


def compute_alphabet_similarities_from_embeddings_dict(embeddings_dic, alphabets):
    alphabet_distances = {}

    for i in range(len(alphabets) - 1):
        a1 = alphabets[i]
        Z1 = np.stack(list(embeddings_dic[a1].values()), axis=0).astype(np.float32, copy=False)
        print(f"[{i + 1}/{len(alphabets) - 1}] {a1}: {Z1.shape[0]} glyphs")

        for j in range(i + 1, len(alphabets)):
            a2 = alphabets[j]
            Z2 = np.stack(list(embeddings_dic[a2].values()), axis=0).astype(np.float32, copy=False)

            dist_dict = dist_dict_from_embeddings_cosine(Z1, Z2)
            sim = similarity_between_languages(dist_dict)
            alphabet_distances[(i, j)] = sim

    return alphabet_distances


if __name__ == "__main__":
    EMB_PKL_NAME = "alphabets_unicode_embeddings_css_kmedoids.pkl"
    load_path = ALPHABETS_DIR / EMB_PKL_NAME
    embeddings_dic = load_glyph_dictionary(load_path)
    print(f"[LOAD] {load_path} ({len(embeddings_dic)} entries)")

    alphabets = sorted([a for a in embeddings_dic.keys() if not a.endswith(" Full")])
    alphabets_short = [short_name_from_filename(a) for a in alphabets]

    alphabet_distances = compute_alphabet_similarities_from_embeddings_dict(
        embeddings_dic, alphabets
    )

    print(
        "Computed",
        len(alphabet_distances),
        "alphabet-pair similarities out of",
        len(alphabets),
        "alphabets.",
    )
    print("alphabets:", alphabets_short)

    out_path = PROCESSED_DIR / "global distances" / "alphabet_similarity_matrix_cosine.pkl"

    global_sim_dic = {"alphabets": alphabets, "alphabet_distances": alphabet_distances}
    save_glyph_dictionary(global_sim_dic, out_path)

    print(f"Saved alphabet similarities to: {out_path}")
