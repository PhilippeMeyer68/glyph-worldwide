"""
Pickle helpers and glyph-level cosine distances.
"""

import numpy as np
import pickle



def save_glyph_dictionary(glyph_dic, path_dic):
    with open(path_dic, "wb") as f:
        pickle.dump(glyph_dic, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_glyph_dictionary(path_dic):
    with open(path_dic, "rb") as f:
        glyph_dic = pickle.load(f)
    return glyph_dic


def short_name_from_filename(fname: str) -> str:
    if fname.endswith(".npy"):
        stem = fname[:-4]
    else:
        stem = fname
    if stem.startswith("alphabet_"):
        stem = stem[len("alphabet_"):]
    if stem.endswith("_reduced"):
        stem = stem[:-8]
    return stem


def cosine_distance_matrix_block(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    A: (na, d), B: (nb, d)
    Both must already be L2-normalized.
    Returns (na, nb) with d(a,b) = 1 - cosine_similarity(a,b)
    """
    sims = A @ B.T
    return (1.0 - sims).astype(np.float32)


def dist_dict_from_embeddings_cosine(
    Z1: np.ndarray,
    Z2: np.ndarray,
    block_i: int = 64,
    block_j: int = 512,
) -> dict[tuple[int, int], float]:
    """
    Replaces old Siamese-head distance with cosine distance on normalized embeddings.
    """
    Z1 = np.asarray(Z1, dtype=np.float32)
    Z2 = np.asarray(Z2, dtype=np.float32)

    # L2 normalization for safety
    Z1 = Z1 / (np.linalg.norm(Z1, axis=1, keepdims=True) + 1e-12)
    Z2 = Z2 / (np.linalg.norm(Z2, axis=1, keepdims=True) + 1e-12)

    n1, d1 = Z1.shape
    n2, d2 = Z2.shape
    if d1 != d2:
        raise ValueError(f"Dim mismatch: Z1 {Z1.shape}, Z2 {Z2.shape}")

    out: dict[tuple[int, int], float] = {}

    for i0 in range(0, n1, block_i):
        i1 = min(i0 + block_i, n1)
        A = Z1[i0:i1]

        for j0 in range(0, n2, block_j):
            j1 = min(j0 + block_j, n2)
            B = Z2[j0:j1]

            dist = cosine_distance_matrix_block(A, B)

            for ii in range(i1 - i0):
                base_i = i0 + ii
                row = dist[ii]
                for jj in range(j1 - j0):
                    out[(base_i, j0 + jj)] = float(row[jj])

    return out
