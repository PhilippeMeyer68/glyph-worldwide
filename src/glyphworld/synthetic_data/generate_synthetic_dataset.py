from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from glyphworld.core.paths import ALPHABETS_DIR, RESULTS_DIR
from glyphworld.core.utils import load_glyph_dictionary, save_glyph_dictionary
from glyphworld.data.data_affine_distortion import (
    to_black_and_white,
    recenter_like_create_glyph,
    sample_valid_augmentation,
)


# ============================================================
# Global config
# ============================================================

ALPHABETS_PKL_NAME = "alphabets_unicode.pkl"
OUTPUT_DIR = RESULTS_DIR / "validation"

# Common distortion settings
BORDER_MARGIN = 1
MAX_TRIES = 300

TRAIN_FRAC = 0.8
VALID_FRAC = 0.1
TEST_FRAC = 0.1


# ============================================================
# Dataset-specific configs
# ============================================================

DATASET_CONFIGS = [
    {
        "dataset_name": "metric_selection",
        "output_name": "synthetic_pairs_metric_selection.pkl",
        "seed": 42,

        # Larger dataset because pairwise metric evaluation is cheaper.
        "n_close_pairs": 100,
        "n_distant_pairs": 100,
        "n_negative_pairs": 100,

        # Broad range to test metric robustness.
        "min_glyph_draws": 5,
        "max_glyph_draws": 80,
        "max_tokens_per_type": 3,
    },
    {
        "dataset_name": "clustering_validation",
        "output_name": "synthetic_pairs_clustering_validation.pkl",
        "seed": 123,

        # Smaller dataset because clustering needs all-vs-all distances.
        # This gives at most 300 synthetic scripts.
        "n_close_pairs": 100,
        "n_distant_pairs": 100,
        "n_negative_pairs": 100,

        # Avoid very tiny scripts for clustering stability.
        "min_glyph_draws": 5,
        "max_glyph_draws": 80,
        "max_tokens_per_type": 3,
    },
]


# ============================================================
# Tier-specific parameters
# ============================================================

# CLOSE pairs: high overlap, light distortion
CLOSE_CORE_OVERLAP_RANGE = (0.6, 0.9)
CLOSE_ROTATION_RANGE = (-10, 10)
CLOSE_SHEAR_RANGE = (-0.3, 0.3)
CLOSE_ZOOM_RANGE = (0.8, 1.2)
CLOSE_SHIFT_RANGE = (-2, 2)

# DISTANT pairs: low overlap, heavy distortion
DISTANT_CORE_OVERLAP_RANGE = (0.3, 0.5)
DISTANT_ROTATION_RANGE = (-30, 30)
DISTANT_SHEAR_RANGE = (-0.6, 0.6)
DISTANT_ZOOM_RANGE = (0.6, 1.4)
DISTANT_SHIFT_RANGE = (-5, 5)

# NEGATIVE pairs: different ancestors
NEG_ROTATION_RANGE = (-10, 10)
NEG_SHEAR_RANGE = (-0.3, 0.3)
NEG_ZOOM_RANGE = (0.8, 1.2)
NEG_SHIFT_RANGE = (-2, 2)


# ============================================================
# Metadata containers
# ============================================================

@dataclass
class SyntheticScript:
    script_id: str
    ancestor_alphabet: str
    glyph_images: list[np.ndarray]
    source_keys: list[str]
    draw_count: int


@dataclass
class PairRecord:
    pair_id: str
    left_id: str
    right_id: str
    label: float
    split: str
    left_ancestor_alphabet: str
    right_ancestor_alphabet: str
    left_draw_count: int
    right_draw_count: int


# ============================================================
# Helpers
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_glyph(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.uint8)
    img = to_black_and_white(img)
    img = recenter_like_create_glyph(img)
    return img.astype(np.uint8, copy=False)


def distort_glyph(
    img: np.ndarray,
    rotation_range: tuple[int, int],
    shear_range: tuple[float, float],
    zoom_range: tuple[float, float],
    shift_range: tuple[int, int],
) -> np.ndarray:
    img = normalize_glyph(img)
    aug = sample_valid_augmentation(
        img,
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        shift_range=shift_range,
        margin=BORDER_MARGIN,
        max_tries=MAX_TRIES,
    )
    return aug.astype(np.uint8, copy=False)


def sample_split(rng: random.Random) -> str:
    u = rng.random()
    if u < TRAIN_FRAC:
        return "train"
    if u < TRAIN_FRAC + VALID_FRAC:
        return "valid"
    return "test"


def safe_script_prefix(alphabet_name: str) -> str:
    return (
        alphabet_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("`", "")
        .replace("'", "")
    )


# ============================================================
# Script construction
# ============================================================

def _draw_with_allographic_expansion(
    rng: random.Random,
    pool_keys: list[str],
    glyph_dict: dict[str, np.ndarray],
    target_count: int,
    max_tokens_per_type: int,
    rotation_range: tuple[int, int],
    shear_range: tuple[float, float],
    zoom_range: tuple[float, float],
    shift_range: tuple[int, int],
) -> tuple[list[str], list[np.ndarray]]:

    source_keys: list[str] = []
    images: list[np.ndarray] = []

    while len(images) < target_count:
        key = rng.choice(pool_keys)
        remaining = target_count - len(images)
        n_tokens = min(rng.randint(1, max_tokens_per_type), remaining)
        base_img = glyph_dict[key]

        for _ in range(n_tokens):
            images.append(
                distort_glyph(
                    base_img,
                    rotation_range=rotation_range,
                    shear_range=shear_range,
                    zoom_range=zoom_range,
                    shift_range=shift_range,
                )
            )
            source_keys.append(key)

    return source_keys, images


def make_synthetic_script(
    rng: random.Random,
    alphabet_name: str,
    pool_keys: list[str],
    glyph_dict: dict[str, np.ndarray],
    synthetic_idx: int,
    dataset_name: str,
    rotation_range: tuple[int, int],
    shear_range: tuple[float, float],
    zoom_range: tuple[float, float],
    shift_range: tuple[int, int],
    min_draws: int,
    max_draws: int,
    max_tokens_per_type: int,
) -> SyntheticScript:

    n_target = rng.randint(min_draws, max_draws)

    source_keys, distorted = _draw_with_allographic_expansion(
        rng=rng,
        pool_keys=pool_keys,
        glyph_dict=glyph_dict,
        target_count=n_target,
        max_tokens_per_type=max_tokens_per_type,
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        shift_range=shift_range,
    )

    script_id = (
        f"{dataset_name}_synthetic_"
        f"{safe_script_prefix(alphabet_name)}_"
        f"{synthetic_idx:07d}"
    )

    return SyntheticScript(
        script_id=script_id,
        ancestor_alphabet=alphabet_name,
        glyph_images=distorted,
        source_keys=source_keys,
        draw_count=len(distorted),
    )


def _split_into_pools(
    rng: random.Random,
    all_keys: list[str],
    core_overlap_range: tuple[float, float],
) -> tuple[list[str], list[str], list[str]]:

    n = len(all_keys)

    if n <= 2:
        return list(all_keys), [], []

    core_frac = rng.uniform(*core_overlap_range)
    n_core = max(1, int(round(core_frac * n)))
    n_core = min(n_core, n)

    shuffled = list(all_keys)
    rng.shuffle(shuffled)

    core = shuffled[:n_core]
    rest = shuffled[n_core:]

    split_point = len(rest) // 2
    private_left = rest[:split_point]
    private_right = rest[split_point:]

    return core, private_left, private_right


def _store_script(store: dict[str, dict[str, Any]], script: SyntheticScript) -> None:
    store[script.script_id] = {
        "ancestor_alphabet": script.ancestor_alphabet,
        "glyph_images": script.glyph_images,
        "source_keys": script.source_keys,
        "draw_count": script.draw_count,
    }


# ============================================================
# Pair builders
# ============================================================

def _build_related_pairs(
    alphabets_dic: dict[str, dict[str, np.ndarray]],
    n_pairs: int,
    seed: int,
    label: float,
    prefix: str,
    start_index: int,
    dataset_name: str,
    min_glyph_draws: int,
    max_glyph_draws: int,
    max_tokens_per_type: int,
    core_overlap_range: tuple[float, float],
    rotation_range: tuple[int, int],
    shear_range: tuple[float, float],
    zoom_range: tuple[float, float],
    shift_range: tuple[int, int],
) -> tuple[list[PairRecord], dict[str, dict[str, Any]]]:

    rng = random.Random(seed)
    alphabet_names = sorted(alphabets_dic.keys())

    pair_records: list[PairRecord] = []
    script_store: dict[str, dict[str, Any]] = {}

    for i in range(n_pairs):
        alphabet = rng.choice(alphabet_names)
        glyph_dict = alphabets_dic[alphabet]
        all_keys = sorted(glyph_dict.keys())

        core, priv_left, priv_right = _split_into_pools(
            rng=rng,
            all_keys=all_keys,
            core_overlap_range=core_overlap_range,
        )

        left_pool = core + priv_left
        right_pool = core + priv_right

        idx = start_index + 2 * i

        left = make_synthetic_script(
            rng=rng,
            alphabet_name=alphabet,
            pool_keys=left_pool,
            glyph_dict=glyph_dict,
            synthetic_idx=idx,
            dataset_name=dataset_name,
            rotation_range=rotation_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            shift_range=shift_range,
            min_draws=min_glyph_draws,
            max_draws=max_glyph_draws,
            max_tokens_per_type=max_tokens_per_type,
        )

        right = make_synthetic_script(
            rng=rng,
            alphabet_name=alphabet,
            pool_keys=right_pool,
            glyph_dict=glyph_dict,
            synthetic_idx=idx + 1,
            dataset_name=dataset_name,
            rotation_range=rotation_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            shift_range=shift_range,
            min_draws=min_glyph_draws,
            max_draws=max_glyph_draws,
            max_tokens_per_type=max_tokens_per_type,
        )

        _store_script(script_store, left)
        _store_script(script_store, right)

        pair_records.append(
            PairRecord(
                pair_id=f"{dataset_name}_{prefix}_{i:07d}",
                left_id=left.script_id,
                right_id=right.script_id,
                label=label,
                split=sample_split(rng),
                left_ancestor_alphabet=alphabet,
                right_ancestor_alphabet=alphabet,
                left_draw_count=left.draw_count,
                right_draw_count=right.draw_count,
            )
        )

        if (i + 1) % 500 == 0:
            print(f"[{dataset_name}:{prefix}] built {i + 1}/{n_pairs}")

    return pair_records, script_store


def build_close_pairs(
    alphabets_dic: dict[str, dict[str, np.ndarray]],
    cfg: dict[str, Any],
    start_index: int,
) -> tuple[list[PairRecord], dict[str, dict[str, Any]]]:

    return _build_related_pairs(
        alphabets_dic=alphabets_dic,
        n_pairs=cfg["n_close_pairs"],
        seed=cfg["seed"],
        label=1.0,
        prefix="close",
        start_index=start_index,
        dataset_name=cfg["dataset_name"],
        min_glyph_draws=cfg["min_glyph_draws"],
        max_glyph_draws=cfg["max_glyph_draws"],
        max_tokens_per_type=cfg["max_tokens_per_type"],
        core_overlap_range=CLOSE_CORE_OVERLAP_RANGE,
        rotation_range=CLOSE_ROTATION_RANGE,
        shear_range=CLOSE_SHEAR_RANGE,
        zoom_range=CLOSE_ZOOM_RANGE,
        shift_range=CLOSE_SHIFT_RANGE,
    )


def build_distant_pairs(
    alphabets_dic: dict[str, dict[str, np.ndarray]],
    cfg: dict[str, Any],
    start_index: int,
) -> tuple[list[PairRecord], dict[str, dict[str, Any]]]:

    return _build_related_pairs(
        alphabets_dic=alphabets_dic,
        n_pairs=cfg["n_distant_pairs"],
        seed=cfg["seed"] + 1,
        label=0.5,
        prefix="distant",
        start_index=start_index,
        dataset_name=cfg["dataset_name"],
        min_glyph_draws=cfg["min_glyph_draws"],
        max_glyph_draws=cfg["max_glyph_draws"],
        max_tokens_per_type=cfg["max_tokens_per_type"],
        core_overlap_range=DISTANT_CORE_OVERLAP_RANGE,
        rotation_range=DISTANT_ROTATION_RANGE,
        shear_range=DISTANT_SHEAR_RANGE,
        zoom_range=DISTANT_ZOOM_RANGE,
        shift_range=DISTANT_SHIFT_RANGE,
    )


def build_negative_pairs(
    alphabets_dic: dict[str, dict[str, np.ndarray]],
    cfg: dict[str, Any],
    start_index: int,
) -> tuple[list[PairRecord], dict[str, dict[str, Any]]]:

    rng = random.Random(cfg["seed"] + 2)
    alphabet_names = sorted(alphabets_dic.keys())

    pair_records: list[PairRecord] = []
    script_store: dict[str, dict[str, Any]] = {}

    for i in range(cfg["n_negative_pairs"]):
        left_alphabet, right_alphabet = rng.sample(alphabet_names, 2)

        left_glyph_dict = alphabets_dic[left_alphabet]
        right_glyph_dict = alphabets_dic[right_alphabet]

        idx = start_index + 2 * i

        left = make_synthetic_script(
            rng=rng,
            alphabet_name=left_alphabet,
            pool_keys=sorted(left_glyph_dict.keys()),
            glyph_dict=left_glyph_dict,
            synthetic_idx=idx,
            dataset_name=cfg["dataset_name"],
            rotation_range=NEG_ROTATION_RANGE,
            shear_range=NEG_SHEAR_RANGE,
            zoom_range=NEG_ZOOM_RANGE,
            shift_range=NEG_SHIFT_RANGE,
            min_draws=cfg["min_glyph_draws"],
            max_draws=cfg["max_glyph_draws"],
            max_tokens_per_type=cfg["max_tokens_per_type"],
        )

        right = make_synthetic_script(
            rng=rng,
            alphabet_name=right_alphabet,
            pool_keys=sorted(right_glyph_dict.keys()),
            glyph_dict=right_glyph_dict,
            synthetic_idx=idx + 1,
            dataset_name=cfg["dataset_name"],
            rotation_range=NEG_ROTATION_RANGE,
            shear_range=NEG_SHEAR_RANGE,
            zoom_range=NEG_ZOOM_RANGE,
            shift_range=NEG_SHIFT_RANGE,
            min_draws=cfg["min_glyph_draws"],
            max_draws=cfg["max_glyph_draws"],
            max_tokens_per_type=cfg["max_tokens_per_type"],
        )

        _store_script(script_store, left)
        _store_script(script_store, right)

        pair_records.append(
            PairRecord(
                pair_id=f"{cfg['dataset_name']}_neg_{i:07d}",
                left_id=left.script_id,
                right_id=right.script_id,
                label=0.0,
                split=sample_split(rng),
                left_ancestor_alphabet=left_alphabet,
                right_ancestor_alphabet=right_alphabet,
                left_draw_count=left.draw_count,
                right_draw_count=right.draw_count,
            )
        )

        if (i + 1) % 500 == 0:
            print(f"[{cfg['dataset_name']}:neg] built {i + 1}/{cfg['n_negative_pairs']}")

    return pair_records, script_store


# ============================================================
# Dataset assembly
# ============================================================

def build_dataset(
    alphabets_dic: dict[str, dict[str, np.ndarray]],
    cfg: dict[str, Any],
    allowed_alphabets: list[str],
) -> dict[str, Any]:

    set_seed(cfg["seed"])

    alphabets_subset = {
        name: alphabets_dic[name]
        for name in allowed_alphabets
    }

    print("\n" + "=" * 70)
    print(f"[dataset] building: {cfg['dataset_name']}")
    print(f"[dataset] seed: {cfg['seed']}")
    print(f"[dataset] source alphabets: {len(alphabets_subset)}")
    print(
        "[dataset] pairs: "
        f"close={cfg['n_close_pairs']}, "
        f"distant={cfg['n_distant_pairs']}, "
        f"negative={cfg['n_negative_pairs']}"
    )
    print(
        "[dataset] glyph draws: "
        f"{cfg['min_glyph_draws']}–{cfg['max_glyph_draws']}"
    )

    close_pairs, close_scripts = build_close_pairs(
        alphabets_dic=alphabets_subset,
        cfg=cfg,
        start_index=0,
    )

    distant_pairs, distant_scripts = build_distant_pairs(
        alphabets_dic=alphabets_subset,
        cfg=cfg,
        start_index=2 * cfg["n_close_pairs"],
    )

    neg_pairs, neg_scripts = build_negative_pairs(
        alphabets_dic=alphabets_subset,
        cfg=cfg,
        start_index=2 * (cfg["n_close_pairs"] + cfg["n_distant_pairs"]),
    )

    all_pairs = close_pairs + distant_pairs + neg_pairs

    rng = random.Random(cfg["seed"])
    rng.shuffle(all_pairs)

    dataset = {
        "config": {
            "dataset_name": cfg["dataset_name"],
            "seed": cfg["seed"],
            "alphabets_pkl_name": ALPHABETS_PKL_NAME,
            "source_alphabets": allowed_alphabets,
            "n_source_alphabets": len(allowed_alphabets),
            "n_close_pairs": cfg["n_close_pairs"],
            "n_distant_pairs": cfg["n_distant_pairs"],
            "n_negative_pairs": cfg["n_negative_pairs"],
            "min_glyph_draws": cfg["min_glyph_draws"],
            "max_glyph_draws": cfg["max_glyph_draws"],
            "max_tokens_per_type": cfg["max_tokens_per_type"],
            "close_core_overlap_range": CLOSE_CORE_OVERLAP_RANGE,
            "close_distortion": {
                "rotation": CLOSE_ROTATION_RANGE,
                "shear": CLOSE_SHEAR_RANGE,
                "zoom": CLOSE_ZOOM_RANGE,
                "shift": CLOSE_SHIFT_RANGE,
            },
            "distant_core_overlap_range": DISTANT_CORE_OVERLAP_RANGE,
            "distant_distortion": {
                "rotation": DISTANT_ROTATION_RANGE,
                "shear": DISTANT_SHEAR_RANGE,
                "zoom": DISTANT_ZOOM_RANGE,
                "shift": DISTANT_SHIFT_RANGE,
            },
            "negative_distortion": {
                "rotation": NEG_ROTATION_RANGE,
                "shear": NEG_SHEAR_RANGE,
                "zoom": NEG_ZOOM_RANGE,
                "shift": NEG_SHIFT_RANGE,
            },
            "border_margin": BORDER_MARGIN,
            "max_tries": MAX_TRIES,
            "train_frac": TRAIN_FRAC,
            "valid_frac": VALID_FRAC,
            "test_frac": TEST_FRAC,
        },
        "pairs": [asdict(p) for p in all_pairs],
        "synthetic_scripts": {
            **close_scripts,
            **distant_scripts,
            **neg_scripts,
        },
    }

    return dataset


def save_dataset(dataset: dict[str, Any], output_name: str) -> None:
    ensure_dir(OUTPUT_DIR)

    save_path = OUTPUT_DIR / output_name
    save_glyph_dictionary(dataset, save_path)

    labels = np.array([float(pair["label"]) for pair in dataset["pairs"]])

    print(f"[saved] {save_path}")
    print(f"[saved] total_pairs={len(dataset['pairs'])}")
    print(f"[saved] close={int(np.sum(labels == 1.0))}")
    print(f"[saved] distant={int(np.sum(labels == 0.5))}")
    print(f"[saved] negative={int(np.sum(labels == 0.0))}")
    print(f"[saved] total_synthetic_scripts={len(dataset['synthetic_scripts'])}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)

    load_path = ALPHABETS_DIR / ALPHABETS_PKL_NAME
    alphabets_dic = load_glyph_dictionary(load_path)

    print(f"[load] {load_path}")
    print(f"[load] alphabets={len(alphabets_dic)}")
    print(f"[output] {OUTPUT_DIR}")

    alphabet_names = sorted(alphabets_dic.keys())

    rng = random.Random(999)
    rng.shuffle(alphabet_names)

    split_idx = len(alphabet_names) // 2

    metric_alphabets = alphabet_names[:split_idx]
    clustering_alphabets = alphabet_names[split_idx:]

    assert set(metric_alphabets).isdisjoint(set(clustering_alphabets))

    print("\n[split] disjoint source alphabets")
    print(f"[split] metric_selection alphabets: {len(metric_alphabets)}")
    print(f"[split] clustering_validation alphabets: {len(clustering_alphabets)}")

    for cfg in DATASET_CONFIGS:
        if cfg["dataset_name"] == "metric_selection":
            allowed_alphabets = metric_alphabets
        elif cfg["dataset_name"] == "clustering_validation":
            allowed_alphabets = clustering_alphabets
        else:
            raise ValueError(f"Unknown dataset_name: {cfg['dataset_name']}")

        dataset = build_dataset(
            alphabets_dic=alphabets_dic,
            cfg=cfg,
            allowed_alphabets=allowed_alphabets,
        )

        save_dataset(dataset, cfg["output_name"])


if __name__ == "__main__":
    main()