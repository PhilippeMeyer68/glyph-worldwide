# -*- coding: utf-8 -*-
"""
Augment glyph images.
"""

from __future__ import annotations

import random
import numpy as np
import cv2
import scipy.ndimage

from glyphworld.core.paths import ALPHABETS_DIR
from glyphworld.core.utils import load_glyph_dictionary, save_glyph_dictionary
from glyphworld.data.build_alphabet_data import get_nonwhite_bounds

WHITE = 255
OUT_H = 105
OUT_W = 105


# -------------------------
# Bounds / validity
# -------------------------
def is_out_of_box(img: np.ndarray, margin: int = 0) -> bool:
    """
    True if non-white pixels touch (or get too close to) borders.

    margin=0 reproduces the 'touch border' test.
    margin=1 means within 1px of border is considered invalid, etc.
    """
    h, w = img.shape
    x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
    if x_nonwhite.size == 0 or y_nonwhite.size == 0:
        return True  # empty / invalid

    return (
        x_nonwhite[0] <= margin
        or y_nonwhite[0] <= margin
        or x_nonwhite[-1] >= (w - 1 - margin)
        or y_nonwhite[-1] >= (h - 1 - margin)
    )


def recenter_like_create_glyph(img: np.ndarray) -> np.ndarray:
    """
    Recenter using the SAME offset logic as create_glyph,
    but applied as a shift on the already-rendered image.
    """
    h, w = img.shape
    x_nonwhite, y_nonwhite = get_nonwhite_bounds(img)
    if x_nonwhite.size == 0 or y_nonwhite.size == 0:
        return img.astype(np.uint8, copy=False)

    width = x_nonwhite[-1] - x_nonwhite[0] + 1
    height = y_nonwhite[-1] - y_nonwhite[0] + 1

    x_offset = (w - width) // 2 - x_nonwhite[0]
    y_offset = (h - height) // 2 - y_nonwhite[0]

    shifted = scipy.ndimage.shift(
        img.astype(np.float32),
        shift=(y_offset, x_offset),
        order=0,  # nearest-neighbor to preserve binary edges
        cval=WHITE,
    )
    return shifted.astype(np.uint8)


# -------------------------
# Pixel helpers
# -------------------------
def to_black_and_white(img: np.ndarray) -> np.ndarray:
    """Binarize to {0,255}."""
    return (img > 127.5).astype(np.uint8) * WHITE


def pad_canvas(img: np.ndarray, pad: int = 20, cval: int = WHITE) -> np.ndarray:
    return np.pad(img, ((pad, pad), (pad, pad)), mode="constant", constant_values=cval)


def center_crop(img: np.ndarray, out_h: int = OUT_H, out_w: int = OUT_W) -> np.ndarray:
    h, w = img.shape
    y0 = (h - out_h) // 2
    x0 = (w - out_w) // 2
    return img[y0 : y0 + out_h, x0 : x0 + out_w]


# -------------------------
# OpenCV affine helpers
# -------------------------
def _warp_affine_nn_white(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Warp with nearest-neighbor, constant white background, same size."""
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=WHITE,
    )


def padded_zoom(img: np.ndarray, zoomfactor: float = 0.8) -> np.ndarray:
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, zoomfactor)
    return _warp_affine_nn_white(img, M)


def image_aug_raw(
    image: np.ndarray,
    rotation_range,
    shear_range,
    zoom_range,
    shift_range,
    p_rotate: float = 0.5,
    p_shear: float = 0.5,
    p_zoom: float = 0.5,
    p_shift: float = 0.5,
) -> np.ndarray:
    """
    Apply random rotation/shear/zoom/shift using OpenCV.

    shear_range is assumed to be in radians (same as your previous skimage setup).
    """
    image2 = image.astype(np.float32)
    h, w = image2.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    if random.random() < p_rotate:
        angle = random.uniform(*rotation_range)  # degrees
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image2 = _warp_affine_nn_white(image2, M)

    if random.random() < p_shear:
        shear_val = random.uniform(*shear_range)  # radians
        sh = np.tan(shear_val)  # angle (rad) -> shear factor
        # x' = x + sh*(y - cy)
        M = np.array([[1.0, sh, -sh * cy],
                      [0.0, 1.0, 0.0]], dtype=np.float32)
        image2 = _warp_affine_nn_white(image2, M)

    if random.random() < p_zoom:
        zoom_val = random.uniform(*zoom_range)
        image2 = padded_zoom(image2, zoom_val)

    if random.random() < p_shift:
        dy = random.uniform(*shift_range)
        dx = random.uniform(*shift_range)
        M = np.array([[1.0, 0.0, dx],
                      [0.0, 1.0, dy]], dtype=np.float32)
        image2 = _warp_affine_nn_white(image2, M)

    return image2


def sample_valid_augmentation(
    original_bw: np.ndarray,
    rotation_range,
    shear_range,
    zoom_range,
    shift_range,
    margin: int,
    max_tries: int,
    pad: int = 20,
) -> np.ndarray:
    for _ in range(max_tries):
        big = pad_canvas(original_bw, pad=pad, cval=WHITE)

        aug_big = image_aug_raw(
            big,
            rotation_range=rotation_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            shift_range=shift_range,
        )

        aug_big = to_black_and_white(aug_big)
        aug_big = recenter_like_create_glyph(aug_big)

        # crop back to 105x105 AFTER recenter
        aug = center_crop(aug_big, OUT_H, OUT_W).astype(np.uint8)

        if not is_out_of_box(aug, margin=margin):
            return aug

    # fallback
    return recenter_like_create_glyph(original_bw)


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    rotation_range = (-10, 10)
    shear_range = (-0.3, 0.3)   # radians
    zoom_range = (0.8, 1.2)
    shift_range = (-2, 2)

    n_aug = 20
    border_margin = 1
    max_tries = 300

    in_medoids_name = "alphabets_unicode_medoids"
    out_aug_name = "alphabets_unicode_medoids_augmented"

    load_path = ALPHABETS_DIR / in_medoids_name
    alphabets_medoids = load_glyph_dictionary(load_path)
    alphabets_unicode_augmented: dict[str, dict[str, list[np.ndarray]]] = {}

    for alphabet, glyphs in alphabets_medoids.items():
        if alphabet.endswith(" Full"):
            continue

        print(f"[BUILD] {alphabet}  n_glyphs={len(glyphs)}")
        out_glyphs: dict[str, list[np.ndarray]] = {}

        for glyph_key, img in glyphs.items():
            original_np = np.asarray(img, dtype=np.uint8)
            original_bw = to_black_and_white(original_np)
            original_bw = recenter_like_create_glyph(original_bw)

            if is_out_of_box(original_bw, margin=border_margin):
                print(f"[WARN] original touches border: {alphabet}/{glyph_key}")

            out_imgs = [original_bw]
            for _ in range(n_aug):
                aug = sample_valid_augmentation(
                    original_bw,
                    rotation_range=rotation_range,
                    shear_range=shear_range,
                    zoom_range=zoom_range,
                    shift_range=shift_range,
                    margin=border_margin,
                    max_tries=max_tries,
                )
                out_imgs.append(aug)

            out_glyphs[glyph_key] = out_imgs

        alphabets_unicode_augmented[alphabet] = out_glyphs

    # save
    save_path = ALPHABETS_DIR / out_aug_name
    save_glyph_dictionary(alphabets_unicode_augmented, save_path)
    print(f"[SAVED] {save_path}")
