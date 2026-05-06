# -*- coding: utf-8 -*-
"""
Compute and save latent representations (embeddings) for each alphabet
using the autonomous glyphworld teacher/student pipeline.

No dependency on glyphssl.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from glyphworld.core.paths import ALPHABETS_DIR, MODELS_DIR
from glyphworld.core.utils import load_glyph_dictionary, save_glyph_dictionary
from glyphworld.models.model import (
    load_student_teacher_from_pt,
    load_teacher_from_pt,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODEL_KIND = "student_teacher"   # "student_teacher" | "teacher_only"
USE_TEACHER = False              # only used if MODEL_KIND == "student_teacher"

if MODEL_KIND == "student_teacher":
    CKPT_PATH = MODELS_DIR / "css" / "student_teacher_final.pt"
elif MODEL_KIND == "teacher_only":
    CKPT_PATH = MODELS_DIR / "css" / "teacher_pretrained.pt"
else:
    raise ValueError(f"Unknown MODEL_KIND: {MODEL_KIND}")

BATCH_SIZE = 256
IMAGE_SIZE = 105
MEAN = [0.5]
STD = [0.5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_transform(image_size: int = 105) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )


def imgs_to_batch(images, tf, device):
    xs = []
    for img in images:
        x = tf(img)
        if x.ndim == 3 and x.shape[0] == 3:
            x = x.mean(dim=0, keepdim=True)
        xs.append(x)
    return torch.stack(xs, dim=0).to(device, non_blocking=True)


@torch.no_grad()
def compute_embeddings(
    model,
    images,
    tf,
    batch_size: int = 256,
    use_teacher: bool = True,
    model_kind: str = "student_teacher",
):
    all_z = []

    for i in range(0, len(images), batch_size):
        batch_imgs = images[i : i + batch_size]
        batch = imgs_to_batch(batch_imgs, tf=tf, device=DEVICE)

        if model_kind == "student_teacher":
            z = model.teacher(batch) if use_teacher else model.student(batch)
        elif model_kind == "teacher_only":
            z = model(batch)
        else:
            raise ValueError(f"Unknown MODEL_KIND: {model_kind}")

        z = F.normalize(z, dim=1, p=2)
        all_z.append(z.cpu().numpy().astype(np.float32, copy=False))

    return np.concatenate(all_z, axis=0)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    tf = build_transform(IMAGE_SIZE)

    if MODEL_KIND == "student_teacher":
        model = load_student_teacher_from_pt(
            ckpt_path=CKPT_PATH,
            device=DEVICE,
            normalize=True,
            strict=True,
        )
        model.eval()

    elif MODEL_KIND == "teacher_only":
        model = load_teacher_from_pt(
            ckpt_path=CKPT_PATH,
            device=DEVICE,
            normalize=True,
            strict=True,
        )
        model.eval()

    else:
        raise ValueError(f"Unknown MODEL_KIND: {MODEL_KIND}")

    load_path = ALPHABETS_DIR / "alphabets_unicode.pkl"
    alphabets_dic = load_glyph_dictionary(load_path)
    print("Loaded alphabets:", len(alphabets_dic))

    embeddings_dic = {}

    for alphabet, glyphs in alphabets_dic.items():
        keys = list(glyphs.keys())
        imgs = [glyphs[k] for k in keys]

        print(f"\nAlphabet: {alphabet} | glyphs: {len(keys)}")
        Z = compute_embeddings(
            model=model,
            images=imgs,
            tf=tf,
            batch_size=BATCH_SIZE,
            use_teacher=USE_TEACHER,
            model_kind=MODEL_KIND,
        )

        embeddings_dic[alphabet] = {
            k: Z[i].astype(np.float32, copy=False)
            for i, k in enumerate(keys)
        }

    save_name = f"alphabets_unicode_embeddings_css.pkl"
    save_path = ALPHABETS_DIR / save_name

    save_glyph_dictionary(embeddings_dic, save_path)
    print(f"\n[SAVED] {save_path}")
