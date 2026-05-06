# -*- coding: utf-8 -*-
"""
Autonomous glyphworld model definitions for inference.

This module intentionally removes the dependency on glyphssl for:
- ContrastiveEncoder
- StudentTeacherEMA minimal wrapper
- checkpoint loading for:
    * teacher_only
    * student_teacher

Supported backbone:
- contrastive only
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Backbone: ContrastiveEncoder
# Copied/adapted from glyphssl.backbones.ContrastiveEncoder
# ---------------------------------------------------------------------
class ContrastiveEncoder(nn.Module):
    """
    Lightweight CNN encoder for glyph images.

    Architecture:
    - Conv-BN-ReLU-MaxPool x3
    - AdaptiveAvgPool
    - FC(128->256)
    - FC(256->embedding_dim)

    Optionally includes a projection head for training-time use.
    """

    def __init__(
        self,
        input_channels: int = 1,
        embedding_dim: int = 128,
        normalize: bool = True,
        use_projection: bool = False,
    ):
        super().__init__()
        self.normalize = normalize
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, embedding_dim)

        if use_projection:
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embedding_dim),
            )
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        z = self.fc2(x)

        if self.normalize:
            z = F.normalize(z, p=2, dim=1)

        return z

    def forward_for_loss(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        z = self.fc2(x)

        if self.projection is not None:
            z = self.projection(z)

        if self.normalize:
            z = F.normalize(z, p=2, dim=1)

        return z


# ---------------------------------------------------------------------
# Minimal StudentTeacherEMA wrapper for inference/checkpoint loading
# ---------------------------------------------------------------------
class StudentTeacherEMA(nn.Module):
    """
    Minimal version of the teacher-student model sufficient for loading
    and inference from saved checkpoints.
    """

    def __init__(
        self,
        pretrained_teacher: nn.Module,
        embedding_dim: Optional[int] = None,
        pred_hidden_dim: int = 512,
        ema_decay: float = 0.996,
    ):
        super().__init__()

        self.teacher = deepcopy(pretrained_teacher)
        self.student = deepcopy(pretrained_teacher)

        if embedding_dim is None:
            embedding_dim = get_embedding_dim(pretrained_teacher, fallback=128)

        self.embedding_dim = int(embedding_dim)
        self.ema_decay = float(ema_decay)

        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, int(pred_hidden_dim)),
            nn.BatchNorm1d(int(pred_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(pred_hidden_dim), self.embedding_dim),
        )

        for p in self.teacher.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, use_teacher: bool = True) -> torch.Tensor:
        return self.teacher(x) if use_teacher else self.student(x)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_backbone(
    name: str,
    input_channels: int = 1,
    embedding_dim: int = 128,
    normalize: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Build the supported backbone.

    Only 'contrastive' is supported in glyphworld autonomous mode.
    """
    name = name.lower().strip()

    if name != "contrastive":
        raise ValueError(
            f"Unsupported backbone in glyphworld autonomous mode: {name!r}. "
            "Only 'contrastive' is supported."
        )

    return ContrastiveEncoder(
        input_channels=input_channels,
        embedding_dim=int(embedding_dim),
        normalize=normalize,
        use_projection=bool(kwargs.get("use_projection", False)),
    )


def get_embedding_dim(backbone: nn.Module, fallback: int = 128) -> int:
    if hasattr(backbone, "embedding_dim"):
        try:
            return int(getattr(backbone, "embedding_dim"))
        except Exception:
            pass
    return int(fallback)


def _extract_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    """
    Supports:
    - {'model_state_dict': ...}
    - raw state_dict
    """
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        return ckpt
    raise ValueError("Unrecognized checkpoint format.")


def infer_pred_hidden_dim_from_state(state: dict[str, torch.Tensor]) -> int:
    """
    Infer predictor hidden dim from predictor.0.weight shape.
    """
    candidates = [
        "predictor.0.weight",
        "student.predictor.0.weight",
        "teacher.predictor.0.weight",
    ]

    for key in candidates:
        if key in state and hasattr(state[key], "shape") and len(state[key].shape) == 2:
            return int(state[key].shape[0])

    for key, val in state.items():
        if "predictor.0.weight" in key and hasattr(val, "shape") and len(val.shape) == 2:
            return int(val.shape[0])

    return 512


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_teacher_from_pt(
    ckpt_path: Union[str, Path],
    device: Optional[torch.device] = None,
    input_channels: int = 1,
    normalize: bool = True,
    strict: bool = True,
) -> nn.Module:
    """
    Load a teacher-only checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = _extract_state_dict(ckpt)

    backbone_name = ckpt.get("backbone_name", "contrastive") if isinstance(ckpt, dict) else "contrastive"
    backbone_kwargs = ckpt.get("backbone_kwargs", {}) if isinstance(ckpt, dict) else {}
    embedding_dim = int(ckpt.get("embedding_dim", 128)) if isinstance(ckpt, dict) else 128

    teacher = build_backbone(
        backbone_name,
        input_channels=input_channels,
        embedding_dim=embedding_dim,
        normalize=normalize,
        **(backbone_kwargs or {}),
    ).to(device)

    teacher.load_state_dict(state, strict=strict)
    teacher.eval()

    print(
        f"[load_teacher] {ckpt_path} | backbone={backbone_name} | "
        f"embedding_dim={get_embedding_dim(teacher)}"
    )
    return teacher


def load_student_teacher_from_pt(
    ckpt_path: Union[str, Path],
    device: Optional[torch.device] = None,
    input_channels: int = 1,
    normalize: bool = True,
    strict: bool = True,
) -> StudentTeacherEMA:
    """
    Load a student-teacher checkpoint.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = _extract_state_dict(ckpt)

    backbone_name = ckpt.get("backbone_name", "contrastive") if isinstance(ckpt, dict) else "contrastive"
    backbone_kwargs = ckpt.get("backbone_kwargs", {}) if isinstance(ckpt, dict) else {}
    embedding_dim = int(ckpt.get("embedding_dim", 128)) if isinstance(ckpt, dict) else 128
    ema_decay = float(ckpt.get("ema_decay", 0.996)) if isinstance(ckpt, dict) else 0.996

    pred_hidden_dim = infer_pred_hidden_dim_from_state(state)

    backbone = build_backbone(
        backbone_name,
        input_channels=input_channels,
        embedding_dim=embedding_dim,
        normalize=normalize,
        **(backbone_kwargs or {}),
    )

    model = StudentTeacherEMA(
        pretrained_teacher=backbone,
        embedding_dim=embedding_dim,
        pred_hidden_dim=pred_hidden_dim,
        ema_decay=ema_decay,
    ).to(device)

    model.load_state_dict(state, strict=strict)
    model.eval()

    print(
        f"[load_student_teacher] {ckpt_path} | backbone={backbone_name} | "
        f"embedding_dim={get_embedding_dim(model.teacher)} | "
        f"pred_hidden_dim={pred_hidden_dim}"
    )
    return model
