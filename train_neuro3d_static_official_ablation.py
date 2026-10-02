#!/usr/bin/env python3
"""
Neuro-3D official-component static-only ablation.

Purpose
-------
Use the publicly released Neuro-3D encoder/head components as closely as
possible, but remove the dynamic EEG branch and dynamic-static cross-attention.
The static EEG feature is sent directly to the official downstream blocks:

    static EEG [B,64,250]
      -> static temporal Transformer (EEGAttention)
      -> static Linear(250,250)
      -> official ConvBlock
      -> official linear_projection
      -> official temporal_aggregation
      -> geometry head [1024] -> 72-class shape/category classifier
      -> appearance head [1024] -> 6-class color classifier

Both geometry and appearance features are aligned to the paired released
CLIP video feature, following the public classification objective:

    L = 0.99 * 10 * (MSE(geometry, visual) + MSE(appearance, visual))
      + 0.01 * 10 * (CLIP(geometry, visual) + CLIP(appearance, visual))
      + 0.1 * (CE_shape + weighted_CE_color)

This is a static-only ABLATION built from the official public components.
It is NOT the full Neuro-3D model because the dynamic EEG stream and the
cross-attention fusion are intentionally omitted.

Development protocol
--------------------
Official train EEG array: [72,8,2,64,250]
  train objects: all official train objects except --val_objects
  train repetitions: the two repetitions are independent examples
  validation objects: --val_objects (default 06,07)
  validation EEG: mean of the two repetitions
  checkpoint selection: minimum validation 72-category CE

Final protocol
--------------
  train objects 00..07 with both repetitions as independent examples
  official test objects 08,09, four repetitions averaged
  fixed --epochs chosen during development
  test is evaluated once after training

No additional z-score/min-max normalization is applied.

The public repository uses test performance for model selection. This script
does NOT do that. Development model selection is kept inside official train
objects so the released 08/09 test objects can remain final-only.
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "pandas is required to read color_label.xlsx. "
        "Install the Neuro-3D requirements first."
    ) from exc


CLASS_COUNT = 72
COLOR_COUNT = 6
TRAIN_OBJECTS = 8
TEST_OBJECTS = 2
TRAIN_REPS = 2
TEST_REPS = 4
CHANNELS = 64
STATIC_SAMPLES = 250
SEMANTIC_DIM = 1024

# Public retri_shape_color.py:
# class_weights = torch.tensor([1.0, 1.0, 0.1, 0.1, 1.0, 1.0])
OFFICIAL_COLOR_WEIGHTS = (1.0, 1.0, 0.1, 0.1, 1.0, 1.0)


# =============================================================================
# Utilities
# =============================================================================

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_object_list(value: str) -> Sequence[int]:
    values = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not values or len(set(values)) != len(values):
        raise ValueError(f"Invalid object list: {value}")
    if any(x < 0 or x >= TRAIN_OBJECTS for x in values):
        raise ValueError("Development val objects must be in 00..07")
    return values


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def topk_counts(logits: torch.Tensor, labels: torch.Tensor, k: int):
    k = min(k, logits.shape[1])
    pred = logits.topk(k, dim=1).indices
    return int(pred.eq(labels[:, None]).any(dim=1).sum().item())


# =============================================================================
# Released EEG / multimodal metadata
# =============================================================================

def load_eeg_array(data_path: str, sub_id: str, split: str):
    root = Path(data_path).expanduser() / "EEGdata" / sub_id
    path = root / f"{sub_id}_{split}_data_1s_250Hz.npy"
    if not path.is_file():
        raise FileNotFoundError(path)

    value = np.load(path, mmap_mode="r")

    if split == "train":
        expected_no_sub = (
            CLASS_COUNT, TRAIN_OBJECTS, TRAIN_REPS, CHANNELS, STATIC_SAMPLES
        )
    elif split == "test":
        expected_no_sub = (
            CLASS_COUNT, TEST_OBJECTS, TEST_REPS, CHANNELS, STATIC_SAMPLES
        )
    else:
        raise ValueError(split)

    # Public files are commonly [72,...], while some earlier local loaders
    # accepted a leading singleton subject dimension.
    if tuple(value.shape) == (1,) + expected_no_sub:
        value = value[0]

    if tuple(value.shape) != expected_no_sub:
        raise ValueError(
            f"{path}: expected {expected_no_sub} or {(1,) + expected_no_sub}, "
            f"got {tuple(value.shape)}"
        )
    if not np.isfinite(value).all():
        raise ValueError(f"{path} contains NaN or Inf")

    return value, str(path)


def build_official_name_matrices(data_path: str):
    """
    Reproduce the public EEGdataset.py filename filtering/order.
    """
    video_dir = Path(data_path).expanduser() / "video_new"
    if not video_dir.is_dir():
        raise FileNotFoundError(video_dir)

    all_names = sorted(p.name for p in video_dir.iterdir())

    def collect(remove_suffixes):
        names = []
        for name in all_names:
            if not name.endswith(".mp4"):
                continue
            # Public code uses name[-6:-4] to identify object 00..09.
            if name[-6:-4] in remove_suffixes:
                continue
            names.append(name[:-4])
        return names

    train = collect({"08", "09"})
    test = collect({f"{x:02d}" for x in range(8)})

    if len(train) != CLASS_COUNT * TRAIN_OBJECTS:
        raise ValueError(
            f"Expected {CLASS_COUNT * TRAIN_OBJECTS} train names, got {len(train)}"
        )
    if len(test) != CLASS_COUNT * TEST_OBJECTS:
        raise ValueError(
            f"Expected {CLASS_COUNT * TEST_OBJECTS} test names, got {len(test)}"
        )

    train = np.asarray(train, dtype=object).reshape(CLASS_COUNT, TRAIN_OBJECTS)
    test = np.asarray(test, dtype=object).reshape(CLASS_COUNT, TEST_OBJECTS)
    return train, test


def load_color_mapping(data_path: str) -> Dict[str, int]:
    path = Path(data_path).expanduser() / "color_label.xlsx"
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_excel(path)
    if "name" not in frame.columns or "label" not in frame.columns:
        raise ValueError(
            f"{path} must contain columns named 'name' and 'label'"
        )

    mapping = {}
    for _, row in frame.iterrows():
        mapping[str(row["name"])] = int(row["label"])

    labels = sorted(set(mapping.values()))
    if not labels:
        raise ValueError("No color labels found")
    if min(labels) < 0 or max(labels) >= COLOR_COUNT:
        raise ValueError(
            f"Expected color labels in 0..{COLOR_COUNT - 1}; found range "
            f"{min(labels)}..{max(labels)}"
        )
    return mapping


def load_clip_video_features(
    data_path: str,
    train_names: np.ndarray,
    test_names: np.ndarray,
    feature_path: Optional[str] = None,
):
    path = (
        Path(feature_path).expanduser()
        if feature_path is not None
        else Path(data_path).expanduser() / "clip_feature.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(path)

    feature_dict = safe_torch_load(path)
    if not isinstance(feature_dict, dict):
        raise ValueError(f"Expected dict in {path}, got {type(feature_dict)}")

    def one(name):
        # Public EEGdataset.py accesses clip_features[name[3:]].
        key = str(name)[3:]
        if key not in feature_dict:
            raise KeyError(
                f"Missing CLIP key '{key}' for stimulus '{name}' in {path}"
            )
        entry = feature_dict[key]
        if "video" not in entry:
            raise KeyError(f"CLIP entry '{key}' has no 'video' feature")
        x = entry["video"]
        if torch.is_tensor(x):
            x = x.detach().cpu().float().numpy()
        x = np.asarray(x, dtype=np.float32).squeeze()
        if x.shape != (SEMANTIC_DIM,):
            raise ValueError(
                f"Expected video feature [{SEMANTIC_DIM}], got {x.shape} "
                f"for {name}"
            )
        return x

    train = np.empty(
        (CLASS_COUNT, TRAIN_OBJECTS, SEMANTIC_DIM), dtype=np.float32
    )
    test = np.empty(
        (CLASS_COUNT, TEST_OBJECTS, SEMANTIC_DIM), dtype=np.float32
    )

    for c in range(CLASS_COUNT):
        for o in range(TRAIN_OBJECTS):
            train[c, o] = one(train_names[c, o])
        for o in range(TEST_OBJECTS):
            test[c, o] = one(test_names[c, o])

    return train, test, str(path)


def make_color_label_matrix(names: np.ndarray, color_map: Dict[str, int]):
    result = np.empty(names.shape, dtype=np.int64)
    for c in range(names.shape[0]):
        for o in range(names.shape[1]):
            # Public code uses self.name_list[ii,jj][3:].
            key = str(names[c, o])[3:]
            if key not in color_map:
                raise KeyError(
                    f"Color label missing for '{key}' derived from '{names[c,o]}'"
                )
            result[c, o] = int(color_map[key])
    return result


# =============================================================================
# Datasets
# =============================================================================

class TrainStaticDataset(Dataset):
    """
    Two repetitions are independent dataset entries, as in public EEGdataset.py.
    """

    def __init__(
        self,
        raw,
        object_ids: Sequence[int],
        visual_features,
        color_labels,
        augmentation=True,
        mean_probability=0.25,
        noise_probability=0.60,
        noise_scale=0.2,
    ):
        self.raw = raw
        self.object_ids = list(object_ids)
        self.visual = visual_features
        self.color_labels = color_labels
        self.augmentation = bool(augmentation)
        self.mean_probability = float(mean_probability)
        self.noise_probability = float(noise_probability)
        self.noise_scale = float(noise_scale)

        self.index = []
        for cls in range(CLASS_COUNT):
            for obj in self.object_ids:
                for rep in range(TRAIN_REPS):
                    self.index.append((cls, obj, rep))

    def __len__(self):
        return len(self.index)

    def _add_noise(self, eeg: torch.Tensor):
        # Public add_noise(): std over time dimension for [64,T].
        std = eeg.std(dim=1, keepdim=True)
        std = torch.nan_to_num(std, nan=0.0)
        return eeg + torch.randn_like(eeg) * std * self.noise_scale

    def __getitem__(self, idx):
        cls, obj, rep = self.index[idx]

        if self.augmentation and np.random.rand() < self.mean_probability:
            eeg = np.asarray(
                self.raw[cls, obj].mean(axis=0), dtype=np.float32
            ).copy()
        else:
            eeg = np.asarray(
                self.raw[cls, obj, rep], dtype=np.float32
            ).copy()

        eeg = torch.from_numpy(eeg)

        if self.augmentation and np.random.random() < self.noise_probability:
            eeg = self._add_noise(eeg)

        return {
            "eeg": eeg,
            "shape_label": torch.tensor(cls, dtype=torch.long),
            "color_label": torch.tensor(
                int(self.color_labels[cls, obj]), dtype=torch.long
            ),
            "visual": torch.from_numpy(
                np.asarray(self.visual[cls, obj], dtype=np.float32)
            ),
            "object_id": torch.tensor(obj, dtype=torch.long),
            "rep_id": torch.tensor(rep, dtype=torch.long),
        }


class AveragedStaticDataset(Dataset):
    """
    Development validation: mean over 2 repetitions.
    Final test: mean over 4 repetitions.
    """

    def __init__(
        self,
        raw,
        object_indices: Sequence[int],
        external_object_ids: Sequence[int],
        visual_features,
        color_labels,
    ):
        if len(object_indices) != len(external_object_ids):
            raise ValueError("object index/id length mismatch")

        self.samples = []
        for cls in range(CLASS_COUNT):
            for local_obj, external_obj in zip(
                object_indices, external_object_ids
            ):
                eeg = np.asarray(
                    raw[cls, local_obj].mean(axis=0), dtype=np.float32
                )
                self.samples.append(
                    (
                        cls,
                        local_obj,
                        external_obj,
                        eeg,
                    )
                )

        self.visual = visual_features
        self.color_labels = color_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls, local_obj, external_obj, eeg = self.samples[idx]
        return {
            "eeg": torch.from_numpy(eeg.copy()),
            "shape_label": torch.tensor(cls, dtype=torch.long),
            "color_label": torch.tensor(
                int(self.color_labels[cls, local_obj]), dtype=torch.long
            ),
            "visual": torch.from_numpy(
                np.asarray(self.visual[cls, local_obj], dtype=np.float32)
            ),
            "object_id": torch.tensor(external_obj, dtype=torch.long),
        }


# =============================================================================
# Official public model components
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(
            0, max_len, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model + 1, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(
            position * div_term[: d_model // 2]
        )
        pe[:, 1::2] = torch.cos(
            position * div_term[: d_model // 2]
        )
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [T,B,C]
        pe = (
            self.pe[: x.size(0), :]
            .unsqueeze(1)
            .repeat(1, x.size(1), 1)
        )
        return x + pe


class EEGAttention(nn.Module):
    """
    Public Neuro-3D temporal self-attention block.
    """

    def __init__(self, channel, d_model, nhead, max_len=600):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model, max_len=max_len
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=1,
        )

    def forward(self, src):
        # [B,C,T] -> [T,B,C]
        src = src.permute(2, 0, 1)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output.permute(1, 2, 0)


class ConvBlock(nn.Module):
    """
    Public Neuro-3D ConvBlock.
    """

    def __init__(self, num_channels, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(
            num_channels, num_features,
            kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv1d(
            num_features, num_features,
            kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv1d(
            num_features, num_features,
            kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        self.norm3 = nn.LayerNorm(num_features)
        self.residual_conv = nn.Conv1d(
            num_channels, num_features, kernel_size=1
        )

    def forward(self, x):
        residual = self.residual_conv(x)

        x = F.gelu(self.conv1(x))
        x = self.norm1(x)

        x = F.gelu(self.conv2(x))
        x = self.norm2(x)

        x = F.gelu(self.conv3(x))
        x = self.norm3(x)

        return x + residual


class MLPHead(nn.Module):
    """
    Public MLPHead:
      [B,C,L] -> [B,L,C] -> LN -> Linear -> GELU -> Dropout -> flatten.
    """

    def __init__(self, in_features, num_latents, dropout_rate=0.25):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, num_latents)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = F.gelu(self.linear(x))
        x = self.dropout(x)
        return x.reshape(x.shape[0], -1)


class Neuro3DStaticOfficialAblation(nn.Module):
    """
    Static-only forward built from VideoImageEEGClassifyColor3 components.

    Omitted intentionally:
      attention_model (dynamic)
      dynamic_linear
      dynamic_static cross-attention

    Retained:
      static_attention
      static_linear
      conv_blocks
      linear_projection
      temporal_aggregation
      geometry / appearance MLP heads
      shape / color classifiers
      trainable logit_scale
    """

    def __init__(
        self,
        num_channels=64,
        sequence_length2=250,
        num_latents=1024,
        num_blocks=1,
        cls_num=72,
    ):
        super().__init__()

        self.static_attention = EEGAttention(
            num_channels, num_channels, nhead=1
        )
        self.static_linear = nn.Linear(
            sequence_length2, sequence_length2
        )

        self.conv_blocks = nn.Sequential(
            *[
                ConvBlock(num_channels, sequence_length2)
                for _ in range(num_blocks)
            ]
        )

        # Official sequence of rearranges + Linear:
        # after ConvBlock: [B,250,250]
        # official outer Rearrange -> [B,L,C], which is still [B,250,250]
        # then linear_projection rearranges back before Linear(250,1024).
        self.linear_projection = nn.Linear(
            sequence_length2, num_latents
        )
        self.temporal_aggregation = nn.Linear(
            sequence_length2, 1
        )

        self.geometry_head = MLPHead(
            num_latents, num_latents
        )
        self.shape_classifier = nn.Linear(
            num_latents, cls_num
        )

        self.appearance_head = MLPHead(
            num_latents, num_latents
        )
        self.color_classifier = nn.Linear(
            num_latents, COLOR_COUNT
        )

        # Public model initializes this to log(1/0.01).
        # Public training/retrieval code passes/multiplies this parameter
        # directly, so this ablation follows that observable behavior.
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / 0.01)
        )

    def forward(self, static_eeg):
        if static_eeg.ndim != 3 or tuple(static_eeg.shape[1:]) != (
            CHANNELS, STATIC_SAMPLES
        ):
            raise ValueError(
                f"Expected [B,{CHANNELS},{STATIC_SAMPLES}], "
                f"got {tuple(static_eeg.shape)}"
            )

        x = self.static_attention(static_eeg)
        x = self.static_linear(x)

        # Official conv_blocks ends with Rearrange('B C L -> B L C').
        x = self.conv_blocks(x)
        x = x.permute(0, 2, 1)  # [B,L,C]

        # Official linear_projection:
        # Rearrange('B L C -> B C L')
        x = x.permute(0, 2, 1)
        x = self.linear_projection(x)
        # Rearrange('B C L -> B L C')
        x = x.permute(0, 2, 1)

        # [B,1024,250] -> [B,1024,1]
        x_tem = self.temporal_aggregation(x)

        geometry = self.geometry_head(x_tem)
        shape_logits = self.shape_classifier(geometry)

        appearance = self.appearance_head(x_tem)
        color_logits = self.color_classifier(appearance)

        return {
            "geometry": geometry,
            "appearance": appearance,
            "shape_logits": shape_logits,
            "color_logits": color_logits,
        }


# =============================================================================
# Official-style objective
# =============================================================================

def clip_loss_public_style(
    eeg_features: torch.Tensor,
    visual_features: torch.Tensor,
    logit_scale: torch.Tensor,
):
    """
    Symmetric CLIP-style batch contrastive objective.

    The public retri_shape_color.py passes the model's logit_scale parameter
    directly to its ClipLoss and uses the same parameter directly in retrieval.
    This standalone version mirrors that observable interface:

        logits = logit_scale * eeg_features @ visual_features.T

    No extra feature normalization is introduced here.
    """
    logits_eeg = logit_scale * (eeg_features @ visual_features.T)
    logits_visual = logits_eeg.T
    labels = torch.arange(
        logits_eeg.shape[0], device=logits_eeg.device
    )
    return 0.5 * (
        F.cross_entropy(logits_eeg, labels)
        + F.cross_entropy(logits_visual, labels)
    )


def compute_losses(
    model,
    outputs,
    visual,
    shape_label,
    color_label,
    color_weight_tensor,
    alpha=0.99,
):
    geometry = outputs["geometry"]
    appearance = outputs["appearance"]

    mse_geometry = F.mse_loss(geometry, visual)
    mse_appearance = F.mse_loss(appearance, visual)
    regress_loss = mse_geometry + mse_appearance

    clip_geometry = clip_loss_public_style(
        geometry, visual, model.logit_scale
    )
    clip_appearance = clip_loss_public_style(
        appearance, visual, model.logit_scale
    )
    contrastive_loss = clip_geometry + clip_appearance

    shape_ce = F.cross_entropy(
        outputs["shape_logits"], shape_label
    )
    color_ce = F.cross_entropy(
        outputs["color_logits"],
        color_label,
        weight=color_weight_tensor,
    )
    classification_loss = shape_ce + color_ce

    total = (
        alpha * regress_loss * 10.0
        + (1.0 - alpha) * contrastive_loss * 10.0
        + classification_loss * 0.1
    )

    return {
        "total": total,
        "shape_ce": shape_ce,
        "color_ce": color_ce,
        "regress": regress_loss,
        "contrastive": contrastive_loss,
        "mse_geometry": mse_geometry,
        "mse_appearance": mse_appearance,
        "clip_geometry": clip_geometry,
        "clip_appearance": clip_appearance,
    }


# =============================================================================
# Training / evaluation
# =============================================================================

def run_train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    device,
    color_weight_tensor,
    alpha,
    grad_clip,
):
    model.train()

    sums = {
        "total": 0.0,
        "shape_ce": 0.0,
        "color_ce": 0.0,
        "regress": 0.0,
        "contrastive": 0.0,
    }
    shape_top1 = shape_top5 = 0
    color_top1 = color_top2 = 0
    total_n = 0

    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True).float()
        shape_label = (
            batch["shape_label"].to(device, non_blocking=True).long()
        )
        color_label = (
            batch["color_label"].to(device, non_blocking=True).long()
        )
        visual = (
            batch["visual"].to(device, non_blocking=True).float()
        )

        optimizer.zero_grad(set_to_none=True)

        outputs = model(eeg)
        losses = compute_losses(
            model,
            outputs,
            visual,
            shape_label,
            color_label,
            color_weight_tensor,
            alpha=alpha,
        )
        losses["total"].backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )

        optimizer.step()
        scheduler.step()

        n = shape_label.numel()
        total_n += n

        for key in sums:
            sums[key] += float(losses[key].item()) * n

        shape_top1 += int(
            (outputs["shape_logits"].argmax(1) == shape_label)
            .sum()
            .item()
        )
        shape_top5 += topk_counts(
            outputs["shape_logits"], shape_label, 5
        )
        color_top1 += int(
            (outputs["color_logits"].argmax(1) == color_label)
            .sum()
            .item()
        )
        color_top2 += topk_counts(
            outputs["color_logits"], color_label, 2
        )

    return {
        **{k: v / total_n for k, v in sums.items()},
        "shape_top1": shape_top1 / total_n,
        "shape_top5": shape_top5 / total_n,
        "color_top1": color_top1 / total_n,
        "color_top2": color_top2 / total_n,
        "n": total_n,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    color_weight_tensor,
    alpha,
    predictions_path: Optional[Path] = None,
):
    model.eval()

    sums = {
        "total": 0.0,
        "shape_ce": 0.0,
        "color_ce": 0.0,
        "regress": 0.0,
        "contrastive": 0.0,
    }
    shape_top1 = shape_top5 = 0
    color_top1 = color_top2 = 0
    total_n = 0

    all_shape_logits = []
    all_color_logits = []
    all_shape_labels = []
    all_color_labels = []
    all_objects = []

    for batch in loader:
        eeg = batch["eeg"].to(device).float()
        shape_label = batch["shape_label"].to(device).long()
        color_label = batch["color_label"].to(device).long()
        visual = batch["visual"].to(device).float()

        outputs = model(eeg)
        losses = compute_losses(
            model,
            outputs,
            visual,
            shape_label,
            color_label,
            color_weight_tensor,
            alpha=alpha,
        )

        n = shape_label.numel()
        total_n += n

        for key in sums:
            sums[key] += float(losses[key].item()) * n

        shape_top1 += int(
            (outputs["shape_logits"].argmax(1) == shape_label)
            .sum()
            .item()
        )
        shape_top5 += topk_counts(
            outputs["shape_logits"], shape_label, 5
        )
        color_top1 += int(
            (outputs["color_logits"].argmax(1) == color_label)
            .sum()
            .item()
        )
        color_top2 += topk_counts(
            outputs["color_logits"], color_label, 2
        )

        if predictions_path is not None:
            all_shape_logits.append(
                outputs["shape_logits"].detach().cpu().numpy()
            )
            all_color_logits.append(
                outputs["color_logits"].detach().cpu().numpy()
            )
            all_shape_labels.append(
                shape_label.detach().cpu().numpy()
            )
            all_color_labels.append(
                color_label.detach().cpu().numpy()
            )
            all_objects.append(
                batch["object_id"].detach().cpu().numpy()
            )

    result = {
        **{k: v / total_n for k, v in sums.items()},
        "shape_top1": shape_top1 / total_n,
        "shape_top5": shape_top5 / total_n,
        "color_top1": color_top1 / total_n,
        "color_top2": color_top2 / total_n,
        "n": total_n,
    }

    if predictions_path is not None:
        np.savez(
            predictions_path,
            shape_logits=np.concatenate(all_shape_logits),
            color_logits=np.concatenate(all_color_logits),
            shape_labels=np.concatenate(all_shape_labels),
            color_labels=np.concatenate(all_color_labels),
            object_ids=np.concatenate(all_objects),
        )

    return result


def clone_state(model):
    return {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    args,
    epoch,
    history,
    metadata,
):
    torch.save(
        {
            "model": clone_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "history": history,
            "metadata": metadata,
        },
        path,
    )


# =============================================================================
# Main
# =============================================================================

def train(args):
    out = Path(args.out_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"Use a fresh output directory: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        or not args.device.startswith("cuda")
        else "cpu"
    )

    train_raw, train_eeg_path = load_eeg_array(
        args.data_path, args.sub_id, "train"
    )

    train_names, test_names = build_official_name_matrices(
        args.data_path
    )
    color_map = load_color_mapping(args.data_path)

    train_color_labels = make_color_label_matrix(
        train_names, color_map
    )
    test_color_labels = make_color_label_matrix(
        test_names, color_map
    )

    (
        train_visual,
        test_visual,
        visual_path,
    ) = load_clip_video_features(
        args.data_path,
        train_names,
        test_names,
        args.visual_feature_path,
    )

    if args.protocol == "development":
        val_objects = list(parse_object_list(args.val_objects))
        train_objects = [
            x for x in range(TRAIN_OBJECTS)
            if x not in val_objects
        ]
        if not train_objects:
            raise ValueError("No training objects remain")

        train_dataset = TrainStaticDataset(
            train_raw,
            train_objects,
            train_visual,
            train_color_labels,
            augmentation=args.official_augmentation,
            mean_probability=args.mean_probability,
            noise_probability=args.noise_probability,
            noise_scale=args.augmentation_noise_scale,
        )

        # Validation uses the two-repetition mean for each held-out train object.
        val_dataset = AveragedStaticDataset(
            train_raw,
            object_indices=val_objects,
            external_object_ids=val_objects,
            visual_features=train_visual,
            color_labels=train_color_labels,
        )
        eval_dataset = val_dataset
        eval_name = "VAL"

    else:
        train_objects = list(range(TRAIN_OBJECTS))
        val_objects = []

        if args.epochs is None:
            raise ValueError(
                "--protocol final requires fixed --epochs selected in development"
            )

        train_dataset = TrainStaticDataset(
            train_raw,
            train_objects,
            train_visual,
            train_color_labels,
            augmentation=args.official_augmentation,
            mean_probability=args.mean_probability,
            noise_probability=args.noise_probability,
            noise_scale=args.augmentation_noise_scale,
        )

        test_raw, test_eeg_path = load_eeg_array(
            args.data_path, args.sub_id, "test"
        )
        test_dataset = AveragedStaticDataset(
            test_raw,
            object_indices=[0, 1],
            external_object_ids=[8, 9],
            visual_features=test_visual,
            color_labels=test_color_labels,
        )
        eval_dataset = test_dataset
        eval_name = "TEST"

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.drop_last,
        generator=generator,
        **loader_kwargs,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    # Deterministic non-augmented training evaluation.
    train_eval_dataset = TrainStaticDataset(
        train_raw,
        train_objects,
        train_visual,
        train_color_labels,
        augmentation=False,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = Neuro3DStaticOfficialAblation(
        num_channels=CHANNELS,
        sequence_length2=STATIC_SAMPLES,
        num_latents=SEMANTIC_DIM,
        num_blocks=1,
        cls_num=CLASS_COUNT,
    ).to(device)

    color_weight_tensor = torch.tensor(
        OFFICIAL_COLOR_WEIGHTS,
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
    )

    schedule_epochs = args.schedule_epochs
    total_steps = (
        schedule_epochs + args.onecycle_extra_epochs
    ) * len(train_loader)

    pct_start = (
        args.onecycle_pct_start
        if args.onecycle_pct_start is not None
        else 2.0 / schedule_epochs
    )

    if total_steps < 2:
        raise ValueError("OneCycleLR requires more training steps")

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        final_div_factor=args.onecycle_final_div_factor,
        pct_start=pct_start,
        last_epoch=-1,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    metadata = {
        "model": "Neuro3D official-component static-only ablation",
        "protocol": args.protocol,
        "train_objects": train_objects,
        "val_objects": val_objects,
        "train_eeg_path": train_eeg_path,
        "visual_feature_path": visual_path,
        "dynamic_branch": False,
        "cross_attention": False,
        "additional_eeg_normalization": "none",
        "train_repetitions": "independent",
        "validation_repetitions": (
            "mean2" if args.protocol == "development" else None
        ),
        "test_repetitions": (
            "mean4" if args.protocol == "final" else None
        ),
        "checkpoint_monitor": (
            "validation_shape_ce"
            if args.protocol == "development"
            else "none_fixed_epoch"
        ),
        "parameters": {
            "total": num_params,
            "trainable": num_trainable,
        },
    }

    write_json(out / "config.json", vars(args))
    write_json(out / "metadata.json", metadata)

    print(
        f"[data] protocol={args.protocol} "
        f"train_objects={train_objects} "
        f"val_objects={val_objects} "
        f"train_samples={len(train_dataset)} "
        f"eval_samples={len(eval_dataset)} "
        f"extra_norm=none device={device}",
        flush=True,
    )
    print(
        "[model] official static_attention/static_linear/ConvBlock/"
        "linear_projection/temporal_aggregation + "
        "geometry/appearance heads; dynamic fusion disabled",
        flush=True,
    )
    print(
        f"[params] total={num_params:,} trainable={num_trainable:,}",
        flush=True,
    )
    print(
        f"[loss] alpha={args.alpha} => "
        f"{args.alpha * 10:.3f}*regress_sum + "
        f"{(1-args.alpha)*10:.3f}*contrastive_sum + "
        f"0.1*(shape_CE+weighted_color_CE)",
        flush=True,
    )
    print(
        f"[schedule] epochs={args.epochs} "
        f"horizon={schedule_epochs} "
        f"batches/epoch={len(train_loader)} "
        f"total_steps={total_steps} "
        f"pct_start={pct_start:.6f}",
        flush=True,
    )

    history = []
    best_state = None
    best_shape_ce = float("inf")
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        train_stats = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            color_weight_tensor,
            args.alpha,
            args.grad_clip,
        )

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total": train_stats["total"],
            "train_shape_ce": train_stats["shape_ce"],
            "train_color_ce": train_stats["color_ce"],
            "train_regress": train_stats["regress"],
            "train_contrastive": train_stats["contrastive"],
            "train_shape_top1": train_stats["shape_top1"],
            "train_shape_top5": train_stats["shape_top5"],
            "train_color_top1": train_stats["color_top1"],
            "train_color_top2": train_stats["color_top2"],
        }

        if args.protocol == "development":
            eval_stats = evaluate(
                model,
                eval_loader,
                device,
                color_weight_tensor,
                args.alpha,
            )
            row.update(
                {
                    "val_total": eval_stats["total"],
                    "val_shape_ce": eval_stats["shape_ce"],
                    "val_color_ce": eval_stats["color_ce"],
                    "val_regress": eval_stats["regress"],
                    "val_contrastive": eval_stats["contrastive"],
                    "val_shape_top1": eval_stats["shape_top1"],
                    "val_shape_top5": eval_stats["shape_top5"],
                    "val_color_top1": eval_stats["color_top1"],
                    "val_color_top2": eval_stats["color_top2"],
                }
            )

            # Selection criterion requested for clean development:
            # 72-category classifier CE only.
            if eval_stats["shape_ce"] < best_shape_ce - args.min_delta:
                best_shape_ce = eval_stats["shape_ce"]
                best_epoch = epoch
                best_state = clone_state(model)
                stale = 0
                torch.save(
                    {
                        "model": best_state,
                        "epoch": epoch,
                        "val_stats": eval_stats,
                        "args": vars(args),
                        "metadata": metadata,
                    },
                    out / "best.pt",
                )
            else:
                stale += 1

        history.append(row)

        if (
            epoch == 1
            or epoch % args.print_every == 0
            or epoch == args.epochs
        ):
            msg = (
                f"[EPOCH {epoch:03d}] "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"train_shape_top1={train_stats['shape_top1']:.4%} "
                f"train_shape_top5={train_stats['shape_top5']:.4%} "
                f"shape_ce={train_stats['shape_ce']:.4f} "
                f"color_ce={train_stats['color_ce']:.4f} "
                f"reg={train_stats['regress']:.4f} "
                f"con={train_stats['contrastive']:.4f}"
            )
            if args.protocol == "development":
                msg += (
                    f" val_shape_top1={eval_stats['shape_top1']:.4%}"
                    f" val_shape_top5={eval_stats['shape_top5']:.4%}"
                    f" val_shape_ce={eval_stats['shape_ce']:.4f}"
                    f" val_color_top1={eval_stats['color_top1']:.4%}"
                    f" stale={stale}/{args.patience}"
                )
            print(msg, flush=True)

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
        ):
            save_checkpoint(
                out / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                args,
                epoch,
                history,
                metadata,
            )

        if (
            args.protocol == "development"
            and args.patience > 0
            and stale >= args.patience
        ):
            print(
                f"[EARLY-STOP] epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"monitor=val_shape_ce",
                flush=True,
            )
            break

    completed_epoch = epoch

    # Save true last state before loading development best.
    save_checkpoint(
        out / "last.pt",
        model,
        optimizer,
        scheduler,
        args,
        completed_epoch,
        history,
        metadata,
    )
    write_csv(out / "train_log.csv", history)

    if args.protocol == "development":
        if best_state is None:
            raise RuntimeError("No development checkpoint was selected")

        model.load_state_dict(best_state, strict=True)

        final_eval = evaluate(
            model,
            eval_loader,
            device,
            color_weight_tensor,
            args.alpha,
            predictions_path=out / "validation_predictions.npz",
        )
        train_eval = evaluate(
            model,
            train_eval_loader,
            device,
            color_weight_tensor,
            args.alpha,
        )

        print(
            f"[VAL best] epoch={best_epoch} "
            f"shape_top1={final_eval['shape_top1']:.4%} "
            f"shape_top5={final_eval['shape_top5']:.4%} "
            f"shape_ce={final_eval['shape_ce']:.4f} "
            f"color_top1={final_eval['color_top1']:.4%} "
            f"color_top2={final_eval['color_top2']:.4%}",
            flush=True,
        )
        print(
            f"[TRAIN eval] "
            f"shape_top1={train_eval['shape_top1']:.4%} "
            f"shape_top5={train_eval['shape_top5']:.4%} "
            f"color_top1={train_eval['color_top1']:.4%}",
            flush=True,
        )

        summary = {
            "metadata": metadata,
            "epochs_completed": completed_epoch,
            "selected_epoch": best_epoch,
            "selected_checkpoint": str(out / "best.pt"),
            "validation": final_eval,
            "train_eval": train_eval,
            "seconds": time.time() - started,
        }

    else:
        # Fixed-epoch final model, official test evaluated once.
        final_eval = evaluate(
            model,
            eval_loader,
            device,
            color_weight_tensor,
            args.alpha,
            predictions_path=out / "test_predictions.npz",
        )
        train_eval = evaluate(
            model,
            train_eval_loader,
            device,
            color_weight_tensor,
            args.alpha,
        )

        print(
            f"[TEST final] "
            f"shape_top1={final_eval['shape_top1']:.4%} "
            f"shape_top5={final_eval['shape_top5']:.4%} "
            f"shape_ce={final_eval['shape_ce']:.4f} "
            f"color_top1={final_eval['color_top1']:.4%} "
            f"color_top2={final_eval['color_top2']:.4%}",
            flush=True,
        )

        summary = {
            "metadata": metadata,
            "epochs_completed": completed_epoch,
            "selected_epoch": completed_epoch,
            "selected_checkpoint": str(out / "last.pt"),
            "test": final_eval,
            "train_eval": train_eval,
            "seconds": time.time() - started,
        }

    write_json(out / "summary.json", summary)
    print(f"[done] outputs={out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--data_path",
        default="/data/jionkim/neuro_3D",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)

    p.add_argument(
        "--protocol",
        choices=["development", "final"],
        default="development",
    )
    p.add_argument(
        "--val_objects",
        default="06,07",
        help="Used only for development protocol.",
    )

    p.add_argument(
        "--visual_feature_path",
        default=None,
        help="Default: <data_path>/clip_feature.pth",
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Development default=200. "
            "Final requires an explicit fixed epoch count."
        ),
    )
    p.add_argument(
        "--schedule_epochs",
        type=int,
        default=200,
        help="OneCycle horizon; default matches public 200-epoch setup.",
    )
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--test_batch_size", type=int, default=72)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.999)

    p.add_argument(
        "--onecycle_extra_epochs",
        type=int,
        default=5,
    )
    p.add_argument(
        "--onecycle_pct_start",
        type=float,
        default=None,
        help="Default=2/schedule_epochs, matching public training code.",
    )
    p.add_argument(
        "--onecycle_final_div_factor",
        type=float,
        default=10000.0,
    )

    p.add_argument(
        "--official_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--mean_probability",
        type=float,
        default=0.25,
    )
    p.add_argument(
        "--noise_probability",
        type=float,
        default=0.60,
    )
    p.add_argument(
        "--augmentation_noise_scale",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--drop_last",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Public classification train loader uses drop_last=True.",
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Public objective alpha.",
    )
    p.add_argument("--grad_clip", type=float, default=0.0)

    p.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Development only; 0 disables early stopping.",
    )
    p.add_argument(
        "--min_delta",
        type=float,
        default=1e-3,
    )
    p.add_argument("--save_every", type=int, default=20)
    p.add_argument("--print_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")

    args = p.parse_args()

    if args.epochs is None:
        if args.protocol == "development":
            args.epochs = 200
        else:
            p.error(
                "--protocol final requires explicit --epochs "
                "selected during development"
            )

    if args.epochs <= 0 or args.schedule_epochs <= 0:
        p.error("epoch counts must be positive")
    if args.epochs > args.schedule_epochs:
        p.error("--epochs cannot exceed --schedule_epochs")
    if args.batch_size <= 0 or args.test_batch_size <= 0:
        p.error("batch sizes must be positive")
    if args.lr <= 0:
        p.error("--lr must be positive")
    if args.weight_decay < 0:
        p.error("--weight_decay must be nonnegative")
    if not (0.0 <= args.alpha <= 1.0):
        p.error("--alpha must be in [0,1]")
    if not (0.0 <= args.mean_probability <= 1.0):
        p.error("--mean_probability must be in [0,1]")
    if not (0.0 <= args.noise_probability <= 1.0):
        p.error("--noise_probability must be in [0,1]")
    if args.augmentation_noise_scale < 0:
        p.error("--augmentation_noise_scale must be nonnegative")
    if args.patience < 0 or args.min_delta < 0:
        p.error("patience/min_delta must be nonnegative")

    train(args)


if __name__ == "__main__":
    main()
