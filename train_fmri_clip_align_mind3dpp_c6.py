#!/usr/bin/env python3
"""
Train an fMRI -> CLIP text aligner on the MinD-3D++ / fMRI-Objaverse dataset.

Semantic-Visual Dual Alignment (E3) on top of the E1-B objective.
Preserves original fMRI frame CNN/temporal encoder, semantic-text forward, split,
and checkpoint['model'] key compatibility with inference_fmri_mind3dpp.py.
Adds a visual head supervised by frozen multi-view GT CLIP image embeddings.
Optional coarse 3D shape histogram regression uses TRAIN GT GLBs only for training.
Validation GT teachers are used solely for diagnostic metrics, never gradients.

Loss:
  batch_nce_weight * symmetric_batch_NCE
  + full_ce_weight * train_UID_full_object_CE
  + category_ce_weight * train_category_CE

Checkpoints (always saved):
  fmri_clip_aligner_best_objce.pt : minimum validation full-object CE
  fmri_clip_aligner_best_10way.pt : maximum validation 10-way accuracy
  fmri_clip_aligner_best.pt       : legacy inference alias (CE by default)
  fmri_clip_aligner_last.pt       : final epoch

For complete 100-epoch curves, set --patience 0. Official test stays disabled.


Expected fMRI-Objaverse root
----------------------------
<data_path>/
  annotations/
  stimuli/
  sub-0001/
    <uid>.npy
    ...
  sub-0006/
  sub-0007/
  sub-0008/
  sub-0015/
  Object_caption_Cap3D.csv
  train_list.txt
  test_list.txt

The baseline fMRI-Objaverse loader stores each object's fMRI as
<subject>/<uid>.npy. C6 adds a train-only sampling ablation (random/center/uniform).
Evaluation ALWAYS uses the middle 6 frames to match legacy E3 inference.

The GLB files are not required for CLIP-text alignment. If --gt_mesh_root is
provided, the script only validates that the object path from train/test list
maps to an existing GLB under:
  <gt_mesh_root>/<folder>/<uid>.glb

Text supervision
----------------
Default: Cap3D captions from Object_caption_Cap3D.csv.
Fallback: category prompt from train/test list when a caption is unavailable.

Model
-----
Because fMRI samples are 2D spatial frames rather than EEG channel x time
signals, the EEG mean/std MLP is replaced by:
  per-frame 2D CNN -> temporal Transformer -> MLP projection -> CLIP space.

The frozen OpenCLIP text encoder supplies target embeddings.

Multi-subject training
----------------------
If several subjects are used, the same object/caption can appear multiple
times in one batch. A multi-positive symmetric contrastive loss is used so
same-object samples are positives rather than false negatives.

Validation
----------
By default, 10% of train-list OBJECT IDs are held out for validation. The
split is made by UID before subject expansion, preventing the same object from
appearing in both train and validation through different subjects.

Checkpoint selection uses FULL-validation object-retrieval cross-entropy:
every validation fMRI embedding is compared against every validation object
caption candidate at once. This replaces the earlier mini-batch validation
contrastive-loss criterion.

The official test_list.txt is disabled by default and can be evaluated once
with --eval_test after the validation rule/model has been fixed.

Semantic N-way evaluation
-------------------------
In addition to full 271-way object retrieval, the script reports deterministic
expected 2-way and 10-way Top-1 object-identification accuracy. Each N-way task
contains the correct UID-caption plus N-1 distractor objects sampled uniformly
without replacement. The exact expectation over all possible distractor sets is
computed, avoiding Monte-Carlo variance.

Note: MinD-3D++ reports N-way semantic accuracy by comparing reconstructed
rendered images against GT rendered images. This script applies the same N-way
candidate-identification logic in the fMRI-to-Cap3D CLIP embedding space, so it
is a representation-level analogue rather than a numerically identical image-
reconstruction metric.
"""

import argparse
import csv
import json
import hashlib
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from PIL import Image



# -----------------------------------------------------------------------------
# Reproducibility / helpers
# -----------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_uid(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem


def pretty_category(value: str) -> str:
    return str(value).strip().replace("_", " ").replace("-", " ")


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Dataset metadata
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectRecord:
    category: str
    object_path: str
    uid: str


def read_object_list(path: Path) -> List[ObjectRecord]:
    """
    Accepts the official two-column text format:
        category   000-000/<uid>.glb

    It also tolerates comma-separated lines.
    """
    if not path.is_file():
        raise FileNotFoundError(path)

    records: List[ObjectRecord] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Parse comma-delimited MinD-3D++ lines before whitespace splitting.
            # E.g. "computer_monitor, 000-035/<uid>.glb" is otherwise parsed
            # as category="computer_monitor," rather than "computer_monitor".
            if "," in line:
                parts = next(csv.reader([line]))
                parts = [p.strip() for p in parts if p.strip()]
            else:
                parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 2 columns "
                    f"(category and object path), got: {line}"
                )

            object_path = parts[-1].strip()
            category = " ".join(parts[:-1]).strip()

            # Ignore a simple header if present.
            if line_no == 1:
                low0 = category.lower()
                low1 = object_path.lower()
                if ("category" in low0) and any(
                    x in low1 for x in ("uid", "path", "object")
                ):
                    continue

            uid = normalize_uid(object_path)
            if not uid:
                raise ValueError(f"{path}:{line_no}: empty UID")

            records.append(
                ObjectRecord(
                    category=category,
                    object_path=object_path.replace("\\", "/"),
                    uid=uid,
                )
            )

    if not records:
        raise ValueError(f"No records found in {path}")

    # An object should occur once in a list.
    seen = set()
    dup = []
    for r in records:
        if r.uid in seen:
            dup.append(r.uid)
        seen.add(r.uid)
    if dup:
        print(
            f"[Warning] {path.name}: {len(dup)} duplicated UIDs; "
            "multi-positive loss will handle duplicates.",
            flush=True,
        )

    return records


def _looks_like_uid(value: str) -> bool:
    s = str(value).strip().replace("\\", "/")
    stem = Path(s).stem
    if re.fullmatch(r"[0-9a-fA-F]{24,40}", stem):
        return True
    if s.lower().endswith((".glb", ".npy")) and len(stem) >= 16:
        return True
    return False


def load_caption_map(csv_path: Path) -> Dict[str, str]:
    """
    Load Object_caption_Cap3D.csv using the observed MinD-3D++ format:

        category,uid,caption

    Captions can contain commas, so each physical line is split at most twice:
        category_raw, uid_raw, caption_raw = line.split(",", 2)

    Example:
        Bible,000-107/adde....,an old brown book with gold trim, handle, and clasps.

    becomes:
        category = Bible
        uid      = adde....
        caption  = an old brown book with gold trim, handle, and clasps.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    mapping: Dict[str, str] = {}
    malformed = []
    duplicate_uids = 0

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split(",", 2)
            if len(parts) != 3:
                malformed.append(
                    (line_no, "expected category,uid,caption", line[:180])
                )
                continue

            category_raw, uid_raw, caption_raw = parts

            category = category_raw.strip().strip('"').strip("'")
            uid_field = uid_raw.strip().strip('"').strip("'")
            caption = caption_raw.strip()

            # Header:
            # category,uid,caption
            if (
                category.lower() == "category"
                and uid_field.lower() == "uid"
            ):
                continue

            uid = normalize_uid(uid_field)

            # Captions may still be wrapped in CSV-style double quotes.
            if len(caption) >= 2 and caption[0] == '"' and caption[-1] == '"':
                caption = caption[1:-1].replace('""', '"').strip()

            if not uid:
                malformed.append(
                    (line_no, "empty UID", line[:180])
                )
                continue
            if not caption:
                malformed.append(
                    (line_no, "empty caption", line[:180])
                )
                continue

            if not _looks_like_uid(uid):
                malformed.append(
                    (line_no, f"suspicious UID '{uid}'", line[:180])
                )

            if uid in mapping:
                duplicate_uids += 1

            mapping[uid] = caption

    if not mapping:
        preview = "\n".join(
            f"line {n}: {reason}: {content}"
            for n, reason, content in malformed[:10]
        )
        raise ValueError(
            f"No UID-caption pairs could be parsed from {csv_path}."
            + (f"\nExamples:\n{preview}" if preview else "")
        )

    print(
        f"[*] Caption CSV: {len(mapping)} UID captions "
        f"(format=category,uid,caption; split(',', 2))",
        flush=True,
    )

    if duplicate_uids:
        print(
            f"[Warning] Caption CSV contained {duplicate_uids} duplicate UID rows; "
            f"the last caption for each duplicated UID was kept.",
            flush=True,
        )

    if malformed:
        print(
            f"[Warning] Caption CSV contained {len(malformed)} "
            f"malformed/suspicious rows. First examples:",
            flush=True,
        )
        for line_no, reason, content in malformed[:5]:
            print(
                f"  line {line_no}: {reason}: {content}",
                flush=True,
            )

    return mapping

def build_texts(
    records: Sequence[ObjectRecord],
    caption_map: Dict[str, str],
    text_source: str,
) -> Dict[str, str]:
    result: Dict[str, str] = {}
    missing = 0

    for r in records:
        if text_source == "caption":
            text = caption_map.get(r.uid)
            if not text:
                missing += 1
                text = f"a 3D object of {pretty_category(r.category)}"
        elif text_source == "category":
            text = f"a 3D object of {pretty_category(r.category)}"
        else:
            raise ValueError(text_source)

        result[r.uid] = text

    if missing:
        print(
            f"[Warning] {missing} objects had no caption; "
            "category prompts were used as fallback.",
            flush=True,
        )
    return result


# -----------------------------------------------------------------------------
# fMRI dataset
# -----------------------------------------------------------------------------

class FMRIObjaverseDataset(Dataset):
    """
    Each item = one (subject, object) pair.

    train_mode=True:
      C6-configurable random / center / uniform selection of num_frames.

    train_mode=False:
      use the middle num_frames, matching the official evaluation convention.
    """

    def __init__(
        self,
        data_root: Path,
        subjects: Sequence[str],
        records: Sequence[ObjectRecord],
        text_embeddings: Dict[str, torch.Tensor],
        category_to_index: Dict[str, int],
        train_mode: bool,
        num_frames: int = 6,
        fmri_norm: str = "none",
        train_frame_sampling: str = "random",
        strict_files: bool = True,
        visual_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        geometry_targets: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.visual_embeddings = visual_embeddings
        self.geometry_targets = geometry_targets
        self.data_root = Path(data_root)
        self.subjects = list(subjects)
        self.records = list(records)
        self.text_embeddings = text_embeddings
        self.category_to_index = category_to_index
        self.train_mode = bool(train_mode)
        self.num_frames = int(num_frames)
        self.fmri_norm = fmri_norm
        self.train_frame_sampling = str(train_frame_sampling)
        if self.train_frame_sampling not in ("random", "center", "uniform"):
            raise ValueError("Unknown train_frame_sampling: " + self.train_frame_sampling)

        index = []
        missing = []

        for subject_idx, subject in enumerate(self.subjects):
            subject_dir = self.data_root / subject
            if not subject_dir.is_dir():
                raise FileNotFoundError(subject_dir)

            for record_idx, record in enumerate(self.records):
                fmri_path = subject_dir / f"{record.uid}.npy"
                if fmri_path.is_file():
                    index.append((subject_idx, record_idx, fmri_path))
                else:
                    missing.append(str(fmri_path))

        if missing:
            msg = (
                f"{len(missing)} fMRI files are missing for the requested "
                f"subjects/objects."
            )
            if strict_files:
                preview = "\n".join(missing[:10])
                raise FileNotFoundError(f"{msg}\nFirst missing files:\n{preview}")
            print(f"[Warning] {msg} They will be skipped.", flush=True)

        if not index:
            raise ValueError("No usable fMRI samples found")

        self.index = index

    def __len__(self):
        return len(self.index)

    def _select_frames(self, fmri: np.ndarray) -> np.ndarray:
        if fmri.ndim == 4 and fmri.shape[1] == 1:
            fmri = fmri[:, 0]

        if fmri.ndim != 3:
            raise ValueError(
                f"Expected fMRI shape [frames,H,W], got {fmri.shape}"
            )

        n = fmri.shape[0]
        if n < self.num_frames:
            raise ValueError(
                f"Need >= {self.num_frames} fMRI frames, got {n}"
            )

        # Only the training sampler varies in the C6 ablation.
        # Validation / test / deterministic train diagnostics always use center.
        sampling = self.train_frame_sampling if self.train_mode else "center"
        if sampling == "random":
            ids = np.random.choice(n, size=self.num_frames, replace=False)
            ids.sort()  # Preserve the chronological ordering (original E3 behavior).
        elif sampling == "center":
            start = (n - self.num_frames) // 2
            ids = np.arange(start, start + self.num_frames)
        elif sampling == "uniform":
            ids = np.rint(np.linspace(0, n - 1, self.num_frames)).astype(np.int64)
            if len(np.unique(ids)) != self.num_frames:
                raise RuntimeError(f"Uniform frame sampling returned duplicates: {ids}")
        else:
            raise AssertionError(sampling)
        return fmri[ids]

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.fmri_norm == "none":
            return x

        if self.fmri_norm == "sample_zscore":
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            return (x - mean) / std

        if self.fmri_norm == "frame_zscore":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return (x - mean) / std

        raise ValueError(self.fmri_norm)

    def __getitem__(self, idx):
        subject_idx, record_idx, fmri_path = self.index[idx]
        record = self.records[record_idx]

        fmri = np.load(fmri_path)
        fmri = self._select_frames(np.asarray(fmri))
        fmri = torch.from_numpy(
            np.asarray(fmri, dtype=np.float32).copy()
        )
        fmri = self._normalize(fmri)

        output = {
            "fmri": fmri,  # [F,H,W]
            "text_embedding": self.text_embeddings[record.uid].clone(),
            "uid": record.uid,
            "category": record.category,
            "category_index": torch.tensor(
                self.category_to_index[record.category], dtype=torch.long
            ),
            "subject_index": torch.tensor(subject_idx, dtype=torch.long),
            "record_index": torch.tensor(record_idx, dtype=torch.long),
        }
        if self.visual_embeddings is not None:
            output["visual_embedding"] = self.visual_embeddings[record.uid]
        if self.geometry_targets is not None:
            output["geometry_target"] = self.geometry_targets[record.uid]
        return output


# -----------------------------------------------------------------------------
# fMRI -> CLIP model
# -----------------------------------------------------------------------------

class FrameCNN(nn.Module):
    def __init__(self, out_dim=256, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),

            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(self.features(x))


class FMRIClipAligner(nn.Module):
    def __init__(
        self,
        clip_dim: int,
        num_frames: int = 6,
        frame_dim: int = 256,
        hidden_dim: int = 1024,
        temporal_layers: int = 1,
        temporal_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        if frame_dim % temporal_heads != 0:
            raise ValueError("frame_dim must be divisible by temporal_heads")

        self.num_frames = num_frames
        self.frame_encoder = FrameCNN(
            out_dim=frame_dim,
            dropout=dropout,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames, frame_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim,
            nhead=temporal_heads,
            dim_feedforward=frame_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=temporal_layers,
        )

        self.pool_query = nn.Parameter(
            torch.randn(1, 1, frame_dim) * 0.02
        )
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=frame_dim,
            num_heads=temporal_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.net = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * math.log(1.0 / 0.07)
        )

    def encode_pooled(self, fmri):
        # fmri: [B,F,H,W]
        if fmri.ndim != 4:
            raise ValueError(
                f"Expected fMRI [B,F,H,W], got {tuple(fmri.shape)}"
            )

        b, f, h, w = fmri.shape
        if f != self.num_frames:
            raise ValueError(
                f"Expected {self.num_frames} frames, got {f}"
            )

        x = fmri.reshape(b * f, 1, h, w)
        x = self.frame_encoder(x)
        x = x.reshape(b, f, -1)

        x = x + self.pos_embed[:, :f]
        x = self.temporal_encoder(x)

        q = self.pool_query.expand(b, -1, -1)
        pooled, _ = self.pool_attn(q, x, x, need_weights=False)
        pooled = pooled[:, 0]

        return pooled

    def forward(self, fmri):
        # Identical semantic embedding and state_dict keys to E1-B.
        return F.normalize(self.net(self.encode_pooled(fmri)), dim=-1)


class DualFMRIClipAligner(FMRIClipAligner):
    """Shared fMRI features with separate semantic and visual predictions.

    Inference forward() remains the original semantic CLIP embedding.
    The visual and optional geometry heads are for auxiliary supervision.
    """
    def __init__(self, *args, visual_hidden_dim=512, geometry_dim=0, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.pos_embed.shape[-1]
        clip_dim = self.net[-1].out_features
        self.visual_net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, visual_hidden_dim),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.2)),
            nn.Linear(visual_hidden_dim, clip_dim),
        )
        self.visual_logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))
        self.geometry_head = (
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                          nn.Linear(256, geometry_dim), nn.Sigmoid())
            if geometry_dim > 0 else None
        )

    def forward_dual(self, fmri):
        pooled = self.encode_pooled(fmri)
        z_text = F.normalize(self.net(pooled), dim=-1)
        z_visual = F.normalize(self.visual_net(pooled), dim=-1)
        z_geometry = self.geometry_head(pooled) if self.geometry_head is not None else None
        return z_text, z_visual, z_geometry


def semantic_only_state_dict(model):
    """Exact legacy FMRIClipAligner keys for inference_fmri_mind3dpp.py."""
    return {k: v for k, v in model.state_dict().items()
            if not (k.startswith("visual_net.") or k.startswith("geometry_head.")
                    or k == "visual_logit_scale")}


def checkpoint_state(model):
    return {"model": semantic_only_state_dict(model),
            "dual_model": model.state_dict()}


# -----------------------------------------------------------------------------
# Text encoding
# -----------------------------------------------------------------------------

@torch.no_grad()
def encode_text_table(
    texts: Dict[str, str],
    clip_model,
    clip_tokenizer,
    device,
    batch_size=256,
) -> Dict[str, torch.Tensor]:
    keys = list(texts.keys())
    values = [texts[k] for k in keys]

    result = {}

    for start in tqdm(
        range(0, len(keys), batch_size),
        desc="Encoding captions",
    ):
        end = min(start + batch_size, len(keys))
        tokens = clip_tokenizer(values[start:end]).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1).cpu()

        for k, e in zip(keys[start:end], emb):
            result[k] = e.clone()

    return result


@torch.no_grad()
def encode_category_embeddings(
    categories: Sequence[str],
    clip_model,
    clip_tokenizer,
    device,
):
    prompts = [
        f"a 3D object of {pretty_category(c)}"
        for c in categories
    ]
    tokens = clip_tokenizer(prompts).to(device)
    features = clip_model.encode_text(tokens)
    return F.normalize(features.float(), dim=-1).cpu()


# -----------------------------------------------------------------------------
# Contrastive objective
# -----------------------------------------------------------------------------

def build_positive_mask(uids: Sequence[str], device) -> torch.Tensor:
    # Same UID across subjects = positive, not negative.
    uid_to_int = {}
    ids = []
    for uid in uids:
        if uid not in uid_to_int:
            uid_to_int[uid] = len(uid_to_int)
        ids.append(uid_to_int[uid])

    ids = torch.tensor(ids, device=device)
    return ids[:, None].eq(ids[None, :])


def multi_positive_nce(logits, positive_mask):
    """
    -log( sum exp(positive logits) / sum exp(all logits) )
    averaged over rows.
    """
    neg_inf = torch.finfo(logits.dtype).min
    pos_logits = logits.masked_fill(~positive_mask, neg_inf)
    numerator = torch.logsumexp(pos_logits, dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return -(numerator - denominator).mean()


def symmetric_multi_positive_clip_loss(
    brain_embeddings,
    text_embeddings,
    uids: Sequence[str],
    logit_scale,
):
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * brain_embeddings @ text_embeddings.T

    positive_mask = build_positive_mask(
        uids, brain_embeddings.device
    )

    loss_b2t = multi_positive_nce(
        logits, positive_mask
    )
    loss_t2b = multi_positive_nce(
        logits.T, positive_mask.T
    )

    with torch.no_grad():
        pred = logits.argmax(dim=1)
        top1_positive = positive_mask[
            torch.arange(logits.shape[0], device=logits.device),
            pred,
        ]
        retrieval_acc = top1_positive.float().mean()

    return 0.5 * (loss_b2t + loss_t2b), retrieval_acc


# -----------------------------------------------------------------------------
# E1: full-training-candidate object CE + category CE
# -----------------------------------------------------------------------------

def prepare_e1_candidate_banks(
    train_records: Sequence[ObjectRecord],
    text_embeddings: Dict[str, torch.Tensor],
    category_names: Sequence[str],
    category_embeddings: torch.Tensor,
    device,
):
    """Build frozen banks from TRAIN records only (never val/test objects).

    Repeated objects across subjects share the same positive UID index.
    Category negatives are also restricted to categories represented in TRAIN.
    """
    unique_uids = list(dict.fromkeys(r.uid for r in train_records))
    if not unique_uids:
        raise ValueError("E1 training object bank is empty")
    uid_to_index = {uid: i for i, uid in enumerate(unique_uids)}
    object_bank = torch.stack(
        [text_embeddings[uid] for uid in unique_uids], dim=0
    ).float().to(device)
    object_bank = F.normalize(object_bank, dim=-1).detach()

    train_category_names = sorted({r.category for r in train_records})
    train_category_to_index = {
        name: i for i, name in enumerate(train_category_names)
    }
    global_category_to_index = {
        name: i for i, name in enumerate(category_names)
    }
    category_indices = [
        global_category_to_index[name] for name in train_category_names
    ]
    category_bank = category_embeddings[category_indices].float().to(device)
    category_bank = F.normalize(category_bank, dim=-1).detach()
    return (
        object_bank,
        uid_to_index,
        category_bank,
        train_category_to_index,
    )


def full_candidate_object_category_loss(
    brain_embeddings: torch.Tensor,
    uids: Sequence[str],
    categories: Sequence[str],
    object_bank: torch.Tensor,
    uid_to_index: Dict[str, int],
    category_bank: torch.Tensor,
    category_to_index: Dict[str, int],
    logit_scale: torch.Tensor,
    category_loss_weight: float = 0.2,
):
    """Object CE over ALL unique train UIDs + weighted TRAIN-category CE.

    The full candidate table is fixed/frozen. Gradients flow through the fMRI
    encoder and logit_scale only. Validation and test UIDs are never negatives
    in this training loss.
    """
    device = brain_embeddings.device
    targets_obj = torch.tensor(
        [uid_to_index[uid] for uid in uids],
        dtype=torch.long, device=device,
    )
    targets_cat = torch.tensor(
        [category_to_index[c] for c in categories],
        dtype=torch.long, device=device,
    )
    scale = logit_scale.exp().clamp(max=100.0)
    logits_obj = scale * (brain_embeddings.float() @ object_bank.T)
    logits_cat = scale * (brain_embeddings.float() @ category_bank.T)
    loss_obj = F.cross_entropy(logits_obj, targets_obj)
    loss_cat = F.cross_entropy(logits_cat, targets_cat)
    loss = loss_obj + float(category_loss_weight) * loss_cat

    with torch.no_grad():
        acc_obj = logits_obj.argmax(dim=1).eq(targets_obj).float().mean()
        acc_cat = logits_cat.argmax(dim=1).eq(targets_cat).float().mean()
    return loss, loss_obj, loss_cat, acc_obj, acc_cat


@torch.no_grad()
def batch_positive_retrieval_accuracy(
    brain_embeddings: torch.Tensor,
    target_text: torch.Tensor,
    uids: Sequence[str],
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Keep Train_Batch_Retrieval_Acc comparable with the old baseline."""
    logits = logit_scale.exp().clamp(max=100.0) * (brain_embeddings @ target_text.T)
    positive_mask = build_positive_mask(uids, brain_embeddings.device)
    predicted = logits.argmax(dim=1)
    return positive_mask[
        torch.arange(len(uids), device=brain_embeddings.device), predicted
    ].float().mean()


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
def extract_fmri_embeddings(model, loader, device):
    model.eval()

    embeddings = []
    uids = []
    categories = []

    for batch in tqdm(loader, desc="Embedding fMRI", leave=False):
        fmri = batch["fmri"].to(
            device, non_blocking=True
        ).float()
        emb = model(fmri).cpu()

        embeddings.append(emb)
        uids.extend(list(batch["uid"]))
        categories.extend(list(batch["category"]))

    return torch.cat(embeddings, dim=0), uids, categories


def topk_accuracy(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    ks=(1, 5),
):
    result = {}
    for k in ks:
        k_eff = min(k, similarities.shape[1])
        pred = similarities.topk(
            k_eff, dim=1
        ).indices
        correct = pred.eq(targets[:, None]).any(dim=1)
        result[k] = float(correct.float().mean().item())
    return result

def expected_nway_top1(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    n_way: int,
) -> float:
    """
    Deterministic expected N-way Top-1 accuracy.

    For each query, the candidate set contains:
      - the ground-truth object
      - n_way - 1 distractors sampled uniformly without replacement
        from all other candidates.

    Rather than Monte-Carlo sampling many candidate sets, this function
    computes the exact expectation over all possible distractor sets.

    Ties are handled by random tie-breaking in expectation:
    if j sampled distractors tie the positive score for the maximum,
    the positive receives probability 1/(j+1).

    This is the embedding-space counterpart of the standard N-way
    identification protocol. For MinD-3D++, the published semantic metric
    is applied to reconstructed-vs-GT rendered images; here it is applied
    to fMRI embeddings vs object-specific Cap3D text embeddings.
    """
    if similarities.ndim != 2:
        raise ValueError(
            f"similarities must be [N,C], got {tuple(similarities.shape)}"
        )

    num_queries, num_candidates = similarities.shape
    if not (2 <= n_way <= num_candidates):
        raise ValueError(
            f"n_way must be in [2,{num_candidates}], got {n_way}"
        )

    sims = similarities.detach().cpu().double()
    targets = targets.detach().cpu().long()
    k = n_way - 1
    denom = math.comb(num_candidates - 1, k)

    probs = []

    for i in range(num_queries):
        target_idx = int(targets[i].item())
        pos = float(sims[i, target_idx].item())

        row = sims[i]
        other_mask = torch.ones(
            num_candidates, dtype=torch.bool
        )
        other_mask[target_idx] = False
        others = row[other_mask]

        # Use a very small tolerance only for numerical equality.
        tol = 1e-12
        higher = int((others > pos + tol).sum().item())
        equal = int((torch.abs(others - pos) <= tol).sum().item())
        lower = (num_candidates - 1) - higher - equal

        # To win, no strictly-higher distractor may be sampled.
        # If j equal-score distractors are sampled, expected tie-break
        # probability is 1/(j+1).
        p_win_num = 0.0

        j_min = max(0, k - lower)
        j_max = min(equal, k)

        for j in range(j_min, j_max + 1):
            remaining = k - j
            if remaining > lower:
                continue

            ways = math.comb(equal, j) * math.comb(lower, remaining)
            p_win_num += ways / (j + 1)

        probs.append(p_win_num / denom)

    return float(np.mean(probs))


@torch.no_grad()
def evaluate_retrieval(
    model,
    loader,
    records: Sequence[ObjectRecord],
    text_embeddings: Dict[str, torch.Tensor],
    category_names: Sequence[str],
    category_embeddings: torch.Tensor,
    device,
):
    """
    Full-candidate retrieval evaluation.

    Checkpoint-selection quantity:
        object_ce = CE(
            scale * brain_embeddings @ all_split_object_text_embeddings.T,
            true_object_index
        )

    This is computed over ALL candidate objects in the split at once, rather
    than averaging contrastive losses from small validation mini-batches.

    Category CE is also reported over the full category vocabulary but is not
    used for checkpoint selection by default.
    """
    brain, sample_uids, sample_categories = extract_fmri_embeddings(
        model, loader, device
    )

    # -------------------------------------------------------------
    # Object-caption retrieval: unique object candidates in this split.
    # -------------------------------------------------------------
    candidate_uids = []
    seen = set()
    for r in records:
        if r.uid not in seen:
            seen.add(r.uid)
            candidate_uids.append(r.uid)

    candidate_text = torch.stack(
        [text_embeddings[u] for u in candidate_uids],
        dim=0,
    )
    candidate_text = F.normalize(candidate_text.float(), dim=-1)

    uid_to_idx = {
        uid: i for i, uid in enumerate(candidate_uids)
    }
    object_targets = torch.tensor(
        [uid_to_idx[u] for u in sample_uids],
        dtype=torch.long,
    )

    # model() already returns normalized brain embeddings.
    brain = F.normalize(brain.float(), dim=-1)

    scale = float(
        model.logit_scale.detach()
        .exp()
        .clamp(max=100.0)
        .cpu()
        .item()
    )

    object_logits = scale * (brain @ candidate_text.T)
    object_ce = float(
        F.cross_entropy(
            object_logits,
            object_targets,
        ).item()
    )
    object_acc = topk_accuracy(
        object_logits,
        object_targets,
        ks=(1, 5),
    )

    # MinD-3D++-style N-way semantic identification counterpart.
    # Each query is compared with its correct object plus randomly
    # selected distractor objects. We compute the exact expected
    # accuracy over all possible distractor sets.
    object_2way_top1 = expected_nway_top1(
        object_logits,
        object_targets,
        n_way=2,
    )
    object_10way_top1 = expected_nway_top1(
        object_logits,
        object_targets,
        n_way=10,
    )

    # -------------------------------------------------------------
    # Category retrieval over the full category vocabulary.
    # -------------------------------------------------------------
    cat_to_idx = {
        c: i for i, c in enumerate(category_names)
    }
    category_targets = torch.tensor(
        [cat_to_idx[c] for c in sample_categories],
        dtype=torch.long,
    )

    category_text = F.normalize(
        category_embeddings.float(),
        dim=-1,
    )
    category_logits = scale * (brain @ category_text.T)
    category_ce = float(
        F.cross_entropy(
            category_logits,
            category_targets,
        ).item()
    )
    category_acc = topk_accuracy(
        category_logits,
        category_targets,
        ks=(1, 5),
    )

    return {
        "object_ce": object_ce,
        "category_ce": category_ce,
        "object_2way_top1": object_2way_top1,
        "object_10way_top1": object_10way_top1,
        "object_top1": object_acc[1],
        "object_top5": object_acc[5],
        "category_top1": category_acc[1],
        "category_top5": category_acc[5],
        "num_samples": len(sample_uids),
        "num_object_candidates": len(candidate_uids),
        "num_category_candidates": len(category_names),
        "logit_scale": scale,
    }

# -----------------------------------------------------------------------------
# Frozen GT multi-view and optional coarse 3D-shape teachers
# -----------------------------------------------------------------------------

VISUAL_CACHE_VERSION = "multi_view_CLIP_mean_v1"
GEOMETRY_CACHE_VERSION = "rot_invariant_surface_hist_v1"
GEOMETRY_DIM = 30  # radial hist 12 + pair hist 12 + eigenvalues 3 + bbox ratios 3


def _unique_records(*groups):
    records = {}
    for g in groups:
        for r in g:
            if r.uid in records and records[r.uid].object_path != r.object_path:
                raise ValueError(f"Conflicting object_path for UID {r.uid}")
            records[r.uid] = r
    return records


def _load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _checked_cache(path, metadata, uid_list):
    if not path.is_file():
        return None
    obj = _load_cache(path)
    if not isinstance(obj, dict) or obj.get("metadata") != metadata:
        raise ValueError(
            f"Cached teacher features at {path} have different settings. "
            "Use a new --visual_cache_path / --geometry_cache_path or delete the old cache."
        )
    features = obj.get("embeddings")
    if not isinstance(features, dict):
        raise ValueError(f"Malformed teacher cache: {path}")
    missing = [u for u in uid_list if u not in features]
    if missing:
        raise ValueError(f"Teacher cache is missing {len(missing)} required UIDs, e.g. {missing[:3]}")
    return {u: features[u].float() for u in uid_list}


@torch.no_grad()
def prepare_multiview_clip_targets(
    records, views_root, clip_model, clip_preprocess, device,
    view_count=6, batch_size=48, cache_path=None,
    clip_name="ViT-B-32", clip_weights="openai",
):
    """Create normalized per-object CLIP image targets from GT rendered views.

    A strictly disjoint held-out validation UID set may be encoded for evaluation
    but gradients use only records from the training split (enforced downstream).
    """
    items = _unique_records(records)
    uid_list = sorted(items)
    views_root = Path(views_root).expanduser().resolve()
    if not views_root.is_dir():
        raise FileNotFoundError(
            f"GT view root does not exist: {views_root}. "
            "Render train_list.txt GT meshes before running E3."
        )
    metadata = {
        "version": VISUAL_CACHE_VERSION, "views_root": str(views_root),
        "clip_model": clip_name, "clip_pretrained": clip_weights,
        "view_count": view_count, "uids": uid_list,
    }
    cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
    if cache_path is not None:
        cached = _checked_cache(cache_path, metadata, uid_list)
        if cached is not None:
            print(f"[*] Loaded GT CLIP multi-view cache: {cache_path} ({len(cached)} UIDs)", flush=True)
            return cached

    files = []
    for uid in uid_list:
        for j in range(view_count):
            files.append((uid, views_root / uid / f"{j:02d}.png"))
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        examples = "\n".join(missing[:6])
        raise FileNotFoundError(
            f"Missing {len(missing)} GT view files for E3 visual targets. "
            f"First missing:\n{examples}\n"
            "The GT renderer must process train_list.txt, not only test_list.txt."
        )

    totals, counts = {}, {}
    clip_model.eval()
    for start in tqdm(range(0, len(files), batch_size), desc="Encoding GT 6-view CLIP"):
        block = files[start:start + batch_size]
        tensors = []
        for _, path in block:
            with Image.open(path) as img:
                tensors.append(clip_preprocess(img.convert("RGB")))
        x = torch.stack(tensors).to(device)
        embeddings = F.normalize(clip_model.encode_image(x).float(), dim=-1).cpu()
        for (uid, _), emb in zip(block, embeddings):
            totals[uid] = totals.get(uid, torch.zeros_like(emb)) + emb
            counts[uid] = counts.get(uid, 0) + 1
    output = {}
    for uid in uid_list:
        if counts.get(uid, 0) != view_count:
            raise RuntimeError(f"Unexpected view count for {uid}")
        output[uid] = F.normalize(totals[uid], dim=-1)
    dim = output[uid_list[0]].numel()
    print(f"[*] Multi-view visual teachers: {len(output)} UIDs, CLIP dimension={dim}", flush=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": metadata, "embeddings": output}, cache_path)
        print(f"[*] Saved GT CLIP cache to {cache_path}", flush=True)
    return output


def _shape_descriptor_from_glb(mesh_path, uid, points_count=2048):
    """Coarse rotation-invariant shape histogram (not a pretrained 3D encoder)."""
    import trimesh
    scene = trimesh.load(str(mesh_path), process=False, force="scene")
    if isinstance(scene, trimesh.Scene):
        if hasattr(scene, "to_mesh"):
            mesh = scene.to_mesh()
        else:
            mesh = scene.dump(concatenate=True)
        if isinstance(mesh, list):
            mesh = trimesh.util.concatenate(mesh)
    else:
        mesh = scene
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) < 3 or len(faces) == 0:
        raise ValueError(f"No valid mesh triangles: {mesh_path}")
    tri = vertices[faces]
    weights = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    weights = np.maximum(weights, 0)
    if not np.isfinite(weights).all() or weights.sum() < 1e-12:
        raise ValueError(f"Degenerate mesh faces: {mesh_path}")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], "little"))
    face_idx = rng.choice(len(faces), size=points_count, replace=True, p=weights/weights.sum())
    v = tri[face_idx]
    u, w = rng.random(points_count), rng.random(points_count)
    a = np.sqrt(u)
    p = (1-a)[:, None]*v[:, 0] + (a*(1-w))[:, None]*v[:, 1] + (a*w)[:, None]*v[:, 2]
    extent = np.ptp(vertices, axis=0)
    bbox_diag = max(float(np.linalg.norm(extent)), 1e-8)
    center = (vertices.min(axis=0) + vertices.max(axis=0))*0.5
    p = (p - center) / bbox_diag
    radial = np.linalg.norm(p, axis=1)
    radial_hist = np.histogram(radial, bins=12, range=(0, 0.75))[0].astype(np.float32)
    radial_hist /= max(radial_hist.sum(), 1.)
    a_idx = rng.integers(0, points_count, size=points_count)
    b_idx = rng.integers(0, points_count, size=points_count)
    pair_dist = np.linalg.norm(p[a_idx] - p[b_idx], axis=1)
    pair_hist = np.histogram(pair_dist, bins=12, range=(0, 1.5))[0].astype(np.float32)
    pair_hist /= max(pair_hist.sum(), 1.)
    cov = np.cov(p, rowvar=False)
    eigs = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0))[::-1]
    eigs /= max(float(eigs.sum()), 1e-12)
    ratios = np.sort(extent)[::-1] / max(float(extent.max()), 1e-8)
    feat = np.concatenate([radial_hist, pair_hist, eigs.astype(np.float32),
                           ratios.astype(np.float32)]).astype(np.float32)
    if feat.size != GEOMETRY_DIM or not np.isfinite(feat).all():
        raise RuntimeError(f"Invalid GT geometry descriptor for {uid}: {feat}")
    return torch.from_numpy(feat)


def prepare_shape_targets(records, gt_mesh_root, cache_path=None):
    """Build coarse shape descriptor targets from GT GLBs (train/val only)."""
    items = _unique_records(records)
    uids = sorted(items)
    root = Path(gt_mesh_root).expanduser().resolve()
    meta = {"version": GEOMETRY_CACHE_VERSION, "gt_mesh_root": str(root), "uids": uids}
    cache_path = Path(cache_path).expanduser().resolve() if cache_path else None
    if cache_path is not None:
        cached = _checked_cache(cache_path, meta, uids)
        if cached is not None:
            print(f"[*] Loaded GT shape cache: {cache_path} ({len(cached)} UIDs)", flush=True)
            return cached
    result = {}
    for uid in tqdm(uids, desc="GT geometry descriptors"):
        mesh = root / items[uid].object_path
        if not mesh.is_file():
            raise FileNotFoundError(f"Missing GT mesh required for geometry loss: {mesh}")
        result[uid] = _shape_descriptor_from_glb(mesh, uid)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"metadata": meta, "embeddings": result}, cache_path)
    return result


@torch.no_grad()
def evaluate_dual_visual(model, loader, records, visual_targets, geometry_targets, device):
    model.eval()
    embeddings, uids, geometry_preds = [], [], []
    for batch in loader:
        fmri = batch["fmri"].to(device).float()
        _, zvisual, zgeometry = model.forward_dual(fmri)
        embeddings.append(zvisual.cpu())
        if zgeometry is not None:
            geometry_preds.append(zgeometry.cpu())
        uids.extend(batch["uid"])
    zvisual = F.normalize(torch.cat(embeddings).float(), dim=-1)
    candidate_uids = list(dict.fromkeys(r.uid for r in records))
    bank = F.normalize(torch.stack([visual_targets[uid] for uid in candidate_uids]), dim=-1)
    targets = torch.tensor([candidate_uids.index(uid) for uid in uids], dtype=torch.long)
    scale = float(model.visual_logit_scale.detach().exp().clamp(max=100).item())
    logits = scale * (zvisual @ bank.T)
    pos = bank[targets]
    metrics = {
        "visual_ce": float(F.cross_entropy(logits, targets).item()),
        "visual_2way": expected_nway_top1(logits, targets, 2),
        "visual_10way": expected_nway_top1(logits, targets, min(10, len(candidate_uids)))
                         if len(candidate_uids) >= 2 else float("nan"),
        "visual_top1": topk_accuracy(logits, targets)[1],
        "visual_top5": topk_accuracy(logits, targets)[5],
        "visual_cos": float((zvisual*pos).sum(dim=-1).mean().item()),
    }
    if geometry_targets is not None:
        pred_geo = torch.cat(geometry_preds)
        gt_geo = torch.stack([geometry_targets[u] for u in uids])
        metrics["geometry_mae"] = float(F.l1_loss(pred_geo, gt_geo).item())
    return metrics


# -----------------------------------------------------------------------------
# Split / mesh checks
# -----------------------------------------------------------------------------

def split_train_val_by_uid(
    records: Sequence[ObjectRecord],
    val_fraction: float,
    seed: int,
):
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0,1)")

    indices = list(range(len(records)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    n_val = max(1, int(round(len(indices) * val_fraction)))
    val_idx = set(indices[:n_val])

    train_records = [
        r for i, r in enumerate(records)
        if i not in val_idx
    ]
    val_records = [
        r for i, r in enumerate(records)
        if i in val_idx
    ]

    return train_records, val_records


def validate_mesh_paths(
    records: Sequence[ObjectRecord],
    gt_mesh_root: Optional[str],
):
    if not gt_mesh_root:
        return

    root = Path(gt_mesh_root).expanduser()
    missing = []

    for r in records:
        path = root / r.object_path
        if not path.is_file():
            missing.append(str(path))

    if missing:
        print(
            f"[Warning] {len(missing)} GLB paths were not found under "
            f"{root}. First examples:",
            flush=True,
        )
        for p in missing[:5]:
            print(f"  {p}", flush=True)
    else:
        print(
            f"[*] GLB validation: all {len(records)} object paths found.",
            flush=True,
        )


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train MinD-3D++ fMRI -> CLIP text aligner"
    )

    p.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "fMRI-Objaverse root containing sub-0001, train_list.txt, "
            "Object_caption_Cap3D.csv, ..."
        ),
    )
    p.add_argument(
        "--subjects",
        type=str,
        default="sub-0001",
        help=(
            "Comma-separated subjects, e.g. "
            "sub-0001,sub-0006,sub-0007,sub-0008,sub-0015"
        ),
    )
    p.add_argument(
        "--train_list",
        type=str,
        default=None,
        help="Default: <data_path>/train_list.txt",
    )
    p.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Default: <data_path>/test_list.txt",
    )
    p.add_argument(
        "--caption_csv",
        type=str,
        default=None,
        help="Default: <data_path>/Object_caption_Cap3D.csv",
    )
    p.add_argument(
        "--gt_mesh_root",
        type=str,
        default=None,
        help=(
            "Optional. Root corresponding to GTmeshes/objaverse/glbs. "
            "Used only to validate object paths."
        ),
    )

    p.add_argument(
        "--text_source",
        choices=["caption", "category"],
        default="caption",
    )
    p.add_argument("--num_frames", type=int, default=6)
    p.add_argument(
        "--train_frame_sampling",
        choices=["random", "center", "uniform"],
        default="random",
        help=("C6 train-only sampling ablation. Validation/test ALWAYS use the "
              "middle num_frames to match existing inference preprocessing."),
    )
    p.add_argument(
        "--fmri_norm",
        choices=["none", "sample_zscore", "frame_zscore"],
        default="none",
    )
    p.add_argument(
        "--strict_files",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_fraction", type=float, default=0.10)
    p.add_argument(
        "--split_seed",
        type=int,
        default=0,
        help=(
            "Seed used ONLY to split train_list.txt into train/validation UIDs. "
            "Keep this fixed when comparing different model --seed values."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Model initialization / stochastic training seed.",
    )
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min_delta", type=float, default=1e-4)

    # E1 changes only the training loss; architecture / split / validation stay
    # unchanged. --objective batch_nce reproduces the original training loss.
    p.add_argument(
        "--objective", choices=["full_ce", "batch_nce"], default="full_ce",
        help="E1 full-training-bank CE or original symmetric batch NCE.",
    )
    p.add_argument(
        "--category_loss_weight", type=float, default=0.2,
        help="Coefficient of TRAIN-category CE for --objective full_ce.",
    )
    p.add_argument(
        "--batch_nce_weight", type=float, default=None,
        help=("Symmetric in-batch CLIP NCE coefficient. Supplying any of the "
              "three new weights activates explicit weighted-loss mode; "
              "unspecified weights become zero."),
    )
    p.add_argument(
        "--full_ce_weight", type=float, default=None,
        help="Full train-UID caption-bank cross entropy coefficient.",
    )
    p.add_argument(
        "--category_ce_weight", type=float, default=None,
        help="Train-category CLIP-text cross entropy coefficient.",
    )
    p.add_argument(
        "--checkpoint_metric", choices=["both", "objce", "10way"],
        default="both",
        help=("Always save best_objce.pt and best_10way.pt. "
              "For the legacy best.pt alias, '10way' selects 10-way; "
              "'both' and 'objce' select object CE."),
    )
    p.add_argument(
        "--nway_min_delta", type=float, default=0.0,
        help="Minimum validation 10-way improvement for updating best_10way.",
    )

    p.add_argument("--init_semantic_ckpt", type=str, default="",
                   help="Optional E1-B checkpoint for warm-start of semantic branch. "
                        "Only for a separate fine-tuning experiment.")
    p.add_argument(
        "--visual_views_root", type=str, default="",
        help="GT 6-view image root: <root>/<uid>/00.png ... 05.png. Required for visual loss.",
    )
    p.add_argument("--visual_cache_path", type=str, default="",
                   help="Reusable per-UID frozen CLIP image target cache across seeds.")
    p.add_argument("--visual_batch_size", type=int, default=48)
    p.add_argument("--visual_view_count", type=int, default=6)
    p.add_argument("--visual_hidden_dim", type=int, default=512)
    p.add_argument("--visual_loss_weight", type=float, default=0.25)
    p.add_argument("--visual_nce_weight", type=float, default=1.0)
    p.add_argument("--visual_cos_weight", type=float, default=0.5)
    p.add_argument("--geometry_loss_weight", type=float, default=0.0,
                   help="Optional GT-GLB coarse shape descriptor regression; requires --gt_mesh_root")
    p.add_argument("--geometry_cache_path", type=str, default="")

    p.add_argument("--frame_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--temporal_layers", type=int, default=1)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="Keeps the original script's OpenCLIP backbone by default.",
    )
    p.add_argument(
        "--clip_pretrained",
        type=str,
        default="openai",
    )
    p.add_argument(
        "--text_encode_batch_size",
        type=int,
        default=256,
    )

    p.add_argument(
        "--eval_test",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate test_list.txt once after loading the best validation "
            "checkpoint. Disabled by default to keep test final-only."
        ),
    )

    p.add_argument(
        "--out_dir",
        type=str,
        required=True,
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    return p.parse_args()


def main():
    args = parse_args()
    if not math.isfinite(args.category_loss_weight) or args.category_loss_weight < 0:
        raise ValueError("--category_loss_weight must be finite and nonnegative")

    explicit = any(
        x is not None for x in (
            args.batch_nce_weight, args.full_ce_weight, args.category_ce_weight
        )
    )
    if explicit:
        # Explicit weights take precedence over --objective/--category_loss_weight.
        weight_nce = float(args.batch_nce_weight or 0.0)
        weight_full = float(args.full_ce_weight or 0.0)
        weight_cat = float(args.category_ce_weight or 0.0)
    elif args.objective == "batch_nce":
        weight_nce, weight_full, weight_cat = 1.0, 0.0, 0.0
    else:
        weight_nce, weight_full = 0.0, 1.0
        weight_cat = float(args.category_loss_weight)

    for name, value in (
        ("batch_nce_weight", weight_nce),
        ("full_ce_weight", weight_full),
        ("category_ce_weight", weight_cat),
        ("nway_min_delta", args.nway_min_delta),
        ("visual_loss_weight", args.visual_loss_weight),
        ("visual_nce_weight", args.visual_nce_weight),
        ("visual_cos_weight", args.visual_cos_weight),
        ("geometry_loss_weight", args.geometry_loss_weight),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name} must be finite and nonnegative")
    if (weight_nce + weight_full + weight_cat) <= 0:
        raise ValueError("At least one training-loss coefficient must be positive")
    if weight_full > 0 and args.text_source != "caption":
        raise ValueError(
            "Nonzero --full_ce_weight requires --text_source caption; "
            "category-only texts produce duplicated object candidates."
        )
    if args.geometry_loss_weight > 0 and args.visual_loss_weight <= 0:
        raise ValueError("Geometry auxiliary branch requires visual supervision in E3")
    if args.visual_loss_weight > 0 and not args.visual_views_root:
        raise ValueError("Set --visual_views_root for Semantic-Visual Dual Alignment")
    if args.visual_loss_weight > 0 and (args.visual_nce_weight + args.visual_cos_weight) <= 0:
        raise ValueError("At least one visual objective component must be nonzero")
    if args.geometry_loss_weight > 0 and not args.gt_mesh_root:
        raise ValueError("--geometry_loss_weight requires --gt_mesh_root")
    if args.visual_view_count < 2 or args.visual_batch_size < 1:
        raise ValueError("--visual_view_count >= 2 and --visual_batch_size >= 1 required")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0 (0 disables early stopping)")

    # Save the RESOLVED weights, not None, so the checkpoint unambiguously
    # identifies which loss was actually trained.
    args.batch_nce_weight = weight_nce
    args.full_ce_weight = weight_full
    args.category_ce_weight = weight_cat
    print(
        f"[*] Objective weights: BatchNCE={weight_nce:g}, "
        f"FullObjectCE={weight_full:g}, CategoryCE={weight_cat:g}; "
        f"checkpoint_metric={args.checkpoint_metric}",
        flush=True,
    )
    seed_everything(args.seed)

    data_root = Path(args.data_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [
        s.strip()
        for s in args.subjects.split(",")
        if s.strip()
    ]
    if not subjects:
        raise ValueError("No subjects were specified")

    train_list_path = (
        Path(args.train_list).expanduser()
        if args.train_list
        else data_root / "train_list.txt"
    )
    test_list_path = (
        Path(args.test_list).expanduser()
        if args.test_list
        else data_root / "test_list.txt"
    )
    caption_csv_path = (
        Path(args.caption_csv).expanduser()
        if args.caption_csv
        else data_root / "Object_caption_Cap3D.csv"
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        or not args.device.startswith("cuda")
        else "cpu"
    )

    print(f"[*] Device: {device}", flush=True)
    print(f"[*] Data root: {data_root}", flush=True)
    print(f"[*] Subjects: {subjects}", flush=True)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    full_train_records = read_object_list(train_list_path)
    test_records = read_object_list(test_list_path)

    print(
        f"[*] Lists: train={len(full_train_records)} "
        f"test={len(test_records)}",
        flush=True,
    )

    validate_mesh_paths(
        full_train_records + test_records,
        args.gt_mesh_root,
    )

    caption_map = (
        load_caption_map(caption_csv_path)
        if args.text_source == "caption"
        else {}
    )

    all_records = full_train_records + test_records
    uid_to_text = build_texts(
        all_records,
        caption_map,
        args.text_source,
    )

    if args.text_source == "caption":
        all_uids = {r.uid for r in all_records}
        caption_uids = set(caption_map.keys())
        matched = len(all_uids & caption_uids)
        missing = len(all_uids - caption_uids)
        coverage = matched / max(1, len(all_uids))
        print(
            f"[*] Caption coverage: {matched}/{len(all_uids)} "
            f"({coverage:.2%}), missing={missing}",
            flush=True,
        )

    # Object-level holdout inside official train list.
    train_records, val_records = split_train_val_by_uid(
        full_train_records,
        args.val_fraction,
        args.split_seed,
    )

    print(
        f"[*] Object split: train={len(train_records)} "
        f"val={len(val_records)} test={len(test_records)} "
        f"split_seed={args.split_seed} model_seed={args.seed}",
        flush=True,
    )

    # Category vocabulary is global for stable indices.
    category_names = sorted(
        {r.category for r in all_records}
    )
    category_to_index = {
        c: i for i, c in enumerate(category_names)
    }
    print(
        f"[*] Categories: {len(category_names)}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Frozen OpenCLIP text encoder
    # ------------------------------------------------------------------
    print(
        f"[*] Loading OpenCLIP {args.clip_model} "
        f"({args.clip_pretrained})...",
        flush=True,
    )
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model,
        pretrained=args.clip_pretrained,
    )
    clip_model = clip_model.to(device).eval()
    clip_model.requires_grad_(False)
    clip_tokenizer = open_clip.get_tokenizer(
        args.clip_model
    )

    text_embeddings = encode_text_table(
        uid_to_text,
        clip_model,
        clip_tokenizer,
        device,
        batch_size=args.text_encode_batch_size,
    )

    category_embeddings = encode_category_embeddings(
        category_names,
        clip_model,
        clip_tokenizer,
        device,
    )

    clip_dim = next(
        iter(text_embeddings.values())
    ).numel()
    print(
        f"[*] CLIP text embedding dim: {clip_dim}",
        flush=True,
    )

    # Obtain frozen GT visual targets once (UID-only, no subject duplication).
    # Validation features are allowed as metric reference only, NEVER training bank.
    visual_targets = None
    if args.visual_loss_weight > 0:
        visual_targets = prepare_multiview_clip_targets(
            train_records + val_records,
            args.visual_views_root, clip_model, clip_preprocess, device,
            view_count=args.visual_view_count,
            batch_size=args.visual_batch_size,
            cache_path=args.visual_cache_path or None,
            clip_name=args.clip_model,
            clip_weights=args.clip_pretrained,
        )
        if any(u in {r.uid for r in test_records} for u in visual_targets):
            raise RuntimeError("Official test UID leaked into visual target features")

    # Free frozen CLIP GPU memory before fMRI model training.
    clip_model = clip_model.cpu()
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Frozen candidate banks use strictly train-list UIDs only. For E0 these
    # banks are diagnostics and do not affect training gradients.
    # ------------------------------------------------------------------
    (
        train_object_bank,
        train_uid_to_index,
        train_category_bank,
        train_category_to_index,
    ) = prepare_e1_candidate_banks(
        train_records, text_embeddings, category_names,
        category_embeddings, device,
    )
    train_uid_set = set(train_uid_to_index)
    if train_uid_set.intersection(r.uid for r in val_records):
        raise RuntimeError("Validation UID leaked into training caption bank")
    if train_uid_set.intersection(r.uid for r in test_records):
        raise RuntimeError("Test UID leaked into training caption bank")
    print(
        f"[*] Training banks: objects={len(train_uid_to_index)}, "
        f"categories={len(train_category_to_index)} "
        "(validation/test UIDs excluded)",
        flush=True,
    )

    geometry_targets = None
    if args.geometry_loss_weight > 0:
        geometry_targets = prepare_shape_targets(
            train_records + val_records, args.gt_mesh_root,
            cache_path=args.geometry_cache_path or None,
        )
        if any(u in {r.uid for r in test_records} for u in geometry_targets):
            raise RuntimeError("Official test UID leaked into geometry teacher")
    # Train targets MUST be indexed using the actual training UIDs only.
    train_visual_targets = ({r.uid: visual_targets[r.uid] for r in train_records}
                            if visual_targets is not None else None)
    train_geometry_targets = ({r.uid: geometry_targets[r.uid] for r in train_records}
                              if geometry_targets is not None else None)

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    print(
        f"[*] C6: train sampling={args.train_frame_sampling}; "
        f"eval sampling=center; fmri_norm={args.fmri_norm}; "
        f"dropout={args.dropout:g}; weight_decay={args.weight_decay:g}",
        flush=True,
    )
    train_dataset = FMRIObjaverseDataset(
        data_root=data_root,
        subjects=subjects,
        records=train_records,
        text_embeddings=text_embeddings,
        category_to_index=category_to_index,
        train_mode=True,
        visual_embeddings=train_visual_targets,
        geometry_targets=train_geometry_targets,
        num_frames=args.num_frames,
        fmri_norm=args.fmri_norm,
        train_frame_sampling=args.train_frame_sampling,
        strict_files=args.strict_files,
    )
    val_dataset = FMRIObjaverseDataset(
        data_root=data_root,
        subjects=subjects,
        records=val_records,
        text_embeddings=text_embeddings,
        category_to_index=category_to_index,
        train_mode=False,
        visual_embeddings=visual_targets,
        geometry_targets=geometry_targets,
        num_frames=args.num_frames,
        fmri_norm=args.fmri_norm,
        strict_files=args.strict_files,
    )

    # Deterministic train-list evaluation: middle frames, no random frame draw.
    # This is used only after best-checkpoint reload to quantify overfitting
    # under the same full-candidate retrieval protocol as validation.
    train_eval_dataset = FMRIObjaverseDataset(
        data_root=data_root,
        subjects=subjects,
        records=train_records,
        text_embeddings=text_embeddings,
        category_to_index=category_to_index,
        train_mode=False,
        num_frames=args.num_frames,
        fmri_norm=args.fmri_norm,
        strict_files=args.strict_files,
    )

    test_dataset = None
    if args.eval_test:
        test_dataset = FMRIObjaverseDataset(
            data_root=data_root,
            subjects=subjects,
            records=test_records,
            text_embeddings=text_embeddings,
            category_to_index=category_to_index,
            train_mode=False,
            num_frames=args.num_frames,
            fmri_norm=args.fmri_norm,
            strict_files=args.strict_files,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        if test_dataset is not None
        else None
    )

    print(
        f"[*] Expanded samples: train={len(train_dataset)} "
        f"val={len(val_dataset)} "
        f"test={len(test_dataset) if test_dataset is not None else 0}",
        flush=True,
    )

    # Inspect one fMRI sample before training.
    first = train_dataset[0]
    print(
        f"[*] fMRI sample shape={tuple(first['fmri'].shape)} "
        f"dtype={first['fmri'].dtype} "
        f"range=[{first['fmri'].min().item():.4f}, "
        f"{first['fmri'].max().item():.4f}]",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Model / optimizer
    # ------------------------------------------------------------------
    model = DualFMRIClipAligner(
        clip_dim=clip_dim,
        num_frames=args.num_frames,
        frame_dim=args.frame_dim,
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
        visual_hidden_dim=args.visual_hidden_dim,
        geometry_dim=GEOMETRY_DIM if args.geometry_loss_weight > 0 else 0,
    ).to(device)

    # Optional E1-B warm-start; visual/geometry heads remain newly initialized.
    if args.init_semantic_ckpt:
        init_path = Path(args.init_semantic_ckpt).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(init_path)
        init_ckpt = _load_cache(init_path)
        init_state = init_ckpt.get("model", init_ckpt)
        old_keys = set(semantic_only_state_dict(model))
        if set(init_state) != old_keys:
            raise ValueError("Warm-start checkpoint semantic weights don't match "
                             "the E1-B architecture (check all model dimensions).")
        incompatible = model.load_state_dict(init_state, strict=False)
        if incompatible.unexpected_keys or any(
            not k.startswith(("visual_", "geometry_")) for k in incompatible.missing_keys
        ):
            raise RuntimeError(f"Unexpected warm-start key mismatch: {incompatible}")
        print(f"[*] E1-B semantic warm-start from {init_path}; "
              "visual/geometry heads are randomly initialized.", flush=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    param_count = sum(
        p.numel() for p in model.parameters()
    )
    print(
        f"[*] fMRI aligner parameters: {param_count:,}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_path = out_dir / "training_log.csv"
    log_headers = [
        "Epoch", "Loss", "Train_Batch_Retrieval_Acc",
        "Val_Full_Object_CE", "Val_Full_Category_CE",
        "Val_2Way_Top1", "Val_10Way_Top1",
        "Val_Object_Top1", "Val_Object_Top5",
        "Val_Category_Top1", "Val_Category_Top5", "LR",
        "Train_Full_Object_CE", "Train_Full_Category_CE",
        "Train_Full_Object_Top1", "Train_Full_Category_Top1",
        "Train_Batch_NCE", "Train_Logit_Scale",
        "Train_Visual_NCE", "Train_Visual_Cos_Loss", "Train_Geometry_Loss",
        "Val_Visual_CE", "Val_Visual_2Way", "Val_Visual_10Way",
        "Val_Visual_Top1", "Val_Visual_Top5", "Val_Visual_Cos",
        "Val_Geometry_MAE", "Visual_Logit_Scale",
    ]
    with log_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(log_headers)

    config = vars(args).copy()
    config.update(
        {
            "resolved_data_path": str(data_root),
            "subjects_list": subjects,
            "clip_dim": clip_dim,
            "num_categories": len(category_names),
            "num_train_objects": len(train_records),
            "num_val_objects": len(val_records),
            "num_test_objects": len(test_records),
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }
    )
    write_json(out_dir / "config.json", config)

    # Save the object split explicitly.
    write_json(
        out_dir / "split.json",
        {
            "split_seed": args.split_seed,
            "model_seed": args.seed,
            "train_uids": [r.uid for r in train_records],
            "val_uids": [r.uid for r in val_records],
            "test_uids": [r.uid for r in test_records],
        },
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_val_object_ce = float("inf")
    best_val_10way = float("-inf")
    best_objce_epoch = 0
    best_10way_epoch = 0
    stale = 0

    best_objce_path = out_dir / "fmri_clip_aligner_best_objce.pt"
    best_10way_path = out_dir / "fmri_clip_aligner_best_10way.pt"
    # Keep the original checkpoint name usable by the existing inference code.
    best_model_path = out_dir / "fmri_clip_aligner_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        total_train_nce = 0.0
        total_train_object_ce = 0.0
        total_train_category_ce = 0.0
        total_train_object_acc = 0.0
        total_train_category_acc = 0.0
        total_visual_nce = 0.0
        total_visual_cos_loss = 0.0
        total_geometry_loss = 0.0
        batches = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        ):
            optimizer.zero_grad(set_to_none=True)

            fmri = batch["fmri"].to(device, non_blocking=True).float()
            target_text = batch["text_embedding"].to(
                device, non_blocking=True
            ).float()
            target_text = F.normalize(target_text, dim=-1)
            uids = list(batch["uid"])
            categories = list(batch["category"])

            fmri_embeds, visual_embeds, geometry_pred = model.forward_dual(fmri)

            # Match E0 exactly when batch NCE is the only active objective.
            if weight_nce > 0:
                loss_nce, batch_acc = symmetric_multi_positive_clip_loss(
                    fmri_embeds, target_text, uids, model.logit_scale,
                )
            else:
                with torch.no_grad():
                    loss_nce, batch_acc = symmetric_multi_positive_clip_loss(
                        fmri_embeds, target_text, uids, model.logit_scale,
                    )

            # Object/category losses are evaluated for every experiment to
            # obtain comparable training diagnostics. For pure E0, this pass is
            # no-grad and does not modify the baseline objective.
            if weight_full > 0 or weight_cat > 0:
                (_, loss_obj, loss_cat, acc_obj, acc_cat) = (
                    full_candidate_object_category_loss(
                        brain_embeddings=fmri_embeds,
                        uids=uids,
                        categories=categories,
                        object_bank=train_object_bank,
                        uid_to_index=train_uid_to_index,
                        category_bank=train_category_bank,
                        category_to_index=train_category_to_index,
                        logit_scale=model.logit_scale,
                        category_loss_weight=0.0,
                    )
                )
            else:
                with torch.no_grad():
                    (_, loss_obj, loss_cat, acc_obj, acc_cat) = (
                        full_candidate_object_category_loss(
                            brain_embeddings=fmri_embeds,
                            uids=uids,
                            categories=categories,
                            object_bank=train_object_bank,
                            uid_to_index=train_uid_to_index,
                            category_bank=train_category_bank,
                            category_to_index=train_category_to_index,
                            logit_scale=model.logit_scale,
                            category_loss_weight=0.0,
                        )
                    )

            visual_nce = fmri_embeds.new_zeros(())
            visual_cos_loss = fmri_embeds.new_zeros(())
            geometry_loss = fmri_embeds.new_zeros(())
            if visual_targets is not None:
                target_visual = F.normalize(
                    batch["visual_embedding"].to(device).float(), dim=-1)
                if args.visual_nce_weight > 0:
                    visual_nce, _ = symmetric_multi_positive_clip_loss(
                        visual_embeds, target_visual, uids, model.visual_logit_scale)
                if args.visual_cos_weight > 0:
                    visual_cos_loss = (1 - (visual_embeds * target_visual).sum(dim=-1)).mean()
            if geometry_targets is not None:
                target_geometry = batch["geometry_target"].to(device).float()
                geometry_loss = F.smooth_l1_loss(geometry_pred, target_geometry)

            # Only positive-weight terms are added, avoiding unnecessary
            # autograd paths and preserving the exact E0/E1 loss definitions.
            loss = None
            for coeff, component in (
                (weight_nce, loss_nce),
                (weight_full, loss_obj),
                (weight_cat, loss_cat),
                (args.visual_loss_weight * args.visual_nce_weight, visual_nce),
                (args.visual_loss_weight * args.visual_cos_weight, visual_cos_loss),
                (args.geometry_loss_weight, geometry_loss),
            ):
                if coeff > 0:
                    term = coeff * component
                    loss = term if loss is None else loss + term
            if loss is None or not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite or empty combined training loss")

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += float(batch_acc.item())
            total_train_nce += float(loss_nce.detach().item())
            total_train_object_ce += float(loss_obj.detach().item())
            total_train_category_ce += float(loss_cat.detach().item())
            total_train_object_acc += float(acc_obj.item())
            total_train_category_acc += float(acc_cat.item())
            total_visual_nce += float(visual_nce.detach().item())
            total_visual_cos_loss += float(visual_cos_loss.detach().item())
            total_geometry_loss += float(geometry_loss.detach().item())
            batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        if batches == 0:
            raise RuntimeError("No training batches; reduce --batch_size")

        avg_loss = total_loss / batches
        train_acc = total_acc / batches
        avg_nce = total_train_nce / batches
        avg_obj_ce = total_train_object_ce / batches
        avg_cat_ce = total_train_category_ce / batches
        avg_obj_acc = total_train_object_acc / batches
        avg_cat_acc = total_train_category_acc / batches
        avg_visual_nce = total_visual_nce / batches
        avg_visual_cos_loss = total_visual_cos_loss / batches
        avg_geometry_loss = total_geometry_loss / batches
        visual_scale = float(model.visual_logit_scale.detach().exp().clamp(max=100).item())
        logit_scale = float(model.logit_scale.detach().exp().clamp(max=100).item())

        val_metrics = evaluate_retrieval(
            model, val_loader, val_records, text_embeddings,
            category_names, category_embeddings, device,
        )
        visual_val = (evaluate_dual_visual(model, val_loader, val_records,
                                           visual_targets, geometry_targets, device)
                      if visual_targets is not None else {})
        val_object_ce = float(val_metrics["object_ce"])
        val_category_ce = float(val_metrics["category_ce"])
        val_10way = float(val_metrics["object_10way_top1"])
        if not math.isfinite(val_object_ce) or not math.isfinite(val_10way):
            raise RuntimeError("Validation object CE or 10-way became non-finite")

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | NCE={avg_nce:.4f} | "
            f"TrainBatchAcc={train_acc:.4f} | "
            f"TrainFullObjCE={avg_obj_ce:.4f} | "
            f"TrainCatCE={avg_cat_ce:.4f} | "
            f"TrainObj@1={avg_obj_acc:.4f} | "
            f"TrainCat@1={avg_cat_acc:.4f} | "
            f"ValObjCE={val_object_ce:.4f} | "
            f"ValCatCE={val_category_ce:.4f} | "
            f"Val2Way={val_metrics['object_2way_top1']:.4f} | "
            f"Val10Way={val_10way:.4f} | "
            f"ValObj@1={val_metrics['object_top1']:.4f} | "
            f"ValObj@5={val_metrics['object_top5']:.4f} | "
            f"ValCat@1={val_metrics['category_top1']:.4f} | "
            f"ValCat@5={val_metrics['category_top5']:.4f} | "
            f"VisNCE={avg_visual_nce:.4f} | VisCosLoss={avg_visual_cos_loss:.4f} | "
            f"ValVisCE={visual_val.get('visual_ce', float('nan')):.4f} | "
            f"ValVis10Way={visual_val.get('visual_10way', float('nan')):.4f} | "
            f"ValVisCos={visual_val.get('visual_cos', float('nan')):.4f} | "
            f"ValGeoMAE={visual_val.get('geometry_mae', float('nan')):.4f} | "
            f"Scale={logit_scale:.4f} | VisScale={visual_scale:.4f} | LR={current_lr:.6g}",
            flush=True,
        )

        with log_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{avg_loss:.8f}", f"{train_acc:.8f}",
                f"{val_object_ce:.8f}", f"{val_category_ce:.8f}",
                f"{val_metrics['object_2way_top1']:.8f}",
                f"{val_10way:.8f}",
                f"{val_metrics['object_top1']:.8f}",
                f"{val_metrics['object_top5']:.8f}",
                f"{val_metrics['category_top1']:.8f}",
                f"{val_metrics['category_top5']:.8f}",
                f"{current_lr:.10f}",
                f"{avg_obj_ce:.8f}", f"{avg_cat_ce:.8f}",
                f"{avg_obj_acc:.8f}", f"{avg_cat_acc:.8f}",
                f"{avg_nce:.8f}", f"{logit_scale:.8f}",
                f"{avg_visual_nce:.8f}", f"{avg_visual_cos_loss:.8f}",
                f"{avg_geometry_loss:.8f}",
                f"{visual_val.get('visual_ce', float('nan')):.8f}",
                f"{visual_val.get('visual_2way', float('nan')):.8f}",
                f"{visual_val.get('visual_10way', float('nan')):.8f}",
                f"{visual_val.get('visual_top1', float('nan')):.8f}",
                f"{visual_val.get('visual_top5', float('nan')):.8f}",
                f"{visual_val.get('visual_cos', float('nan')):.8f}",
                f"{visual_val.get('geometry_mae', float('nan')):.8f}",
                f"{visual_scale:.8f}",
            ])

        # Always save both selection criteria, and update the legacy best.pt
        # according to --checkpoint_metric (default: object CE).
        ckpt_info = {
            **checkpoint_state(model),
            "epoch": epoch,
            "val_full_object_ce": val_object_ce,
            "val_full_category_ce": val_category_ce,
            "val_metrics": val_metrics,
            "val_visual_metrics": visual_val,
            "config": config,
        }

        if val_object_ce < best_val_object_ce - args.min_delta:
            best_val_object_ce = val_object_ce
            best_objce_epoch = epoch
            stale = 0
            payload = dict(ckpt_info, checkpoint_metric="full_validation_object_ce")
            torch.save(payload, best_objce_path)
            if args.checkpoint_metric != "10way":
                torch.save(payload, best_model_path)
            print(
                f"[*] BestObjCE updated: epoch={epoch}, "
                f"ValObjCE={val_object_ce:.6f}", flush=True,
            )
        else:
            stale += 1

        if val_10way > best_val_10way + args.nway_min_delta:
            best_val_10way = val_10way
            best_10way_epoch = epoch
            payload = dict(ckpt_info, checkpoint_metric="validation_10way_top1")
            torch.save(payload, best_10way_path)
            if args.checkpoint_metric == "10way":
                torch.save(payload, best_model_path)
            print(
                f"[*] Best10Way updated: epoch={epoch}, "
                f"Val10Way={val_10way:.6f}", flush=True,
            )

        if args.patience > 0 and stale >= args.patience:
            print(
                f"[*] Early stopping at epoch {epoch}; "
                f"best_objce_epoch={best_objce_epoch}; "
                f"best_10way_epoch={best_10way_epoch}", flush=True,
            )
            break

    last_model_path = out_dir / "fmri_clip_aligner_last.pt"
    torch.save(
        {
            **checkpoint_state(model),
            "epoch": epoch,
            "config": config,
        },
        last_model_path,
    )

    # ------------------------------------------------------------------
    # Reload both selected checkpoints and report under the SAME validation
    # candidate set. Default best.pt remains the CE-selected checkpoint for
    # backward compatibility with inference_fmri_mind3dpp.py.
    # ------------------------------------------------------------------
    def load_and_evaluate(path):
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["dual_model"], strict=True)
        validation_metrics = evaluate_retrieval(
            model, val_loader, val_records, text_embeddings,
            category_names, category_embeddings, device,
        )
        training_metrics = evaluate_retrieval(
            model, train_eval_loader, train_records, text_embeddings,
            category_names, category_embeddings, device,
        )
        if visual_targets is not None:
            validation_metrics.update(evaluate_dual_visual(
                model, val_loader, val_records, visual_targets, geometry_targets, device))
        return checkpoint, validation_metrics, training_metrics

    ce_best, ce_val, ce_train = load_and_evaluate(best_objce_path)
    nw_best, nw_val, nw_train = load_and_evaluate(best_10way_path)

    def show_metrics(tag, ckpt, metrics):
        print(
            f"[{tag}] epoch={ckpt['epoch']} "
            f"ObjCE={metrics['object_ce']:.4f} "
            f"CatCE={metrics['category_ce']:.4f} "
            f"2Way={metrics['object_2way_top1']:.4%} "
            f"10Way={metrics['object_10way_top1']:.4%} "
            f"Obj@1={metrics['object_top1']:.4%} "
            f"Obj@5={metrics['object_top5']:.4%} "
            f"Cat@1={metrics['category_top1']:.4%} "
            f"Cat@5={metrics['category_top5']:.4%}",
            flush=True,
        )

    show_metrics("VAL best-objce", ce_best, ce_val)
    if visual_targets is not None:
        print("[VAL best-objce visual] "
              f"CE={ce_val['visual_ce']:.4f} 10Way={ce_val['visual_10way']:.4%} "
              f"Cos={ce_val['visual_cos']:.4f}", flush=True)
    show_metrics("VAL best-10way", nw_best, nw_val)
    show_metrics("TRAIN best-objce", ce_best, ce_train)
    show_metrics("TRAIN best-10way", nw_best, nw_train)

    # For the optional one-time official test, evaluate only the chosen legacy
    # inference checkpoint, never use test accuracy to choose the objective.
    primary = "10way" if args.checkpoint_metric == "10way" else "objce"
    primary_ckpt = nw_best if primary == "10way" else ce_best
    primary_val = nw_val if primary == "10way" else ce_val
    primary_train = nw_train if primary == "10way" else ce_train

    summary = {
        "objective_weights": {
            "batch_nce_weight": weight_nce,
            "full_ce_weight": weight_full,
            "category_ce_weight": weight_cat,
            "visual_loss_weight": float(args.visual_loss_weight),
            "visual_nce_weight": float(args.visual_nce_weight),
            "visual_cos_weight": float(args.visual_cos_weight),
            "geometry_loss_weight": float(args.geometry_loss_weight),
        },
        "checkpoint_metric": primary_ckpt["checkpoint_metric"],
        "checkpoint_selection_mode": args.checkpoint_metric,
        "primary_checkpoint": str(best_model_path),
        "best_epoch": int(primary_ckpt["epoch"]),
        "best_val_full_object_ce": float(primary_val["object_ce"]),
        "best_val_full_category_ce": float(primary_val["category_ce"]),
        "train_full_retrieval": primary_train,
        "validation": primary_val,
        "best_objce": {
            "checkpoint": str(best_objce_path),
            "epoch": int(ce_best["epoch"]),
            "validation": ce_val,
            "train_full_retrieval": ce_train,
        },
        "best_10way": {
            "checkpoint": str(best_10way_path),
            "epoch": int(nw_best["epoch"]),
            "validation": nw_val,
            "train_full_retrieval": nw_train,
        },
    }

    if test_loader is not None:
        model.load_state_dict(primary_ckpt["dual_model"], strict=True)
        test_metrics = evaluate_retrieval(
            model, test_loader, test_records, text_embeddings,
            category_names, category_embeddings, device,
        )
        summary["test"] = test_metrics
        show_metrics("TEST final", primary_ckpt, test_metrics)

    write_json(out_dir / "summary.json", summary)

    print(
        f"[*] Best ObjCE checkpoint: {best_objce_path} "
        f"(epoch={best_objce_epoch})\n"
        f"[*] Best 10-way checkpoint: {best_10way_path} "
        f"(epoch={best_10way_epoch})\n"
        f"[*] Legacy inference checkpoint: {best_model_path} "
        f"(selection={primary})\n"
        f"[*] Last checkpoint: {last_model_path}\n"
        f"[*] Summary: {out_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
