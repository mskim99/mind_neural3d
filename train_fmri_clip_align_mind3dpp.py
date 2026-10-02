#!/usr/bin/env python3
"""
Train an fMRI -> CLIP text aligner on the MinD-3D++ / fMRI-Objaverse dataset.

This script is a replacement for the EEG-specific train_eeg_clip_align.py.

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

The official MinD-3D++ fMRI-Objaverse loader stores each object's fMRI as
<subject>/<uid>.npy. During training it randomly selects 6 fMRI frames;
during evaluation it uses the middle 6 frames. This script follows that
sampling rule.

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
"""

import argparse
import csv
import json
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

            # First try whitespace-separated official format.
            parts = line.split()

            # Also accept CSV-like list files.
            if len(parts) < 2 and "," in line:
                parts = next(csv.reader([line]))
                parts = [p.strip() for p in parts if p.strip()]

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
      randomly choose num_frames from all fMRI frames.

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
        strict_files: bool = True,
    ):
        self.data_root = Path(data_root)
        self.subjects = list(subjects)
        self.records = list(records)
        self.text_embeddings = text_embeddings
        self.category_to_index = category_to_index
        self.train_mode = bool(train_mode)
        self.num_frames = int(num_frames)
        self.fmri_norm = fmri_norm

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

        if self.train_mode:
            ids = np.random.choice(
                n, size=self.num_frames, replace=False
            )
            ids.sort()
        else:
            start = (n - self.num_frames) // 2
            ids = np.arange(start, start + self.num_frames)

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

        return {
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

    def forward(self, fmri):
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

        features = self.net(pooled)
        return F.normalize(features, dim=-1)


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
    clip_model, _, _ = open_clip.create_model_and_transforms(
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

    # Free frozen CLIP GPU memory before fMRI model training.
    clip_model = clip_model.cpu()
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_dataset = FMRIObjaverseDataset(
        data_root=data_root,
        subjects=subjects,
        records=train_records,
        text_embeddings=text_embeddings,
        category_to_index=category_to_index,
        train_mode=True,
        num_frames=args.num_frames,
        fmri_norm=args.fmri_norm,
        strict_files=args.strict_files,
    )
    val_dataset = FMRIObjaverseDataset(
        data_root=data_root,
        subjects=subjects,
        records=val_records,
        text_embeddings=text_embeddings,
        category_to_index=category_to_index,
        train_mode=False,
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
    model = FMRIClipAligner(
        clip_dim=clip_dim,
        num_frames=args.num_frames,
        frame_dim=args.frame_dim,
        hidden_dim=args.hidden_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        dropout=args.dropout,
    ).to(device)

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
    with log_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Epoch",
                "Loss",
                "Train_Batch_Retrieval_Acc",
                "Val_Full_Object_CE",
                "Val_Full_Category_CE",
                "Val_Object_Top1",
                "Val_Object_Top5",
                "Val_Category_Top1",
                "Val_Category_Top5",
                "LR",
            ]
        )

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
    best_epoch = 0
    stale = 0
    best_model_path = out_dir / "fmri_clip_aligner_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for batch in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
        ):
            optimizer.zero_grad(set_to_none=True)

            fmri = batch["fmri"].to(
                device, non_blocking=True
            ).float()
            target_text = batch[
                "text_embedding"
            ].to(device, non_blocking=True).float()
            target_text = F.normalize(
                target_text, dim=-1
            )
            uids = list(batch["uid"])

            fmri_embeds = model(fmri)

            loss, batch_acc = (
                symmetric_multi_positive_clip_loss(
                    fmri_embeds,
                    target_text,
                    uids,
                    model.logit_scale,
                )
            )

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_acc += float(batch_acc.item())
            batches += 1

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / max(1, batches)
        train_acc = total_acc / max(1, batches)

        # Full-candidate validation retrieval.
        # The returned object_ce compares every validation fMRI embedding
        # against ALL validation object-caption candidates at once.
        val_metrics = evaluate_retrieval(
            model,
            val_loader,
            val_records,
            text_embeddings,
            category_names,
            category_embeddings,
            device,
        )
        val_object_ce = val_metrics["object_ce"]
        val_category_ce = val_metrics["category_ce"]

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={avg_loss:.4f} | "
            f"TrainBatchAcc={train_acc:.4f} | "
            f"ValObjCE={val_object_ce:.4f} | "
            f"ValCatCE={val_category_ce:.4f} | "
            f"ValObj@1={val_metrics['object_top1']:.4f} | "
            f"ValObj@5={val_metrics['object_top5']:.4f} | "
            f"ValCat@1={val_metrics['category_top1']:.4f} | "
            f"ValCat@5={val_metrics['category_top5']:.4f} | "
            f"LR={current_lr:.6g}",
            flush=True,
        )

        with log_path.open(
            "a", encoding="utf-8", newline=""
        ) as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    f"{avg_loss:.8f}",
                    f"{train_acc:.8f}",
                    f"{val_object_ce:.8f}",
                    f"{val_category_ce:.8f}",
                    f"{val_metrics['object_top1']:.8f}",
                    f"{val_metrics['object_top5']:.8f}",
                    f"{val_metrics['category_top1']:.8f}",
                    f"{val_metrics['category_top5']:.8f}",
                    f"{current_lr:.10f}",
                ]
            )

        # Checkpoint selection is based ONLY on full-validation object CE.
        # This directly matches the full object-retrieval candidate set and
        # replaces the previous small-mini-batch validation NCE criterion.
        if val_object_ce < best_val_object_ce - args.min_delta:
            best_val_object_ce = val_object_ce
            best_epoch = epoch
            stale = 0

            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_full_object_ce": val_object_ce,
                    "val_full_category_ce": val_category_ce,
                    "val_metrics": val_metrics,
                    "checkpoint_metric": "full_validation_object_ce",
                    "config": config,
                },
                best_model_path,
            )
            print(
                f"[*] Best checkpoint updated: "
                f"epoch={epoch}, full_val_object_ce={val_object_ce:.6f}",
                flush=True,
            )
        else:
            stale += 1

        if (
            args.patience > 0
            and stale >= args.patience
        ):
            print(
                f"[*] Early stopping at epoch {epoch}; "
                f"best_epoch={best_epoch}",
                flush=True,
            )
            break

    last_model_path = out_dir / "fmri_clip_aligner_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        last_model_path,
    )

    # ------------------------------------------------------------------
    # Reload best and report validation / test once
    # ------------------------------------------------------------------
    best = torch.load(
        best_model_path,
        map_location=device,
    )
    model.load_state_dict(best["model"])

    best_val_metrics = evaluate_retrieval(
        model,
        val_loader,
        val_records,
        text_embeddings,
        category_names,
        category_embeddings,
        device,
    )

    best_train_metrics = evaluate_retrieval(
        model,
        train_eval_loader,
        train_records,
        text_embeddings,
        category_names,
        category_embeddings,
        device,
    )

    summary = {
        "best_epoch": int(best["epoch"]),
        "checkpoint_metric": "full_validation_object_ce",
        "best_val_full_object_ce": float(best["val_full_object_ce"]),
        "best_val_full_category_ce": float(best["val_full_category_ce"]),
        "train_full_retrieval": best_train_metrics,
        "validation": best_val_metrics,
    }

    print(
        "[TRAIN full] "
        f"ObjCE={best_train_metrics['object_ce']:.4f} "
        f"Obj@1={best_train_metrics['object_top1']:.4%} "
        f"Obj@5={best_train_metrics['object_top5']:.4%} "
        f"Cat@1={best_train_metrics['category_top1']:.4%} "
        f"Cat@5={best_train_metrics['category_top5']:.4%}",
        flush=True,
    )

    print(
        "[VAL best] "
        f"epoch={best['epoch']} "
        f"ObjCE={best_val_metrics['object_ce']:.4f} "
        f"CatCE={best_val_metrics['category_ce']:.4f} "
        f"Obj@1={best_val_metrics['object_top1']:.4%} "
        f"Obj@5={best_val_metrics['object_top5']:.4%} "
        f"Cat@1={best_val_metrics['category_top1']:.4%} "
        f"Cat@5={best_val_metrics['category_top5']:.4%}",
        flush=True,
    )

    if test_loader is not None:
        test_metrics = evaluate_retrieval(
            model,
            test_loader,
            test_records,
            text_embeddings,
            category_names,
            category_embeddings,
            device,
        )
        summary["test"] = test_metrics

        print(
            "[TEST final] "
            f"ObjCE={test_metrics['object_ce']:.4f} "
            f"CatCE={test_metrics['category_ce']:.4f} "
            f"Obj@1={test_metrics['object_top1']:.4%} "
            f"Obj@5={test_metrics['object_top5']:.4%} "
            f"Cat@1={test_metrics['category_top1']:.4%} "
            f"Cat@5={test_metrics['category_top5']:.4%}",
            flush=True,
        )

    write_json(out_dir / "summary.json", summary)

    print(
        f"[*] Best model: {best_model_path}\n"
        f"[*] Last model: {last_model_path}\n"
        f"[*] Summary: {out_dir / 'summary.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
