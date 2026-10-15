#!/usr/bin/env python3
"""
MinD-3D++ fMRI inference with C6/E3 Semantic-Visual TRAIN-bank fusion.

Pipeline
--------
MinD-3D++ fMRI (.npy)
    -> FMRIClipAligner checkpoint
    -> TRAIN-only CLIP caption/GT multi-view bank retrieval
       * semantic alpha=1 or z-score fusion alpha in [0,1]
       * train_caption: nearest caption among TRAIN UIDs only (default)
       * category: nearest training-category prompt
       * hybrid: predicted category + nearest TRAIN caption
    -> Embedding-space object identification evaluation (GT captions;
       held-out split only; evaluation-only, never generation prompting)
    -> SDXL
    -> rembg
    -> Zero123++
    -> cleaned 6-view images

Important
---------
- This script DOES NOT use Neuro-3D EEG or train_static_eeg_mlp.py.
- Test-object ground-truth captions are never used as generation candidates.
  Caption prompting retrieves only from training UIDs (preferably from the
  checkpoint directory's split.json).
- fMRI frame selection and normalization are read from the training checkpoint
  config to keep train/inference preprocessing identical.
- Object 2-way/10-way uses the SAME expected-N-way counting method as
  train_fmri_clip_align_mind3dpp_dual.py, not the paper's rendered-image metric.
- split.json is REQUIRED to prevent validation objects from entering the
  train-caption generation bank.
- Visual features come from TRAIN UID cache only, aligned by UID with caption
  candidates; no held-out GT images or captions participate in prompting.
- Per-query zscore fusion matches evaluate_fmri_classification.py on the TRAIN
  candidate bank. Object N-way metrics remain SEMANTIC GT-caption diagnostics
  and are NEVER used to choose a generation prompt.
- The official TEST split must not be used to tune alpha or checkpoint choice.
- For reproducibility, the N-way candidate pool is the complete requested
  evaluation split (271 validation or 432 official test objects), even if only
  a subset of queries is processed with --max_samples / --start_index.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import open_clip
from huggingface_hub import hf_hub_download

try:
    import rembg
except ImportError:
    rembg = None

try:
    from diffusers import (
        StableDiffusionXLPipeline,
        DiffusionPipeline,
        EulerAncestralDiscreteScheduler,
    )
except ImportError:
    StableDiffusionXLPipeline = None
    DiffusionPipeline = None
    EulerAncestralDiscreteScheduler = None


DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_ZERO123_MODEL = "sudo-ai/zero123plus-v1.1"
DEFAULT_SDXL_NEGATIVE_PROMPT = (
    "multiple objects, duplicate object, extra object, group, collection, cluster, "
    "cluttered background, complex background, perspective distortion, "
    "text, watermark, logo, cropped object, partial object, "
    "deformed geometry, distorted geometry, blurry, noisy, low quality"
)


# =============================================================================
# Metadata
# =============================================================================

@dataclass(frozen=True)
class ObjectRecord:
    category: str
    object_path: str
    uid: str


def normalize_uid(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    return Path(value).stem


def pretty_category(value: str) -> str:
    return str(value).strip().replace("_", " ").replace("-", " ")


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value).strip())
    return value.strip("_") or "sample"


def read_object_list(path: Path) -> List[ObjectRecord]:
    if not path.is_file():
        raise FileNotFoundError(path)

    records: List[ObjectRecord] = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            # Match the E3 training parser: commas must take precedence over
            # whitespace, otherwise "chair, 000-035/uid.glb" becomes "chair,".
            if "," in line:
                parts = next(csv.reader([line]))
                parts = [p.strip() for p in parts if p.strip()]
            else:
                parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{line_no}: expected category + object path, got {line!r}"
                )

            object_path = parts[-1].strip()
            category = " ".join(parts[:-1]).strip()

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

    return records


def build_eval_object_caption_candidates(
    records: Sequence[ObjectRecord],
    caption_map: Dict[str, str],
    text_source: str = "caption",
):
    """Reference candidate pool for *evaluation*, NOT generation.

    Mirrors training `build_texts()` with the saved text_source option:
    captions with category fallback, or category prompts for all objects.
    The caller must pass ALL records of the requested split, not only queried
    records, and train-caption generation must use a different candidate bank.
    """
    if text_source not in ("caption", "category"):
        raise ValueError(f"Unexpected checkpoint text_source: {text_source!r}")
    candidates = []
    texts = []
    seen = set()
    missing = []
    for r in records:
        if r.uid in seen:
            continue
        seen.add(r.uid)
        candidates.append(r)
        caption = caption_map.get(r.uid) if text_source == "caption" else None
        if not caption:
            if text_source == "caption":
                missing.append(r.uid)
            caption = f"a 3D object of {pretty_category(r.category)}"
        texts.append(caption)

    if not candidates:
        raise RuntimeError("Object identification candidate bank is empty")
    if missing:
        print(
            f"[Warning] {len(missing)} evaluation UIDs lack Cap3D captions; "
            "using E3 training's category-prompt fallback. Examples: "
            f"{missing[:5]}", flush=True,
        )
    return candidates, texts


def expected_nway_top1_per_query(
    similarities: torch.Tensor,
    targets: torch.Tensor,
    n_way: int,
) -> np.ndarray:
    """Exact E3 expected N-way accuracy per query (uniform distractors).

    Every candidate set contains the ground-truth object and `n_way - 1`
    distinct distractors drawn uniformly from the OTHER objects in the same
    evaluation split. This computes the expectation over all candidate sets
    rather than randomly sampling candidate sets. Equal scores are resolved
    by uniform random tie-breaking in expectation.

    The counting logic is deliberately kept identical to E3 training's
    `expected_nway_top1` to make validation results comparable.
    """
    if similarities.ndim != 2:
        raise ValueError(f"Expected similarities [N,C], got {tuple(similarities.shape)}")
    num_queries, num_candidates = similarities.shape
    if targets.ndim != 1 or len(targets) != num_queries:
        raise ValueError("Object target count does not match the number of queries")
    if not (2 <= n_way <= num_candidates):
        raise ValueError(f"n_way must be in [2,{num_candidates}], got {n_way}")

    sims = similarities.detach().cpu().double()
    targets = targets.detach().cpu().long()
    k = n_way - 1
    denom = math.comb(num_candidates - 1, k)
    probs = []

    for i in range(num_queries):
        target_idx = int(targets[i].item())
        if not (0 <= target_idx < num_candidates):
            raise ValueError(f"Target index out of range: {target_idx}")
        pos = float(sims[i, target_idx].item())
        row = sims[i]
        other_mask = torch.ones(num_candidates, dtype=torch.bool)
        other_mask[target_idx] = False
        others = row[other_mask]

        tol = 1e-12
        higher = int((others > pos + tol).sum().item())
        equal = int((torch.abs(others - pos) <= tol).sum().item())
        lower = (num_candidates - 1) - higher - equal

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

    return np.asarray(probs, dtype=np.float64)


def compute_object_identification_metrics(
    object_logits: torch.Tensor,
    object_targets: torch.Tensor,
):
    """Return split-level metrics plus per-query N-way expectations and ranks."""
    logits = object_logits.detach().cpu().float()
    targets = object_targets.detach().cpu().long()
    if not torch.isfinite(logits).all():
        raise ValueError("Object logits contain non-finite values")

    n, c = logits.shape
    if n == 0 or c < 2:
        raise ValueError("Object identification requires queries and >=2 candidates")

    ranks = (torch.argsort(logits, dim=1, descending=True)
             == targets[:, None]).nonzero(as_tuple=False)
    if ranks.shape[0] != n:
        raise RuntimeError("Every evaluation query must have exactly one positive UID")
    ranks = (ranks[:, 1] + 1).cpu().numpy()

    p2 = expected_nway_top1_per_query(logits, targets, n_way=2)
    p10 = (expected_nway_top1_per_query(logits, targets, n_way=10)
           if c >= 10 else None)
    metrics = {
        "object_ce": float(F.cross_entropy(logits, targets).item()),
        "object_top1": float((ranks == 1).mean()),
        "object_top5": float((ranks <= min(5, c)).mean()),
        "object_2way_top1": float(np.mean(p2)),
        "object_10way_top1": float(np.mean(p10)) if p10 is not None else None,
        "num_object_candidates": int(c),
        "num_queries": int(n),
    }
    return metrics, ranks, p2, p10


def load_caption_map(csv_path: Path) -> Dict[str, str]:
    """
    Observed MinD-3D++ format:
        category,uid,caption

    Split at most twice so commas inside captions remain part of the caption.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    mapping: Dict[str, str] = {}

    with csv_path.open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue

            parts = line.split(",", 2)
            if len(parts) != 3:
                continue

            category_raw, uid_raw, caption_raw = parts
            category = category_raw.strip().strip('"').strip("'")
            uid_field = uid_raw.strip().strip('"').strip("'")
            caption = caption_raw.strip()

            if category.lower() == "category" and uid_field.lower() == "uid":
                continue

            uid = normalize_uid(uid_field)
            if len(caption) >= 2 and caption[0] == '"' and caption[-1] == '"':
                caption = caption[1:-1].replace('""', '"').strip()

            if uid and caption:
                mapping[uid] = caption

    if not mapping:
        raise ValueError(f"No captions parsed from {csv_path}")

    return mapping


def load_split_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# Model: identical to training
# =============================================================================

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
            torch.ones([]) * np.log(1.0 / 0.07)
        )

    def encode_pooled(self, fmri):
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
        return F.normalize(self.net(self.encode_pooled(fmri)), dim=-1)


class DualFMRIClipAligner(FMRIClipAligner):
    """Exactly the C6/E3 dual-head architecture; no training dependencies."""

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
        self.visual_logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / 0.07)
        )
        self.geometry_head = (
            nn.Sequential(
                nn.LayerNorm(d), nn.Linear(d, 256), nn.GELU(),
                nn.Linear(256, geometry_dim), nn.Sigmoid(),
            ) if geometry_dim > 0 else None
        )

    def forward_dual(self, fmri):
        pooled = self.encode_pooled(fmri)
        semantic = F.normalize(self.net(pooled), dim=-1)
        visual = F.normalize(self.visual_net(pooled), dim=-1)
        geometry = self.geometry_head(pooled) if self.geometry_head is not None else None
        return semantic, visual, geometry


def torch_load_checkpoint(path: Path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def load_fmri_decoder(checkpoint_path: str, device, require_visual: bool = False):
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    checkpoint = torch_load_checkpoint(ckpt_path, device="cpu")

    if "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain 'model' and 'config', as saved by "
            "train_fmri_clip_align_mind3dpp.py"
        )

    config = checkpoint["config"]

    required = [
        "clip_dim",
        "num_frames",
        "frame_dim",
        "hidden_dim",
        "temporal_layers",
        "temporal_heads",
        "dropout",
    ]
    missing = [k for k in required if k not in config]
    if missing:
        raise KeyError(f"Checkpoint config is missing: {missing}")

    common_kwargs = dict(
        clip_dim=int(config["clip_dim"]),
        num_frames=int(config["num_frames"]),
        frame_dim=int(config["frame_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        temporal_layers=int(config["temporal_layers"]),
        temporal_heads=int(config["temporal_heads"]),
        dropout=float(config["dropout"]),
    )
    dual_state = checkpoint.get("dual_model")
    if dual_state is None and require_visual:
        raise ValueError(
            "Fusion alpha < 1 requires checkpoint['dual_model'] with visual_net. "
            "This checkpoint contains only a Semantic Head."
        )
    if dual_state is not None:
        if any(k.startswith("classifier_head.") for k in dual_state):
            raise ValueError("C3 classifier checkpoint is not supported by this C6/E3 inference")
        if "visual_net.4.weight" not in dual_state:
            raise ValueError("dual_model lacks trained visual_net weights")
        geometry_dim = (
            int(dual_state["geometry_head.3.weight"].shape[0])
            if "geometry_head.3.weight" in dual_state else 0
        )
        # Infer Visual Head width from the actual tensor; enforce strict loading.
        visual_hidden_dim = int(dual_state["visual_net.1.weight"].shape[0])
        model = DualFMRIClipAligner(
            **common_kwargs,
            visual_hidden_dim=visual_hidden_dim,
            geometry_dim=geometry_dim,
        )
        model.load_state_dict(dual_state, strict=True)
        # Compatibility check: the semantic path in dual_model matches the
        # legacy model state_dict saved for existing inference.
        semantic_sd = {k: v for k, v in dual_state.items() if
                       not k.startswith(("visual_net.", "geometry_head."))
                       and k != "visual_logit_scale"}
        for k, v in checkpoint["model"].items():
            if k not in semantic_sd or not torch.equal(semantic_sd[k], v):
                raise RuntimeError(f"dual_model and model semantic weights disagree: {k}")
    else:
        model = FMRIClipAligner(**common_kwargs)
        model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    return model, config, checkpoint


# =============================================================================
# fMRI inference dataset
# =============================================================================

class FMRIEvalDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        subject: str,
        records: Sequence[ObjectRecord],
        num_frames: int,
        fmri_norm: str,
    ):
        self.data_root = Path(data_root)
        self.subject = subject
        self.records = list(records)
        self.num_frames = int(num_frames)
        self.fmri_norm = str(fmri_norm)

        self.subject_dir = self.data_root / subject
        if not self.subject_dir.is_dir():
            raise FileNotFoundError(self.subject_dir)

        missing = []
        for r in self.records:
            p = self.subject_dir / f"{r.uid}.npy"
            if not p.is_file():
                missing.append(str(p))

        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} fMRI files are missing.\nFirst missing:\n{preview}"
            )

    def __len__(self):
        return len(self.records)

    def _select_middle_frames(self, fmri: np.ndarray):
        if fmri.ndim == 4 and fmri.shape[1] == 1:
            fmri = fmri[:, 0]

        if fmri.ndim != 3:
            raise ValueError(
                f"Expected [frames,H,W], got {fmri.shape}"
            )

        n = fmri.shape[0]
        if n < self.num_frames:
            raise ValueError(
                f"Need >= {self.num_frames} frames, got {n}"
            )

        start = (n - self.num_frames) // 2
        ids = np.arange(start, start + self.num_frames)
        return fmri[ids]

    def _normalize(self, x: torch.Tensor):
        if self.fmri_norm == "none":
            return x

        if self.fmri_norm == "sample_zscore":
            return (x - x.mean()) / x.std().clamp_min(1e-6)

        if self.fmri_norm == "frame_zscore":
            mean = x.mean(dim=(-2, -1), keepdim=True)
            std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return (x - mean) / std

        raise ValueError(f"Unknown fmri_norm: {self.fmri_norm}")

    def __getitem__(self, idx):
        r = self.records[idx]
        path = self.subject_dir / f"{r.uid}.npy"

        x = np.load(path)
        x = self._select_middle_frames(np.asarray(x))
        x = torch.from_numpy(
            np.asarray(x, dtype=np.float32).copy()
        )
        x = self._normalize(x)

        return {
            "fmri": x,
            "uid": r.uid,
            "category": r.category,
            "object_path": r.object_path,
        }


# =============================================================================
# CLIP text candidates
# =============================================================================

@torch.no_grad()
def encode_texts(
    texts: Sequence[str],
    clip_model,
    tokenizer,
    device,
    batch_size=256,
):
    chunks = []

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Encoding text candidates",
    ):
        end = min(start + batch_size, len(texts))
        tokens = tokenizer(list(texts[start:end])).to(device)
        emb = clip_model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1).cpu()
        chunks.append(emb)

    return torch.cat(chunks, dim=0)


def build_train_caption_candidates(
    full_train_records: Sequence[ObjectRecord],
    caption_map: Dict[str, str],
    split_json,
):
    # Fail closed: fMRI E3 uses UID-level train/val splitting within train_list.
    # Falling back to the whole train_list would leak held-out validation UIDs
    # into the generation prompt bank.
    if not split_json or not split_json.get("train_uids"):
        raise ValueError("A non-empty split.json train_uids list is required")
    allowed = set(split_json["train_uids"])
    available = {r.uid for r in full_train_records}
    missing_uids = allowed - available
    if missing_uids:
        raise ValueError(
            "split.json includes training UIDs absent from train_list.txt: "
            f"{sorted(missing_uids)[:5]}"
        )

    records = [r for r in full_train_records if r.uid in allowed]
    if not records:
        raise RuntimeError("No train-caption candidates were found")

    # Match the E3 training text bank: missing Cap3D captions fall back to
    # category prompts, without ever adding non-training UIDs.
    texts = [
        caption_map.get(r.uid) or f"a 3D object of {pretty_category(r.category)}"
        for r in records
    ]
    missing_captions = sum(r.uid not in caption_map for r in records)
    if missing_captions:
        print(
            f"[Warning] {missing_captions} train UIDs lack Cap3D captions; "
            "using category prompts as in E3 training.",
            flush=True,
        )
    return records, texts, "split.json train_uids"


def build_category_candidates(
    full_train_records: Sequence[ObjectRecord],
):
    categories = sorted({r.category for r in full_train_records})
    prompts = [
        f"a 3D object of {pretty_category(c)}"
        for c in categories
    ]
    return categories, prompts


def normalize_candidate_scores(scores: torch.Tensor, kind: str) -> torch.Tensor:
    """Identical to C1/C6 fusion evaluation: per-query, per-modality bank zscore."""
    if kind == "none":
        return scores
    if kind == "zscore":
        return ((scores - scores.mean(dim=1, keepdim=True)) /
                scores.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6))
    raise ValueError(f"Unknown score normalization: {kind}")


def combine_train_bank_scores(semantic_scores: torch.Tensor,
                              visual_scores: torch.Tensor | None,
                              alpha: float, score_normalize: str) -> torch.Tensor:
    """Semantic score (alpha=1) or normalized Semantic-Visual score fusion."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--fusion_alpha must be in [0,1]")
    if alpha == 1.0:
        # Preserve the old semantic-only ranking; zscore would preserve it too.
        return semantic_scores
    if visual_scores is None or visual_scores.shape != semantic_scores.shape:
        raise ValueError("Visual and semantic TRAIN banks are absent/misaligned")
    return (alpha * normalize_candidate_scores(semantic_scores, score_normalize)
            + (1.0-alpha) * normalize_candidate_scores(visual_scores, score_normalize))


def load_train_visual_bank(cache_path: str, train_caption_records,
                           config, train_uid_set, val_uid_set, test_uid_set):
    """Load frozen GT multi-view CLIP features for TRAIN IDs only.

    Original E3 cache includes all train_list UIDs (train + val). Deliberately
    select only the exact caption candidate order; never add val/test targets.
    """
    path = Path(cache_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Visual CLIP cache not found: {path}")
    payload = torch_load_checkpoint(path, device="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid visual feature cache: {path}")
    meta = payload.get("metadata", {})
    features = payload.get("embeddings", {})
    if not isinstance(meta, dict) or not isinstance(features, dict):
        raise ValueError("Visual cache must have metadata and embeddings dictionaries")
    if meta.get("clip_model") != config.get("clip_model", "ViT-B-32"):
        raise ValueError("Visual CLIP cache model differs from checkpoint")
    if meta.get("clip_pretrained") != config.get("clip_pretrained", "openai"):
        raise ValueError("Visual CLIP cache weights differ from checkpoint")
    if int(meta.get("view_count", -1)) != int(config.get("visual_view_count", 6)):
        raise ValueError("Visual CLIP cache view count differs from checkpoint")
    candidates = [r.uid for r in train_caption_records]
    if len(set(candidates)) != len(candidates):
        raise ValueError("Duplicate TRAIN caption UID candidates")
    if not set(candidates).issubset(train_uid_set):
        raise RuntimeError("Visual candidate bank includes non-training UIDs")
    if set(candidates) & (val_uid_set | test_uid_set):
        raise RuntimeError("Validation/test UIDs detected in TRAIN candidate list")
    absent = [uid for uid in candidates if uid not in features]
    if absent:
        raise ValueError(f"Visual cache is missing {len(absent)} TRAIN UIDs: {absent[:5]}")
    # Array order is EXACTLY train_caption_records order, not sorted cache order.
    bank = torch.stack([torch.as_tensor(features[uid]).float().reshape(-1)
                        for uid in candidates])
    if bank.ndim != 2 or bank.shape[1] != int(config["clip_dim"]):
        raise ValueError(f"Visual cache embeddings have wrong shape {tuple(bank.shape)}")
    if not torch.isfinite(bank).all() or torch.any(bank.norm(dim=-1) < 1e-8):
        raise ValueError("Visual cache contains non-finite or zero embeddings")
    bank = F.normalize(bank, dim=-1)
    print(f"[*] Visual TRAIN bank: {len(candidates)} UIDs, aligned with captions "
          f"from {path}", flush=True)
    return bank


# =============================================================================
# Image utilities
# =============================================================================

def bytes_or_pil_to_rgba(value):
    if isinstance(value, Image.Image):
        return value.convert("RGBA")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGBA")
    return Image.open(value).convert("RGBA")


def tensor_to_pil(image_tensor: torch.Tensor):
    x = image_tensor.detach().float().cpu().clamp(0, 1)
    arr = (
        x.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(image: Image.Image):
    arr = (
        np.asarray(image.convert("RGB"), dtype=np.float32)
        / 255.0
    )
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def save_tensor_image(image_tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(image_tensor).save(path)


def clean_generated_grid_background(
    generated_grid,
    rembg_session,
    alpha_threshold=8,
    binary_alpha=False,
):
    _, height, width = generated_grid.shape
    tile_h = height // 3
    tile_w = width // 2

    clean = torch.ones(
        (3, height, width),
        dtype=torch.float32,
    )
    masks = torch.zeros(
        (1, height, width),
        dtype=torch.float32,
    )

    threshold = float(alpha_threshold) / 255.0

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)

        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w

        tile = generated_grid[:, y0:y1, x0:x1]
        tile_pil = tensor_to_pil(tile).convert("RGBA")

        rgba = rembg.remove(
            tile_pil,
            session=rembg_session,
            alpha_matting=False,
        )
        rgba = bytes_or_pil_to_rgba(rgba)
        rgba_np = np.asarray(rgba, dtype=np.uint8)

        rgb = rgba_np[..., :3].astype(np.float32) / 255.0
        alpha = rgba_np[..., 3].astype(np.float32) / 255.0

        alpha[alpha < threshold] = 0.0

        if binary_alpha:
            alpha = (alpha >= 0.5).astype(np.float32)

        clean_rgb = (
            rgb * alpha[..., None]
            + (1.0 - alpha[..., None])
        )

        clean_tile = (
            torch.from_numpy(clean_rgb)
            .permute(2, 0, 1)
            .contiguous()
        )
        mask_tile = (
            torch.from_numpy(alpha)
            .unsqueeze(0)
            .contiguous()
        )

        clean[:, y0:y1, x0:x1] = clean_tile
        masks[:, y0:y1, x0:x1] = mask_tile

    return clean.clamp(0, 1), masks.clamp(0, 1)


def split_grid_to_six_views(grid_tensor):
    _, height, width = grid_tensor.shape
    tile_h = height // 3
    tile_w = width // 2

    views = []

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w
        views.append(
            grid_tensor[:, y0:y1, x0:x1].contiguous()
        )

    return views


def save_six_individual_views(grid_tensor, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for view_idx, view in enumerate(
        split_grid_to_six_views(grid_tensor)
    ):
        save_tensor_image(
            view,
            output_dir / f"{view_idx:02d}.png",
        )


# =============================================================================
# SDXL / Zero123++
# =============================================================================

def require_generation_dependencies():
    if rembg is None:
        raise ImportError(
            "rembg is required. Example: pip install 'rembg[cpu]==2.0.61'"
        )

    if (
        StableDiffusionXLPipeline is None
        or DiffusionPipeline is None
        or EulerAncestralDiscreteScheduler is None
    ):
        raise ImportError(
            "diffusers is required for SDXL/Zero123++ generation"
        )


def load_generation_pipelines(args, device):
    require_generation_dependencies()

    dtype = (
        torch.float16
        if device.type == "cuda"
        else torch.float32
    )

    print("[*] Loading SDXL:", args.sdxl_model)
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.sdxl_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    lora_path = args.sdxl_lora_path

    if args.sdxl_lora_repo_id and args.sdxl_lora_filename:
        lora_path = hf_hub_download(
            repo_id=args.sdxl_lora_repo_id,
            filename=args.sdxl_lora_filename,
        )

    if lora_path:
        if not os.path.isfile(lora_path):
            raise FileNotFoundError(lora_path)
        sdxl_pipe.load_lora_weights(lora_path)
        print(
            f"[*] LoRA loaded: {lora_path}, scale={args.sdxl_lora_scale}"
        )

    if args.cpu_offload and device.type == "cuda":
        sdxl_pipe.enable_model_cpu_offload()
    else:
        sdxl_pipe = sdxl_pipe.to(device)

    sdxl_pipe.set_progress_bar_config(disable=True)

    print("[*] Loading Zero123++:", args.zero123_model)
    zero123_pipe = DiffusionPipeline.from_pretrained(
        args.zero123_model,
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=dtype,
    )

    zero123_pipe.scheduler = (
        EulerAncestralDiscreteScheduler.from_config(
            zero123_pipe.scheduler.config,
            timestep_spacing="trailing",
        )
    )

    # Zero123++ uses a third-party custom pipeline which manually moves its
    # preprocessed image to `self.vae.device` BEFORE calling `vae.encode`.
    # With Diffusers/Accelerate model CPU offload, `vae.device` may report CPU
    # even though its forward hook moves VAE weights to CUDA. This creates the
    # CPU-input/CUDA-weight mismatch seen inside `vae.encoder.conv_in`.
    #
    # Keep SDXL's CPU offload, but never attach model CPU-offload hooks to the
    # Zero123++ custom pipeline. It must execute as an ordinary GPU pipeline.
    # For constrained VRAM, --zero123_cpu_swap keeps it on CPU between calls.
    zero123_swap = bool(getattr(args, "zero123_cpu_swap", False)) and device.type == "cuda"
    if args.cpu_offload and device.type == "cuda":
        print("[*] CPU offload: SDXL enabled; Zero123++ custom pipeline uses "
              "explicit device placement (prevents VAE CPU/CUDA mismatch).", flush=True)
    if zero123_swap:
        zero123_pipe = zero123_pipe.to("cpu")
        print("[*] Zero123++ CPU swap enabled: move to GPU only during Zero123++ inference "
              "(lower VRAM, slower per sample).", flush=True)
    else:
        zero123_pipe = zero123_pipe.to(device)

    zero123_pipe.set_progress_bar_config(disable=True)

    rembg_session = rembg.new_session()

    return sdxl_pipe, zero123_pipe, rembg_session


@torch.inference_mode()
def run_sdxl_base_image(
    pipe,
    prompt,
    args,
    seed,
):
    effective_prompt = (
        f"A standalone solo object, {prompt}, only one object, "
        "orthographic side profile view, flat shading, "
        "zero perspective distortion, perfectly centered, "
        "isolated on a pure solid white background, "
        "high quality 3D asset"
    )

    generator_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    generator = torch.Generator(
        device=generator_device
    )
    generator.manual_seed(int(seed))

    has_lora = bool(
        args.sdxl_lora_path
        or (
            args.sdxl_lora_repo_id
            and args.sdxl_lora_filename
        )
    )

    cross_attention_kwargs = (
        {"scale": args.sdxl_lora_scale}
        if has_lora
        else None
    )

    result = pipe(
        prompt=effective_prompt,
        negative_prompt=args.sdxl_negative_prompt,
        height=args.sdxl_resolution,
        width=args.sdxl_resolution,
        num_inference_steps=args.sdxl_steps,
        guidance_scale=args.sdxl_cfg,
        cross_attention_kwargs=cross_attention_kwargs,
        generator=generator,
    )

    return result.images[0].convert("RGB")


def prepare_zero123_input(
    image,
    rembg_session,
):
    rgba = rembg.remove(
        image,
        session=rembg_session,
        alpha_matting=False,
    )
    rgba = bytes_or_pil_to_rgba(rgba)

    gray_bg = Image.new(
        "RGBA",
        rgba.size,
        (127, 127, 127, 255),
    )
    gray_bg.paste(rgba, (0, 0), rgba)

    return gray_bg.convert("RGB")


@torch.inference_mode()
def run_zero123plus(
    pipe,
    image,
    args,
    seed,
):
    generator_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    generator = torch.Generator(
        device=generator_device
    )
    generator.manual_seed(int(seed))

    result = pipe(
        image,
        num_inference_steps=args.zero123_steps,
        generator=generator,
    )

    return result.images[0].convert("RGB")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MinD-3D++ fMRI -> trained CLIP aligner -> semantic prompt "
            "-> SDXL -> Zero123++"
        )
    )

    p.add_argument(
        "--semantic_ckpt",
        required=True,
        help="fmri_clip_aligner_best.pt from the MinD-3D++ training script",
    )
    p.add_argument(
        "--data_path",
        required=True,
        help="MinD-3D++ fMRI-Objaverse root",
    )
    p.add_argument(
        "--subject",
        default="sub-0001/npy_data",
    )
    p.add_argument(
        "--eval_split",
        choices=["test", "val", "train"],
        default="test",
    )
    p.add_argument(
        "--split_json",
        default="",
        help=(
            "Optional split.json. If omitted, sibling split.json next to "
            "the checkpoint is used when present."
        ),
    )

    p.add_argument(
        "--train_list",
        default="",
    )
    p.add_argument(
        "--test_list",
        default="",
    )
    p.add_argument(
        "--caption_csv",
        default="",
    )

    p.add_argument(
        "--prompt_source",
        choices=["train_caption", "category", "hybrid"],
        default="train_caption",
        help=(
            "train_caption = nearest caption among TRAIN UIDs only; "
            "category = nearest training-category prompt; "
            "hybrid = predicted category + nearest train caption."
        ),
    )

    p.add_argument(
        "--fusion_alpha", type=float, default=1.0,
        help=("Semantic/Visual TRAIN-bank fusion: alpha=1 semantic only; "
              "alpha=0.5 equal; alpha=0.25 visual 75%%. "
              "Choose on validation, not on official test."),
    )
    p.add_argument(
        "--visual_cache", default="",
        help="E3/C6 frozen GT 6-view CLIP cache (.pt), required when fusion_alpha < 1.",
    )
    p.add_argument(
        "--score_normalize", choices=["none", "zscore"], default="zscore",
        help="Per-query modality scores normalized over the TRAIN UID candidates, "
             "matching evaluate_fmri_classification.py.",
    )

    p.add_argument(
        "--batchsize",
        type=int,
        default=8,
        help="fMRI semantic inference batch size; generation is still sample-wise.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
    )
    p.add_argument(
        "--start_index",
        type=int,
        default=0,
    )

    p.add_argument(
        "--semantic_only",
        action="store_true",
        help="Only decode fMRI semantics and write CSV; skip SDXL/Zero123++.",
    )
    p.add_argument(
        "--no_object_metrics", action="store_true",
        help=(
            "Disable GT-caption object identification (default: enabled). "
            "Evaluation GT captions are never used as generation prompts."
        ),
    )

    p.add_argument(
        "--out_dir",
        required=True,
    )

    # SDXL / LoRA
    p.add_argument(
        "--sdxl_model",
        default=DEFAULT_SDXL_MODEL,
    )
    p.add_argument(
        "--sdxl_lora_path",
        default="",
    )
    p.add_argument(
        "--sdxl_lora_repo_id",
        default="",
    )
    p.add_argument(
        "--sdxl_lora_filename",
        default="",
    )
    p.add_argument(
        "--sdxl_lora_scale",
        type=float,
        default=0.7,
    )
    p.add_argument(
        "--sdxl_steps",
        type=int,
        default=30,
    )
    p.add_argument(
        "--sdxl_cfg",
        type=float,
        default=6.0,
    )
    p.add_argument(
        "--sdxl_resolution",
        type=int,
        default=1024,
    )
    p.add_argument(
        "--sdxl_negative_prompt",
        default=DEFAULT_SDXL_NEGATIVE_PROMPT,
    )

    # Zero123++
    p.add_argument(
        "--zero123_model",
        default=DEFAULT_ZERO123_MODEL,
    )
    p.add_argument(
        "--zero123_steps",
        type=int,
        default=75,
    )

    p.add_argument(
        "--bg_clean_alpha_threshold",
        type=int,
        default=8,
    )
    p.add_argument(
        "--bg_clean_binary_alpha",
        action="store_true",
    )

    p.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Enable SDXL model CPU offload. Zero123++ uses explicit device placement "
             "because its custom pipeline is incompatible with model CPU-offload hooks.",
    )
    p.add_argument(
        "--zero123_cpu_swap",
        action="store_true",
        help="Keep Zero123++ weights on CPU during SDXL inference and move them to "
             "the selected GPU only for Zero123++ generation. Reduces simultaneous "
             "GPU memory at the cost of large CPU/GPU transfers for every sample.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--device",
        default="cuda:0",
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

@torch.no_grad()
def main():
    args = parse_args()
    if not (0 <= args.fusion_alpha <= 1) or not np.isfinite(args.fusion_alpha):
        raise ValueError("--fusion_alpha must be finite and between 0 and 1")
    if args.fusion_alpha < 1 and not args.visual_cache:
        raise ValueError("--visual_cache must be specified for --fusion_alpha < 1")

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        or not args.device.startswith("cuda")
        else "cpu"
    )

    data_root = Path(args.data_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = out_dir / "base_images"
    render_dir = out_dir / "render"
    views_dir = out_dir / "views"

    for d in (base_dir, render_dir, views_dir):
        d.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        Path(args.semantic_ckpt)
        .expanduser()
        .resolve()
    )

    print("[*] Loading C6/E3 fMRI checkpoint...")
    model, config, checkpoint = load_fmri_decoder(
        str(checkpoint_path), device,
        require_visual=(args.fusion_alpha < 1),
    )
    print(f"[*] Fusion alpha={args.fusion_alpha:.2f} "
          f"(score_normalize={args.score_normalize}), dual_model="
          f"{isinstance(model, DualFMRIClipAligner)}", flush=True)

    print("[*] Checkpoint epoch:", checkpoint.get("epoch"))
    print("[*] Checkpoint preprocessing:")
    print("    num_frames =", config["num_frames"])
    print("    fmri_norm  =", config.get("fmri_norm", "none"))
    print("    clip_model =", config.get("clip_model", "ViT-B-32"))
    print("    pretrained =", config.get("clip_pretrained", "openai"))

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

    full_train_records = read_object_list(
        train_list_path
    )
    test_records = read_object_list(
        test_list_path
    )
    caption_map = load_caption_map(
        caption_csv_path
    )

    # Resolve the exact train/val split used by the checkpoint when available.
    if args.split_json:
        split_json_path = (
            Path(args.split_json)
            .expanduser()
            .resolve()
        )
    else:
        split_json_path = (
            checkpoint_path.parent / "split.json"
        )

    split_json = load_split_json(
        split_json_path
    )

    if split_json is None:
        raise FileNotFoundError(
            "The E3 training split.json is required to restrict generation "
            "caption retrieval to training UIDs only. Place split.json next "
            "to --semantic_ckpt or pass --split_json explicitly: "
            f"{split_json_path}"
        )
    if not split_json.get("train_uids"):
        raise ValueError("split.json must contain a non-empty train_uids list")
    print("[*] Using split metadata:", split_json_path)

    # Select evaluation split. Keep the COMPLETE split for evaluation object
    # candidates even if only a subset of fMRI queries is being generated.
    if args.eval_split == "test":
        full_eval_records = test_records

    elif args.eval_split in ("train", "val"):
        if split_json is None:
            raise ValueError(
                f"--eval_split {args.eval_split!r} requires split.json"
            )

        key = (
            "train_uids"
            if args.eval_split == "train"
            else "val_uids"
        )
        uid_set = set(split_json[key])

        full_eval_records = [
            r for r in full_train_records
            if r.uid in uid_set
        ]
        if {r.uid for r in full_eval_records} != uid_set:
            raise RuntimeError(
                f"split.json {key} and train_list.txt contain different UID sets"
            )

    else:
        raise ValueError(args.eval_split)

    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    eval_records = full_eval_records[
        args.start_index:
    ]

    if args.max_samples is not None:
        eval_records = eval_records[
            :args.max_samples
        ]

    if not eval_records:
        raise RuntimeError("No inference records remain")

    if split_json is not None:
        train_uid_set = set(split_json.get("train_uids", []))
        val_uid_set = set(split_json.get("val_uids", []))
        test_uid_set = {r.uid for r in test_records}
        if train_uid_set & val_uid_set:
            raise RuntimeError("train and validation UID lists overlap")
        if (train_uid_set | val_uid_set) & test_uid_set:
            raise RuntimeError("Official test UIDs overlap train/val split")

    print(
        f"[*] fMRI evaluation split={args.eval_split}: "
        f"queries={len(eval_records)} / full_split={len(full_eval_records)}",
        flush=True,
    )

    # Same deterministic evaluation preprocessing as training validation/test.
    dataset = FMRIEvalDataset(
        data_root=data_root,
        subject=args.subject,
        records=eval_records,
        num_frames=int(config["num_frames"]),
        fmri_norm=str(
            config.get("fmri_norm", "none")
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    # -------------------------------------------------------------------------
    # Build text space using the SAME OpenCLIP model as training.
    # -------------------------------------------------------------------------
    clip_model_name = str(
        config.get("clip_model", "ViT-B-32")
    )
    clip_pretrained = str(
        config.get("clip_pretrained", "openai")
    )

    print(
        f"[*] Loading OpenCLIP {clip_model_name} ({clip_pretrained}) "
        "for inference candidates..."
    )

    clip_model, _, _ = (
        open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
        )
    )
    clip_model = clip_model.to(device).eval()
    clip_model.requires_grad_(False)

    tokenizer = open_clip.get_tokenizer(
        clip_model_name
    )

    category_names, category_prompts = (
        build_category_candidates(
            # Training constructs its global category vocabulary from BOTH
            # train_list and test_list for stable category indices/prompts.
            full_train_records + test_records
        )
    )
    category_embeddings = encode_texts(
        category_prompts,
        clip_model,
        tokenizer,
        device,
    )

    (
        train_caption_records,
        train_caption_texts,
        train_caption_source,
    ) = build_train_caption_candidates(
        full_train_records,
        caption_map,
        split_json,
    )

    train_caption_embeddings = encode_texts(
        train_caption_texts,
        clip_model,
        tokenizer,
        device,
    )

    # This is deliberately a SECOND, evaluation-only bank.  The GT caption of
    # an unseen val/test UID MUST NOT enter train_caption_embeddings or the
    # prompt branch below.  Metrics use all distinct UIDs of the requested
    # split, exactly like evaluate_retrieval() in E3 training.
    object_eval_candidates = []
    object_eval_embeddings = None
    object_eval_uid_to_index = {}
    if not args.no_object_metrics:
        object_eval_candidates, object_eval_texts = (
            build_eval_object_caption_candidates(
                full_eval_records, caption_map,
                text_source=str(config.get("text_source", "caption")),
            )
        )
        object_eval_uid_to_index = {
            r.uid: i for i, r in enumerate(object_eval_candidates)
        }
        missing = [r.uid for r in eval_records if r.uid not in object_eval_uid_to_index]
        if missing:
            raise RuntimeError(
                f"{len(missing)} evaluation query UIDs are absent from the "
                f"{args.eval_split} object-candidate bank: {missing[:5]}"
            )
        # Sanity check: held-out GT captions must not leak into prompt bank.
        if args.eval_split in ("val", "test"):
            candidate_uids = {r.uid for r in object_eval_candidates}
            prompt_uids = {r.uid for r in train_caption_records}
            overlapping = candidate_uids & prompt_uids
            if overlapping:
                raise RuntimeError(
                    "Evaluation objects entered the generation prompt bank: "
                    f"{sorted(overlapping)[:5]}. Check split.json."
                )
        object_eval_embeddings = encode_texts(
            object_eval_texts, clip_model, tokenizer, device,
        )
        print(
            f"[*] Object {args.eval_split} evaluation: "
            f"{len(object_eval_candidates)} distinct GT-caption candidates "
            "(metrics ONLY; never used for generation)",
            flush=True,
        )

    train_visual_embeddings = None
    if args.fusion_alpha < 1:
        train_visual_embeddings = load_train_visual_bank(
            args.visual_cache, train_caption_records, config,
            train_uid_set, val_uid_set, test_uid_set,
        )
        # Fail before generation if any candidate order or UID lookup is bad.
        if train_visual_embeddings.shape[0] != len(train_caption_texts):
            raise RuntimeError("Visual TRAIN bank and caption bank sizes disagree")

    print(
        f"[*] Train-caption candidate pool: "
        f"{len(train_caption_records)} ({train_caption_source})"
    )
    print(
        f"[*] Category candidate pool: {len(category_names)}"
    )

    # No longer needed after candidate embedding creation.
    clip_model = clip_model.cpu()
    del clip_model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Generation models are loaded only if requested.
    # -------------------------------------------------------------------------
    if args.semantic_only:
        sdxl_pipe = None
        zero123_pipe = None
        rembg_session = None
    else:
        sdxl_pipe, zero123_pipe, rembg_session = (
            load_generation_pipelines(
                args,
                device,
            )
        )

    records_out = []
    category_correct = 0
    category_top5_correct = 0
    train_caption_category_correct = 0
    train_caption_category_top5_correct = 0
    object_logits_rows = []
    object_target_rows = []
    total = 0
    global_idx = args.start_index

    category_embeddings = category_embeddings.to(device)
    train_caption_embeddings = (
        train_caption_embeddings.to(device)
    )
    if train_visual_embeddings is not None:
        train_visual_embeddings = train_visual_embeddings.to(device)
    if object_eval_embeddings is not None:
        object_eval_embeddings = object_eval_embeddings.to(device)

    for batch in tqdm(
        loader,
        desc="fMRI -> CLIP semantics -> SDXL -> Zero123++",
    ):
        fmri = batch["fmri"].to(
            device,
            non_blocking=True,
        ).float()

        if isinstance(model, DualFMRIClipAligner):
            brain, brain_visual, _ = model.forward_dual(fmri)
        else:
            brain, brain_visual = model(fmri), None

        scale = (
            model.logit_scale
            .detach()
            .exp()
            .clamp(max=100.0)
        )

        category_logits = (
            scale
            * brain
            @ category_embeddings.T
        )
        semantic_scores = brain @ train_caption_embeddings.T
        visual_scores = (
            brain_visual @ train_visual_embeddings.T
            if train_visual_embeddings is not None else None
        )
        # Evaluation's fusion implementation scores all TRAIN objects in
        # precisely the same candidate order, then normalizes each modality
        # across that row/bank before weighted averaging.
        caption_scores = combine_train_bank_scores(
            semantic_scores, visual_scores,
            args.fusion_alpha, args.score_normalize,
        )
        caption_logits = scale * caption_scores
        if object_eval_embeddings is not None:
            eval_object_logits = (
                scale * (brain @ object_eval_embeddings.T)
            )
            object_logits_rows.append(eval_object_logits.float().cpu())
            object_target_rows.extend(
                object_eval_uid_to_index[str(uid)]
                for uid in batch["uid"]
            )

        cat_top5 = (
            category_logits
            .topk(
                min(5, category_logits.shape[1]),
                dim=1,
            )
            .indices
            .cpu()
            .tolist()
        )
        cat_pred = (
            category_logits
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        cat_conf = (
            category_logits
            .softmax(dim=1)
            .max(dim=1)
            .values
            .cpu()
            .tolist()
        )

        cap_top5 = (
            caption_logits
            .topk(
                min(5, caption_logits.shape[1]),
                dim=1,
            )
            .indices
            .cpu()
            .tolist()
        )
        cap_pred = (
            caption_logits
            .argmax(dim=1)
            .cpu()
            .tolist()
        )
        cap_conf = (
            caption_logits
            .softmax(dim=1)
            .max(dim=1)
            .values
            .cpu()
            .tolist()
        )

        batch_size = len(batch["uid"])

        for i in range(batch_size):
            true_uid = str(batch["uid"][i])
            true_category = str(
                batch["category"][i]
            )

            pred_category = category_names[
                int(cat_pred[i])
            ]
            pred_caption_record = (
                train_caption_records[
                    int(cap_pred[i])
                ]
            )
            pred_caption = (
                train_caption_texts[
                    int(cap_pred[i])
                ]
            )

            top5_categories = [
                category_names[int(j)]
                for j in cat_top5[i]
            ]
            top5_caption_uids = [
                train_caption_records[int(j)].uid
                for j in cap_top5[i]
            ]

            is_category_correct = int(
                pred_category == true_category
            )
            is_category_top5 = int(
                true_category in top5_categories
            )
            top5_caption_categories = [
                train_caption_records[int(j)].category
                for j in cap_top5[i]
            ]
            is_train_caption_category_correct = int(
                pred_caption_record.category == true_category
            )
            is_train_caption_category_top5 = int(
                true_category in top5_caption_categories
            )

            category_correct += is_category_correct
            category_top5_correct += is_category_top5
            train_caption_category_correct += is_train_caption_category_correct
            train_caption_category_top5_correct += is_train_caption_category_top5
            total += 1

            if args.prompt_source == "category":
                generation_prompt = pretty_category(
                    pred_category
                )

            elif args.prompt_source == "train_caption":
                generation_prompt = pred_caption

            elif args.prompt_source == "hybrid":
                generation_prompt = (
                    f"{pretty_category(pred_category)}. "
                    f"{pred_caption}"
                )

            else:
                raise ValueError(args.prompt_source)

            sample_id = (
                f"{global_idx:05d}__"
                f"{sanitize_name(true_category)}__"
                f"{true_uid[:12]}"
            )

            base_image_path = ""
            render_grid_path = ""
            pred_view_dir = ""

            if not args.semantic_only:
                sample_seed = args.seed + global_idx

                base_image = run_sdxl_base_image(
                    sdxl_pipe,
                    generation_prompt,
                    args,
                    sample_seed,
                )

                base_image_path = str(
                    base_dir
                    / f"{sample_id}_base.png"
                )
                base_image.save(
                    base_image_path
                )

                cond_image = prepare_zero123_input(
                    base_image,
                    rembg_session,
                )

                # The custom Zero123++ pipeline expects inputs and model weights
                # on the same device. For low-memory runs, explicitly swap the
                # complete pipeline to CUDA for this call, then back to CPU.
                if args.zero123_cpu_swap and device.type == "cuda":
                    zero123_pipe = zero123_pipe.to(device)
                try:
                    grid_pil = run_zero123plus(
                        zero123_pipe,
                        cond_image,
                        args,
                        sample_seed,
                    )
                finally:
                    if args.zero123_cpu_swap and device.type == "cuda":
                        zero123_pipe = zero123_pipe.to("cpu")
                        torch.cuda.empty_cache()

                raw_grid = (
                    pil_to_tensor(grid_pil)
                    .clamp(0, 1)
                )

                clean_grid, _ = (
                    clean_generated_grid_background(
                        raw_grid,
                        rembg_session,
                        alpha_threshold=args.bg_clean_alpha_threshold,
                        binary_alpha=args.bg_clean_binary_alpha,
                    )
                )

                render_grid_path = str(
                    render_dir / f"{sample_id}.png"
                )
                save_tensor_image(
                    clean_grid,
                    render_grid_path,
                )

                pred_view_dir = str(
                    views_dir / sample_id
                )
                save_six_individual_views(
                    clean_grid,
                    pred_view_dir,
                )

                del (
                    base_image,
                    cond_image,
                    grid_pil,
                    raw_grid,
                    clean_grid,
                )
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            records_out.append(
                {
                    "sample_index": global_idx,
                    "uid": true_uid,
                    "true_category": true_category,
                    "pred_category": pred_category,
                    "category_top5": " | ".join(
                        top5_categories
                    ),
                    "category_confidence": float(
                        cat_conf[i]
                    ),
                    "category_correct": is_category_correct,
                    "category_top5_correct": is_category_top5,
                    "train_caption_category_correct": is_train_caption_category_correct,
                    "train_caption_category_top5_correct": is_train_caption_category_top5,
                    "retrieval_alpha": args.fusion_alpha,
                    "retrieval_score_normalize": args.score_normalize,
                    "retrieval_top1_score": float(caption_scores[i, int(cap_pred[i])].item()),
                    "pred_train_uid": pred_caption_record.uid,
                    "pred_train_category": pred_caption_record.category,
                    "pred_train_caption": pred_caption,
                    "caption_top5_train_uids": " | ".join(
                        top5_caption_uids
                    ),
                    "caption_confidence": float(
                        cap_conf[i]
                    ),
                    "prompt_source": args.prompt_source,
                    "generation_prompt": generation_prompt,
                    "base_image": base_image_path,
                    "render_grid": render_grid_path,
                    "views_dir": pred_view_dir,
                }
            )

            global_idx += 1

    # Object identification uses ONLY the held-out GT candidate bank, not
    # train-caption generation results.  It is therefore meaningful even when
    # the generation prompt was retrieved from a disjoint training UID.
    object_metrics = None
    if object_eval_embeddings is not None:
        object_logits = torch.cat(object_logits_rows, dim=0)
        object_targets = torch.tensor(object_target_rows, dtype=torch.long)
        object_metrics, object_ranks, object_p2, object_p10 = (
            compute_object_identification_metrics(object_logits, object_targets)
        )
        if len(records_out) != len(object_ranks):
            raise RuntimeError("Object metric rows do not match saved inference records")
        for i, record in enumerate(records_out):
            record["object_gt_caption_rank"] = int(object_ranks[i])
            record["object_2way_expected"] = float(object_p2[i])
            record["object_10way_expected"] = (
                float(object_p10[i]) if object_p10 is not None else None
            )

    csv_path = out_dir / "inference_records.csv"

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        fieldnames = list(
            records_out[0].keys()
        )
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records_out)

    summary = {
        "semantic_checkpoint": str(checkpoint_path),
        "retrieval_alpha": args.fusion_alpha,
        "retrieval_score_normalize": args.score_normalize,
        "retrieval_mode": ("semantic_only" if args.fusion_alpha == 1 else "semantic_visual_train_fusion"),
        "visual_cache": (str(Path(args.visual_cache).expanduser().resolve())
                          if args.fusion_alpha < 1 else None),
        "retrieval_score_definition": (
            "per-query per-modality cosine scores over identical TRAIN UID candidates; "
            "optional candidate-bank zscore; alpha * Semantic + "
            "(1-alpha) * Visual. Alpha=1 preserves legacy semantic-only scoring."
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "eval_split": args.eval_split,
        "subject": args.subject,
        "num_samples": total,
        "prompt_source": args.prompt_source,
        "caption_candidate_policy": (
            "training UIDs only; test captions are not generation candidates"
        ),
        "category_top1": (
            category_correct / total
            if total
            else None
        ),
        "category_top5": (
            category_top5_correct / total
            if total
            else None
        ),
        "train_caption_category_top1": (
            train_caption_category_correct / total if total else None
        ),
        "train_caption_category_top5": (
            train_caption_category_top5_correct / total if total else None
        ),
        "object_metrics": object_metrics,
        "object_metric_definition": (
            "SEMANTIC-only embedding-space exact expected N-way identification "
            "(independent of retrieval alpha); "
            "fMRI semantic embedding vs GT Cap3D caption embedding, "
            "positive + uniformly selected distinct negative UIDs "
            "from the COMPLETE evaluation split; reference captions "
            "are never generation candidates"
            if object_metrics is not None else None
        ),
        "num_train_caption_candidates": len(
            train_caption_records
        ),
        "num_category_candidates": len(
            category_names
        ),
        "semantic_only": bool(
            args.semantic_only
        ),
    }

    with (
        out_dir / "summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 72)
    print("[INFERENCE COMPLETE]")
    print("samples        :", total)
    print("eval split     :", args.eval_split)
    print("retrieval      :", f"Semantic {args.fusion_alpha:.2f} + Visual {1-args.fusion_alpha:.2f}",
          f"({args.score_normalize})")
    if object_metrics is not None:
        print("object N-way   : Semantic vs GT captions (NOT fused / NOT mesh)")
        print(
            "object GT pool :",
            object_metrics["num_object_candidates"],
            f"({args.eval_split} UIDs; eval only)",
        )
        print(
            "object 2-way  :",
            f"{object_metrics['object_2way_top1']:.4%}",
            "(exact expectation)",
        )
        print(
            "object 10-way :",
            f"{object_metrics['object_10way_top1']:.4%}"
            if object_metrics["object_10way_top1"] is not None else "N/A",
            "(exact expectation)",
        )
        print(
            "object Top-1  :",
            f"{object_metrics['object_top1']:.4%}",
        )
        print(
            "object Top-5  :",
            f"{object_metrics['object_top5']:.4%}",
        )
        print(
            "object CE     :",
            f"{object_metrics['object_ce']:.4f}",
        )
    print(
        "category Top-1:",
        f"{category_correct / total:.4%}"
        if total else "N/A",
    )
    print(
        "category Top-5:",
        f"{category_top5_correct / total:.4%}"
        if total else "N/A",
    )
    print("train-caption category metrics: top-k from "
          "the selected Semantic/Visual retrieval scores")
    print(
        "train caption category Top-1:",
        f"{train_caption_category_correct / total:.4%}"
        if total else "N/A",
    )
    print(
        "train caption category Top-5:",
        f"{train_caption_category_top5_correct / total:.4%}"
        if total else "N/A",
    )
    print(
        "prompt source :",
        args.prompt_source,
    )
    print(
        "caption pool  :",
        len(train_caption_records),
        "(TRAIN UIDs only)",
    )
    print("records        :", csv_path)
    print("=" * 72)


if __name__ == "__main__":
    main()
