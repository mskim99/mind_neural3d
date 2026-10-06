#!/usr/bin/env python3
"""
MinD-3D++ fMRI inference matched to train_fmri_clip_align_mind3dpp.py

Pipeline
--------
MinD-3D++ fMRI (.npy)
    -> FMRIClipAligner checkpoint
    -> CLIP-space semantic retrieval
       * train_caption: nearest caption among TRAIN UIDs only (default)
       * category: nearest training-category prompt
       * hybrid: predicted category + nearest TRAIN caption
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
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
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

            parts = line.split()
            if len(parts) < 2 and "," in line:
                parts = next(csv.reader([line]))
                parts = [p.strip() for p in parts if p.strip()]

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

    def forward(self, fmri):
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


def torch_load_checkpoint(path: Path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location=device)


def load_fmri_decoder(checkpoint_path: str, device):
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

    model = FMRIClipAligner(
        clip_dim=int(config["clip_dim"]),
        num_frames=int(config["num_frames"]),
        frame_dim=int(config["frame_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        temporal_layers=int(config["temporal_layers"]),
        temporal_heads=int(config["temporal_heads"]),
        dropout=float(config["dropout"]),
    )

    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.to(device).eval()

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
    if split_json is not None and split_json.get("train_uids"):
        allowed = set(split_json["train_uids"])
        source = "split.json train_uids"
    else:
        allowed = {r.uid for r in full_train_records}
        source = "all train_list UIDs"

    records = [
        r for r in full_train_records
        if r.uid in allowed and r.uid in caption_map
    ]

    if not records:
        raise RuntimeError("No train-caption candidates were found")

    texts = [caption_map[r.uid] for r in records]
    return records, texts, source


def build_category_candidates(
    full_train_records: Sequence[ObjectRecord],
):
    categories = sorted({r.category for r in full_train_records})
    prompts = [
        f"a 3D object of {pretty_category(c)}"
        for c in categories
    ]
    return categories, prompts


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

    if args.cpu_offload and device.type == "cuda":
        zero123_pipe.enable_model_cpu_offload()
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

    print("[*] Loading fMRI semantic checkpoint...")
    model, config, checkpoint = load_fmri_decoder(
        str(checkpoint_path),
        device,
    )

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

    if split_json is not None:
        print("[*] Using split metadata:", split_json_path)
    else:
        print(
            "[Warning] split.json not found. "
            "train_caption candidates will use all train_list UIDs."
        )

    # Select inference records.
    if args.eval_split == "test":
        eval_records = test_records

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

        eval_records = [
            r for r in full_train_records
            if r.uid in uid_set
        ]

    else:
        raise ValueError(args.eval_split)

    if args.start_index < 0:
        raise ValueError("--start_index must be >= 0")

    eval_records = eval_records[
        args.start_index:
    ]

    if args.max_samples is not None:
        eval_records = eval_records[
            :args.max_samples
        ]

    if not eval_records:
        raise RuntimeError("No inference records remain")

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
            full_train_records
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
    total = 0
    global_idx = args.start_index

    category_embeddings = category_embeddings.to(device)
    train_caption_embeddings = (
        train_caption_embeddings.to(device)
    )

    for batch in tqdm(
        loader,
        desc="fMRI -> CLIP semantics -> SDXL -> Zero123++",
    ):
        fmri = batch["fmri"].to(
            device,
            non_blocking=True,
        ).float()

        brain = model(fmri)

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
        caption_logits = (
            scale
            * brain
            @ train_caption_embeddings.T
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

            category_correct += is_category_correct
            category_top5_correct += is_category_top5
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

                grid_pil = run_zero123plus(
                    zero123_pipe,
                    cond_image,
                    args,
                    sample_seed,
                )

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
