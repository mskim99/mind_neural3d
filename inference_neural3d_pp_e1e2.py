#!/usr/bin/env python3
"""
E1-r / E2 / Oracle / Unconditional six-view inference for Neuro-3D
===================================================================

This replaces the legacy end-to-end EEG inference path.

DEV protocol (default)
----------------------
train semantics/generator : objects 00..05
inference holdout          : object 06 from train=True EEG split
object07                   : NOT USED
final08/09                 : NOT LOADED

Condition modes
---------------
e1     : raw EEG -> frozen E1-r PCA/student -> soft semantic prior
e2     : E1 prior + fixed alpha * frozen E2 residual
oracle : GT class -> fixed category prototype (DEV diagnostic only)
uncond : empty text condition

All modes use the SAME Oracle Zero123++ generator and, within each batch,
the SAME initial diffusion noise. EEG spatial cond_lat is always zero, exactly
as in E1-r/E2 training.

Outputs
-------
<out_dir>/<mode>/render/<sample_id>.png
<out_dir>/<mode>/views/<sample_id>/00.png ... 05.png
<out_dir>/<mode>/evaluation_pairs.csv

The packed 3x2 layout remains row-major:
    0 | 1
    2 | 3
    4 | 5
No crop / recenter / resize / reorder is applied before optional per-view rembg.
"""

import os
import csv
import json
import shutil
import tempfile
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf

try:
    import rembg
except ImportError:
    rembg = None

from src.mvdiffusion_semantic_prior import MVDiffusion, unscale_image
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


VALID_MODES = ("e1", "e2", "oracle", "uncond")


def parse_args():
    p = argparse.ArgumentParser(
        description="E1-r/E2 semantic-prior six-view inference"
    )
    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument(
        "--gt_render_root",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)

    p.add_argument("--oracle_ckpt", required=True)
    p.add_argument("--e1_ckpt", required=True)
    p.add_argument(
        "--e2_ckpt",
        default=None,
        help="Required only when condition_modes contains e2.",
    )

    p.add_argument(
        "--condition_modes",
        nargs="+",
        choices=VALID_MODES,
        default=list(VALID_MODES),
        help="Generate one or more of: e1 e2 oracle uncond.",
    )

    p.add_argument(
        "--holdout_suffix",
        default="06",
        choices=[f"{i:02d}" for i in range(7)],
        help="DEV holdout object suffix. Default 06.",
    )
    p.add_argument(
        "--trial_mode",
        choices=["individual", "average"],
        default="individual",
        help="individual: 2 train trials/object; average: mean the 2 EEG trials.",
    )

    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--num_steps", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--student_temperature",
        type=float,
        default=None,
        help="Default: exact value stored in E1-r checkpoint.",
    )
    p.add_argument(
        "--residual_alpha",
        type=float,
        default=None,
        help="Default: exact value stored in E2 checkpoint (normally 0.03).",
    )

    p.add_argument(
        "--no_bg_clean",
        action="store_true",
        help="Disable per-view rembg. Default keeps the existing white-background cleanup.",
    )
    p.add_argument("--bg_clean_alpha_threshold", type=int, default=8)
    p.add_argument("--bg_clean_binary_alpha", action="store_true")
    p.add_argument("--allow_missing_gt", action="store_true")
    return p.parse_args()


# =============================================================================
# DEV holdout dataset: explicitly use train=True and select object06
# =============================================================================
def _object_suffix(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[-1]
    return ""


def _selected_object_column(base, suffix):
    matches = []
    for o in range(int(base.obj_num)):
        seen = {
            _object_suffix(base.name_list[c, o])
            for c in range(int(base.cls_num))
        }
        if len(seen) != 1:
            raise RuntimeError(
                f"object column {o} has inconsistent suffixes: {seen}"
            )
        if next(iter(seen)) == str(suffix):
            matches.append(o)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one object column for suffix={suffix}, got {matches}"
        )
    return int(matches[0])


class DevHoldoutDataset(Dataset):
    """Wrap the original train=True dataset but expose only one strict holdout object."""

    def __init__(self, base, holdout_suffix="06", trial_mode="individual"):
        self.base = base
        self.holdout_suffix = str(holdout_suffix)
        self.trial_mode = str(trial_mode)
        self.obj_col = _selected_object_column(base, self.holdout_suffix)

        self.S = int(base.eeg_data.shape[0])
        self.C = int(base.cls_num)
        self.O = int(base.obj_num)
        self.R = int(base.trails_num)
        self.items = []
        for s in range(self.S):
            for c in range(self.C):
                if self.trial_mode == "individual":
                    for r in range(self.R):
                        self.items.append((s, c, self.obj_col, r))
                else:
                    self.items.append((s, c, self.obj_col, None))

    def __len__(self):
        return len(self.items)

    def _base_linear_index(self, s, c, o, r):
        return s * (self.C * self.O * self.R) + c * (self.O * self.R) + o * self.R + r

    def __getitem__(self, index):
        s, c, o, r = self.items[index]
        label = str(self.base.name_list[c, o])[3:]
        class_prefix = label.rsplit("_", 1)[0]

        # Use the original item for image/name metadata when possible.
        base_r = 0 if r is None else int(r)
        base_idx = self._base_linear_index(s, c, o, base_r)
        item = dict(self.base[base_idx])

        if r is None:
            eeg = np.asarray(
                self.base.eeg_data[s, c, o, :], dtype=np.float32
            ).mean(axis=0, dtype=np.float32)
            trial_index = -1
        else:
            eeg = np.asarray(
                self.base.eeg_data[s, c, o, int(r)], dtype=np.float32
            )
            trial_index = int(r)

        item["eeg_data"] = torch.from_numpy(eeg)
        item["cls_index"] = int(c)
        item["obj_index"] = int(o)
        item["trial_index"] = trial_index
        item["subject_index"] = int(s)
        item["name"] = str(self.base.name_list[c, o])
        item["label"] = label
        item["class_prefix"] = class_prefix
        return item


# =============================================================================
# E1-r student and E2 residual: exact training definitions
# =============================================================================
def preprocess_raw_eeg(eeg: torch.Tensor) -> torch.Tensor:
    eeg = eeg.float()
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    eeg = (eeg - mean) / std
    return eeg.reshape(eeg.shape[0], -1)


class RawEEGSemanticStudent(nn.Module):
    def __init__(self, pca_mean, pca_components, hidden_dim=256, num_classes=72):
        super().__init__()
        pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32).reshape(-1)
        pca_components = torch.as_tensor(pca_components, dtype=torch.float32)
        self.register_buffer("pca_mean", pca_mean, persistent=True)
        self.register_buffer("pca_components", pca_components, persistent=True)
        pca_dim = int(pca_components.shape[0])
        raw_dim = int(pca_components.shape[1])
        if raw_dim != 64 * 600:
            raise RuntimeError(f"Expected raw EEG dim 38400, got {raw_dim}")
        self.adapter = nn.Sequential(
            nn.LayerNorm(pca_dim),
            nn.Linear(pca_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def encode_pca(self, eeg):
        raw = preprocess_raw_eeg(eeg)
        centered = raw - self.pca_mean.view(1, -1)
        return centered @ self.pca_components.t()

    def forward(self, eeg, temperature=0.5):
        feat = self.encode_pca(eeg)
        logits = self.adapter(feat)
        probs = F.softmax(logits / float(temperature), dim=-1)
        return logits, probs, feat


class EEGResidualHead(nn.Module):
    def __init__(self, pca_dim=512, rank=64, out_dim=1024, dropout=0.0):
        super().__init__()
        layers = [
            nn.LayerNorm(int(pca_dim)),
            nn.Linear(int(pca_dim), int(rank)),
            nn.GELU(),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(rank), int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, pca_feat):
        x = self.net(pca_feat.float())
        return F.normalize(x, dim=-1, eps=1e-6)


def load_e1_student(path, device):
    path = Path(path).expanduser().resolve()
    pack = torch.load(path, map_location="cpu")
    if "student" not in pack:
        raise RuntimeError(f"E1 checkpoint has no 'student': {path}")
    state = pack["student"]
    for k in ("pca_mean", "pca_components", "adapter.1.weight", "adapter.3.weight"):
        if k not in state:
            raise RuntimeError(f"E1 checkpoint missing key: {k}")
    hidden_dim = int(state["adapter.1.weight"].shape[0])
    num_classes = int(state["adapter.3.weight"].shape[0])
    student = RawEEGSemanticStudent(
        state["pca_mean"],
        state["pca_components"],
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    student.load_state_dict(state, strict=True)
    student.to(device).eval().requires_grad_(False)
    e1_args = pack.get("args", {}) or {}
    print(
        f"[*] E1-r loaded: {path} step={pack.get('step', 'unknown')} "
        f"PCA={tuple(state['pca_components'].shape)} hidden={hidden_dim}"
    )
    return student, pack, e1_args


def load_e2_residual(path, student, device):
    path = Path(path).expanduser().resolve()
    pack = torch.load(path, map_location="cpu")
    if "residual_head" not in pack:
        raise RuntimeError(f"E2 checkpoint has no 'residual_head': {path}")
    e2_args = pack.get("args", {}) or {}
    pca_dim = int(student.pca_components.shape[0])
    rank = int(e2_args.get("residual_rank", 64))
    dropout = float(e2_args.get("residual_dropout", 0.0))
    head = EEGResidualHead(
        pca_dim=pca_dim,
        rank=rank,
        out_dim=1024,
        dropout=dropout,
    )
    head.load_state_dict(pack["residual_head"], strict=True)
    head.to(device).eval().requires_grad_(False)
    print(
        f"[*] E2 residual loaded: {path} step={pack.get('step', 'unknown')} "
        f"rank={rank} alpha={float(e2_args.get('residual_alpha', 0.03)):.4f}"
    )
    return head, pack, e2_args


# =============================================================================
# Oracle generator / conditioning
# =============================================================================
def build_oracle_generator(config_path, oracle_ckpt, runtime_logdir, device, temperature):
    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)
    # Must match the constructor shape used by Oracle training.
    OmegaConf.update(cfg, "semantic_prior_pca_dim", 512, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_rank", 64, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_scale", 0.0, merge=False)
    OmegaConf.update(cfg, "semantic_prior_spatial_scale", 0.0, merge=False)
    OmegaConf.update(cfg, "semantic_prior_temperature", float(temperature), merge=False)

    stable_cfg = OmegaConf.select(cfg, "model.params.stable_diffusion_config", default=None)
    fmri_cfg = OmegaConf.select(cfg, "model.params.fmri_encoder_config", default=None)
    if stable_cfg is None:
        raise KeyError("config missing model.params.stable_diffusion_config")

    model = MVDiffusion(
        cfg,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=runtime_logdir,
        num_classes=72,
    ).to(device)

    pack = torch.load(oracle_ckpt, map_location="cpu")
    state = pack["model"] if isinstance(pack, dict) and "model" in pack else pack
    state = {
        (k[len("module."):] if str(k).startswith("module.") else str(k)): v
        for k, v in state.items()
    }
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[ORACLE CHECKPOINT MISMATCH]")
        print(" missing   :", list(incompatible.missing_keys)[:30])
        print(" unexpected:", list(incompatible.unexpected_keys)[:30])
        raise RuntimeError("Oracle checkpoint/model architecture mismatch.")

    model.requires_grad_(False)
    model.eval()
    print(f"[*] Oracle generator loaded: {oracle_ckpt} step={pack.get('step', 'unknown') if isinstance(pack, dict) else 'unknown'}")
    return model, pack


def semantic_prior_from_probs(probs, fixed_prototypes):
    return F.normalize(probs.float() @ fixed_prototypes.float(), dim=-1)


def build_prompt_from_prior(model, prior):
    B = prior.shape[0]
    dtype = next(model.pipeline.unet.parameters()).dtype
    empty = model.get_empty_text_embeds(B)
    with torch.autocast("cuda", enabled=False):
        semantic_cond = model.semantic_to_cross(prior.float()).unsqueeze(1)
    semantic_cond = semantic_cond.to(dtype)
    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(1, -1, 1)
    prompt = empty + semantic_cond * ramp
    spatial = torch.zeros(B, 4, 64, 64, device=prior.device, dtype=dtype)
    return prompt, spatial


def build_unconditional(model, batch_size, device):
    dtype = next(model.pipeline.unet.parameters()).dtype
    prompt = model.get_empty_text_embeds(batch_size)
    spatial = torch.zeros(batch_size, 4, 64, 64, device=device, dtype=dtype)
    return prompt, spatial


def build_mode_condition(
    mode,
    model,
    eeg,
    labels,
    semantic_student,
    residual_head,
    student_temperature,
    residual_alpha,
):
    B = eeg.shape[0]
    if mode == "uncond":
        return build_unconditional(model, B, eeg.device), None

    if mode == "oracle":
        prior = F.normalize(model.fixed_text_prototypes[labels].float(), dim=-1)
        return build_prompt_from_prior(model, prior), {
            "prior": prior,
            "logits": None,
            "probs": None,
            "pca_feat": None,
        }

    logits, probs, feat = semantic_student(
        eeg, temperature=student_temperature
    )
    base_prior = semantic_prior_from_probs(probs, model.fixed_text_prototypes)

    if mode == "e1":
        return build_prompt_from_prior(model, base_prior), {
            "prior": base_prior,
            "logits": logits,
            "probs": probs,
            "pca_feat": feat,
        }

    if mode == "e2":
        if residual_head is None:
            raise RuntimeError("condition mode e2 requires --e2_ckpt")
        residual = residual_head(feat)
        prior = F.normalize(
            base_prior.float() + float(residual_alpha) * residual.float(),
            dim=-1,
        )
        return build_prompt_from_prior(model, prior), {
            "prior": prior,
            "base_prior": base_prior,
            "residual": residual,
            "logits": logits,
            "probs": probs,
            "pca_feat": feat,
        }

    raise ValueError(mode)


def generate_with_cfg(
    model,
    prompt_cond,
    spatial_cond,
    initial_latents,
    num_steps,
    guidance_scale,
    sampling_seed,
):
    scheduler = model.pipeline.scheduler
    device = initial_latents.device
    dtype = initial_latents.dtype
    B = initial_latents.shape[0]

    scheduler.set_timesteps(int(num_steps), device=device)
    latents = initial_latents.clone()
    prompt_uncond, spatial_uncond = build_unconditional(model, B, device)

    prompt = torch.cat([prompt_uncond, prompt_cond], dim=0)
    spatial = torch.cat([spatial_uncond, spatial_cond], dim=0)

    amp_enabled = device.type == "cuda"
    # Some schedulers can sample additional noise inside scheduler.step().
    # Reset RNG for every condition mode so E1/E2/Oracle/Uncond share the
    # exact same stochastic denoising path, not only the same initial latent.
    fork_devices = [device.index if device.index is not None else 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(int(sampling_seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(sampling_seed))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            for t in scheduler.timesteps:
                latent_in = torch.cat([latents, latents], dim=0)
                latent_in = scheduler.scale_model_input(latent_in, t)
                pred = model.forward_unet(latent_in, t, prompt, spatial)
                pred_u, pred_c = pred.chunk(2)
                pred = pred_u + float(guidance_scale) * (pred_c - pred_u)
                latents = scheduler.step(pred, t, latents).prev_sample

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        images = model.pipeline.vae.decode(
            latents / model.pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        images = unscale_image(images)
        images = (images * 0.5 + 0.5).clamp(0, 1)
    return images



def _tensor_grid_to_pil(image_tensor):
    """
    [3,H,W] float tensor in [0,1] -> RGB PIL image.
    No normalization is applied.
    """
    x = image_tensor.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def _pil_to_tensor(image):
    """
    RGB PIL -> [3,H,W] float32 in [0,1].
    """
    arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _save_exact_tensor_image(image_tensor, path):
    """
    Save a [3,H,W] tensor directly as RGB PNG without make_grid/normalize.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _tensor_grid_to_pil(image_tensor).save(path)



def _canonical_label_from_dataset_name(name):
    """
    Original Neuro-3D dataset uses name[3:] as the object key for CLIP/point-cloud
    lookup. The GT renderer directory follows the same key, e.g.
        dataset name -> ...airplane_00
        label        -> airplane_00
    """
    name = str(name)
    if len(name) <= 3:
        raise ValueError(f"Dataset name is too short to derive label: {name!r}")
    return name[3:]


def split_grid_to_six_views(grid_tensor):
    """
    [3, H, W] packed grid -> list of six [3, H/3, W/2] tensors.

    Fixed canonical order:
        0 | 1
        2 | 3
        4 | 5

    NO resize / crop-to-object / recenter / flip / reorder.
    """
    if grid_tensor.ndim != 3 or grid_tensor.shape[0] != 3:
        raise ValueError(
            f"Expected RGB grid [3,H,W], got {tuple(grid_tensor.shape)}"
        )

    _, H, W = grid_tensor.shape
    if H % 3 != 0 or W % 2 != 0:
        raise ValueError(
            f"Grid must be divisible into 3x2 tiles, got H={H}, W={W}"
        )

    tile_h, tile_w = H // 3, W // 2
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
    """
    Save a packed prediction in the SAME representation as GT renderer output:
        <output_dir>/00.png
        ...
        <output_dir>/05.png
    """
    os.makedirs(output_dir, exist_ok=True)
    views = split_grid_to_six_views(grid_tensor)

    saved = []
    for view_idx, view in enumerate(views):
        path = os.path.join(output_dir, f"{view_idx:02d}.png")
        _save_exact_tensor_image(view, path)
        saved.append(path)
    return saved


def validate_gt_view_dir(gt_root, label):
    """
    Return (<gt_dir>, <ok>, <missing_files>).
    """
    gt_dir = os.path.join(gt_root, label)
    missing = [
        f"{i:02d}.png"
        for i in range(6)
        if not os.path.isfile(os.path.join(gt_dir, f"{i:02d}.png"))
    ]
    return gt_dir, len(missing) == 0, missing


def save_sample_metadata(
    out_dir,
    sample_id,
    dataset_name,
    label,
    gt_dir,
    raw_view_dir,
    clean_view_dir,
    raw_grid_path,
    clean_grid_path,
):
    meta_dir = os.path.join(out_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    payload = {
        "sample_id": sample_id,
        "dataset_name": dataset_name,
        "label": label,
        "view_order": [
            ["00", "01"],
            ["02", "03"],
            ["04", "05"],
        ],
        "gt_dir": gt_dir,
        "raw_view_dir": raw_view_dir,
        "clean_view_dir": clean_view_dir,
        "raw_grid": raw_grid_path,
        "clean_grid": clean_grid_path,
    }

    path = os.path.join(meta_dir, f"{sample_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def write_evaluation_manifests(out_dir, records):
    """
    Write:
      inference_index.csv   : full label/path mapping
      eval_pairs_raw.csv    : GT six views <-> raw predicted six views
      eval_pairs_clean.csv  : GT six views <-> cleaned predicted six views

    These CSVs enforce one-to-one matching by `label`.
    """
    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(out_dir, "inference_index.csv")
    fieldnames = [
        "sample_id",
        "dataset_name",
        "label",
        "gt_dir",
        "pred_raw_dir",
        "pred_clean_dir",
        "raw_grid",
        "clean_grid",
    ]

    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    raw_path = os.path.join(out_dir, "eval_pairs_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "label", "gt_dir", "pred_dir"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "sample_id": r["sample_id"],
                "label": r["label"],
                "gt_dir": r["gt_dir"],
                "pred_dir": r["pred_raw_dir"],
            })

    clean_path = os.path.join(out_dir, "eval_pairs_clean.csv")
    with open(clean_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "label", "gt_dir", "pred_dir"]
        )
        writer.writeheader()
        for r in records:
            if r.get("pred_clean_dir"):
                writer.writerow({
                    "sample_id": r["sample_id"],
                    "label": r["label"],
                    "gt_dir": r["gt_dir"],
                    "pred_dir": r["pred_clean_dir"],
                })

    return index_path, raw_path, clean_path




def write_evaluation_pairs(out_dir, records):
    """
    Root-level 1:1 GT/prediction mapping for evaluation.
    """
    csv_path = os.path.join(out_dir, "evaluation_pairs.csv")
    fieldnames = [
        "sample_id",
        "dataset_name",
        "label",
        "class_prefix",
        "cls_index",
        "obj_index",
        "trial_index",
        "subject_index",
        "gt_dir",
        "pred_dir",
        "render_grid",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    return csv_path


def prepare_output_layout(out_dir):
    """
    Keep only two output directories:
        render/ : cleaned packed 3x2 grids for InstantMesh
        views/  : cleaned 00.png ... 05.png folders for evaluation
    """
    os.makedirs(out_dir, exist_ok=True)

    legacy_dirs = [
        "raw_for_instantmesh",
        "raw_views",
        "clean_for_instantmesh",
        "clean_views",
        "gt_pred",
        "bg_masks",
        "bg_compare",
        "metadata",
        "images",
        "images_val",
    ]
    for name in legacy_dirs:
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)

    render_dir = os.path.join(out_dir, "render")
    views_dir = os.path.join(out_dir, "views")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    return render_dir, views_dir



def clean_generated_grid_background(
    generated_grid,
    rembg_session,
    alpha_threshold=8,
    binary_alpha=False,
):
    """
    Experiment B.

    Input:
        generated_grid: [3, 960, 640] (3 rows x 2 columns of 320x320 views)

    Processing:
        1. Split into the canonical six fixed tiles.
        2. Run rembg independently on each tile.
        3. Composite foreground onto exact RGB=(255,255,255).
        4. Reassemble in the SAME locations/order.

    IMPORTANT:
        - NO crop
        - NO recenter
        - NO resize
        - NO view reordering

    Returns:
        clean_grid: [3,H,W], float32 [0,1]
        mask_grid:  [1,H,W], float32 [0,1]
    """
    if rembg is None:
        raise ImportError(
            "Experiment B requires `rembg`. Install it in this environment, e.g. "
            "`pip install rembg onnxruntime-gpu` (or onnxruntime for CPU)."
        )

    if generated_grid.ndim != 3 or generated_grid.shape[0] != 3:
        raise ValueError(
            f"Expected generated grid [3,H,W], got {tuple(generated_grid.shape)}"
        )

    _, H, W = generated_grid.shape
    if H % 3 != 0 or W % 2 != 0:
        raise ValueError(
            f"Generated grid must be divisible into 3x2 tiles, got H={H}, W={W}"
        )

    tile_h = H // 3
    tile_w = W // 2

    clean = torch.ones((3, H, W), dtype=torch.float32)
    masks = torch.zeros((1, H, W), dtype=torch.float32)

    threshold = float(alpha_threshold) / 255.0

    for view_idx in range(6):
        row, col = divmod(view_idx, 2)
        y0, y1 = row * tile_h, (row + 1) * tile_h
        x0, x1 = col * tile_w, (col + 1) * tile_w

        tile = generated_grid[:, y0:y1, x0:x1]
        tile_pil = _tensor_grid_to_pil(tile).convert('RGBA')

        rgba = rembg.remove(
            tile_pil,
            session=rembg_session,
            alpha_matting=False,
        )
        if not isinstance(rgba, Image.Image):
            rgba = Image.open(rgba).convert('RGBA')
        else:
            rgba = rgba.convert('RGBA')

        rgba_np = np.asarray(rgba, dtype=np.uint8)
        rgb = rgba_np[..., :3].astype(np.float32) / 255.0
        alpha = rgba_np[..., 3].astype(np.float32) / 255.0

        # Suppress tiny residual alpha values that can otherwise become
        # large faint shells after 3D reconstruction.
        alpha[alpha < threshold] = 0.0

        if binary_alpha:
            alpha = (alpha >= 0.5).astype(np.float32)

        # Exact white background. Preserve original tile coordinates.
        clean_rgb = rgb * alpha[..., None] + (1.0 - alpha[..., None])

        clean_tile = torch.from_numpy(clean_rgb).permute(2, 0, 1).contiguous()
        mask_tile = torch.from_numpy(alpha).unsqueeze(0).contiguous()

        clean[:, y0:y1, x0:x1] = clean_tile
        masks[:, y0:y1, x0:x1] = mask_tile

    return clean.clamp(0, 1), masks.clamp(0, 1)


def save_experiment_b_outputs(
    out_dir,
    sample_idx,
    raw_grid,
    clean_grid,
    mask_grid=None,
):
    """
    Save files in paths that can be passed directly to InstantMesh.

    raw_for_instantmesh/infer_XXXXX.png
    clean_for_instantmesh/infer_XXXXX.png
    bg_masks/infer_XXXXX.png
    bg_compare/infer_XXXXX.png
    """
    stem = f"infer_{sample_idx:05d}.png"

    raw_path = os.path.join(out_dir, "raw_for_instantmesh", stem)
    clean_path = os.path.join(out_dir, "clean_for_instantmesh", stem)

    _save_exact_tensor_image(raw_grid, raw_path)
    _save_exact_tensor_image(clean_grid, clean_path)

    if mask_grid is not None:
        mask_rgb = mask_grid.repeat(3, 1, 1)
        mask_path = os.path.join(out_dir, "bg_masks", stem)
        _save_exact_tensor_image(mask_rgb, mask_path)

    # Diagnostic only: left/raw and right/clean.
    # Do NOT feed this comparison image to InstantMesh.
    compare = torch.cat([raw_grid, clean_grid], dim=2)
    compare_path = os.path.join(out_dir, "bg_compare", stem)
    _save_exact_tensor_image(compare, compare_path)

    return raw_path, clean_path





@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[*] device={device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    modes = []
    for m in args.condition_modes:
        if m not in modes:
            modes.append(m)
    if "e2" in modes and not args.e2_ckpt:
        raise ValueError("--e2_ckpt is required when condition_modes contains e2")

    runtime_logdir = tempfile.mkdtemp(prefix="neural3d_e1e2_infer_")

    # E1 first: its checkpoint defines the exact PCA/student architecture and temperature.
    semantic_student, e1_pack, e1_args = load_e1_student(args.e1_ckpt, device)
    e1_holdout = str(e1_args.get("holdout_suffix", ""))
    if e1_holdout and e1_holdout != str(args.holdout_suffix):
        raise RuntimeError(
            f"E1 checkpoint holdout={e1_holdout}, inference holdout={args.holdout_suffix}"
        )
    student_temperature = (
        float(args.student_temperature)
        if args.student_temperature is not None
        else float(e1_args.get("student_temperature", 0.50))
    )

    model, oracle_pack = build_oracle_generator(
        args.config,
        args.oracle_ckpt,
        runtime_logdir,
        device,
        temperature=student_temperature,
    )

    # Verify E1 was distilled from this Oracle when the path was stored.
    stored_e1_oracle = str(e1_pack.get("oracle_ckpt", ""))
    if stored_e1_oracle:
        print(f"[*] E1 stored oracle: {stored_e1_oracle}")

    residual_head = None
    e2_pack = None
    residual_alpha = None
    if args.e2_ckpt:
        residual_head, e2_pack, e2_args = load_e2_residual(
            args.e2_ckpt, semantic_student, device
        )
        e2_holdout = str(e2_args.get("holdout_suffix", ""))
        if e2_holdout and e2_holdout != str(args.holdout_suffix):
            raise RuntimeError(
                f"E2 checkpoint holdout={e2_holdout}, inference holdout={args.holdout_suffix}"
            )
        residual_alpha = (
            float(args.residual_alpha)
            if args.residual_alpha is not None
            else float(e2_args.get("residual_alpha", 0.03))
        )
        print(f"[*] inference residual_alpha={residual_alpha:.4f}")

    # IMPORTANT: train=True, then explicit object06 subset. Do not touch final08/09.
    print(f"[*] Preparing strict DEV holdout object{args.holdout_suffix}")
    base = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )
    dataset = DevHoldoutDataset(
        base,
        holdout_suffix=args.holdout_suffix,
        trial_mode=args.trial_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"[*] DEV samples={len(dataset)} trial_mode={args.trial_mode}")

    if args.no_bg_clean:
        rembg_session = None
        print("[*] background cleanup: OFF")
    else:
        if rembg is None:
            raise ImportError(
                "Background cleanup is enabled but rembg is unavailable. "
                "Install rembg/onnxruntime-gpu or pass --no_bg_clean."
            )
        rembg_session = rembg.new_session()
        print("[*] background cleanup: per-view rembg -> exact white")

    mode_layout = {}
    records = {m: [] for m in modes}
    for mode in modes:
        root = Path(args.out_dir) / mode
        render_root, views_root = prepare_output_layout(str(root))
        mode_layout[mode] = (root, Path(render_root), Path(views_root))

    dtype = next(model.pipeline.unet.parameters()).dtype
    conditioning_rows = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Generating 4-way")):
        eeg = batch["eeg_data"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        labels = batch["cls_index"].to(
            device, dtype=torch.long, non_blocking=True
        )
        B = eeg.shape[0]

        # One initial latent per EEG sample; clone it for every condition mode.
        gen = torch.Generator(device=device)
        gen.manual_seed(int(args.seed) + int(batch_idx))
        initial_latents = torch.randn(
            (B, 4, 960 // 8, 640 // 8),
            generator=gen,
            device=device,
            dtype=dtype,
        )
        initial_latents = initial_latents * model.pipeline.scheduler.init_noise_sigma

        # Compute EEG student once; mode helper remains explicit for clarity.
        cached_conditions = {}
        for mode in modes:
            (prompt_cond, spatial_cond), aux = build_mode_condition(
                mode,
                model,
                eeg,
                labels,
                semantic_student,
                residual_head,
                student_temperature,
                residual_alpha if residual_alpha is not None else 0.0,
            )
            cached_conditions[mode] = (prompt_cond, spatial_cond, aux)

        # Semantic metadata once per sample, using E1 student output.
        logits_e1, probs_e1, _ = semantic_student(
            eeg, temperature=student_temperature
        )
        pred_cls = logits_e1.argmax(dim=-1)
        entropy = -(
            probs_e1.clamp_min(1e-8) * probs_e1.clamp_min(1e-8).log()
        ).sum(dim=-1)

        for mode in modes:
            prompt_cond, spatial_cond, aux = cached_conditions[mode]
            images_pred = generate_with_cfg(
                model,
                prompt_cond,
                spatial_cond,
                initial_latents,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                sampling_seed=int(args.seed) + int(batch_idx),
            )

            root, render_root, views_root = mode_layout[mode]

            for i in range(B):
                label = str(batch["label"][i])
                class_prefix = str(batch["class_prefix"][i])
                dataset_name = str(batch["name"][i])
                cls_index = int(batch["cls_index"][i])
                obj_index = int(batch["obj_index"][i])
                trial_index = int(batch["trial_index"][i])
                subject_index = int(batch["subject_index"][i])

                # Never key predicted views only by label: two individual trials share the label.
                if args.trial_mode == "individual":
                    sample_id = f"{label}__trial{trial_index:02d}"
                else:
                    sample_id = f"{label}__avg"
                if subject_index != 0:
                    sample_id += f"__sub{subject_index:02d}"

                gt_dir, gt_ok, missing_gt = validate_gt_view_dir(
                    args.gt_render_root, label
                )
                if not gt_ok and not args.allow_missing_gt:
                    raise FileNotFoundError(
                        f"GT six-view missing for {label}: {missing_gt} in {gt_dir}"
                    )

                raw_pred = images_pred[i].detach().float().cpu().clamp(0, 1)
                if rembg_session is not None:
                    pred, _ = clean_generated_grid_background(
                        raw_pred,
                        rembg_session=rembg_session,
                        alpha_threshold=args.bg_clean_alpha_threshold,
                        binary_alpha=args.bg_clean_binary_alpha,
                    )
                else:
                    pred = raw_pred

                render_path = render_root / f"{sample_id}.png"
                _save_exact_tensor_image(pred, str(render_path))

                pred_view_dir = views_root / sample_id
                save_six_individual_views(pred, str(pred_view_dir))

                records[mode].append({
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "label": label,
                    "class_prefix": class_prefix,
                    "cls_index": cls_index,
                    "obj_index": obj_index,
                    "trial_index": trial_index,
                    "subject_index": subject_index,
                    "gt_dir": gt_dir,
                    "pred_dir": str(pred_view_dir),
                    "render_grid": str(render_path),
                })

                if mode == modes[0]:
                    conditioning_rows.append({
                        "sample_id": sample_id,
                        "label": label,
                        "true_cls_index": cls_index,
                        "e1_pred_cls_index": int(pred_cls[i].item()),
                        "e1_entropy": float(entropy[i].item()),
                        "trial_index": trial_index,
                    })

    for mode in modes:
        root, _, _ = mode_layout[mode]
        csv_path = write_evaluation_pairs(str(root), records[mode])
        print(f"[*] {mode:6s}: {len(records[mode])} samples -> {root}")
        print(f"    evaluation pairs: {csv_path}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cond_csv = out_root / "conditioning_index.csv"
    if conditioning_rows:
        with cond_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(conditioning_rows[0].keys()))
            writer.writeheader()
            writer.writerows(conditioning_rows)

    run_meta = {
        "config": str(Path(args.config).expanduser().resolve()),
        "oracle_ckpt": str(Path(args.oracle_ckpt).expanduser().resolve()),
        "e1_ckpt": str(Path(args.e1_ckpt).expanduser().resolve()),
        "e2_ckpt": str(Path(args.e2_ckpt).expanduser().resolve()) if args.e2_ckpt else None,
        "holdout_suffix": args.holdout_suffix,
        "trial_mode": args.trial_mode,
        "condition_modes": modes,
        "student_temperature": student_temperature,
        "residual_alpha": residual_alpha,
        "guidance_scale": args.guidance_scale,
        "num_steps": args.num_steps,
        "seed": args.seed,
        "protocol": "strict DEV: train=True object holdout only; final08/09 not loaded",
        "same_initial_noise_across_modes": True,
        "spatial_eeg_condition": "zero",
    }
    (out_root / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    shutil.rmtree(runtime_logdir, ignore_errors=True)
    print("[*] Inference complete.")
    print(f"[*] Root: {out_root}")


if __name__ == "__main__":
    main()
