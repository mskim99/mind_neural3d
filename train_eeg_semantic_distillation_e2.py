#!/usr/bin/env python3
"""
E2: Frozen E1-r semantic prior + fixed-scale EEG residual distillation
=====================================================================

Purpose
-------
Test whether continuous EEG information adds generation utility on top of the
already-trained E1-r semantic prior.

Strict DEV
----------
train  : object 00..05
val    : object 06
unused : object07
final  : 08/09 NOT INSTANTIATED

Frozen
------
- Oracle semantic generator (Zero123++ base + LoRA + semantic_to_cross)
- E1-r raw-EEG -> PCA -> semantic student
- PCA transform stored inside the E1-r student checkpoint
- fixed TRAIN-only category text prototypes

Trainable
---------
- residual_head only:
      PCA feature [512]
        -> LayerNorm
        -> Linear 512->64
        -> GELU
        -> Linear 64->1024
        -> L2 normalize

Condition
---------
base_semantic = normalize(p_EEG @ category_prototypes)
residual      = normalize(residual_head(PCA_feature))
E2_prior      = normalize(base_semantic + alpha * residual)

alpha is FIXED (default 0.03). There is no learnable gate.

Loss
----
L = MSE(v_E2, stopgrad(v_oracle))

No hard CE.
No soft semantic KL.
No exact CLIP regression.
No learnable gate.
No spatial EEG condition.
No deprecated EEG checkpoint.

Validation reports BOTH:
- E1 base condition (frozen semantic student only)
- E2 residual condition
against the same Oracle / unconditional generator using the same noisy latent,
noise realization, and timestep. Therefore E2_gain_delta directly measures the
added value of the residual branch.
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.mvdiffusion_semantic_prior import MVDiffusion
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# =============================================================================
# Reproducibility / strict split
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def object_suffix(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[-1]
    return ""


def category_from_name(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[0]
    return key


def selected_object_columns(dataset, suffixes):
    wanted = set(suffixes)
    cols = []
    for o in range(int(dataset.obj_num)):
        seen = {
            object_suffix(dataset.name_list[c, o])
            for c in range(int(dataset.cls_num))
        }
        if len(seen) != 1:
            raise RuntimeError(
                f"Object column {o} has inconsistent suffixes: {seen}"
            )
        suffix = next(iter(seen))
        if suffix in wanted:
            cols.append(o)

    found = {object_suffix(dataset.name_list[0, o]) for o in cols}
    missing = wanted - found
    if missing:
        raise RuntimeError(f"Missing object suffixes: {sorted(missing)}")
    return cols


def full_dataset_indices(dataset, suffixes):
    cols = selected_object_columns(dataset, suffixes)
    S = int(dataset.eeg_data.shape[0])
    C = int(dataset.cls_num)
    O = int(dataset.obj_num)
    R = int(dataset.trails_num)

    out = []
    for s in range(S):
        for c in range(C):
            for o in cols:
                for r in range(R):
                    out.append(
                        s * (C * O * R) + c * (O * R) + o * R + r
                    )
    return out


class EEGOnlyDataset(Dataset):
    def __init__(self, base, suffixes, mode="individual"):
        self.base = base
        self.mode = mode
        self.cols = selected_object_columns(base, suffixes)
        S = int(base.eeg_data.shape[0])
        C = int(base.cls_num)
        R = int(base.trails_num)

        self.items = []
        for s in range(S):
            for c in range(C):
                for o in self.cols:
                    if mode == "individual":
                        for r in range(R):
                            self.items.append((s, c, o, r))
                    elif mode == "averaged":
                        self.items.append((s, c, o, None))
                    else:
                        raise ValueError(mode)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s, c, o, r = self.items[idx]
        if r is None:
            eeg = np.asarray(
                self.base.eeg_data[s, c, o, :], dtype=np.float32
            ).mean(axis=0, dtype=np.float32)
            trial_index = -1
        else:
            eeg = np.asarray(
                self.base.eeg_data[s, c, o, r], dtype=np.float32
            )
            trial_index = int(r)

        return {
            "eeg_data": torch.from_numpy(eeg),
            "cls_index": int(c),
            "obj_index": int(o),
            "trial_index": trial_index,
            "name": str(self.base.name_list[c, o]),
        }


# =============================================================================
# TRAIN-only category text geometry
# =============================================================================
def to_flat_feature(x):
    if torch.is_tensor(x):
        return x.detach().float().cpu().reshape(-1)
    return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1)


def build_fixed_category_text_prototypes(
    base_dataset,
    source_suffixes,
    expected_dim=1024,
):
    source_cols = selected_object_columns(base_dataset, source_suffixes)
    prototypes, categories, source_objects = [], [], []

    for c in range(int(base_dataset.cls_num)):
        feats, cats, objs = [], [], []
        for o in source_cols:
            name = str(base_dataset.name_list[c, o])
            key = name[3:]
            cat = category_from_name(name)
            feat = to_flat_feature(base_dataset.clip_features[key]["text"])
            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"{key}: expected text dim {expected_dim}, got {feat.numel()}"
                )
            feats.append(F.normalize(feat, dim=0))
            cats.append(cat)
            objs.append(key)

        if len(set(cats)) != 1:
            raise RuntimeError(
                f"cls_index={c} maps to multiple categories: {set(cats)}"
            )

        prototypes.append(
            F.normalize(torch.stack(feats).mean(dim=0), dim=0)
        )
        categories.append(cats[0])
        source_objects.append(objs)

    prototypes = torch.stack(prototypes, dim=0)
    if tuple(prototypes.shape) != (72, expected_dim):
        raise RuntimeError(
            f"Expected [72,{expected_dim}], got {tuple(prototypes.shape)}"
        )
    return prototypes, categories, source_objects


# =============================================================================
# Frozen E1 semantic student (PCA E1-r OR temporal E1-T)
# =============================================================================
def preprocess_raw_eeg(eeg: torch.Tensor) -> torch.Tensor:
    """Per-sample, per-channel z-normalization, identical to PCA E1-r."""
    eeg = eeg.float()
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    eeg = (eeg - mean) / std
    return eeg.reshape(eeg.shape[0], -1)


def normalize_eeg_channels(eeg: torch.Tensor) -> torch.Tensor:
    """EEG [B,64,600] -> deterministic per-sample/per-channel z-normalization."""
    eeg = eeg.float()
    if eeg.ndim != 3 or eeg.shape[1:] != (64, 600):
        raise RuntimeError(f"Expected EEG [B,64,600], got {tuple(eeg.shape)}")
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (eeg - mean) / std


class RawEEGSemanticStudent(nn.Module):
    """Legacy PCA-based E1-r student."""
    def __init__(
        self,
        pca_mean,
        pca_components,
        hidden_dim=256,
        num_classes=72,
    ):
        super().__init__()
        pca_mean = torch.as_tensor(pca_mean, dtype=torch.float32).reshape(-1)
        pca_components = torch.as_tensor(
            pca_components, dtype=torch.float32
        )
        self.register_buffer("pca_mean", pca_mean, persistent=True)
        self.register_buffer(
            "pca_components", pca_components, persistent=True
        )

        pca_dim = int(pca_components.shape[0])
        raw_dim = int(pca_components.shape[1])
        if raw_dim != 64 * 600:
            raise RuntimeError(
                f"Expected raw EEG dimension {64*600}, got {raw_dim}"
            )

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


class TemporalEEGSemanticStudent(nn.Module):
    """Low-capacity temporal E1-T student used by the revised E1."""
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=72):
        super().__init__()
        if int(feature_dim) != 512:
            raise ValueError(
                "temporal_dwconv_v1 currently expects feature_dim=512"
            )

        self.temporal = nn.Sequential(
            nn.Conv1d(
                64, 64, kernel_size=15, stride=2, padding=7,
                groups=64, bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv1d(
                128, 128, kernel_size=9, stride=2, padding=4,
                groups=128, bias=False,
            ),
            nn.GroupNorm(16, 128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.GroupNorm(32, 256),
            nn.GELU(),
        )

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.adapter = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def encode_temporal(self, eeg):
        x = normalize_eeg_channels(eeg)
        x = self.temporal(x)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        pooled = torch.cat([mean, std], dim=-1)
        return self.feature_proj(pooled)

    def forward(self, eeg, temperature=0.5):
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        feat = self.encode_temporal(eeg)
        logits = self.adapter(feat)
        probs = F.softmax(logits / float(temperature), dim=-1)
        return logits, probs, feat


def load_frozen_e1_student(path, device):
    """Load ONLY the revised temporal E1-T checkpoint."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"E1 checkpoint not found: {path}")

    pack = torch.load(path, map_location="cpu")
    if not isinstance(pack, dict) or "student" not in pack:
        raise RuntimeError(
            "Temporal E1 checkpoint must contain checkpoint['student']."
        )

    state = pack["student"]
    stored_arch = str(pack.get("student_arch", "")).strip()
    has_temporal_keys = any(k.startswith("temporal.") for k in state)

    if stored_arch != "temporal_dwconv_v1" and not has_temporal_keys:
        raise RuntimeError(
            "This script requires the revised temporal E1-T checkpoint, not the "
            "legacy PCA E1-r checkpoint. "
            f"student_arch={stored_arch!r}; first_keys={list(state)[:8]}"
        )

    required = [
        "feature_proj.1.weight",
        "adapter.1.weight",
        "adapter.3.weight",
    ]
    missing = [k for k in required if k not in state]
    if missing:
        raise RuntimeError(
            f"Unexpected temporal E1-T checkpoint layout; missing={missing}"
        )

    feature_dim = int(state["feature_proj.1.weight"].shape[0])
    hidden_dim = int(state["adapter.1.weight"].shape[0])
    num_classes = int(state["adapter.3.weight"].shape[0])

    student = TemporalEEGSemanticStudent(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )
    student.load_state_dict(state, strict=True)
    student.to(device)
    student.requires_grad_(False)
    student.eval()

    e1_args = pack.get("args", {}) or {}
    e1_holdout = str(e1_args.get("holdout_suffix", ""))
    print(
        f"[E1-T] loaded {path}; step={pack.get('step', 'unknown')}; "
        f"arch=temporal_dwconv_v1; feature_dim={feature_dim}; "
        f"hidden={hidden_dim}; classes={num_classes}"
    )
    return student, pack, e1_holdout, feature_dim, "temporal_dwconv_v1"


# =============================================================================
# E2 fixed-scale residual head
# =============================================================================
class EEGResidualHead(nn.Module):
    def __init__(self, feature_dim=512, rank=64, out_dim=1024, dropout=0.0):
        super().__init__()
        layers = [
            nn.LayerNorm(int(feature_dim)),
            nn.Linear(int(feature_dim), int(rank)),
            nn.GELU(),
        ]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(rank), int(out_dim)))
        self.net = nn.Sequential(*layers)

        # Small random initialization. Do NOT zero-initialize a vector that is
        # immediately L2-normalized: the gradient around an exact zero vector
        # is numerically ill-conditioned. E1 is evaluated separately as an
        # explicit frozen baseline, so E2 does not need to start identically.
        final = self.net[-1]
        nn.init.normal_(final.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final.bias)

    def forward(self, eeg_feat):
        x = self.net(eeg_feat.float())
        return F.normalize(x, dim=-1, eps=1e-6)


def combine_semantic_and_residual(base_prior, residual, alpha):
    if alpha < 0:
        raise ValueError("residual alpha must be >= 0")
    return F.normalize(
        base_prior.float() + float(alpha) * residual.float(),
        dim=-1,
    )


# =============================================================================
# Oracle generator
# =============================================================================
def build_generator_model(args, out_dir):
    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)

    # Constructor compatibility only. Old EEG prior modules are unused.
    OmegaConf.update(cfg, "semantic_prior_pca_dim", 512, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_rank", 64, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_scale", 0.0, merge=False)
    OmegaConf.update(cfg, "semantic_prior_spatial_scale", 0.0, merge=False)
    OmegaConf.update(
        cfg,
        "semantic_prior_temperature",
        float(args.student_temperature),
        merge=False,
    )

    stable_cfg = OmegaConf.select(
        cfg, "model.params.stable_diffusion_config", default=None
    )
    fmri_cfg = OmegaConf.select(
        cfg, "model.params.fmri_encoder_config", default=None
    )
    if stable_cfg is None:
        raise KeyError(
            f"{config_path}: missing model.params.stable_diffusion_config"
        )

    OmegaConf.save(cfg, Path(out_dir) / "resolved_config.yaml")
    return MVDiffusion(
        cfg,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
    )


def load_oracle_checkpoint(model, path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Oracle checkpoint not found: {path}")
    pack = torch.load(path, map_location="cpu")
    if not isinstance(pack, dict) or "model" not in pack:
        raise RuntimeError("Oracle checkpoint must contain checkpoint['model'].")
    missing, unexpected = model.load_state_dict(pack["model"], strict=False)
    if missing:
        print(f"[oracle load warning] missing={missing}")
    if unexpected:
        print(f"[oracle load warning] unexpected={unexpected}")
    print(f"[oracle] loaded {path}; step={pack.get('step', 'unknown')}")
    return pack


def freeze_generator(model):
    model.requires_grad_(False)
    model.eval()
    model.fmri_encoder.eval()
    model.semantic_to_cross.eval()
    model.unet.eval()
    model.pipeline.vae.eval()
    model.pipeline.text_encoder.eval()
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Generator freeze failed.")
    print("[generator] fully frozen")


# =============================================================================
# Conditioning / diffusion
# =============================================================================
def semantic_prior_from_probs(probs, prototypes):
    return F.normalize(probs @ prototypes.float(), dim=-1)


def build_oracle_condition(model, labels, unconditional=False):
    B = labels.shape[0]
    dtype = next(model.pipeline.unet.parameters()).dtype
    empty_prompt = model.get_empty_text_embeds(B)

    if unconditional:
        spatial = torch.zeros(
            B, 4, 64, 64, device=labels.device, dtype=dtype
        )
        return empty_prompt, spatial

    with torch.autocast("cuda", enabled=False):
        prior = F.normalize(
            model.fixed_text_prototypes[labels].float(), dim=-1
        )
        semantic_cond = model.semantic_to_cross(prior).unsqueeze(1)

    semantic_cond = semantic_cond.to(dtype)
    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(1, -1, 1)
    prompt = empty_prompt + semantic_cond * ramp
    spatial = torch.zeros(
        B, 4, 64, 64, device=labels.device, dtype=dtype
    )
    return prompt, spatial


def build_condition_from_prior(model, prior):
    B = prior.shape[0]
    dtype = next(model.pipeline.unet.parameters()).dtype
    empty_prompt = model.get_empty_text_embeds(B)

    with torch.autocast("cuda", enabled=False):
        semantic_cond = model.semantic_to_cross(prior.float()).unsqueeze(1)

    semantic_cond = semantic_cond.to(dtype)
    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(1, -1, 1)
    prompt = empty_prompt + semantic_cond * ramp
    spatial = torch.zeros(
        B, 4, 64, 64, device=prior.device, dtype=dtype
    )
    return prompt, spatial


def prepare_noisy_latents(model, batch, device):
    _, target_imgs = model.prepare_batch_data(batch)
    labels = batch["cls_index"].to(
        device, dtype=torch.long, non_blocking=True
    )
    B = labels.shape[0]
    t = torch.randint(0, model.num_timesteps, (B,), device=device).long()
    latents = model.encode_target_images(target_imgs)
    noise = torch.randn_like(latents)
    noisy = model.train_scheduler.add_noise(latents, noise, t)
    v_target = model.get_v(latents, noise, t)
    return labels, t, noisy, v_target


def forward_unet(model, noisy, t, prompt, spatial):
    return model.forward_unet(noisy, t, prompt, spatial)


def gt_diffusion_loss(model, pred, target):
    loss, _ = model.compute_loss(pred, target)
    return loss


# =============================================================================
# Validation: E1 baseline vs E2 residual on identical noise/timestep
# =============================================================================
@torch.no_grad()
def evaluate_generation_effect(
    model,
    semantic_student,
    residual_head,
    val_subset,
    args,
    device,
):
    loader = DataLoader(
        val_subset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    semantic_student.eval()
    residual_head.eval()

    e1_distills = []
    e2_distills = []
    e1_gt = []
    e2_gt = []
    oracle_gt = []
    uncond_gt = []

    with torch.random.fork_rng(
        devices=[0] if device.type == "cuda" else []
    ):
        torch.manual_seed(args.seed + 8119)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 8119)

        for i, batch in enumerate(loader):
            if i >= args.val_diff_batches:
                break

            eeg = batch["eeg_data"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            labels, t, noisy, v_target = prepare_noisy_latents(
                model, batch, device
            )

            _, probs, feat = semantic_student(
                eeg, temperature=args.student_temperature
            )
            base_prior = semantic_prior_from_probs(
                probs, model.fixed_text_prototypes
            )
            residual = residual_head(feat)
            e2_prior = combine_semantic_and_residual(
                base_prior, residual, args.residual_alpha
            )

            e1_prompt, e1_spatial = build_condition_from_prior(
                model, base_prior
            )
            e2_prompt, e2_spatial = build_condition_from_prior(
                model, e2_prior
            )
            o_prompt, o_spatial = build_oracle_condition(
                model, labels, unconditional=False
            )
            u_prompt, u_spatial = build_oracle_condition(
                model, labels, unconditional=True
            )

            v_e1 = forward_unet(model, noisy, t, e1_prompt, e1_spatial)
            v_e2 = forward_unet(model, noisy, t, e2_prompt, e2_spatial)
            v_oracle = forward_unet(model, noisy, t, o_prompt, o_spatial)
            v_uncond = forward_unet(model, noisy, t, u_prompt, u_spatial)

            e1_distills.append(
                float(F.mse_loss(v_e1.float(), v_oracle.float()))
            )
            e2_distills.append(
                float(F.mse_loss(v_e2.float(), v_oracle.float()))
            )
            e1_gt.append(float(gt_diffusion_loss(model, v_e1, v_target)))
            e2_gt.append(float(gt_diffusion_loss(model, v_e2, v_target)))
            oracle_gt.append(
                float(gt_diffusion_loss(model, v_oracle, v_target))
            )
            uncond_gt.append(
                float(gt_diffusion_loss(model, v_uncond, v_target))
            )

    if not e2_gt:
        raise RuntimeError("No validation diffusion batches evaluated.")

    e1_diff = float(np.mean(e1_gt))
    e2_diff = float(np.mean(e2_gt))
    oracle_diff = float(np.mean(oracle_gt))
    uncond_diff = float(np.mean(uncond_gt))

    e1_gain = uncond_diff - e1_diff
    e2_gain = uncond_diff - e2_diff
    oracle_gain = uncond_diff - oracle_diff

    e1_recovery = e1_gain / oracle_gain if oracle_gain > 1e-8 else float("nan")
    e2_recovery = e2_gain / oracle_gain if oracle_gain > 1e-8 else float("nan")

    return {
        "e1_distill": float(np.mean(e1_distills)),
        "e2_distill": float(np.mean(e2_distills)),
        "e1_diff": e1_diff,
        "e2_diff": e2_diff,
        "oracle_diff": oracle_diff,
        "uncond_diff": uncond_diff,
        "e1_gain": e1_gain,
        "e2_gain": e2_gain,
        "oracle_gain": oracle_gain,
        "e1_recovery": e1_recovery,
        "e2_recovery": e2_recovery,
        "e2_gain_delta": e2_gain - e1_gain,
        "e2_recovery_delta": e2_recovery - e1_recovery,
        "n_batches": len(e2_gt),
    }


# =============================================================================
# Logging / checkpoint
# =============================================================================
def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_e2_checkpoint(
    path,
    residual_head,
    optimizer,
    scheduler,
    step,
    args,
    extra,
):
    torch.save(
        {
            "residual_head": residual_head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "e1_ckpt": str(Path(args.e1_ckpt).expanduser().resolve()),
            "oracle_ckpt": str(
                Path(args.oracle_ckpt).expanduser().resolve()
            ),
            "extra": extra,
        },
        path,
    )


def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = min(int(warmup_steps), max(0, int(total_steps) - 1))

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        remain = max(1, int(total_steps) - warmup_steps)
        return max(0.0, float(int(total_steps) - step) / float(remain))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "E2: freeze E1-r semantic student and learn only a fixed-scale "
            "continuous EEG residual against the frozen Oracle generator."
        )
    )
    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--oracle_ckpt", required=True)
    p.add_argument("--e1_ckpt", required=True)

    p.add_argument(
        "--holdout_suffix",
        default="06",
        choices=[f"{i:02d}" for i in range(7)],
    )
    p.add_argument("--student_temperature", type=float, default=0.50)

    p.add_argument("--residual_rank", type=int, default=64)
    p.add_argument("--residual_alpha", type=float, default=0.03)
    p.add_argument("--residual_dropout", type=float, default=0.0)

    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--residual_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=250)
    p.add_argument("--val_diff_batches", type=int, default=32)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    print(f"[code] E2-T temporal-only v2 | file={Path(__file__).resolve()}")
    seed_everything(args.seed)

    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be > 0")
    if args.residual_rank <= 0:
        raise ValueError("--residual_rank must be > 0")
    if not (0.0 < args.residual_alpha <= 0.25):
        raise ValueError("Use --residual_alpha in (0,0.25].")
    if args.student_temperature <= 0:
        raise ValueError("--student_temperature must be > 0")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[device] {device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    universe = tuple(f"{i:02d}" for i in range(7))
    train_suffixes = tuple(
        s for s in universe if s != args.holdout_suffix
    )
    val_suffixes = (args.holdout_suffix,)
    print(
        f"[strict] train={train_suffixes}; holdout={val_suffixes}; "
        "object07=NOT_USED; final08_09=NOT_USED"
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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

    train_subset = Subset(
        base, full_dataset_indices(base, train_suffixes)
    )
    val_subset = Subset(
        base, full_dataset_indices(base, val_suffixes)
    )
    print(
        f"[samples] generation_train={len(train_subset)} "
        f"generation_val={len(val_subset)}"
    )

    # Strict TRAIN-only category geometry.
    prototypes, categories, source_objects = (
        build_fixed_category_text_prototypes(base, train_suffixes)
    )
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": list(train_suffixes),
        },
        out_dir / "fixed_category_text_prototypes.pt",
    )

    # Frozen Oracle generator.
    model = build_generator_model(args, out_dir).to(device)
    oracle_pack = load_oracle_checkpoint(model, args.oracle_ckpt)
    model.set_fixed_prototypes(prototypes.to(device))
    freeze_generator(model)

    # Frozen E1 semantic student: supports legacy PCA E1-r and revised temporal E1-T.
    semantic_student, e1_pack, e1_holdout, e1_feature_dim, e1_arch = (
        load_frozen_e1_student(args.e1_ckpt, device)
    )
    if e1_holdout and e1_holdout != str(args.holdout_suffix):
        raise RuntimeError(
            f"E1 holdout={e1_holdout} but E2 requested holdout={args.holdout_suffix}."
        )

    e1_oracle = str(e1_pack.get("oracle_ckpt", ""))
    if e1_oracle:
        e1_oracle_resolved = str(Path(e1_oracle).expanduser().resolve())
        requested_oracle = str(Path(args.oracle_ckpt).expanduser().resolve())
        if e1_oracle_resolved != requested_oracle:
            raise RuntimeError(
                "E1 and E2 must use the same Oracle generator.\n"
                f"E1 oracle: {e1_oracle_resolved}\n"
                f"E2 oracle   : {requested_oracle}"
            )

    residual_head = EEGResidualHead(
        feature_dim=e1_feature_dim,
        rank=args.residual_rank,
        out_dim=1024,
        dropout=args.residual_dropout,
    ).to(device)

    n_res = sum(p.numel() for p in residual_head.parameters())
    print(
        f"[E2 residual] trainable={n_res/1e6:.4f}M "
        f"alpha={args.residual_alpha:.4f} rank={args.residual_rank}"
    )

    optimizer = torch.optim.AdamW(
        residual_head.parameters(),
        lr=args.residual_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer, args.max_steps, args.warmup_steps
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    train_iter = cycle_loader(train_loader)

    writer = SummaryWriter(str(out_dir / "logs"))
    log_path = out_dir / "train_log.csv"

    # Baseline validation BEFORE any E2 update. E1 is evaluated explicitly,
    # so we can measure E2 gain delta even though the residual head has a small
    # non-zero initialization for numerical stability.
    baseline = evaluate_generation_effect(
        model,
        semantic_student,
        residual_head,
        val_subset,
        args,
        device,
    )
    print(
        "[BASELINE] "
        f"E1Gain={baseline['e1_gain']:+.5f} "
        f"E1Recovery={baseline['e1_recovery']:+.3f} | "
        f"E2Gain={baseline['e2_gain']:+.5f} "
        f"Delta={baseline['e2_gain_delta']:+.5f}"
    )

    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    best_key = None
    best_step = 0

    while step < args.max_steps:
        batch = next(train_iter)
        eeg = batch["eeg_data"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        labels, t, noisy, _ = prepare_noisy_latents(model, batch, device)

        # Frozen Oracle teacher.
        with torch.no_grad():
            o_prompt, o_spatial = build_oracle_condition(
                model, labels, unconditional=False
            )
            v_teacher = forward_unet(
                model, noisy, t, o_prompt, o_spatial
            ).detach()

            # Frozen E1 semantic student and its 512-D EEG feature.
            _, probs, feat = semantic_student(
                eeg, temperature=args.student_temperature
            )
            base_prior = semantic_prior_from_probs(
                probs, model.fixed_text_prototypes
            )
            feat = feat.detach()
            base_prior = base_prior.detach()

        # ONLY residual_head is trainable.
        residual = residual_head(feat)
        e2_prior = combine_semantic_and_residual(
            base_prior, residual, args.residual_alpha
        )
        e2_prompt, e2_spatial = build_condition_from_prior(model, e2_prior)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            v_e2 = forward_unet(
                model, noisy, t, e2_prompt, e2_spatial
            )

        loss = F.mse_loss(v_e2.float(), v_teacher.float())
        (loss / args.accumulation_steps).backward()
        micro += 1

        if micro % args.accumulation_steps != 0:
            continue

        if args.grad_clip > 0:
            clip_grad_norm_(residual_head.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % args.print_every == 0:
            with torch.no_grad():
                residual_cos = float(
                    F.cosine_similarity(
                        base_prior.float(), residual.float(), dim=-1
                    ).mean()
                )
            print(
                f"[TRAIN] step={step:06d} "
                f"Distill={loss.item():.6f} "
                f"ResidualCos={residual_cos:+.4f}"
            )
            writer.add_scalar("train/distill", loss.item(), step)
            writer.add_scalar("train/residual_cos", residual_cos, step)

        if step % args.validate_every == 0 or step == args.max_steps:
            val = evaluate_generation_effect(
                model,
                semantic_student,
                residual_head,
                val_subset,
                args,
                device,
            )

            print(
                f"[VAL] step={step:06d} "
                f"E1Diff={val['e1_diff']:.5f} "
                f"E2Diff={val['e2_diff']:.5f} "
                f"OracleDiff={val['oracle_diff']:.5f} "
                f"UncondDiff={val['uncond_diff']:.5f} | "
                f"E1Gain={val['e1_gain']:+.5f} "
                f"E2Gain={val['e2_gain']:+.5f} "
                f"Delta={val['e2_gain_delta']:+.5f} | "
                f"E1Recovery={val['e1_recovery']:+.3f} "
                f"E2Recovery={val['e2_recovery']:+.3f} "
                f"DeltaRec={val['e2_recovery_delta']:+.3f} | "
                f"E2Distill={val['e2_distill']:.6f}"
            )

            row = {"step": step, **val}
            append_csv(log_path, row)
            for k, v in row.items():
                if k == "step" or not isinstance(v, (int, float)):
                    continue
                if np.isfinite(v):
                    writer.add_scalar(f"val/{k}", float(v), step)

            # Primary: E2's ADDED utility over the frozen E1-r baseline.
            # Secondary: absolute E2 utility over unconditional.
            # Tertiary: closer Oracle-effect match.
            key = (
                float(val["e2_gain_delta"]),
                float(val["e2_gain"]),
                -float(val["e2_distill"]),
            )

            extra = {
                "strict_train_suffixes": train_suffixes,
                "strict_holdout_suffix": args.holdout_suffix,
                "object07": "NOT_USED",
                "final08_09": "NOT_USED",
                "oracle_step": oracle_pack.get("step", None),
                "e1_step": e1_pack.get("step", None),
                "e1_student_arch": e1_arch,
                "e1_feature_dim": int(e1_feature_dim),
                "residual_alpha": float(args.residual_alpha),
                "residual_rank": int(args.residual_rank),
                "val": row,
            }

            save_e2_checkpoint(
                ckpt_dir / "last.pt",
                residual_head,
                optimizer,
                scheduler,
                step,
                args,
                extra,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                save_e2_checkpoint(
                    ckpt_dir / "best.pt",
                    residual_head,
                    optimizer,
                    scheduler,
                    step,
                    args,
                    extra,
                )
                print(
                    f"[BEST] step={step} "
                    f"E2Gain={val['e2_gain']:+.5f} "
                    f"GainDelta={val['e2_gain_delta']:+.5f} "
                    f"Recovery={val['e2_recovery']:+.3f}"
                )

        if step % args.save_every == 0:
            save_e2_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                residual_head,
                optimizer,
                scheduler,
                step,
                args,
                {
                    "strict_train_suffixes": train_suffixes,
                    "strict_holdout_suffix": args.holdout_suffix,
                },
            )

    writer.close()

    summary = {
        "best_step": int(best_step),
        "train_suffixes": list(train_suffixes),
        "holdout_suffix": args.holdout_suffix,
        "object07": "NOT_USED",
        "final08_09": "NOT_USED",
        "oracle_checkpoint": str(
            Path(args.oracle_ckpt).expanduser().resolve()
        ),
        "e1_checkpoint": str(Path(args.e1_ckpt).expanduser().resolve()),
        "deprecated_eeg_checkpoint_used": False,
        "semantic_student_frozen": True,
        "e1_student_arch": e1_arch,
        "e1_feature_dim": int(e1_feature_dim),
        "generator_frozen": True,
        "trainable_component": "residual_head_only",
        "residual_alpha": float(args.residual_alpha),
        "residual_rank": int(args.residual_rank),
        "best_checkpoint": str(ckpt_dir / "best.pt"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[done] best_step={best_step}; best={ckpt_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
