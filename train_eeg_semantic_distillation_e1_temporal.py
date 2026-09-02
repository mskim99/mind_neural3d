#!/usr/bin/env python3
"""
E1-T: Temporal EEG Semantic Distillation
========================================

Goal
----
Replace the fixed raw-EEG PCA front-end of E1-r with a small trainable temporal
encoder while keeping the Oracle semantic generator fully frozen.

Strict DEV protocol
-------------------
train  : object 00..05
val    : object 06
unused : object 07
final  : object 08/09 are NOT instantiated

Student
-------
EEG [B,64,600]
  -> per-sample/per-channel z-normalization
  -> depthwise temporal Conv1d(k=15,s=2)
  -> pointwise Conv1d(64->128)
  -> depthwise temporal Conv1d(k=9,s=2)
  -> pointwise Conv1d(128->256)
  -> temporal mean + std pooling [512]
  -> projection [512]
  -> semantic adapter -> 72 logits
  -> soft prototype mixture
  -> frozen Oracle semantic_to_cross
  -> frozen Oracle Zero123++/LoRA

Loss
----
L = lambda_distill * MSE(v_student, stopgrad(v_oracle))
  + lambda_cross   * (1 - cos(cross_student, cross_oracle))
  + lambda_soft(t) * KL(q_text_geometry || p_EEG)

lambda_soft(t) linearly decays from lambda_soft_sem to zero during the first
soft_sem_decay_steps optimizer steps. No hard CE is used.

Validation
----------
1) individual EEG generation effect
2) raw-trial-averaged semantic diagnostic
3) condition-level trial ensemble generation effect:
      mean logits across trials -> softmax -> prototype mixture
   This matches the intended final 08/09 protocol, where multiple EEG trials
   are available for each object/category.

The generator, LoRA, semantic_to_cross, VAE, and text encoder stay frozen.
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
# Reproducibility
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Strict split helpers
# =============================================================================
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
                    idx = s * (C * O * R) + c * (O * R) + o * R + r
                    out.append(idx)
    return out


class EEGOnlyDataset(Dataset):
    """EEG-only validation view. mode='averaged' averages RAW trials."""

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


class TrialEnsembleGenerationDataset(Dataset):
    """
    One item per (subject, category, object), with all EEG trials stacked.
    The first trial's ordinary dataset item supplies the same rendered target.
    """

    def __init__(self, base, suffixes):
        self.base = base
        self.cols = selected_object_columns(base, suffixes)
        self.S = int(base.eeg_data.shape[0])
        self.C = int(base.cls_num)
        self.O = int(base.obj_num)
        self.R = int(base.trails_num)

        self.groups = []
        for s in range(self.S):
            for c in range(self.C):
                for o in self.cols:
                    self.groups.append((s, c, o))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        s, c, o = self.groups[idx]
        flat0 = s * (self.C * self.O * self.R) + c * (self.O * self.R) + o * self.R
        item = dict(self.base[flat0])

        trials = np.asarray(
            self.base.eeg_data[s, c, o, :], dtype=np.float32
        )
        item["eeg_trials"] = torch.from_numpy(trials)
        item["num_trials"] = int(self.R)
        item["ensemble_trial_index"] = -1
        return item


# =============================================================================
# Fixed train-only category text geometry
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
            dataset_name = str(base_dataset.name_list[c, o])
            key = dataset_name[3:]
            cat = category_from_name(dataset_name)
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

        prototypes.append(F.normalize(torch.stack(feats).mean(dim=0), dim=0))
        categories.append(cats[0])
        source_objects.append(objs)

    prototypes = torch.stack(prototypes, dim=0)
    if tuple(prototypes.shape) != (72, expected_dim):
        raise RuntimeError(
            f"Expected [72,{expected_dim}], got {tuple(prototypes.shape)}"
        )
    return prototypes, categories, source_objects


def save_prototypes(out_dir, prototypes, categories, source_objects, suffixes):
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": list(suffixes),
            "construction": (
                "normalize each TRAIN-object text feature -> category mean -> normalize"
            ),
        },
        Path(out_dir) / "fixed_category_text_prototypes.pt",
    )


# =============================================================================
# Temporal EEG student
# =============================================================================
def normalize_eeg_channels(eeg: torch.Tensor) -> torch.Tensor:
    """eeg [B,64,600] -> deterministic per-sample/per-channel z-normalization."""
    eeg = eeg.float()
    if eeg.ndim != 3 or eeg.shape[1:] != (64, 600):
        raise RuntimeError(f"Expected EEG [B,64,600], got {tuple(eeg.shape)}")
    mean = eeg.mean(dim=-1, keepdim=True)
    std = eeg.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (eeg - mean) / std


class TemporalEEGSemanticStudent(nn.Module):
    """
    Low-capacity trainable temporal encoder.

    [B,64,600]
      -> DW temporal 64ch, k15 s2
      -> PW 64->128
      -> DW temporal 128ch, k9 s2
      -> PW 128->256
      -> temporal mean/std -> 512
      -> feature projection -> 512
      -> semantic adapter -> 72 logits
    """

    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=72):
        super().__init__()
        if int(feature_dim) != 512:
            raise ValueError("This architecture currently expects feature_dim=512")

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


# =============================================================================
# Oracle generator loading
# =============================================================================
def build_generator_model(args, out_dir):
    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)

    # Constructor compatibility only; legacy EEG prior modules are not used.
    OmegaConf.update(cfg, "semantic_prior_pca_dim", 512, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_rank", 64, merge=False)
    OmegaConf.update(cfg, "semantic_prior_residual_scale", 0.0, merge=False)
    OmegaConf.update(cfg, "semantic_prior_spatial_scale", 0.0, merge=False)
    OmegaConf.update(
        cfg, "semantic_prior_temperature", float(args.student_temperature), merge=False
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


def load_oracle_checkpoint(model, oracle_ckpt):
    path = Path(oracle_ckpt).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Oracle checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise RuntimeError("Oracle checkpoint must contain checkpoint['model'].")

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[oracle load warning] missing={missing}")
    if unexpected:
        print(f"[oracle load warning] unexpected={unexpected}")
    print(f"[oracle] loaded {path}; step={ckpt.get('step', 'unknown')}")
    return ckpt


def freeze_generator(model):
    model.requires_grad_(False)
    model.eval()
    if hasattr(model, "fmri_encoder"):
        model.fmri_encoder.eval()
    model.semantic_to_cross.eval()
    if hasattr(model, "unet"):
        model.unet.eval()
    model.pipeline.unet.eval()
    model.pipeline.vae.eval()
    model.pipeline.text_encoder.eval()

    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Generator freeze failed: trainable parameter remains.")
    print("[generator] fully frozen")


# =============================================================================
# Conditioning and diffusion helpers
# =============================================================================
def build_oracle_condition(model, labels, unconditional=False):
    B = labels.shape[0]
    unet_dtype = next(model.pipeline.unet.parameters()).dtype
    empty_prompt = model.get_empty_text_embeds(B)

    if unconditional:
        spatial = torch.zeros(
            B, 4, 64, 64, device=labels.device, dtype=unet_dtype
        )
        return empty_prompt, spatial

    with torch.autocast("cuda", enabled=False):
        prior = F.normalize(model.fixed_text_prototypes[labels].float(), dim=-1)
        semantic_cond = model.semantic_to_cross(prior).unsqueeze(1)

    semantic_cond = semantic_cond.to(unet_dtype)
    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(1, -1, 1)
    prompt = empty_prompt + semantic_cond * ramp
    spatial = torch.zeros(
        B, 4, 64, 64, device=labels.device, dtype=unet_dtype
    )
    return prompt, spatial


def semantic_cross_embedding(model, semantic_prior):
    """Frozen mapping, but keep input gradient for the EEG student."""
    with torch.autocast("cuda", enabled=False):
        return model.semantic_to_cross(semantic_prior.float())


def build_student_condition(model, semantic_prior):
    B = semantic_prior.shape[0]
    unet_dtype = next(model.pipeline.unet.parameters()).dtype
    empty_prompt = model.get_empty_text_embeds(B)

    semantic_cond = semantic_cross_embedding(model, semantic_prior).unsqueeze(1)
    semantic_cond = semantic_cond.to(unet_dtype)
    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(1, -1, 1)
    prompt = empty_prompt + semantic_cond * ramp
    spatial = torch.zeros(
        B, 4, 64, 64, device=semantic_prior.device, dtype=unet_dtype
    )
    return prompt, spatial


def prepare_noisy_latents(model, batch, device):
    _, target_imgs = model.prepare_batch_data(batch)
    labels = batch["cls_index"].to(device, dtype=torch.long, non_blocking=True)
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
# Semantic geometry / losses
# =============================================================================
def build_soft_semantic_table(prototypes, temperature):
    if temperature <= 0:
        raise ValueError("teacher semantic temperature must be > 0")
    p = F.normalize(prototypes.float(), dim=-1)
    sim = p @ p.t()
    return F.softmax(sim / float(temperature), dim=-1)


def soft_semantic_kl(logits, labels, q_table, student_temperature):
    target_q = q_table[labels]
    log_p = F.log_softmax(logits / float(student_temperature), dim=-1)
    return F.kl_div(log_p, target_q, reduction="batchmean")


def semantic_prior_from_probs(probs, fixed_prototypes):
    return F.normalize(probs @ fixed_prototypes.float(), dim=-1)


def cross_condition_loss(student_cross, oracle_cross):
    return (1.0 - F.cosine_similarity(
        student_cross.float(), oracle_cross.float(), dim=-1
    )).mean()


def soft_weight_at_step(step, max_weight, decay_steps):
    if max_weight <= 0 or decay_steps <= 0:
        return 0.0
    frac = max(0.0, 1.0 - float(step) / float(decay_steps))
    return float(max_weight) * frac


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def evaluate_semantic(student, dataset, q_table, args, device):
    loader = DataLoader(
        dataset,
        batch_size=max(1, args.semantic_eval_batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = student.training
    student.eval()

    total = 0
    top1_sum = 0
    kl_sum = 0.0
    entropy_sum = 0.0

    for batch in loader:
        eeg = batch["eeg_data"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        labels = batch["cls_index"].to(
            device, dtype=torch.long, non_blocking=True
        )
        logits, probs, _ = student(eeg, temperature=args.student_temperature)
        kl = soft_semantic_kl(
            logits, labels, q_table, args.student_temperature
        )
        n = labels.shape[0]
        total += n
        top1_sum += int((logits.argmax(dim=-1) == labels).sum().item())
        kl_sum += float(kl) * n
        ent = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
        entropy_sum += float(ent.sum())

    if was_training:
        student.train()
    total = max(1, total)
    return {
        "top1": top1_sum / total,
        "soft_kl": kl_sum / total,
        "entropy": entropy_sum / total,
    }


@torch.no_grad()
def _generation_metrics_from_values(distill_values, student_gt, oracle_gt, uncond_gt):
    if not distill_values:
        raise RuntimeError("No validation diffusion batches evaluated.")
    student_diff = float(np.mean(student_gt))
    oracle_diff = float(np.mean(oracle_gt))
    uncond_diff = float(np.mean(uncond_gt))
    student_gain = uncond_diff - student_diff
    oracle_gain = uncond_diff - oracle_diff
    recovery = student_gain / oracle_gain if oracle_gain > 1e-8 else float("nan")
    return {
        "distill": float(np.mean(distill_values)),
        "student_diff": student_diff,
        "oracle_diff": oracle_diff,
        "uncond_diff": uncond_diff,
        "student_gain": student_gain,
        "oracle_gain": oracle_gain,
        "oracle_gain_recovery": recovery,
        "n_batches": len(distill_values),
    }


@torch.no_grad()
def evaluate_generation_effect(model, student, val_subset, args, device):
    loader = DataLoader(
        val_subset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = student.training
    student.eval()

    distill_values, student_gt, oracle_gt, uncond_gt = [], [], [], []

    with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
        torch.manual_seed(args.seed + 8119)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 8119)

        for i, batch in enumerate(loader):
            if i >= args.val_diff_batches:
                break
            eeg = batch["eeg_data"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            labels, t, noisy, v_target = prepare_noisy_latents(model, batch, device)
            logits, probs, _ = student(eeg, temperature=args.student_temperature)
            prior = semantic_prior_from_probs(probs, model.fixed_text_prototypes)

            s_prompt, s_spatial = build_student_condition(model, prior)
            o_prompt, o_spatial = build_oracle_condition(model, labels, False)
            u_prompt, u_spatial = build_oracle_condition(model, labels, True)

            v_student = forward_unet(model, noisy, t, s_prompt, s_spatial)
            v_oracle = forward_unet(model, noisy, t, o_prompt, o_spatial)
            v_uncond = forward_unet(model, noisy, t, u_prompt, u_spatial)

            distill_values.append(float(F.mse_loss(v_student.float(), v_oracle.float())))
            student_gt.append(float(gt_diffusion_loss(model, v_student, v_target)))
            oracle_gt.append(float(gt_diffusion_loss(model, v_oracle, v_target)))
            uncond_gt.append(float(gt_diffusion_loss(model, v_uncond, v_target)))

    if was_training:
        student.train()
    return _generation_metrics_from_values(
        distill_values, student_gt, oracle_gt, uncond_gt
    )


@torch.no_grad()
def evaluate_trial_ensemble_generation(model, student, ensemble_dataset, args, device):
    """
    Condition-level ensemble:
      each trial -> student logits -> MEAN LOGITS -> softmax -> prototype mixture.
    This is intentionally different from averaging raw EEG first.
    """
    loader = DataLoader(
        ensemble_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    was_training = student.training
    student.eval()

    distill_values, student_gt, oracle_gt, uncond_gt = [], [], [], []

    with torch.random.fork_rng(devices=[0] if device.type == "cuda" else []):
        torch.manual_seed(args.seed + 9127)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 9127)

        for i, batch in enumerate(loader):
            if i >= args.val_ensemble_batches:
                break

            trials = batch["eeg_trials"].to(
                device, dtype=torch.float32, non_blocking=True
            )  # [B,R,64,600]
            B, R, E, T = trials.shape
            flat = trials.reshape(B * R, E, T)
            logits_flat, _, _ = student(
                flat, temperature=args.student_temperature
            )
            mean_logits = logits_flat.reshape(B, R, -1).mean(dim=1)
            probs = F.softmax(
                mean_logits / float(args.student_temperature), dim=-1
            )
            prior = semantic_prior_from_probs(
                probs, model.fixed_text_prototypes
            )

            labels, t, noisy, v_target = prepare_noisy_latents(model, batch, device)
            s_prompt, s_spatial = build_student_condition(model, prior)
            o_prompt, o_spatial = build_oracle_condition(model, labels, False)
            u_prompt, u_spatial = build_oracle_condition(model, labels, True)

            v_student = forward_unet(model, noisy, t, s_prompt, s_spatial)
            v_oracle = forward_unet(model, noisy, t, o_prompt, o_spatial)
            v_uncond = forward_unet(model, noisy, t, u_prompt, u_spatial)

            distill_values.append(float(F.mse_loss(v_student.float(), v_oracle.float())))
            student_gt.append(float(gt_diffusion_loss(model, v_student, v_target)))
            oracle_gt.append(float(gt_diffusion_loss(model, v_oracle, v_target)))
            uncond_gt.append(float(gt_diffusion_loss(model, v_uncond, v_target)))

    if was_training:
        student.train()
    return _generation_metrics_from_values(
        distill_values, student_gt, oracle_gt, uncond_gt
    )


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


def save_student_checkpoint(path, student, optimizer, scheduler, step, args, extra):
    torch.save(
        {
            "student": student.state_dict(),
            "student_arch": "temporal_dwconv_v1",
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": int(step),
            "args": vars(args),
            "oracle_ckpt": str(Path(args.oracle_ckpt).expanduser().resolve()),
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
            "E1-T: raw EEG -> low-capacity temporal encoder -> semantic prior -> "
            "frozen Oracle generator distillation"
        )
    )
    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument(
        "--rendered_view_path", default="/data/jionkim/neuro_3D/render_grid_v4"
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--oracle_ckpt", required=True)

    p.add_argument(
        "--holdout_suffix",
        default="06",
        choices=[f"{i:02d}" for i in range(7)],
    )

    p.add_argument("--feature_dim", type=int, default=512)
    p.add_argument("--student_hidden_dim", type=int, default=256)
    p.add_argument("--semantic_eval_batch_size", type=int, default=64)

    p.add_argument("--teacher_semantic_temperature", type=float, default=0.10)
    p.add_argument("--student_temperature", type=float, default=0.50)

    p.add_argument("--lambda_distill", type=float, default=1.0)
    p.add_argument("--lambda_cross", type=float, default=0.05)
    p.add_argument("--lambda_soft_sem", type=float, default=0.005)
    p.add_argument("--soft_sem_decay_steps", type=int, default=1000)

    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--student_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=250)
    p.add_argument("--val_diff_batches", type=int, default=32)
    p.add_argument("--val_ensemble_batches", type=int, default=32)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--selection_metric",
        choices=["individual_gain", "ensemble_gain"],
        default="ensemble_gain",
        help=(
            "Best-checkpoint primary metric. ensemble_gain is recommended because "
            "final 08/09 have multiple EEG trials."
        ),
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    seed_everything(args.seed)

    if args.student_temperature <= 0:
        raise ValueError("--student_temperature must be > 0")
    if args.teacher_semantic_temperature <= 0:
        raise ValueError("--teacher_semantic_temperature must be > 0")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be > 0")
    if args.feature_dim != 512:
        raise ValueError("--feature_dim must currently be 512")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[device] {device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    universe = tuple(f"{i:02d}" for i in range(7))
    train_suffixes = tuple(s for s in universe if s != args.holdout_suffix)
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
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Only train=True is instantiated. 08/09 are never touched.
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

    train_full_subset = Subset(base, full_dataset_indices(base, train_suffixes))
    val_full_subset = Subset(base, full_dataset_indices(base, val_suffixes))
    val_eeg_ind = EEGOnlyDataset(base, val_suffixes, mode="individual")
    val_eeg_rawavg = EEGOnlyDataset(base, val_suffixes, mode="averaged")
    val_ensemble = TrialEnsembleGenerationDataset(base, val_suffixes)

    print(
        f"[samples] generation_train={len(train_full_subset)} "
        f"generation_val_ind={len(val_full_subset)} "
        f"val_sem_ind={len(val_eeg_ind)} "
        f"val_sem_rawavg={len(val_eeg_rawavg)} "
        f"val_condition_ensemble={len(val_ensemble)}"
    )

    prototypes, categories, source_objects = build_fixed_category_text_prototypes(
        base, train_suffixes
    )
    save_prototypes(
        out_dir, prototypes, categories, source_objects, train_suffixes
    )

    model = build_generator_model(args, out_dir).to(device)
    oracle_pack = load_oracle_checkpoint(model, args.oracle_ckpt)
    model.set_fixed_prototypes(prototypes.to(device))
    freeze_generator(model)

    student = TemporalEEGSemanticStudent(
        feature_dim=args.feature_dim,
        hidden_dim=args.student_hidden_dim,
        num_classes=72,
    ).to(device)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] arch=temporal_dwconv_v1 trainable={trainable/1e6:.4f}M")

    # Lightweight shape sanity check before expensive training.
    with torch.no_grad():
        dummy = torch.zeros(2, 64, 600, device=device)
        dl, dp, df = student(dummy, temperature=args.student_temperature)
        if dl.shape != (2, 72) or df.shape != (2, 512):
            raise RuntimeError(
                f"Temporal student shape check failed: logits={dl.shape}, feat={df.shape}"
            )
    print("[student] shape check passed: EEG[64,600] -> feat512 -> logits72")

    q_table = build_soft_semantic_table(
        prototypes.to(device), args.teacher_semantic_temperature
    ).detach()

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.student_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args.max_steps, args.warmup_steps)

    train_loader = DataLoader(
        train_full_subset,
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

        # Frozen Oracle teacher, same noisy latent/timestep.
        with torch.no_grad():
            oracle_prior = F.normalize(
                model.fixed_text_prototypes[labels].float(), dim=-1
            )
            oracle_cross = semantic_cross_embedding(model, oracle_prior).detach()
            oracle_prompt, oracle_spatial = build_oracle_condition(
                model, labels, unconditional=False
            )
            v_teacher = forward_unet(
                model, noisy, t, oracle_prompt, oracle_spatial
            ).detach()

        # Trainable temporal EEG student.
        logits, probs, _ = student(
            eeg, temperature=args.student_temperature
        )
        prior = semantic_prior_from_probs(
            probs, model.fixed_text_prototypes
        )
        student_cross = semantic_cross_embedding(model, prior)
        student_prompt, student_spatial = build_student_condition(model, prior)

        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
        ):
            v_student = forward_unet(
                model, noisy, t, student_prompt, student_spatial
            )

        loss_distill = F.mse_loss(v_student.float(), v_teacher.float())
        loss_cross = cross_condition_loss(student_cross, oracle_cross)
        loss_soft = soft_semantic_kl(
            logits, labels, q_table, args.student_temperature
        )
        soft_w = soft_weight_at_step(
            step, args.lambda_soft_sem, args.soft_sem_decay_steps
        )

        total = (
            args.lambda_distill * loss_distill
            + args.lambda_cross * loss_cross
            + soft_w * loss_soft
        )

        (total / args.accumulation_steps).backward()
        micro += 1
        if micro % args.accumulation_steps != 0:
            continue

        if args.grad_clip > 0:
            clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % args.print_every == 0:
            batch_top1 = float((logits.argmax(dim=-1) == labels).float().mean())
            entropy = float(
                -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log())
                .sum(dim=-1)
                .mean()
            )
            print(
                f"[TRAIN] step={step:06d} L={total.item():.6f} "
                f"Distill={loss_distill.item():.6f} "
                f"Cross={loss_cross.item():.5f} "
                f"SoftKL={loss_soft.item():.4f} SoftW={soft_w:.6f} "
                f"Top1={batch_top1:.3f} H={entropy:.3f}"
            )
            writer.add_scalar("train/total", total.item(), step)
            writer.add_scalar("train/distill", loss_distill.item(), step)
            writer.add_scalar("train/cross", loss_cross.item(), step)
            writer.add_scalar("train/soft_kl", loss_soft.item(), step)
            writer.add_scalar("train/soft_weight", soft_w, step)

        if step % args.validate_every == 0 or step == args.max_steps:
            sem_ind = evaluate_semantic(
                student, val_eeg_ind, q_table, args, device
            )
            sem_rawavg = evaluate_semantic(
                student, val_eeg_rawavg, q_table, args, device
            )
            gen_ind = evaluate_generation_effect(
                model, student, val_full_subset, args, device
            )
            gen_ens = evaluate_trial_ensemble_generation(
                model, student, val_ensemble, args, device
            )

            print(
                f"[VAL] step={step:06d} "
                f"IND Gain={gen_ind['student_gain']:+.5f} "
                f"Rec={gen_ind['oracle_gain_recovery']:+.3f} "
                f"Distill={gen_ind['distill']:.6f} | "
                f"ENS Gain={gen_ens['student_gain']:+.5f} "
                f"Rec={gen_ens['oracle_gain_recovery']:+.3f} "
                f"Distill={gen_ens['distill']:.6f} | "
                f"IndTop1={sem_ind['top1']:.4f} "
                f"RawAvgTop1={sem_rawavg['top1']:.4f} "
                f"RawAvgKL={sem_rawavg['soft_kl']:.4f}"
            )

            row = {
                "step": step,
                "val_ind_distill": gen_ind["distill"],
                "val_ind_student_diff": gen_ind["student_diff"],
                "val_ind_oracle_diff": gen_ind["oracle_diff"],
                "val_ind_uncond_diff": gen_ind["uncond_diff"],
                "val_ind_student_gain": gen_ind["student_gain"],
                "val_ind_oracle_gain": gen_ind["oracle_gain"],
                "val_ind_recovery": gen_ind["oracle_gain_recovery"],
                "val_ens_distill": gen_ens["distill"],
                "val_ens_student_diff": gen_ens["student_diff"],
                "val_ens_oracle_diff": gen_ens["oracle_diff"],
                "val_ens_uncond_diff": gen_ens["uncond_diff"],
                "val_ens_student_gain": gen_ens["student_gain"],
                "val_ens_oracle_gain": gen_ens["oracle_gain"],
                "val_ens_recovery": gen_ens["oracle_gain_recovery"],
                "val_sem_ind_top1": sem_ind["top1"],
                "val_sem_ind_soft_kl": sem_ind["soft_kl"],
                "val_sem_rawavg_top1": sem_rawavg["top1"],
                "val_sem_rawavg_soft_kl": sem_rawavg["soft_kl"],
            }
            append_csv(log_path, row)
            for k, v in row.items():
                if k == "step" or not np.isfinite(v):
                    continue
                writer.add_scalar(f"val/{k}", float(v), step)

            primary = gen_ens if args.selection_metric == "ensemble_gain" else gen_ind
            key = (
                float(primary["student_gain"]),
                -float(primary["distill"]),
                float(gen_ind["student_gain"]),
            )

            extra = {
                "student_arch": "temporal_dwconv_v1",
                "strict_train_suffixes": train_suffixes,
                "strict_holdout_suffix": args.holdout_suffix,
                "object07": "NOT_USED",
                "final08_09": "NOT_USED",
                "oracle_step": oracle_pack.get("step", None),
                "raw_eeg_preprocessing": "per-sample per-channel z-normalize",
                "trial_ensemble": "mean logits across trials -> softmax -> prototype mixture",
                "selection_metric": args.selection_metric,
                "val": row,
            }

            save_student_checkpoint(
                ckpt_dir / "last.pt",
                student,
                optimizer,
                scheduler,
                step,
                args,
                extra,
            )

            if best_key is None or key > best_key:
                best_key = key
                best_step = step
                save_student_checkpoint(
                    ckpt_dir / "best.pt",
                    student,
                    optimizer,
                    scheduler,
                    step,
                    args,
                    extra,
                )
                print(
                    f"[BEST] step={step} metric={args.selection_metric} "
                    f"PrimaryGain={primary['student_gain']:+.5f} "
                    f"IND={gen_ind['student_gain']:+.5f} "
                    f"ENS={gen_ens['student_gain']:+.5f}"
                )

        if step % args.save_every == 0:
            save_student_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt",
                student,
                optimizer,
                scheduler,
                step,
                args,
                {
                    "student_arch": "temporal_dwconv_v1",
                    "strict_train_suffixes": train_suffixes,
                    "strict_holdout_suffix": args.holdout_suffix,
                },
            )

    writer.close()
    summary = {
        "best_step": int(best_step),
        "student_arch": "temporal_dwconv_v1",
        "student_trainable_parameters": int(trainable),
        "train_suffixes": list(train_suffixes),
        "holdout_suffix": args.holdout_suffix,
        "object07": "NOT_USED",
        "final08_09": "NOT_USED",
        "oracle_checkpoint": str(Path(args.oracle_ckpt).expanduser().resolve()),
        "deprecated_eeg_checkpoint_used": False,
        "pca_used": False,
        "raw_eeg_preprocessing": "per-sample per-channel z-normalize",
        "loss": {
            "lambda_distill": args.lambda_distill,
            "lambda_cross": args.lambda_cross,
            "lambda_soft_sem_initial": args.lambda_soft_sem,
            "soft_sem_decay_steps": args.soft_sem_decay_steps,
        },
        "trial_ensemble": "mean logits across trials -> softmax -> prototype mixture",
        "selection_metric": args.selection_metric,
        "best_student_checkpoint": str(ckpt_dir / "best.pt"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] best_step={best_step}; best={ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
