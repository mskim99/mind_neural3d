#!/usr/bin/env python3
"""
Final trainer for the recommended EEG semantic structure.

DEV protocol (default)
----------------------
train objects: 00..06
validation object: 07
final test objects 08..09: NEVER instantiated during training/tuning

Stage-1:
    EEG -> semantic -> semantic_to_clip -> frozen train-only category prototypes
    loss = ProtoCE + optional tiny Ortho
    best checkpoint selected by object-07 fixed-prototype margin

Stage-2:
    restore BEST Stage-1 checkpoint
    freeze shared EEG trunk (including BN running statistics)
    ProtoCE updates semantic_routing + semantic_to_clip
    diffusion reads DETACHED z_clip
    variation_routing + variation_to_latent + LoRA are trained by diffusion
    loss = Diffusion + ProtoCE + optional tiny Ortho

FINAL protocol
--------------
train objects: 00..07
prototype source: 00..07
no validation/test is instantiated.
Use the Stage-1 / Stage-2 step counts chosen in DEV, then evaluate 08..09 once
with a separate inference/evaluation script.

Removed objectives:
- learned 72-way classifier CE
- SupCon / memory bank
- AvgCons
- direct CLIP image/text cosine regression
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.mvdiffusion_var_protofinal import MVDiffusion
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# =========================================================================
# CLI
# =========================================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--config", default="./configs/mind3d_pp.yaml")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D/")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", required=True)

    p.add_argument(
        "--protocol",
        choices=["dev", "final"],
        default="dev",
        help=(
            "dev: train 00..06 / validate 07 / never load 08..09. "
            "final: train 00..07 with no validation/test."
        ),
    )

    # Optimizer-step counts, not micro-batches.
    p.add_argument("--semantic_steps", type=int, default=30000)
    p.add_argument("--joint_steps", type=int, default=30000)
    p.add_argument("--stage1_only", action="store_true")
    p.add_argument(
        "--stage2_from",
        default="",
        help=(
            "Skip Stage-1 and start Stage-2 from a selected semantic checkpoint "
            "produced by this final trainer."
        ),
    )

    p.add_argument("--batchsize", type=int, default=1)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    # Semantic geometry.
    p.add_argument("--prototype_temperature", type=float, default=0.07)
    p.add_argument("--semantic_lr", type=float, default=1e-4)
    p.add_argument("--semantic_head_lr", type=float, default=1e-4)
    p.add_argument("--joint_semantic_lr", type=float, default=3e-5)
    p.add_argument("--generation_head_lr", type=float, default=1e-4)
    p.add_argument(
        "--lora_lr",
        type=float,
        default=None,
        help="Defaults to `learning_rate` from mind3d_pp.yaml.",
    )

    # Only retained auxiliary objective. Default OFF.
    p.add_argument("--lambda_ortho", type=float, default=0.0)

    # Stage-2 raw loss weights. Branches are gradient-isolated, so these are
    # not intended to compensate for competing gradients in one shared branch.
    p.add_argument("--lambda_diff", type=float, default=1.0)
    p.add_argument("--lambda_proto_joint", type=float, default=1.0)

    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--validate_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)

    p.add_argument(
        "--aug_data",
        action="store_true",
        help="Applied only to Stage-2 full-image dataset items.",
    )

    return p.parse_args()


# =========================================================================
# Reproducibility
# =========================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================================
# Config
# =========================================================================
def resolve_model_configs(cfg, config_path):
    stable_cfg = OmegaConf.select(
        cfg, "model.params.stable_diffusion_config", default=None
    )
    fmri_cfg = OmegaConf.select(
        cfg, "model.params.fmri_encoder_config", default=None
    )
    if stable_cfg is None:
        raise KeyError(
            f"{config_path} is not the MinD-3D++ config. Missing "
            "`model.params.stable_diffusion_config`."
        )

    if OmegaConf.select(cfg, "learning_rate", default=None) is None:
        raise KeyError(
            "`learning_rate` missing from config. It is used as default LoRA LR."
        )
    return stable_cfg, fmri_cfg, cfg


# =========================================================================
# Dataset split helpers
# =========================================================================
def object_suffix(name):
    key = str(name)[3:]
    suffix = key.rsplit("_", 1)[-1] if "_" in key else ""
    return suffix


def category_from_name(name):
    key = str(name)[3:]
    if "_" in key and key.rsplit("_", 1)[-1].isdigit():
        return key.rsplit("_", 1)[0]
    return key


def protocol_suffixes(protocol):
    if protocol == "dev":
        return tuple(f"{i:02d}" for i in range(7)), ("07",)
    if protocol == "final":
        return tuple(f"{i:02d}" for i in range(8)), tuple()
    raise ValueError(protocol)


def selected_object_columns(dataset, suffixes):
    suffixes = set(suffixes)
    cols = []
    for o in range(int(dataset.obj_num)):
        col_suffixes = {
            object_suffix(dataset.name_list[c, o])
            for c in range(int(dataset.cls_num))
        }
        if len(col_suffixes) != 1:
            raise RuntimeError(
                f"Object column {o} has inconsistent suffixes: {col_suffixes}"
            )
        suffix = next(iter(col_suffixes))
        if suffix in suffixes:
            cols.append(o)

    found = {
        object_suffix(dataset.name_list[0, o])
        for o in cols
    }
    missing = suffixes - found
    if missing:
        raise RuntimeError(
            f"Requested object suffixes not found in train dataset: {sorted(missing)}"
        )
    return cols


def full_dataset_indices(dataset, suffixes):
    cols = selected_object_columns(dataset, suffixes)
    S = int(dataset.eeg_data.shape[0])
    C = int(dataset.cls_num)
    O = int(dataset.obj_num)
    R = int(dataset.trails_num)

    indices = []
    for s in range(S):
        for c in range(C):
            for o in cols:
                for r in range(R):
                    idx = s * (C * O * R) + c * (O * R) + o * R + r
                    indices.append(idx)
    return indices


class SemanticObjectDataset(Dataset):
    """
    EEG/label-only view over the train=True raw tensor.

    This avoids loading six rendered images during Stage-1 and semantic
    validation. mode='averaged' averages trials BEFORE the encoder.
    """
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
        raw = self.base.eeg_data

        if r is None:
            eeg = np.asarray(
                raw[s, c, o, :], dtype=np.float32
            ).mean(axis=0, dtype=np.float32)
            trial_index = -1
        else:
            eeg = np.asarray(raw[s, c, o, r], dtype=np.float32)
            trial_index = int(r)

        name = str(self.base.name_list[c, o])
        return {
            "name": name,
            "label": name[3:],
            "cls_index": int(c),
            "obj_index": int(o),
            "trial_index": trial_index,
            "subject_index": int(s),
            "eeg_data": torch.from_numpy(eeg),
        }


# =========================================================================
# Fixed category prototypes
# =========================================================================
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
    prototypes = []
    categories = []
    source_objects = []

    for c in range(int(base_dataset.cls_num)):
        feats, cats, objs = [], [], []
        for o in source_cols:
            dataset_name = str(base_dataset.name_list[c, o])
            key = dataset_name[3:]
            cat = category_from_name(dataset_name)

            feat = to_flat_feature(
                base_dataset.clip_features[key]["text"]
            )
            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"{key}: expected text dim {expected_dim}, "
                    f"got {feat.numel()}"
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


def save_prototype_artifacts(
    out_dir, prototypes, categories, source_objects, source_suffixes
):
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": list(source_suffixes),
            "construction": (
                "normalize each allowed train-object text feature -> "
                "class mean -> normalize"
            ),
        },
        out_dir / "fixed_category_text_prototypes.pt",
    )

    with (out_dir / "fixed_category_text_prototypes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["cls_index", "category", "source_objects"])
        for c, (cat, objs) in enumerate(zip(categories, source_objects)):
            w.writerow([c, cat, ";".join(objs)])


# =========================================================================
# Optimizer / scheduler
# =========================================================================
def trainable_params(module):
    return [p for p in module.parameters() if p.requires_grad]


def build_optimizer(model, stage, args, cfg):
    if stage == "semantic":
        groups = [
            {
                "params": trainable_params(model.fmri_encoder),
                "lr": args.semantic_lr,
                "name": "eeg_encoder",
            },
            {
                "params": trainable_params(model.semantic_to_clip),
                "lr": args.semantic_head_lr,
                "name": "semantic_to_clip",
            },
        ]
    else:
        lora_lr = (
            float(args.lora_lr)
            if args.lora_lr is not None
            else float(cfg.learning_rate)
        )
        groups = [
            {
                "params": [
                    p for p in model.unet.parameters() if p.requires_grad
                ],
                "lr": lora_lr,
                "name": "unet_lora",
            },
            {
                "params": trainable_params(
                    model.fmri_encoder.disentanglement.semantic_routing
                ),
                "lr": args.joint_semantic_lr,
                "name": "semantic_routing",
            },
            {
                "params": trainable_params(model.semantic_to_clip),
                "lr": args.joint_semantic_lr,
                "name": "semantic_to_clip",
            },
            {
                "params": trainable_params(
                    model.fmri_encoder.disentanglement.variation_routing
                ),
                "lr": args.generation_head_lr,
                "name": "variation_routing",
            },
            {
                "params": trainable_params(model.semantic_to_cross),
                "lr": args.generation_head_lr,
                "name": "semantic_to_cross",
            },
            {
                "params": trainable_params(model.variation_to_latent),
                "lr": args.generation_head_lr,
                "name": "variation_to_latent",
            },
        ]

    groups = [g for g in groups if len(g["params"]) > 0]
    if not groups:
        raise RuntimeError(f"No trainable parameters for stage={stage}")

    print(f"[optimizer:{stage}]")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        print(
            f"  {g['name']:<22s} {n/1e6:8.3f}M  lr={g['lr']:.3e}"
        )

    return torch.optim.AdamW(groups, betas=(0.9, 0.95))


def build_scheduler(optimizer, total_steps, warmup_steps):
    warmup_steps = min(int(warmup_steps), max(0, total_steps - 1))

    def fn(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        remain = max(1, total_steps - warmup_steps)
        return max(
            0.0,
            float(total_steps - step) / float(remain)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# =========================================================================
# Semantic evaluation
# =========================================================================
@torch.no_grad()
def evaluate_semantic(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
    temperature,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    was_training = model.training
    model.eval()

    totals = {
        "n": 0,
        "proto_loss": 0.0,
        "top1": 0.0,
        "margin": 0.0,
        "correct_cosine": 0.0,
        "ortho": 0.0,
    }

    for batch in loader:
        out = model.semantic_validation(
            batch, prototype_temperature=temperature
        )
        n = len(batch["cls_index"])
        totals["n"] += n
        totals["proto_loss"] += float(out["proto_loss"]) * n
        totals["top1"] += float(out["proto_top1"]) * n
        totals["margin"] += float(out["proto_margin"]) * n
        totals["correct_cosine"] += (
            float(out["proto_correct_cosine"]) * n
        )
        totals["ortho"] += float(out["ortho_loss"]) * n

    if was_training:
        model.train()

    n = max(1, totals["n"])
    return {
        "proto_loss": totals["proto_loss"] / n,
        "top1": totals["top1"] / n,
        "margin": totals["margin"] / n,
        "correct_cosine": totals["correct_cosine"] / n,
        "ortho": totals["ortho"] / n,
    }


def print_semantic_eval(prefix, metrics, step):
    print(
        f"[{prefix}] step={step:06d} "
        f"Proto={metrics['proto_loss']:.4f} "
        f"Top1={metrics['top1']:.4f} "
        f"Margin={metrics['margin']:+.5f} "
        f"Cos={metrics['correct_cosine']:.4f} "
        f"Ortho={metrics['ortho']:.4f}"
    )


# =========================================================================
# Checkpoints
# =========================================================================
def save_checkpoint(
    path,
    model,
    stage,
    step,
    args,
    optimizer=None,
    scheduler=None,
    extra=None,
):
    payload = {
        "model": model.state_dict(),
        "stage": stage,
        "step": int(step),
        "args": vars(args),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_model_only(path, model, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[LOAD MISMATCH]")
        print(" missing   :", incompatible.missing_keys[:30])
        print(" unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError("Checkpoint architecture mismatch.")
    return ckpt


# =========================================================================
# Training loops
# =========================================================================
def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def stage1_train(
    model,
    train_sem_dataset,
    val_ind_dataset,
    val_avg_dataset,
    args,
    cfg,
    device,
    out_dir,
    writer,
):
    model.set_training_stage("semantic")
    model.train()

    loader = DataLoader(
        train_sem_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    it = cycle_loader(loader)

    optimizer = build_optimizer(model, "semantic", args, cfg)
    scheduler = build_scheduler(
        optimizer, args.semantic_steps, args.warmup_steps
    )
    optimizer.zero_grad(set_to_none=True)

    best_margin = -float("inf")
    best_top1 = -float("inf")
    best_step = -1
    best_path = out_dir / "checkpoints" / "best_semantic.pt"

    micro = 0
    step = 0
    start = time.time()

    while step < args.semantic_steps:
        batch = next(it)
        out = model(
            batch,
            stage="semantic",
            prototype_temperature=args.prototype_temperature,
        )

        total = out["proto_loss"] + args.lambda_ortho * out["ortho_loss"]
        (total / args.accumulation_steps).backward()
        micro += 1

        if micro % args.accumulation_steps != 0:
            continue

        if args.grad_clip > 0:
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % args.print_every == 0:
            print(
                f"[S1] step={step:06d} "
                f"Total={total.item():.4f} "
                f"Proto={out['proto_loss'].item():.4f} "
                f"PAcc={out['proto_top1'].item():.3f} "
                f"PMargin={out['proto_margin'].item():+.5f} "
                f"Ortho={out['ortho_loss'].item():.4f} "
                f"time={(time.time()-start)/60:.1f}m"
            )
            writer.add_scalar("s1/train_total", total.item(), step)
            writer.add_scalar("s1/train_proto", out["proto_loss"].item(), step)
            writer.add_scalar("s1/train_top1", out["proto_top1"].item(), step)
            writer.add_scalar("s1/train_margin", out["proto_margin"].item(), step)

        if (
            args.protocol == "dev"
            and step % args.validate_every == 0
        ):
            vi = evaluate_semantic(
                model, val_ind_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            va = evaluate_semantic(
                model, val_avg_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            print_semantic_eval("VAL07 individual", vi, step)
            print_semantic_eval("VAL07 averaged  ", va, step)

            for k, v in vi.items():
                writer.add_scalar(f"s1/val07_ind/{k}", v, step)
            for k, v in va.items():
                writer.add_scalar(f"s1/val07_avg/{k}", v, step)

            improved = (
                vi["margin"] > best_margin + 1e-8
                or (
                    abs(vi["margin"] - best_margin) <= 1e-8
                    and vi["top1"] > best_top1
                )
            )
            if improved:
                best_margin = vi["margin"]
                best_top1 = vi["top1"]
                best_step = step
                save_checkpoint(
                    best_path,
                    model,
                    stage="semantic",
                    step=step,
                    args=args,
                    optimizer=None,
                    scheduler=None,
                    extra={
                        "val07_individual": vi,
                        "val07_averaged": va,
                        "selection_metric": "val07 individual margin",
                    },
                )
                print(
                    f"[BEST S1] step={step} "
                    f"val07 margin={best_margin:+.5f} "
                    f"top1={best_top1:.4f}"
                )

        if step % args.save_every == 0:
            save_checkpoint(
                out_dir / "checkpoints" / f"semantic_{step:06d}.pt",
                model,
                stage="semantic",
                step=step,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    last_path = out_dir / "checkpoints" / "semantic_last.pt"
    save_checkpoint(
        last_path,
        model,
        stage="semantic",
        step=step,
        args=args,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if args.protocol == "dev":
        if best_step < 0:
            # In case semantic_steps < validate_every.
            vi = evaluate_semantic(
                model, val_ind_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            va = evaluate_semantic(
                model, val_avg_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            best_step = step
            best_margin = vi["margin"]
            best_top1 = vi["top1"]
            save_checkpoint(
                best_path,
                model,
                stage="semantic",
                step=step,
                args=args,
                extra={
                    "val07_individual": vi,
                    "val07_averaged": va,
                    "selection_metric": "val07 individual margin",
                },
            )

        print(
            f"[S1 SELECTED] best_step={best_step}, "
            f"val07_margin={best_margin:+.5f}, "
            f"val07_top1={best_top1:.4f}"
        )
        return best_path, best_step

    return last_path, step


def stage2_train(
    model,
    full_train_subset,
    val_ind_dataset,
    val_avg_dataset,
    args,
    cfg,
    device,
    out_dir,
    writer,
):
    model.set_training_stage("joint")
    model.train()  # overridden model.train() keeps frozen EEG trunk in eval mode.

    loader = DataLoader(
        full_train_subset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
        drop_last=True,
    )
    it = cycle_loader(loader)

    optimizer = build_optimizer(model, "joint", args, cfg)
    scheduler = build_scheduler(
        optimizer, args.joint_steps, args.warmup_steps
    )
    optimizer.zero_grad(set_to_none=True)

    micro = 0
    step = 0
    start = time.time()

    while step < args.joint_steps:
        batch = next(it)
        out = model(
            batch,
            stage="joint",
            prototype_temperature=args.prototype_temperature,
        )

        total = (
            args.lambda_diff * out["diff_loss"]
            + args.lambda_proto_joint * out["proto_loss"]
            + args.lambda_ortho * out["ortho_loss"]
        )

        (total / args.accumulation_steps).backward()
        micro += 1

        if micro % args.accumulation_steps != 0:
            continue

        if args.grad_clip > 0:
            clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step == 1 or step % args.print_every == 0:
            print(
                f"[S2] step={step:06d} "
                f"Total={total.item():.4f} "
                f"Diff={out['diff_loss'].item():.4f} "
                f"Proto={out['proto_loss'].item():.4f} "
                f"PAcc={out['proto_top1'].item():.3f} "
                f"PMargin={out['proto_margin'].item():+.5f} "
                f"Ortho={out['ortho_loss'].item():.4f} "
                f"time={(time.time()-start)/60:.1f}m"
            )
            writer.add_scalar("s2/train_total", total.item(), step)
            writer.add_scalar("s2/train_diff", out["diff_loss"].item(), step)
            writer.add_scalar("s2/train_proto", out["proto_loss"].item(), step)
            writer.add_scalar("s2/train_top1", out["proto_top1"].item(), step)
            writer.add_scalar("s2/train_margin", out["proto_margin"].item(), step)

        if (
            args.protocol == "dev"
            and step % args.validate_every == 0
        ):
            vi = evaluate_semantic(
                model, val_ind_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            va = evaluate_semantic(
                model, val_avg_dataset, device,
                batch_size=max(1, args.batchsize * 16),
                num_workers=args.num_workers,
                temperature=args.prototype_temperature,
            )
            print_semantic_eval("S2 VAL07 individual", vi, step)
            print_semantic_eval("S2 VAL07 averaged  ", va, step)
            for k, v in vi.items():
                writer.add_scalar(f"s2/val07_ind/{k}", v, step)
            for k, v in va.items():
                writer.add_scalar(f"s2/val07_avg/{k}", v, step)

        if step % args.save_every == 0:
            save_checkpoint(
                out_dir / "checkpoints" / f"joint_{step:06d}.pt",
                model,
                stage="joint",
                step=step,
                args=args,
                optimizer=optimizer,
                scheduler=scheduler,
            )

    final_path = out_dir / "checkpoints" / "joint_final.pt"
    save_checkpoint(
        final_path,
        model,
        stage="joint",
        step=step,
        args=args,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    return final_path


# =========================================================================
# Main
# =========================================================================
def main():
    args = parse_args()

    if args.semantic_steps <= 0:
        raise ValueError("--semantic_steps must be > 0")
    if args.joint_steps < 0:
        raise ValueError("--joint_steps must be >= 0")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be > 0")
    if args.prototype_temperature <= 0:
        raise ValueError("--prototype_temperature must be > 0")
    if args.lambda_ortho < 0:
        raise ValueError("--lambda_ortho must be >= 0")

    seed_everything(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(
        f"[device] {device}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    config_path = Path(args.config).expanduser().resolve()
    cfg = OmegaConf.load(config_path)
    stable_cfg, fmri_cfg, model_args = resolve_model_configs(
        cfg, config_path
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_path, out_dir / "config.yaml")
    with (out_dir / "train_args.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    train_suffixes, val_suffixes = protocol_suffixes(args.protocol)
    print(
        f"[protocol={args.protocol}] train={train_suffixes}, "
        f"val={val_suffixes if val_suffixes else 'NONE'}, "
        "final_test=NOT_LOADED"
    )

    # Important: ONLY train=True dataset is instantiated here.
    # In DEV this contains 00..07; subsets below prevent 07 from training.
    # 08..09 are never loaded by this trainer.
    base_train = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[args.sub_id],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=args.aug_data,
        strict_rendered_views=True,
    )

    stage1_train_dataset = SemanticObjectDataset(
        base_train, train_suffixes, mode="individual"
    )

    full_indices = full_dataset_indices(
        base_train, train_suffixes
    )
    stage2_train_subset = Subset(base_train, full_indices)

    if args.protocol == "dev":
        val_ind_dataset = SemanticObjectDataset(
            base_train, val_suffixes, mode="individual"
        )
        val_avg_dataset = SemanticObjectDataset(
            base_train, val_suffixes, mode="averaged"
        )
    else:
        val_ind_dataset = None
        val_avg_dataset = None

    print(
        f"[samples] S1 train={len(stage1_train_dataset)}, "
        f"S2 train={len(stage2_train_subset)}, "
        f"val_ind={len(val_ind_dataset) if val_ind_dataset else 0}, "
        f"val_avg={len(val_avg_dataset) if val_avg_dataset else 0}"
    )

    prototypes, categories, source_objects = (
        build_fixed_category_text_prototypes(
            base_train, train_suffixes
        )
    )
    save_prototype_artifacts(
        out_dir,
        prototypes,
        categories,
        source_objects,
        train_suffixes,
    )
    print(
        f"[fixed prototypes] shape={tuple(prototypes.shape)}, "
        f"source_suffixes={train_suffixes}"
    )

    model = MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
    ).to(device)
    model.set_fixed_prototypes(prototypes.to(device))

    writer = SummaryWriter(str(out_dir / "logs"))

    if args.stage2_from:
        if args.stage1_only:
            raise ValueError("--stage1_only and --stage2_from cannot be used together.")
        selected_path = Path(args.stage2_from).expanduser().resolve()
        selected_step = -1
        print(f"[Stage-1 skipped] loading selected semantic checkpoint: {selected_path}")
        load_model_only(selected_path, model, device="cpu")
        model.set_fixed_prototypes(prototypes.to(device))
    else:
        selected_path, selected_step = stage1_train(
            model,
            stage1_train_dataset,
            val_ind_dataset,
            val_avg_dataset,
            args,
            cfg,
            device,
            out_dir,
            writer,
        )

        print(
            f"[Stage-1 complete] selected={selected_path}, "
            f"step={selected_step}"
        )

        if args.stage1_only or args.joint_steps == 0:
            print("[done] Stage-1 only.")
            writer.close()
            return

        # Critical: Stage-2 always starts from the selected Stage-1 model.
        load_model_only(selected_path, model, device="cpu")
        model.set_fixed_prototypes(prototypes.to(device))

    final_path = stage2_train(
        model,
        stage2_train_subset,
        val_ind_dataset,
        val_avg_dataset,
        args,
        cfg,
        device,
        out_dir,
        writer,
    )

    writer.close()
    print(f"[done] final checkpoint: {final_path}")
    if args.protocol == "dev":
        print(
            "[IMPORTANT] 08/09 were never instantiated. "
            "Choose architecture/steps using object-07 only."
        )
    else:
        print(
            "[IMPORTANT] final protocol used 00..07. "
            "Evaluate 08/09 exactly once with separate inference code."
        )


if __name__ == "__main__":
    main()
