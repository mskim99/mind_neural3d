#!/usr/bin/env python3
"""
Oracle Semantic Generator
=========================

Strict DEV
----------
train  : object 00..05
val    : object 06
unused : object07
final  : 08/09 NOT LOADED

Condition
---------
GT category
 -> fixed TRAIN-only category text prototype
 -> semantic_to_cross
 -> Zero123++ cross-attention

Trainable
---------
semantic_to_cross
Zero123++ LoRA

Disabled
--------
EEG semantic classifier
EEG residual
EEG spatial cond_lat

Validation
----------
Same latent / noise / timestep:

Oracle category condition -> diffusion loss
Unconditional             -> diffusion loss

We want:
    OracleDiff < UncondDiff
"""

import argparse
import csv
import json
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

from src.mvdiffusion_semantic_prior import MVDiffusion
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# =========================================================
# Basic utilities
# =========================================================

def seed_everything(seed):
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
        suffixes_here = {
            object_suffix(dataset.name_list[c, o])
            for c in range(int(dataset.cls_num))
        }

        if len(suffixes_here) != 1:
            raise RuntimeError(
                f"Object column {o} inconsistent: {suffixes_here}"
            )

        suffix = next(iter(suffixes_here))

        if suffix in wanted:
            cols.append(o)

    found = {
        object_suffix(dataset.name_list[0, o])
        for o in cols
    }

    missing = wanted - found

    if missing:
        raise RuntimeError(
            f"Missing object suffixes: {sorted(missing)}"
        )

    return cols


def full_dataset_indices(dataset, suffixes):
    cols = selected_object_columns(
        dataset,
        suffixes,
    )

    S = int(dataset.eeg_data.shape[0])
    C = int(dataset.cls_num)
    O = int(dataset.obj_num)
    R = int(dataset.trails_num)

    indices = []

    for s in range(S):
        for c in range(C):
            for o in cols:
                for r in range(R):
                    idx = (
                        s * C * O * R
                        + c * O * R
                        + o * R
                        + r
                    )
                    indices.append(idx)

    return indices


class EEGOnlyDataset(Dataset):

    def __init__(
        self,
        base,
        suffixes,
        mode="individual",
    ):
        self.base = base
        self.mode = mode

        self.cols = selected_object_columns(
            base,
            suffixes,
        )

        S = int(base.eeg_data.shape[0])
        C = int(base.cls_num)
        R = int(base.trails_num)

        self.items = []

        for s in range(S):
            for c in range(C):
                for o in self.cols:

                    if mode == "individual":

                        for r in range(R):
                            self.items.append(
                                (s, c, o, r)
                            )

                    elif mode == "averaged":

                        self.items.append(
                            (s, c, o, None)
                        )

                    else:
                        raise ValueError(mode)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        s, c, o, r = self.items[idx]

        if r is None:

            eeg = np.asarray(
                self.base.eeg_data[
                    s,
                    c,
                    o,
                    :,
                ],
                dtype=np.float32,
            ).mean(
                axis=0,
                dtype=np.float32,
            )

            trial_index = -1

        else:

            eeg = np.asarray(
                self.base.eeg_data[
                    s,
                    c,
                    o,
                    r,
                ],
                dtype=np.float32,
            )

            trial_index = int(r)

        return {
            "eeg_data": torch.from_numpy(eeg),
            "cls_index": int(c),
            "obj_index": int(o),
            "trial_index": trial_index,
            "name": str(
                self.base.name_list[c, o]
            ),
        }


# =========================================================
# Fixed TRAIN-only category text prototypes
# =========================================================

def to_flat_feature(x):

    if torch.is_tensor(x):
        return x.detach().float().cpu().reshape(-1)

    return torch.as_tensor(
        np.asarray(x),
        dtype=torch.float32,
    ).reshape(-1)


def build_fixed_category_text_prototypes(
    base,
    source_suffixes,
    expected_dim=1024,
):

    source_cols = selected_object_columns(
        base,
        source_suffixes,
    )

    prototypes = []
    categories = []
    source_objects = []

    for c in range(int(base.cls_num)):

        feats = []
        cats = []
        objs = []

        for o in source_cols:

            name = str(
                base.name_list[c, o]
            )

            key = name[3:]

            feat = to_flat_feature(
                base.clip_features[key]["text"]
            )

            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"{key}: expected {expected_dim}, "
                    f"got {feat.numel()}"
                )

            feats.append(
                F.normalize(feat, dim=0)
            )

            cats.append(
                category_from_name(name)
            )

            objs.append(key)

        if len(set(cats)) != 1:
            raise RuntimeError(
                f"class {c}: category mismatch {set(cats)}"
            )

        prototype = F.normalize(
            torch.stack(feats).mean(dim=0),
            dim=0,
        )

        prototypes.append(prototype)
        categories.append(cats[0])
        source_objects.append(objs)

    prototypes = torch.stack(
        prototypes,
        dim=0,
    )

    if tuple(prototypes.shape) != (72, 1024):
        raise RuntimeError(
            f"Expected [72,1024], got {tuple(prototypes.shape)}"
        )

    return (
        prototypes,
        categories,
        source_objects,
    )


# =========================================================
# Model
# =========================================================

def build_model(
    args,
    out_dir,
):

    config_path = Path(
        args.config
    ).expanduser().resolve()

    cfg = OmegaConf.load(
        config_path
    )

    # Only needed so MVDiffusion constructor has fixed shapes.
    OmegaConf.update(
        cfg,
        "semantic_prior_pca_dim",
        512,
        merge=False,
    )

    OmegaConf.update(
        cfg,
        "semantic_prior_residual_rank",
        64,
        merge=False,
    )

    OmegaConf.update(
        cfg,
        "semantic_prior_residual_scale",
        0.0,
        merge=False,
    )

    OmegaConf.update(
        cfg,
        "semantic_prior_spatial_scale",
        0.0,
        merge=False,
    )

    stable_cfg = OmegaConf.select(
        cfg,
        "model.params.stable_diffusion_config",
    )

    fmri_cfg = OmegaConf.select(
        cfg,
        "model.params.fmri_encoder_config",
        default=None,
    )

    OmegaConf.save(
        cfg,
        out_dir / "resolved_config.yaml",
    )

    model = MVDiffusion(
        cfg,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(out_dir),
        num_classes=72,
    )

    return model, cfg


def set_oracle_training(model):

    model.requires_grad_(False)

    model.semantic_to_cross.requires_grad_(True)

    for name, p in model.unet.named_parameters():

        p.requires_grad_(
            name
            in model._unet_lora_trainable_names
        )

    model.fmri_encoder.eval()
    model.pipeline.vae.eval()
    model.pipeline.text_encoder.eval()

    model.semantic_to_cross.train()
    model.unet.train()

    n = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"[oracle trainable] {n / 1e6:.3f}M"
    )


def build_oracle_condition(
    model,
    labels,
    unconditional=False,
):

    B = labels.shape[0]

    unet_dtype = next(
        model.pipeline.unet.parameters()
    ).dtype

    empty_prompt = model.get_empty_text_embeds(B)

    if unconditional:

        semantic_cond = torch.zeros(
            B,
            1,
            1024,
            device=labels.device,
            dtype=unet_dtype,
        )

        spatial = torch.zeros(
            B,
            4,
            64,
            64,
            device=labels.device,
            dtype=unet_dtype,
        )

        return empty_prompt, spatial

    with torch.autocast(
        "cuda",
        enabled=False,
    ):

        prior = F.normalize(
            model.fixed_text_prototypes[
                labels
            ].float(),
            dim=-1,
        )

        semantic_cond = (
            model.semantic_to_cross(
                prior
            )
            .unsqueeze(1)
        )

    semantic_cond = semantic_cond.to(
        unet_dtype
    )

    ramp = semantic_cond.new_tensor(
        model.pipeline.config.ramping_coefficients
    ).view(
        1,
        -1,
        1,
    )

    prompt = (
        empty_prompt
        + semantic_cond * ramp
    )

    spatial = torch.zeros(
        B,
        4,
        64,
        64,
        device=labels.device,
        dtype=unet_dtype,
    )

    return prompt, spatial


def prepare_noisy_latents(
    model,
    batch,
    device,
):

    _, target_imgs = model.prepare_batch_data(
        batch
    )

    labels = batch[
        "cls_index"
    ].to(
        device,
        dtype=torch.long,
        non_blocking=True,
    )

    B = labels.shape[0]

    t = torch.randint(
        0,
        model.num_timesteps,
        (B,),
        device=device,
    ).long()

    latents = model.encode_target_images(
        target_imgs
    )

    noise = torch.randn_like(
        latents
    )

    noisy = model.train_scheduler.add_noise(
        latents,
        noise,
        t,
    )

    v_target = model.get_v(
        latents,
        noise,
        t,
    )

    return (
        labels,
        t,
        noisy,
        v_target,
    )


def diffusion_loss(
    model,
    noisy,
    t,
    prompt,
    spatial,
    v_target,
):

    pred = model.forward_unet(
        noisy,
        t,
        prompt,
        spatial,
    )

    loss, _ = model.compute_loss(
        pred,
        v_target,
    )

    return loss


# =========================================================
# Validation
# =========================================================

@torch.no_grad()
def validate(
    model,
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
    )

    model.unet.eval()
    model.semantic_to_cross.eval()

    oracle_losses = []
    uncond_losses = []

    with torch.random.fork_rng(
        devices=[0]
        if device.type == "cuda"
        else []
    ):

        torch.manual_seed(
            args.seed + 7103
        )

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                args.seed + 7103
            )

        for i, batch in enumerate(loader):

            if i >= args.val_batches:
                break

            (
                labels,
                t,
                noisy,
                v_target,
            ) = prepare_noisy_latents(
                model,
                batch,
                device,
            )

            oracle_prompt, oracle_spatial = (
                build_oracle_condition(
                    model,
                    labels,
                    unconditional=False,
                )
            )

            uncond_prompt, uncond_spatial = (
                build_oracle_condition(
                    model,
                    labels,
                    unconditional=True,
                )
            )

            oracle_losses.append(
                float(
                    diffusion_loss(
                        model,
                        noisy,
                        t,
                        oracle_prompt,
                        oracle_spatial,
                        v_target,
                    )
                )
            )

            uncond_losses.append(
                float(
                    diffusion_loss(
                        model,
                        noisy,
                        t,
                        uncond_prompt,
                        uncond_spatial,
                        v_target,
                    )
                )
            )

    model.unet.train()
    model.semantic_to_cross.train()

    oracle = float(
        np.mean(oracle_losses)
    )

    uncond = float(
        np.mean(uncond_losses)
    )

    return {
        "oracle_diff": oracle,
        "uncond_diff": uncond,
        "condition_gain": uncond - oracle,
    }


# =========================================================
# Main
# =========================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--config",
        default="./configs/mind3d_pp.yaml",
    )

    p.add_argument(
        "--data_path",
        default="/data/jionkim/neuro_3D/",
    )

    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )

    p.add_argument(
        "--sub_id",
        default="sub01",
    )

    p.add_argument(
        "--out_dir",
        default=(
            "/data/jionkim/mind_3d_output/"
            "oracle_semantic_generator_h06"
        ),
    )

    p.add_argument(
        "--holdout_suffix",
        default="06",
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--batchsize",
        type=int,
        default=1,
    )

    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
    )

    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    p.add_argument(
        "--generation_head_lr",
        type=float,
        default=5e-5,
    )

    p.add_argument(
        "--lora_lr",
        type=float,
        default=None,
    )

    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--condition_drop_prob",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--validate_every",
        type=int,
        default=500,
    )

    p.add_argument(
        "--val_batches",
        type=int,
        default=64,
    )

    p.add_argument(
        "--print_every",
        type=int,
        default=20,
    )

    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main():

    args = parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    universe = tuple(
        f"{i:02d}"
        for i in range(7)
    )

    train_suffixes = tuple(
        x
        for x in universe
        if x != args.holdout_suffix
    )

    val_suffixes = (
        args.holdout_suffix,
    )

    print(
        f"[strict] train={train_suffixes}; "
        f"val={val_suffixes}; "
        "object07=NOT_USED; final08_09=NOT_USED"
    )

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    ckpt_dir = out_dir / "checkpoints"

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
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
        base,
        full_dataset_indices(
            base,
            train_suffixes,
        ),
    )

    val_subset = Subset(
        base,
        full_dataset_indices(
            base,
            val_suffixes,
        ),
    )

    (
        prototypes,
        categories,
        source_objects,
    ) = build_fixed_category_text_prototypes(
        base,
        train_suffixes,
    )

    torch.save(
        {
            "prototypes": prototypes,
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": train_suffixes,
        },
        out_dir
        / "fixed_category_text_prototypes.pt",
    )

    model, cfg = build_model(
        args,
        out_dir,
    )

    model = model.to(device)

    model.set_fixed_prototypes(
        prototypes.to(device)
    )

    set_oracle_training(model)

    lora_lr = (
        args.lora_lr
        if args.lora_lr is not None
        else float(
            OmegaConf.select(
                cfg,
                "learning_rate",
                default=1e-4,
            )
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for p in model.semantic_to_cross.parameters()
                    if p.requires_grad
                ],
                "lr": args.generation_head_lr,
            },
            {
                "params": [
                    p
                    for p in model.unet.parameters()
                    if p.requires_grad
                ],
                "lr": lora_lr,
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    loader = DataLoader(
        train_subset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=True,
    )

    writer = SummaryWriter(
        str(out_dir / "logs")
    )

    step = 0
    micro = 0
    best = None
    best_step = 0

    optimizer.zero_grad(
        set_to_none=True
    )

    while step < args.max_steps:

        for batch in loader:

            (
                labels,
                t,
                noisy,
                v_target,
            ) = prepare_noisy_latents(
                model,
                batch,
                device,
            )

            drop = (
                random.random()
                < args.condition_drop_prob
            )

            prompt, spatial = (
                build_oracle_condition(
                    model,
                    labels,
                    unconditional=drop,
                )
            )

            loss = diffusion_loss(
                model,
                noisy,
                t,
                prompt,
                spatial,
                v_target,
            )

            (
                loss
                / args.accumulation_steps
            ).backward()

            micro += 1

            if (
                micro
                % args.accumulation_steps
                != 0
            ):
                continue

            clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                args.grad_clip,
            )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            step += 1

            if (
                step == 1
                or step
                % args.print_every
                == 0
            ):
                print(
                    f"[TRAIN] step={step:06d} "
                    f"Diff={loss.item():.5f} "
                    f"Drop={int(drop)}"
                )

            if (
                step
                % args.validate_every
                == 0
            ):

                val = validate(
                    model,
                    val_subset,
                    args,
                    device,
                )

                print(
                    f"[VAL] step={step:06d} "
                    f"OracleDiff={val['oracle_diff']:.5f} "
                    f"UncondDiff={val['uncond_diff']:.5f} "
                    f"Gain={val['condition_gain']:+.5f}"
                )

                key = (
                    -val["oracle_diff"],
                    val["condition_gain"],
                )

                payload = {
                    "model": model.state_dict(),
                    "step": step,
                    "args": vars(args),
                    "train_suffixes": train_suffixes,
                    "holdout_suffix": args.holdout_suffix,
                    "val": val,
                }

                torch.save(
                    payload,
                    ckpt_dir / "last.pt",
                )

                if (
                    best is None
                    or key > best
                ):

                    best = key
                    best_step = step

                    torch.save(
                        payload,
                        ckpt_dir / "best.pt",
                    )

                    print(
                        f"[BEST] step={step}"
                    )

            if step >= args.max_steps:
                break

    writer.close()

    print(
        f"[done] best_step={best_step}"
    )


if __name__ == "__main__":
    main()