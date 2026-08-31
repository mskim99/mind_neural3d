#!/usr/bin/env python3
"""
Semantic-Prior-Conditioned EEG -> Zero123++ Generation
======================================================

This trainer is the actual transition away from exact unseen-object CLIP
regression.

Core separation
---------------
1) EEG semantic estimator:
      frozen TemporalStem
        -> train-only PCA
        -> 72-way category distribution
   Optimized ONLY by category CE.

2) Semantic generation prior:
      soft category distribution
        -> mixture of fixed TRAIN-ONLY category text prototypes
        -> tiny gated EEG residual
        -> semantic_to_cross
        -> Zero123++ cross-attention

3) Generator:
      Zero123++ UNet LoRA + semantic_to_cross
   Optimized by diffusion v-prediction loss.

4) High-capacity old EEG variation path:
      DISABLED by default.
   `--spatial_scale 0.0` produces semantic-only generation conditioning.

Strict development split
------------------------
Default:
    train objects = 00..05
    holdout       = 06
    object07      = NOT USED
    final08/09    = NOT USED

The frozen TemporalStem is initialized from the selected strict semantic
checkpoint, but ONLY TemporalStem weights are loaded from it.

PCA is fitted only on individual EEG trials from train objects.

Teacher-anchor curriculum
-------------------------
During generation training the generator initially receives the ground-truth
category prototype and gradually transitions to the predicted soft semantic
prior:

    teacher_mix = 1.0 -> 0.0

The predicted distribution is detached from diffusion. Thus diffusion gradients
never teach the category classifier to memorize seen objects.

At inference:
    teacher_mix = 0.0 always.
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
from sklearn.decomposition import PCA

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.mvdiffusion_semantic_prior import MVDiffusion
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# =============================================================================
# CLI
# =============================================================================
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
        required=True,
    )

    p.add_argument(
        "--holdout_suffix",
        default="06",
        choices=[
            f"{i:02d}"
            for i in range(7)
        ],
    )

    p.add_argument(
        "--temporal_ckpt",
        default=(
            "/data/jionkim/mind_3d_output/holdout_06/"
            "checkpoints/best_semantic.pt"
        ),
        help=(
            "Used ONLY to initialize frozen fmri_encoder.temporal_stem."
        ),
    )

    # Fixed EEG representation.
    p.add_argument(
        "--pca_dim",
        type=int,
        default=512,
    )
    p.add_argument(
        "--feature_batch_size",
        type=int,
        default=64,
    )

    # Semantic-prior model.
    p.add_argument(
        "--prior_temperature",
        type=float,
        default=0.50,
    )
    p.add_argument(
        "--residual_rank",
        type=int,
        default=64,
    )
    p.add_argument(
        "--residual_scale",
        type=float,
        default=0.02,
        help=(
            "Maximum semantic EEG residual magnitude. Keep small."
        ),
    )
    p.add_argument(
        "--spatial_scale",
        type=float,
        default=0.0,
        help=(
            "0 disables EEG spatial cond_lat. Recommended first run."
        ),
    )
    p.add_argument(
        "--label_smoothing",
        type=float,
        default=0.05,
    )

    # Training.
    p.add_argument(
        "--max_steps",
        type=int,
        default=10000,
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
        "--classifier_lr",
        type=float,
        default=2e-4,
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
        help=(
            "Defaults to config learning_rate."
        ),
    )
    p.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--lambda_diff",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--lambda_cls",
        type=float,
        default=0.30,
    )
    p.add_argument(
        "--lambda_gate",
        type=float,
        default=0.05,
    )

    p.add_argument(
        "--teacher_mix_start",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--teacher_mix_end",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--teacher_mix_steps",
        type=int,
        default=5000,
    )

    p.add_argument(
        "--warmup_steps",
        type=int,
        default=300,
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--print_every",
        type=int,
        default=20,
    )
    p.add_argument(
        "--validate_every",
        type=int,
        default=500,
    )
    p.add_argument(
        "--val_diff_batches",
        type=int,
        default=32,
    )
    p.add_argument(
        "--save_every",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--aug_data",
        action="store_true",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


# =============================================================================
# Reproducibility / config
# =============================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_configs(
    cfg,
    config_path,
):
    stable_cfg = OmegaConf.select(
        cfg,
        "model.params.stable_diffusion_config",
        default=None,
    )

    fmri_cfg = OmegaConf.select(
        cfg,
        "model.params.fmri_encoder_config",
        default=None,
    )

    if stable_cfg is None:
        raise KeyError(
            f"{config_path}: missing "
            "`model.params.stable_diffusion_config`."
        )

    if OmegaConf.select(
        cfg,
        "learning_rate",
        default=None,
    ) is None:
        raise KeyError(
            "`learning_rate` missing from config."
        )

    return (
        stable_cfg,
        fmri_cfg,
        cfg,
    )


# =============================================================================
# Strict split helpers
# =============================================================================
def object_suffix(name):
    key = str(name)[3:]
    return (
        key.rsplit(
            "_",
            1,
        )[-1]
        if "_"
        in key
        else ""
    )


def category_from_name(name):
    key = str(name)[3:]

    if (
        "_"
        in key
        and key.rsplit(
            "_",
            1,
        )[-1].isdigit()
    ):
        return key.rsplit(
            "_",
            1,
        )[0]

    return key


def selected_object_columns(
    dataset,
    suffixes,
):
    suffixes = set(
        suffixes
    )

    cols = []

    for o in range(
        int(
            dataset.obj_num
        )
    ):
        col_suffixes = {
            object_suffix(
                dataset.name_list[
                    c,
                    o,
                ]
            )
            for c in range(
                int(
                    dataset.cls_num
                )
            )
        }

        if len(
            col_suffixes
        ) != 1:
            raise RuntimeError(
                f"Object column {o} has inconsistent suffixes: "
                f"{col_suffixes}"
            )

        suffix = next(
            iter(
                col_suffixes
            )
        )

        if suffix in suffixes:
            cols.append(
                o
            )

    found = {
        object_suffix(
            dataset.name_list[
                0,
                o,
            ]
        )
        for o in cols
    }

    missing = (
        suffixes
        - found
    )

    if missing:
        raise RuntimeError(
            f"Missing requested suffixes: {sorted(missing)}"
        )

    return cols


def full_dataset_indices(
    dataset,
    suffixes,
):
    cols = selected_object_columns(
        dataset,
        suffixes,
    )

    S = int(
        dataset.eeg_data.shape[
            0
        ]
    )
    C = int(
        dataset.cls_num
    )
    O = int(
        dataset.obj_num
    )
    R = int(
        dataset.trails_num
    )

    indices = []

    for s in range(
        S
    ):
        for c in range(
            C
        ):
            for o in cols:
                for r in range(
                    R
                ):
                    idx = (
                        s
                        * (
                            C
                            * O
                            * R
                        )
                        + c
                        * (
                            O
                            * R
                        )
                        + o
                        * R
                        + r
                    )

                    indices.append(
                        idx
                    )

    return indices


class EEGOnlyDataset(Dataset):
    """
    No rendered image I/O. Used for PCA and semantic validation.
    """
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

        S = int(
            base.eeg_data.shape[
                0
            ]
        )
        C = int(
            base.cls_num
        )
        R = int(
            base.trails_num
        )

        self.items = []

        for s in range(
            S
        ):
            for c in range(
                C
            ):
                for o in self.cols:
                    if mode == "individual":
                        for r in range(
                            R
                        ):
                            self.items.append(
                                (
                                    s,
                                    c,
                                    o,
                                    r,
                                )
                            )

                    elif mode == "averaged":
                        self.items.append(
                            (
                                s,
                                c,
                                o,
                                None,
                            )
                        )

                    else:
                        raise ValueError(
                            mode
                        )

    def __len__(self):
        return len(
            self.items
        )

    def __getitem__(
        self,
        idx,
    ):
        s, c, o, r = self.items[
            idx
        ]

        raw = self.base.eeg_data

        if r is None:
            eeg = np.asarray(
                raw[
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
                raw[
                    s,
                    c,
                    o,
                    r,
                ],
                dtype=np.float32,
            )

            trial_index = int(
                r
            )

        name = str(
            self.base.name_list[
                c,
                o,
            ]
        )

        return {
            "name": name,
            "cls_index": int(
                c
            ),
            "obj_index": int(
                o
            ),
            "trial_index": trial_index,
            "eeg_data": torch.from_numpy(
                eeg
            ),
        }


# =============================================================================
# Fixed train-only category text geometry
# =============================================================================
def to_flat_feature(x):
    if torch.is_tensor(
        x
    ):
        return x.detach().float().cpu().reshape(
            -1
        )

    return torch.as_tensor(
        np.asarray(
            x
        ),
        dtype=torch.float32,
    ).reshape(
        -1
    )


def build_fixed_category_text_prototypes(
    base_dataset,
    source_suffixes,
    expected_dim=1024,
):
    source_cols = selected_object_columns(
        base_dataset,
        source_suffixes,
    )

    prototypes = []
    categories = []
    source_objects = []

    for c in range(
        int(
            base_dataset.cls_num
        )
    ):
        feats = []
        cats = []
        objs = []

        for o in source_cols:
            dataset_name = str(
                base_dataset.name_list[
                    c,
                    o,
                ]
            )

            key = dataset_name[
                3:
            ]

            cat = category_from_name(
                dataset_name
            )

            feat = to_flat_feature(
                base_dataset.clip_features[
                    key
                ][
                    "text"
                ]
            )

            if feat.numel() != expected_dim:
                raise RuntimeError(
                    f"{key}: expected text dim {expected_dim}, "
                    f"got {feat.numel()}"
                )

            feats.append(
                F.normalize(
                    feat,
                    dim=0,
                )
            )

            cats.append(
                cat
            )

            objs.append(
                key
            )

        if len(
            set(
                cats
            )
        ) != 1:
            raise RuntimeError(
                f"cls_index={c} maps to multiple categories: "
                f"{set(cats)}"
            )

        prototypes.append(
            F.normalize(
                torch.stack(
                    feats
                ).mean(
                    dim=0
                ),
                dim=0,
            )
        )

        categories.append(
            cats[
                0
            ]
        )

        source_objects.append(
            objs
        )

    prototypes = torch.stack(
        prototypes,
        dim=0,
    )

    if tuple(
        prototypes.shape
    ) != (
        72,
        expected_dim,
    ):
        raise RuntimeError(
            f"Expected [72,{expected_dim}], "
            f"got {tuple(prototypes.shape)}"
        )

    return (
        prototypes,
        categories,
        source_objects,
    )


def save_prototypes(
    out_dir,
    prototypes,
    categories,
    source_objects,
    source_suffixes,
):
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "categories": categories,
            "source_objects": source_objects,
            "source_suffixes": list(
                source_suffixes
            ),
            "construction": (
                "normalize each TRAIN-object text feature "
                "-> category mean -> normalize"
            ),
        },
        out_dir
        / "fixed_category_text_prototypes.pt",
    )


# =============================================================================
# TemporalStem checkpoint + train-only PCA
# =============================================================================
def load_only_temporal_stem(
    model,
    checkpoint_path,
):
    path = Path(
        checkpoint_path
    ).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Temporal checkpoint not found: {path}"
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    state = (
        ckpt[
            "model"
        ]
        if isinstance(
            ckpt,
            dict,
        )
        and "model"
        in ckpt
        else ckpt
    )

    if not isinstance(
        state,
        dict,
    ):
        raise RuntimeError(
            "Temporal checkpoint state is not a dict."
        )

    target = model.fmri_encoder.temporal_stem.state_dict()
    extracted = {}

    for key, value in state.items():
        k = str(
            key
        )

        if k.startswith(
            "module."
        ):
            k = k[
                len(
                    "module."
                ):
            ]

        prefix = (
            "fmri_encoder.temporal_stem."
        )

        if k.startswith(
            prefix
        ):
            extracted[
                k[
                    len(
                        prefix
                    ):
                ]
            ] = value

    if set(
        extracted
    ) != set(
        target
    ):
        raise RuntimeError(
            "Failed to extract exact TemporalStem state.\n"
            f"expected={sorted(target.keys())}\n"
            f"got={sorted(extracted.keys())}"
        )

    model.fmri_encoder.temporal_stem.load_state_dict(
        extracted,
        strict=True,
    )

    model.fmri_encoder.temporal_stem.eval()

    for p in model.fmri_encoder.temporal_stem.parameters():
        p.requires_grad_(
            False
        )

    print(
        f"[TemporalStem] loaded only from {path}"
    )


@torch.no_grad()
def collect_temporal_matrix(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    feats = []

    for batch in loader:
        eeg = batch[
            "eeg_data"
        ].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        f = model.extract_semantic_prior_temporal(
            eeg
        )

        feats.append(
            f.cpu()
        )

    return torch.cat(
        feats,
        dim=0,
    ).numpy().astype(
        np.float32
    )


def fit_train_only_pca(
    model,
    train_eeg_dataset,
    device,
    pca_dim,
    batch_size,
    num_workers,
    seed,
):
    matrix = collect_temporal_matrix(
        model=model,
        dataset=train_eeg_dataset,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    actual = min(
        int(
            pca_dim
        ),
        matrix.shape[
            0
        ]
        - 1,
        matrix.shape[
            1
        ],
    )

    if actual != int(
        pca_dim
    ):
        raise RuntimeError(
            f"Requested PCA={pca_dim}, but strict train data "
            f"supports only {actual}."
        )

    pca = PCA(
        n_components=actual,
        whiten=False,
        svd_solver="randomized",
        random_state=int(
            seed
        ),
    )

    pca.fit(
        matrix
    )

    model.set_semantic_prior_pca(
        pca.mean_,
        pca.components_,
    )

    return (
        pca,
        matrix.shape,
    )


def save_pca(
    path,
    pca,
    train_suffixes,
):
    np.savez_compressed(
        path,
        mean=pca.mean_.astype(
            np.float32
        ),
        components=pca.components_.astype(
            np.float32
        ),
        explained_variance_ratio=pca.explained_variance_ratio_.astype(
            np.float32
        ),
        train_suffixes=np.asarray(
            train_suffixes
        ),
    )


# =============================================================================
# Optimizer / schedule
# =============================================================================
def trainable_params(
    module,
):
    return [
        p
        for p in module.parameters()
        if p.requires_grad
    ]


def build_optimizer(
    model,
    args,
    cfg,
):
    lora_lr = (
        float(
            args.lora_lr
        )
        if args.lora_lr
        is not None
        else float(
            cfg.learning_rate
        )
    )

    groups = [
        {
            "params": trainable_params(
                model.semantic_prior_classifier_norm
            )
            + trainable_params(
                model.semantic_prior_classifier
            ),
            "lr": args.classifier_lr,
            "name": "category_classifier_CE_only",
        },
        {
            "params": trainable_params(
                model.semantic_prior_residual_norm
            )
            + trainable_params(
                model.semantic_prior_residual
            )
            + trainable_params(
                model.semantic_prior_gate
            )
            + trainable_params(
                model.semantic_to_cross
            ),
            "lr": args.generation_head_lr,
            "name": "semantic_prior_generation",
        },
        {
            "params": [
                p
                for p in model.unet.parameters()
                if p.requires_grad
            ],
            "lr": lora_lr,
            "name": "unet_lora",
        },
    ]

    if model.semantic_prior_spatial_scale > 0:
        groups.append(
            {
                "params": trainable_params(
                    model.semantic_prior_spatial
                )
                + trainable_params(
                    model.variation_to_latent
                ),
                "lr": args.generation_head_lr,
                "name": "weak_spatial_modulation",
            }
        )

    groups = [
        g
        for g in groups
        if g[
            "params"
        ]
    ]

    print(
        "[optimizer]"
    )

    for g in groups:
        n = sum(
            p.numel()
            for p in g[
                "params"
            ]
        )

        print(
            f"  {g['name']:<30s} "
            f"{n/1e6:8.3f}M "
            f"lr={g['lr']:.3e}"
        )

    return torch.optim.AdamW(
        groups,
        betas=(
            0.9,
            0.95,
        ),
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    optimizer,
    total_steps,
    warmup_steps,
):
    warmup_steps = min(
        int(
            warmup_steps
        ),
        max(
            0,
            total_steps
            - 1,
        ),
    )

    def fn(step):
        if (
            warmup_steps
            > 0
            and step
            < warmup_steps
        ):
            return max(
                1e-8,
                float(
                    step
                    + 1
                )
                / float(
                    warmup_steps
                ),
            )

        remain = max(
            1,
            total_steps
            - warmup_steps,
        )

        return max(
            0.0,
            float(
                total_steps
                - step
            )
            / float(
                remain
            ),
        )

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        fn,
    )


def teacher_mix_at_step(
    step,
    args,
):
    if args.teacher_mix_steps <= 0:
        return float(
            args.teacher_mix_end
        )

    progress = min(
        1.0,
        max(
            0.0,
            float(
                step
            )
            / float(
                args.teacher_mix_steps
            ),
        ),
    )

    return (
        float(
            args.teacher_mix_start
        )
        + progress
        * (
            float(
                args.teacher_mix_end
            )
            - float(
                args.teacher_mix_start
            )
        )
    )


# =============================================================================
# Validation
# =============================================================================
@torch.no_grad()
def evaluate_prior(
    model,
    dataset,
    device,
    batch_size,
    num_workers,
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

    total_n = 0
    total_loss = 0.0
    total_top1 = 0.0
    total_entropy = 0.0
    total_gate = 0.0

    for batch in loader:
        eeg = batch[
            "eeg_data"
        ].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = batch[
            "cls_index"
        ].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        out = model.semantic_prior_forward(
            eeg,
            cls_target=labels,
            teacher_mix=0.0,
            detach_predicted_prior_for_diffusion=True,
        )

        n = labels.shape[
            0
        ]

        total_n += n
        total_loss += float(
            out[
                "cls_loss"
            ]
        ) * n
        total_top1 += float(
            out[
                "top1"
            ]
        ) * n
        total_entropy += float(
            out[
                "entropy"
            ]
        ) * n
        total_gate += float(
            out[
                "gate"
            ].mean()
        ) * n

    if was_training:
        model.train()

    n = max(
        1,
        total_n,
    )

    return {
        "cls_loss": total_loss / n,
        "top1": total_top1 / n,
        "entropy": total_entropy / n,
        "gate": total_gate / n,
    }


@torch.no_grad()
def evaluate_diffusion(
    model,
    full_val_subset,
    args,
    device,
    step,
):
    """
    Predicted-prior generation validation:
      teacher_mix=0.0
      drop condition remains model-defined random CFG behavior.

    Fixed RNG seed per validation step keeps comparisons substantially less noisy.
    """
    loader = DataLoader(
        full_val_subset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    was_training = model.training
    model.eval()

    diff_values = []
    cls_values = []

    devices = [
        device
    ] if device.type == "cuda" else []

    with torch.random.fork_rng(
        devices=[
            device.index
            if device.index is not None
            else 0
        ] if device.type == "cuda" else []
    ):
        torch.manual_seed(
            args.seed
            + 12345
        )

        if device.type == "cuda":
            torch.cuda.manual_seed_all(
                args.seed
                + 12345
            )

        for i, batch in enumerate(
            loader
        ):
            if i >= args.val_diff_batches:
                break

            out = model(
                batch,
                stage="prior_generation",
                teacher_mix=0.0,
            )

            diff_values.append(
                float(
                    out[
                        "diff_loss"
                    ]
                )
            )

            cls_values.append(
                float(
                    out[
                        "prior_cls_loss"
                    ]
                )
            )

    if was_training:
        model.train()

    if not diff_values:
        raise RuntimeError(
            "No validation diffusion batches were evaluated."
        )

    return {
        "diff_loss": float(
            np.mean(
                diff_values
            )
        ),
        "cls_loss": float(
            np.mean(
                cls_values
            )
        ),
        "n_batches": int(
            len(
                diff_values
            )
        ),
    }


# =============================================================================
# Checkpoint / logging
# =============================================================================
def append_csv(
    path,
    row,
):
    path = Path(
        path
    )

    exists = path.exists()

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            ),
        )

        if not exists:
            writer.writeheader()

        writer.writerow(
            row
        )


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    args,
    step,
    extra=None,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stage": "prior_generation",
            "step": int(
                step
            ),
            "args": vars(
                args
            ),
            "extra": extra
            or {},
        },
        path,
    )


def cycle_loader(
    loader,
):
    while True:
        for batch in loader:
            yield batch


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    if args.pca_dim <= 0:
        raise ValueError(
            "--pca_dim must be > 0"
        )

    if args.accumulation_steps <= 0:
        raise ValueError(
            "--accumulation_steps must be > 0"
        )

    if args.prior_temperature <= 0:
        raise ValueError(
            "--prior_temperature must be > 0"
        )

    if not (
        0.0
        <= args.residual_scale
        <= 0.25
    ):
        raise ValueError(
            "Keep --residual_scale in [0,0.25]."
        )

    if args.spatial_scale < 0:
        raise ValueError(
            "--spatial_scale must be >= 0"
        )

    seed_everything(
        args.seed
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"[device] {device}; "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    universe = tuple(
        f"{i:02d}"
        for i in range(
            7
        )
    )

    train_suffixes = tuple(
        s
        for s in universe
        if s
        != args.holdout_suffix
    )

    val_suffixes = (
        args.holdout_suffix,
    )

    print(
        f"[strict] train={train_suffixes}; "
        f"holdout={val_suffixes}; "
        "object07=NOT_USED; final08_09=NOT_USED"
    )

    # -------------------------------------------------------------------------
    # Config: inject semantic-prior dimensions BEFORE model construction.
    # -------------------------------------------------------------------------
    config_path = Path(
        args.config
    ).expanduser().resolve()

    cfg = OmegaConf.load(
        config_path
    )

    OmegaConf.update(
        cfg,
        "semantic_prior_pca_dim",
        int(
            args.pca_dim
        ),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "semantic_prior_residual_rank",
        int(
            args.residual_rank
        ),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "semantic_prior_residual_scale",
        float(
            args.residual_scale
        ),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "semantic_prior_temperature",
        float(
            args.prior_temperature
        ),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "semantic_prior_spatial_scale",
        float(
            args.spatial_scale
        ),
        merge=False,
    )
    OmegaConf.update(
        cfg,
        "semantic_prior_label_smoothing",
        float(
            args.label_smoothing
        ),
        merge=False,
    )

    (
        stable_cfg,
        fmri_cfg,
        model_args,
    ) = resolve_model_configs(
        cfg,
        config_path,
    )

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()

    (
        out_dir
        / "checkpoints"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "logs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copyfile(
        config_path,
        out_dir
        / "base_config.yaml",
    )

    OmegaConf.save(
        cfg,
        out_dir
        / "semantic_prior_config.yaml",
    )

    (
        out_dir
        / "train_args.json"
    ).write_text(
        json.dumps(
            vars(
                args
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Only train=True is instantiated. This contains 00..07.
    # Explicit subsets below exclude 07; 08/09 are never instantiated.
    # -------------------------------------------------------------------------
    base_train = AllDataFeatureTwoEEG(
        data_path=args.data_path,
        sub_list=[
            args.sub_id
        ],
        train=True,
        test_mean=False,
        num_frames=6,
        rendered_view_path=args.rendered_view_path,
        aug_data=args.aug_data,
        strict_rendered_views=True,
    )

    train_full_indices = full_dataset_indices(
        base_train,
        train_suffixes,
    )

    val_full_indices = full_dataset_indices(
        base_train,
        val_suffixes,
    )

    train_full_subset = Subset(
        base_train,
        train_full_indices,
    )

    val_full_subset = Subset(
        base_train,
        val_full_indices,
    )

    train_eeg_ind = EEGOnlyDataset(
        base_train,
        train_suffixes,
        mode="individual",
    )

    val_eeg_ind = EEGOnlyDataset(
        base_train,
        val_suffixes,
        mode="individual",
    )

    val_eeg_avg = EEGOnlyDataset(
        base_train,
        val_suffixes,
        mode="averaged",
    )

    print(
        f"[samples] generation_train={len(train_full_subset)}, "
        f"generation_val={len(val_full_subset)}, "
        f"PCA_train={len(train_eeg_ind)}, "
        f"val_ind={len(val_eeg_ind)}, "
        f"val_avg={len(val_eeg_avg)}"
    )

    # -------------------------------------------------------------------------
    # Strict train-only fixed semantic geometry.
    # -------------------------------------------------------------------------
    (
        prototypes,
        categories,
        source_objects,
    ) = build_fixed_category_text_prototypes(
        base_train,
        train_suffixes,
    )

    save_prototypes(
        out_dir,
        prototypes,
        categories,
        source_objects,
        train_suffixes,
    )

    print(
        f"[fixed category text prototypes] "
        f"shape={tuple(prototypes.shape)}; "
        f"source={train_suffixes}"
    )

    # -------------------------------------------------------------------------
    # Model.
    # -------------------------------------------------------------------------
    model = MVDiffusion(
        model_args,
        stable_cfg,
        fmri_encoder_config=fmri_cfg,
        logdir=str(
            out_dir
        ),
        num_classes=72,
    ).to(
        device
    )

    model.set_fixed_prototypes(
        prototypes.to(
            device
        )
    )

    # Only frozen TemporalStem is inherited from the previous semantic model.
    load_only_temporal_stem(
        model,
        args.temporal_ckpt,
    )

    # -------------------------------------------------------------------------
    # Strict train-only PCA.
    # -------------------------------------------------------------------------
    (
        pca,
        pca_train_shape,
    ) = fit_train_only_pca(
        model=model,
        train_eeg_dataset=train_eeg_ind,
        device=device,
        pca_dim=args.pca_dim,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    pca_path = (
        out_dir
        / "semantic_prior_pca_train_only.npz"
    )

    save_pca(
        pca_path,
        pca,
        train_suffixes,
    )

    print(
        f"[PCA] train matrix={pca_train_shape}; "
        f"dim={pca.n_components_}; "
        f"explained_var={pca.explained_variance_ratio_.sum():.4f}"
    )

    model.set_semantic_prior_training()
    model.train()

    # -------------------------------------------------------------------------
    # Optimizer.
    # -------------------------------------------------------------------------
    optimizer = build_optimizer(
        model,
        args,
        cfg,
    )

    scheduler = build_scheduler(
        optimizer,
        args.max_steps,
        args.warmup_steps,
    )

    train_loader = DataLoader(
        train_full_subset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers
        > 0,
        pin_memory=True,
        drop_last=True,
    )

    train_iter = cycle_loader(
        train_loader
    )

    writer = SummaryWriter(
        str(
            out_dir
            / "logs"
        )
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # -------------------------------------------------------------------------
    # Train.
    # -------------------------------------------------------------------------
    step = 0
    micro = 0
    start_time = time.time()

    best_key = None
    best_step = 0

    log_csv = (
        out_dir
        / "train_log.csv"
    )

    while step < args.max_steps:
        batch = next(
            train_iter
        )

        teacher_mix = teacher_mix_at_step(
            step,
            args,
        )

        out = model(
            batch,
            stage="prior_generation",
            teacher_mix=teacher_mix,
        )

        total = (
            float(
                args.lambda_diff
            )
            * out[
                "diff_loss"
            ]
            + float(
                args.lambda_cls
            )
            * out[
                "prior_cls_loss"
            ]
            + float(
                args.lambda_gate
            )
            * out[
                "prior_gate"
            ]
        )

        (
            total
            / args.accumulation_steps
        ).backward()

        micro += 1

        if (
            micro
            % args.accumulation_steps
            != 0
        ):
            continue

        if args.grad_clip > 0:
            clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                args.grad_clip,
            )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        step += 1

        if (
            step
            == 1
            or step
            % args.print_every
            == 0
        ):
            print(
                f"[TRAIN] step={step:06d} "
                f"Total={total.item():.4f} "
                f"Diff={out['diff_loss'].item():.4f} "
                f"Cls={out['prior_cls_loss'].item():.4f} "
                f"Top1={out['prior_top1'].item():.3f} "
                f"H={out['prior_entropy'].item():.3f} "
                f"Gate={out['prior_gate'].item():.4f} "
                f"Teacher={teacher_mix:.3f} "
                f"time={(time.time()-start_time)/60:.1f}m"
            )

            writer.add_scalar(
                "train/total",
                total.item(),
                step,
            )
            writer.add_scalar(
                "train/diff",
                out[
                    "diff_loss"
                ].item(),
                step,
            )
            writer.add_scalar(
                "train/prior_cls",
                out[
                    "prior_cls_loss"
                ].item(),
                step,
            )
            writer.add_scalar(
                "train/prior_top1",
                out[
                    "prior_top1"
                ].item(),
                step,
            )
            writer.add_scalar(
                "train/prior_entropy",
                out[
                    "prior_entropy"
                ].item(),
                step,
            )
            writer.add_scalar(
                "train/residual_gate",
                out[
                    "prior_gate"
                ].item(),
                step,
            )
            writer.add_scalar(
                "train/teacher_mix",
                teacher_mix,
                step,
            )

        # ---------------------------------------------------------------------
        # Strict holdout validation.
        # ---------------------------------------------------------------------
        if (
            step
            % args.validate_every
            == 0
            or step
            == args.max_steps
        ):
            val_ind = evaluate_prior(
                model=model,
                dataset=val_eeg_ind,
                device=device,
                batch_size=max(
                    1,
                    args.batchsize
                    * 16,
                ),
                num_workers=args.num_workers,
            )

            val_avg = evaluate_prior(
                model=model,
                dataset=val_eeg_avg,
                device=device,
                batch_size=max(
                    1,
                    args.batchsize
                    * 16,
                ),
                num_workers=args.num_workers,
            )

            val_diff = evaluate_diffusion(
                model=model,
                full_val_subset=val_full_subset,
                args=args,
                device=device,
                step=step,
            )

            print(
                f"[VAL] step={step:06d} "
                f"Diff(pred-prior)={val_diff['diff_loss']:.4f} | "
                f"IndCls={val_ind['cls_loss']:.4f} "
                f"IndTop1={val_ind['top1']:.4f} | "
                f"AvgCls={val_avg['cls_loss']:.4f} "
                f"AvgTop1={val_avg['top1']:.4f} "
                f"AvgH={val_avg['entropy']:.3f} "
                f"Gate={val_avg['gate']:.4f}"
            )

            row = {
                "step": step,
                "teacher_mix": teacher_mix,
                "train_diff": float(
                    out[
                        "diff_loss"
                    ].item()
                ),
                "train_cls": float(
                    out[
                        "prior_cls_loss"
                    ].item()
                ),
                "val_diff_predicted_prior": val_diff[
                    "diff_loss"
                ],
                "val_ind_cls": val_ind[
                    "cls_loss"
                ],
                "val_ind_top1": val_ind[
                    "top1"
                ],
                "val_ind_entropy": val_ind[
                    "entropy"
                ],
                "val_avg_cls": val_avg[
                    "cls_loss"
                ],
                "val_avg_top1": val_avg[
                    "top1"
                ],
                "val_avg_entropy": val_avg[
                    "entropy"
                ],
                "val_avg_gate": val_avg[
                    "gate"
                ],
            }

            append_csv(
                log_csv,
                row,
            )

            for k, v in row.items():
                if k == "step":
                    continue

                writer.add_scalar(
                    f"val/{k}",
                    float(
                        v
                    ),
                    step,
                )

            # Generation objective is primary.
            # Smaller diffusion loss, then better averaged category Top1/CE.
            current_key = (
                -float(
                    val_diff[
                        "diff_loss"
                    ]
                ),
                float(
                    val_avg[
                        "top1"
                    ]
                ),
                -float(
                    val_avg[
                        "cls_loss"
                    ]
                ),
            )

            extra = {
                "strict_train_suffixes": train_suffixes,
                "strict_holdout_suffix": args.holdout_suffix,
                "pca_path": str(
                    pca_path
                ),
                "pca_explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "temporal_ckpt": str(
                    Path(
                        args.temporal_ckpt
                    ).expanduser().resolve()
                ),
                "val": row,
            }

            save_checkpoint(
                out_dir
                / "checkpoints"
                / "last.pt",
                model,
                optimizer,
                scheduler,
                args,
                step,
                extra=extra,
            )

            if (
                best_key
                is None
                or current_key
                > best_key
            ):
                best_key = current_key
                best_step = step

                save_checkpoint(
                    out_dir
                    / "checkpoints"
                    / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    args,
                    step,
                    extra=extra,
                )

                print(
                    f"[BEST] step={step}; "
                    f"valDiff={val_diff['diff_loss']:.4f}; "
                    f"valAvgTop1={val_avg['top1']:.4f}"
                )

        if (
            step
            % args.save_every
            == 0
        ):
            save_checkpoint(
                out_dir
                / "checkpoints"
                / f"step_{step:06d}.pt",
                model,
                optimizer,
                scheduler,
                args,
                step,
                extra={
                    "strict_train_suffixes": train_suffixes,
                    "strict_holdout_suffix": args.holdout_suffix,
                },
            )

    writer.close()

    summary = {
        "best_step": int(
            best_step
        ),
        "strict_train_suffixes": list(
            train_suffixes
        ),
        "strict_holdout_suffix": args.holdout_suffix,
        "object07": "NOT_USED",
        "final08_09": "NOT_USED",
        "temporal_ckpt": str(
            Path(
                args.temporal_ckpt
            ).expanduser().resolve()
        ),
        "pca_dim": int(
            pca.n_components_
        ),
        "pca_explained_variance_ratio_sum": float(
            pca.explained_variance_ratio_.sum()
        ),
        "residual_scale": float(
            args.residual_scale
        ),
        "spatial_scale": float(
            args.spatial_scale
        ),
        "teacher_mix": {
            "start": float(
                args.teacher_mix_start
            ),
            "end": float(
                args.teacher_mix_end
            ),
            "steps": int(
                args.teacher_mix_steps
            ),
        },
        "best_checkpoint": str(
            out_dir
            / "checkpoints"
            / "best.pt"
        ),
    }

    (
        out_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"[done] best_step={best_step}; "
        f"best={out_dir / 'checkpoints' / 'best.pt'}"
    )


if __name__ == "__main__":
    main()