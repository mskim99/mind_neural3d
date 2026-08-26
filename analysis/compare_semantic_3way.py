#!/usr/bin/env python3
"""
Three-way semantic diagnosis for the current 72-way EEG model.

Compares the SAME validation semantic feature using:

  1) learned classifier
       semantic_feature -> semantic_cls_head

  2) train semantic prototype
       semantic_feature -> cosine to 72 class means computed from TRAIN EEG semantic features

  3) CLIP text prototype
       semantic_feature -> cosine to 72 class means computed from the TRAIN dataset's
                           precomputed `txt_fea`

Why use dataset `txt_fea` instead of re-encoding class-name strings?
---------------------------------------------------------------
The current Neuro-3D training pipeline already supplies `txt_fea` as the CLIP-space
text target associated with each EEG/object sample. Using exactly those saved
features makes this audit match the semantic space actually used during training,
without introducing another CLIP model/version/prompt-template mismatch.

The script loads ONLY:
    - EEG_Detangling_Disentanglement_Model
    - semantic_cls_head

It does NOT instantiate Zero123++ / VAE / UNet.

Main diagnostic question
------------------------
If validation behaves like:

    learned classifier     ~ 2%
    train semantic proto   >> 2%
    CLIP text proto        >> 2%

then the semantic encoder still contains useful validation structure and the
learned classification head is the main failure.

If all three are near chance, the semantic encoder itself is not generalizing
to unseen validation objects.

If train semantic prototype works but CLIP text prototype fails, category
structure exists in the learned semantic space but CLIP alignment is weak.

If CLIP text prototype works while train semantic prototype is poor, the
semantic feature may preserve external semantic alignment even though
train-instance geometry is distorted.

Outputs
-------
<out_dir>/
  summary.json
  method_comparison.csv
  val_predictions_3way.csv
  per_class_3way.csv
  prototype_alignment.csv
  train_semantic_prototypes.npy
  clip_text_prototypes.npy
  val_semantic_features.npy
  val_targets.npy

  learned_confusion.png
  train_proto_confusion.png
  clip_text_confusion.png
  method_accuracy.png
  correct_class_cosine.png

Dependencies
------------
PyTorch, numpy, pandas, matplotlib.
No open_clip package is required because the dataset's precomputed `txt_fea`
is used.

Example
-------
CUDA_VISIBLE_DEVICES=1 python compare_semantic_3way.py \
    --ckpt ./stage2_semantic_cls/checkpoints/model_final.pt \
    --data_path /data/jionkim/neuro_3D \
    --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
    --sub_id sub01 \
    --out_dir ./semantic_3way \
    --batch_size 32
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.mvdiffusion_var_semantic_cls import (
    EEG_Detangling_Disentanglement_Model,
)
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare learned classifier vs train semantic prototypes "
            "vs CLIP text prototypes."
        )
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint from train_neural3d_pp_semantic_cls.py",
    )
    p.add_argument(
        "--data_path",
        default="/data/jionkim/neuro_3D",
    )
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", default="./semantic_3way")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_classes", type=int, default=72)

    p.add_argument(
        "--prototype_mode",
        choices=["mean_then_normalize", "normalize_then_mean"],
        default="mean_then_normalize",
        help=(
            "How to form class prototypes. Default: average raw features "
            "within each class and L2-normalize the resulting class mean."
        ),
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Optional cosine-logit temperature for prototype softmax diagnostics. "
            "Predicted class is unaffected by any positive value."
        ),
    )
    p.add_argument(
        "--save_train_features",
        action="store_true",
        help="Also save all extracted train semantic features and labels.",
    )
    return p.parse_args()


# ============================================================================
# Model: semantic encoder + trained classifier only
# ============================================================================

class SemanticClassifierOnly(nn.Module):
    """
    Architecture matched to the semantic-class-supervised checkpoint.

    This is intentionally the same lightweight architecture used in the
    preceding validation-confusion audit.
    """
    def __init__(self, num_classes=72):
        super().__init__()

        self.fmri_encoder = EEG_Detangling_Disentanglement_Model(
            num_electrodes=64,
            seq_len=600,
            embed_dim=1024,
        )

        self.semantic_cls_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, eeg):
        sem_feat, var_feat, bias_feat = self.fmri_encoder(eeg)
        logits = self.semantic_cls_head(sem_feat.float())
        return logits, sem_feat, var_feat, bias_feat


def strip_module_prefix(state):
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def load_semantic_checkpoint(model, ckpt_path):
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = (
        ckpt["model"]
        if isinstance(ckpt, dict) and "model" in ckpt
        else ckpt
    )
    state = strip_module_prefix(state)

    wanted = {}
    for key, value in state.items():
        if key.startswith("fmri_encoder."):
            wanted[key] = value
        elif key.startswith("semantic_cls_head."):
            wanted[key] = value

    encoder_keys = [
        k for k in wanted
        if k.startswith("fmri_encoder.")
    ]
    classifier_keys = [
        k for k in wanted
        if k.startswith("semantic_cls_head.")
    ]

    if not encoder_keys:
        raise RuntimeError(
            "Checkpoint contains no `fmri_encoder.*` weights."
        )
    if not classifier_keys:
        raise RuntimeError(
            "Checkpoint contains no `semantic_cls_head.*` weights."
        )

    incompatible = model.load_state_dict(
        wanted,
        strict=False,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[checkpoint mismatch]")
        print("missing   :", incompatible.missing_keys[:30])
        print("unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError(
            "Semantic encoder/classifier checkpoint architecture mismatch."
        )

    step = None
    stage = None
    if isinstance(ckpt, dict):
        step = ckpt.get(
            "optimizer_step",
            ckpt.get("global_step", None),
        )
        stage = ckpt.get("stage", None)

    print(f"[checkpoint] {ckpt_path}")
    if step is not None:
        print(f"[checkpoint] optimizer_step={step}")
    if stage is not None:
        print(f"[checkpoint] stage={stage}")

    return {
        "optimizer_step": step,
        "stage": stage,
    }


# ============================================================================
# Dataset
# ============================================================================

def ensure_trailing_sep(path):
    path = str(Path(path).expanduser().resolve())
    if not path.endswith(os.sep):
        path += os.sep
    return path


def build_dataset(args, train):
    return AllDataFeatureTwoEEG(
        data_path=ensure_trailing_sep(args.data_path),
        sub_list=[args.sub_id],
        train=train,
        num_frames=6,
        rendered_view_path=str(
            Path(args.rendered_view_path)
            .expanduser()
            .resolve()
        ),
        aug_data=False,
        strict_rendered_views=True,
    )


def category_from_dataset_name(name):
    """
    13_can_08 -> can
    01_airplane_00 -> airplane
    """
    name = str(name)
    key = name[3:]
    if "_" in key:
        return key.rsplit("_", 1)[0]
    return key


def build_class_metadata(ds):
    names = []
    prefixes = []

    for cls_idx in range(ds.cls_num):
        raw = str(ds.name_list[cls_idx, 0])
        prefixes.append(raw[:3])
        names.append(
            category_from_dataset_name(raw)
        )

    return names, prefixes


# ============================================================================
# Robust batch metadata
# ============================================================================

def tensor_or_list_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_feature_to_2d(x, expected_batch=None):
    """
    Robustly convert features to [B,D].

    Supports:
        [B,D]
        [B,1,D]
        [B,D,1]
        singleton-heavy variants
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    x = x.float()

    if expected_batch is not None and x.shape[0] != expected_batch:
        raise ValueError(
            f"Feature batch mismatch: got {tuple(x.shape)}, "
            f"expected first dim={expected_batch}"
        )

    # Keep batch dim; remove singleton dimensions after it.
    while x.ndim > 2:
        squeeze_dim = None
        for d in range(1, x.ndim):
            if x.shape[d] == 1:
                squeeze_dim = d
                break
        if squeeze_dim is None:
            # If no singleton exists, flatten non-batch dims.
            x = x.reshape(x.shape[0], -1)
            break
        x = x.squeeze(squeeze_dim)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    return x


# ============================================================================
# Feature extraction
# ============================================================================

@torch.no_grad()
def extract_split(model, ds, args, split_name):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)

    all_semantic = []
    all_text = []
    all_logits = []
    all_targets = []
    all_names = []
    all_labels = []
    all_obj = []
    all_trial = []

    for batch in loader:
        eeg = batch["eeg_data"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if "cls_index" not in batch:
            raise KeyError(
                "Current dataset must return `cls_index`."
            )

        target = batch["cls_index"].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        logits, sem_feat, _, _ = model(eeg)

        sem_feat = squeeze_feature_to_2d(
            sem_feat,
            expected_batch=eeg.shape[0],
        )

        txt_fea = squeeze_feature_to_2d(
            batch["txt_fea"],
            expected_batch=eeg.shape[0],
        )

        if sem_feat.shape[-1] != txt_fea.shape[-1]:
            raise RuntimeError(
                "Semantic feature and txt_fea live in different dimensions: "
                f"semantic={tuple(sem_feat.shape)}, "
                f"text={tuple(txt_fea.shape)}. "
                "A projection used during training must be applied before "
                "CLIP-text prototype comparison."
            )

        all_semantic.append(
            sem_feat.detach().float().cpu()
        )
        all_text.append(
            txt_fea.detach().float().cpu()
        )
        all_logits.append(
            logits.detach().float().cpu()
        )
        all_targets.append(
            target.detach().cpu()
        )

        names = batch.get(
            "name",
            [""] * eeg.shape[0],
        )
        labels = batch.get(
            "label",
            [""] * eeg.shape[0],
        )

        if isinstance(names, (tuple, list)):
            all_names.extend(
                [str(x) for x in names]
            )
        else:
            all_names.extend(
                [str(x) for x in names]
            )

        if isinstance(labels, (tuple, list)):
            all_labels.extend(
                [str(x) for x in labels]
            )
        else:
            all_labels.extend(
                [str(x) for x in labels]
            )

        obj = batch.get(
            "obj_index",
            torch.full(
                (eeg.shape[0],),
                -1,
                dtype=torch.long,
            ),
        )
        trial = batch.get(
            "trial_index",
            torch.full(
                (eeg.shape[0],),
                -1,
                dtype=torch.long,
            ),
        )

        all_obj.extend(
            tensor_or_list_to_numpy(obj)
            .reshape(-1)
            .astype(int)
            .tolist()
        )
        all_trial.extend(
            tensor_or_list_to_numpy(trial)
            .reshape(-1)
            .astype(int)
            .tolist()
        )

    semantic = torch.cat(
        all_semantic,
        dim=0,
    )
    text = torch.cat(
        all_text,
        dim=0,
    )
    logits = torch.cat(
        all_logits,
        dim=0,
    )
    targets = torch.cat(
        all_targets,
        dim=0,
    ).long()

    if len(all_names) != len(targets):
        raise RuntimeError(
            f"{split_name}: metadata length mismatch: "
            f"names={len(all_names)}, targets={len(targets)}"
        )

    print(
        f"[{split_name}] semantic={tuple(semantic.shape)}, "
        f"text={tuple(text.shape)}, "
        f"logits={tuple(logits.shape)}, "
        f"targets={tuple(targets.shape)}"
    )

    return {
        "semantic": semantic,
        "text": text,
        "logits": logits,
        "targets": targets,
        "names": all_names,
        "labels": all_labels,
        "obj_index": all_obj,
        "trial_index": all_trial,
    }


# ============================================================================
# Prototypes
# ============================================================================

def build_prototypes(
    features,
    targets,
    num_classes,
    mode,
):
    """
    features: [N,D]
    targets : [N]

    mean_then_normalize:
        p_c = normalize(mean(x_i))

    normalize_then_mean:
        p_c = normalize(mean(normalize(x_i)))
    """
    features = features.float()
    targets = targets.long()

    prototypes = []
    counts = []

    for cls_idx in range(num_classes):
        mask = targets == cls_idx
        x = features[mask]

        if len(x) == 0:
            raise RuntimeError(
                f"No training samples for class {cls_idx}."
            )

        if mode == "normalize_then_mean":
            x = F.normalize(
                x,
                dim=-1,
            )

        proto = x.mean(
            dim=0,
        )
        proto = F.normalize(
            proto,
            dim=-1,
        )

        prototypes.append(proto)
        counts.append(int(mask.sum().item()))

    return (
        torch.stack(prototypes, dim=0),
        np.asarray(counts, dtype=np.int64),
    )


def cosine_logits(
    features,
    prototypes,
    temperature=1.0,
):
    if temperature <= 0:
        raise ValueError(
            "--temperature must be > 0"
        )

    f = F.normalize(
        features.float(),
        dim=-1,
    )
    p = F.normalize(
        prototypes.float(),
        dim=-1,
    )

    return (f @ p.T) / float(temperature)


# ============================================================================
# Metrics
# ============================================================================

def topk_accuracy(logits, targets, k):
    k = min(
        int(k),
        logits.shape[1],
    )
    top = logits.topk(
        k=k,
        dim=1,
    ).indices
    correct = (
        top
        == targets[:, None]
    ).any(dim=1)

    return float(
        correct.float().mean().item()
    )


def effective_num_classes(pred, num_classes):
    pred = pred.detach().cpu().numpy()
    counts = np.bincount(
        pred,
        minlength=num_classes,
    ).astype(np.float64)

    p = counts / max(
        counts.sum(),
        1.0,
    )
    p = p[p > 0]

    entropy = float(
        -(p * np.log2(p)).sum()
    )
    effective = float(
        2.0 ** entropy
    )

    return {
        "classes_used": int(
            np.count_nonzero(counts)
        ),
        "prediction_entropy_bits": entropy,
        "effective_classes": effective,
        "prediction_counts": counts.astype(int),
    }


def classification_metrics(
    logits,
    targets,
    num_classes,
):
    pred = logits.argmax(
        dim=1,
    )

    concentration = effective_num_classes(
        pred,
        num_classes,
    )

    return {
        "top1": topk_accuracy(
            logits,
            targets,
            1,
        ),
        "top5": topk_accuracy(
            logits,
            targets,
            5,
        ),
        "classes_used": concentration[
            "classes_used"
        ],
        "prediction_entropy_bits": concentration[
            "prediction_entropy_bits"
        ],
        "effective_classes": concentration[
            "effective_classes"
        ],
        "prediction_counts": concentration[
            "prediction_counts"
        ],
        "pred": pred,
    }


def cosine_diagnostics(
    logits,
    targets,
):
    """
    For cosine prototype methods:
        correct similarity
        best wrong similarity
        margin = correct - best_wrong
    """
    n = logits.shape[0]

    correct = logits[
        torch.arange(n),
        targets,
    ]

    wrong = logits.clone()
    wrong[
        torch.arange(n),
        targets,
    ] = -float("inf")

    best_wrong, best_wrong_cls = wrong.max(
        dim=1,
    )

    margin = correct - best_wrong

    return {
        "correct_similarity": correct,
        "best_wrong_similarity": best_wrong,
        "best_wrong_cls": best_wrong_cls,
        "margin": margin,
        "mean_correct_similarity": float(
            correct.mean().item()
        ),
        "mean_best_wrong_similarity": float(
            best_wrong.mean().item()
        ),
        "mean_margin": float(
            margin.mean().item()
        ),
        "positive_margin_rate": float(
            (margin > 0)
            .float()
            .mean()
            .item()
        ),
    }


def confusion_matrix(
    pred,
    targets,
    num_classes,
):
    matrix = np.zeros(
        (
            num_classes,
            num_classes,
        ),
        dtype=np.int64,
    )

    p = pred.detach().cpu().numpy()
    t = targets.detach().cpu().numpy()

    for gt, pr in zip(t, p):
        matrix[
            int(gt),
            int(pr),
        ] += 1

    return matrix


# ============================================================================
# Prototype-to-prototype semantic alignment
# ============================================================================

def analyze_prototype_alignment(
    semantic_proto,
    text_proto,
    class_names,
):
    similarity = (
        F.normalize(
            semantic_proto,
            dim=-1,
        )
        @ F.normalize(
            text_proto,
            dim=-1,
        ).T
    )

    rows = []

    top1 = 0
    top5 = 0

    for cls_idx in range(
        similarity.shape[0]
    ):
        order = torch.argsort(
            similarity[cls_idx],
            descending=True,
        )

        rank = int(
            (
                order == cls_idx
            )
            .nonzero(
                as_tuple=False
            )[0]
            .item()
        ) + 1

        top1 += int(
            rank == 1
        )
        top5 += int(
            rank <= 5
        )

        best_idx = int(
            order[0].item()
        )

        rows.append(
            {
                "cls_index": cls_idx,
                "category": class_names[cls_idx],
                "same_class_cosine": float(
                    similarity[
                        cls_idx,
                        cls_idx,
                    ].item()
                ),
                "same_class_rank": rank,
                "best_text_cls": best_idx,
                "best_text_category": class_names[
                    best_idx
                ],
                "best_text_cosine": float(
                    similarity[
                        cls_idx,
                        best_idx,
                    ].item()
                ),
            }
        )

    df = pd.DataFrame(
        rows
    )

    summary = {
        "semantic_proto_to_text_top1": (
            top1
            / similarity.shape[0]
        ),
        "semantic_proto_to_text_top5": (
            top5
            / similarity.shape[0]
        ),
        "mean_same_class_cosine": float(
            torch.diag(
                similarity
            ).mean().item()
        ),
        "median_same_class_rank": float(
            df[
                "same_class_rank"
            ].median()
        ),
    }

    return (
        similarity,
        df,
        summary,
    )


# ============================================================================
# Per-class / per-sample tables
# ============================================================================

def build_sample_table(
    val,
    learned_logits,
    semantic_logits,
    text_logits,
    class_names,
):
    targets = val[
        "targets"
    ]

    learned_pred = learned_logits.argmax(
        dim=1
    )
    semantic_pred = semantic_logits.argmax(
        dim=1
    )
    text_pred = text_logits.argmax(
        dim=1
    )

    sem_diag = cosine_diagnostics(
        semantic_logits,
        targets,
    )
    text_diag = cosine_diagnostics(
        text_logits,
        targets,
    )

    learned_prob = torch.softmax(
        learned_logits,
        dim=-1,
    )
    learned_conf = learned_prob.max(
        dim=1
    ).values
    learned_gt_prob = learned_prob[
        torch.arange(
            len(targets)
        ),
        targets,
    ]

    rows = []

    for i in range(
        len(targets)
    ):
        gt = int(
            targets[i].item()
        )
        l = int(
            learned_pred[i].item()
        )
        s = int(
            semantic_pred[i].item()
        )
        t = int(
            text_pred[i].item()
        )

        rows.append(
            {
                "sample_index": i,
                "dataset_name": val[
                    "names"
                ][i],
                "label": val[
                    "labels"
                ][i],
                "obj_index": val[
                    "obj_index"
                ][i],
                "trial_index": val[
                    "trial_index"
                ][i],

                "gt_cls": gt,
                "gt_category": class_names[
                    gt
                ],

                "learned_pred_cls": l,
                "learned_pred_category": class_names[
                    l
                ],
                "learned_correct": int(
                    l == gt
                ),
                "learned_confidence": float(
                    learned_conf[i].item()
                ),
                "learned_gt_probability": float(
                    learned_gt_prob[i].item()
                ),

                "semantic_proto_pred_cls": s,
                "semantic_proto_pred_category": class_names[
                    s
                ],
                "semantic_proto_correct": int(
                    s == gt
                ),
                "semantic_correct_cosine": float(
                    sem_diag[
                        "correct_similarity"
                    ][i].item()
                ),
                "semantic_best_wrong_cosine": float(
                    sem_diag[
                        "best_wrong_similarity"
                    ][i].item()
                ),
                "semantic_margin": float(
                    sem_diag[
                        "margin"
                    ][i].item()
                ),

                "clip_text_pred_cls": t,
                "clip_text_pred_category": class_names[
                    t
                ],
                "clip_text_correct": int(
                    t == gt
                ),
                "clip_text_correct_cosine": float(
                    text_diag[
                        "correct_similarity"
                    ][i].item()
                ),
                "clip_text_best_wrong_cosine": float(
                    text_diag[
                        "best_wrong_similarity"
                    ][i].item()
                ),
                "clip_text_margin": float(
                    text_diag[
                        "margin"
                    ][i].item()
                ),

                "all_three_agree": int(
                    l == s == t
                ),
                "semantic_and_text_agree": int(
                    s == t
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_per_class_table(
    sample_df,
    class_names,
):
    rows = []

    for cls_idx, category in enumerate(
        class_names
    ):
        d = sample_df[
            sample_df["gt_cls"]
            == cls_idx
        ]

        if len(d) == 0:
            continue

        rows.append(
            {
                "cls_index": cls_idx,
                "category": category,
                "n": len(d),

                "learned_acc": float(
                    d[
                        "learned_correct"
                    ].mean()
                ),
                "semantic_proto_acc": float(
                    d[
                        "semantic_proto_correct"
                    ].mean()
                ),
                "clip_text_acc": float(
                    d[
                        "clip_text_correct"
                    ].mean()
                ),

                "semantic_correct_cosine": float(
                    d[
                        "semantic_correct_cosine"
                    ].mean()
                ),
                "semantic_margin": float(
                    d[
                        "semantic_margin"
                    ].mean()
                ),

                "clip_text_correct_cosine": float(
                    d[
                        "clip_text_correct_cosine"
                    ].mean()
                ),
                "clip_text_margin": float(
                    d[
                        "clip_text_margin"
                    ].mean()
                ),

                "learned_mode_pred": (
                    d[
                        "learned_pred_category"
                    ]
                    .value_counts()
                    .index[0]
                ),
                "semantic_mode_pred": (
                    d[
                        "semantic_proto_pred_category"
                    ]
                    .value_counts()
                    .index[0]
                ),
                "text_mode_pred": (
                    d[
                        "clip_text_pred_category"
                    ]
                    .value_counts()
                    .index[0]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================================
# Plotting
# ============================================================================

def plot_confusion(
    matrix,
    class_names,
    path,
    title,
):
    row_sum = matrix.sum(
        axis=1,
        keepdims=True,
    )
    norm = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(
            matrix,
            dtype=np.float64,
        ),
        where=row_sum > 0,
    )

    n = len(
        class_names
    )

    fig = plt.figure(
        figsize=(18, 16)
    )
    ax = fig.add_subplot(
        111
    )

    im = ax.imshow(
        norm,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ticks = np.arange(
        n
    )
    ax.set_xticks(
        ticks
    )
    ax.set_yticks(
        ticks
    )
    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=5,
    )
    ax.set_yticklabels(
        class_names,
        fontsize=5,
    )

    ax.set_xlabel(
        "Predicted class"
    )
    ax.set_ylabel(
        "Ground-truth class"
    )
    ax.set_title(
        title
    )

    ax.plot(
        [-0.5, n - 0.5],
        [-0.5, n - 0.5],
        linewidth=0.7,
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_method_accuracy(
    method_rows,
    path,
):
    df = pd.DataFrame(
        method_rows
    )

    x = np.arange(
        len(df)
    )
    width = 0.35

    fig = plt.figure(
        figsize=(9, 5)
    )
    ax = fig.add_subplot(
        111
    )

    ax.bar(
        x - width / 2,
        df["top1"].values,
        width,
        label="Top-1",
    )
    ax.bar(
        x + width / 2,
        df["top5"].values,
        width,
        label="Top-5",
    )

    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        df["method"].values,
        rotation=15,
    )
    ax.set_ylim(
        0,
        1,
    )
    ax.set_ylabel(
        "Accuracy"
    )
    ax.set_title(
        "Validation semantic decoding: 3-way comparison"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_correct_cosine(
    sample_df,
    path,
):
    data = [
        sample_df[
            "semantic_correct_cosine"
        ].values,
        sample_df[
            "clip_text_correct_cosine"
        ].values,
    ]

    fig = plt.figure(
        figsize=(8, 5)
    )
    ax = fig.add_subplot(
        111
    )

    ax.boxplot(
        data,
        labels=[
            "Train semantic prototype",
            "CLIP text prototype",
        ],
        showfliers=True,
    )

    ax.set_ylabel(
        "Cosine to correct class prototype"
    )
    ax.set_title(
        "Validation correct-class cosine"
    )

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


# ============================================================================
# Diagnosis logic
# ============================================================================

def automatic_diagnosis(
    learned,
    semantic,
    text,
):
    l = learned["top1"]
    s = semantic["top1"]
    t = text["top1"]

    best_proto = max(
        s,
        t,
    )

    evidence = []
    conclusion = None

    # Large gain from cosine prototypes over learned head.
    if (
        l < 0.15
        and best_proto >= 0.20
        and best_proto - l >= 0.15
    ):
        conclusion = (
            "classifier_head_failure_likely"
        )
        evidence.append(
            "Validation prototype decoding substantially exceeds "
            "the learned classifier."
        )

    # All decoders fail.
    elif (
        l < 0.15
        and s < 0.15
        and t < 0.15
    ):
        conclusion = (
            "semantic_encoder_generalization_failure_likely"
        )
        evidence.append(
            "Learned classifier, train-semantic prototype, and "
            "CLIP-text prototype are all near chance/very low."
        )

    # Internal category geometry survives, external semantic alignment does not.
    elif (
        s >= 0.20
        and t < 0.15
        and s - t >= 0.10
    ):
        conclusion = (
            "internal_semantic_structure_survives_but_clip_alignment_is_weak"
        )
        evidence.append(
            "Train semantic prototypes decode validation much better "
            "than CLIP text prototypes."
        )

    # External semantic alignment survives better.
    elif (
        t >= 0.20
        and s < 0.15
        and t - s >= 0.10
    ):
        conclusion = (
            "clip_semantic_alignment_survives_better_than_train_instance_geometry"
        )
        evidence.append(
            "CLIP text prototypes decode validation substantially better "
            "than train semantic prototypes."
        )

    else:
        conclusion = (
            "mixed_or_intermediate_failure"
        )
        evidence.append(
            "No single failure mode dominates by the simple thresholds."
        )

    evidence.append(
        f"Top-1 learned={l:.4f}, "
        f"train-proto={s:.4f}, "
        f"clip-text={t:.4f}."
    )

    return {
        "label": conclusion,
        "evidence": evidence,
        "note": (
            "This is a heuristic diagnosis. Inspect per-class results, "
            "cosine margins, and confusion matrices before changing training."
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    out_dir = (
        Path(args.out_dir)
        .expanduser()
        .resolve()
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    device = torch.device(
        args.device
    )

    # ----------------------------------------------------------------------
    # Load lightweight semantic model.
    # ----------------------------------------------------------------------
    model = SemanticClassifierOnly(
        num_classes=args.num_classes
    )

    ckpt_meta = load_semantic_checkpoint(
        model,
        args.ckpt,
    )

    model = model.to(
        device
    ).eval()

    # ----------------------------------------------------------------------
    # Datasets.
    # ----------------------------------------------------------------------
    train_ds = build_dataset(
        args,
        train=True,
    )
    val_ds = build_dataset(
        args,
        train=False,
    )

    train_class_names, train_prefixes = build_class_metadata(
        train_ds
    )
    val_class_names, val_prefixes = build_class_metadata(
        val_ds
    )

    if train_class_names != val_class_names:
        raise RuntimeError(
            "Train/test semantic category order differs. "
            "Run the alignment audit before interpreting this diagnostic."
        )

    class_names = train_class_names

    print(
        f"[train] n={len(train_ds)}, "
        f"name_list={train_ds.name_list.shape}, "
        f"eeg={train_ds.eeg_data.shape}"
    )
    print(
        f"[val]   n={len(val_ds)}, "
        f"name_list={val_ds.name_list.shape}, "
        f"eeg={val_ds.eeg_data.shape}"
    )

    # ----------------------------------------------------------------------
    # Extract semantic and text features.
    # ----------------------------------------------------------------------
    print(
        "\n[1/5] Extracting TRAIN semantic/text features ..."
    )
    train = extract_split(
        model,
        train_ds,
        args,
        split_name="train",
    )

    print(
        "\n[2/5] Extracting VAL semantic/text features ..."
    )
    val = extract_split(
        model,
        val_ds,
        args,
        split_name="val",
    )

    # ----------------------------------------------------------------------
    # Build the two prototype banks FROM TRAIN ONLY.
    # ----------------------------------------------------------------------
    print(
        "\n[3/5] Building TRAIN prototype banks ..."
    )

    semantic_proto, semantic_counts = build_prototypes(
        train["semantic"],
        train["targets"],
        args.num_classes,
        args.prototype_mode,
    )

    text_proto, text_counts = build_prototypes(
        train["text"],
        train["targets"],
        args.num_classes,
        args.prototype_mode,
    )

    if not np.array_equal(
        semantic_counts,
        text_counts,
    ):
        raise RuntimeError(
            "Semantic/text prototype sample counts differ."
        )

    print(
        "[prototype] samples/class min/max = "
        f"{semantic_counts.min()}/{semantic_counts.max()}"
    )

    # ----------------------------------------------------------------------
    # Three validation decoders.
    # ----------------------------------------------------------------------
    print(
        "\n[4/5] Running 3-way validation decoding ..."
    )

    learned_logits = val[
        "logits"
    ]

    semantic_logits = cosine_logits(
        val["semantic"],
        semantic_proto,
        temperature=args.temperature,
    )

    text_logits = cosine_logits(
        val["semantic"],
        text_proto,
        temperature=args.temperature,
    )

    learned_metrics = classification_metrics(
        learned_logits,
        val["targets"],
        args.num_classes,
    )
    semantic_metrics = classification_metrics(
        semantic_logits,
        val["targets"],
        args.num_classes,
    )
    text_metrics = classification_metrics(
        text_logits,
        val["targets"],
        args.num_classes,
    )

    semantic_cos = cosine_diagnostics(
        semantic_logits,
        val["targets"],
    )
    text_cos = cosine_diagnostics(
        text_logits,
        val["targets"],
    )

    # ----------------------------------------------------------------------
    # Train-reference metrics using same banks.
    # Useful to see whether prototype banks themselves are sensible.
    # ----------------------------------------------------------------------
    train_learned_metrics = classification_metrics(
        train["logits"],
        train["targets"],
        args.num_classes,
    )

    train_semantic_logits = cosine_logits(
        train["semantic"],
        semantic_proto,
        temperature=args.temperature,
    )
    train_text_logits = cosine_logits(
        train["semantic"],
        text_proto,
        temperature=args.temperature,
    )

    train_semantic_metrics = classification_metrics(
        train_semantic_logits,
        train["targets"],
        args.num_classes,
    )
    train_text_metrics = classification_metrics(
        train_text_logits,
        train["targets"],
        args.num_classes,
    )

    # ----------------------------------------------------------------------
    # Prototype alignment itself.
    # ----------------------------------------------------------------------
    (
        proto_alignment_matrix,
        proto_alignment_df,
        proto_alignment_summary,
    ) = analyze_prototype_alignment(
        semantic_proto,
        text_proto,
        class_names,
    )

    # ----------------------------------------------------------------------
    # Sample and class tables.
    # ----------------------------------------------------------------------
    sample_df = build_sample_table(
        val,
        learned_logits,
        semantic_logits,
        text_logits,
        class_names,
    )

    per_class_df = build_per_class_table(
        sample_df,
        class_names,
    )

    # ----------------------------------------------------------------------
    # Compact comparison table.
    # ----------------------------------------------------------------------
    method_rows = [
        {
            "method": "learned_classifier",
            "top1": learned_metrics["top1"],
            "top5": learned_metrics["top5"],
            "classes_used": learned_metrics["classes_used"],
            "effective_classes": learned_metrics["effective_classes"],
            "mean_correct_cosine": np.nan,
            "mean_margin": np.nan,
            "positive_margin_rate": np.nan,
        },
        {
            "method": "train_semantic_prototype",
            "top1": semantic_metrics["top1"],
            "top5": semantic_metrics["top5"],
            "classes_used": semantic_metrics["classes_used"],
            "effective_classes": semantic_metrics["effective_classes"],
            "mean_correct_cosine": semantic_cos[
                "mean_correct_similarity"
            ],
            "mean_margin": semantic_cos[
                "mean_margin"
            ],
            "positive_margin_rate": semantic_cos[
                "positive_margin_rate"
            ],
        },
        {
            "method": "clip_text_prototype",
            "top1": text_metrics["top1"],
            "top5": text_metrics["top5"],
            "classes_used": text_metrics["classes_used"],
            "effective_classes": text_metrics["effective_classes"],
            "mean_correct_cosine": text_cos[
                "mean_correct_similarity"
            ],
            "mean_margin": text_cos[
                "mean_margin"
            ],
            "positive_margin_rate": text_cos[
                "positive_margin_rate"
            ],
        },
    ]

    comparison_df = pd.DataFrame(
        method_rows
    )

    # ----------------------------------------------------------------------
    # Automatic diagnosis.
    # ----------------------------------------------------------------------
    diagnosis = automatic_diagnosis(
        learned_metrics,
        semantic_metrics,
        text_metrics,
    )

    # ----------------------------------------------------------------------
    # Summary.
    # ----------------------------------------------------------------------
    def metric_json(m):
        return {
            "top1": m["top1"],
            "top5": m["top5"],
            "classes_used": m["classes_used"],
            "prediction_entropy_bits": m[
                "prediction_entropy_bits"
            ],
            "effective_classes": m[
                "effective_classes"
            ],
        }

    summary = {
        "checkpoint": str(
            Path(args.ckpt)
            .expanduser()
            .resolve()
        ),
        "checkpoint_meta": ckpt_meta,
        "subject": args.sub_id,
        "num_classes": args.num_classes,
        "prototype_mode": args.prototype_mode,
        "temperature": args.temperature,

        "validation": {
            "learned_classifier": metric_json(
                learned_metrics
            ),
            "train_semantic_prototype": {
                **metric_json(
                    semantic_metrics
                ),
                "mean_correct_cosine": semantic_cos[
                    "mean_correct_similarity"
                ],
                "mean_best_wrong_cosine": semantic_cos[
                    "mean_best_wrong_similarity"
                ],
                "mean_margin": semantic_cos[
                    "mean_margin"
                ],
                "positive_margin_rate": semantic_cos[
                    "positive_margin_rate"
                ],
            },
            "clip_text_prototype": {
                **metric_json(
                    text_metrics
                ),
                "mean_correct_cosine": text_cos[
                    "mean_correct_similarity"
                ],
                "mean_best_wrong_cosine": text_cos[
                    "mean_best_wrong_similarity"
                ],
                "mean_margin": text_cos[
                    "mean_margin"
                ],
                "positive_margin_rate": text_cos[
                    "positive_margin_rate"
                ],
            },
        },

        "train_reference": {
            "learned_classifier": metric_json(
                train_learned_metrics
            ),
            "train_semantic_prototype": metric_json(
                train_semantic_metrics
            ),
            "clip_text_prototype": metric_json(
                train_text_metrics
            ),
        },

        "prototype_alignment": proto_alignment_summary,
        "diagnosis": diagnosis,
    }

    # ----------------------------------------------------------------------
    # Save arrays / tables.
    # ----------------------------------------------------------------------
    print(
        "\n[5/5] Saving diagnostics ..."
    )

    comparison_df.to_csv(
        out_dir / "method_comparison.csv",
        index=False,
    )
    sample_df.to_csv(
        out_dir / "val_predictions_3way.csv",
        index=False,
    )
    per_class_df.to_csv(
        out_dir / "per_class_3way.csv",
        index=False,
    )
    proto_alignment_df.to_csv(
        out_dir / "prototype_alignment.csv",
        index=False,
    )

    np.save(
        out_dir / "train_semantic_prototypes.npy",
        semantic_proto.numpy(),
    )
    np.save(
        out_dir / "clip_text_prototypes.npy",
        text_proto.numpy(),
    )
    np.save(
        out_dir / "val_semantic_features.npy",
        val["semantic"].numpy(),
    )
    np.save(
        out_dir / "val_targets.npy",
        val["targets"].numpy(),
    )
    np.save(
        out_dir / "semantic_text_prototype_similarity.npy",
        proto_alignment_matrix.numpy(),
    )

    if args.save_train_features:
        np.save(
            out_dir / "train_semantic_features.npy",
            train["semantic"].numpy(),
        )
        np.save(
            out_dir / "train_targets.npy",
            train["targets"].numpy(),
        )

    with open(
        out_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ----------------------------------------------------------------------
    # Confusions / figures.
    # ----------------------------------------------------------------------
    learned_conf = confusion_matrix(
        learned_metrics["pred"],
        val["targets"],
        args.num_classes,
    )
    semantic_conf = confusion_matrix(
        semantic_metrics["pred"],
        val["targets"],
        args.num_classes,
    )
    text_conf = confusion_matrix(
        text_metrics["pred"],
        val["targets"],
        args.num_classes,
    )

    plot_confusion(
        learned_conf,
        class_names,
        out_dir / "learned_confusion.png",
        "Validation: learned classifier",
    )
    plot_confusion(
        semantic_conf,
        class_names,
        out_dir / "train_proto_confusion.png",
        "Validation: nearest TRAIN semantic prototype",
    )
    plot_confusion(
        text_conf,
        class_names,
        out_dir / "clip_text_confusion.png",
        "Validation: nearest CLIP text prototype",
    )
    plot_method_accuracy(
        method_rows,
        out_dir / "method_accuracy.png",
    )
    plot_correct_cosine(
        sample_df,
        out_dir / "correct_class_cosine.png",
    )

    # ----------------------------------------------------------------------
    # Console report.
    # ----------------------------------------------------------------------
    print()
    print("=" * 96)
    print("SEMANTIC FEATURE 3-WAY VALIDATION DIAGNOSIS")
    print("=" * 96)
    print(
        f"{'Method':<30}"
        f"{'Top-1':>10}"
        f"{'Top-5':>10}"
        f"{'Used':>10}"
        f"{'Eff.cls':>12}"
        f"{'Correct cos':>14}"
        f"{'Margin':>12}"
    )
    print("-" * 96)

    for row in method_rows:
        correct_cos = row[
            "mean_correct_cosine"
        ]
        margin = row[
            "mean_margin"
        ]

        cos_str = (
            "   -"
            if np.isnan(correct_cos)
            else f"{correct_cos:.4f}"
        )
        margin_str = (
            "   -"
            if np.isnan(margin)
            else f"{margin:.4f}"
        )

        print(
            f"{row['method']:<30}"
            f"{row['top1']:>10.4f}"
            f"{row['top5']:>10.4f}"
            f"{row['classes_used']:>10d}"
            f"{row['effective_classes']:>12.2f}"
            f"{cos_str:>14}"
            f"{margin_str:>12}"
        )

    print()
    print("Train reference:")
    print(
        f"  learned classifier     Top-1 = "
        f"{train_learned_metrics['top1']:.4f}"
    )
    print(
        f"  train semantic proto   Top-1 = "
        f"{train_semantic_metrics['top1']:.4f}"
    )
    print(
        f"  CLIP text proto        Top-1 = "
        f"{train_text_metrics['top1']:.4f}"
    )

    print()
    print("Train semantic prototype <-> CLIP text prototype:")
    print(
        f"  retrieval Top-1        = "
        f"{proto_alignment_summary['semantic_proto_to_text_top1']:.4f}"
    )
    print(
        f"  retrieval Top-5        = "
        f"{proto_alignment_summary['semantic_proto_to_text_top5']:.4f}"
    )
    print(
        f"  mean diagonal cosine   = "
        f"{proto_alignment_summary['mean_same_class_cosine']:.4f}"
    )
    print(
        f"  median diagonal rank   = "
        f"{proto_alignment_summary['median_same_class_rank']:.1f}"
    )

    print()
    print("Automatic diagnosis:")
    print(
        f"  {diagnosis['label']}"
    )
    for evidence in diagnosis[
        "evidence"
    ]:
        print(
            f"  - {evidence}"
        )

    print("=" * 96)
    print(
        f"Saved: {out_dir}"
    )
    print(
        f"Main summary: {out_dir / 'summary.json'}"
    )
    print(
        f"Comparison  : {out_dir / 'method_comparison.csv'}"
    )
    print(
        f"Per-sample  : {out_dir / 'val_predictions_3way.csv'}"
    )


if __name__ == "__main__":
    main()
