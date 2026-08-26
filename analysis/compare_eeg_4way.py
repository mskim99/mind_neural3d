#!/usr/bin/env python3
"""
Four-way EEG condition audit for the current semantic encoder/checkpoint.

Compares the SAME trained checkpoint under four input conditions:

    1) train_individual
       seen train objects (00~07), each raw EEG trial separately

    2) train_averaged
       seen train objects (00~07), raw EEG trials averaged BEFORE the encoder

    3) test_individual
       unseen test objects (08~09), each raw EEG trial separately

    4) test_averaged
       unseen test objects (08~09), raw EEG trials averaged BEFORE the encoder

This directly separates two possible failure sources:

    - trial-averaging / preprocessing distribution shift
    - unseen-object generalization failure

For each condition the script reports THREE semantic decoders:

    A. learned_classifier
       semantic -> trained semantic_cls_head

    B. train_semantic_prototype
       semantic -> cosine to 72 semantic prototypes built ONLY from
                   train_individual features

    C. clip_text_prototype
       semantic -> cosine to 72 CLIP text prototypes built ONLY from
                   train data's precomputed txt_fea

Important design choice
-----------------------
All prototype banks are fixed from TRAIN INDIVIDUAL samples.
Only the EEG input condition changes. This makes the four-way comparison fair.

The averaging is performed on raw EEG:
    [trial, 64, 600] -> mean over trial -> [64, 600]
BEFORE the nonlinear EEG encoder. It therefore matches the intended
`test_mean=True` behavior rather than averaging semantic features afterward.

Expected raw shapes for the current Neuro-3D dataset:
    train raw: [1,72,8,2,64,600]
    test raw : [1,72,2,4,64,600]

Expected condition sample counts:
    train_individual : 72*8*2 = 1152
    train_averaged   : 72*8   = 576
    test_individual  : 72*2*4 = 576
    test_averaged    : 72*2   = 144

Outputs
-------
<out_dir>/
  summary.json
  condition_comparison.csv
  predictions_all_conditions.csv
  per_class_all_conditions.csv

  train_individual_confusion.png
  train_averaged_confusion.png
  test_individual_confusion.png
  test_averaged_confusion.png

  learned_top1_4way.png
  prototype_top1_4way.png
  correct_cosine_margin_4way.png

  train_semantic_prototypes.npy
  clip_text_prototypes.npy

Recommended run
---------------
CUDA_VISIBLE_DEVICES=1 python compare_eeg_4way.py \
  --ckpt ./stage2_semantic_cls/checkpoints/model_final.pt \
  --data_path /data/jionkim/neuro_3D \
  --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
  --sub_id sub01 \
  --out_dir ./eeg_4way \
  --batch_size 32
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
            "Compare train individual / train averaged / "
            "test individual / test averaged EEG conditions."
        )
    )
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--out_dir", default="./eeg_4way")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_classes", type=int, default=72)
    p.add_argument(
        "--prototype_mode",
        choices=["mean_then_normalize", "normalize_then_mean"],
        default="mean_then_normalize",
    )
    p.add_argument(
        "--strict_rendered_views",
        action="store_true",
        help=(
            "Validate all six rendered PNGs while constructing the source "
            "dataset. Not needed for this EEG-only diagnostic."
        ),
    )
    return p.parse_args()


# ============================================================================
# Lightweight semantic model
# ============================================================================

class SemanticClassifierOnly(nn.Module):
    def __init__(self, num_classes=72):
        super().__init__()

        self.fmri_encoder = EEG_Detangling_Disentanglement_Model(
            num_electrodes=64,
            seq_len=600,
            embed_dim=1024,
        )

        # Must match current semantic-class checkpoint.
        self.semantic_cls_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, eeg):
        sem, var, bias = self.fmri_encoder(eeg)
        logits = self.semantic_cls_head(sem.float())
        return logits, sem, var, bias


def strip_module_prefix(state):
    out = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_checkpoint(model, ckpt_path):
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = strip_module_prefix(state)

    wanted = {
        k: v
        for k, v in state.items()
        if k.startswith("fmri_encoder.")
        or k.startswith("semantic_cls_head.")
    }

    if not any(k.startswith("fmri_encoder.") for k in wanted):
        raise RuntimeError("Checkpoint contains no fmri_encoder.* weights.")
    if not any(k.startswith("semantic_cls_head.") for k in wanted):
        raise RuntimeError("Checkpoint contains no semantic_cls_head.* weights.")

    incompatible = model.load_state_dict(wanted, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[checkpoint mismatch]")
        print("  missing   :", incompatible.missing_keys[:30])
        print("  unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError("Semantic-only checkpoint architecture mismatch.")

    meta = {}
    if isinstance(ckpt, dict):
        meta["optimizer_step"] = ckpt.get(
            "optimizer_step",
            ckpt.get("global_step", None),
        )
        meta["stage"] = ckpt.get("stage", None)

    print(f"[checkpoint] loaded {ckpt_path}")
    print(f"[checkpoint] meta={meta}")
    return meta


# ============================================================================
# Source dataset
# ============================================================================

def ensure_trailing_sep(path):
    path = str(Path(path).expanduser().resolve())
    if not path.endswith(os.sep):
        path += os.sep
    return path


def build_raw_source_dataset(args, train):
    """
    IMPORTANT:
    test_mean=False ensures the TEST source keeps its original trial axis.

    We then create both individual and averaged conditions ourselves from
    this same raw source, avoiding any ambiguity between dataset instances.
    """
    return AllDataFeatureTwoEEG(
        data_path=ensure_trailing_sep(args.data_path),
        sub_list=[args.sub_id],
        train=train,
        test_mean=False,
        num_frames=6,
        rendered_view_path=str(
            Path(args.rendered_view_path).expanduser().resolve()
        ),
        aug_data=False,
        strict_rendered_views=args.strict_rendered_views,
    )


def category_from_name(name):
    # 13_can_08 -> can
    key = str(name)[3:]
    return key.rsplit("_", 1)[0] if "_" in key else key


def class_names_from_source(source):
    return [
        category_from_name(source.name_list[c, 0])
        for c in range(source.cls_num)
    ]


# ============================================================================
# Deterministic raw-EEG condition wrapper
# ============================================================================

class EEGConditionDataset(Dataset):
    """
    Creates a deterministic condition directly from source.eeg_data.

    source.eeg_data:
        [S, C, O, R, 64, 600]

    mode="individual":
        one item per S,C,O,R

    mode="averaged":
        one item per S,C,O; average raw EEG over R BEFORE encoder
    """

    def __init__(self, source, split, mode):
        super().__init__()

        if mode not in {"individual", "averaged"}:
            raise ValueError(mode)

        self.source = source
        self.split = split
        self.mode = mode

        shape = tuple(int(x) for x in source.eeg_data.shape)
        if len(shape) != 6:
            raise RuntimeError(
                f"Expected raw EEG [S,C,O,R,64,600], got {shape}"
            )

        self.S, self.C, self.O, self.R, self.E, self.T = shape

        if self.C != 72:
            raise RuntimeError(f"Expected 72 classes, got {self.C}")
        if self.E != 64 or self.T != 600:
            raise RuntimeError(
                f"Expected EEG [64,600], got [{self.E},{self.T}]"
            )

        if source.name_list.shape != (self.C, self.O):
            raise RuntimeError(
                "name_list and EEG class/object axes disagree: "
                f"name_list={source.name_list.shape}, eeg={shape}"
            )

        self.index = []
        for s in range(self.S):
            for c in range(self.C):
                for o in range(self.O):
                    if mode == "individual":
                        for r in range(self.R):
                            self.index.append((s, c, o, r))
                    else:
                        self.index.append((s, c, o, None))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s, c, o, r = self.index[idx]

        if self.mode == "individual":
            eeg = np.asarray(
                self.source.eeg_data[s, c, o, r],
                dtype=np.float32,
            )
            trial_index = int(r)
        else:
            # Raw trial averaging BEFORE nonlinear encoder.
            eeg = np.asarray(
                self.source.eeg_data[s, c, o, :],
                dtype=np.float32,
            ).mean(axis=0, dtype=np.float32)
            trial_index = -1

        name = str(self.source.name_list[c, o])
        key = name[3:]

        # Use the same precomputed CLIP text feature used by the source dataset.
        txt = self.source.clip_features[key]["text"]
        if not torch.is_tensor(txt):
            txt = torch.as_tensor(txt)

        return {
            "eeg_data": torch.from_numpy(np.ascontiguousarray(eeg)).float(),
            "cls_index": torch.tensor(c, dtype=torch.long),
            "subject_index": torch.tensor(s, dtype=torch.long),
            "obj_index": torch.tensor(o, dtype=torch.long),
            "trial_index": torch.tensor(trial_index, dtype=torch.long),
            "name": name,
            "label": key,
            "txt_fea": txt.float(),
            "condition": f"{self.split}_{self.mode}",
        }


# ============================================================================
# Sanity checks
# ============================================================================

def audit_expected_shapes(train_source, test_source):
    train_shape = tuple(int(x) for x in train_source.eeg_data.shape)
    test_shape = tuple(int(x) for x in test_source.eeg_data.shape)

    print(f"[source train raw] eeg={train_shape}")
    print(f"[source test  raw] eeg={test_shape}")

    # We do not hard-fail on exact R values because dataset variants may differ,
    # but report strongly because these are expected for the current project.
    if train_shape[:4] != (1, 72, 8, 2):
        print(
            "[WARNING] Expected current train leading shape "
            "(1,72,8,2), got",
            train_shape[:4],
        )

    if test_shape[:4] != (1, 72, 2, 4):
        print(
            "[WARNING] Expected current test leading shape "
            "(1,72,2,4), got",
            test_shape[:4],
        )


def print_condition_sizes(conditions):
    print("\n[condition sizes]")
    for name, ds in conditions.items():
        print(
            f"  {name:<20} n={len(ds):4d} | "
            f"S={ds.S}, C={ds.C}, O={ds.O}, raw_trials={ds.R}"
        )


# ============================================================================
# Feature extraction
# ============================================================================

def squeeze_feature(x, batch_size):
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.float()

    if x.shape[0] != batch_size:
        raise RuntimeError(
            f"Feature batch mismatch: shape={tuple(x.shape)}, B={batch_size}"
        )

    while x.ndim > 2:
        singleton = next(
            (d for d in range(1, x.ndim) if x.shape[d] == 1),
            None,
        )
        if singleton is None:
            x = x.reshape(x.shape[0], -1)
            break
        x = x.squeeze(singleton)

    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


@torch.no_grad()
def extract_condition(model, ds, args, condition_name):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)

    sem_all = []
    text_all = []
    logits_all = []
    target_all = []

    names = []
    labels = []
    obj_idx = []
    trial_idx = []

    for batch in loader:
        eeg = batch["eeg_data"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )
        target = batch["cls_index"].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        logits, sem, _, _ = model(eeg)

        sem = squeeze_feature(sem, eeg.shape[0])
        txt = squeeze_feature(batch["txt_fea"], eeg.shape[0])

        if sem.shape[-1] != txt.shape[-1]:
            raise RuntimeError(
                "Semantic and text dimensions differ: "
                f"semantic={tuple(sem.shape)}, text={tuple(txt.shape)}"
            )

        sem_all.append(sem.cpu())
        text_all.append(txt.cpu())
        logits_all.append(logits.float().cpu())
        target_all.append(target.cpu())

        names.extend([str(x) for x in batch["name"]])
        labels.extend([str(x) for x in batch["label"]])
        obj_idx.extend(batch["obj_index"].cpu().numpy().astype(int).tolist())
        trial_idx.extend(
            batch["trial_index"].cpu().numpy().astype(int).tolist()
        )

    result = {
        "semantic": torch.cat(sem_all, dim=0),
        "text": torch.cat(text_all, dim=0),
        "logits": torch.cat(logits_all, dim=0),
        "targets": torch.cat(target_all, dim=0).long(),
        "names": names,
        "labels": labels,
        "obj_index": obj_idx,
        "trial_index": trial_idx,
    }

    print(
        f"[extract {condition_name:<20}] "
        f"semantic={tuple(result['semantic'].shape)}, "
        f"logits={tuple(result['logits'].shape)}"
    )

    return result


# ============================================================================
# Prototype banks
# ============================================================================

def build_prototypes(features, targets, num_classes, mode):
    features = features.float()
    targets = targets.long()

    protos = []
    counts = []

    for c in range(num_classes):
        x = features[targets == c]
        if len(x) == 0:
            raise RuntimeError(f"No samples for train class {c}")

        if mode == "normalize_then_mean":
            x = F.normalize(x, dim=-1)

        p = x.mean(dim=0)
        p = F.normalize(p, dim=-1)

        protos.append(p)
        counts.append(len(x))

    return torch.stack(protos), np.asarray(counts, dtype=np.int64)


def cosine_logits(features, prototypes):
    return (
        F.normalize(features.float(), dim=-1)
        @ F.normalize(prototypes.float(), dim=-1).T
    )


# ============================================================================
# Metrics
# ============================================================================

def topk_acc(logits, targets, k):
    k = min(k, logits.shape[1])
    idx = logits.topk(k=k, dim=1).indices
    return float(
        (idx == targets[:, None])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )


def concentration_metrics(pred, num_classes):
    counts = np.bincount(
        pred.cpu().numpy(),
        minlength=num_classes,
    ).astype(np.float64)

    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]

    entropy = float(-(p * np.log2(p)).sum())
    return {
        "classes_used": int(np.count_nonzero(counts)),
        "prediction_entropy_bits": entropy,
        "effective_classes": float(2.0 ** entropy),
    }


def classification_metrics(logits, targets, num_classes):
    pred = logits.argmax(dim=1)
    conc = concentration_metrics(pred, num_classes)

    return {
        "top1": topk_acc(logits, targets, 1),
        "top5": topk_acc(logits, targets, 5),
        "pred": pred,
        **conc,
    }


def prototype_margin_metrics(logits, targets):
    n = logits.shape[0]
    rows = torch.arange(n)

    correct = logits[rows, targets]

    wrong = logits.clone()
    wrong[rows, targets] = -float("inf")
    best_wrong, best_wrong_cls = wrong.max(dim=1)

    margin = correct - best_wrong

    return {
        "correct_cosine": correct,
        "best_wrong_cosine": best_wrong,
        "best_wrong_cls": best_wrong_cls,
        "margin": margin,
        "mean_correct_cosine": float(correct.mean().item()),
        "mean_best_wrong_cosine": float(best_wrong.mean().item()),
        "mean_margin": float(margin.mean().item()),
        "positive_margin_rate": float((margin > 0).float().mean().item()),
    }


def confusion_matrix(pred, targets, num_classes):
    C = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pr in zip(
        targets.cpu().numpy(),
        pred.cpu().numpy(),
    ):
        C[int(gt), int(pr)] += 1
    return C


# ============================================================================
# Distribution / feature-shift diagnostics
# ============================================================================

def feature_shift_against_train_individual(
    train_ind_sem,
    condition_sem,
):
    """
    Coarse global distribution-shift indicators.
    These are descriptive, not inferential statistics.
    """
    a = train_ind_sem.float()
    b = condition_sem.float()

    a_centroid = F.normalize(a.mean(dim=0), dim=0)
    b_centroid = F.normalize(b.mean(dim=0), dim=0)

    centroid_cos = float(
        torch.dot(a_centroid, b_centroid).item()
    )

    # Feature norm statistics can reveal averaging-induced scale shifts.
    a_norm = a.norm(dim=-1)
    b_norm = b.norm(dim=-1)

    return {
        "centroid_cosine_to_train_individual": centroid_cos,
        "semantic_norm_mean": float(b_norm.mean().item()),
        "semantic_norm_std": float(b_norm.std().item()),
        "train_individual_norm_mean_reference": float(a_norm.mean().item()),
    }


# ============================================================================
# Tables
# ============================================================================

METHODS = [
    "learned_classifier",
    "train_semantic_prototype",
    "clip_text_prototype",
]


def evaluate_one_condition(
    extracted,
    semantic_proto,
    text_proto,
    num_classes,
):
    logits_map = {
        "learned_classifier": extracted["logits"],
        "train_semantic_prototype": cosine_logits(
            extracted["semantic"],
            semantic_proto,
        ),
        "clip_text_prototype": cosine_logits(
            extracted["semantic"],
            text_proto,
        ),
    }

    result = {}

    for method, logits in logits_map.items():
        cls = classification_metrics(
            logits,
            extracted["targets"],
            num_classes,
        )

        one = {
            "top1": cls["top1"],
            "top5": cls["top5"],
            "classes_used": cls["classes_used"],
            "prediction_entropy_bits": cls["prediction_entropy_bits"],
            "effective_classes": cls["effective_classes"],
            "pred": cls["pred"],
            "logits": logits,
        }

        if method != "learned_classifier":
            margin = prototype_margin_metrics(
                logits,
                extracted["targets"],
            )
            one.update({
                "mean_correct_cosine": margin["mean_correct_cosine"],
                "mean_best_wrong_cosine": margin["mean_best_wrong_cosine"],
                "mean_margin": margin["mean_margin"],
                "positive_margin_rate": margin["positive_margin_rate"],
                "_margin_raw": margin,
            })

        result[method] = one

    return result


def build_prediction_rows(
    condition_name,
    extracted,
    eval_result,
    class_names,
):
    targets = extracted["targets"]

    rows = []

    sem_margin = eval_result[
        "train_semantic_prototype"
    ]["_margin_raw"]
    text_margin = eval_result[
        "clip_text_prototype"
    ]["_margin_raw"]

    learned_prob = torch.softmax(
        eval_result["learned_classifier"]["logits"],
        dim=-1,
    )

    for i in range(len(targets)):
        gt = int(targets[i].item())

        lpred = int(
            eval_result["learned_classifier"]["pred"][i].item()
        )
        spred = int(
            eval_result["train_semantic_prototype"]["pred"][i].item()
        )
        tpred = int(
            eval_result["clip_text_prototype"]["pred"][i].item()
        )

        rows.append({
            "condition": condition_name,
            "sample_index": i,
            "dataset_name": extracted["names"][i],
            "label": extracted["labels"][i],
            "obj_index": extracted["obj_index"][i],
            "trial_index": extracted["trial_index"][i],
            "gt_cls": gt,
            "gt_category": class_names[gt],

            "learned_pred_cls": lpred,
            "learned_pred_category": class_names[lpred],
            "learned_correct": int(lpred == gt),
            "learned_confidence": float(
                learned_prob[i, lpred].item()
            ),
            "learned_gt_probability": float(
                learned_prob[i, gt].item()
            ),

            "semantic_proto_pred_cls": spred,
            "semantic_proto_pred_category": class_names[spred],
            "semantic_proto_correct": int(spred == gt),
            "semantic_correct_cosine": float(
                sem_margin["correct_cosine"][i].item()
            ),
            "semantic_margin": float(
                sem_margin["margin"][i].item()
            ),

            "clip_text_pred_cls": tpred,
            "clip_text_pred_category": class_names[tpred],
            "clip_text_correct": int(tpred == gt),
            "clip_text_correct_cosine": float(
                text_margin["correct_cosine"][i].item()
            ),
            "clip_text_margin": float(
                text_margin["margin"][i].item()
            ),
        })

    return rows


def build_per_class_table(pred_df):
    rows = []

    for condition in pred_df["condition"].unique():
        cd = pred_df[pred_df["condition"] == condition]

        for cls_idx in sorted(cd["gt_cls"].unique()):
            d = cd[cd["gt_cls"] == cls_idx]

            rows.append({
                "condition": condition,
                "cls_index": int(cls_idx),
                "category": d["gt_category"].iloc[0],
                "n": len(d),

                "learned_acc": float(d["learned_correct"].mean()),
                "semantic_proto_acc": float(
                    d["semantic_proto_correct"].mean()
                ),
                "clip_text_acc": float(
                    d["clip_text_correct"].mean()
                ),

                "semantic_correct_cosine": float(
                    d["semantic_correct_cosine"].mean()
                ),
                "semantic_margin": float(
                    d["semantic_margin"].mean()
                ),
                "clip_text_correct_cosine": float(
                    d["clip_text_correct_cosine"].mean()
                ),
                "clip_text_margin": float(
                    d["clip_text_margin"].mean()
                ),

                "learned_mode_pred": (
                    d["learned_pred_category"]
                    .value_counts()
                    .index[0]
                ),
                "semantic_mode_pred": (
                    d["semantic_proto_pred_category"]
                    .value_counts()
                    .index[0]
                ),
                "clip_text_mode_pred": (
                    d["clip_text_pred_category"]
                    .value_counts()
                    .index[0]
                ),
            })

    return pd.DataFrame(rows)


# ============================================================================
# Plotting
# ============================================================================

def plot_confusion(
    matrix,
    class_names,
    out_path,
    title,
):
    row_sum = matrix.sum(axis=1, keepdims=True)
    norm = np.divide(
        matrix,
        row_sum,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sum > 0,
    )

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        norm,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=5,
    )
    ax.set_yticklabels(
        class_names,
        fontsize=5,
    )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title)

    ax.plot(
        [-0.5, len(class_names) - 0.5],
        [-0.5, len(class_names) - 0.5],
        linewidth=0.7,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_top1_4way(
    comparison_df,
    method,
    out_path,
    title,
):
    d = comparison_df[
        comparison_df["method"] == method
    ].copy()

    order = [
        "train_individual",
        "train_averaged",
        "test_individual",
        "test_averaged",
    ]
    d["condition"] = pd.Categorical(
        d["condition"],
        categories=order,
        ordered=True,
    )
    d = d.sort_values("condition")

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111)

    x = np.arange(len(d))
    ax.bar(x, d["top1"].values)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Train\nindividual",
            "Train\naveraged",
            "Test\nindividual",
            "Test\naveraged",
        ]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(title)

    for i, v in enumerate(d["top1"].values):
        ax.text(
            i,
            min(v + 0.025, 0.98),
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_prototype_top1_4way(
    comparison_df,
    out_path,
):
    order = [
        "train_individual",
        "train_averaged",
        "test_individual",
        "test_averaged",
    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    x = np.arange(4)
    width = 0.35

    vals_sem = []
    vals_txt = []

    for cond in order:
        vals_sem.append(
            float(
                comparison_df[
                    (comparison_df["condition"] == cond)
                    & (
                        comparison_df["method"]
                        == "train_semantic_prototype"
                    )
                ]["top1"].iloc[0]
            )
        )
        vals_txt.append(
            float(
                comparison_df[
                    (comparison_df["condition"] == cond)
                    & (
                        comparison_df["method"]
                        == "clip_text_prototype"
                    )
                ]["top1"].iloc[0]
            )
        )

    ax.bar(
        x - width / 2,
        vals_sem,
        width,
        label="Train semantic prototype",
    )
    ax.bar(
        x + width / 2,
        vals_txt,
        width,
        label="CLIP text prototype",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Train\nindividual",
            "Train\naveraged",
            "Test\nindividual",
            "Test\naveraged",
        ]
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("Prototype decoding across four EEG conditions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_margin_4way(
    comparison_df,
    out_path,
):
    order = [
        "train_individual",
        "train_averaged",
        "test_individual",
        "test_averaged",
    ]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    x = np.arange(4)
    width = 0.35

    sem = []
    txt = []

    for cond in order:
        sem.append(
            float(
                comparison_df[
                    (comparison_df["condition"] == cond)
                    & (
                        comparison_df["method"]
                        == "train_semantic_prototype"
                    )
                ]["mean_margin"].iloc[0]
            )
        )
        txt.append(
            float(
                comparison_df[
                    (comparison_df["condition"] == cond)
                    & (
                        comparison_df["method"]
                        == "clip_text_prototype"
                    )
                ]["mean_margin"].iloc[0]
            )
        )

    ax.bar(
        x - width / 2,
        sem,
        width,
        label="Train semantic prototype",
    )
    ax.bar(
        x + width / 2,
        txt,
        width,
        label="CLIP text prototype",
    )

    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Train\nindividual",
            "Train\naveraged",
            "Test\nindividual",
            "Test\naveraged",
        ]
    )
    ax.set_ylabel("Mean correct-vs-best-wrong cosine margin")
    ax.set_title("Semantic margin across four EEG conditions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Automatic interpretation
# ============================================================================

def get_top1(comparison_df, condition, method="learned_classifier"):
    return float(
        comparison_df[
            (comparison_df["condition"] == condition)
            & (comparison_df["method"] == method)
        ]["top1"].iloc[0]
    )


def automatic_interpretation(comparison_df):
    """
    Use learned classifier first because it is the actual trained decoder,
    then cross-check prototype methods.
    """
    ti = get_top1(comparison_df, "train_individual")
    ta = get_top1(comparison_df, "train_averaged")
    vi = get_top1(comparison_df, "test_individual")
    va = get_top1(comparison_df, "test_averaged")

    train_avg_drop = ti - ta
    unseen_drop_individual = ti - vi
    test_avg_drop = vi - va

    # Same deltas for train semantic prototype.
    pti = get_top1(
        comparison_df,
        "train_individual",
        "train_semantic_prototype",
    )
    pta = get_top1(
        comparison_df,
        "train_averaged",
        "train_semantic_prototype",
    )
    pvi = get_top1(
        comparison_df,
        "test_individual",
        "train_semantic_prototype",
    )
    pva = get_top1(
        comparison_df,
        "test_averaged",
        "train_semantic_prototype",
    )

    evidence = []

    # Heuristic thresholds; intended for diagnosis, not statistical testing.
    averaging_large_train = train_avg_drop >= 0.25
    averaging_large_test = test_avg_drop >= 0.15
    unseen_large = unseen_drop_individual >= 0.40

    if averaging_large_train and not unseen_large:
        label = "trial_averaging_shift_dominant"
        evidence.append(
            "Seen-object train accuracy collapses after raw trial averaging."
        )
    elif unseen_large and not averaging_large_train:
        label = "unseen_object_generalization_dominant"
        evidence.append(
            "Train-averaged remains relatively stable, but test-individual "
            "collapses on unseen objects."
        )
    elif unseen_large and (averaging_large_train or averaging_large_test):
        label = "both_unseen_object_and_averaging_shift"
        evidence.append(
            "There is a large seen->unseen drop and an additional averaging drop."
        )
    elif vi > va + 0.15:
        label = "test_averaging_hurts_substantially"
        evidence.append(
            "Test-individual is substantially better than test-averaged."
        )
    else:
        label = "mixed_or_weakly_separated"
        evidence.append(
            "The simple four-way contrasts do not isolate one dominant factor."
        )

    evidence.extend([
        (
            f"Learned head Top-1: train individual={ti:.4f}, "
            f"train averaged={ta:.4f}, test individual={vi:.4f}, "
            f"test averaged={va:.4f}."
        ),
        (
            f"Train-prototype Top-1: train individual={pti:.4f}, "
            f"train averaged={pta:.4f}, test individual={pvi:.4f}, "
            f"test averaged={pva:.4f}."
        ),
        (
            f"Learned-head deltas: train averaging drop={train_avg_drop:.4f}, "
            f"seen->unseen individual drop={unseen_drop_individual:.4f}, "
            f"test averaging drop={test_avg_drop:.4f}."
        ),
    ])

    return {
        "label": label,
        "learned_head_deltas": {
            "train_averaging_drop": train_avg_drop,
            "seen_to_unseen_individual_drop": unseen_drop_individual,
            "test_averaging_drop": test_avg_drop,
        },
        "evidence": evidence,
        "note": (
            "Heuristic diagnostic only. Confirm with the three decoder methods "
            "and per-class margins before changing training."
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)

    # ----------------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------------
    model = SemanticClassifierOnly(
        num_classes=args.num_classes,
    )
    ckpt_meta = load_checkpoint(
        model,
        args.ckpt,
    )
    model = model.to(device).eval()

    # ----------------------------------------------------------------------
    # Raw train/test source datasets.
    # test_mean=False is essential here.
    # ----------------------------------------------------------------------
    print("\n[1/6] Loading raw train/test source datasets ...")

    train_source = build_raw_source_dataset(
        args,
        train=True,
    )
    test_source = build_raw_source_dataset(
        args,
        train=False,
    )

    audit_expected_shapes(
        train_source,
        test_source,
    )

    train_names = class_names_from_source(train_source)
    test_names = class_names_from_source(test_source)

    if train_names != test_names:
        raise RuntimeError(
            "Train/test class ordering differs. "
            "Run label-alignment audit first."
        )

    class_names = train_names

    # ----------------------------------------------------------------------
    # Four deterministic raw-EEG conditions.
    # ----------------------------------------------------------------------
    conditions = {
        "train_individual": EEGConditionDataset(
            train_source,
            "train",
            "individual",
        ),
        "train_averaged": EEGConditionDataset(
            train_source,
            "train",
            "averaged",
        ),
        "test_individual": EEGConditionDataset(
            test_source,
            "test",
            "individual",
        ),
        "test_averaged": EEGConditionDataset(
            test_source,
            "test",
            "averaged",
        ),
    }

    print_condition_sizes(conditions)

    # ----------------------------------------------------------------------
    # Extract all four conditions.
    # ----------------------------------------------------------------------
    print("\n[2/6] Extracting semantic features ...")

    extracted = {}
    for condition_name, ds in conditions.items():
        extracted[condition_name] = extract_condition(
            model,
            ds,
            args,
            condition_name,
        )

    # ----------------------------------------------------------------------
    # Fixed prototype banks from TRAIN INDIVIDUAL ONLY.
    # ----------------------------------------------------------------------
    print("\n[3/6] Building fixed train-individual prototype banks ...")

    ref = extracted["train_individual"]

    semantic_proto, semantic_counts = build_prototypes(
        ref["semantic"],
        ref["targets"],
        args.num_classes,
        args.prototype_mode,
    )
    text_proto, text_counts = build_prototypes(
        ref["text"],
        ref["targets"],
        args.num_classes,
        args.prototype_mode,
    )

    if not np.array_equal(semantic_counts, text_counts):
        raise RuntimeError("Semantic/text prototype counts differ.")

    print(
        "[prototype] train-individual samples/class min/max = "
        f"{semantic_counts.min()}/{semantic_counts.max()}"
    )

    np.save(
        out_dir / "train_semantic_prototypes.npy",
        semantic_proto.numpy(),
    )
    np.save(
        out_dir / "clip_text_prototypes.npy",
        text_proto.numpy(),
    )

    # ----------------------------------------------------------------------
    # Evaluate all conditions with all three decoders.
    # ----------------------------------------------------------------------
    print("\n[4/6] Evaluating four conditions x three decoders ...")

    all_eval = {}
    comparison_rows = []
    prediction_rows = []

    for condition_name in conditions:
        ex = extracted[condition_name]

        result = evaluate_one_condition(
            ex,
            semantic_proto,
            text_proto,
            args.num_classes,
        )
        all_eval[condition_name] = result

        shift = feature_shift_against_train_individual(
            extracted["train_individual"]["semantic"],
            ex["semantic"],
        )

        for method in METHODS:
            r = result[method]

            row = {
                "condition": condition_name,
                "method": method,
                "n": len(ex["targets"]),
                "top1": r["top1"],
                "top5": r["top5"],
                "classes_used": r["classes_used"],
                "prediction_entropy_bits": r[
                    "prediction_entropy_bits"
                ],
                "effective_classes": r["effective_classes"],
                **shift,
            }

            if method != "learned_classifier":
                row.update({
                    "mean_correct_cosine": r[
                        "mean_correct_cosine"
                    ],
                    "mean_best_wrong_cosine": r[
                        "mean_best_wrong_cosine"
                    ],
                    "mean_margin": r["mean_margin"],
                    "positive_margin_rate": r[
                        "positive_margin_rate"
                    ],
                })
            else:
                row.update({
                    "mean_correct_cosine": np.nan,
                    "mean_best_wrong_cosine": np.nan,
                    "mean_margin": np.nan,
                    "positive_margin_rate": np.nan,
                })

            comparison_rows.append(row)

        prediction_rows.extend(
            build_prediction_rows(
                condition_name,
                ex,
                result,
                class_names,
            )
        )

        # Learned-head confusion is the direct operational comparison.
        C = confusion_matrix(
            result["learned_classifier"]["pred"],
            ex["targets"],
            args.num_classes,
        )
        plot_confusion(
            C,
            class_names,
            out_dir / f"{condition_name}_confusion.png",
            title=(
                f"{condition_name}: learned classifier "
                "(row-normalized)"
            ),
        )

    comparison_df = pd.DataFrame(comparison_rows)
    pred_df = pd.DataFrame(prediction_rows)
    per_class_df = build_per_class_table(pred_df)

    comparison_df.to_csv(
        out_dir / "condition_comparison.csv",
        index=False,
    )
    pred_df.to_csv(
        out_dir / "predictions_all_conditions.csv",
        index=False,
    )
    per_class_df.to_csv(
        out_dir / "per_class_all_conditions.csv",
        index=False,
    )

    # ----------------------------------------------------------------------
    # Automatic diagnosis.
    # ----------------------------------------------------------------------
    print("\n[5/6] Computing contrasts / diagnosis ...")

    diagnosis = automatic_interpretation(
        comparison_df,
    )

    # Build compact JSON-safe nested metrics.
    condition_summary = {}

    for condition_name in conditions:
        condition_summary[condition_name] = {}

        for method in METHODS:
            r = all_eval[condition_name][method]

            entry = {
                "top1": r["top1"],
                "top5": r["top5"],
                "classes_used": r["classes_used"],
                "prediction_entropy_bits": r[
                    "prediction_entropy_bits"
                ],
                "effective_classes": r["effective_classes"],
            }

            if method != "learned_classifier":
                entry.update({
                    "mean_correct_cosine": r[
                        "mean_correct_cosine"
                    ],
                    "mean_best_wrong_cosine": r[
                        "mean_best_wrong_cosine"
                    ],
                    "mean_margin": r["mean_margin"],
                    "positive_margin_rate": r[
                        "positive_margin_rate"
                    ],
                })

            condition_summary[condition_name][method] = entry

        condition_summary[condition_name][
            "feature_shift"
        ] = feature_shift_against_train_individual(
            extracted["train_individual"]["semantic"],
            extracted[condition_name]["semantic"],
        )

    summary = {
        "checkpoint": str(
            Path(args.ckpt).expanduser().resolve()
        ),
        "checkpoint_meta": ckpt_meta,
        "subject": args.sub_id,
        "num_classes": args.num_classes,
        "source_shapes": {
            "train_raw": list(
                map(int, train_source.eeg_data.shape)
            ),
            "test_raw": list(
                map(int, test_source.eeg_data.shape)
            ),
        },
        "condition_sizes": {
            k: len(v)
            for k, v in conditions.items()
        },
        "prototype_reference": (
            "train_individual only"
        ),
        "conditions": condition_summary,
        "diagnosis": diagnosis,
    }

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
    # Plots.
    # ----------------------------------------------------------------------
    print("\n[6/6] Saving plots ...")

    plot_top1_4way(
        comparison_df,
        "learned_classifier",
        out_dir / "learned_top1_4way.png",
        "Learned semantic classifier: four-way EEG comparison",
    )

    plot_prototype_top1_4way(
        comparison_df,
        out_dir / "prototype_top1_4way.png",
    )

    plot_margin_4way(
        comparison_df,
        out_dir / "correct_cosine_margin_4way.png",
    )

    # ----------------------------------------------------------------------
    # Console report.
    # ----------------------------------------------------------------------
    print()
    print("=" * 110)
    print("FOUR-WAY EEG CONDITION DIAGNOSIS")
    print("=" * 110)
    print(
        f"{'Condition':<20}"
        f"{'Method':<28}"
        f"{'Top-1':>10}"
        f"{'Top-5':>10}"
        f"{'Used':>8}"
        f"{'Eff.cls':>10}"
        f"{'Margin':>12}"
    )
    print("-" * 110)

    order = [
        "train_individual",
        "train_averaged",
        "test_individual",
        "test_averaged",
    ]

    for condition in order:
        for method in METHODS:
            row = comparison_df[
                (comparison_df["condition"] == condition)
                & (comparison_df["method"] == method)
            ].iloc[0]

            margin = row["mean_margin"]
            margin_str = (
                "-"
                if pd.isna(margin)
                else f"{margin:.4f}"
            )

            print(
                f"{condition:<20}"
                f"{method:<28}"
                f"{row['top1']:>10.4f}"
                f"{row['top5']:>10.4f}"
                f"{int(row['classes_used']):>8d}"
                f"{row['effective_classes']:>10.2f}"
                f"{margin_str:>12}"
            )

        print("-" * 110)

    print()
    print("Automatic diagnosis:")
    print(f"  {diagnosis['label']}")
    for e in diagnosis["evidence"]:
        print(f"  - {e}")

    print()
    print("Interpretation:")
    print(
        "  * train_individual -> train_averaged drop "
        "= pure averaging sensitivity on SEEN objects."
    )
    print(
        "  * train_individual -> test_individual drop "
        "= unseen-object generalization gap WITHOUT averaging."
    )
    print(
        "  * test_individual -> test_averaged drop "
        "= additional averaging penalty on UNSEEN objects."
    )
    print(
        "  * Prototype methods cross-check whether the effect is "
        "already present in semantic feature geometry."
    )
    print("=" * 110)
    print(f"Saved: {out_dir}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(
        f"Table  : {out_dir / 'condition_comparison.csv'}"
    )


if __name__ == "__main__":
    main()
