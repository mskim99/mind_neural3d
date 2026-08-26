#!/usr/bin/env python3
"""
Validation confusion-matrix audit for the 72-way EEG semantic classifier.

Purpose
-------
Distinguish between:

A) hidden class permutation / semantic-axis mismatch
   - each GT class maps consistently to one OTHER predicted class
   - row-wise confusion is sharp
   - row argmax predictions are often unique across GT classes
   - Hungarian-permuted accuracy can be high even when raw accuracy is low

B) severe overfitting / class collapse
   - predictions are diffuse or collapse to a small subset of classes
   - row-wise entropy is high or many GT classes share the same predicted class
   - Hungarian-permuted accuracy remains low

This script intentionally loads ONLY:
    - EEG_Detangling_Disentanglement_Model
    - semantic_cls_head

from the checkpoint. It does NOT instantiate Zero123++ / VAE / UNet, so it is
much lighter than full inference.

Expected checkpoint:
    stage2_semantic_cls/checkpoints/model_XXXXXX.pt

Expected dataset:
    src.data.egg_dataset_ext_el.AllDataFeatureTwoEEG

Outputs
-------
<out_dir>/
  confusion_counts.csv
  confusion_row_normalized.csv
  per_class_metrics.csv
  top_confusions.csv
  sample_predictions.csv
  summary.json
  confusion_counts.png
  confusion_row_normalized.png
  prediction_histogram.png

Example
-------
CUDA_VISIBLE_DEVICES=1 python analyze_val_confusion.py \
  --ckpt ./stage2_semantic_cls/checkpoints/model_final.pt \
  --data_path /data/jionkim/neuro_3D \
  --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
  --sub_id sub01 \
  --out_dir ./val_confusion
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
from torch.utils.data import DataLoader

# Headless-safe plotting.
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.mvdiffusion_var_semantic_cls import EEG_Detangling_Disentanglement_Model
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze validation confusion matrix of the 72-way EEG semantic classifier."
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint created by train_neural3d_pp_semantic_cls.py",
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
    p.add_argument("--out_dir", default="./val_confusion")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_classes", type=int, default=72)
    p.add_argument(
        "--also_train",
        action="store_true",
        help="Also evaluate train split and save train_* outputs.",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# Model: semantic path only
# -------------------------------------------------------------------------

class SemanticClassifierOnly(nn.Module):
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


def _strip_module_prefix(state_dict):
    out = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        out[k] = v
    return out


def load_semantic_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _strip_module_prefix(state)

    wanted = {}
    for k, v in state.items():
        if k.startswith("fmri_encoder."):
            wanted[k] = v
        elif k.startswith("semantic_cls_head."):
            wanted[k] = v

    fmri_keys = [k for k in wanted if k.startswith("fmri_encoder.")]
    cls_keys = [k for k in wanted if k.startswith("semantic_cls_head.")]

    if not fmri_keys:
        raise RuntimeError(
            "Checkpoint has no `fmri_encoder.*` keys. "
            "This is not a compatible semantic-class-supervised checkpoint."
        )
    if not cls_keys:
        raise RuntimeError(
            "Checkpoint has no `semantic_cls_head.*` keys. "
            "This is not a compatible 72-way semantic-classifier checkpoint."
        )

    incompatible = model.load_state_dict(wanted, strict=False)

    # Missing keys outside these two submodules should not exist because this
    # lightweight model contains only those modules.
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("[checkpoint load warning]")
        print("  missing   :", incompatible.missing_keys[:30])
        print("  unexpected:", incompatible.unexpected_keys[:30])
        raise RuntimeError("Semantic-only checkpoint loading mismatch.")

    step = None
    if isinstance(ckpt, dict):
        step = ckpt.get("optimizer_step", ckpt.get("global_step", None))

    print(f"[checkpoint] loaded semantic encoder + classifier from {ckpt_path}")
    if step is not None:
        print(f"[checkpoint] optimizer_step={step}")


# -------------------------------------------------------------------------
# Dataset / labels
# -------------------------------------------------------------------------

def ensure_trailing_sep(path):
    path = str(Path(path).expanduser().resolve())
    if not path.endswith(os.sep):
        path += os.sep
    return path


def build_dataset(args, train=False):
    return AllDataFeatureTwoEEG(
        data_path=ensure_trailing_sep(args.data_path),
        sub_list=[args.sub_id],
        train=train,
        num_frames=6,
        rendered_view_path=str(Path(args.rendered_view_path).expanduser().resolve()),
        aug_data=False,
        strict_rendered_views=True,
    )


def category_name_from_dataset_name(name):
    """
    Example:
        13_can_08 -> can
        01_airplane_09 -> airplane

    The first 3 chars are the dataset class prefix.
    The final _XX is the object-instance suffix.
    """
    key = str(name)[3:]
    if "_" in key:
        return key.rsplit("_", 1)[0]
    return key


def build_class_names(ds):
    names = []
    prefixes = []
    for cls_idx in range(ds.cls_num):
        raw = str(ds.name_list[cls_idx, 0])
        names.append(category_name_from_dataset_name(raw))
        prefixes.append(raw[:3])
    return names, prefixes


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model, ds, args, split_name):
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device(args.device)
    class_names, class_prefixes = build_class_names(ds)

    confusion = np.zeros(
        (args.num_classes, args.num_classes),
        dtype=np.int64,
    )

    sample_rows = []

    total = 0
    correct1 = 0
    correct5 = 0

    pred_counter = Counter()

    for batch_idx, batch in enumerate(loader):
        eeg = batch["eeg_data"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        if "cls_index" not in batch:
            raise KeyError(
                "Dataset batch does not contain `cls_index`. "
                "Use the current egg_dataset_ext_el loader."
            )

        target = batch["cls_index"].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        logits, sem_feat, _, _ = model(eeg)
        probs = torch.softmax(logits, dim=-1)

        pred = logits.argmax(dim=-1)
        top5 = logits.topk(
            k=min(5, args.num_classes),
            dim=-1,
        ).indices

        correct1 += int((pred == target).sum().item())
        correct5 += int(
            (top5 == target.unsqueeze(1)).any(dim=1).sum().item()
        )
        total += int(target.numel())

        target_cpu = target.cpu().numpy()
        pred_cpu = pred.cpu().numpy()
        probs_cpu = probs.cpu().numpy()
        top5_cpu = top5.cpu().numpy()

        # Explicit sample metadata.
        names = batch.get("name", [""] * len(target_cpu))
        labels = batch.get("label", [""] * len(target_cpu))
        obj_indices = batch.get("obj_index", [None] * len(target_cpu))
        trial_indices = batch.get("trial_index", [None] * len(target_cpu))

        for i in range(len(target_cpu)):
            gt = int(target_cpu[i])
            pr = int(pred_cpu[i])

            confusion[gt, pr] += 1
            pred_counter[pr] += 1

            if torch.is_tensor(obj_indices):
                obj_idx = int(obj_indices[i].item())
            else:
                obj_idx = obj_indices[i]

            if torch.is_tensor(trial_indices):
                trial_idx = int(trial_indices[i].item())
            else:
                trial_idx = trial_indices[i]

            name = str(names[i])
            label = str(labels[i]) if len(labels) > i else name[3:]

            sample_rows.append(
                {
                    "split": split_name,
                    "sample_index": len(sample_rows),
                    "dataset_name": name,
                    "label": label,
                    "obj_index": obj_idx,
                    "trial_index": trial_idx,
                    "gt_cls": gt,
                    "gt_prefix": class_prefixes[gt],
                    "gt_category": class_names[gt],
                    "pred_cls": pr,
                    "pred_prefix": class_prefixes[pr],
                    "pred_category": class_names[pr],
                    "correct": int(gt == pr),
                    "pred_confidence": float(probs_cpu[i, pr]),
                    "gt_probability": float(probs_cpu[i, gt]),
                    "top5_cls": ",".join(map(str, top5_cpu[i].tolist())),
                    "top5_category": ",".join(
                        class_names[j] for j in top5_cpu[i].tolist()
                    ),
                }
            )

    accuracy1 = correct1 / max(total, 1)
    accuracy5 = correct5 / max(total, 1)

    return {
        "confusion": confusion,
        "sample_df": pd.DataFrame(sample_rows),
        "class_names": class_names,
        "class_prefixes": class_prefixes,
        "pred_counter": pred_counter,
        "accuracy1": accuracy1,
        "accuracy5": accuracy5,
        "total": total,
    }


# -------------------------------------------------------------------------
# Confusion analysis
# -------------------------------------------------------------------------

def entropy_from_counts(row):
    row = np.asarray(row, dtype=np.float64)
    total = row.sum()
    if total <= 0:
        return 0.0
    p = row / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def analyze_confusion(result):
    C = result["confusion"]
    class_names = result["class_names"]
    class_prefixes = result["class_prefixes"]

    row_sum = C.sum(axis=1, keepdims=True)
    row_norm = np.divide(
        C,
        row_sum,
        out=np.zeros_like(C, dtype=np.float64),
        where=row_sum > 0,
    )

    rows = []
    top_confusions = []

    row_argmax = C.argmax(axis=1)
    unique_row_argmax = len(set(row_argmax.tolist()))

    for gt in range(C.shape[0]):
        total = int(C[gt].sum())
        correct = int(C[gt, gt])

        order = np.argsort(-C[gt])
        best = int(order[0])
        second = int(order[1]) if len(order) > 1 else best

        row_entropy = entropy_from_counts(C[gt])

        rows.append(
            {
                "gt_cls": gt,
                "gt_prefix": class_prefixes[gt],
                "gt_category": class_names[gt],
                "samples": total,
                "correct": correct,
                "accuracy": correct / max(total, 1),
                "row_argmax_cls": best,
                "row_argmax_category": class_names[best],
                "row_argmax_count": int(C[gt, best]),
                "row_argmax_fraction": float(
                    C[gt, best] / max(total, 1)
                ),
                "second_cls": second,
                "second_category": class_names[second],
                "second_count": int(C[gt, second]),
                "row_entropy_bits": row_entropy,
                "row_argmax_is_gt": bool(best == gt),
            }
        )

        for pred in order:
            pred = int(pred)
            if pred == gt or C[gt, pred] == 0:
                continue
            top_confusions.append(
                {
                    "gt_cls": gt,
                    "gt_category": class_names[gt],
                    "pred_cls": pred,
                    "pred_category": class_names[pred],
                    "count": int(C[gt, pred]),
                    "fraction_of_gt": float(
                        C[gt, pred] / max(total, 1)
                    ),
                }
            )

    per_class_df = pd.DataFrame(rows)

    top_confusions_df = pd.DataFrame(top_confusions)
    if len(top_confusions_df):
        top_confusions_df = top_confusions_df.sort_values(
            ["count", "fraction_of_gt"],
            ascending=False,
        ).reset_index(drop=True)

    # Prediction concentration.
    pred_counts = C.sum(axis=0)
    pred_probs = pred_counts / max(pred_counts.sum(), 1)
    nonzero = pred_probs[pred_probs > 0]
    pred_entropy = float(
        -(nonzero * np.log2(nonzero)).sum()
    )
    effective_pred_classes = float(
        2.0 ** pred_entropy
    )

    # How many GT classes map to each row-argmax class?
    argmax_counts = Counter(row_argmax.tolist())
    most_common_argmax_cls, most_common_argmax_n = (
        argmax_counts.most_common(1)[0]
    )

    # Hungarian permutation diagnostic.
    hungarian = {
        "available": False,
        "permuted_accuracy": None,
        "assignment_unique_classes": None,
    }
    try:
        from scipy.optimize import linear_sum_assignment

        # Maximize counts -> minimize negative counts.
        gt_idx, pred_idx = linear_sum_assignment(-C)
        matched = int(C[gt_idx, pred_idx].sum())
        total = int(C.sum())
        permuted_accuracy = matched / max(total, 1)

        hungarian = {
            "available": True,
            "permuted_accuracy": float(permuted_accuracy),
            "assignment_unique_classes": int(len(pred_idx)),
            "assignment": [
                {
                    "gt_cls": int(g),
                    "gt_category": class_names[int(g)],
                    "mapped_pred_cls": int(p),
                    "mapped_pred_category": class_names[int(p)],
                    "count": int(C[int(g), int(p)]),
                }
                for g, p in zip(gt_idx, pred_idx)
            ],
        }
    except Exception as e:
        hungarian["error"] = repr(e)

    summary = {
        "raw_top1_accuracy": float(result["accuracy1"]),
        "raw_top5_accuracy": float(result["accuracy5"]),
        "num_samples": int(result["total"]),
        "num_classes": int(C.shape[0]),
        "num_predicted_classes_used": int(
            np.count_nonzero(pred_counts)
        ),
        "prediction_entropy_bits": pred_entropy,
        "effective_num_predicted_classes": effective_pred_classes,
        "row_argmax_unique_pred_classes": int(unique_row_argmax),
        "row_argmax_correct_classes": int(
            np.sum(row_argmax == np.arange(C.shape[0]))
        ),
        "most_common_row_argmax_cls": int(
            most_common_argmax_cls
        ),
        "most_common_row_argmax_category": class_names[
            int(most_common_argmax_cls)
        ],
        "num_gt_classes_mapping_to_most_common_pred": int(
            most_common_argmax_n
        ),
        "mean_row_argmax_fraction": float(
            per_class_df["row_argmax_fraction"].mean()
        ),
        "mean_row_entropy_bits": float(
            per_class_df["row_entropy_bits"].mean()
        ),
        "hungarian": hungarian,
    }

    return row_norm, per_class_df, top_confusions_df, summary


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def plot_confusion(
    matrix,
    class_names,
    out_path,
    title,
    normalized=False,
):
    n = len(class_names)

    fig = plt.figure(figsize=(18, 16))
    ax = fig.add_subplot(111)

    im = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title(title)

    # 72 labels are dense; use category names but small font.
    ticks = np.arange(n)
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

    # Diagonal guide.
    ax.plot(
        [-0.5, n - 0.5],
        [-0.5, n - 0.5],
        linewidth=0.7,
    )

    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_prediction_histogram(
    confusion,
    class_names,
    out_path,
    title,
):
    counts = confusion.sum(axis=0)
    x = np.arange(len(class_names))

    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111)
    ax.bar(x, counts)
    ax.set_xticks(x)
    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=6,
    )
    ax.set_ylabel("Number of predictions")
    ax.set_xlabel("Predicted class")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(
        out_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# -------------------------------------------------------------------------
# Save one split
# -------------------------------------------------------------------------

def save_split_outputs(result, out_dir, prefix="val"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    C = result["confusion"]
    class_names = result["class_names"]

    row_norm, per_class_df, top_confusions_df, summary = (
        analyze_confusion(result)
    )

    labels = [
        f"{i:02d}:{name}"
        for i, name in enumerate(class_names)
    ]

    pd.DataFrame(
        C,
        index=labels,
        columns=labels,
    ).to_csv(
        out_dir / f"{prefix}_confusion_counts.csv"
    )

    pd.DataFrame(
        row_norm,
        index=labels,
        columns=labels,
    ).to_csv(
        out_dir / f"{prefix}_confusion_row_normalized.csv"
    )

    per_class_df.to_csv(
        out_dir / f"{prefix}_per_class_metrics.csv",
        index=False,
    )

    top_confusions_df.to_csv(
        out_dir / f"{prefix}_top_confusions.csv",
        index=False,
    )

    result["sample_df"].to_csv(
        out_dir / f"{prefix}_sample_predictions.csv",
        index=False,
    )

    with open(
        out_dir / f"{prefix}_summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    plot_confusion(
        C,
        class_names,
        out_dir / f"{prefix}_confusion_counts.png",
        title=f"{prefix.upper()} confusion matrix (counts)",
        normalized=False,
    )

    plot_confusion(
        row_norm,
        class_names,
        out_dir / f"{prefix}_confusion_row_normalized.png",
        title=f"{prefix.upper()} confusion matrix (row-normalized)",
        normalized=True,
    )

    plot_prediction_histogram(
        C,
        class_names,
        out_dir / f"{prefix}_prediction_histogram.png",
        title=f"{prefix.upper()} predicted-class distribution",
    )

    return summary


# -------------------------------------------------------------------------
# Console diagnosis
# -------------------------------------------------------------------------

def print_diagnosis(summary, prefix="VAL"):
    print()
    print("=" * 80)
    print(f"{prefix} CONFUSION-MATRIX DIAGNOSIS")
    print("=" * 80)
    print(
        f"Top-1 accuracy                 : "
        f"{summary['raw_top1_accuracy']:.4f}"
    )
    print(
        f"Top-5 accuracy                 : "
        f"{summary['raw_top5_accuracy']:.4f}"
    )
    print(
        f"Predicted classes actually used: "
        f"{summary['num_predicted_classes_used']}/"
        f"{summary['num_classes']}"
    )
    print(
        f"Effective predicted classes    : "
        f"{summary['effective_num_predicted_classes']:.2f}"
    )
    print(
        f"Unique row-argmax predictions  : "
        f"{summary['row_argmax_unique_pred_classes']}/"
        f"{summary['num_classes']}"
    )
    print(
        f"GT classes with correct row max: "
        f"{summary['row_argmax_correct_classes']}/"
        f"{summary['num_classes']}"
    )
    print(
        f"Mean row-argmax fraction       : "
        f"{summary['mean_row_argmax_fraction']:.3f}"
    )
    print(
        f"Mean row entropy               : "
        f"{summary['mean_row_entropy_bits']:.3f} bits"
    )

    h = summary["hungarian"]
    if h.get("available", False):
        print(
            f"Hungarian-permuted accuracy    : "
            f"{h['permuted_accuracy']:.4f}"
        )

    print()
    print("Interpretation guide:")

    raw = summary["raw_top1_accuracy"]
    perm = h.get("permuted_accuracy", None)
    unique = summary["row_argmax_unique_pred_classes"]
    mean_peak = summary["mean_row_argmax_fraction"]
    eff = summary["effective_num_predicted_classes"]

    if (
        perm is not None
        and raw < 0.15
        and perm > 0.60
        and unique > summary["num_classes"] * 0.6
        and mean_peak > 0.6
    ):
        print(
            "  -> Strong hidden-permutation signature:\n"
            "     raw accuracy is low, but a one-to-one class permutation "
            "recovers high accuracy."
        )
    elif (
        eff < summary["num_classes"] * 0.25
        or unique < summary["num_classes"] * 0.25
    ):
        print(
            "  -> Strong class-collapse signature:\n"
            "     many GT classes are being mapped into a small subset "
            "of predicted classes."
        )
    elif raw < 0.15 and mean_peak < 0.5:
        print(
            "  -> Diffuse generalization failure:\n"
            "     predictions are wrong and not consistently mapped to "
            "one alternate class."
        )
    elif raw < 0.15:
        print(
            "  -> Low validation accuracy with structured confusions.\n"
            "     Inspect the row-normalized matrix and Hungarian mapping."
        )
    else:
        print(
            "  -> Validation classification is not catastrophically failed."
        )

    print("=" * 80)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    # Model.
    model = SemanticClassifierOnly(
        num_classes=args.num_classes
    )
    load_semantic_weights(
        model,
        args.ckpt,
    )
    model = model.to(device).eval()

    # Validation/test split.
    val_ds = build_dataset(
        args,
        train=False,
    )
    print(
        f"[val] dataset length={len(val_ds)}, "
        f"name_list={val_ds.name_list.shape}, "
        f"eeg_data={val_ds.eeg_data.shape}"
    )

    val_result = evaluate_split(
        model,
        val_ds,
        args,
        split_name="val",
    )
    val_summary = save_split_outputs(
        val_result,
        out_dir,
        prefix="val",
    )
    print_diagnosis(
        val_summary,
        prefix="VAL",
    )

    if args.also_train:
        train_ds = build_dataset(
            args,
            train=True,
        )
        print(
            f"[train] dataset length={len(train_ds)}, "
            f"name_list={train_ds.name_list.shape}, "
            f"eeg_data={train_ds.eeg_data.shape}"
        )

        train_result = evaluate_split(
            model,
            train_ds,
            args,
            split_name="train",
        )
        train_summary = save_split_outputs(
            train_result,
            out_dir,
            prefix="train",
        )
        print_diagnosis(
            train_summary,
            prefix="TRAIN",
        )

        comparison = {
            "train_top1": train_summary["raw_top1_accuracy"],
            "val_top1": val_summary["raw_top1_accuracy"],
            "generalization_gap": (
                train_summary["raw_top1_accuracy"]
                - val_summary["raw_top1_accuracy"]
            ),
        }
        with open(
            out_dir / "train_val_comparison.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                comparison,
                f,
                indent=2,
            )

    print()
    print("Saved to:", out_dir)
    print("Key files:")
    print("  val_confusion_row_normalized.png")
    print("  val_per_class_metrics.csv")
    print("  val_top_confusions.csv")
    print("  val_sample_predictions.csv")
    print("  val_summary.json")


if __name__ == "__main__":
    main()
