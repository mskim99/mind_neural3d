#!/usr/bin/env python3
"""
Audit EEG tensor index <-> class/object label <-> rendered GT images.

This script is designed for the current Neuro-3D loader:
    src.data.egg_dataset_ext_el.AllDataFeatureTwoEEG

It answers three different questions separately:

A. INDEXING AUDIT (deterministic)
   Does dataset[idx] actually read:
       eeg_data[sub_index, cls_index, obj_index, trial_index]
   and attach the name/label expected from:
       name_list[cls_index, obj_index] ?

B. GT IMAGE AUDIT (deterministic)
   Does the same sample load the six PNGs belonging to that exact object label,
   and are dataset['rotation_images'] numerically identical to those files after
   the dataset's own resize/normalization?

C. CROSS-SPLIT EEG PROTOTYPE AUDIT (diagnostic, NOT a proof of semantics)
   Do train/test EEG class prototypes show diagonal similarity, i.e. does
   test cls_index=c tend to be closest to train cls_index=c?
   Low similarity alone does NOT prove misalignment because EEG category
   separability may be weak. A strong systematic off-diagonal permutation,
   however, is suspicious.

Outputs:
    <out_dir>/
      audit_report.json
      audit_samples.csv
      mapping_train_checked.csv
      mapping_test_checked.csv
      eeg_prototype_similarity.csv
      eeg_similarity_matrix.npy
      visuals/
        train/*.png
        test/*.png

Recommended:
    python audit_eeg_label_gt_alignment.py \
        --data_path /data/jionkim/neuro_3D \
        --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
        --sub_id 0001 \
        --mapping_train ./stage2_semantic_cls/mapping_train.csv \
        --mapping_test ./stage2_semantic_cls/mapping_test.csv \
        --out_dir ./alignment_audit \
        --visualize_per_split 12

For the exact loader used in training, keep aug_data=False during auditing.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Same loader imported by the user's current training script.
from src.data.egg_dataset_ext_el import AllDataFeatureTwoEEG


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Audit Neuro-3D EEG tensor index <-> label <-> GT six-view alignment"
    )
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
        help="Root containing <label>/00.png ... 05.png or <dataset_name>/00.png ... 05.png",
    )
    p.add_argument("--sub_id", default="0001")
    p.add_argument(
        "--mapping_train",
        default="./stage2_semantic_cls/mapping_train.csv",
    )
    p.add_argument(
        "--mapping_test",
        default="./stage2_semantic_cls/mapping_test.csv",
    )
    p.add_argument("--out_dir", default="./alignment_audit")
    p.add_argument(
        "--visualize_per_split",
        type=int,
        default=12,
        help="Number of representative train/test samples to save as visual audit sheets.",
    )
    p.add_argument(
        "--focus_labels",
        default="",
        help="Comma-separated substrings such as 'can,airplane,chair'. These are always visualized if found.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 audits every dataset item. Positive value limits deterministic sample audit.",
    )
    p.add_argument(
        "--skip_prototype_audit",
        action="store_true",
        help="Skip train-vs-test raw EEG prototype similarity diagnostic.",
    )
    return p.parse_args()


# -------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------

def ensure_trailing_sep(path):
    # Dataset implementation concatenates data_path + 'video_new/' etc.
    path = str(Path(path).expanduser().resolve())
    if not path.endswith(os.sep):
        path += os.sep
    return path


def tensor_sha256(x):
    if torch.is_tensor(x):
        arr = x.detach().cpu().contiguous().numpy()
    else:
        arr = np.ascontiguousarray(x)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def decode_flat_index(ds, idx):
    """
    Reproduce AllDataFeatureTwoEEG.__getitem__ flat-index decoding.
    """
    per_subject = ds.cls_num * ds.obj_num * ds.trails_num

    sub_index = idx // per_subject
    sub_other = idx % per_subject

    cls_index = sub_other // (ds.obj_num * ds.trails_num)
    cls_other = sub_other % (ds.obj_num * ds.trails_num)

    obj_index = cls_other // ds.trails_num
    trial_index = cls_other % ds.trails_num

    return int(sub_index), int(cls_index), int(obj_index), int(trial_index)


def expected_dataset_index(ds, sub_index, cls_index, obj_index, trial_index):
    return (
        sub_index * ds.cls_num * ds.obj_num * ds.trails_num
        + cls_index * ds.obj_num * ds.trails_num
        + obj_index * ds.trails_num
        + trial_index
    )


def get_item_string(item, key, fallback=""):
    if key not in item:
        return str(fallback)
    value = item[key]
    if torch.is_tensor(value) and value.numel() == 1:
        value = value.item()
    return str(value)


def get_item_int(item, key, fallback=None):
    if key not in item:
        return fallback
    value = item[key]
    if torch.is_tensor(value):
        value = value.item()
    return int(value)


def resolve_render_dir(ds, name):
    # Prefer the loader's actual resolver so this audit checks exactly what
    # training/inference uses.
    if hasattr(ds, "_resolve_render_dir"):
        return Path(ds._resolve_render_dir(name))

    root = Path(ds.rendered_view_path)
    candidates = [root / str(name), root / str(name)[3:]]
    for candidate in candidates:
        if candidate.is_dir() and all(
            (candidate / f"{i:02d}.png").is_file()
            for i in range(6)
        ):
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve six GT views for name={name}; tried {candidates}"
    )


def load_png_like_dataset(ds, path):
    """
    Reproduce the rendered-image path used by the dataset as closely as possible.
    """
    if hasattr(ds, "_load_rgb_view"):
        return ds._load_rgb_view(Path(path)).float()

    with Image.open(path) as im:
        im = im.convert("RGB")
        size = int(getattr(ds, "rendered_image_size", 320))
        if im.size != (size, size):
            im = im.resize((size, size), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


# -------------------------------------------------------------------------
# Mapping CSV checks
# -------------------------------------------------------------------------

def load_mapping(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "cls_index",
        "obj_index",
        "class_prefix",
        "dataset_name",
        "label",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"{path}: missing mapping columns {sorted(missing)}; "
            f"available={list(df.columns)}"
        )
    return df


def audit_mapping_against_dataset(ds, mapping_df, split):
    """
    Verify mapping CSV against ds.name_list exactly.
    """
    rows = []
    failures = []

    for cls_index in range(ds.name_list.shape[0]):
        for obj_index in range(ds.name_list.shape[1]):
            actual_name = str(ds.name_list[cls_index, obj_index])
            actual_label = actual_name[3:]
            actual_prefix = actual_name[:3]

            match = mapping_df[
                (mapping_df["cls_index"] == cls_index)
                & (mapping_df["obj_index"] == obj_index)
            ]

            if len(match) != 1:
                failures.append(
                    f"{split}: cls={cls_index}, obj={obj_index}: "
                    f"mapping rows={len(match)}"
                )
                csv_name = csv_label = csv_prefix = "<missing/nonunique>"
            else:
                r = match.iloc[0]
                csv_name = str(r["dataset_name"])
                csv_label = str(r["label"])
                csv_prefix = str(r["class_prefix"])

                if csv_name != actual_name:
                    failures.append(
                        f"{split}: cls={cls_index}, obj={obj_index}: "
                        f"dataset_name CSV={csv_name} dataset={actual_name}"
                    )
                if csv_label != actual_label:
                    failures.append(
                        f"{split}: cls={cls_index}, obj={obj_index}: "
                        f"label CSV={csv_label} dataset={actual_label}"
                    )
                if csv_prefix != actual_prefix:
                    failures.append(
                        f"{split}: cls={cls_index}, obj={obj_index}: "
                        f"prefix CSV={csv_prefix} dataset={actual_prefix}"
                    )

            rows.append(
                {
                    "split": split,
                    "cls_index": cls_index,
                    "obj_index": obj_index,
                    "dataset_name": actual_name,
                    "label": actual_label,
                    "class_prefix": actual_prefix,
                    "csv_dataset_name": csv_name,
                    "csv_label": csv_label,
                    "csv_class_prefix": csv_prefix,
                    "mapping_exact": (
                        csv_name == actual_name
                        and csv_label == actual_label
                        and csv_prefix == actual_prefix
                    ),
                }
            )

    return pd.DataFrame(rows), failures


# -------------------------------------------------------------------------
# Dataset shape checks
# -------------------------------------------------------------------------

def audit_dataset_shapes(ds, split):
    eeg_shape = tuple(int(x) for x in ds.eeg_data.shape)
    name_shape = tuple(int(x) for x in ds.name_list.shape)

    problems = []

    if len(eeg_shape) < 6:
        problems.append(
            f"{split}: expected eeg_data with >=6 dims "
            f"[sub,class,obj,trial/channel...?], got {eeg_shape}"
        )

    if name_shape[0] != ds.cls_num:
        problems.append(
            f"{split}: name_list classes={name_shape[0]} != cls_num={ds.cls_num}"
        )

    if name_shape[1] != ds.obj_num:
        problems.append(
            f"{split}: name_list objects/class={name_shape[1]} != obj_num={ds.obj_num}"
        )

    if eeg_shape[0] != len(ds.sub_list):
        problems.append(
            f"{split}: eeg subject axis={eeg_shape[0]} != len(sub_list)={len(ds.sub_list)}"
        )

    if len(eeg_shape) >= 4:
        if eeg_shape[1] != ds.cls_num:
            problems.append(
                f"{split}: eeg class axis={eeg_shape[1]} != cls_num={ds.cls_num}"
            )
        if eeg_shape[2] != ds.obj_num:
            problems.append(
                f"{split}: eeg object axis={eeg_shape[2]} != obj_num={ds.obj_num}"
            )
        if eeg_shape[3] != ds.trails_num:
            problems.append(
                f"{split}: eeg trial axis={eeg_shape[3]} != trails_num={ds.trails_num}"
            )

    expected_len = (
        len(ds.sub_list) * ds.cls_num * ds.obj_num * ds.trails_num
    )
    if len(ds) != expected_len:
        problems.append(
            f"{split}: len(dataset)={len(ds)} != explicit index product={expected_len}"
        )

    return {
        "split": split,
        "eeg_shape": eeg_shape,
        "name_list_shape": name_shape,
        "cls_num": int(ds.cls_num),
        "obj_num": int(ds.obj_num),
        "trials_num": int(ds.trails_num),
        "dataset_len": int(len(ds)),
        "expected_len": int(expected_len),
        "problems": problems,
    }


# -------------------------------------------------------------------------
# Per-sample deterministic audit
# -------------------------------------------------------------------------

def audit_one_sample(ds, split, idx):
    sub_idx, cls_idx, obj_idx, trial_idx = decode_flat_index(ds, idx)

    roundtrip_idx = expected_dataset_index(
        ds, sub_idx, cls_idx, obj_idx, trial_idx
    )

    name = str(ds.name_list[cls_idx, obj_idx])
    expected_label = name[3:]
    expected_prefix = name[:3]

    # IMPORTANT: aug_data must be False so dataset item should be exactly equal
    # to the selected underlying EEG tensor.
    item = ds[idx]

    returned_name = get_item_string(item, "name", fallback=name)
    returned_label = get_item_string(
        item,
        "label",
        fallback=returned_name[3:] if len(returned_name) >= 3 else expected_label,
    )

    # Explicit metadata exists in the newer loader; fall back gracefully.
    returned_cls = get_item_int(item, "cls_index", cls_idx)
    returned_obj = get_item_int(item, "obj_index", obj_idx)
    returned_trial = get_item_int(item, "trial_index", trial_idx)
    returned_sub = get_item_int(item, "subject_index", sub_idx)

    direct_eeg_np = np.asarray(
        ds.eeg_data[sub_idx, cls_idx, obj_idx, trial_idx]
    )
    direct_eeg = torch.from_numpy(np.asarray(direct_eeg_np)).float()
    item_eeg = item["eeg_data"].detach().cpu().float()

    same_shape = tuple(item_eeg.shape) == tuple(direct_eeg.shape)
    if same_shape:
        eeg_max_abs_diff = float(
            (item_eeg - direct_eeg).abs().max().item()
        )
        eeg_equal = bool(torch.equal(item_eeg, direct_eeg))
        eeg_allclose = bool(
            torch.allclose(item_eeg, direct_eeg, atol=0.0, rtol=0.0)
        )
    else:
        eeg_max_abs_diff = float("inf")
        eeg_equal = False
        eeg_allclose = False

    render_dir = resolve_render_dir(ds, name)
    gt_paths = [
        render_dir / f"{i:02d}.png"
        for i in range(6)
    ]

    gt_exist = all(p.is_file() for p in gt_paths)

    rotation = item["rotation_images"].detach().cpu().float()
    gt_tensor = torch.stack(
        [load_png_like_dataset(ds, p) for p in gt_paths],
        dim=0,
    )

    image_shape_equal = tuple(rotation.shape) == tuple(gt_tensor.shape)
    if image_shape_equal:
        image_max_abs_diff = float(
            (rotation - gt_tensor).abs().max().item()
        )
        images_equal = bool(
            torch.allclose(rotation, gt_tensor, atol=1e-7, rtol=0.0)
        )
    else:
        image_max_abs_diff = float("inf")
        images_equal = False

    checks = {
        "index_roundtrip": roundtrip_idx == idx,
        "returned_name": returned_name == name,
        "returned_label": returned_label == expected_label,
        "returned_cls": returned_cls == cls_idx,
        "returned_obj": returned_obj == obj_idx,
        "returned_trial": returned_trial == trial_idx,
        "returned_sub": returned_sub == sub_idx,
        "eeg_exact": eeg_equal or eeg_allclose,
        "gt_six_exist": gt_exist,
        "rotation_matches_png": images_equal,
    }
    passed = all(checks.values())

    return {
        "split": split,
        "flat_idx": idx,
        "subject_index": sub_idx,
        "cls_index": cls_idx,
        "obj_index": obj_idx,
        "trial_index": trial_idx,
        "dataset_name": name,
        "label": expected_label,
        "class_prefix": expected_prefix,
        "returned_name": returned_name,
        "returned_label": returned_label,
        "returned_cls_index": returned_cls,
        "returned_obj_index": returned_obj,
        "returned_trial_index": returned_trial,
        "returned_subject_index": returned_sub,
        "eeg_shape": str(tuple(item_eeg.shape)),
        "eeg_mean": float(item_eeg.mean().item()),
        "eeg_std": float(item_eeg.std().item()),
        "eeg_min": float(item_eeg.min().item()),
        "eeg_max": float(item_eeg.max().item()),
        "eeg_sha256": tensor_sha256(item_eeg),
        "direct_eeg_sha256": tensor_sha256(direct_eeg),
        "eeg_max_abs_diff": eeg_max_abs_diff,
        "render_dir": str(render_dir),
        "rotation_shape": str(tuple(rotation.shape)),
        "image_max_abs_diff": image_max_abs_diff,
        **{f"check_{k}": bool(v) for k, v in checks.items()},
        "passed": bool(passed),
    }


def representative_indices(ds, count, focus_labels):
    """
    Deterministic spread across classes plus explicitly requested labels.
    """
    chosen = []

    if focus_labels:
        for cls_idx in range(ds.cls_num):
            for obj_idx in range(ds.obj_num):
                name = str(ds.name_list[cls_idx, obj_idx])
                label = name[3:].lower()
                if any(token in label for token in focus_labels):
                    idx = expected_dataset_index(
                        ds, 0, cls_idx, obj_idx, 0
                    )
                    chosen.append(idx)
                    break

    if count > 0:
        classes = np.linspace(
            0,
            ds.cls_num - 1,
            num=min(count, ds.cls_num),
            dtype=int,
        )
        for cls_idx in classes:
            idx = expected_dataset_index(
                ds, 0, int(cls_idx), 0, 0
            )
            chosen.append(idx)

    # preserve order + unique
    out = []
    seen = set()
    for x in chosen:
        if x not in seen:
            out.append(int(x))
            seen.add(x)
    return out


# -------------------------------------------------------------------------
# Visual audit sheet
# -------------------------------------------------------------------------

def save_visual_audit(ds, split, idx, row, out_path):
    """
    Saves:
      - EEG heatmap
      - first few EEG channel traces
      - exact six GT images loaded for this sample

    No predictions are involved: this is a source-alignment audit.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)

    import matplotlib.pyplot as plt
    item = ds[idx]
    eeg = item["eeg_data"].detach().cpu().float().numpy()

    # Collapse unexpected leading dimensions conservatively for display.
    while eeg.ndim > 2:
        eeg = eeg[0]

    name = row["dataset_name"]
    render_dir = Path(row["render_dir"])
    images = [Image.open(render_dir / f"{i:02d}.png").convert("RGB") for i in range(6)]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 4)

    ax0 = fig.add_subplot(gs[0, :2])
    if eeg.ndim == 2:
        ax0.imshow(eeg, aspect="auto")
        ax0.set_xlabel("Time")
        ax0.set_ylabel("EEG channel")
    else:
        ax0.plot(np.ravel(eeg))
        ax0.set_xlabel("Flattened EEG index")
    ax0.set_title("EEG tensor")

    ax1 = fig.add_subplot(gs[0, 2:])
    if eeg.ndim == 2:
        n_ch = min(6, eeg.shape[0])
        for c in range(n_ch):
            ax1.plot(eeg[c])
        ax1.set_title(f"First {n_ch} EEG channels")
        ax1.set_xlabel("Time")
    else:
        ax1.plot(np.ravel(eeg))
        ax1.set_title("EEG trace")

    for view_idx in range(6):
        r = 1 + view_idx // 3
        c = view_idx % 3
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(images[view_idx])
        ax.set_title(f"GT {view_idx:02d}.png")
        ax.axis("off")

    # Metadata / pass status in last grid cells if available.
    ax_meta = fig.add_subplot(gs[1:, 3])
    ax_meta.axis("off")
    meta = (
        f"split: {split}\n"
        f"flat_idx: {idx}\n"
        f"sub: {row['subject_index']}\n"
        f"cls: {row['cls_index']}\n"
        f"obj: {row['obj_index']}\n"
        f"trial: {row['trial_index']}\n"
        f"name: {name}\n"
        f"label: {row['label']}\n\n"
        f"EEG direct match: {row['check_eeg_exact']}\n"
        f"GT 6 files: {row['check_gt_six_exist']}\n"
        f"rotation=PNG: {row['check_rotation_matches_png']}\n"
        f"ALL PASS: {row['passed']}\n\n"
        f"EEG max diff:\n{row['eeg_max_abs_diff']:.3g}\n"
        f"Image max diff:\n{row['image_max_abs_diff']:.3g}"
    )
    ax_meta.text(0.0, 1.0, meta, va="top", family="monospace")

    fig.suptitle(
        f"{split.upper()} alignment audit | {name} | cls={row['cls_index']}",
        fontsize=14,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    for im in images:
        im.close()


# -------------------------------------------------------------------------
# Cross-split EEG prototype audit
# -------------------------------------------------------------------------

def instance_standardize_eeg(x):
    """
    Match the model's channel-wise EEG standardization:
        (x - mean_time) / (std_time + 1e-5)

    x expected [..., C, T].
    """
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-5)


def class_eeg_prototypes(ds):
    """
    One prototype per cls_index by averaging standardized EEG across
    subject/object/trial. Returns [72, D] normalized vectors.
    """
    prototypes = []

    for cls_idx in range(ds.cls_num):
        # [S, O, R, C, T]
        x = np.asarray(ds.eeg_data[:, cls_idx], dtype=np.float64)
        x = instance_standardize_eeg(x)

        # Average across subject/object/trial.
        while x.ndim > 2:
            x = x.mean(axis=0)

        vec = x.reshape(-1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        prototypes.append(vec)

    dims = {len(x) for x in prototypes}
    if len(dims) != 1:
        raise RuntimeError(
            f"Prototype dimensions differ across classes: {sorted(dims)}"
        )

    return np.stack(prototypes, axis=0)


def eeg_prototype_audit(train_ds, test_ds):
    train_proto = class_eeg_prototypes(train_ds)
    test_proto = class_eeg_prototypes(test_ds)

    if train_proto.shape[1] != test_proto.shape[1]:
        raise RuntimeError(
            "Train/test EEG feature dimensions differ after standardization: "
            f"{train_proto.shape} vs {test_proto.shape}"
        )

    sim = test_proto @ train_proto.T

    rows = []
    for cls_idx in range(test_ds.cls_num):
        order = np.argsort(-sim[cls_idx])
        rank = int(np.where(order == cls_idx)[0][0]) + 1

        top5 = order[:5].tolist()
        rows.append(
            {
                "test_cls_index": cls_idx,
                "test_class_prefix": str(test_ds.name_list[cls_idx, 0])[:3],
                "test_category": str(test_ds.name_list[cls_idx, 0])[3:].rsplit("_", 1)[0],
                "same_class_cosine": float(sim[cls_idx, cls_idx]),
                "same_class_rank": rank,
                "nearest_train_cls": int(order[0]),
                "nearest_train_prefix": str(train_ds.name_list[order[0], 0])[:3],
                "nearest_train_category": str(train_ds.name_list[order[0], 0])[3:].rsplit("_", 1)[0],
                "nearest_cosine": float(sim[cls_idx, order[0]]),
                "same_is_top1": bool(order[0] == cls_idx),
                "same_is_top5": bool(cls_idx in top5),
                "top5_train_cls": ",".join(map(str, top5)),
            }
        )

    df = pd.DataFrame(rows)

    diag = np.diag(sim)
    mask = ~np.eye(sim.shape[0], dtype=bool)
    offdiag = sim[mask]

    summary = {
        "same_class_top1_rate": float(df["same_is_top1"].mean()),
        "same_class_top5_rate": float(df["same_is_top5"].mean()),
        "median_same_class_rank": float(df["same_class_rank"].median()),
        "mean_same_class_rank": float(df["same_class_rank"].mean()),
        "diag_cosine_mean": float(diag.mean()),
        "diag_cosine_std": float(diag.std()),
        "offdiag_cosine_mean": float(offdiag.mean()),
        "offdiag_cosine_std": float(offdiag.std()),
        "warning": (
            "Prototype similarity is only a diagnostic. Weak category-level EEG "
            "separability can produce low diagonal accuracy even when indexing is correct."
        ),
    }
    return sim, df, summary


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = ensure_trailing_sep(args.data_path)
    rendered_view_path = str(
        Path(args.rendered_view_path).expanduser().resolve()
    )

    focus_labels = [
        x.strip().lower()
        for x in args.focus_labels.split(",")
        if x.strip()
    ]

    print("=" * 80)
    print("EEG tensor index <-> label <-> GT six-view alignment audit")
    print("=" * 80)
    print(f"data_path          : {data_path}")
    print(f"rendered_view_path : {rendered_view_path}")
    print(f"subject            : {args.sub_id}")
    print()

    # aug_data=False is critical: otherwise dataset can average/add noise and
    # exact tensor-equality checks would intentionally fail.
    train_ds = AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=[args.sub_id],
        train=True,
        num_frames=6,
        rendered_view_path=rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )
    test_ds = AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=[args.sub_id],
        train=False,
        num_frames=6,
        rendered_view_path=rendered_view_path,
        aug_data=False,
        strict_rendered_views=True,
    )

    mapping_train = load_mapping(args.mapping_train)
    mapping_test = load_mapping(args.mapping_test)

    report = {
        "subject": args.sub_id,
        "data_path": data_path,
        "rendered_view_path": rendered_view_path,
        "shape_audit": {},
        "mapping_audit": {},
        "sample_audit": {},
        "prototype_audit": None,
    }

    # 1) Shape/index-axis audit.
    for split, ds in [("train", train_ds), ("test", test_ds)]:
        shape_result = audit_dataset_shapes(ds, split)
        report["shape_audit"][split] = shape_result

        print(f"[{split}] eeg_data shape : {shape_result['eeg_shape']}")
        print(f"[{split}] name_list shape: {shape_result['name_list_shape']}")
        print(
            f"[{split}] cls/obj/trial   : "
            f"{shape_result['cls_num']}/"
            f"{shape_result['obj_num']}/"
            f"{shape_result['trials_num']}"
        )
        if shape_result["problems"]:
            for problem in shape_result["problems"]:
                print("  [FAIL]", problem)
        else:
            print(f"[{split}] shape/index contract: PASS")

    # 2) CSV mapping vs runtime dataset.name_list.
    train_map_checked, train_map_fail = audit_mapping_against_dataset(
        train_ds, mapping_train, "train"
    )
    test_map_checked, test_map_fail = audit_mapping_against_dataset(
        test_ds, mapping_test, "test"
    )

    train_map_checked.to_csv(
        out_dir / "mapping_train_checked.csv",
        index=False,
    )
    test_map_checked.to_csv(
        out_dir / "mapping_test_checked.csv",
        index=False,
    )

    report["mapping_audit"]["train"] = {
        "rows": len(train_map_checked),
        "failures": train_map_fail,
        "all_exact": len(train_map_fail) == 0,
    }
    report["mapping_audit"]["test"] = {
        "rows": len(test_map_checked),
        "failures": test_map_fail,
        "all_exact": len(test_map_fail) == 0,
    }

    print()
    print(
        f"[mapping train] exact={len(train_map_fail) == 0} "
        f"failures={len(train_map_fail)}"
    )
    print(
        f"[mapping test ] exact={len(test_map_fail) == 0} "
        f"failures={len(test_map_fail)}"
    )

    # 3) Every actual dataset sample: direct EEG tensor and GT image check.
    audit_rows = []

    for split, ds in [("train", train_ds), ("test", test_ds)]:
        total = len(ds)
        n = total if args.max_samples <= 0 else min(total, args.max_samples)

        print()
        print(f"[{split}] auditing {n}/{total} actual dataset items ...")

        failed = 0
        for idx in range(n):
            row = audit_one_sample(ds, split, idx)
            audit_rows.append(row)
            if not row["passed"]:
                failed += 1
                if failed <= 20:
                    failed_checks = [
                        k.replace("check_", "")
                        for k, v in row.items()
                        if k.startswith("check_") and not bool(v)
                    ]
                    print(
                        f"  [FAIL] idx={idx} "
                        f"name={row['dataset_name']} "
                        f"checks={failed_checks}"
                    )

        report["sample_audit"][split] = {
            "audited": n,
            "total": total,
            "failed": failed,
            "passed": n - failed,
            "all_pass": failed == 0,
        }
        print(
            f"[{split}] sample audit: "
            f"PASS={n - failed}/{n}, FAIL={failed}/{n}"
        )

        # Visualize representative source samples.
        vis_indices = representative_indices(
            ds,
            args.visualize_per_split,
            focus_labels,
        )
        for idx in vis_indices:
            # Reuse result if already audited; otherwise audit it now.
            candidates = [
                r for r in audit_rows
                if r["split"] == split and r["flat_idx"] == idx
            ]
            if candidates:
                row = candidates[0]
            else:
                row = audit_one_sample(ds, split, idx)

            safe_name = row["dataset_name"].replace("/", "_")
            out_path = (
                out_dir
                / "visuals"
                / split
                / f"{idx:05d}_{safe_name}.png"
            )
            save_visual_audit(
                ds,
                split,
                idx,
                row,
                out_path,
            )

    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(
        out_dir / "audit_samples.csv",
        index=False,
    )

    # 4) Train/test EEG prototype diagonal diagnostic.
    if not args.skip_prototype_audit:
        print()
        print("[prototype] computing train-vs-test class EEG similarity ...")
        try:
            sim, prototype_df, proto_summary = eeg_prototype_audit(
                train_ds,
                test_ds,
            )

            np.save(
                out_dir / "eeg_similarity_matrix.npy",
                sim,
            )
            prototype_df.to_csv(
                out_dir / "eeg_prototype_similarity.csv",
                index=False,
            )
            report["prototype_audit"] = proto_summary

            print(
                "[prototype] same-class top1="
                f"{proto_summary['same_class_top1_rate']:.3f}, "
                "top5="
                f"{proto_summary['same_class_top5_rate']:.3f}, "
                "median rank="
                f"{proto_summary['median_same_class_rank']:.1f}"
            )
            print(
                "[prototype] diagonal cosine mean="
                f"{proto_summary['diag_cosine_mean']:.4f}, "
                "off-diagonal mean="
                f"{proto_summary['offdiag_cosine_mean']:.4f}"
            )
            print(
                "[prototype] NOTE: low prototype accuracy alone is NOT proof "
                "of label misalignment."
            )
        except Exception as e:
            report["prototype_audit"] = {
                "error": repr(e),
                "warning": "Deterministic index/image audit is still valid.",
            }
            print("[prototype] skipped due to:", repr(e))

    # Final decision: only deterministic checks.
    deterministic_failures = []
    for split in ["train", "test"]:
        deterministic_failures.extend(
            report["shape_audit"][split]["problems"]
        )
        deterministic_failures.extend(
            report["mapping_audit"][split]["failures"]
        )
        if not report["sample_audit"][split]["all_pass"]:
            deterministic_failures.append(
                f"{split}: one or more actual dataset samples failed"
            )

    report["deterministic_alignment_pass"] = (
        len(deterministic_failures) == 0
    )
    report["deterministic_failures"] = deterministic_failures

    with open(
        out_dir / "audit_report.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 80)
    if report["deterministic_alignment_pass"]:
        print(
            "DETERMINISTIC ALIGNMENT: PASS\n"
            "dataset[idx] -> EEG tensor index -> name/label -> GT six PNGs "
            "are internally consistent."
        )
        print(
            "This does NOT by itself prove that the original EEG acquisition's "
            "class axis was semantically annotated correctly; the prototype "
            "diagnostic provides additional, non-conclusive evidence."
        )
    else:
        print("DETERMINISTIC ALIGNMENT: FAIL")
        for problem in deterministic_failures[:30]:
            print(" -", problem)
    print("=" * 80)
    print(f"Report: {out_dir / 'audit_report.json'}")
    print(f"Samples: {out_dir / 'audit_samples.csv'}")
    print(f"Visuals: {out_dir / 'visuals'}")


if __name__ == "__main__":
    main()
