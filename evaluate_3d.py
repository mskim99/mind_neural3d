#!/usr/bin/env python3
"""
Evaluation aligned to the CURRENT Neuro-3D training + inference pipeline.

This evaluator is designed for:

Training
--------
train_neural3d_pp_semantic_cls.py / semantic-invariant revision
  - explicit cls_index / class_prefix / label metadata
  - unseen test objects (normally suffix 08/09)
  - object-invariant semantic training
  - optional final EEG semantic probe

Inference
---------
inference_neural3d_pp_render_views_el_fixed.py
  - evaluation_pairs.csv:
      sample_id,dataset_name,label,class_prefix,cls_index,obj_index,
      trial_index,subject_index,gt_dir,pred_dir,render_grid
  - pred_dir contains cleaned canonical six views:
      00.png ... 05.png
  - gt_dir contains the exact matching canonical six GT views
  - render_grid is the packed 3x2 image sent to InstantMesh

Mesh evaluation
---------------
A later InstantMesh rendering CSV can instead provide:
  - pred_mesh_view_dir
  - gt_view_dir (or gt_dir)

The evaluator can therefore quantify BOTH:
  1) generated six-view images BEFORE InstantMesh
  2) six views rendered FROM the reconstructed mesh

Core Brain3D-style metrics
--------------------------
  - 2-way Top-1
  - 10-way Top-1 / Top-2
  - 50-way Top-1 / Top-2
  - CLIPScore (OpenCLIP ViT-B/16 image-image cosine)
  - LPIPS (AlexNet)
  - Inception Score
  - FID

Additional current-pipeline diagnostics
---------------------------------------
  - strict inference/training mapping audit
  - per-category metrics (72 semantic classes)
  - per-object-suffix metrics (e.g. unseen 08 vs 09)
  - optional 72-way CLIP text/category retrieval on predicted views
  - optional EEG semantic checkpoint probe matching the revised trainer:
        train individual
        train averaged
        test individual
        test averaged
    using train semantic prototypes and the learned classifier

Recommended generated-view evaluation
-------------------------------------
CUDA_VISIBLE_DEVICES=0 python evaluate_3d_semantic_aligned.py \
  --pairs_csv ./inference_results/evaluation_pairs.csv \
  --target generated \
  --out_dir ./inference_results/evaluation_generated \
  --reference_mode six_view \
  --strict_mapping \
  --device cuda \
  --batch_size 16 \
  --nway_trials 100

Recommended mesh-render evaluation
----------------------------------
CUDA_VISIBLE_DEVICES=0 python evaluate_3d_semantic_aligned.py \
  --pairs_csv ./mesh_results/mesh_pairs.csv \
  --target mesh \
  --out_dir ./mesh_results/evaluation \
  --reference_mode six_view \
  --strict_mapping \
  --device cuda \
  --batch_size 16 \
  --nway_trials 100

Optional final EEG semantic probe
---------------------------------
Add:
  --semantic_ckpt ./stage2_semantic_invariant/checkpoints/model_final.pt \
  --data_path /data/jionkim/neuro_3D \
  --rendered_view_path /data/jionkim/neuro_3D/render_grid_v4 \
  --sub_id sub01

Dependencies
------------
pip install open_clip_torch lpips torchmetrics torch-fidelity pandas pillow tqdm
"""

import argparse
import json
import os
import random
import re
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights,
    vit_b_16, ViT_B_16_Weights,
)

try:
    import open_clip
except ImportError as e:
    raise ImportError(
        "Install open_clip_torch: pip install open_clip_torch"
    ) from e

try:
    import lpips
except ImportError as e:
    raise ImportError(
        "Install lpips: pip install lpips"
    ) from e

try:
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError as e:
    raise ImportError(
        "Install torchmetrics + torch-fidelity: "
        "pip install torchmetrics torch-fidelity"
    ) from e


BRAIN3D_REFERENCE = {
    "BrainVis": [
        0.880, 0.706, 0.796, 0.578, 0.649,
        0.617, 16.590, 204.015, 0.789,
    ],
    "DreamDiffusion": [
        0.730, 0.314, 0.480, 0.164, 0.206,
        0.564, 14.871, 232.256, 0.780,
    ],
    "EEG-CLIP": [
        0.857, 0.655, 0.742, 0.545, 0.602,
        0.608, 17.173, 156.631, 0.788,
    ],
    "GWIT": [
        0.946, 0.854, 0.906, 0.763, 0.822,
        0.648, 17.195, 153.295, 0.783,
    ],
}


# ============================================================================
# CLI / metadata
# ============================================================================

@dataclass
class EvalObject:
    sample_id: str
    label: str
    category: str
    pred_views: List[str]
    gt_views: List[str]

    dataset_name: str = ""
    class_prefix: str = ""
    cls_index: int = -1
    obj_index: int = -1
    trial_index: int = -1
    subject_index: int = -1
    object_suffix: str = ""
    pred_source: str = ""
    pred_dir: str = ""
    gt_dir: str = ""


def parse_args():
    p = argparse.ArgumentParser(
        description="Neuro-3D / Brain3D aligned 3D evaluation"
    )

    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", default="./brain3d_mesh_eval")

    p.add_argument(
        "--target",
        choices=["auto", "generated", "mesh"],
        default="auto",
        help=(
            "generated: evaluate inference pred_dir before InstantMesh; "
            "mesh: evaluate pred_mesh_view_dir / mesh_root; "
            "auto: prefer mesh column when available, otherwise generated."
        ),
    )
    p.add_argument(
        "--reference_mode",
        choices=["six_view", "single"],
        default="six_view",
    )

    # Explicit override. Leave blank for automatic schema resolution.
    p.add_argument("--pred_dir_col", default="")
    p.add_argument("--gt_dir_col", default="")
    p.add_argument("--gt_image_col", default="gt_image")

    p.add_argument(
        "--mesh_root",
        default="",
        help=(
            "Optional mesh-render root when pairs_csv is the inference "
            "evaluation_pairs.csv. Expected <mesh_root>/<label>/00.png... "
            "or <mesh_root>/<sample_id>/00.png..."
        ),
    )
    p.add_argument(
        "--mesh_key",
        choices=["label", "sample_id"],
        default="label",
    )

    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument(
        "--classifier",
        choices=["resnet50", "resnet18", "vit_b_16"],
        default="resnet50",
    )
    p.add_argument("--nway_trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--skip_is_fid", action="store_true")
    p.add_argument("--save_reference_table", action="store_true")

    # Current Neuro-3D mapping audit.
    p.add_argument(
        "--strict_mapping",
        action="store_true",
        help=(
            "Require inference metadata to be internally consistent: "
            "label == dataset_name[3:], one cls_index -> one category, etc."
        ),
    )
    p.add_argument(
        "--strict_test_suffix",
        action="store_true",
        help="Require every evaluated object label to end in _08 or _09.",
    )

    # Training-aligned output-semantic metric.
    p.add_argument(
        "--skip_clip_category",
        action="store_true",
        help=(
            "Disable external CLIP text/category retrieval diagnostic."
        ),
    )
    p.add_argument(
        "--clip_prompt_template",
        default="a photo of a {}",
        help=(
            "Template for the OPTIONAL CLIP category retrieval diagnostic. "
            "Use exactly one '{}' placeholder."
        ),
    )

    # Optional final checkpoint EEG semantic probe.
    p.add_argument(
        "--semantic_ckpt",
        default="",
        help=(
            "Optional semantic-invariant training checkpoint. "
            "If supplied, run the final four-way EEG semantic probe."
        ),
    )
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    p.add_argument(
        "--rendered_view_path",
        default="/data/jionkim/neuro_3D/render_grid_v4",
    )
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--semantic_probe_batchsize", type=int, default=64)

    return p.parse_args()


def _safe_int(value, default=-1):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def category_from_label(label: str) -> str:
    """
    Current labels are object identities:
        airplane_08 -> semantic category airplane
        office_chair_09 -> semantic category office_chair
    """
    label = str(label).strip()
    m = re.match(r"^(.*)_([0-9]{2})$", label)
    if m:
        return m.group(1)
    return label


def object_suffix_from_label(label: str) -> str:
    m = re.match(r"^.*_([0-9]{2})$", str(label).strip())
    return m.group(1) if m else ""


# ============================================================================
# Path / CSV alignment
# ============================================================================

def load_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def canonical_view_paths(directory: str, num_views: int = 6) -> List[str]:
    directory = str(Path(directory).expanduser())
    paths = [
        os.path.join(directory, f"{i:02d}.png")
        for i in range(num_views)
    ]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing canonical views under {directory}: "
            + ", ".join(os.path.basename(x) for x in missing)
        )
    return paths


def _resolve_existing_column(
    df: pd.DataFrame,
    requested: str,
    aliases: Sequence[str],
    role: str,
) -> Optional[str]:
    if requested:
        if requested in df.columns:
            return requested
        raise KeyError(
            f"Requested {role} column {requested!r} not found. "
            f"Available={list(df.columns)}"
        )

    for candidate in aliases:
        if candidate in df.columns:
            return candidate

    return None


def resolve_prediction_source(args, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    if args.pred_dir_col:
        if args.pred_dir_col not in df.columns:
            raise KeyError(f"--pred_dir_col={args.pred_dir_col!r} not in CSV.")
        name = "mesh" if "mesh" in args.pred_dir_col else "generated"
        return name, args.pred_dir_col

    mesh_col = _resolve_existing_column(
        df, "", ["pred_mesh_view_dir", "mesh_view_dir", "pred_mesh_dir"], "mesh prediction directory"
    )
    generated_col = _resolve_existing_column(
        df, "", ["pred_dir", "pred_clean_dir", "pred_views_dir", "views_dir"], "generated-view directory"
    )

    if args.target == "mesh":
        if args.mesh_root:
            return "mesh_root", None
        if mesh_col:
            return "mesh", mesh_col
        raise KeyError("--target mesh requested, but no mesh-view column exists and --mesh_root was not provided.")

    if args.target == "generated":
        if generated_col:
            return "generated", generated_col
        # [PATCH] 컬럼이 없어도 에러를 띄우지 않고 generated 모드로 진행
        return "generated", None

    # auto
    if mesh_col:
        return "mesh", mesh_col
    if args.mesh_root:
        return "mesh_root", None
    if generated_col:
        return "generated", generated_col

    # [PATCH] 기본값으로 generated 모드 반환
    return "generated", None


def resolve_gt_source(args, df):
    if args.reference_mode == "six_view":
        col = _resolve_existing_column(
            df,
            args.gt_dir_col,
            ["gt_view_dir", "gt_dir"],
            "GT six-view directory",
        )
        # [PATCH] raise KeyError 제거, 찾지 못하면 None 반환
        return col, None

    image_col = _resolve_existing_column(
        df,
        args.gt_image_col,
        ["gt_image", "stimulus_image", "reference_image"],
        "GT stimulus image",
    )
    if image_col is None:
        raise KeyError("single mode requires a GT stimulus-image column.")
    return None, image_col


def resolve_mesh_root_dir(
    mesh_root: str,
    row,
    mesh_key: str,
) -> str:
    root = Path(mesh_root).expanduser()
    key = str(row[mesh_key]).strip()

    candidates = [
        root / key,
        root / "views" / key,
        root / "renders" / key,
        root / "mesh_views" / key,
    ]

    for c in candidates:
        if all(
            (c / f"{i:02d}.png").is_file()
            for i in range(6)
        ):
            return str(c)

    raise FileNotFoundError(
        f"Could not find six mesh-render views for {mesh_key}={key!r} "
        f"under {root}. Tried: {[str(c) for c in candidates]}"
    )


def audit_mapping_row(row_idx: int, row) -> Dict:
    dataset_name = (
        str(row["dataset_name"]).strip()
        if "dataset_name" in row.index
        and not pd.isna(row["dataset_name"])
        else ""
    )
    label = (
        str(row["label"]).strip()
        if "label" in row.index
        and not pd.isna(row["label"])
        else ""
    )
    category = category_from_label(label)

    cls_index = (
        _safe_int(row["cls_index"])
        if "cls_index" in row.index
        else -1
    )
    class_prefix = (
        str(row["class_prefix"]).strip()
        if "class_prefix" in row.index
        and not pd.isna(row["class_prefix"])
        else ""
    )

    checks = {
        "row": row_idx,
        "sample_id": (
            str(row["sample_id"]).strip()
            if "sample_id" in row.index
            else ""
        ),
        "dataset_name": dataset_name,
        "label": label,
        "category": category,
        "cls_index": cls_index,
        "class_prefix": class_prefix,
        "label_matches_name": True,
        "prefix_matches_cls_index": True,
        "test_suffix_ok": object_suffix_from_label(label) in {"08", "09"},
    }

    if dataset_name:
        checks["label_matches_name"] = (
            label == dataset_name[3:]
        )

    if class_prefix and cls_index >= 0:
        try:
            checks["prefix_matches_cls_index"] = (
                int(class_prefix) == cls_index + 1
            )
        except ValueError:
            # Keep non-numeric prefixes auditable without inventing a rule.
            checks["prefix_matches_cls_index"] = True

    return checks


def load_eval_objects(args):
    csv_path = Path(args.pairs_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns:
        raise KeyError(
            "pairs_csv must contain sample_id. "
            f"Available={list(df.columns)}"
        )

    pred_source, pred_col = resolve_prediction_source(args, df)
    gt_dir_col, gt_image_col = resolve_gt_source(args, df)

    print(f"[CSV] file              : {csv_path}")
    print(f"[CSV] prediction source : {pred_source}")
    if pred_col:
        print(f"[CSV] prediction column : {pred_col}")
    if args.mesh_root:
        print(f"[CSV] mesh root         : {args.mesh_root}")

    if args.reference_mode == "six_view":
        print(f"[CSV] GT column         : {gt_dir_col}")
    else:
        print(f"[CSV] GT image column   : {gt_image_col}")

    audit_rows = []
    objects = []
    resolved_rows = []
    seen_ids = set()

    for row_idx, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()
        if not sample_id or sample_id.lower() == "nan":
            raise ValueError(
                f"Invalid sample_id at CSV row {row_idx}"
            )
        if sample_id in seen_ids:
            raise ValueError(
                f"Duplicate sample_id: {sample_id}"
            )
        seen_ids.add(sample_id)

        label = (
            str(row["label"]).strip()
            if "label" in df.columns
            else sample_id
        )
        category = category_from_label(label)
        suffix = object_suffix_from_label(label)

        audit = audit_mapping_row(row_idx, row)
        audit_rows.append(audit)

        if args.strict_mapping:
            if not audit["label_matches_name"]:
                raise RuntimeError(
                    f"Mapping error row {row_idx}: "
                    f"label={label!r} != dataset_name[3:]"
                )
            if not audit["prefix_matches_cls_index"]:
                raise RuntimeError(
                    f"Mapping error row {row_idx}: "
                    f"class_prefix={audit['class_prefix']!r}, "
                    f"cls_index={audit['cls_index']}"
                )
        '''
        if args.strict_test_suffix and suffix not in {"08", "09"}:
            raise RuntimeError(
                f"Expected unseen test suffix 08/09, got label={label!r}"
            )
        '''
        if pred_source == "mesh_root":
            if args.mesh_key not in row.index:
                raise KeyError(f"--mesh_key {args.mesh_key!r} not in pairs CSV.")
            pred_dir = resolve_mesh_root_dir(args.mesh_root, row, args.mesh_key)
        else:
            # [PATCH] pred_col이 없으면 CSV가 위치한 폴더의 'views' 하위 폴더로 자동 추론
            if pred_col and pred_col in row.index:
                pred_dir = str(row[pred_col]).strip()
            else:
                csv_dir = Path(args.pairs_csv).parent
                pred_dir = str(csv_dir / "views" / sample_id)

            # [PATCH] GT 누락 시 패스(Skip) + gt_dir 자동 추론
        try:
            pred_views = canonical_view_paths(pred_dir, args.num_views)

            if args.reference_mode == "six_view":
                # gt_dir_col이 없으면 인자로 받은 --rendered_view_path에서 자동으로 매칭
                if gt_dir_col and gt_dir_col in row.index:
                    gt_dir = str(row[gt_dir_col]).strip()
                else:
                    gt_dir = str(Path(args.rendered_view_path).expanduser() / label)

                gt_views = canonical_view_paths(gt_dir, args.num_views)
            else:
                gt_dir = ""
                gt_image = str(row[gt_image_col]).strip()
                if not os.path.isfile(gt_image):
                    raise FileNotFoundError(gt_image)
                gt_views = [gt_image] * args.num_views
        except FileNotFoundError as e:
            print(f"[Warning] Skipping evaluation for '{sample_id}' due to missing views: {e}")
            continue

        obj = EvalObject(
            sample_id=sample_id,
            label=label,
            category=category,
            pred_views=pred_views,
            gt_views=gt_views,
            dataset_name=(
                str(row["dataset_name"]).strip()
                if "dataset_name" in df.columns
                else ""
            ),
            class_prefix=(
                str(row["class_prefix"]).strip()
                if "class_prefix" in df.columns
                else ""
            ),
            cls_index=(
                _safe_int(row["cls_index"])
                if "cls_index" in df.columns
                else -1
            ),
            obj_index=(
                _safe_int(row["obj_index"])
                if "obj_index" in df.columns
                else -1
            ),
            trial_index=(
                _safe_int(row["trial_index"])
                if "trial_index" in df.columns
                else -1
            ),
            subject_index=(
                _safe_int(row["subject_index"])
                if "subject_index" in df.columns
                else -1
            ),
            object_suffix=suffix,
            pred_source=pred_source,
            pred_dir=pred_dir,
            gt_dir=gt_dir,
        )
        objects.append(obj)

        resolved_rows.append({
            "sample_id": sample_id,
            "dataset_name": obj.dataset_name,
            "label": label,
            "category": category,
            "object_suffix": suffix,
            "class_prefix": obj.class_prefix,
            "cls_index": obj.cls_index,
            "obj_index": obj.obj_index,
            "trial_index": obj.trial_index,
            "subject_index": obj.subject_index,
            "pred_source": pred_source,
            "pred_dir": pred_dir,
            "gt_dir": gt_dir,
        })

    if not objects:
        raise RuntimeError("No evaluation samples found.")

    # One cls_index must correspond to one semantic category.
    known = [
        (o.cls_index, o.category)
        for o in objects
        if o.cls_index >= 0
    ]
    if known:
        class_map = {}
        for c, name in known:
            if c in class_map and class_map[c] != name:
                raise RuntimeError(
                    f"cls_index={c} maps to both "
                    f"{class_map[c]!r} and {name!r}."
                )
            class_map[c] = name

    print(
        f"[data] objects={len(objects)}, "
        f"categories={len(set(o.category for o in objects))}, "
        f"views/object={args.num_views}, "
        f"reference_mode={args.reference_mode}"
    )

    return (
        objects,
        pd.DataFrame(audit_rows),
        pd.DataFrame(resolved_rows),
        pred_source,
    )


# ============================================================================
# Image metrics
# ============================================================================

def image_to_unit_tensor(
    image: Image.Image,
    size: Tuple[int, int] = None,
) -> torch.Tensor:
    x = TF.to_tensor(image.convert("RGB"))
    if (
        size is not None
        and tuple(x.shape[-2:]) != tuple(size)
    ):
        x = TF.resize(
            x,
            list(size),
            interpolation=TF.InterpolationMode.BICUBIC,
            antialias=True,
        )
    return x.clamp(0, 1)


def image_to_uint8_tensor(
    image: Image.Image,
    size=(299, 299),
) -> torch.Tensor:
    x = image_to_unit_tensor(
        image,
        size=size,
    )
    return (
        x * 255.0
    ).round().to(torch.uint8)


class CLIPScorer:
    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess = (
            open_clip.create_model_and_transforms(
                "ViT-B-16",
                pretrained="openai",
            )
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(
        self,
        paths: Sequence[str],
        batch_size: int,
    ) -> torch.Tensor:
        out = []
        for s in range(0, len(paths), batch_size):
            x = torch.stack([
                self.preprocess(load_rgb(p))
                for p in paths[s:s + batch_size]
            ]).to(self.device)

            f = F.normalize(
                self.model.encode_image(x).float(),
                dim=-1,
            )
            out.append(f.cpu())

        return torch.cat(out, dim=0)

    @torch.no_grad()
    def encode_categories(
        self,
        categories: Sequence[str],
        template: str,
    ) -> torch.Tensor:
        if template.count("{}") != 1:
            raise ValueError(
                "--clip_prompt_template must contain exactly one '{}'."
            )

        prompts = [
            template.format(
                str(c).replace("_", " ")
            )
            for c in categories
        ]

        tokens = open_clip.tokenize(prompts).to(self.device)
        feat = F.normalize(
            self.model.encode_text(tokens).float(),
            dim=-1,
        )
        return feat.cpu()


class LPIPSScorer:
    def __init__(self, device):
        self.device = device
        self.model = lpips.LPIPS(
            net="alex"
        ).to(device).eval()

    @torch.no_grad()
    def pair_scores(
        self,
        gt_paths,
        pred_paths,
        batch_size,
    ):
        values = []

        for s in range(0, len(gt_paths), batch_size):
            gb, pb = [], []

            for gp, pp in zip(
                gt_paths[s:s + batch_size],
                pred_paths[s:s + batch_size],
            ):
                gi = load_rgb(gp)
                pi = load_rgb(pp)

                target_hw = (
                    gi.height,
                    gi.width,
                )

                g = image_to_unit_tensor(gi)
                p = image_to_unit_tensor(
                    pi,
                    target_hw,
                )

                gb.append(g * 2 - 1)
                pb.append(p * 2 - 1)

            g = torch.stack(gb).to(self.device)
            p = torch.stack(pb).to(self.device)

            values.extend(
                self.model(
                    g,
                    p,
                ).reshape(-1).float().cpu().tolist()
            )

        return np.asarray(
            values,
            dtype=np.float64,
        )


def build_classifier(name: str, device):
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
    elif name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)

    return (
        model.to(device).eval(),
        weights.transforms(),
        weights.meta["categories"],
    )


@torch.no_grad()
def classifier_probabilities(
    paths,
    model,
    preprocess,
    device,
    batch_size,
):
    out = []

    for s in range(0, len(paths), batch_size):
        x = torch.stack([
            preprocess(load_rgb(p))
            for p in paths[s:s + batch_size]
        ]).to(device)

        out.append(
            torch.softmax(
                model(x).float(),
                dim=-1,
            ).cpu()
        )

    return torch.cat(
        out,
        dim=0,
    ).numpy()


# ============================================================================
# Brain3D N-way
# ============================================================================

def nway_trials_for_pair(
    pred_prob,
    positive_class,
    n_way,
    top_k,
    trials,
    rng,
):
    K = pred_prob.shape[0]

    neg_pool = np.concatenate([
        np.arange(
            0,
            positive_class,
            dtype=np.int64,
        ),
        np.arange(
            positive_class + 1,
            K,
            dtype=np.int64,
        ),
    ])

    success = np.zeros(
        trials,
        dtype=np.float64,
    )

    for t in range(trials):
        neg = rng.choice(
            neg_pool,
            size=n_way - 1,
            replace=False,
        )

        candidates = np.concatenate([
            [positive_class],
            neg,
        ])

        scores = pred_prob[candidates]
        rank = np.argsort(-scores)

        positive_rank = int(
            np.where(rank == 0)[0][0]
        )

        success[t] = float(
            positive_rank < top_k
        )

    return success


def compute_all_nway(
    gt_probs,
    pred_probs,
    object_index,
    num_objects,
    trials,
    seed,
):
    settings = [
        ("2way_top1", 2, 1),
        ("10way_top1", 10, 1),
        ("10way_top2", 10, 2),
        ("50way_top1", 50, 1),
        ("50way_top2", 50, 2),
    ]

    positive = gt_probs.argmax(axis=1)

    summary = {}
    per_object = {}
    trial_rows = []

    for setting_idx, (
        name,
        n_way,
        top_k,
    ) in enumerate(settings):
        pair_trial = np.zeros(
            (
                len(pred_probs),
                trials,
            ),
            dtype=np.float64,
        )

        for pair_idx in range(len(pred_probs)):
            rng = np.random.default_rng(
                seed
                + setting_idx * 1_000_003
                + pair_idx * 9_973
            )

            pair_trial[pair_idx] = (
                nway_trials_for_pair(
                    pred_probs[pair_idx],
                    int(positive[pair_idx]),
                    n_way,
                    top_k,
                    trials,
                    rng,
                )
            )

        obj_scores = np.zeros(
            num_objects,
            dtype=np.float64,
        )

        for oi in range(num_objects):
            obj_scores[oi] = pair_trial[
                object_index == oi
            ].mean()

        per_object[name] = obj_scores

        global_trials = []

        for t in range(trials):
            vals = [
                pair_trial[
                    object_index == oi,
                    t,
                ].mean()
                for oi in range(num_objects)
            ]

            v = float(np.mean(vals))
            global_trials.append(v)

            trial_rows.append({
                "metric": name,
                "trial": t,
                "value": v,
            })

        global_trials = np.asarray(
            global_trials
        )

        summary[name] = {
            "mean": float(
                global_trials.mean()
            ),
            "std": float(
                global_trials.std(ddof=0)
            ),
        }

    return (
        summary,
        per_object,
        trial_rows,
        positive,
    )


# ============================================================================
# CLIP category-space diagnostic
# ============================================================================

def _topk_bool(logits, target, k):
    k = min(
        k,
        logits.shape[1],
    )
    top = logits.topk(
        k,
        dim=1,
    ).indices

    return (
        top == target[:, None]
    ).any(dim=1)


def compute_clip_category_metrics(
    objects,
    gt_features,
    pred_features,
    object_index,
    text_features,
    categories,
):
    category_to_idx = {
        c: i
        for i, c in enumerate(categories)
    }

    target = torch.tensor([
        category_to_idx[
            objects[int(oi)].category
        ]
        for oi in object_index
    ], dtype=torch.long)

    pred_logits = (
        F.normalize(
            pred_features.float(),
            dim=-1,
        )
        @ F.normalize(
            text_features.float(),
            dim=-1,
        ).t()
    )

    gt_logits = (
        F.normalize(
            gt_features.float(),
            dim=-1,
        )
        @ F.normalize(
            text_features.float(),
            dim=-1,
        ).t()
    )

    pred_view_top1 = (
        pred_logits.argmax(dim=1)
        == target
    )
    pred_view_top5 = _topk_bool(
        pred_logits,
        target,
        5,
    )

    gt_view_top1 = (
        gt_logits.argmax(dim=1)
        == target
    )
    gt_view_top5 = _topk_bool(
        gt_logits,
        target,
        5,
    )

    # Object-level category retrieval:
    # average all six normalized-view similarities.
    object_pred_logits = []
    object_gt_logits = []
    object_target = []

    for oi, obj in enumerate(objects):
        mask = (
            torch.from_numpy(
                object_index
            ) == oi
        )

        object_pred_logits.append(
            pred_logits[mask].mean(dim=0)
        )
        object_gt_logits.append(
            gt_logits[mask].mean(dim=0)
        )
        object_target.append(
            category_to_idx[obj.category]
        )

    object_pred_logits = torch.stack(
        object_pred_logits,
        dim=0,
    )
    object_gt_logits = torch.stack(
        object_gt_logits,
        dim=0,
    )
    object_target = torch.tensor(
        object_target,
        dtype=torch.long,
    )

    pred_obj_top1 = (
        object_pred_logits.argmax(dim=1)
        == object_target
    )
    pred_obj_top5 = _topk_bool(
        object_pred_logits,
        object_target,
        5,
    )

    gt_obj_top1 = (
        object_gt_logits.argmax(dim=1)
        == object_target
    )
    gt_obj_top5 = _topk_bool(
        object_gt_logits,
        object_target,
        5,
    )

    return {
        "pred_view_top1": pred_view_top1.numpy(),
        "pred_view_top5": pred_view_top5.numpy(),
        "gt_view_top1": gt_view_top1.numpy(),
        "gt_view_top5": gt_view_top5.numpy(),
        "pred_object_top1": pred_obj_top1.numpy(),
        "pred_object_top5": pred_obj_top5.numpy(),
        "gt_object_top1": gt_obj_top1.numpy(),
        "gt_object_top5": gt_obj_top5.numpy(),
        "pred_view_logits": pred_logits.numpy(),
        "gt_view_logits": gt_logits.numpy(),
        "categories": list(categories),
        "summary": {
            "num_categories": len(categories),
            "pred_view_top1": float(
                pred_view_top1.float().mean().item()
            ),
            "pred_view_top5": float(
                pred_view_top5.float().mean().item()
            ),
            "pred_object_top1": float(
                pred_obj_top1.float().mean().item()
            ),
            "pred_object_top5": float(
                pred_obj_top5.float().mean().item()
            ),
            "gt_view_top1": float(
                gt_view_top1.float().mean().item()
            ),
            "gt_view_top5": float(
                gt_view_top5.float().mean().item()
            ),
            "gt_object_top1": float(
                gt_obj_top1.float().mean().item()
            ),
            "gt_object_top5": float(
                gt_obj_top5.float().mean().item()
            ),
        },
    }


# ============================================================================
# IS / FID
# ============================================================================

@torch.no_grad()
def compute_is_fid(
    gt_paths,
    pred_paths,
    device,
    batch_size,
    is_splits,
):
    inception = InceptionScore(
        splits=is_splits,
        normalize=False,
    ).to(device)

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=False,
    ).to(device)

    for kind, paths in [
        ("gt", gt_paths),
        ("pred", pred_paths),
    ]:
        for s in tqdm(
            range(0, len(paths), batch_size),
            desc=kind,
            leave=False,
        ):
            x = torch.stack([
                image_to_uint8_tensor(
                    load_rgb(p),
                    (299, 299),
                )
                for p in paths[
                    s:s + batch_size
                ]
            ]).to(device)

            if kind == "gt":
                fid.update(
                    x,
                    real=True,
                )
            else:
                fid.update(
                    x,
                    real=False,
                )
                inception.update(x)

    is_mean, is_std = inception.compute()

    return {
        "is": float(
            is_mean.item()
        ),
        "is_std": float(
            is_std.item()
        ),
        "fid": float(
            fid.compute().item()
        ),
    }


# ============================================================================
# Optional checkpoint-level EEG semantic probe
# ============================================================================

class _SemanticClassifierOnly(nn.Module):
    def __init__(
        self,
        encoder_cls,
        num_classes=72,
    ):
        super().__init__()

        self.fmri_encoder = encoder_cls(
            num_electrodes=64,
            seq_len=600,
            embed_dim=1024,
        )

        self.semantic_cls_head = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, eeg):
        sem, _, _ = self.fmri_encoder(eeg)
        logits = self.semantic_cls_head(
            sem.float()
        )
        return logits, sem


def _strip_module(state):
    return {
        (
            k[len("module."):]
            if k.startswith("module.")
            else k
        ): v
        for k, v in state.items()
    }


def _load_semantic_model(
    ckpt_path,
    device,
):
    from src.mvdiffusion_var_semantic_cls import (
        EEG_Detangling_Disentanglement_Model,
    )

    model = _SemanticClassifierOnly(
        EEG_Detangling_Disentanglement_Model,
        num_classes=72,
    )

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
    )

    state = (
        ckpt["model"]
        if isinstance(ckpt, dict)
        and "model" in ckpt
        else ckpt
    )
    state = _strip_module(state)

    wanted = {
        k: v
        for k, v in state.items()
        if (
            k.startswith("fmri_encoder.")
            or k.startswith(
                "semantic_cls_head."
            )
        )
    }

    incompatible = model.load_state_dict(
        wanted,
        strict=False,
    )

    if (
        incompatible.missing_keys
        or incompatible.unexpected_keys
    ):
        raise RuntimeError(
            "Semantic checkpoint mismatch.\n"
            f"missing={incompatible.missing_keys[:20]}\n"
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )

    meta = {}
    if isinstance(ckpt, dict):
        meta = {
            "optimizer_step": ckpt.get(
                "optimizer_step",
                ckpt.get("global_step", None),
            ),
            "stage": ckpt.get(
                "stage",
                None,
            ),
            "training_args": ckpt.get(
                "args",
                {},
            ),
        }

    return model.to(device).eval(), meta


def _build_probe_dataset(
    args,
    train,
):
    from src.data.egg_dataset_ext_el import (
        AllDataFeatureTwoEEG,
    )

    data_path = str(
        Path(args.data_path)
        .expanduser()
        .resolve()
    )
    if not data_path.endswith(os.sep):
        data_path += os.sep

    return AllDataFeatureTwoEEG(
        data_path=data_path,
        sub_list=[args.sub_id],
        train=train,
        test_mean=False,
        num_frames=6,
        rendered_view_path=str(
            Path(args.rendered_view_path)
            .expanduser()
            .resolve()
        ),
        aug_data=False,
        strict_rendered_views=False,
    )


@torch.no_grad()
def _encode_eeg_raw_condition(
    model,
    dataset,
    mode,
    device,
    batch_size,
):
    raw = dataset.eeg_data

    if raw.ndim != 6:
        raise RuntimeError(
            f"Expected raw EEG [S,C,O,R,64,600], got {raw.shape}"
        )

    S, C, O, R, E, T = map(
        int,
        raw.shape,
    )

    indices = []

    for s in range(S):
        for c in range(C):
            for o in range(O):
                if mode == "individual":
                    for r in range(R):
                        indices.append(
                            (s, c, o, r)
                        )
                elif mode == "averaged":
                    indices.append(
                        (s, c, o, None)
                    )
                else:
                    raise ValueError(mode)

    feat = []
    logits = []
    y_all = []

    for start in range(
        0,
        len(indices),
        batch_size,
    ):
        chunk = indices[
            start:start + batch_size
        ]

        eeg_np = []
        y = []

        for s, c, o, r in chunk:
            if r is None:
                x = np.asarray(
                    raw[s, c, o, :],
                    dtype=np.float32,
                ).mean(
                    axis=0,
                    dtype=np.float32,
                )
            else:
                x = np.asarray(
                    raw[s, c, o, r],
                    dtype=np.float32,
                )

            eeg_np.append(x)
            y.append(c)

        eeg = torch.from_numpy(
            np.stack(eeg_np)
        ).to(
            device=device,
            dtype=torch.float32,
        )

        l, f = model(eeg)

        feat.append(
            f.float().cpu()
        )
        logits.append(
            l.float().cpu()
        )
        y_all.append(
            torch.tensor(
                y,
                dtype=torch.long,
            )
        )

    return (
        torch.cat(feat),
        torch.cat(logits),
        torch.cat(y_all),
    )


def _build_probe_prototypes(
    features,
    targets,
    num_classes=72,
):
    proto = []

    for c in range(num_classes):
        x = features[
            targets == c
        ].float()

        if len(x) == 0:
            raise RuntimeError(
                f"No train EEG for class {c}"
            )

        proto.append(
            F.normalize(
                x.mean(dim=0),
                dim=-1,
            )
        )

    return torch.stack(
        proto,
        dim=0,
    )


def _semantic_probe_metrics(
    features,
    learned_logits,
    targets,
    prototypes,
):
    proto_logits = (
        F.normalize(
            features.float(),
            dim=-1,
        )
        @ F.normalize(
            prototypes.float(),
            dim=-1,
        ).t()
    )

    rows = torch.arange(
        len(targets)
    )

    correct = proto_logits[
        rows,
        targets,
    ]

    wrong = proto_logits.clone()
    wrong[
        rows,
        targets,
    ] = -torch.inf

    best_wrong = wrong.max(
        dim=1,
    ).values

    margin = (
        correct - best_wrong
    )

    learned_top1 = (
        learned_logits.argmax(dim=1)
        == targets
    ).float().mean()

    learned_top5 = (
        learned_logits.topk(
            k=5,
            dim=1,
        ).indices
        == targets[:, None]
    ).any(dim=1).float().mean()

    proto_top1 = (
        proto_logits.argmax(dim=1)
        == targets
    ).float().mean()

    proto_top5 = (
        proto_logits.topk(
            k=5,
            dim=1,
        ).indices
        == targets[:, None]
    ).any(dim=1).float().mean()

    return {
        "learned_top1": float(
            learned_top1.item()
        ),
        "learned_top5": float(
            learned_top5.item()
        ),
        "prototype_top1": float(
            proto_top1.item()
        ),
        "prototype_top5": float(
            proto_top5.item()
        ),
        "prototype_correct_cosine": float(
            correct.mean().item()
        ),
        "prototype_margin": float(
            margin.mean().item()
        ),
        "prototype_positive_margin_rate": float(
            (margin > 0)
            .float()
            .mean()
            .item()
        ),
        "semantic_norm": float(
            features.norm(
                dim=-1
            ).mean().item()
        ),
    }


@torch.no_grad()
def run_final_eeg_semantic_probe(
    args,
    device,
):
    model, ckpt_meta = (
        _load_semantic_model(
            args.semantic_ckpt,
            device,
        )
    )

    train_ds = _build_probe_dataset(
        args,
        train=True,
    )
    test_ds = _build_probe_dataset(
        args,
        train=False,
    )

    ti_f, ti_l, ti_y = (
        _encode_eeg_raw_condition(
            model,
            train_ds,
            "individual",
            device,
            args.semantic_probe_batchsize,
        )
    )

    prototypes = _build_probe_prototypes(
        ti_f,
        ti_y,
        num_classes=72,
    )

    conditions = {
        "train_individual": (
            ti_f,
            ti_l,
            ti_y,
        ),
        "train_averaged": (
            *_encode_eeg_raw_condition(
                model,
                train_ds,
                "averaged",
                device,
                args.semantic_probe_batchsize,
            ),
        ),
        "test_individual": (
            *_encode_eeg_raw_condition(
                model,
                test_ds,
                "individual",
                device,
                args.semantic_probe_batchsize,
            ),
        ),
        "test_averaged": (
            *_encode_eeg_raw_condition(
                model,
                test_ds,
                "averaged",
                device,
                args.semantic_probe_batchsize,
            ),
        ),
    }

    result = {}

    for name, (
        feat,
        logits,
        target,
    ) in conditions.items():
        result[name] = (
            _semantic_probe_metrics(
                feat,
                logits,
                target,
                prototypes,
            )
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "checkpoint": str(
            Path(args.semantic_ckpt)
            .expanduser()
            .resolve()
        ),
        "checkpoint_meta": ckpt_meta,
        "train_raw_shape": list(
            map(
                int,
                train_ds.eeg_data.shape,
            )
        ),
        "test_raw_shape": list(
            map(
                int,
                test_ds.eeg_data.shape,
            )
        ),
        "conditions": result,
    }


# ============================================================================
# Aggregation helpers
# ============================================================================

def _macro_summary(
    df,
    metric_cols,
):
    out = {}

    for col in metric_cols:
        if col in df.columns:
            values = pd.to_numeric(
                df[col],
                errors="coerce",
            )
            out[col] = {
                "mean": float(
                    values.mean()
                ),
                "std": float(
                    values.std(ddof=0)
                ),
            }

    return out


def _json_float(value):
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if np.isnan(v):
        return None
    return v


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    out_dir = Path(
        args.out_dir
    ).expanduser().resolve()
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        objects,
        mapping_audit_df,
        resolved_pairs_df,
        pred_source,
    ) = load_eval_objects(args)

    mapping_audit_df.to_csv(
        out_dir / "mapping_audit.csv",
        index=False,
    )
    resolved_pairs_df.to_csv(
        out_dir / "resolved_pairs.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Flatten exact canonical view pairs.
    # ------------------------------------------------------------------
    gt_paths = []
    pred_paths = []
    obj_idx = []
    view_idx = []

    for oi, obj in enumerate(objects):
        for vi in range(args.num_views):
            gt_paths.append(
                obj.gt_views[vi]
            )
            pred_paths.append(
                obj.pred_views[vi]
            )
            obj_idx.append(oi)
            view_idx.append(vi)

    obj_idx = np.asarray(
        obj_idx,
        dtype=np.int64,
    )
    view_idx = np.asarray(
        view_idx,
        dtype=np.int64,
    )

    # ------------------------------------------------------------------
    # 1) CLIP image-image + optional category-space diagnostic.
    # ------------------------------------------------------------------
    print("[1/5] CLIP image embeddings / CLIPScore ...")

    clip_model = CLIPScorer(
        device
    )

    gt_clip = clip_model.encode_images(
        gt_paths,
        args.batch_size,
    )
    pred_clip = clip_model.encode_images(
        pred_paths,
        args.batch_size,
    )

    clip_pair = (
        gt_clip * pred_clip
    ).sum(
        dim=-1
    ).numpy()

    clip_category = None

    if not args.skip_clip_category:
        # Prefer explicit cls_index order when it is available and complete.
        known = [
            (o.cls_index, o.category)
            for o in objects
            if o.cls_index >= 0
        ]

        if known:
            class_map = {}
            for idx, name in known:
                class_map[idx] = name
            categories = [
                class_map[k]
                for k in sorted(class_map)
            ]
        else:
            categories = sorted(
                set(
                    o.category
                    for o in objects
                )
            )

        text_features = (
            clip_model.encode_categories(
                categories,
                args.clip_prompt_template,
            )
        )

        clip_category = (
            compute_clip_category_metrics(
                objects,
                gt_clip,
                pred_clip,
                obj_idx,
                text_features,
                categories,
            )
        )

        print(
            "[CLIP category] "
            f"categories={len(categories)}, "
            f"pred object Top-1="
            f"{clip_category['summary']['pred_object_top1']:.3f}, "
            f"GT sanity Top-1="
            f"{clip_category['summary']['gt_object_top1']:.3f}"
        )

    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 2) LPIPS
    # ------------------------------------------------------------------
    print("[2/5] LPIPS ...")

    lpips_model = LPIPSScorer(
        device
    )

    lpips_pair = (
        lpips_model.pair_scores(
            gt_paths,
            pred_paths,
            args.batch_size,
        )
    )

    del lpips_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3) Brain3D N-way
    # ------------------------------------------------------------------
    print(
        f"[3/5] N-way ({args.classifier}, "
        f"{args.nway_trials} trials) ..."
    )

    clf, clf_pre, classifier_categories = (
        build_classifier(
            args.classifier,
            device,
        )
    )

    gt_probs = classifier_probabilities(
        gt_paths,
        clf,
        clf_pre,
        device,
        args.batch_size,
    )

    pred_probs = classifier_probabilities(
        pred_paths,
        clf,
        clf_pre,
        device,
        args.batch_size,
    )

    (
        nway,
        nway_obj,
        trial_rows,
        gt_positive,
    ) = compute_all_nway(
        gt_probs,
        pred_probs,
        obj_idx,
        len(objects),
        args.nway_trials,
        args.seed,
    )

    del clf
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4) IS / FID
    # ------------------------------------------------------------------
    global_metrics = {
        "is": float("nan"),
        "is_std": float("nan"),
        "fid": float("nan"),
    }

    if not args.skip_is_fid:
        print("[4/5] IS / FID ...")

        fid_gt = (
            gt_paths
            if args.reference_mode == "six_view"
            else [
                o.gt_views[0]
                for o in objects
            ]
        )

        if len(fid_gt) < 2:
            warnings.warn(
                "FID with fewer than two real images "
                "is not meaningful."
            )

        global_metrics = compute_is_fid(
            fid_gt,
            pred_paths,
            device,
            args.batch_size,
            args.is_splits,
        )
    else:
        print("[4/5] IS/FID skipped")

    # ------------------------------------------------------------------
    # 5) Optional checkpoint EEG semantics.
    # ------------------------------------------------------------------
    eeg_semantic = None

    if args.semantic_ckpt:
        print("[5/5] Final EEG semantic checkpoint probe ...")

        eeg_semantic = (
            run_final_eeg_semantic_probe(
                args,
                device,
            )
        )

        primary = (
            eeg_semantic["conditions"][
                "test_individual"
            ]
        )

        print(
            "[EEG semantic] test individual: "
            f"prototype Top-1={primary['prototype_top1']:.3f}, "
            f"margin={primary['prototype_margin']:+.4f}"
        )
    else:
        print("[5/5] EEG semantic checkpoint probe skipped")

    # ------------------------------------------------------------------
    # Per-view table.
    # ------------------------------------------------------------------
    per_view_rows = []

    for i in range(len(gt_paths)):
        oi = int(obj_idx[i])
        obj = objects[oi]

        cls_id = int(
            gt_positive[i]
        )

        row = {
            "sample_id": obj.sample_id,
            "dataset_name": obj.dataset_name,
            "label": obj.label,
            "category": obj.category,
            "object_suffix": obj.object_suffix,
            "class_prefix": obj.class_prefix,
            "cls_index": obj.cls_index,
            "obj_index": obj.obj_index,
            "trial_index": obj.trial_index,
            "subject_index": obj.subject_index,
            "view": int(view_idx[i]),
            "gt_path": gt_paths[i],
            "pred_path": pred_paths[i],
            "clipscore": float(
                clip_pair[i]
            ),
            "lpips": float(
                lpips_pair[i]
            ),
            "gt_classifier_id": cls_id,
            "gt_classifier_name": (
                classifier_categories[
                    cls_id
                ]
            ),
        }

        if clip_category is not None:
            row.update({
                "clip_category_pred_top1": int(
                    clip_category[
                        "pred_view_top1"
                    ][i]
                ),
                "clip_category_pred_top5": int(
                    clip_category[
                        "pred_view_top5"
                    ][i]
                ),
                "clip_category_gt_top1": int(
                    clip_category[
                        "gt_view_top1"
                    ][i]
                ),
                "clip_category_gt_top5": int(
                    clip_category[
                        "gt_view_top5"
                    ][i]
                ),
            })

        per_view_rows.append(row)

    per_view_df = pd.DataFrame(
        per_view_rows
    )
    per_view_df.to_csv(
        out_dir / "per_view_metrics.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Per-object table.
    # ------------------------------------------------------------------
    object_rows = []

    for oi, obj in enumerate(objects):
        mask = (
            obj_idx == oi
        )

        gt_classes = (
            gt_positive[mask]
        )

        unique, counts = np.unique(
            gt_classes,
            return_counts=True,
        )

        majority = int(
            unique[
                np.argmax(counts)
            ]
        )

        row = {
            "sample_id": obj.sample_id,
            "dataset_name": obj.dataset_name,
            "label": obj.label,
            "category": obj.category,
            "object_suffix": obj.object_suffix,
            "class_prefix": obj.class_prefix,
            "cls_index": obj.cls_index,
            "obj_index": obj.obj_index,
            "trial_index": obj.trial_index,
            "subject_index": obj.subject_index,
            "pred_source": obj.pred_source,

            "2way_top1": float(
                nway_obj[
                    "2way_top1"
                ][oi]
            ),
            "10way_top1": float(
                nway_obj[
                    "10way_top1"
                ][oi]
            ),
            "10way_top2": float(
                nway_obj[
                    "10way_top2"
                ][oi]
            ),
            "50way_top1": float(
                nway_obj[
                    "50way_top1"
                ][oi]
            ),
            "50way_top2": float(
                nway_obj[
                    "50way_top2"
                ][oi]
            ),
            "clipscore": float(
                clip_pair[mask].mean()
            ),
            "lpips": float(
                lpips_pair[mask].mean()
            ),
            "gt_classifier_majority_id": majority,
            "gt_classifier_majority_name": (
                classifier_categories[
                    majority
                ]
            ),
            "gt_classifier_agreement": float(
                counts.max()
                / len(gt_classes)
            ),
        }

        if clip_category is not None:
            row.update({
                "clip_category_pred_view_top1": float(
                    clip_category[
                        "pred_view_top1"
                    ][mask].mean()
                ),
                "clip_category_pred_view_top5": float(
                    clip_category[
                        "pred_view_top5"
                    ][mask].mean()
                ),
                "clip_category_gt_view_top1": float(
                    clip_category[
                        "gt_view_top1"
                    ][mask].mean()
                ),
                "clip_category_gt_view_top5": float(
                    clip_category[
                        "gt_view_top5"
                    ][mask].mean()
                ),
                "clip_category_pred_object_top1": int(
                    clip_category[
                        "pred_object_top1"
                    ][oi]
                ),
                "clip_category_pred_object_top5": int(
                    clip_category[
                        "pred_object_top5"
                    ][oi]
                ),
                "clip_category_gt_object_top1": int(
                    clip_category[
                        "gt_object_top1"
                    ][oi]
                ),
                "clip_category_gt_object_top5": int(
                    clip_category[
                        "gt_object_top5"
                    ][oi]
                ),
            })

        object_rows.append(row)

    object_df = pd.DataFrame(
        object_rows
    )
    object_df.to_csv(
        out_dir / "per_object_metrics.csv",
        index=False,
    )

    pd.DataFrame(
        trial_rows
    ).to_csv(
        out_dir / "nway_trials.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Per-category aggregation.
    # This directly reflects the 72-way semantic structure used in training.
    # ------------------------------------------------------------------
    numeric_object_cols = [
        "2way_top1",
        "10way_top1",
        "10way_top2",
        "50way_top1",
        "50way_top2",
        "clipscore",
        "lpips",
        "gt_classifier_agreement",
        "clip_category_pred_view_top1",
        "clip_category_pred_view_top5",
        "clip_category_gt_view_top1",
        "clip_category_gt_view_top5",
        "clip_category_pred_object_top1",
        "clip_category_pred_object_top5",
        "clip_category_gt_object_top1",
        "clip_category_gt_object_top5",
    ]

    existing_numeric = [
        c
        for c in numeric_object_cols
        if c in object_df.columns
    ]

    category_agg = {
        c: "mean"
        for c in existing_numeric
    }
    category_agg["sample_id"] = "count"

    per_category_df = (
        object_df
        .groupby(
            ["cls_index", "category"],
            dropna=False,
        )
        .agg(category_agg)
        .reset_index()
        .rename(
            columns={
                "sample_id": "num_objects"
            }
        )
    )

    per_category_df.to_csv(
        out_dir / "per_category_metrics.csv",
        index=False,
    )

    # Object-suffix aggregation: useful for 08 vs 09 unseen instances.
    if object_df[
        "object_suffix"
    ].astype(str).str.len().gt(0).any():
        suffix_rows = []

        for suffix, sdf in object_df.groupby(
            "object_suffix"
        ):
            if not str(suffix):
                continue

            r = {
                "object_suffix": suffix,
                "num_objects": len(sdf),
            }

            for c in existing_numeric:
                r[c] = float(
                    pd.to_numeric(
                        sdf[c],
                        errors="coerce",
                    ).mean()
                )

            suffix_rows.append(r)

        suffix_df = pd.DataFrame(
            suffix_rows
        )
        suffix_df.to_csv(
            out_dir / "per_object_suffix_metrics.csv",
            index=False,
        )
    else:
        suffix_df = pd.DataFrame()

    # ------------------------------------------------------------------
    # Brain3D-compatible table row.
    # ------------------------------------------------------------------
    result = {
        "Backbone": "Ours",
        "2-way Top-1": (
            nway["2way_top1"]["mean"]
        ),
        "10-way Top-1": (
            nway["10way_top1"]["mean"]
        ),
        "10-way Top-2": (
            nway["10way_top2"]["mean"]
        ),
        "50-way Top-1": (
            nway["50way_top1"]["mean"]
        ),
        "50-way Top-2": (
            nway["50way_top2"]["mean"]
        ),
        "CLIPScore": float(
            object_df[
                "clipscore"
            ].mean()
        ),
        "IS": global_metrics["is"],
        "FID": global_metrics["fid"],
        "LPIPS": float(
            object_df[
                "lpips"
            ].mean()
        ),
    }

    pd.DataFrame([
        result
    ]).to_csv(
        out_dir / "brain3d_table_row.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    category_macro = _macro_summary(
        per_category_df,
        existing_numeric,
    )

    summary = {
        "protocol": {
            "pairs_csv": str(
                Path(args.pairs_csv)
                .expanduser()
                .resolve()
            ),
            "prediction_source": pred_source,
            "reference_mode": args.reference_mode,
            "num_objects": len(objects),
            "num_categories": len(
                set(
                    o.category
                    for o in objects
                )
            ),
            "num_views_per_object": args.num_views,
            "classifier": args.classifier,
            "nway_trials": args.nway_trials,
            "seed": args.seed,
            "strict_mapping": bool(
                args.strict_mapping
            ),
        },
        "mapping_audit": {
            "rows": len(mapping_audit_df),
            "label_matches_name_rate": float(
                mapping_audit_df[
                    "label_matches_name"
                ].mean()
            )
            if len(mapping_audit_df)
            else None,
            "prefix_matches_cls_index_rate": float(
                mapping_audit_df[
                    "prefix_matches_cls_index"
                ].mean()
            )
            if len(mapping_audit_df)
            else None,
            "test_suffix_08_09_rate": float(
                mapping_audit_df[
                    "test_suffix_ok"
                ].mean()
            )
            if len(mapping_audit_df)
            else None,
        },
        "brain3d_metrics": {
            "2way_top1": nway[
                "2way_top1"
            ],
            "10way_top1": nway[
                "10way_top1"
            ],
            "10way_top2": nway[
                "10way_top2"
            ],
            "50way_top1": nway[
                "50way_top1"
            ],
            "50way_top2": nway[
                "50way_top2"
            ],
            "clipscore": {
                "mean": float(
                    object_df[
                        "clipscore"
                    ].mean()
                ),
                "std": float(
                    object_df[
                        "clipscore"
                    ].std(ddof=0)
                ),
            },
            "is": {
                "mean": _json_float(
                    global_metrics["is"]
                ),
                "std": _json_float(
                    global_metrics["is_std"]
                ),
            },
            "fid": _json_float(
                global_metrics["fid"]
            ),
            "lpips": {
                "mean": float(
                    object_df[
                        "lpips"
                    ].mean()
                ),
                "std": float(
                    object_df[
                        "lpips"
                    ].std(ddof=0)
                ),
            },
        },
        "category_macro_metrics": category_macro,
        "clip_category_diagnostic": (
            clip_category["summary"]
            if clip_category is not None
            else None
        ),
        "eeg_semantic_probe": eeg_semantic,
    }

    if len(suffix_df):
        summary[
            "object_suffix_metrics"
        ] = (
            suffix_df.to_dict(
                orient="records"
            )
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

    if args.save_reference_table:
        cols = [
            "Backbone",
            "2-way Top-1",
            "10-way Top-1",
            "10-way Top-2",
            "50-way Top-1",
            "50-way Top-2",
            "CLIPScore",
            "IS",
            "FID",
            "LPIPS",
        ]

        ref_rows = []

        for name, vals in (
            BRAIN3D_REFERENCE.items()
        ):
            ref_rows.append(
                dict(
                    zip(
                        cols,
                        [name] + vals,
                    )
                )
            )

        pd.concat(
            [
                pd.DataFrame(ref_rows),
                pd.DataFrame([result]),
            ],
            ignore_index=True,
        ).to_csv(
            out_dir
            / "brain3d_comparison_table.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------
    print("\n" + "=" * 118)
    print(
        f"{'Method':<10}"
        f"{'2w-T1':>9}"
        f"{'10w-T1':>10}"
        f"{'10w-T2':>10}"
        f"{'50w-T1':>10}"
        f"{'50w-T2':>10}"
        f"{'CLIP':>10}"
        f"{'IS':>10}"
        f"{'FID':>11}"
        f"{'LPIPS':>10}"
    )
    print("-" * 118)
    print(
        f"{'Ours':<10}"
        f"{result['2-way Top-1']:>9.3f}"
        f"{result['10-way Top-1']:>10.3f}"
        f"{result['10-way Top-2']:>10.3f}"
        f"{result['50-way Top-1']:>10.3f}"
        f"{result['50-way Top-2']:>10.3f}"
        f"{result['CLIPScore']:>10.3f}"
        f"{result['IS']:>10.3f}"
        f"{result['FID']:>11.3f}"
        f"{result['LPIPS']:>10.3f}"
    )
    print("=" * 118)

    if clip_category is not None:
        cs = clip_category[
            "summary"
        ]
        print(
            "[semantic output] "
            f"CLIP category object Top-1="
            f"{cs['pred_object_top1']:.3f}, "
            f"Top-5={cs['pred_object_top5']:.3f}, "
            f"GT sanity Top-1={cs['gt_object_top1']:.3f}"
        )

    print(
        "[mapping] "
        f"label/name="
        f"{summary['mapping_audit']['label_matches_name_rate']:.3f}, "
        f"prefix/cls="
        f"{summary['mapping_audit']['prefix_matches_cls_index_rate']:.3f}"
    )

    print(f"Saved to: {out_dir}")
    print(f"  summary                 : {out_dir / 'summary.json'}")
    print(f"  per-view                : {out_dir / 'per_view_metrics.csv'}")
    print(f"  per-object              : {out_dir / 'per_object_metrics.csv'}")
    print(f"  per-category            : {out_dir / 'per_category_metrics.csv'}")
    print(f"  resolved pairs          : {out_dir / 'resolved_pairs.csv'}")
    print(f"  mapping audit           : {out_dir / 'mapping_audit.csv'}")


if __name__ == "__main__":
    main()