#!/usr/bin/env python3
"""
Brain3D-style evaluation for reconstructed 3D mesh renders.

Metrics:
  - 2-way Top-1 accuracy
  - 10-way Top-1 / Top-2 accuracy
  - 50-way Top-1 / Top-2 accuracy
  - CLIPScore (CLIP ViT-B/16 image-image cosine similarity)
  - Inception Score (IS)
  - Fréchet Inception Distance (FID)
  - LPIPS (AlexNet)

Reference modes
---------------
six_view (recommended for the current Neuro-3D pipeline):
    GT/00.png <-> Pred/00.png, ..., GT/05.png <-> Pred/05.png

single (Brain3D paper protocol):
    one GT stimulus image x is compared with all six reconstructed mesh views.

Expected CSV for six_view mode (mesh_pairs.csv):
    sample_id,label,...,pred_mesh_view_dir,gt_view_dir,...

Dependencies:
    pip install open_clip_torch lpips torchmetrics torch-fidelity pandas pillow tqdm

Example:
    python evaluate_brain3d_mesh.py \
        --pairs_csv ./mesh_results/mesh_pairs.csv \
        --out_dir ./mesh_results/evaluation \
        --reference_mode six_view \
        --device cuda \
        --batch_size 16 \
        --nway_trials 100
"""

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
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
    raise ImportError("Install open_clip_torch: pip install open_clip_torch") from e

try:
    import lpips
except ImportError as e:
    raise ImportError("Install lpips: pip install lpips") from e

try:
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError as e:
    raise ImportError(
        "Install torchmetrics + torch-fidelity: pip install torchmetrics torch-fidelity"
    ) from e


BRAIN3D_REFERENCE = {
    "BrainVis": [0.880, 0.706, 0.796, 0.578, 0.649, 0.617, 16.590, 204.015, 0.789],
    "DreamDiffusion": [0.730, 0.314, 0.480, 0.164, 0.206, 0.564, 14.871, 232.256, 0.780],
    "EEG-CLIP": [0.857, 0.655, 0.742, 0.545, 0.602, 0.608, 17.173, 156.631, 0.788],
    "GWIT": [0.946, 0.854, 0.906, 0.763, 0.822, 0.648, 17.195, 153.295, 0.783],
}


@dataclass
class EvalObject:
    sample_id: str
    label: str
    pred_views: List[str]
    gt_views: List[str]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", default="./brain3d_mesh_eval")
    p.add_argument("--reference_mode", choices=["six_view", "single"], default="six_view")
    p.add_argument("--pred_dir_col", default="pred_mesh_view_dir")
    p.add_argument("--gt_dir_col", default="gt_view_dir")
    p.add_argument("--gt_image_col", default="gt_image")
    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--classifier", choices=["resnet50", "resnet18", "vit_b_16"], default="resnet50")
    p.add_argument("--nway_trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--skip_is_fid", action="store_true")
    p.add_argument("--save_reference_table", action="store_true")
    return p.parse_args()


def load_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def canonical_view_paths(directory: str, num_views: int = 6) -> List[str]:
    paths = [os.path.join(directory, f"{i:02d}.png") for i in range(num_views)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing views under {directory}: " + ", ".join(os.path.basename(x) for x in missing)
        )
    return paths


def image_to_unit_tensor(image: Image.Image, size: Tuple[int, int] = None) -> torch.Tensor:
    x = TF.to_tensor(image.convert("RGB"))
    if size is not None and tuple(x.shape[-2:]) != tuple(size):
        x = TF.resize(
            x, list(size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True
        )
    return x.clamp(0, 1)


def image_to_uint8_tensor(image: Image.Image, size=(299, 299)) -> torch.Tensor:
    x = image_to_unit_tensor(image, size=size)
    return (x * 255.0).round().to(torch.uint8)


def load_eval_objects(args) -> List[EvalObject]:
    csv_path = Path(args.pairs_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    required = {"sample_id", args.pred_dir_col}
    required.add(args.gt_dir_col if args.reference_mode == "six_view" else args.gt_image_col)
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing CSV columns: {missing}; available={list(df.columns)}")

    objects, seen = [], set()
    for row_idx, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()
        label = str(row["label"]).strip() if "label" in df.columns else sample_id
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen.add(sample_id)

        pred_views = canonical_view_paths(str(row[args.pred_dir_col]).strip(), args.num_views)
        if args.reference_mode == "six_view":
            gt_views = canonical_view_paths(str(row[args.gt_dir_col]).strip(), args.num_views)
        else:
            gt_image = str(row[args.gt_image_col]).strip()
            if not os.path.isfile(gt_image):
                raise FileNotFoundError(gt_image)
            gt_views = [gt_image] * args.num_views

        objects.append(EvalObject(sample_id, label, pred_views, gt_views))

    if not objects:
        raise RuntimeError("No evaluation samples found.")
    print(f"[data] objects={len(objects)}, views/object={args.num_views}, mode={args.reference_mode}")
    return objects


class CLIPImageScorer:
    def __init__(self, device):
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="openai"
        )
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def encode(self, paths: Sequence[str], batch_size: int) -> torch.Tensor:
        out = []
        for s in range(0, len(paths), batch_size):
            x = torch.stack([self.preprocess(load_rgb(p)) for p in paths[s:s+batch_size]]).to(self.device)
            f = F.normalize(self.model.encode_image(x).float(), dim=-1)
            out.append(f.cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def pair_scores(self, gt_paths, pred_paths, batch_size):
        g = self.encode(gt_paths, batch_size)
        p = self.encode(pred_paths, batch_size)
        return (g * p).sum(dim=-1).numpy()


class LPIPSScorer:
    def __init__(self, device):
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def pair_scores(self, gt_paths, pred_paths, batch_size):
        values = []
        for s in range(0, len(gt_paths), batch_size):
            gb, pb = [], []
            for gp, pp in zip(gt_paths[s:s+batch_size], pred_paths[s:s+batch_size]):
                gi, pi = load_rgb(gp), load_rgb(pp)
                target_hw = (gi.height, gi.width)
                g = image_to_unit_tensor(gi)
                p = image_to_unit_tensor(pi, target_hw)
                gb.append(g * 2 - 1)
                pb.append(p * 2 - 1)
            g = torch.stack(gb).to(self.device)
            p = torch.stack(pb).to(self.device)
            values.extend(self.model(g, p).reshape(-1).float().cpu().tolist())
        return np.asarray(values, dtype=np.float64)


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
    return model.to(device).eval(), weights.transforms(), weights.meta["categories"]


@torch.no_grad()
def classifier_probabilities(paths, model, preprocess, device, batch_size):
    out = []
    for s in range(0, len(paths), batch_size):
        x = torch.stack([preprocess(load_rgb(p)) for p in paths[s:s+batch_size]]).to(device)
        out.append(torch.softmax(model(x).float(), dim=-1).cpu())
    return torch.cat(out, dim=0).numpy()


def nway_trials_for_pair(pred_prob, positive_class, n_way, top_k, trials, rng):
    K = pred_prob.shape[0]
    neg_pool = np.concatenate([
        np.arange(0, positive_class, dtype=np.int64),
        np.arange(positive_class + 1, K, dtype=np.int64),
    ])
    success = np.zeros(trials, dtype=np.float64)
    for t in range(trials):
        neg = rng.choice(neg_pool, size=n_way - 1, replace=False)
        candidates = np.concatenate([[positive_class], neg])
        scores = pred_prob[candidates]
        rank = np.argsort(-scores)
        positive_rank = int(np.where(rank == 0)[0][0])
        success[t] = float(positive_rank < top_k)
    return success


def compute_all_nway(gt_probs, pred_probs, object_index, num_objects, trials, seed):
    settings = [
        ("2way_top1", 2, 1),
        ("10way_top1", 10, 1),
        ("10way_top2", 10, 2),
        ("50way_top1", 50, 1),
        ("50way_top2", 50, 2),
    ]
    positive = gt_probs.argmax(axis=1)
    summary, per_object, trial_rows = {}, {}, []

    for setting_idx, (name, n_way, top_k) in enumerate(settings):
        pair_trial = np.zeros((len(pred_probs), trials), dtype=np.float64)
        for pair_idx in range(len(pred_probs)):
            rng = np.random.default_rng(seed + setting_idx * 1_000_003 + pair_idx * 9_973)
            pair_trial[pair_idx] = nway_trials_for_pair(
                pred_probs[pair_idx], int(positive[pair_idx]), n_way, top_k, trials, rng
            )

        obj_scores = np.zeros(num_objects, dtype=np.float64)
        for oi in range(num_objects):
            obj_scores[oi] = pair_trial[object_index == oi].mean()
        per_object[name] = obj_scores

        global_trials = []
        for t in range(trials):
            vals = [pair_trial[object_index == oi, t].mean() for oi in range(num_objects)]
            v = float(np.mean(vals))
            global_trials.append(v)
            trial_rows.append({"metric": name, "trial": t, "value": v})
        global_trials = np.asarray(global_trials)
        summary[name] = {
            "mean": float(global_trials.mean()),
            "std": float(global_trials.std(ddof=0)),
        }

    return summary, per_object, trial_rows, positive


@torch.no_grad()
def compute_is_fid(gt_paths, pred_paths, device, batch_size, is_splits):
    inception = InceptionScore(splits=is_splits, normalize=False).to(device)
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    for kind, paths in [("gt", gt_paths), ("pred", pred_paths)]:
        for s in tqdm(range(0, len(paths), batch_size), desc=kind, leave=False):
            x = torch.stack([
                image_to_uint8_tensor(load_rgb(p), (299, 299))
                for p in paths[s:s+batch_size]
            ]).to(device)
            if kind == "gt":
                fid.update(x, real=True)
            else:
                fid.update(x, real=False)
                inception.update(x)

    is_mean, is_std = inception.compute()
    return {
        "is": float(is_mean.item()),
        "is_std": float(is_std.item()),
        "fid": float(fid.compute().item()),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects = load_eval_objects(args)

    gt_paths, pred_paths, obj_idx, view_idx = [], [], [], []
    for oi, obj in enumerate(objects):
        for vi in range(args.num_views):
            gt_paths.append(obj.gt_views[vi])
            pred_paths.append(obj.pred_views[vi])
            obj_idx.append(oi)
            view_idx.append(vi)
    obj_idx = np.asarray(obj_idx, dtype=np.int64)
    view_idx = np.asarray(view_idx, dtype=np.int64)

    print("[1/4] CLIPScore ...")
    clip_model = CLIPImageScorer(device)
    clip_pair = clip_model.pair_scores(gt_paths, pred_paths, args.batch_size)
    del clip_model
    if device.type == "cuda": torch.cuda.empty_cache()

    print("[2/4] LPIPS ...")
    lpips_model = LPIPSScorer(device)
    lpips_pair = lpips_model.pair_scores(gt_paths, pred_paths, args.batch_size)
    del lpips_model
    if device.type == "cuda": torch.cuda.empty_cache()

    print(f"[3/4] N-way ({args.classifier}, {args.nway_trials} trials) ...")
    clf, clf_pre, categories = build_classifier(args.classifier, device)
    gt_probs = classifier_probabilities(gt_paths, clf, clf_pre, device, args.batch_size)
    pred_probs = classifier_probabilities(pred_paths, clf, clf_pre, device, args.batch_size)
    nway, nway_obj, trial_rows, gt_positive = compute_all_nway(
        gt_probs, pred_probs, obj_idx, len(objects), args.nway_trials, args.seed
    )
    del clf
    if device.type == "cuda": torch.cuda.empty_cache()

    global_metrics = {"is": float("nan"), "is_std": float("nan"), "fid": float("nan")}
    if not args.skip_is_fid:
        print("[4/4] IS / FID ...")
        fid_gt = gt_paths if args.reference_mode == "six_view" else [o.gt_views[0] for o in objects]
        if len(fid_gt) < 2:
            warnings.warn("FID with fewer than two real images is not meaningful.")
        global_metrics = compute_is_fid(fid_gt, pred_paths, device, args.batch_size, args.is_splits)
    else:
        print("[4/4] IS/FID skipped")

    object_rows = []
    for oi, obj in enumerate(objects):
        mask = obj_idx == oi
        gt_classes = gt_positive[mask]
        unique, counts = np.unique(gt_classes, return_counts=True)
        maj = int(unique[np.argmax(counts)])
        row = {
            "sample_id": obj.sample_id,
            "label": obj.label,
            "2way_top1": float(nway_obj["2way_top1"][oi]),
            "10way_top1": float(nway_obj["10way_top1"][oi]),
            "10way_top2": float(nway_obj["10way_top2"][oi]),
            "50way_top1": float(nway_obj["50way_top1"][oi]),
            "50way_top2": float(nway_obj["50way_top2"][oi]),
            "clipscore": float(clip_pair[mask].mean()),
            "lpips": float(lpips_pair[mask].mean()),
            "gt_classifier_majority_id": maj,
            "gt_classifier_majority_name": categories[maj],
            "gt_classifier_agreement": float(counts.max() / len(gt_classes)),
        }
        object_rows.append(row)

    object_df = pd.DataFrame(object_rows)
    object_df.to_csv(out_dir / "per_object_metrics.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(out_dir / "nway_trials.csv", index=False)

    per_view_rows = []
    for i in range(len(gt_paths)):
        cls_id = int(gt_positive[i])
        per_view_rows.append({
            "sample_id": objects[obj_idx[i]].sample_id,
            "label": objects[obj_idx[i]].label,
            "view": int(view_idx[i]),
            "gt_path": gt_paths[i],
            "pred_path": pred_paths[i],
            "clipscore": float(clip_pair[i]),
            "lpips": float(lpips_pair[i]),
            "gt_classifier_id": cls_id,
            "gt_classifier_name": categories[cls_id],
        })
    pd.DataFrame(per_view_rows).to_csv(out_dir / "per_view_metrics.csv", index=False)

    result = {
        "Backbone": "Ours",
        "2-way Top-1": nway["2way_top1"]["mean"],
        "10-way Top-1": nway["10way_top1"]["mean"],
        "10-way Top-2": nway["10way_top2"]["mean"],
        "50-way Top-1": nway["50way_top1"]["mean"],
        "50-way Top-2": nway["50way_top2"]["mean"],
        "CLIPScore": float(object_df["clipscore"].mean()),
        "IS": global_metrics["is"],
        "FID": global_metrics["fid"],
        "LPIPS": float(object_df["lpips"].mean()),
    }
    pd.DataFrame([result]).to_csv(out_dir / "brain3d_table_row.csv", index=False)

    summary = {
        "num_objects": len(objects),
        "num_views_per_object": args.num_views,
        "reference_mode": args.reference_mode,
        "classifier": args.classifier,
        "nway_trials": args.nway_trials,
        "metrics": {
            "2way_top1": nway["2way_top1"],
            "10way_top1": nway["10way_top1"],
            "10way_top2": nway["10way_top2"],
            "50way_top1": nway["50way_top1"],
            "50way_top2": nway["50way_top2"],
            "clipscore": {
                "mean": float(object_df["clipscore"].mean()),
                "std": float(object_df["clipscore"].std(ddof=0)),
            },
            "is": {"mean": global_metrics["is"], "std": global_metrics["is_std"]},
            "fid": global_metrics["fid"],
            "lpips": {
                "mean": float(object_df["lpips"].mean()),
                "std": float(object_df["lpips"].std(ddof=0)),
            },
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if args.save_reference_table:
        cols = [
            "Backbone", "2-way Top-1", "10-way Top-1", "10-way Top-2",
            "50-way Top-1", "50-way Top-2", "CLIPScore", "IS", "FID", "LPIPS"
        ]
        ref_rows = []
        for name, vals in BRAIN3D_REFERENCE.items():
            ref_rows.append(dict(zip(cols, [name] + vals)))
        pd.concat([pd.DataFrame(ref_rows), pd.DataFrame([result])], ignore_index=True).to_csv(
            out_dir / "brain3d_comparison_table.csv", index=False
        )

    print("\n" + "=" * 110)
    print(
        f"{'Method':<10}{'2w-T1':>9}{'10w-T1':>10}{'10w-T2':>10}"
        f"{'50w-T1':>10}{'50w-T2':>10}{'CLIP':>10}{'IS':>10}{'FID':>11}{'LPIPS':>10}"
    )
    print("-" * 110)
    print(
        f"{'Ours':<10}{result['2-way Top-1']:>9.3f}{result['10-way Top-1']:>10.3f}"
        f"{result['10-way Top-2']:>10.3f}{result['50-way Top-1']:>10.3f}"
        f"{result['50-way Top-2']:>10.3f}{result['CLIPScore']:>10.3f}"
        f"{result['IS']:>10.3f}{result['FID']:>11.3f}{result['LPIPS']:>10.3f}"
    )
    print("=" * 110)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()