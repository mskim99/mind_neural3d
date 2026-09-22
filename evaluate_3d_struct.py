#!/usr/bin/env python3
"""
Evaluation aligned to the CURRENT Neuro-3D training + inference pipeline.
Includes Textural-Level (PSNR, SSIM) and Structure-Level (FPD, CD, EMD) metrics.
"""
import argparse
import json
import os
import random
import re
import warnings
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights,
    vit_b_16, ViT_B_16_Weights,
)

# Textural-Level Metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_metric
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError as e:
    raise ImportError("Install scikit-image: pip install scikit-image") from e

# Structure-Level Metrics
try:
    import trimesh
    import ot  # Python Optimal Transport (POT) for EMD
except ImportError as e:
    raise ImportError("Install trimesh and POT: pip install trimesh pot") from e

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
    raise ImportError("Install torchmetrics + torch-fidelity: pip install torchmetrics torch-fidelity") from e

BRAIN3D_REFERENCE = {
    "MinD-3D++": [
        0.887, 0.616, 3.025, 1.635, 3.672,
        0.234, 16.44, 0.763,
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
    mesh_path: str = ""
    gt_mesh_path: str = ""


def parse_args():
    p = argparse.ArgumentParser(description="Neuro-3D / Brain3D aligned 3D evaluation with FPD, CD, EMD, PSNR, SSIM")

    p.add_argument("--pairs_csv", required=True)
    p.add_argument("--out_dir", default="./brain3d_mesh_eval")
    p.add_argument("--target", choices=["auto", "generated", "mesh"], default="auto")
    p.add_argument("--reference_mode", choices=["six_view", "single"], default="six_view")

    p.add_argument("--pred_dir_col", default="")
    p.add_argument("--gt_dir_col", default="")
    p.add_argument("--gt_image_col", default="gt_image")

    p.add_argument("--mesh_root", default="")
    p.add_argument("--mesh_key", choices=["label", "sample_id"], default="label")

    # 3D Structure Evaluation Arguments
    p.add_argument("--gt_mesh_root", default="",
                   help="Root directory for original Ground Truth .obj meshes to compute CD and EMD.")
    p.add_argument("--skip_3d_metrics", action="store_true",
                   help="Skip CD, EMD, FPD calculation if GT meshes are missing.")
    p.add_argument("--num_mesh_samples", type=int, default=2048,
                   help="Number of points to sample from each mesh for FPD/CD/EMD.")
    p.add_argument("--fpd_pointnet_ckpt", default="",
                   help="Pretrained TreeGAN PointNet checkpoint. If omitted, the default checkpoint is downloaded.")
    p.add_argument("--fpd_pointnet_url", default=(
        "https://github.com/junzhezhang/shape-inversion/raw/"
        "a1176778330e22546ee81dc01e93c0b1e9e7a37d/evaluation/cls_model_39.pth"
    ))
    p.add_argument("--fpd_batch_size", type=int, default=32,
                   help="Batch size used for PointNet feature extraction.")

    p.add_argument("--num_views", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--classifier", choices=["resnet50", "resnet18", "vit_b_16"], default="resnet50")
    p.add_argument("--nway_trials", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--is_splits", type=int, default=10)
    p.add_argument("--skip_is_fid", action="store_true")
    p.add_argument("--save_reference_table", action="store_true")

    p.add_argument("--strict_mapping", action="store_true")
    p.add_argument("--strict_test_suffix", action="store_true")
    p.add_argument("--skip_clip_category", action="store_true")
    p.add_argument("--clip_prompt_template", default="a photo of a {}")

    p.add_argument("--semantic_ckpt", default="")
    p.add_argument("--data_path", default="/data/jionkim/neuro_3D")
    p.add_argument("--rendered_view_path", default="/data/jionkim/neuro_3D/render_grid_v4")
    p.add_argument("--sub_id", default="sub01")
    p.add_argument("--semantic_probe_batchsize", type=int, default=64)

    return p.parse_args()


def _safe_int(value, default=-1):
    if value is None: return default
    try:
        if pd.isna(value): return default
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return default


def category_from_label(label: str) -> str:
    label = str(label).strip()
    m = re.match(r"^(.*)_([0-9]{2})$", label)
    return m.group(1) if m else label


def object_suffix_from_label(label: str) -> str:
    m = re.match(r"^.*_([0-9]{2})$", str(label).strip())
    return m.group(1) if m else ""


def load_rgb(path: str) -> Image.Image:
    if not os.path.isfile(path): raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def canonical_view_paths(directory: str, num_views: int = 6) -> List[str]:
    directory = str(Path(directory).expanduser())
    paths = [os.path.join(directory, f"{i:02d}.png") for i in range(num_views)]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing views: {', '.join(os.path.basename(x) for x in missing)}")
    return paths


def resolve_prediction_source(args, df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    # ... (생략 없이 이전 코드 유지)
    generated_col = "pred_dir" if "pred_dir" in df.columns else None
    mesh_col = "pred_mesh_view_dir" if "pred_mesh_view_dir" in df.columns else None

    if args.target == "mesh": return "mesh", mesh_col
    if args.target == "generated": return "generated", generated_col
    if mesh_col: return "mesh", mesh_col
    return "generated", generated_col


def resolve_gt_source(args, df):
    col = "gt_view_dir" if "gt_view_dir" in df.columns else None
    return col, None


def load_eval_objects(args):
    csv_path = Path(args.pairs_csv).expanduser().resolve()
    df = pd.read_csv(csv_path)

    pred_source, pred_col = resolve_prediction_source(args, df)
    gt_dir_col, gt_image_col = resolve_gt_source(args, df)

    objects = []
    seen_ids = set()

    for row_idx, row in df.iterrows():
        sample_id = str(row["sample_id"]).strip()
        if sample_id in seen_ids: continue
        seen_ids.add(sample_id)

        label = str(row["label"]).strip() if "label" in df.columns else sample_id
        category = category_from_label(label)
        suffix = object_suffix_from_label(label)

        pred_dir = str(row[pred_col]).strip() if pred_col and pred_col in row.index else str(
            csv_path.parent / "views" / sample_id)

        # -------------------------------------------------------------
        # [PATCH] 1. Predicted Mesh Path 동적 탐색 (폴더 내부 검색)
        # -------------------------------------------------------------
        mesh_path = ""
        if args.mesh_root:
            # 예상 경로: mesh_root / sample_id (예: .../mesh/wineglass_07__trial01)
            pred_mesh_dir = Path(args.mesh_root) / sample_id
            if pred_mesh_dir.is_dir():
                # 폴더 안의 .glb 파일을 모두 찾아서 첫 번째 파일 지정
                glb_files = list(pred_mesh_dir.glob("*.glb"))
                if glb_files:
                    mesh_path = str(glb_files[0])
                else:
                    # glb가 없다면 obj 파일이라도 탐색
                    obj_files = list(pred_mesh_dir.glob("*.obj"))
                    if obj_files:
                        mesh_path = str(obj_files[0])
        elif "mesh_path" in df.columns:
            mesh_path = str(row["mesh_path"]).strip()

        # CSV에 .obj로 기록되어 있더라도 실제 파일이 .glb면 경로 교체 (안전장치)
        if mesh_path and mesh_path.endswith('.obj'):
            glb_fallback = Path(mesh_path).with_suffix('.glb')
            if glb_fallback.is_file():
                mesh_path = str(glb_fallback)

        # -------------------------------------------------------------
        # [PATCH] 2. GT Mesh Path 탐색 (단일 파일)
        # -------------------------------------------------------------
        gt_mesh_path = ""
        if args.gt_mesh_root:
            # 정답 파일은 내부에 별도 디렉토리 없이 label.glb 형태로 존재
            potential_path = Path(args.gt_mesh_root) / f"{label}.glb"
            if potential_path.is_file():
                gt_mesh_path = str(potential_path)
            else:
                # 확장자가 대문자이거나 obj인 경우를 위한 Fallback
                obj_path = Path(args.gt_mesh_root) / f"{label}.obj"
                if obj_path.is_file():
                    gt_mesh_path = str(obj_path)

        try:
            pred_views = canonical_view_paths(pred_dir, args.num_views)
            if args.reference_mode == "six_view":
                gt_dir = str(row[gt_dir_col]).strip() if gt_dir_col and gt_dir_col in row.index else str(
                    Path(args.rendered_view_path).expanduser() / label)
                gt_views = canonical_view_paths(gt_dir, args.num_views)
            else:
                gt_views = []
        except FileNotFoundError as e:
            print(f"[Warning] Skipping '{sample_id}': {e}")
            continue

        obj = EvalObject(
            sample_id=sample_id, label=label, category=category,
            pred_views=pred_views, gt_views=gt_views,
            cls_index=_safe_int(row.get("cls_index")),
            object_suffix=suffix, pred_source=pred_source,
            pred_dir=pred_dir, gt_dir=gt_dir,
            mesh_path=mesh_path, gt_mesh_path=gt_mesh_path
        )
        objects.append(obj)

    print(f"[data] Valid objects: {len(objects)}")
    return objects, pred_source


# ============================================================================
# Textural & Structural Metrics (PSNR, SSIM, CD, EMD)
# ============================================================================

def compute_psnr_ssim(gt_paths, pred_paths):
    psnr_vals, ssim_vals = [], []
    for gp, pp in zip(gt_paths, pred_paths):
        g = np.array(load_rgb(gp))
        p = np.array(load_rgb(pp).resize((g.shape[1], g.shape[0])))

        # skimage expects data range 255 for uint8
        psnr_vals.append(psnr_metric(g, p, data_range=255))
        ssim_vals.append(ssim_metric(g, p, channel_axis=-1, data_range=255))

    return np.mean(psnr_vals), np.mean(ssim_vals)


def load_as_single_mesh(path):
    """ .glb 파일이 Scene으로 로드될 경우 모든 Geometry를 병합하여 단일 메쉬로 반환합니다. """
    scene_or_mesh = trimesh.load(path)
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            raise ValueError(f"No geometry found in {path}")
        # 씬 내부의 모든 메쉬를 하나로 병합
        return trimesh.util.concatenate(list(scene_or_mesh.geometry.values()))
    return scene_or_mesh


def chamfer_distance_pytorch(p1, p2):
    """
    p1: [B, N, 3], p2: [B, M, 3]
    """
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff ** 2, dim=-1)  # [B, N, M]
    dist1 = torch.min(dist, dim=2)[0].mean(dim=1)
    dist2 = torch.min(dist, dim=1)[0].mean(dim=1)
    return (dist1 + dist2).mean()


# ============================================================================
# FPD (Fréchet Point Cloud Distance) Core
# ============================================================================
class STN3d(nn.Module):
    """Input spatial transformer used by the original TreeGAN FPD PointNet."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(batch_size, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        identity = torch.eye(3, dtype=x.dtype, device=x.device).reshape(1, 9).repeat(batch_size, 1)
        return (x + identity).view(-1, 3, 3)


class PointNetFeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return torch.max(x, 2, keepdim=False)[0], trans


class PointNetFPD(nn.Module):
    """PointNet classifier returning the concatenated activation used by FPD."""

    def __init__(self, num_classes=16):
        super().__init__()
        self.feat = PointNetFeat()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x1, trans = self.feat(x)
        x2 = F.relu(self.bn1(self.fc1(x1)))
        x3 = F.relu(self.bn2(self.fc2(x2)))
        logits = self.fc3(x3)
        activation = torch.cat((x1, x2, x3, logits), dim=1)
        return logits, trans, activation


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        checkpoint = checkpoint.state_dict()
    elif isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "pointnet", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("The FPD checkpoint does not contain a state_dict.")

    state_dict = dict(checkpoint)
    for prefix in ("module.", "model.", "pointnet."):
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def load_fpd_pointnet(checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _checkpoint_state_dict(checkpoint)

    fc3_weight = state_dict.get("fc3.weight")
    if fc3_weight is None or fc3_weight.ndim != 2:
        raise ValueError("Incompatible FPD checkpoint: missing fc3.weight.")
    num_classes = int(fc3_weight.shape[0])
    model = PointNetFPD(num_classes=num_classes)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The checkpoint does not match the TreeGAN PointNet architecture; "
            "FPD cannot be compared with the reference score."
        ) from exc
    return model.to(device).eval(), 1024 + 512 + 256 + num_classes


def normalize_pointclouds_unit_sphere(pointclouds):
    pointclouds = np.asarray(pointclouds, dtype=np.float32)
    centered = pointclouds - pointclouds.mean(axis=1, keepdims=True)
    radius = np.linalg.norm(centered, axis=2).max(axis=1, keepdims=True)
    if np.any(~np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("FPD received a degenerate point cloud.")
    return centered / radius[..., None]


@torch.no_grad()
def extract_pointnet_features(model, pointclouds, device, batch_size=32):
    """Extract one global TreeGAN PointNet activation per point cloud."""
    points = torch.as_tensor(pointclouds, dtype=torch.float32)
    features = []
    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size].to(device).transpose(1, 2).contiguous()
        _, _, activation = model(batch)
        features.append(activation.detach().cpu())
    return torch.cat(features, dim=0)


def compute_mean_cov(features):
    features = features.double()
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("FPD requires at least two feature vectors per distribution.")
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / (features.shape[0] - 1)
    return mean, covariance


def matrix_sqrt_psd(matrix, eps=1e-10):
    matrix = (matrix + matrix.T) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)
    return (eigenvectors * torch.sqrt(eigenvalues + eps).unsqueeze(0)) @ eigenvectors.T


def frechet_pointcloud_distance(real_features, generated_features, eps=1e-10):
    """Compute one distribution-level FPD from all GT and predicted objects."""
    real = real_features.double()
    generated = generated_features.double()
    if real.ndim != 2 or generated.ndim != 2 or real.shape[1] != generated.shape[1]:
        raise ValueError("GT and predicted features must be [N, D] with an identical D.")
    if real.shape[0] < 2 or generated.shape[0] < 2:
        raise ValueError("FPD requires at least two GT and two predicted objects.")

    real_mean = real.mean(dim=0)
    generated_mean = generated.mean(dim=0)
    real_scaled = (real - real_mean) / np.sqrt(real.shape[0] - 1)
    generated_scaled = (generated - generated_mean) / np.sqrt(generated.shape[0] - 1)

    # Exact low-rank form of the covariance square-root trace. For the usual
    # case N << D, this replaces a costly D x D eigendecomposition by an
    # N_real x N_generated SVD without changing the Fréchet distance.
    trace_covmean = torch.linalg.svdvals(real_scaled @ generated_scaled.T).sum()
    value = (
        (real_mean - generated_mean).square().sum()
        + real_scaled.square().sum()
        + generated_scaled.square().sum()
        - 2.0 * trace_covmean
    )
    if value < 0 and value > -eps:
        value = value.new_zeros(())
    return float(value.item())


def resolve_fpd_checkpoint(args):
    if args.fpd_pointnet_ckpt:
        path = Path(args.fpd_pointnet_ckpt).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"FPD checkpoint not found: {path}")
        return path

    path = Path(args.out_dir).expanduser() / "cls_model_39.pth"
    if not path.is_file():
        print(f"[*] Downloading the pretrained FPD PointNet checkpoint to {path}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        download_url_to_file(args.fpd_pointnet_url, str(path), progress=True)
    return path


def compute_3d_metrics(pred_mesh_path, gt_mesh_path, num_samples):
    if not (pred_mesh_path and gt_mesh_path and os.path.exists(pred_mesh_path) and os.path.exists(gt_mesh_path)):
        return float('nan'), float('nan'), None, None

    try:
        pred_mesh = load_as_single_mesh(pred_mesh_path)
        gt_mesh = load_as_single_mesh(gt_mesh_path)

        pred_pts_raw, _ = trimesh.sample.sample_surface(pred_mesh, num_samples)
        gt_pts_raw, _ = trimesh.sample.sample_surface(gt_mesh, num_samples)

        # CD/EMD preprocessing is kept unchanged for compatibility with the
        # existing reported values. FPD receives the raw samples and applies
        # PointNet's zero-mean/unit-sphere normalization separately.
        pred_pts = (pred_pts_raw - pred_pts_raw.min(axis=0)) / (
            pred_pts_raw.max(axis=0) - pred_pts_raw.min(axis=0) + 1e-6
        )
        gt_pts = (gt_pts_raw - gt_pts_raw.min(axis=0)) / (
            gt_pts_raw.max(axis=0) - gt_pts_raw.min(axis=0) + 1e-6
        )

        t_p1 = torch.tensor(pred_pts, dtype=torch.float32).unsqueeze(0)
        t_p2 = torch.tensor(gt_pts, dtype=torch.float32).unsqueeze(0)
        cd_val = chamfer_distance_pytorch(t_p1, t_p2).item() * 100.0

        M = ot.dist(pred_pts, gt_pts, metric='euclidean')
        a, b = np.ones((len(pred_pts),)) / len(pred_pts), np.ones((len(gt_pts),)) / len(gt_pts)
        emd_val = ot.emd2(a, b, M) * 100.0

        return cd_val, emd_val, pred_pts_raw, gt_pts_raw
    except Exception as e:
        print(f"[Warning] Failed computing 3D metrics for {pred_mesh_path}: {e}")
        return float('nan'), float('nan'), None, None


# ============================================================================
# Image metrics & LPIPS
# ============================================================================
def image_to_unit_tensor(image: Image.Image, size: Tuple[int, int] = None) -> torch.Tensor:
    x = TF.to_tensor(image.convert("RGB"))
    if size is not None and tuple(x.shape[-2:]) != tuple(size):
        x = TF.resize(x, list(size), interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
    return x.clamp(0, 1)

class LPIPSScorer:
    def __init__(self, device):
        self.device = device
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def pair_scores(self, gt_paths, pred_paths, batch_size):
        values = []
        for s in range(0, len(gt_paths), batch_size):
            gb, pb = [], []
            for gp, pp in zip(gt_paths[s:s + batch_size], pred_paths[s:s + batch_size]):
                gi = load_rgb(gp)
                pi = load_rgb(pp)
                target_hw = (gi.height, gi.width)
                g = image_to_unit_tensor(gi)
                p = image_to_unit_tensor(pi, target_hw)
                gb.append(g * 2 - 1)
                pb.append(p * 2 - 1)
            g = torch.stack(gb).to(self.device)
            p = torch.stack(pb).to(self.device)
            values.extend(self.model(g, p).reshape(-1).float().cpu().tolist())
        return np.asarray(values, dtype=np.float64)

# ============================================================================
# Brain3D N-way (Classifier)
# ============================================================================
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
        x = torch.stack([preprocess(load_rgb(p)) for p in paths[s:s + batch_size]]).to(device)
        out.append(torch.softmax(model(x).float(), dim=-1).cpu())
    return torch.cat(out, dim=0).numpy()

def nway_trials_for_pair(pred_prob, positive_class, n_way, top_k, trials, rng):
    K = pred_prob.shape[0]
    neg_pool = np.concatenate([np.arange(0, positive_class, dtype=np.int64), np.arange(positive_class + 1, K, dtype=np.int64)])
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
    settings = [("2way_top1", 2, 1), ("10way_top1", 10, 1), ("10way_top2", 10, 2), ("50way_top1", 50, 1), ("50way_top2", 50, 2)]
    positive = gt_probs.argmax(axis=1)
    summary, per_object, trial_rows = {}, {}, []
    for setting_idx, (name, n_way, top_k) in enumerate(settings):
        pair_trial = np.zeros((len(pred_probs), trials), dtype=np.float64)
        for pair_idx in range(len(pred_probs)):
            rng = np.random.default_rng(seed + setting_idx * 1_000_003 + pair_idx * 9_973)
            pair_trial[pair_idx] = nway_trials_for_pair(pred_probs[pair_idx], int(positive[pair_idx]), n_way, top_k, trials, rng)
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
        summary[name] = {"mean": float(global_trials.mean()), "std": float(global_trials.std(ddof=0))}
    return summary, per_object, trial_rows, positive


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    device = torch.device(args.device)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    objects, pred_source = load_eval_objects(args)

    gt_paths, pred_paths, obj_idx, view_idx = [], [], [], []
    for oi, obj in enumerate(objects):
        for vi in range(args.num_views):
            gt_paths.append(obj.gt_views[vi])
            pred_paths.append(obj.pred_views[vi])
            obj_idx.append(oi)
            view_idx.append(vi)

    obj_idx = np.asarray(obj_idx, dtype=np.int64)

    # ------------------------------------------------------------------
    # [1] Textural-Level (PSNR, SSIM)
    # ------------------------------------------------------------------
    print("\n[*] Computing Textural-Level Metrics (PSNR, SSIM)...")
    psnr_pair = np.zeros(len(gt_paths))
    ssim_pair = np.zeros(len(gt_paths))
    for i in tqdm(range(len(gt_paths)), desc="PSNR/SSIM"):
        p, s = compute_psnr_ssim([gt_paths[i]], [pred_paths[i]])
        psnr_pair[i] = p
        ssim_pair[i] = s

    # ------------------------------------------------------------------
    # [2] Structure-Level (CD, EMD, FPD)
    # ------------------------------------------------------------------
    print("\n[*] Computing Structure-Level Metrics (CD, EMD, FPD)...")
    cd_scores = np.full(len(objects), np.nan)
    emd_scores = np.full(len(objects), np.nan)
    final_fpd = float('nan')

    pred_pts_list, gt_pts_list = [], []

    if not args.skip_3d_metrics and args.gt_mesh_root:
        valid_3d_pairs = [obj for obj in objects if obj.mesh_path and obj.gt_mesh_path]
        if len(valid_3d_pairs) == 0:
            print("[Warning] 매칭되는 .glb 메쉬 쌍이 0개입니다. 3D 구조 평가를 스킵합니다.")
        else:
            print(f"[*] 유효한 3D 메쉬 매칭 쌍: {len(valid_3d_pairs)}개 발견.")
            for oi, obj in enumerate(tqdm(objects, desc="CD/EMD")):
                cd, emd, p_pts, g_pts = compute_3d_metrics(obj.mesh_path, obj.gt_mesh_path, args.num_mesh_samples)
                cd_scores[oi] = cd
                emd_scores[oi] = emd
                if p_pts is not None and g_pts is not None:
                    pred_pts_list.append(p_pts)
                    gt_pts_list.append(g_pts)

            # FPD is computed once from the complete GT/prediction feature sets.
            if len(pred_pts_list) > 1:
                try:
                    print("[*] Calculating distribution-level FPD...")
                    fpd_checkpoint = resolve_fpd_checkpoint(args)
                    pointnet, feature_dim = load_fpd_pointnet(fpd_checkpoint, device)

                    pred_clouds = normalize_pointclouds_unit_sphere(np.stack(pred_pts_list))
                    gt_clouds = normalize_pointclouds_unit_sphere(np.stack(gt_pts_list))
                    feat_pred = extract_pointnet_features(
                        pointnet, pred_clouds, device, batch_size=args.fpd_batch_size
                    )
                    feat_gt = extract_pointnet_features(
                        pointnet, gt_clouds, device, batch_size=args.fpd_batch_size
                    )
                    final_fpd = frechet_pointcloud_distance(feat_gt, feat_pred)
                    print(
                        f"[*] Computed FPD: {final_fpd:.3f} "
                        f"(objects={len(pred_pts_list)}, feature_dim={feature_dim})"
                    )
                    del pointnet, feat_pred, feat_gt
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[Warning] FPD calculation failed: {e}")
                    print("Provide the reference PointNet checkpoint explicitly with --fpd_pointnet_ckpt.")
            else:
                print("[Warning] FPD requires at least two valid GT/predicted mesh pairs.")
    else:
        print("[Warning] Skipping CD, EMD, FPD (No --gt_mesh_root provided or skipped).")

    # ------------------------------------------------------------------
    # [3] Image Perceptual Metric (LPIPS)
    # ------------------------------------------------------------------
    print("\n[*] Computing LPIPS...")
    lpips_model = LPIPSScorer(device)
    lpips_pair = lpips_model.pair_scores(gt_paths, pred_paths, args.batch_size)

    del lpips_model
    if device.type == "cuda": torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # [4] Semantic N-Way Classification (2w-T1, 10w-T1, etc.)
    # ------------------------------------------------------------------
    print(f"\n[*] Computing N-way Classification ({args.classifier}, {args.nway_trials} trials)...")
    clf, clf_pre, classifier_categories = build_classifier(args.classifier, device)

    gt_probs = classifier_probabilities(gt_paths, clf, clf_pre, device, args.batch_size)
    pred_probs = classifier_probabilities(pred_paths, clf, clf_pre, device, args.batch_size)

    nway, nway_obj, trial_rows, gt_positive = compute_all_nway(
        gt_probs, pred_probs, obj_idx, len(objects), args.nway_trials, args.seed
    )

    del clf
    if device.type == "cuda": torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    final_psnr = np.nanmean(psnr_pair) if not np.isnan(psnr_pair).all() else float('nan')
    final_ssim = np.nanmean(ssim_pair) if not np.isnan(ssim_pair).all() else float('nan')
    final_cd = np.nanmean(cd_scores) if not np.isnan(cd_scores).all() else float('nan')
    final_emd = np.nanmean(emd_scores) if not np.isnan(emd_scores).all() else float('nan')
    final_lpips = float(np.mean(lpips_pair))

    val_2w_t1 = nway["2way_top1"]["mean"]
    val_10w_t1 = nway["10way_top1"]["mean"]

    print("\n" + "=" * 115)
    print(
        f"{'Method':<10}{'2w-T1':>9}{'10w-T1':>10}{'CD(↓)':>10}{'EMD(↓)':>10}{'FPD(↓)':>10}{'LPIPS(↓)':>10}{'PSNR(↑)':>10}{'SSIM(↑)':>10}")
    print("-" * 115)
    print(
        f"{'Ours':<10}{val_2w_t1:>9.3f}{val_10w_t1:>10.3f}{final_cd:>10.3f}{final_emd:>10.3f}{final_fpd:>10.3f}{final_lpips:>10.3f}{final_psnr:>10.3f}{final_ssim:>10.3f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
