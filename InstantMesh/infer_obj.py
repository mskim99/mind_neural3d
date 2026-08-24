#!/usr/bin/env python3
"""
InstantMesh reconstruction aligned with the EEG inference output layout.

Expected EEG inference output:
    <inference_root>/
        render/
            airplane_00.png
            chair_00.png
            ...
        views/
            airplane_00/
                00.png ... 05.png
            ...
        evaluation_pairs.csv

This script can receive either:
    --input_path <inference_root>
or
    --input_path <inference_root>/render
or a single packed 3x2 PNG.

For each packed 3x2 prediction:
    1) split row-major into exactly six canonical views
    2) reconstruct an InstantMesh/FlexiCubes mesh
    3) save the mesh using the same object label
    4) render the reconstructed geometry again from the SAME six canonical
       Zero123++ camera poses and save 00.png ... 05.png for evaluation

Output:
    <output_path>/
        meshes/
            airplane_00.obj
        views/
            airplane_00/
                00.png ... 05.png
        videos/                 # only if --save_video
            airplane_00.mp4
        mesh_pairs.csv

The script intentionally does NOT:
    - remove background again
    - crop object regions
    - recenter
    - reorder views
    - vertically flip the packed grid

GPU convention:
    defaults to physical GPU 1 via CUDA_VISIBLE_DEVICES=1.
"""

# Must be set before importing torch.
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import glob
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import save_video


# -----------------------------------------------------------------------------
# Camera / rendering helpers
# -----------------------------------------------------------------------------

def get_render_cameras(
    batch_size=1,
    M=120,
    radius=4.0,
    elevation=20.0,
    is_flexicubes=False,
):
    c2ws = get_circular_camera_poses(
        M=M,
        radius=radius,
        elevation=elevation,
    )
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = (
            FOV_to_intrinsics(30.0)
            .unsqueeze(0)
            .repeat(M, 1, 1)
            .float()
            .flatten(-2)
        )
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(
    model,
    planes,
    render_cameras,
    render_size=512,
    chunk_size=1,
    is_flexicubes=False,
):
    frames = []
    for i in range(0, render_cameras.shape[1], chunk_size):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )["img"]
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )["images_rgb"]
        frames.append(frame)

    return torch.cat(frames, dim=1)[0]


def input_camera_features_to_flexicubes_w2c(input_cameras):
    """
    InstantMesh input camera feature:
        first 12 values = flattened 3x4 camera-to-world matrix
        final 4 values  = normalized intrinsics

    FlexiCubes rendering expects 4x4 world-to-camera matrices.
    """
    if input_cameras.ndim != 3 or input_cameras.shape[-1] < 12:
        raise ValueError(
            f"Expected camera features [B,V,>=12], got {tuple(input_cameras.shape)}"
        )

    B, V, _ = input_cameras.shape
    c2w_3x4 = input_cameras[..., :12].reshape(B, V, 3, 4)

    bottom = torch.zeros(
        (B, V, 1, 4),
        dtype=c2w_3x4.dtype,
        device=c2w_3x4.device,
    )
    bottom[..., 0, 3] = 1.0

    c2ws = torch.cat([c2w_3x4, bottom], dim=-2)
    return torch.linalg.inv(c2ws)


def tensor_to_pil(x):
    """
    [3,H,W] float tensor, expected [0,1] -> RGB PIL.
    """
    x = x.detach().float().cpu().clamp(0, 1)
    arr = (
        x.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_six_mesh_views(frames, output_dir):
    """
    frames: [6,3,H,W]
    """
    if frames.ndim != 4 or frames.shape[0] != 6:
        raise ValueError(
            f"Expected six rendered frames [6,3,H,W], got {tuple(frames.shape)}"
        )

    os.makedirs(output_dir, exist_ok=True)

    for view_idx in range(6):
        tensor_to_pil(frames[view_idx]).save(
            os.path.join(output_dir, f"{view_idx:02d}.png")
        )


# -----------------------------------------------------------------------------
# Input discovery / label mapping
# -----------------------------------------------------------------------------

def resolve_render_input(input_path):
    """
    Return:
        render_source_dir_or_none,
        input_files,
        inference_root_or_none

    If input_path is an inference root containing render/, automatically use it.
    """
    p = Path(input_path).expanduser().resolve()

    if p.is_file():
        return None, [str(p)], None

    if not p.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    # Preferred: user passes the EEG inference root.
    render_child = p / "render"
    if render_child.is_dir():
        render_dir = render_child
        inference_root = p
    else:
        # Also support passing .../render directly.
        render_dir = p
        inference_root = p.parent if (p.name == "render") else None

    input_files = sorted(
        str(f)
        for f in render_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not input_files:
        raise RuntimeError(f"No packed render images found in: {render_dir}")

    return str(render_dir), input_files, inference_root


def load_evaluation_mapping(inference_root):
    """
    Read the evaluation_pairs.csv produced by
    inference_neural3d_pp_render_views.py.

    Mapping is keyed by render-grid stem / sample_id.
    """
    mapping = {}
    if inference_root is None:
        return mapping

    csv_path = Path(inference_root) / "evaluation_pairs.csv"
    if not csv_path.is_file():
        return mapping

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            mapping[sample_id] = row

    print(f"Loaded label mapping: {csv_path} ({len(mapping)} entries)")
    return mapping


def validate_packed_grid(image, path):
    """
    EEG inference writes a 3(row)x2(col) packed grid.
    It can be any size divisible by 3x2, but each tile is expected to be square.
    """
    W, H = image.size

    if H % 3 != 0 or W % 2 != 0:
        raise ValueError(
            f"{path}: packed image {W}x{H} is not divisible into 3x2."
        )

    tile_h = H // 3
    tile_w = W // 2

    if tile_h != tile_w:
        raise ValueError(
            f"{path}: expected square tiles, got tile {tile_w}x{tile_h} "
            f"from packed grid {W}x{H}."
        )

    return tile_w, tile_h


def grid_image_to_model_tensor(image, device):
    """
    RGB packed 3x2 PIL image -> [1,6,3,320,320]

    Row-major canonical order:
        0 | 1
        2 | 3
        4 | 5
    """
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    grid = (
        torch.from_numpy(image_np)
        .permute(2, 0, 1)
        .contiguous()
        .float()
    )

    views = rearrange(
        grid,
        "c (n h) (m w) -> (n m) c h w",
        n=3,
        m=2,
    )

    # The EEG inference currently produces 320x320 tiles.
    # Resize only if another valid square resolution is supplied.
    if views.shape[-2:] != (320, 320):
        views = v2.functional.resize(
            views,
            [320, 320],
            interpolation=v2.InterpolationMode.BICUBIC,
            antialias=True,
        )

    return views.unsqueeze(0).to(device).clamp(0, 1)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="InstantMesh reconstruction for EEG inference render/ outputs"
)
parser.add_argument("config", type=str, help="Path to InstantMesh config YAML.")
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help=(
        "EEG inference root containing render/, the render/ directory itself, "
        "or a single packed 3x2 image."
    ),
)
parser.add_argument(
    "--output_path",
    type=str,
    default="mesh_results",
    help="Output root. Creates meshes/, views/, and optional videos/.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--scale", type=float, default=1.0)
parser.add_argument("--distance", type=float, default=4.5)
parser.add_argument("--export_texmap", action="store_true")
parser.add_argument("--save_video", action="store_true")
parser.add_argument(
    "--eval_view_resolution",
    type=int,
    default=320,
    help="Resolution for the six reconstructed-mesh evaluation views.",
)
parser.add_argument(
    "--gt_render_root",
    type=str,
    default="/data/jionkim/neuro_3D/render_grid_v4",
    help=(
        "Optional GT canonical-view root used only to validate 1:1 labels. "
        "Expected <root>/<label>/00.png ... 05.png."
    ),
)
parser.add_argument(
    "--allow_missing_gt",
    action="store_true",
    help="Allow mesh creation when no matching GT label directory exists.",
)
args = parser.parse_args()

seed_everything(args.seed)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

config = OmegaConf.load(args.config)
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True
device = torch.device("cuda")

print(f"Using device: {device} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
print("Loading reconstruction model ...")

model = instantiate_from_config(model_config)
model_ckpt_path = infer_config.model_path
state_dict = torch.load(
    model_ckpt_path,
    map_location="cpu",
)["state_dict"]
state_dict = {
    k[14:]: v
    for k, v in state_dict.items()
    if k.startswith("lrm_generator.")
}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(
        device,
        fovy=30.0,
    )
model = model.eval()

# -----------------------------------------------------------------------------
# Output layout
# -----------------------------------------------------------------------------

mesh_root = os.path.join(args.output_path, "meshes")
view_root = os.path.join(args.output_path, "views")
video_root = os.path.join(args.output_path, "videos")

os.makedirs(mesh_root, exist_ok=True)
os.makedirs(view_root, exist_ok=True)
if args.save_video:
    os.makedirs(video_root, exist_ok=True)

# -----------------------------------------------------------------------------
# Input discovery
# -----------------------------------------------------------------------------

render_dir, input_files, inference_root = resolve_render_input(args.input_path)
label_mapping = load_evaluation_mapping(inference_root)

print(f"Total packed input grids: {len(input_files)}")
if render_dir is not None:
    print(f"Input render directory: {render_dir}")

# Fixed canonical input cameras: same convention as GT renderer / Zero123++.
input_cameras_all = get_zero123plus_input_cameras(
    batch_size=1,
    radius=4.0 * args.scale,
).to(device)

if input_cameras_all.shape[1] != 6:
    raise RuntimeError(
        f"Expected 6 canonical input cameras, got {input_cameras_all.shape[1]}"
    )

# The SAME camera poses converted to FlexiCubes W2C for evaluation rendering.
canonical_eval_w2c = input_camera_features_to_flexicubes_w2c(
    input_cameras_all
)

rows_for_csv = []

# -----------------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------------

for idx, image_file in enumerate(input_files):
    sample_id = Path(image_file).stem

    # Prefer label from inference evaluation_pairs.csv.
    map_row = label_mapping.get(sample_id, {})
    label = str(map_row.get("label", sample_id)).strip() or sample_id

    print(
        f"[{idx + 1}/{len(input_files)}] "
        f"Creating mesh: sample_id={sample_id}, label={label}"
    )

    # 1:1 GT label validation.
    gt_view_dir = ""
    if args.gt_render_root:
        gt_view_dir = os.path.join(
            args.gt_render_root,
            label,
        )
        missing_gt = [
            f"{i:02d}.png"
            for i in range(6)
            if not os.path.isfile(
                os.path.join(gt_view_dir, f"{i:02d}.png")
            )
        ]
        if missing_gt and not args.allow_missing_gt:
            raise FileNotFoundError(
                f"No complete 1:1 GT six-view match for '{label}'.\n"
                f"Expected: {gt_view_dir}/00.png ... 05.png\n"
                f"Missing: {missing_gt}"
            )

    # No rembg here: render/ already contains cleaned white-background grids.
    image = Image.open(image_file).convert("RGB")
    tile_w, tile_h = validate_packed_grid(image, image_file)

    print(
        f"  packed={image.size[0]}x{image.size[1]}, "
        f"tile={tile_w}x{tile_h}"
    )

    images = grid_image_to_model_tensor(
        image,
        device=device,
    )

    with torch.inference_mode():
        # Reconstruct triplanes from the exact 6 input views.
        planes = model.forward_planes(
            images,
            input_cameras_all,
        )

        # Extract final mesh.
        mesh_file = os.path.join(
            mesh_root,
            f"{sample_id}.obj",
        )
        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_texmap,
            **infer_config,
        )

        if args.export_texmap:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_file,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(
                vertices,
                faces,
                vertex_colors,
                mesh_file,
            )

        print(f"  -> Mesh: {mesh_file}")

        # --------------------------------------------------------------
        # Re-render the reconstructed FlexiCubes geometry from the SAME
        # six canonical camera poses. These are the views that should be
        # used for final mesh-quality evaluation against GT six views.
        # --------------------------------------------------------------
        mesh_view_dir = os.path.join(
            view_root,
            sample_id,
        )

        canonical_frames = render_frames(
            model,
            planes,
            canonical_eval_w2c,
            render_size=args.eval_view_resolution,
            chunk_size=6,
            is_flexicubes=IS_FLEXICUBES,
        )

        save_six_mesh_views(
            canonical_frames,
            mesh_view_dir,
        )

        print(
            f"  -> Mesh evaluation views: "
            f"{mesh_view_dir}/00.png ... 05.png"
        )

        # Optional 120-view turntable.
        video_file = ""
        if args.save_video:
            video_file = os.path.join(
                video_root,
                f"{sample_id}.mp4",
            )
            render_cameras = get_render_cameras(
                batch_size=1,
                M=120,
                radius=args.distance,
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)

            video_frames = render_frames(
                model,
                planes,
                render_cameras=render_cameras,
                render_size=infer_config.render_resolution,
                chunk_size=20,
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(
                video_frames,
                video_file,
                fps=30,
            )
            print(f"  -> Video: {video_file}")

    rows_for_csv.append({
        "sample_id": sample_id,
        "label": label,
        "input_grid": image_file,
        "mesh_path": mesh_file,
        "pred_mesh_view_dir": mesh_view_dir,
        "gt_view_dir": gt_view_dir,
        "video_path": video_file,
    })

# -----------------------------------------------------------------------------
# Save explicit GT <-> mesh-view mapping for evaluation
# -----------------------------------------------------------------------------

pairs_csv = os.path.join(
    args.output_path,
    "mesh_pairs.csv",
)

with open(
    pairs_csv,
    "w",
    newline="",
    encoding="utf-8",
) as f:
    fieldnames = [
        "sample_id",
        "label",
        "input_grid",
        "mesh_path",
        "pred_mesh_view_dir",
        "gt_view_dir",
        "video_path",
    ]
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows_for_csv)

print("\nInference complete.")
print(f"Meshes          : {mesh_root}")
print(f"Mesh 6-view eval: {view_root}")
if args.save_video:
    print(f"Videos          : {video_root}")
print(f"Evaluation pairs: {pairs_csv}")