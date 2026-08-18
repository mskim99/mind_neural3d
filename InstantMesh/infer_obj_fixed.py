import os
import glob
import argparse
import math
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from tqdm import tqdm
import rembg

# 기존 프로젝트의 유틸리티 모듈 유지
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import save_video


###############################################################################
# Camera / rendering utilities
###############################################################################


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """Get circular rendering camera parameters."""
    c2ws = get_circular_camera_poses(M=M, radius=radius, elevation=elevation)
    if is_flexicubes:
        cameras = torch.linalg.inv(c2ws)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    else:
        extrinsics = c2ws.flatten(-2)
        intrinsics = FOV_to_intrinsics(30.0).unsqueeze(0).repeat(M, 1, 1).float().flatten(-2)
        cameras = torch.cat([extrinsics, intrinsics], dim=-1)
        cameras = cameras.unsqueeze(0).repeat(batch_size, 1, 1)
    return cameras


def render_frames(model, planes, render_cameras, render_size=512, chunk_size=1, is_flexicubes=False):
    """Render circular-view frames from triplanes."""
    frames = []
    for i in tqdm(range(0, render_cameras.shape[1], chunk_size)):
        if is_flexicubes:
            frame = model.forward_geometry(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['img']
        else:
            frame = model.forward_synthesizer(
                planes,
                render_cameras[:, i:i + chunk_size],
                render_size=render_size,
            )['images_rgb']
        frames.append(frame)

    return torch.cat(frames, dim=1)[0]


def _camera_tensor_from_dict(data):
    """
    Build InstantMesh camera conditioning vectors [V, 16].

    Accepted layouts:
      1) cameras: [V, 16]
      2) c2ws: [V, 4, 4] + Ks: [V, 3, 3]
      3) c2ws: [V, 4, 4] + intrinsics: [V, 4] containing fx, fy, cx, cy

    IMPORTANT: intrinsics must use the same normalization/convention as the model's
    training data. No automatic pixel -> normalized conversion is attempted here.
    """
    if 'cameras' in data:
        cameras = np.asarray(data['cameras'], dtype=np.float32)
        if cameras.ndim == 3 and cameras.shape[0] == 1:
            cameras = cameras[0]
        if cameras.ndim != 2 or cameras.shape[-1] != 16:
            raise ValueError(f"'cameras' must have shape [V,16], got {cameras.shape}")
        return torch.from_numpy(cameras).float()

    if 'c2ws' not in data:
        raise ValueError("Camera file must contain either 'cameras' or 'c2ws'.")

    c2ws = np.asarray(data['c2ws'], dtype=np.float32)
    if c2ws.ndim == 4 and c2ws.shape[0] == 1:
        c2ws = c2ws[0]
    if c2ws.ndim != 3 or c2ws.shape[-2:] != (4, 4):
        raise ValueError(f"'c2ws' must have shape [V,4,4], got {c2ws.shape}")

    extrinsics = c2ws.reshape(c2ws.shape[0], 16)[:, :12]

    if 'Ks' in data:
        Ks = np.asarray(data['Ks'], dtype=np.float32)
        if Ks.ndim == 4 and Ks.shape[0] == 1:
            Ks = Ks[0]
        if Ks.ndim != 3 or Ks.shape[-2:] != (3, 3):
            raise ValueError(f"'Ks' must have shape [V,3,3], got {Ks.shape}")
        intrinsics = np.stack(
            [Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]], axis=-1
        )
    elif 'intrinsics' in data:
        intrinsics = np.asarray(data['intrinsics'], dtype=np.float32)
        if intrinsics.ndim == 3 and intrinsics.shape[0] == 1:
            intrinsics = intrinsics[0]
        if intrinsics.ndim != 2 or intrinsics.shape[-1] != 4:
            raise ValueError(
                f"'intrinsics' must have shape [V,4] = [fx,fy,cx,cy], got {intrinsics.shape}"
            )
    else:
        raise ValueError("When using 'c2ws', provide either 'Ks' or 'intrinsics'.")

    cameras = np.concatenate([extrinsics, intrinsics], axis=-1).astype(np.float32)
    return torch.from_numpy(cameras).float()


def load_input_cameras(camera_path, num_views, scale, device):
    """Load custom InstantMesh camera conditioning, or use Zero123++ defaults."""
    if camera_path is None:
        cameras = get_zero123plus_input_cameras(
            batch_size=1,
            radius=4.0 * scale,
        ).float()
        if cameras.shape[1] != num_views:
            raise ValueError(
                f"Zero123++ camera utility returned {cameras.shape[1]} views; expected {num_views}."
            )
        print(
            "[WARNING] --camera_path was not provided. Using Zero123++ predefined cameras.\n"
            "          This is only geometrically valid when the 6 input views follow the\n"
            "          Zero123++ camera convention and ordering."
        )
        return cameras.to(device), 'zero123plus'

    ext = os.path.splitext(camera_path)[1].lower()
    if ext == '.npy':
        arr = np.load(camera_path)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[-1] != 16:
            raise ValueError(
                f".npy camera file must directly contain [V,16] camera vectors, got {arr.shape}"
            )
        cameras = torch.from_numpy(arr.astype(np.float32))
    elif ext == '.npz':
        with np.load(camera_path) as data:
            cameras = _camera_tensor_from_dict({k: data[k] for k in data.files})
    elif ext in ('.pt', '.pth'):
        data = torch.load(camera_path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            cameras = data.float()
            if cameras.ndim == 3 and cameras.shape[0] == 1:
                cameras = cameras[0]
            if cameras.ndim != 2 or cameras.shape[-1] != 16:
                raise ValueError(
                    f"Tensor camera file must contain [V,16], got {tuple(cameras.shape)}"
                )
        elif isinstance(data, dict):
            np_data = {}
            for k, v in data.items():
                if torch.is_tensor(v):
                    np_data[k] = v.detach().cpu().numpy()
                else:
                    np_data[k] = np.asarray(v)
            cameras = _camera_tensor_from_dict(np_data)
        else:
            raise ValueError("Unsupported .pt/.pth camera content.")
    else:
        raise ValueError("--camera_path supports .npy, .npz, .pt, or .pth files.")

    if cameras.shape[0] != num_views:
        raise ValueError(
            f"Camera count ({cameras.shape[0]}) does not match input views ({num_views})."
        )

    cameras = cameras.unsqueeze(0).to(device)
    print(f"Loaded custom cameras: {camera_path}  shape={tuple(cameras.shape)}")
    return cameras, 'custom'


###############################################################################
# Multi-view preprocessing
###############################################################################


def parse_view_order(text, num_views=6):
    """
    Parse a comma-separated list of tile indices.

    The resulting list means:
        model slot 0 <- grid tile order[0]
        model slot 1 <- grid tile order[1]
        ...
    """
    order = [int(x.strip()) for x in text.split(',') if x.strip() != '']
    if len(order) != num_views or sorted(order) != list(range(num_views)):
        raise ValueError(
            f"--view_order must be a permutation of 0..{num_views - 1}; got {order}"
        )
    return order


def crop_validation_top_half(image):
    """Keep the original script's convention for vertically stacked validation images."""
    w, h = image.size
    if h / w > 2.0:
        return image.crop((0, 0, w, h // 2))
    return image


def split_3x2_grid(image):
    """Split a 3-row x 2-column image into six PIL views in row-major order."""
    w, h = image.size
    if w % 2 != 0 or h % 3 != 0:
        raise ValueError(
            f"6-view image must be divisible as 3x2 grid. Got size {w}x{h}."
        )
    tile_w, tile_h = w // 2, h // 3
    views = []
    for r in range(3):
        for c in range(2):
            views.append(image.crop((c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)))
    return views


def make_debug_grid(views, columns=2, background=(255, 255, 255)):
    """Compose same-sized PIL images into a grid for visual inspection."""
    if not views:
        raise ValueError("No views to compose.")
    widths = [im.width for im in views]
    heights = [im.height for im in views]
    if len(set(widths)) != 1 or len(set(heights)) != 1:
        raise ValueError("Debug-grid views must share the same size.")
    rows = math.ceil(len(views) / columns)
    grid = Image.new('RGB', (widths[0] * columns, heights[0] * rows), background)
    for i, im in enumerate(views):
        grid.paste(im.convert('RGB'), ((i % columns) * widths[0], (i // columns) * heights[0]))
    return grid


def remove_background_and_center_views(
    views,
    rembg_session,
    foreground_ratio=0.85,
    alpha_threshold=8,
):
    """
    Remove background from each view and construct a white, object-centered canvas.

    Key detail: all six views use ONE shared pixel scale. We do NOT independently
    resize each object to 85%, because independent resizing would destroy relative
    multi-view scale cues. Instead, we:
      1) remove background per view,
      2) find each alpha bounding box,
      3) use the largest bbox extent across all views to choose one common canvas,
      4) center each foreground crop on that common canvas,
      5) composite onto white.
    """
    if not (0.0 < foreground_ratio < 1.0):
        raise ValueError("--foreground_ratio must be between 0 and 1.")

    rgba_views = []
    bboxes = []

    for i, view in enumerate(views):
        rgba = rembg.remove(view.convert('RGBA'), session=rembg_session)
        if not isinstance(rgba, Image.Image):
            rgba = Image.fromarray(rgba)
        rgba = rgba.convert('RGBA')

        arr = np.asarray(rgba)
        mask = arr[..., 3] > alpha_threshold
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError(
                f"Foreground segmentation failed for view {i}: alpha mask is empty. "
                "Try lowering --alpha_threshold or use --no_remove_background if inputs are already clean."
            )

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        rgba_views.append(rgba)
        bboxes.append((x1, y1, x2, y2))

    max_extent = max(max(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in bboxes)
    canvas_size = max(2, int(math.ceil(max_extent / foreground_ratio)))
    # Keep even spatial dimensions to avoid off-by-one behavior in later resizing.
    if canvas_size % 2 == 1:
        canvas_size += 1

    processed = []
    alpha_debug = []

    for rgba, bbox in zip(rgba_views, bboxes):
        x1, y1, x2, y2 = bbox
        fg = rgba.crop((x1, y1, x2, y2))

        # White RGB canvas; preserve the segmented alpha only for compositing.
        canvas = Image.new('RGBA', (canvas_size, canvas_size), (255, 255, 255, 255))
        px = (canvas_size - fg.width) // 2
        py = (canvas_size - fg.height) // 2
        canvas.alpha_composite(fg, dest=(px, py))
        processed.append(canvas.convert('RGB'))

        alpha_crop = np.asarray(fg)[..., 3]
        alpha_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        alpha_canvas[py:py + fg.height, px:px + fg.width] = alpha_crop
        alpha_debug.append(Image.fromarray(alpha_canvas, mode='L').convert('RGB'))

    return processed, alpha_debug


def views_to_model_tensor(views, size=320):
    """Convert a list of RGB PIL views to [1,V,3,H,W] float tensor."""
    tensors = []
    for view in views:
        arr = np.asarray(view.convert('RGB'), dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
    images = torch.stack(tensors, dim=0).float()
    images = v2.functional.resize(images, (size, size), interpolation=3, antialias=True).clamp(0, 1)
    return images.unsqueeze(0)


###############################################################################
# Arguments
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument(
    '--input_path', type=str, required=True,
    help='Path to a single 6-view 3x2 grid image or a directory of such images.'
)
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--save_name', type=str, default='', help='Suffix for the output directory.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--scale', type=float, default=1.0, help='Camera radius scale.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')

# New: preprocessing controls
parser.add_argument(
    '--no_remove_background', action='store_true',
    help='Disable rembg preprocessing. By default, background is removed per view.'
)
parser.add_argument(
    '--foreground_ratio', type=float, default=0.85,
    help='Largest foreground extent / normalized square canvas size. Default: 0.85.'
)
parser.add_argument(
    '--alpha_threshold', type=int, default=8,
    help='Alpha threshold used to determine rembg foreground bounding boxes. Default: 8.'
)
parser.add_argument(
    '--view_order', type=str, default='0,1,2,3,4,5',
    help=(
        'Permutation mapping grid tiles to model camera slots. '
        'Example: "0,2,4,5,3,1" means slot 0 gets tile 0, slot 1 gets tile 2, etc.'
    )
)

# New: custom camera support
parser.add_argument(
    '--camera_path', type=str, default=None,
    help=(
        'Optional custom cameras (.npy/.npz/.pt/.pth). '
        '.npy should be [6,16]. .npz/.pt dict can contain cameras=[6,16], '
        'or c2ws=[6,4,4] plus Ks=[6,3,3] / intrinsics=[6,4]. '
        'If omitted, Zero123++ predefined cameras are used.'
    )
)
args = parser.parse_args()

seed_everything(args.seed)
view_order = parse_view_order(args.view_order, num_views=6)

###############################################################################
# Stage 0: Configuration & Model Loading
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True
if args.save_name:
    config_name = config_name + '_' + args.save_name

device = torch.device('cuda')

print('Loading reconstruction model ...')
model = instantiate_from_config(model_config)
model_ckpt_path = infer_config.model_path
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
if IS_FLEXICUBES:
    model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
debug_path = os.path.join(args.output_path, config_name, 'debug')
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)
os.makedirs(debug_path, exist_ok=True)

if os.path.isdir(args.input_path):
    input_files = sorted(glob.glob(os.path.join(args.input_path, '*.*')))
    input_files = [f for f in input_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    input_files = [args.input_path]

print(f'Total number of input images: {len(input_files)}')

###############################################################################
# Stage 1: Camera setup + preprocessing session
###############################################################################

# IMPORTANT: keep base cameras immutable across samples. The old code modified
# input_cameras in-place when --view 4, which broke directory inference after
# the first sample.
base_input_cameras, camera_mode = load_input_cameras(
    args.camera_path,
    num_views=6,
    scale=args.scale,
    device=device,
)

chunk_size = 20 if IS_FLEXICUBES else 1
rembg_session = None if args.no_remove_background else rembg.new_session()

###############################################################################
# Stage 2: 3D Reconstruction
###############################################################################

for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).rsplit('.', 1)[0]
    print(f'[{idx + 1}/{len(input_files)}] Creating 3D object for {name} ...')

    image = Image.open(image_file).convert('RGB')
    image = crop_validation_top_half(image)

    # Save the exact raw grid after optional validation cropping.
    image.save(os.path.join(debug_path, f'{name}_00_raw_grid.png'))

    # 1) Split 3x2 grid into six views.
    raw_views = split_3x2_grid(image)
    make_debug_grid(raw_views).save(os.path.join(debug_path, f'{name}_01_split_raw.png'))

    # 2) Remove background + center with ONE shared multi-view scale.
    if args.no_remove_background:
        processed_views = [v.convert('RGB') for v in raw_views]
        print('  [preprocess] Background removal disabled.')
    else:
        processed_views, alpha_views = remove_background_and_center_views(
            raw_views,
            rembg_session=rembg_session,
            foreground_ratio=args.foreground_ratio,
            alpha_threshold=args.alpha_threshold,
        )
        make_debug_grid(alpha_views).save(os.path.join(debug_path, f'{name}_02_alpha_masks.png'))
        make_debug_grid(processed_views).save(os.path.join(debug_path, f'{name}_03_bg_removed_centered.png'))
        print(
            f'  [preprocess] rembg + white composite + shared-scale centering '
            f'(foreground_ratio={args.foreground_ratio:.2f})'
        )

    # 3) Reorder grid tiles into camera/model slot order.
    processed_views = [processed_views[i] for i in view_order]
    make_debug_grid(processed_views).save(os.path.join(debug_path, f'{name}_04_model_order.png'))

    # Camera handling:
    # - Zero123++ mode: view_order maps arbitrary grid tiles -> fixed model camera slots,
    #   so cameras themselves stay in the predefined slot order.
    # - Custom mode: custom camera rows correspond to the ORIGINAL grid tiles, so apply
    #   the same permutation to cameras when images are reordered.
    sample_cameras = base_input_cameras.clone()
    if camera_mode == 'custom':
        camera_order = torch.tensor(view_order, dtype=torch.long, device=device)
        sample_cameras = sample_cameras[:, camera_order]

    images = views_to_model_tensor(processed_views, size=320).to(device)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5], dtype=torch.long, device=device)
        images = images[:, indices]
        sample_cameras = sample_cameras[:, indices]

    print(f'  [input] images={tuple(images.shape)}, cameras={tuple(sample_cameras.shape)}')

    with torch.no_grad():
        # 4) Triplane reconstruction.
        planes = model.forward_planes(images, sample_cameras)

        # 5) FlexiCubes mesh extraction.
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')
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
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)

        print(f'  -> Mesh saved to {mesh_path_idx}')

        # 6) Optional circular-view rendering.
        if args.save_video:
            video_path_idx = os.path.join(video_path, f'{name}.mp4')
            render_size = infer_config.render_resolution
            render_cameras = get_render_cameras(
                batch_size=1,
                M=120,
                radius=args.distance,
                elevation=20.0,
                is_flexicubes=IS_FLEXICUBES,
            ).to(device)

            frames = render_frames(
                model,
                planes,
                render_cameras=render_cameras,
                render_size=render_size,
                chunk_size=chunk_size,
                is_flexicubes=IS_FLEXICUBES,
            )

            save_video(frames, video_path_idx, fps=30)
            print(f'  -> Video saved to {video_path_idx}')
