import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm

# 기존 프로젝트의 유틸리티 모듈 유지
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics,
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import save_video


def get_render_cameras(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False):
    """
    Get the rendering camera parameters.
    """
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
    """
    Render frames from triplanes.
    """
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

    frames = torch.cat(frames, dim=1)[0]
    return frames


###############################################################################
# Arguments
###############################################################################

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--input_path', type=str, required=True,
                    help='Path to a single 6-view grid image or a directory of images.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output directory.')
parser.add_argument('--save_name', type=str, default='', help='Prefix for the output directory.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale of generated object.')
parser.add_argument('--distance', type=float, default=4.5, help='Render distance.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of input views.')
parser.add_argument('--export_texmap', action='store_true', help='Export a mesh with texture map.')
parser.add_argument('--save_video', action='store_true', help='Save a circular-view video.')
args = parser.parse_args()

seed_everything(args.seed)

###############################################################################
# Stage 0: Configuration & Model Loading
###############################################################################

config = OmegaConf.load(args.config)
config_name = os.path.basename(args.config).replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config

IS_FLEXICUBES = True
if args.save_name:
    config_name = config_name + "_" + args.save_name
device = torch.device('cuda')

# 모델 로드 (Diffusion 모델 제외, 복원 모델만 로드)
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

# 출력 디렉토리 생성
mesh_path = os.path.join(args.output_path, config_name, 'meshes')
video_path = os.path.join(args.output_path, config_name, 'videos')
os.makedirs(mesh_path, exist_ok=True)
os.makedirs(video_path, exist_ok=True)

# 입력 파일 목록 구성 (단일 파일 또는 디렉토리)
if os.path.isdir(args.input_path):
    input_files = glob.glob(os.path.join(args.input_path, '*.*'))
    input_files = [f for f in input_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    input_files = [args.input_path]

print(f'Total number of input images: {len(input_files)}')

###############################################################################
# Stage 1: 3D Reconstruction
###############################################################################

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * args.scale).to(device)
chunk_size = 20 if IS_FLEXICUBES else 1

for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx + 1}/{len(input_files)}] Creating 3D object for {name} ...')

    # 1. 6시점 그리드 이미지 로드 및 텐서 변환 (기존 파이프라인의 후처리 방식 차용)
    image = Image.open(image_file).convert('RGB')
    images_np = np.asarray(image, dtype=np.float32) / 255.0
    images_tensor = torch.from_numpy(images_np).permute(2, 0, 1).contiguous().float()  # (3, 960, 640)
    images_tensor = rearrange(images_tensor, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)  # (6, 3, 320, 320)

    images = images_tensor.unsqueeze(0).to(device)
    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # 2. Triplane 추출
        planes = model.forward_planes(images, input_cameras)

        # 3. 메쉬 추출 및 저장
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
        print(f"  -> Mesh saved to {mesh_path_idx}")

        # 4. (옵션) 비디오 렌더링
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

            save_video(
                frames,
                video_path_idx,
                fps=30,
            )
            print(f"  -> Video saved to {video_path_idx}")