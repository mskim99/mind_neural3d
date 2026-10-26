# MinD-3D++ 기준선 전용: 6뷰 격자(images/<이름>.png) → InstantMesh 3D. 제안 기법(PRISM)에는 쓰지 않는다 (PRISM 3D = TRELLIS, 사용자 10-26)
#  --cam v12   : 공식 카메라 (방위 30~330·고도 20/-10·거리 4·FOV 30) — 공개 가중치·v1.2 GT 학습 모델
#  --cam video : 영상 프레임 GT로 학습한 모델용. 고도 0·방위 0,60..300, FOV 30·거리 5.06, 회전 방향은 입력 6뷰 실루엣 IoU로 물체마다 선택
#                (카메라 보정 시험 10-26, tools/cam_calib.py — 원문에 없는 설정)
# 실행 위치: MinD-3D/InstantMesh.  python tools/instantmesh_baseline.py ../configs/mind3d_pp_infer.yaml --lrm_ckpt … --out_dir <images/가 있는 폴더> [--cam video]
import os, sys, csv, time, argparse
import numpy as np, torch
from PIL import Image
from torchvision.transforms import v2
from omegaconf import OmegaConf
from einops import rearrange
sys.path.insert(0, '.')
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras, spherical_camera_pose, FOV_to_intrinsics
from src.utils.mesh_util import save_obj_with_mtl

p = argparse.ArgumentParser()
p.add_argument('config'); p.add_argument('--lrm_ckpt', required=True); p.add_argument('--out_dir', required=True)
p.add_argument('--cam', choices=['v12', 'video'], default='v12')
p.add_argument('--cam_radius', type=float, default=5.06); p.add_argument('--cam_fov', type=float, default=30.0)
p.add_argument('--scale', type=float, default=1.0)
p.add_argument('--mesh_dir', default='meshes'); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
config = OmegaConf.load(a.config); device = torch.device('cuda')
names = sorted(f[:-4] for f in os.listdir(os.path.join(a.out_dir, 'images')) if f.endswith('.png'))[: a.limit or None]
items = [(n, None) for n in names]
os.makedirs(os.path.join(a.out_dir, a.mesh_dir), exist_ok=True)
logf = open(os.path.join(a.out_dir, f'{a.mesh_dir}_timing.csv'), 'a', newline=''); w = csv.writer(logf)
sec_mv = {n: 0.0 for n in names}; mv_peak = 0.0

infer_kwargs = dict(config.infer_config); infer_kwargs.pop('unet_path', None); infer_kwargs['model_path'] = a.lrm_ckpt
model = instantiate_from_config(config.model_config)
st = torch.load(a.lrm_ckpt, map_location='cpu')['state_dict']
model.load_state_dict({k[14:]: v for k, v in st.items() if k.startswith('lrm_generator.')}, strict=True)
model = model.to(device); model.init_flexicubes_geometry(device, fovy=30.0); model = model.eval()
from src.utils.camera_util import spherical_camera_pose, FOV_to_intrinsics

def cams16(az, el, r, fov):
    c2w = spherical_camera_pose(np.array(az, float), np.array(el, float), r).float().flatten(-2)
    K = FOV_to_intrinsics(fov).unsqueeze(0).repeat(len(az), 1, 1).float().flatten(-2)
    return torch.cat([c2w[:, :12], torch.stack([K[:, 0], K[:, 4], K[:, 2], K[:, 5]], -1)], -1).unsqueeze(0).to(device)

if a.cam == 'v12':
    input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0 * a.scale).to(device)
else:   # 영상 자세: 고도 0, 방위 0·60·…·300, 회전 방향은 물체마다 입력 6뷰 실루엣 IoU로 고름 (GT 미사용, tools/cam_calib.py와 같은 기준)
    model.init_flexicubes_geometry(device, fovy=a.cam_fov)
    AZ = {d: [d * 60 * i for i in range(6)] for d in (1, -1)}
    VCAM = {d: cams16(AZ[d], [0] * 6, a.cam_radius, a.cam_fov) for d in (1, -1)}
    VW2C = {d: torch.linalg.inv(spherical_camera_pose(np.array(AZ[d], float), np.zeros(6), a.cam_radius).float()).unsqueeze(0).to(device) for d in (1, -1)}
    dirf = open(os.path.join(a.out_dir, f'{a.mesh_dir}_dir.csv'), 'a', newline=''); dw = csv.writer(dirf)
for i, (name, npy) in enumerate(items):
    mp = os.path.join(a.out_dir, a.mesh_dir, f'{name}.obj')
    if os.path.exists(mp):
        continue
    torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    out = Image.open(os.path.join(a.out_dir, 'images', f'{name}.png'))
    imgs = torch.from_numpy(np.asarray(out, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous().float()
    imgs = rearrange(imgs, 'c (n h) (m w) -> (n m) c h w', n=3, m=2).unsqueeze(0).to(device)
    imgs = v2.functional.resize(imgs, 320, interpolation=3, antialias=True).clamp(0, 1)
    with torch.no_grad():
        if a.cam == 'v12':
            planes = model.forward_planes(imgs, input_cameras)
        else:
            sil = (imgs[0].min(1)[0] < 235 / 255.).cpu().numpy()          # 생성 6뷰 실루엣 (흰 배경)
            best = None
            for d in (1, -1):
                pl = model.forward_planes(imgs, VCAM[d])
                m = model.forward_geometry(pl, VW2C[d], render_size=320)['mask'][0, :, 0].float().cpu().numpy() > 0.5
                iou = float(np.mean([(m[v_] & sil[v_]).sum() / max((m[v_] | sil[v_]).sum(), 1) for v_ in range(6)]))
                if best is None or iou > best[0]: best = (iou, d, pl)
            planes = best[2]; dw.writerow([name, best[1], f'{best[0]:.4f}']); dirf.flush()
        v, f, uvs, tidx, tex = model.extract_mesh(planes, use_texture_map=True, **infer_kwargs)
        save_obj_with_mtl(v.data.cpu().numpy(), uvs.data.cpu().numpy(), f.data.cpu().numpy(),
                          tidx.data.cpu().numpy(), tex.permute(1, 2, 0).data.cpu().numpy(), mp)
    t2 = time.time()
    w.writerow([name, f'{t2-t1:.2f}', f'{max(mv_peak, torch.cuda.max_memory_allocated()/1e9):.2f}']); logf.flush()
    print(f'[lrm {i+1}/{len(items)}] {name} {t2-t1:.1f}s', flush=True)
print('ALL_DONE', flush=True)
