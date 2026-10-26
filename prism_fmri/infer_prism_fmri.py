# PRISM fMRI판 일괄 추론 — Pro6000 MinD-3D++ 추론 스크립트(infer_mind3dpp_batch.py)를 바탕으로 바꾼 것만:
#  ① fMRI 인코더 → PrismFMRIEncoder (의미·변형 헤드가 공식 파이프라인의 forward_encoder_wo_pred 안에서 적용됨 → 파이프라인 수정 없음)
#  ② 6뷰를 TRELLIS 입력 형식으로도 저장: views/<이름>/00.png … 05.png (320×320, 흰 배경) → meshgen run_trellis.py --views-per-object 6
#  ③ --no_lrm: InstantMesh 재구성 건너뛰기 (TRELLIS만 쓸 때)
# 실행 위치: MinD-3D/InstantMesh (공식 InstantMesh/src, configs 사용)
#   python <이 파일> ../configs/mind3d_pp_infer.yaml --dataset obj --list .../test_list.txt --fmri_dir .../sub-0001/npy_data \
#       --unet_path <학습 결과 model_final.pt> --merge_lora --z123 ... --lrm_ckpt ... --out_dir ...
import os, sys, csv, time, argparse
import numpy as np, torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj_with_mtl
from prism_encoder import PrismFMRIEncoder

p = argparse.ArgumentParser()
p.add_argument('config')
p.add_argument('--dataset', choices=['shape', 'obj'], required=True)
p.add_argument('--list', required=True)
p.add_argument('--fmri_dir', required=True)
p.add_argument('--unet_path', required=True)        # PRISM 학습 결과(model_final.pt) 또는 MinD-3D++ 공개 가중치(헤드 없음 → 항등으로 동작)
p.add_argument('--z123', required=True)
p.add_argument('--lrm_ckpt', default='')
p.add_argument('--out_dir', required=True)
p.add_argument('--merge_lora', action='store_true')
p.add_argument('--lora_scale', type=float, default=1.0)
p.add_argument('--limit', type=int, default=0)
p.add_argument('--diffusion_steps', type=int, default=75)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--scale', type=float, default=1.0)
p.add_argument('--no_lrm', action='store_true')
p.add_argument('--cam', choices=['v12', 'video'], default='v12')   # InstantMesh 입력 카메라. v12 = 공식(방위 30~330·고도 20/-10·거리 4·FOV 30)
p.add_argument('--cam_radius', type=float, default=5.06)          # video: 고도 0·방위 0,60..300 (방향 자동), FOV·거리는 보정 시험(10-26) 권장값
p.add_argument('--cam_fov', type=float, default=30.0)
p.add_argument('--lrm_only', action='store_true')                 # images/<이름>.png가 있으면 6뷰 생성을 건너뛰고 3D만
p.add_argument('--mesh_dir', default='meshes')
p.add_argument('--base_unet', default='')           # 학습 중 체크포인트(model_latest.pt: 인코더·LoRA만)일 때 본체 UNet (학습 초기값 diffusion_pytorch_model.bin)
a = p.parse_args()
seed_everything(a.seed)

config = OmegaConf.load(a.config)
device = torch.device('cuda')

pipeline = DiffusionPipeline.from_pretrained(a.z123, custom_pipeline="src/zero123plus", torch_dtype=torch.float16)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config, timestep_spacing='trailing')
pipeline.fmri_encoder = PrismFMRIEncoder(config.model.params.fmri_encoder_config).to(torch.float16).cuda()

sd = torch.load(a.unet_path, map_location='cpu')["model"]
fm = {k.replace("fmri_encoder.", "", 1): v for k, v in sd.items() if k.startswith("fmri_encoder.")}
un = {k.replace("unet.base_model.model.unet.", ""): v for k, v in sd.items() if "unet.base_model.model.unet" in k}
un = {k.replace('.base_layer.', '.'): v for k, v in un.items()}
if a.base_unet:                                       # 체크포인트에 UNet 본체가 없으면 학습 초기값 UNet을 바탕으로
    pipeline.unet.load_state_dict(torch.load(a.base_unet, map_location='cpu'), strict=True)
    print(f'base unet: {a.base_unet}', flush=True)
base_sd = pipeline.unet.state_dict()
n_fill = 0
for k in [k for k in un if '.lora_A.default.weight' in k]:
    b = k.replace('.lora_A.default.weight', '.weight')
    if b not in un:
        un[b] = base_sd[b].clone(); n_fill += 1
if n_fill:
    print(f'UNet 본체가 체크포인트에 없음 → 현재 UNet에서 {n_fill}개 채움 (--base_unet {"있음" if a.base_unet else "없음: zero123plus 본체"})', flush=True)
n_lora = 0
if a.merge_lora:
    for k in [k for k in un if '.lora_A.default.weight' in k]:
        base = k.replace('.lora_A.default.weight', '.weight')
        B = un[k.replace('lora_A', 'lora_B')].float(); A = un[k].float()
        un[base] = (un[base].float() + (B @ A) * a.lora_scale).to(un[base].dtype); n_lora += 1
un = {k: v for k, v in un.items() if 'lora_' not in k}
r = pipeline.fmri_encoder.load_state_dict(fm, strict=False)
heads_missing = [k for k in r.missing_keys if k.split('.')[0] in ('sem_head', 'var_head', 'subj_head')]
other_missing = [k for k in r.missing_keys if k not in heads_missing]
print(f'fmri_encoder: missing core {len(other_missing)}, heads {"없음(항등 = MinD-3D++와 같음)" if heads_missing else "불러옴"}, unexpected {len(r.unexpected_keys)}', flush=True)
assert not other_missing, other_missing[:5]
r = pipeline.unet.load_state_dict(un, strict=False)
print(f'unet: missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)} | merge_lora={a.merge_lora} merged={n_lora}', flush=True)
del sd
pipeline = pipeline.to(device)

lines = [l.strip() for l in open(a.list) if l.strip()]
if a.dataset == 'obj':
    lines = lines[1:]
items = []
for l in lines:
    if a.dataset == 'shape':
        items.append((l.split('/')[-1], os.path.join(a.fmri_dir, 'npy_data', f'{l}.npy')))
    else:
        name = l.split('/')[-1]
        items.append((name, os.path.join(a.fmri_dir, name.replace('.glb', '') + '.npy')))
if a.limit:
    items = items[:a.limit]

for d in ('images', a.mesh_dir, 'views'):
    os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)
logp = os.path.join(a.out_dir, '_timing.csv'); new = not os.path.exists(logp)
logf = open(logp, 'a', newline=''); w = csv.writer(logf)
if new:
    w.writerow(['name', 'sec_mv', 'sec_lrm', 'peak_mem_gb'])

def save_views(grid, name):
    """3×2 격자(640×960) → TRELLIS 입력 폴더 views/<이름>/00.png … 05.png"""
    arr = np.asarray(grid.convert('RGB'))
    h, wd = arr.shape[0] // 3, arr.shape[1] // 2
    od = os.path.join(a.out_dir, 'views', name.replace('.glb', ''))
    os.makedirs(od, exist_ok=True)
    for i in range(6):
        r_, c_ = divmod(i, 2)
        Image.fromarray(arr[r_ * h:(r_ + 1) * h, c_ * wd:(c_ + 1) * wd]).save(os.path.join(od, f'{i:02d}.png'))

sec_mv = {}
for i, (name, npy) in enumerate(items):
    t0 = time.time()
    if a.lrm_only and os.path.exists(os.path.join(a.out_dir, 'images', f'{name}.png')):
        sec_mv[name] = 0.0; continue
    x = np.load(npy)[2:8]
    if len(x) < 6:
        x = np.concatenate([x, x[-(6 - len(x)):]])
    x = torch.from_numpy(x).unsqueeze(0).to(torch.float16).cuda()
    out = pipeline(x, num_inference_steps=a.diffusion_steps).images[0]
    out.save(os.path.join(a.out_dir, 'images', f'{name}.png'))
    save_views(out, name)
    sec_mv[name] = time.time() - t0
    print(f'[mv {i+1}/{len(items)}] {name} {sec_mv[name]:.1f}s', flush=True)
mv_peak = torch.cuda.max_memory_allocated() / 1e9
del pipeline; torch.cuda.empty_cache()

if a.no_lrm:
    for name in sec_mv:
        w.writerow([name, f'{sec_mv[name]:.2f}', '', f'{mv_peak:.2f}'])
    print('ALL_DONE (6뷰만)', flush=True); sys.exit(0)

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
    w.writerow([name, f'{sec_mv[name]:.2f}', f'{t2-t1:.2f}', f'{max(mv_peak, torch.cuda.max_memory_allocated()/1e9):.2f}']); logf.flush()
    print(f'[lrm {i+1}/{len(items)}] {name} {t2-t1:.1f}s', flush=True)
print('ALL_DONE', flush=True)
