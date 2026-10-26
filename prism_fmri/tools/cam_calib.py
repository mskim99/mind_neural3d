# InstantMesh 영상 자세 카메라 보정 시험 (Pro6000, GPU 1). 모델 없이 GT 자극 영상 프레임으로 InstantMesh 입력 카메라를 고른다.
#  입력: 영상 0·64·…·320번(60°마다) 6장 = 학습 GT와 같은 프레임 / 판정: 그 사이 32·96·…·352번 6장 (입력에 안 넣은 각도)
#  후보마다 재구성 → 같은 카메라 규칙으로 입력 각도·판정 각도에서 실루엣 렌더 → 영상 실루엣과 IoU
#  회전 방향(dir ±1)은 둘 다 해 봄. 모델 출력에는 판정 프레임이 없으므로 "입력 각도 IoU로 방향 고르기"가 판정 기준과 맞는지도 같이 봄
#  렌더 좌우·상하 뒤집힘 규약은 모르는 값이라 4가지 다 계산 (입력 IoU가 가장 높은 규약이 맞는 것)
# 실행 위치: MinD-3D/InstantMesh (src 불러오기). 사용: python cam_calib.py ../configs/mind3d_pp_infer.yaml --lrm_ckpt … --frames … --out …csv
import os, sys, csv, glob, time, argparse
import numpy as np, torch
from PIL import Image
from omegaconf import OmegaConf
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import spherical_camera_pose, FOV_to_intrinsics

p = argparse.ArgumentParser()
p.add_argument('config'); p.add_argument('--lrm_ckpt', required=True)
p.add_argument('--frames', required=True)          # <해시>/{00..11}.png (prep_calib_frames.py)
p.add_argument('--out', required=True); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
dev = torch.device('cuda')
cfg = OmegaConf.load(a.config)
model = instantiate_from_config(cfg.model_config)
st = torch.load(a.lrm_ckpt, map_location='cpu')['state_dict']
model.load_state_dict({k[14:]: v for k, v in st.items() if k.startswith('lrm_generator.')}, strict=True)
model = model.to(dev).eval()

R = 320
# (이름, fov, 거리, 고도 규칙). 영상: Blender 거리 1.82·렌즈 35mm(FOV 54.4°). v1.2 학습 GT 보정(거리 2.75 ↔ InstantMesh 4.0)으로 환산하면 FOV 54.4 → 2.65, FOV 30 → 5.06 (추정)
CANDS = [('v12_now', 30.0, 4.0, 'v12')] + [(f'vid_f{f:g}_r{r:g}', f, r, 'vid') for f, r in
         [(30, 4.4), (30, 5.06), (30, 5.7), (40, 3.73), (54.4, 2.3), (54.4, 2.65), (54.4, 3.0)]]

def cams16(az, el, r, fov):
    c2w = spherical_camera_pose(np.array(az, float), np.array(el, float), r).float().flatten(-2)
    K = FOV_to_intrinsics(fov).unsqueeze(0).repeat(len(az), 1, 1).float().flatten(-2)
    return torch.cat([c2w[:, :12], torch.stack([K[:, 0], K[:, 4], K[:, 2], K[:, 5]], -1)], -1).unsqueeze(0).to(dev)

def w2c(az, el, r):
    return torch.linalg.inv(spherical_camera_pose(np.array(az, float), np.array(el, float), r).float()).unsqueeze(0).to(dev)

def angles(rule, d):
    if rule == 'v12':                                   # 지금 방식: 영상 프레임을 v1.2 자세라고 가정
        return [30 + 60 * i for i in range(6)], [20, -10] * 3, [60 + 60 * i for i in range(6)], [0] * 6
    return [d * 60 * i for i in range(6)], [0] * 6, [d * (30 + 60 * i) for i in range(6)], [0] * 6

def sil(im):                                            # 흰 배경 영상 → 실루엣
    return np.asarray(im.convert('RGB')).min(-1) < 235

FLIPS = {'none': lambda m: m, 'ud': lambda m: m[::-1], 'lr': lambda m: m[:, ::-1], 'both': lambda m: m[::-1, ::-1]}
def iou(m, s):
    u = (m | s).sum(); return float((m & s).sum() / u) if u else 0.0

objs = sorted(os.listdir(a.frames))[: a.limit or None]
fo = open(a.out, 'w', newline=''); w = csv.writer(fo); w.writerow(['obj', 'cand', 'dir', 'flip', 'iou_in', 'iou_ho'])
t0 = time.time(); cur_fov = None
for oi, k in enumerate(objs):
    ims = [Image.open(os.path.join(a.frames, k, f'{j:02d}.png')) for j in range(12)]
    x = torch.from_numpy(np.stack([np.asarray(ims[j].convert('RGB'), np.float32) / 255. for j in range(0, 12, 2)])).permute(0, 3, 1, 2).unsqueeze(0).to(dev)
    s_in = [sil(ims[j]) for j in range(0, 12, 2)]; s_ho = [sil(ims[j]) for j in range(1, 12, 2)]
    for name, fov, r, rule in CANDS:
        if fov != cur_fov:
            model.init_flexicubes_geometry(dev, fovy=fov); cur_fov = fov
        for d in ([1] if rule == 'v12' else [1, -1]):
            az_i, el_i, az_h, el_h = angles(rule, d)
            with torch.no_grad():
                planes = model.forward_planes(x, cams16(az_i, el_i, r, fov))
                out = model.forward_geometry(planes, w2c(az_i + az_h, el_i + el_h, r), render_size=R)
            m = (out['mask'][0, :, 0].float().cpu().numpy() > 0.5)
            for fl, f in FLIPS.items():
                w.writerow([k, name, d, fl, f'{np.mean([iou(f(m[v]), s_in[v]) for v in range(6)]):.4f}',
                            f'{np.mean([iou(f(m[6 + v]), s_ho[v]) for v in range(6)]):.4f}'])
    fo.flush()
    print(f'[calib {oi + 1}/{len(objs)}] {k} {time.time() - t0:.0f}s', flush=True)

# 요약: 뒤집힘 규약은 전체 입력 IoU 평균이 가장 높은 것. 방향은 (가) 판정 IoU로 고른 것(정답 쪽) (나) 입력 IoU로 고른 것(모델 출력에 쓸 방법)
fo.close()
import collections
rows = list(csv.DictReader(open(a.out)))
fl = max(FLIPS, key=lambda f: np.mean([float(r['iou_in']) for r in rows if r['flip'] == f]))
print(f'\n== 렌더 뒤집힘 규약: {fl}')
by = collections.defaultdict(dict)
for r in rows:
    if r['flip'] == fl: by[(r['cand'], r['obj'])][int(r['dir'])] = (float(r['iou_in']), float(r['iou_ho']))
print(f'{"후보":16s} 판정IoU(방향=판정으로)  판정IoU(방향=입력으로)  입력IoU  방향 일치율')
for name, *_ in CANDS:
    o = [by[(name, k)] for k in objs]
    best_ho = np.mean([max(v.values(), key=lambda t: t[1])[1] for v in o])
    sel_in = [max(v.values(), key=lambda t: t[0]) for v in o]
    agree = np.mean([max(v, key=lambda dd: v[dd][1]) == max(v, key=lambda dd: v[dd][0]) for v in o])
    print(f'{name:16s} {best_ho:.4f}                 {np.mean([t[1] for t in sel_in]):.4f}                 {np.mean([t[0] for t in sel_in]):.4f}   {agree:.2f}')
print('CALIB_DONE', flush=True)
