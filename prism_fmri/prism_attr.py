# PRISM 성분별 입력 기여 지도 (Pro6000, MinD-3D++ venv, 실행 위치 $W/run — src → MinD-3D/src)
#  11-10: 현재 모델(V = 범주 안 개체 형상, var_heads shape)이면 V 대상 = vt 출력 4차원(cy · w · h · area, 프레임 평균) → V_cy · V_w · V_h · V_area
#         이전 모델(center · scale · rot)이면 예전 그대로 V_center · V_scale · V_rot. 원본 = .보고 자료/261105_8차보고/PRISM_성분지도/scripts/prism_attr.py
#  각 성분 출력의 기울기를 fMRI 입력(프레임 2~7, 256×256 = 피질 평면 그림)까지 내려 |기울기 × 입력|을 픽셀마다 구한다.
#  test 물체마다 계산해 프레임·물체 평균 → 256×256 지도. 대상:
#    S        : Semantic 정답 범주 점수 (sem_clip · 정답 범주 텍스트 프로토타입)
#    V_center / V_scale / V_rot : Variational head 출력 (center x·y, scale, rot6d 6개; 프레임 평균, 차원별 따로 구해 합침)
#    P        : Perceptive → 정답 피험자 점수 (subj_logit)
#  출력 <out>/attr_<sub>.npz : 대상별 abs(|g·x| 평균), sgn(g·x 평균), grad(|g| 평균), 그리고 범주 예측 정확도
#  사용: python prism_attr.py --pr $PR --repo $W/MinD-3D --data_root $DR --ckpt <model_final.pt> --out <폴더> [--subs sub-0001] [--limit N]
import os, sys, time, types, argparse, numpy as np, torch

p = argparse.ArgumentParser()
p.add_argument('--pr', required=True); p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True)
p.add_argument('--subs', default='sub-0001'); p.add_argument('--split', default='test'); p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
sys.path.insert(0, '.'); sys.path.insert(0, a.pr)
from prism import build_from_ckpt

ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()
model.enc_grad = True                       # ViT를 지나 입력까지 기울기
print('load ok | temporal', getattr(model, 'temporal', None) is not None, '| subs', SUBS, flush=True)
PROTO = model.prototypes.float()
ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
ids = ids[:a.limit] if a.limit else ids
os.makedirs(a.out, exist_ok=True)


def targets(out, cls, si):
    """대상 이름 → 스칼라 (배치 1). V는 차원마다 따로 기울기를 구해 그룹 안에서 합친다."""
    logit = out['sem_clip'].float() @ PROTO.t()
    t = {'S': [logit[0, cls]], 'P': [out['subj_logit'].float()[0, si]]}
    if 'vt' in out:                          # 11-10 V = 범주 안 개체 형상 [cy, w, h, area]
        q = out['vt'].float().mean(1)[0]
        for j, nm in enumerate(VT_NAMES[:q.shape[0]]): t[f'V_{nm}'] = [q[j]]
    else:
        c = out['center'].float().mean(1)[0]; s = out['scale'].float().mean(1)[0]; r = out['rot6d'].float().mean(1)[0]
        t['V_center'] = [c[0], c[1]]; t['V_scale'] = [s]; t['V_rot'] = [r[j] for j in range(6)]
    return t


VT_NAMES = ['cy', 'w', 'h', 'area']
_vh = str((ck.get('args') or {}).get('var_heads', 'csr'))
NAMES = ['S'] + ([f'V_{n}' for n in VT_NAMES] if _vh in ('shape', 'sil') else ['V_center', 'V_scale', 'V_rot']) + ['P']
print('대상', NAMES, flush=True)
for sub in a.subs.split(','):
    si = SUBS.index(sub) if sub in SUBS else 0
    acc = {n: {k: np.zeros((256, 256), np.float64) for k in ('abs', 'sgn', 'grad')} for n in NAMES}
    hit = 0; t0 = time.time()
    for i, k in enumerate(ids):
        x = torch.from_numpy(np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES])[None].cuda()
        x.requires_grad_(True)
        fi = torch.tensor(FRAMES)[None].cuda()
        out = model(x, fi, torch.tensor([si]).cuda())        # fp32 (autocast 안 씀 — 기울기 정밀도)
        cls = SYN.index(k.split('/')[0])
        hit += int((out['sem_clip'].float() @ PROTO.t()).argmax(-1).item() == cls)
        tg = targets(out, cls, si); flat = [(n, v) for n in NAMES for v in tg[n]]
        for j, (n, v) in enumerate(flat):
            g, = torch.autograd.grad(v, x, retain_graph=j < len(flat) - 1)
            g = g[0].double(); xx = x.detach()[0].double()
            acc[n]['abs'] += (g * xx).abs().mean(0).cpu().numpy()
            acc[n]['sgn'] += (g * xx).mean(0).cpu().numpy()
            acc[n]['grad'] += g.abs().mean(0).cpu().numpy()
        if (i + 1) % 10 == 0 or i + 1 == len(ids):
            print(f'{sub} {i + 1}/{len(ids)} {time.time() - t0:.0f}s 범주 정답 {hit}/{i + 1}', flush=True)
    res = {f'{n}_{kk}': (acc[n][kk] / len(ids)).astype(np.float32) for n in NAMES for kk in acc[n]}
    np.savez(f'{a.out}/attr_{sub}.npz', names=np.array(ids), cls_acc=hit / len(ids), frames=np.array(FRAMES), ckpt=a.ckpt, **res)
    print(f'{sub}: {len(ids)}개, 범주 정확도 {hit / len(ids):.3f} → {a.out}/attr_{sub}.npz', flush=True)
print('ATTR_DONE', flush=True)
