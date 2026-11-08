# 외형 단계 분리용 검색 (11-09, CPU): test 물체마다 외형(SLat) 조건으로 쓸 실제 train 토큰 하나를 고름
#  형상(SS) = fMRI 예측 토큰 그대로, 외형(SLat) = 예측 범주(Semantic top-1) 안 train 정답 토큰 중 예측 토큰과 가장 가까운 것
#  거리: 토큰마다 affine 없는 layer_norm → 토큰 평균 → train 평균 빼기 → cos (평균을 안 빼면 모두 cos ≈ 1)
#  평균 토큰이 아니라 실제 토큰 하나를 그대로 씀 (실제 분포 안의 조건 → TRELLIS 외형 디코더가 정상 색을 내는지 확인)
#  출력: <out>/<test uid>.npy = 고른 train 토큰 파일로 가는 symlink, <out>/slat_map.json (test uid → train uid · cos · 범주)
# 사용: python slat_retrieve.py --pred <예측 tokens> --comp <components.npz> --tgt <shape_targets> --data_root <fMRI-Shape> --out <폴더>
import os, json, shutil, argparse, numpy as np
p = argparse.ArgumentParser()
p.add_argument('--pred', required=True); p.add_argument('--comp', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--data_root', required=True); p.add_argument('--out', required=True)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
lst = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
tr = [k for k in lst('train') if os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy')]


def pool(f):
    t = np.load(f).astype(np.float32)
    t = (t - t.mean(-1, keepdims=True)) / np.sqrt(t.var(-1, keepdims=True) + 1e-5)
    return t.mean(0)


TV = np.stack([pool(f'{a.tgt}/dino/{k.split("/")[1]}.npy') for k in tr]); MU = TV.mean(0)
nz = lambda v: (v - MU) / (np.linalg.norm(v - MU, axis=-1, keepdims=True) + 1e-8)
TN = nz(TV); TCAT = np.array([k.split('/')[0] for k in tr])
c = np.load(a.comp); classes = list(c['classes']); names = list(c['name']); cp = c['cls_pred']; cl = c['cls']
M, hit = {}, 0
for k, pc, gc in zip(names, cp, cl):
    u = k.split('/')[1]; f = f'{a.pred}/{u}.npy'
    if not os.path.exists(f): print('없음', f, flush=True); continue
    cat = classes[int(pc)]; idx = np.where(TCAT == cat)[0]
    s = TN[idx] @ nz(pool(f)); j = idx[int(s.argmax())]; tu = tr[j].split('/')[1]
    o = f'{a.out}/{u}.npy'
    if os.path.lexists(o): os.unlink(o)
    try: os.symlink(os.path.abspath(f'{a.tgt}/dino/{tu}.npy'), o)
    except OSError: shutil.copy(f'{a.tgt}/dino/{tu}.npy', o)               # symlink 권한 없는 환경 (Windows)
    M[u] = {'train': tr[j], 'cos': round(float(s.max()), 4), 'pred_cat': cat, 'gt_cat': classes[int(gc)]}; hit += int(pc == gc)
json.dump(M, open(f'{a.out}/slat_map.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(f'SLATRET {len(M)}개 | 범주 일치 {hit / max(len(M), 1):.3f} | cos 평균 {np.mean([v["cos"] for v in M.values()]):.3f} | train 후보 {len(tr)}', flush=True)
print('SLATRET_DONE', flush=True)
