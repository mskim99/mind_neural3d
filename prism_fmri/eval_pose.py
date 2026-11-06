# PRISM Variational 자세 일치도 (11-05, 11-06 var_heads sr 지원, MinD-3D++ venv, 실행 위치 $W/run)
#  test fMRI → Variational head 출력을 GT와 비교 (fMRI 프레임 2~7 ↔ 영상 프레임 round(19.2k))
#   var_heads csr: center (cx, cy) · scale · 회전 vs 영상 / sr: scale vs 영상 + 크기 비율 x·y·z (프레임 평균) vs GT 64³ 복셀 범위 (train_prism.py와 같은 식)
#  비교 기준선: 프레임 번호마다 train GT 평균을 그대로 냈을 때 (= fMRI를 안 보고도 나오는 오차)
#   - 회전 GT는 모든 영상이 같은 카메라 궤도라 프레임 번호만으로 정해짐 → 기준선 회전 오차는 0°에 가까움 (회전 head는 fMRI 정보를 거의 안 배워도 됨)
#  출력 <out>/pose_<sub>.csv (물체 × 프레임마다 GT·예측·오차), <out>/pose_<sub>.json (요약), 로그에 POSE 줄
# 사용: python eval_pose.py --repo <MinD-3D> --data_root <fMRI-Shape> --tgt <shape_targets> --ckpt <model_final.pt> --out <폴더> [--subs sub-0001]
import os, sys, math, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt, rot6d_to_matrix
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default='sub-0001'); p.add_argument('--bs', type=int, default=8)
a = p.parse_args()
FRAMES = list(range(2, 8)); VF = [min(191, round(19.2 * f)) for f in FRAMES]


def rot_target(az_deg, elev_deg=30.0):          # train_prism.py와 같은 식
    t, e = math.radians(-az_deg), math.radians(elev_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return (Rx @ Rz).astype(np.float32)


def gt(uid):
    t = np.load(f'{a.tgt}/targets/{uid}.npz')
    return (np.stack([t['cx'][VF], t['cy'][VF]], -1), t['scale'][VF], np.stack([rot_target(float(t['az_deg'][j])) for j in VF]))


def ratio_gt(uid):                              # train_prism.py ratio_target과 같은 식 (없으면 None)
    f = f'{a.tgt}/voxels/{uid}.npy'
    if not os.path.exists(f): return None
    occ = np.unpackbits(np.load(f))[:64 ** 3].reshape(64, 64, 64)
    return np.array([(lambda nz: (nz.max() - nz.min() + 1) / 64 if len(nz) else 0.0)(np.nonzero(occ.any(axis=tuple(j for j in range(3) if j != ax)))[0]) for ax in range(3)])


def rot_err_deg(R1, R2):                         # 두 회전 사이 각도 [°]
    c = (np.einsum('...ij,...ij->...', R1, R2) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


ids = lambda sp: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{sp}_list.txt')) if l.strip()]
# 기준선: 프레임 번호마다 train GT 평균
tr = [gt(k.split('/')[1]) for k in ids('train') if os.path.exists(f'{a.tgt}/targets/{k.split("/")[1]}.npz')]
mc = np.mean([x[0] for x in tr], 0); ms = np.mean([x[1] for x in tr], 0)
mR = np.mean([x[2] for x in tr], 0); U, _, Vt = np.linalg.svd(mR); mR = U @ Vt     # 평균 회전 (가장 가까운 회전 행렬)

ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']
model = build_from_ckpt(ck, a.repo).cuda().eval()
VH = (ck.get('args') or {}).get('var_heads', 'csr')
print('load ok | var_mode', (ck.get('args') or {}).get('var_mode', 'feat'), '| var_heads', VH, flush=True)
if VH == 'sr':                                  # 크기 비율 기준선 = train 평균
    trq = [r for r in (ratio_gt(k.split('/')[1]) for k in ids('train')) if r is not None]; mq = np.mean(trq, 0)
os.makedirs(a.out, exist_ok=True); test = ids('test')
for sub in a.subs.split(','):
    si = SUBS.index(sub); rows = []
    for i in range(0, len(test), a.bs):
        ks = test[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x, fi, torch.full((len(ks),), si, dtype=torch.long, device='cuda'))
        ps = o['scale'].float().cpu().numpy()
        if VH == 'sr':
            pq = o['ratio'].float().mean(1).cpu().numpy()                       # [B, 3] 프레임 평균
            for j, k in enumerate(ks):
                gc, gs, gR = gt(k.split('/')[1]); gq = ratio_gt(k.split('/')[1])
                for f in range(len(FRAMES)):
                    r = dict(uid=k, frame=FRAMES[f], s_gt=gs[f], s_pred=ps[j, f], s_base=ms[f])
                    for xi, ax in enumerate('xyz'):
                        r.update({f'q{ax}_gt': np.nan if gq is None else gq[xi], f'q{ax}_pred': pq[j, xi], f'q{ax}_base': mq[xi]})
                    rows.append(r)
            continue
        pc = o['center'].float().cpu().numpy(); pR = rot6d_to_matrix(o['rot6d'].float()).cpu().numpy()
        for j, k in enumerate(ks):
            gc, gs, gR = gt(k.split('/')[1])
            for f in range(len(FRAMES)):
                rows.append(dict(uid=k, frame=FRAMES[f], cx_gt=gc[f, 0], cx_pred=pc[j, f, 0], cy_gt=gc[f, 1], cy_pred=pc[j, f, 1],
                                 s_gt=gs[f], s_pred=ps[j, f], rot_err=float(rot_err_deg(pR[j, f], gR[f])),
                                 cx_base=mc[f, 0], cy_base=mc[f, 1], s_base=ms[f], rot_err_base=float(rot_err_deg(mR[f], gR[f]))))
    with open(f'{a.out}/pose_{sub}.csv', 'w') as fh:
        fh.write(','.join(rows[0]) + '\n')
        for r in rows: fh.write(','.join(str(v) if isinstance(v, str) else f'{v:.5f}' for v in r.values()) + '\n')
    A = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in rows[0] if k != 'uid'}
    corr = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if np.std(u) > 0 and np.std(v) > 0 else float('nan')
    if VH == 'sr':
        S = {'n': len(rows), 's_mae': float(np.abs(A['s_pred'] - A['s_gt']).mean()), 's_mae_base': float(np.abs(A['s_base'] - A['s_gt']).mean()), 's_r': corr(A['s_pred'], A['s_gt'])}
        ok = ~np.isnan(A['qx_gt'])                                                # 복셀 없는 물체 뺌
        for ax in 'xyz':
            g, pr, b = A[f'q{ax}_gt'][ok], A[f'q{ax}_pred'][ok], A[f'q{ax}_base'][ok]
            S.update({f'q{ax}_mae': float(np.abs(pr - g).mean()), f'q{ax}_mae_base': float(np.abs(b - g).mean()), f'q{ax}_r': corr(pr, g)})
        json.dump(S, open(f'{a.out}/pose_{sub}.json', 'w'), indent=1)
        print(f"POSE {sub} | scale MAE {S['s_mae']:.4f} (기준 {S['s_mae_base']:.4f}, r {S['s_r']:.3f}) | 크기 비율 " +
              ' '.join(f"{ax} MAE {S[f'q{ax}_mae']:.4f} (기준 {S[f'q{ax}_mae_base']:.4f}, r {S[f'q{ax}_r']:.3f})" for ax in 'xyz'), flush=True)
        continue
    S = {'n': len(rows),
         'cx_mae': float(np.abs(A['cx_pred'] - A['cx_gt']).mean()), 'cx_mae_base': float(np.abs(A['cx_base'] - A['cx_gt']).mean()), 'cx_r': corr(A['cx_pred'], A['cx_gt']),
         'cy_mae': float(np.abs(A['cy_pred'] - A['cy_gt']).mean()), 'cy_mae_base': float(np.abs(A['cy_base'] - A['cy_gt']).mean()), 'cy_r': corr(A['cy_pred'], A['cy_gt']),
         's_mae': float(np.abs(A['s_pred'] - A['s_gt']).mean()), 's_mae_base': float(np.abs(A['s_base'] - A['s_gt']).mean()), 's_r': corr(A['s_pred'], A['s_gt']),
         'rot_err_deg': float(A['rot_err'].mean()), 'rot_err_deg_base': float(A['rot_err_base'].mean())}
    json.dump(S, open(f'{a.out}/pose_{sub}.json', 'w'), indent=1)
    print(f"POSE {sub} | cx MAE {S['cx_mae']:.4f} (기준 {S['cx_mae_base']:.4f}, r {S['cx_r']:.3f}) | cy MAE {S['cy_mae']:.4f} (기준 {S['cy_mae_base']:.4f}, r {S['cy_r']:.3f}) | "
          f"scale MAE {S['s_mae']:.4f} (기준 {S['s_mae_base']:.4f}, r {S['s_r']:.3f}) | 회전 {S['rot_err_deg']:.2f}° (기준 {S['rot_err_deg_base']:.2f}°)", flush=True)
print('POSE_DONE', flush=True)
