# V-6 ① · ② (11-25, 5090 CPU, 학습 없음): 생성 결과의 범주 안 몫
#  ① 범주 안 고르기: 생성물마다 같은 범주 test 정답 8개와 비교해 가장 가까운 것이 자기 정답인 비율 (무작위 0.125)
#     - CD: 생성 GLB 표면 2,048점(seed 0) → pc_norm → GT 좌표축(형식 축 +x,+y,+z) vs gt_pc/P2[후보][0] (평가 번들 dist.py와 같은 식, 공식 chamfer)
#     - 실루엣: 평가 렌더 render_R2/<uid>/{0..5}.png 알파 vs gt_render/R2/<후보>/{0..5}.png 알파, 256²로 줄여 IoU 6뷰 평균
#     - 순열 검정: 범주마다 "어느 후보가 정답인가"를 섞어 10,000번 → p
#     - 예측 범주(외형 문장)가 맞음 / 틀림으로 나눔
#  ② 생성 렌더 특징: 평가 렌더 6뷰에서 V-5와 같은 특징(v6_common) → 뷰 평균. 정답 텍스처 렌더(gt_render/R2)도 같은 방식으로 잼 (참고)
# 출력: <out>/ident.json (세트별 요약), <out>/perobj_<세트>.csv, <out>/feat_<세트>.npz · feat_gt_tex.npz
# 사용: python -I v6_ident.py --fe <fair_eval 번들> --out <폴더> --workers 6
import os, sys, json, csv, argparse, numpy as np
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[_v] = '1'
from multiprocessing import Pool
ap = argparse.ArgumentParser()
ap.add_argument('--fe', default='/home/oem/jhj/fair_eval/fair_eval'); ap.add_argument('--m3', default='/home/oem/jhj/MinD-3D++')
ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=6); ap.add_argument('--subs', default='01,02,03,05,06,07,08')
ap.add_argument('--perm', type=int, default=10000)
a = ap.parse_args()
sys.path.insert(0, f'{a.fe}/scripts'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.makedirs(a.out, exist_ok=True)
KEYS = [l.strip() for l in open(f'{a.fe}/core_test_list.txt') if l.strip()]
UID = [k.split('/')[-1] for k in KEYS]; CAT = np.array([k.split('/')[0] for k in KEYS]); IDX = {u: i for i, u in enumerate(UID)}
CANDS = {c: np.where(CAT == c)[0] for c in sorted(set(CAT))}
print('범주별 test 물체 수', {c: len(v) for c, v in CANDS.items()}, flush=True)
SETS = []
for s in a.subs.split(','):
    SETS.append((f'Ours_s{s}', f'yaw_fullausvp_s{s}', f'{a.m3}/prism_full/eval_ausvp/text/s{s}/prompt.json'))
    SETS.append((f'noV_s{s}', f'yaw_abl_noV_s{s}', f'{a.m3}/prism_full_abl/eval_noV/text/s{s}/prompt.json'))


def sil(p, cv2):
    im = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_UNCHANGED) if os.path.exists(p) else None
    if im is None or im.ndim < 3 or im.shape[2] < 4: return None
    return cv2.resize(im[..., 3], (256, 256), interpolation=cv2.INTER_AREA) > 127


GT_SIL = {}
def gt_sil(u, cv2):
    if u not in GT_SIL: GT_SIL[u] = [sil(f'{a.fe}/gt_render/R2/{u}/{v}.png', cv2) for v in range(6)]
    return GT_SIL[u]


def one(job):
    name, d, uid, glb = job
    import cv2, trimesh
    from official_pc import compute_trimesh_chamfer, pc_norm
    from v6_common import render_feat
    i = IDX[uid]; cand = CANDS[CAT[i]]; row = {'uid': uid, 'cat': CAT[i]}
    try:
        gen = pc_norm(trimesh.sample.sample_surface(trimesh.load(glb, force='mesh'), 2048, seed=0)[0]).astype(np.float32)
        row['cd'] = [float(compute_trimesh_chamfer(np.load(f'{a.fe}/gt_pc/P2/{UID[j]}.npy')[0].astype(np.float32), gen) * 100) for j in cand]
    except Exception as e:
        row['cd'] = None; print('CD 실패', name, uid, repr(e)[:80], flush=True)
    ps = [sil(f'{a.fe}/out/{d}/render_R2/{uid}/{v}.png', cv2) for v in range(6)]
    ious = []
    for j in cand:
        g = gt_sil(UID[j], cv2); vals = [(p & q).sum() / max((p | q).sum(), 1) for p, q in zip(ps, g) if p is not None and q is not None]
        ious.append(float(np.mean(vals)) if len(vals) >= 4 else np.nan)
    row['iou'] = ious
    f = render_feat(f'{a.fe}/out/{d}/render_R2/{uid}', cv2); row['feat'] = None if f is None else [float(x) for x in f]
    return name, row


def ident(rows, key, better):
    """rows: 물체별 dict → (정확도, 순열 p, 물체별 맞음 배열, 고른 후보 위치)"""
    ok, choice, selfpos, cats, kept = [], [], [], [], []
    for r in rows:
        v = r.get(key)
        if v is None or np.all(np.isnan(v)): continue
        v = np.array(v, float); c = int(np.nanargmin(v) if better == 'min' else np.nanargmax(v))
        sp = int(np.where(CANDS[r['cat']] == IDX[r['uid']])[0][0]); choice.append(c); selfpos.append(sp); cats.append(r['cat']); ok.append(c == sp); kept.append(r['uid'])
    choice, selfpos, cats = np.array(choice), np.array(selfpos), np.array(cats); acc = float(np.mean(ok))
    rng = np.random.default_rng(0); null = np.zeros(a.perm)
    for c in set(cats):   # 범주 안에서 정답 자리를 섞음 (8개 순열)
        m = cats == c; n = len(CANDS[c]); sp = selfpos[m]; ch = choice[m]
        P = np.argsort(rng.random((a.perm, n)), 1)     # 순열 π: 원래 자리 k → π[k]
        null += (P[:, sp] == ch[None]).sum(1)
    null /= len(choice); p = float((1 + (null >= acc).sum()) / (1 + a.perm))
    return acc, p, np.array(ok), kept


if __name__ == '__main__':
    import cv2
    from v6_common import render_feat, NAMES
    # 정답 텍스처 렌더 특징 (참고, 한 번만)
    if not os.path.exists(f'{a.out}/feat_gt_tex.npz'):
        G = [render_feat(f'{a.fe}/gt_render/R2/{u}', cv2) for u in UID]
        ok = [k for k, g in enumerate(G) if g is not None]
        np.savez(f'{a.out}/feat_gt_tex.npz', keys=np.array([KEYS[k] for k in ok]), feat=np.stack([G[k] for k in ok]).astype(np.float32), names=np.array(NAMES))
        print('GT 텍스처 렌더 특징', len(ok), flush=True)
    jobs = []
    for name, d, _ in SETS:
        if os.path.exists(f'{a.out}/perobj_{name}.json'): continue
        pairs = [l.rstrip('\n').split('\t') for l in open(f'{a.fe}/out/{d}/pairs.tsv') if l.strip()]
        fr = open(f'{a.fe}/out/{d}/frame.txt').read().strip(); assert fr == '+x,+y,+z', f'{d} 좌표축 {fr}'
        jobs += [(name, d, u, g) for u, g in pairs]
    print('할 물체', len(jobs), '· 세트', len(SETS), flush=True)
    R = {}
    with Pool(a.workers) as pool:
        for n, (name, row) in enumerate(pool.imap_unordered(one, jobs, chunksize=4)):
            R.setdefault(name, []).append(row)
            if (n + 1) % 104 == 0: print(f'[{n + 1}/{len(jobs)}] {name}', flush=True)
    for name, rows in R.items(): json.dump(rows, open(f'{a.out}/perobj_{name}.json', 'w'))
    S = {}
    CATWORD = {}
    allp = {}
    for name, d, pj in SETS:
        P = json.load(open(pj)); allp[name] = P
        for k in KEYS:
            w = P.get(k.split('/')[-1]); w and CATWORD.setdefault(k.split('/')[0], []).append(w)
    CATWORD = {c: max(set(v), key=v.count) for c, v in CATWORD.items()}   # 범주마다 가장 많이 나온 문장 = 그 범주의 문장
    print('범주 문장', CATWORD, flush=True)
    for name, d, pj in SETS:
        rows = json.load(open(f'{a.out}/perobj_{name}.json')); P = allp[name]
        CO = {r['uid']: P.get(r['uid']) == CATWORD[r['cat']] for r in rows}
        out = {'n': len(rows), 'cat_acc': float(np.mean(list(CO.values())))}
        for key, better in (('cd', 'min'), ('iou', 'max')):
            acc, p, ok, kept = ident(rows, key, better); co = np.array([CO[u] for u in kept])
            out[key] = {'acc': acc, 'p': p, 'n': len(kept),
                        'acc_catok': float(ok[co].mean()) if co.any() else None, 'n_catok': int(co.sum()),
                        'acc_catwrong': float(ok[~co].mean()) if (~co).any() else None, 'n_catwrong': int((~co).sum())}
        F = [(r['uid'], r['feat']) for r in rows if r['feat'] is not None]
        np.savez(f'{a.out}/feat_{name}.npz', keys=np.array([KEYS[IDX[u]] for u, _ in F]), feat=np.array([f for _, f in F], np.float32), names=np.array(NAMES))
        S[name] = out; print('IDENT|' + name + '|' + json.dumps(out, ensure_ascii=False), flush=True)
    json.dump({'sets': S, 'catword': CATWORD}, open(f'{a.out}/ident.json', 'w'), ensure_ascii=False, indent=1)
    print('V6_IDENT_DONE', flush=True)
