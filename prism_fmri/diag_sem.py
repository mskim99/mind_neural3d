# 성능 개선 1단계 진단 (11-12, 인계 261112_실험Pro6000→실험5090_성능개선1단계) — MinD-3D++ venv, 실행 위치 $W/run
#  A (H2) Semantic 경로 어디서 범주 정보를 잃나: 피험자마다 train 1300 → test 104, 지점마다 같은 선형 해독기
#     해독기 = roi_decode.py fit() 그대로 (선형 커널 ridge, 커널 train 중심화, alpha 열마다 LOO), 특징은 차원마다 train 평균 · 표준편차로 표준화, 커널 = Z Zᵀ / 차원 수
#     지점: vit_pos  고정 ViT 출력 패치 256 × 1024 (프레임 2~7 평균, 펼침, 줄이지 않음)   vit_mean  같은 것의 프레임 · 패치 평균 1024
#           sem_pos  Mixed Features의 Semantic 채널 패치 256 × 384 (프레임 평균)          sem_mean  패치 평균 384
#           clip_pos CLIP Projection 출력 256 × 1024                                     clip_mean 패치 평균 1024
#           sfeat_pos Semantic Features(디코더 입력) 256 × 768                            sfeat_mean 패치 평균 768
#           model    모델 자체 예측 (CLIP 벡터 · 텍스트 프로토타입 cos argmax, 해독기 아님)
#  B (H3) 출력이 범주 평균 수준인가: test 104, 정답 토큰과 1 − cos (토큰마다, 학습 tok_cos와 같은 식)
#     Ours 예측 / train 범주 평균 토큰(정답 범주) / train 범주 평균 토큰(모델 예측 범주) / train 전체 평균 토큰
#     나머지 상관: cos(예측 − 범주평균[정답 범주], 정답 − 범주평균[정답 범주]) 토큰마다 평균. 비교 = 같은 범주 다른 물체의 예측으로 바꾼 값
#  출력 <out>/diag.json, 표 줄 A| · B|
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt, _route

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default='')
p.add_argument('--bs', type=int, default=8); p.add_argument('--skip_a', type=int, default=0)
p.add_argument('--hold', type=int, default=0)   # > 0: train 범주마다 목록 끝 n개(검증 물체)를 빼고 맞춘 뒤 검증 물체 · test 정확도를 따로 (A만, 값은 'hold/test')
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval(); assert model.arch == 'fig' and model.sem_clip == 'front'
subs = a.subs.split(',') if a.subs else SUBS
rd = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
TR, TE = [k for k in rd('train') if k.split('/')[0] in SYN], rd('test'); HO = []
if a.hold:
    by = {}
    for k in TR: by.setdefault(k.split('/')[0], []).append(k)
    Hs = {k for v in by.values() for k in v[-a.hold:]}; HO = [k for k in TR if k in Hs]; TR = [k for k in TR if k not in Hs]
ALL = TR + HO + TE; ntr = len(TR); nho = len(HO)
cls = np.array([SYN.index(k.split('/')[0]) for k in ALL]); NC = len(SYN); Yc = np.eye(NC)[cls]
uid = lambda k: k.split('/')[1]
ALPHA_REL = [10.0 ** k for k in np.arange(-3, 3.01, 0.5)]


def fit(Kt, Y):   # roi_decode.py fit() 그대로
    rm = Kt.mean(0); am = Kt.mean(); s, U = np.linalg.eigh(Kt - rm[None] - rm[:, None] + am); s = np.clip(s, 0, None)
    ym = Y.mean(0); UtY = U.T @ (Y - ym); U2 = U ** 2; err = np.zeros((len(ALPHA_REL), Y.shape[1])); n = len(Kt)
    for ai, r in enumerate(ALPHA_REL):
        h = s / (s + r * s.mean()); d = np.clip(1 - U2 @ h - 1 / n, 1e-6, None)
        err[ai] = ((((Y - ym) - U @ (h[:, None] * UtY)) / d[:, None]) ** 2).sum(0)
    al = np.array(ALPHA_REL)[err.argmin(0)] * s.mean(); C = U @ (UtY / (s[:, None] + al[None]))
    return lambda Ke: (Ke - rm[None] - Ke.mean(1, keepdims=True) + am) @ C + ym


def gram(X):      # X [N, D] (GPU fp16) → K [N, ntr] float64, 차원마다 train 표준화
    N, D = X.shape; K = torch.zeros(N, ntr, device='cuda', dtype=torch.float64)
    for j in range(0, D, 8192):
        c = X[:, j:j + 8192].float(); mu = c[:ntr].mean(0); sd = c[:ntr].std(0) + 1e-6; c = (c - mu) / sd
        K += (c @ c[:ntr].T).double()
    return (K / D).cpu().numpy()


def decode(X):
    K = gram(X); f = fit(K[:ntr], Yc[:ntr]); P = f(K[ntr:]); ok = P.argmax(1) == cls[ntr:]
    return (float(ok[:nho].mean()), float(ok[nho:].mean())) if nho else float(ok.mean())


def run(sub, ks):
    x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
    fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda(); sj = torch.full((len(ks),), SUBS.index(sub), dtype=torch.long, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = model.encode_frames(x, sj).float()
        mixed = model.shared(model.temporal(tok, model.frame_emb(fi)))
        st = mixed.split(model.splits, dim=-1)[0][:, :, 1:].mean(1)
        clip = _route(model.sem_clip_proj, sj, st)
        out = model(x, fi, sj)
    pred = (out['sem_clip'].float() @ model.prototypes.t()).argmax(-1)
    return dict(vit=tok[:, :, 1:].mean(1).half(), sem=st.half(), clip=clip.half(), sfeat=out['sem_feat'].half(), pred=pred, tok=out['cond_tokens'].half())


R = {'A': {}, 'B': {}, 'subs': subs}
# ---- B 준비: 정답 토큰, train 범주 평균 · 전체 평균
gt = lambda k: torch.from_numpy(np.load(f'{a.tgt}/dino/{uid(k)}.npy')).float()
CM = torch.zeros(NC, 1374, 1024); cnt = np.zeros(NC)
for k, c in zip(TR, cls[:ntr]): CM[c] += gt(k); cnt[c] += 1
GM = (CM.sum(0) / cnt.sum()).cuda(); CM = (CM / torch.from_numpy(cnt)[:, None, None].float()).cuda()
GTE = torch.stack([gt(k) for k in TE]).half().cuda(); cte = torch.from_numpy(cls[ntr + nho:]).cuda()
loss = lambda X, Y: (1 - torch.nn.functional.cosine_similarity(X.float(), Y.float(), dim=-1)).mean(-1)   # [B]
rng = np.random.default_rng(0); cn = cls[ntr + nho:]
same = torch.from_numpy(np.array([rng.choice(np.where((cn == cn[i]) & (np.arange(len(TE)) != i))[0]) for i in range(len(TE))])).cuda()

for sub in subs:
    keys = TE if a.skip_a else ALL
    F = {k: [] for k in ('vit', 'sem', 'clip', 'sfeat', 'pred', 'tok')}
    for i in range(0, len(keys), a.bs):
        o = run(sub, keys[i:i + a.bs]); te_part = i >= (0 if a.skip_a else ntr)
        for k in F:
            if k == 'tok' and not (a.skip_a or i + a.bs > ntr): continue
            F[k].append(o[k] if k != 'tok' else o[k][max(0, ntr - i) if not a.skip_a else 0:])
    F = {k: torch.cat(v) for k, v in F.items()}
    pred_te = F['pred'][-len(TE):]
    # A
    if not a.skip_a:
        A = {}
        for nm in ('vit', 'sem', 'clip', 'sfeat'):
            A[nm + '_pos'] = decode(F[nm].flatten(1)); A[nm + '_mean'] = decode(F[nm].float().mean(1).half())
        A['model'] = float((pred_te == cte).float().mean())
        R['A'][sub] = A; print(f'A| {sub} ' + ' '.join(f'{k} {v[0]:.3f}/{v[1]:.3f}' if isinstance(v, tuple) else f'{k} {v:.3f}' for k, v in A.items()), flush=True)
    if a.hold:
        A['model_hold'] = float((F['pred'][ntr:ntr + nho] == torch.from_numpy(cls[ntr:ntr + nho]).cuda()).float().mean()); print(f'A| {sub} model_hold {A["model_hold"]:.3f}', flush=True)
        json.dump(R, open(f'{a.out}/diag.json', 'w'), indent=1); del F; torch.cuda.empty_cache(); continue
    # B
    T = F['tok']; assert len(T) == len(TE)
    B = {'ours': float(loss(T, GTE).mean()), 'catmean_gt': float(loss(CM[cte], GTE).mean()),
         'catmean_pred': float(loss(CM[pred_te], GTE).mean()), 'globalmean': float(loss(GM[None].expand(len(TE), -1, -1), GTE).mean())}
    rp = T.float() - CM[cte]; rg = GTE.float() - CM[cte]
    B['resid_cos'] = float(torch.nn.functional.cosine_similarity(rp, rg, dim=-1).mean())
    B['resid_cos_shuffle'] = float(torch.nn.functional.cosine_similarity(rp[same], rg, dim=-1).mean())
    B['cat_acc_model'] = float((pred_te == cte).float().mean())
    R['B'][sub] = B; print(f'B| {sub} ' + ' '.join(f'{k} {v:.4f}' for k, v in B.items()), flush=True)
    json.dump(R, open(f'{a.out}/diag.json', 'w'), indent=1)
    del F; torch.cuda.empty_cache()

for part in ('A', 'B'):
    if R[part] and not a.hold:
        ks = list(next(iter(R[part].values())).keys())
        print(f'{part}| 평균 ' + ' '.join(f'{k} {np.mean([R[part][s][k] for s in R[part]]):.4f}' for k in ks), flush=True)
print('DIAG_DONE', flush=True)
