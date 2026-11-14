# 새 구조(mem branch) attention 진단 (11-15 사용자: UKB 2만 정성이 안 나오는 이유를 구조에서 찾기 — attention부터, 구조 바꾼 게 괜찮았나)
#  MinD-3D++ venv, 실행 위치 $W/run. 피험자마다 test 104, 토큰만 (렌더 없음)
#  [SEM] Semantic attention pooling (out['sem_attn'] [B, 256])
#    ent      엔트로피 ÷ log 256 (1 = 균등, 0 = 조각 하나)       eff  유효 조각 수 1/Σw²       top10  상위 10개 조각 몫
#    fixed    물체마다 지도와 104개 평균 지도의 cos (1이면 입력과 무관하게 같은 곳)    cat_in · cat_out  같은 범주 · 다른 범주 물체끼리 지도 cos
#    벡터 구성: sv = CLIP Projection(pooled) [+ vit_lin 선형 읽기]. ‖pool‖ · ‖lin‖ · cos(pool, lin)
#    acc_*    프로토타입 범주 정확도: 합 / pool만 / lin만 / attention 대신 균등 평균으로 pool
#  [DEC] 디코더 층마다 (BranchDecoderLayer: self → [Semantic cross + Variational cross] → FFN)
#    블록 출력 크기 self · cross_s · cross_v · ff, 잔차 크기 x
#    dep_*    그 출력이 물체에 따라 달라지는 비율 = 1 − ‖104개 평균‖² ÷ 평균 ‖출력‖² (query마다, 평균). Semantic cross는 key가 1개라 query마다 같은 값(사실상 bias)
#    Variational cross-attention 가중치: head 토큰(T개) 몫 / patch 토큰 몫 (균등 대비 배수), 엔트로피 ÷ log M, 평균 지도와 cos, 물체끼리 cos,
#             값 크기까지 본 head 토큰 몫, patch 위치 상위 5개 몫(프레임 합), 프레임별 patch 몫, query 위치(37×37)와 보는 patch 위치(G×G)의 상관
#  [ABL] 갈래 바꿔 끼우기 (test 104 평균으로): S · V 전체 · V head 토큰 · V patch 토큰 → d_out(1 − cos 원본) · d_gt(정답 cos 변화)
#  출력 <out>/<sub>.json, <out>/<sub>_maps.npz (Semantic pooling 평균 지도 [256] · 범주별 [C, 256], Variational patch 몫 지도 [L, G²])
import os, sys, json, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--ckpt', required=True); p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--bs', type=int, default=4)
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
ck = torch.load(a.ckpt, map_location='cpu'); SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8)); T = len(FRAMES)
model = build_from_ckpt(ck, a.repo).cuda().eval(); assert model.mem == 'branch', 'mem branch 체크포인트만 (cat은 dec_attn.py)'
TE = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_test_list.txt')) if l.strip()]
uid = lambda k: k.split('/')[1]; CAT = np.array([SYN.index(k.split('/')[0]) for k in TE])
G = 16 // model.v_pool; P2 = G * G; GQ, NPRE = 37, 5
NC = P2 if model.v_center else 0                                   # v_center: patch별 프레임 평균 토큰 G²개가 앞에
PROTO = F.normalize(model.prototypes.float(), dim=-1).cpu(); assert PROTO.abs().sum() > 0, '프로토타입이 비어 있음'
REC = {}


def layer_fwd(self, x, ms, mv, _i=0):                              # BranchDecoderLayer.forward와 같은 계산 + 기록 (eval이라 dropout 없음)
    h = self.n1(x); sa = self.sa(h, h, h, need_weights=False)[0]; x = x + sa
    h = self.n2(x); cs = self.ca_s(h, ms, ms, need_weights=False)[0]
    cv, w = self.ca_v(h, mv, mv, need_weights=True, average_attn_weights=False); x = x + cs + cv
    ff = self.f2(F.gelu(self.f1(self.n3(x))))
    m = self.ca_v; E = m.embed_dim; H = m.num_heads; val = mv.float()
    v = (val @ m.in_proj_weight[2 * E:].float().t() + m.in_proj_bias[2 * E:].float()).view(val.shape[0], val.shape[1], H, E // H)
    vn = torch.einsum('bmhd,ehd->bhme', v, m.out_proj.weight.float().view(E, H, E // H)).norm(dim=-1)   # [B, H, M] 값 벡터 크기 (out_proj까지)
    REC[_i] = dict(sa=sa.float(), cs=cs.float(), cv=cv.float(), ff=ff.float(), x=x.float(), w=w.float(), vn=vn)
    return x + ff


for i, l in enumerate(model.dec): l.forward = (lambda *ar, _l=l, _i=i: layer_fwd(_l, *ar, _i=_i))


def run(x, fi, sj, **kw):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): return model(x, fi, sj, **kw)


def load(sub, ks):
    x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
    return x, torch.tensor(FRAMES).expand(len(ks), -1).cuda(), torch.full((len(ks),), SUBS.index(sub), dtype=torch.long, device='cuda')


def sem_parts(x, fi, sj):                                          # Semantic 벡터를 pool · lin으로 나눠 다시 계산 (forward와 같은 식)
    from prism import _route
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = model.encode_frames(x, sj).float()
        if model.feat_norm: tok = (tok - model.feat_mu[sj][:, None]) / model.feat_sd[sj][:, None]
        xx = model.temporal(tok, model.frame_emb(fi)) if model.arch == 'fig' else model.temporal(tok + model.frame_emb(fi)[:, :, None, :])
        s = model.shared(xx).split(model.splits, dim=-1)[0]; st = s[:, :, 1:].mean(1)
        aw = torch.softmax(_route(model.sem_pool, sj, st).squeeze(-1).float(), -1)
        pool = _route(model.sem_clip_proj, sj, (aw[..., None].to(st.dtype) * st).sum(1)).float()
        unif = _route(model.sem_clip_proj, sj, st.mean(1)).float()
        lin = _route(model.sem_vlin, sj, _route(model.sem_vred, sj, tok[:, :, 1:].mean(1)).flatten(1)).float() if hasattr(model, 'sem_vlin') else torch.zeros_like(pool)
    return pool, lin, unif


acc_of = lambda v, y: float(((F.normalize(v, dim=-1) @ PROTO.t()).argmax(-1).cpu().numpy() == y).mean())
dep = lambda S, Q2, n: float((1 - ((S / n) ** 2).sum(-1) / (Q2 / n).clamp_min(1e-12)).mean())
OUT = {'ckpt': a.ckpt, 'args': {k: ck['args'].get(k) for k in ('mem', 'v_pool', 'sv_enc', 'v_center', 'vit_lin', 'feat_norm', 'in_norm', 'max_step')}}
for sub in (a.subs.split(',') if a.subs else SUBS):
    L = len(model.dec); aw_all, pools, lins, unifs, semF, varF, outs = [], [], [], [], [], [], []
    st = [dict(n=0, sa=0., cs=0., cv=0., ff=0., x=0., Ss=None, Qs=None, Sv=None, Qv=None, head=0., ent=0., cvh=0., hd=None, wsum=None, maps=[], frame=None) for _ in range(L)]
    for i in range(0, len(TE), a.bs):
        ks = TE[i:i + a.bs]; x, fi, sj = load(sub, ks); B = len(ks)
        o = run(x, fi, sj); aw_all.append(o['sem_attn'].float().cpu()); semF.append(o['sem_feat'].float().cpu()); varF.append(o['var_feat'].float().cpu()); outs.append(o['cond_tokens'].float().cpu())
        pl, ln, un = sem_parts(x, fi, sj); pools.append(pl.cpu()); lins.append(ln.cpu()); unifs.append(un.cpu())
        for li in range(L):
            R = REC[li]; Z = st[li]; w = R['w']; M = w.shape[-1]; Z['n'] += B
            for k in ('sa', 'cs', 'cv', 'ff', 'x'): Z[k] += float(R[k].norm(dim=-1).mean()) * B
            for k, kk in (('cs', 's'), ('cv', 'v')):
                Z['S' + kk] = R[k].sum(0).double() if Z['S' + kk] is None else Z['S' + kk] + R[k].sum(0).double()
                Z['Q' + kk] = (R[k] ** 2).sum(-1).sum(0).double() if Z['Q' + kk] is None else Z['Q' + kk] + (R[k] ** 2).sum(-1).sum(0).double()
            hm = w[..., :T].sum(-1)                                                   # [B, H, Q] head 토큰 몫
            Z['head'] += float(hm.mean((1, 2)).sum()); Z['hd'] = hm.mean((0, 2)).double().cpu() * B if Z['hd'] is None else Z['hd'] + hm.mean((0, 2)).double().cpu() * B
            Z['ent'] += float((-(w.clamp_min(1e-12).log() * w).sum(-1) / np.log(M)).mean((1, 2)).sum())
            c = w * R['vn'][:, :, None]; Z['cvh'] += float((c[..., :T].sum(-1) / c.sum(-1).clamp_min(1e-12)).mean((1, 2)).sum())
            wm = w.mean(1)                                                            # head 평균 [B, Q, M]
            Z['wsum'] = wm.sum(0).double() if Z['wsum'] is None else Z['wsum'] + wm.sum(0).double(); Z['maps'].append(wm.half().cpu())
            pf = wm[..., T + NC:].reshape(B, -1, T, P2).sum(1)                         # [B, T, G²] (query 합)
            Z['frame'] = pf.sum(0).double().cpu() if Z['frame'] is None else Z['frame'] + pf.sum(0).double().cpu()
    AW = torch.cat(aw_all); y = CAT; n = len(TE); R = {}
    # ---- Semantic pooling
    ent = float((-(AW.clamp_min(1e-12).log() * AW).sum(-1) / np.log(AW.shape[1])).mean()); eff = float((1 / (AW ** 2).sum(-1)).mean())
    top10 = float(AW.sort(-1, descending=True)[0][:, :10].sum(-1).mean()); mu = AW.mean(0)
    fixed = float(F.cosine_similarity(AW, mu[None], dim=-1).mean()); An = F.normalize(AW - mu, dim=-1); Cm = (An @ An.t()).numpy()
    same = (y[:, None] == y[None]) & ~np.eye(n, dtype=bool); diff = y[:, None] != y[None]
    PL, LN, UN = torch.cat(pools), torch.cat(lins), torch.cat(unifs)
    R['sem'] = dict(ent=ent, eff=eff, top10=top10, fixed=fixed, cat_in=float(Cm[same].mean()), cat_out=float(Cm[diff].mean()),
                    n_pool=float(PL.norm(dim=-1).mean()), n_lin=float(LN.norm(dim=-1).mean()), cos_pl=float(F.cosine_similarity(PL, LN, dim=-1).mean()),
                    acc_sum=acc_of(PL + LN, y), acc_pool=acc_of(PL, y), acc_lin=acc_of(LN, y), acc_unif=acc_of(UN + LN, y), acc_unif_pool=acc_of(UN, y))
    # ---- 디코더
    qy, qx = np.divmod(np.arange(GQ * GQ), GQ); py, px = np.divmod(np.arange(P2), G)
    for li in range(L):
        Z = st[li]; N = Z['n']; M = Z['wsum'].shape[-1]; mean = (Z['wsum'] / N).float(); A = torch.cat(Z['maps']).cuda().float()
        fx = float(F.cosine_similarity(A, mean[None], dim=-1).mean()); An = F.normalize(A, dim=-1); S = An.sum(0); pair = float((((S * S).sum(-1) - N) / (N * (N - 1))).mean())
        del A, An
        pm = mean[NPRE:NPRE + GQ * GQ, T + NC:].reshape(GQ * GQ, T, P2).sum(1).cpu().numpy(); pm = pm / pm.sum(-1, keepdims=True).clip(1e-12)
        cx, cy = pm @ px, pm @ py; rx = float(np.corrcoef(qx, cx)[0, 1]); ry = float(np.corrcoef(qy, cy)[0, 1])
        fr = (Z['frame'] / N).numpy(); pos = fr.sum(0) / fr.sum(); top5 = float(np.sort(pos)[::-1][:5].sum())
        R[f'layer{li}'] = dict(x=Z['x'] / N, sa=Z['sa'] / N, cs=Z['cs'] / N, cv=Z['cv'] / N, ff=Z['ff'] / N,
                               dep_s=dep(Z['Ss'], Z['Qs'], N), dep_v=dep(Z['Sv'], Z['Qv'], N),
                               head=Z['head'] / N, head_uni=T / M, head_x=Z['head'] / N / (T / M), head_h=[float(v) for v in (Z['hd'] / N).numpy()], cv_head=Z['cvh'] / N,
                               ent=Z['ent'] / N, fixed=fx, pair=pair, pos_rx=rx, pos_ry=ry, top5_pos=top5, uni5=5 / P2,
                               frame_share=[float(v) for v in (fr.sum(1) / fr.sum())], M=M)
        R[f'layer{li}']['_pos'] = pos
    # ---- 바꿔 끼우기 (test 104 평균으로)
    SF, VF, OO = torch.cat(semF), torch.cat(varF), torch.cat(outs)
    GT = torch.stack([torch.from_numpy(np.load(f'{a.tgt}/dino/{uid(k)}.npy')).float() for k in TE])
    cg = lambda O: F.cosine_similarity(O, GT, dim=-1).mean(-1)
    base = cg(OO); smu, vmu = SF.mean(0, keepdim=True), VF.mean(0, keepdim=True); abl = {}
    for name in ('S', 'V', 'Vhead', 'Vpatch'):
        res = []
        for i in range(0, n, a.bs):
            ks = TE[i:i + a.bs]; x, fi, sj = load(sub, ks); B = len(ks); kw = {}
            if name == 'S': kw['sem_override'] = smu.expand(B, -1, -1).cuda()
            else:
                v = VF[i:i + B].clone()
                if name == 'V': v[:] = vmu
                elif name == 'Vhead': v[:, :T] = vmu[:, :T]
                else: v[:, T:] = vmu[:, T:]
                kw['var_override'] = v.cuda()
            res.append(run(x, fi, sj, **kw)['cond_tokens'].float().cpu())
        O2 = torch.cat(res)
        abl[name] = dict(d_out=float((1 - F.cosine_similarity(O2, OO, dim=-1).mean(-1)).mean()), d_gt=float((cg(O2) - base).mean()))
    O_mean = OO.mean(0, keepdim=True)
    R['abl'] = abl; R['base_gt'] = float(base.mean()); R['dep_out'] = float((1 - (O_mean ** 2).sum(-1) / (OO ** 2).sum(-1).mean(0)).mean())
    R['mean_gt'] = float(cg(O_mean.expand_as(OO)).mean())                   # 모든 물체에 평균 출력을 줬을 때 정답 cos (출력이 물체를 얼마나 구분하나의 바닥)
    np.savez_compressed(f'{a.out}/{sub}_maps.npz', sem=mu.numpy(), sem_cat=np.stack([AW[torch.from_numpy(y == c)].mean(0).numpy() for c in range(len(SYN))]),
                        vpos=np.stack([R[f'layer{li}'].pop('_pos') for li in range(L)]), G=G, classes=np.array(SYN))
    OUT[sub] = R; s = R['sem']
    print(f"SEM| {sub} | ent {s['ent']:.3f} eff {s['eff']:.1f}/256 top10 {s['top10']:.3f} | 평균 지도 cos {s['fixed']:.3f} | 범주 안 {s['cat_in']:.3f} 밖 {s['cat_out']:.3f} | "
          f"‖pool‖ {s['n_pool']:.2f} ‖lin‖ {s['n_lin']:.2f} cos {s['cos_pl']:.2f} | acc 합 {s['acc_sum']:.3f} pool {s['acc_pool']:.3f} lin {s['acc_lin']:.3f} 균등pool+lin {s['acc_unif']:.3f} 균등pool {s['acc_unif_pool']:.3f}", flush=True)
    for li in range(L):
        d = R[f'layer{li}']
        print(f"DEC| {sub} L{li} | 크기 x {d['x']:.2f} self {d['sa']:.2f} cross_s {d['cs']:.2f} cross_v {d['cv']:.2f} ff {d['ff']:.2f} | dep s {d['dep_s']:.3f} v {d['dep_v']:.3f} | "
              f"head 몫 {d['head']:.3f} (균등 {d['head_uni']:.4f}, ×{d['head_x']:.1f}, 값크기 {d['cv_head']:.3f}) | ent {d['ent']:.3f} | 평균지도 cos {d['fixed']:.3f} 물체끼리 {d['pair']:.3f} | "
              f"위치상관 x {d['pos_rx']:.2f} y {d['pos_ry']:.2f} | 위치 상위5 {d['top5_pos']:.3f} (균등 {d['uni5']:.3f}) | 프레임 {' '.join(f'{v:.2f}' for v in d['frame_share'])}", flush=True)
    print(f"ABL| {sub} | 정답 cos {R['base_gt']:.4f} (평균 출력 {R['mean_gt']:.4f}, 출력이 물체에 따라 달라지는 비율 {R['dep_out']:.3f}) | " +
          ' | '.join(f"{k}: d_out {v['d_out']:.4f} d_gt {v['d_gt']:+.4f}" for k, v in abl.items()), flush=True)
json.dump(OUT, open(f'{a.out}/branch_attn.json', 'w'), indent=1, ensure_ascii=False)
print('BRANCH_ATTN_DONE', a.out, flush=True)
