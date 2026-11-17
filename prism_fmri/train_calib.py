# -*- coding: utf-8 -*-
# 11-17 추가 논문 Perceptive 개선안 1단계: 피험자 · 영역별 입력 보정(prism.RegionCalib)만 학습해 피험자들의 뇌를 맞춤
#  고정 ViT(--enc_path, 피험자 공통) 앞에 보정을 두고, 같은 물체를 본 다른 피험자의 ViT 특징이 가까워지게 대조 손실(sep_losses.share_loss)
#  특징 = patch 토큰 프레임 평균 → 토큰마다 LayerNorm(학습 파라미터 없음) → 이웃 2×2 평균(64개) → 펴서 피험자별 평균 빼기(배치 · 평가 세트 안) → 정규화
#  배치 = 물체 m개 × 피험자 k명 (train에서 --holdout 뗀 물체 중 전 피험자에 있는 것)
#  검증 (보정 전 = 0스텝, 이후 --eval_every마다, 끝): [hold] · [test]에서 "다른 피험자 표본 중 같은 물체 고르기" top-1 (우연 = 1/물체 수)과
#    같은 물체 · 다른 피험자 cos 평균 − 다른 물체 cos 평균 (gap). 판정: [hold]가 보정 전보다 올라야 정렬이 된 것 (train만 오르면 외운 것)
#  --fit_subs: 이 피험자들의 보정만 학습 (나머지 행은 고정) — 나중에 새 피험자를 넣을 때 그 피험자 행만 맞추는 용도. 비우면 전부
#  저장: <out_dir>/calib.pt {g, b, lab, subs, args, hist, summary} → train_prism.py --calib 1 --calib_init <이 파일>
# 사용 (실행 위치 $W/run): python train_calib.py --repo $W/MinD-3D --data_root $DR --tgt $TGT --subs sub-0001,sub-0006 --enc_path $U --in_norm 1 --holdout 8 --out_dir $W/prism_calib/s16
import os, sys, json, time, argparse, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prism import RegionCalib, load_encoder
from sep_losses import share_loss
from polite import Throttle

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--subs', required=True); p.add_argument('--enc_path', required=True); p.add_argument('--in_norm', type=int, default=1, choices=[0, 1])
p.add_argument('--holdout', type=int, default=8); p.add_argument('--out_dir', required=True)
p.add_argument('--m', type=int, default=12); p.add_argument('--k', type=int, default=2)          # 배치 = 물체 m × 피험자 k
p.add_argument('--steps', type=int, default=3000); p.add_argument('--lr', type=float, default=1e-2); p.add_argument('--temp', type=float, default=0.05)
p.add_argument('--reg', type=float, default=1e-3)                                                  # 보정값 L2 (항등에서 너무 멀어지지 않게)
p.add_argument('--bound', type=float, default=0.0); p.add_argument('--renorm', type=int, default=0, choices=[0, 1])   # 11-18 크기 묶기 (prism.RegionCalib)
p.add_argument('--keep', type=float, default=0.0)   # 11-18 보존 항: λ · mean(1 − cos(보정 켠 특징, 보정 끈 특징)) — 피험자 간 정렬이 고정 ViT 특징을 바꿔 정보를 잃지 않게
p.add_argument('--eval_every', type=int, default=500); p.add_argument('--print_every', type=int, default=50)
p.add_argument('--polite', type=float, default=0)   # > 0: 같은 GPU 다른 사람 SM%를 이 값 이상으로 유지하게 스텝 사이 쉼 (polite.py)
p.add_argument('--fit_subs', default=''); p.add_argument('--limit', type=int, default=0); p.add_argument('--fake_enc', type=int, default=0)   # fake_enc: 로컬 동작 확인용
a = p.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; FRAMES = list(range(2, 8)); SUBS = a.subs.split(','); K = len(SUBS)
lst = lambda s: [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{s}_list.txt')) if l.strip()]
tr_all = lst('train'); by = {}
for k_ in tr_all: by.setdefault(k_.split('/')[0], []).append(k_)
H = {k_ for v in by.values() for k_ in v[-a.holdout:]} if a.holdout > 0 else set()
has = lambda k_: os.path.exists(f'{a.tgt}/dino/{k_.split("/")[1]}.npy') and all(os.path.exists(os.path.join(a.data_root, s, 'npy_data', k_ + '.npy')) for s in SUBS)
SPL = {'train': [k_ for k_ in tr_all if k_ not in H and has(k_)], 'hold': [k_ for k_ in tr_all if k_ in H and has(k_)], 'test': [k_ for k_ in lst('test') if has(k_)]}
if a.limit: SPL = {s: v[:a.limit] for s, v in SPL.items()}
print('물체 (전 피험자 공통) ' + ' · '.join(f'{s} {len(v)}' for s, v in SPL.items()) + f' | 피험자 {SUBS}', flush=True)
load = lambda si, k_: np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k_ + '.npy')).astype(np.float32)[FRAMES]

if a.fake_enc:                                                                     # 로컬 확인: 패치 임베딩 + 작은 변환 (입력까지 기울기가 감)
    class FE(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.pe = torch.nn.Conv2d(1, 1024, 16, 16); s.cls = torch.nn.Parameter(torch.zeros(1, 1, 1024))
        def forward_encoder_wo_pred(s, x):
            t = s.pe(x).flatten(2).transpose(1, 2); return torch.cat([s.cls.expand(len(t), -1, -1), torch.tanh(t)], 1)
    enc = FE()
else:
    from omegaconf import OmegaConf
    enc = load_encoder(a.enc_path, OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml')))
enc = enc.to(dev).eval().requires_grad_(False)
cal = RegionCalib(K, bound=a.bound, renorm=bool(a.renorm)).to(dev)
fit = [SUBS.index(s) for s in a.fit_subs.split(',')] if a.fit_subs else list(range(K))
mask = torch.zeros(K, 1, device=dev); mask[fit] = 1
for q in (cal.g, cal.b): q.register_hook(lambda g_: g_ * mask)                     # 고를 피험자 행만 학습
opt = torch.optim.Adam(cal.parameters(), lr=a.lr); th = Throttle(a.polite)


def feats(x, subj, on=True):
    """x [B, T, 256, 256] → [B, 64·1024] 정규화 특징 (on=False: 보정 끔)"""
    if a.in_norm:
        m = (x != 0).to(x.dtype); n = m.sum((2, 3), keepdim=True).clamp_min(1); mu = (x * m).sum((2, 3), keepdim=True) / n
        sd = (((x - mu) ** 2 * m).sum((2, 3), keepdim=True) / n).sqrt() + 1e-6; x = (x - mu) / sd * m
    if on: x = cal(x, subj)
    B, T = x.shape[:2]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
        t = enc.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)[:, :, 1:].float().mean(1)   # [B, 256, 1024]
    return F.layer_norm(t, t.shape[-1:]).view(B, 8, 2, 8, 2, -1).mean((2, 4)).flatten(1)   # [B, 64·1024] (정규화 전)


def center(z, subj):
    """피험자마다 자기 표본 평균을 빼고 정규화 — 피험자 고유 성분(물체와 무관)을 없애야 피험자 간 같은 물체 신호가 보임 (본 모델 feat_norm과 같은 이유)"""
    z = z.float().clone()
    for s_ in subj.unique(): mk = subj == s_; z[mk] = z[mk] - z[mk].mean(0, keepdim=True)
    return F.normalize(z, dim=-1)


@torch.no_grad()
def evaluate(split):
    ks = SPL[split]; Z, O, S = [], [], []
    for si in range(K):
        for i in range(0, len(ks), 16):
            x = torch.from_numpy(np.stack([load(si, k_) for k_ in ks[i:i + 16]])).to(dev)
            Z.append(feats(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).half()); O += list(range(i, i + len(x))); S += [si] * len(x)
    O = torch.tensor(O, device=dev); S = torch.tensor(S, device=dev); Z = center(torch.cat(Z), S)
    sim = Z @ Z.t(); other = S[:, None] != S[None]; same = (O[:, None] == O[None]) & other
    top1 = float((O[sim.masked_fill(~other, -2).argmax(-1)] == O).float().mean())
    gap = float(sim[same].mean() - sim[other & ~same].mean())
    return {'top1': round(top1, 4), 'chance': round(1 / len(ks), 4), 'gap': round(gap, 4)}


def eval_all(step):
    r = {s: evaluate(s) for s in ('hold', 'test')}; r['step'] = step; _g, _b = (a.bound * torch.tanh(cal.g / a.bound), a.bound * torch.tanh(cal.b / a.bound)) if a.bound > 0 else (cal.g, cal.b)
    r['|g|'] = round(float(_g.detach().abs().mean()), 4); r['|b|'] = round(float(_b.detach().abs().mean()), 4)
    print(f'[eval] step {step} | hold top1 {r["hold"]["top1"]:.3f} gap {r["hold"]["gap"]:.4f} | test top1 {r["test"]["top1"]:.3f} gap {r["test"]["gap"]:.4f} '
          f'(우연 {r["hold"]["chance"]:.3f}) | |g| {r["|g|"]:.3f} |b| {r["|b|"]:.3f}', flush=True)
    return r


hist = [eval_all(0)]                                                               # 보정 전 (항등)
keys = SPL['train']; t0 = time.time(); step = 0; rng = np.random.default_rng(0)
while step < a.steps:
    for oi in np.array_split(rng.permutation(len(keys)), len(keys) // a.m):
        if step >= a.steps: break
        xs, ss, os_ = [], [], []
        for j in oi[:a.m]:
            for si in rng.choice(K, a.k, replace=False): xs.append(load(si, keys[j])); ss.append(int(si)); os_.append(int(j))
        x = torch.from_numpy(np.stack(xs)).to(dev); subj = torch.tensor(ss, device=dev); obj = torch.tensor(os_, device=dev)
        zr = feats(x, subj); z = center(zr, subj); loss, acc = share_loss(z, obj, a.temp)
        tot = loss + a.reg * (cal.g.pow(2).mean() + cal.b.pow(2).mean())
        if a.keep > 0:
            with torch.no_grad(): z0 = feats(x, subj, on=False)
            kp = (1 - F.cosine_similarity(zr.float(), z0.float(), dim=-1)).mean(); tot = tot + a.keep * kp
        opt.zero_grad(set_to_none=True); tot.backward(); opt.step(); step += 1; th.step()
        if step % a.print_every == 0:
            el = time.time() - t0; print(f'step {step}/{a.steps} loss {float(loss):.4f} acc {float(acc):.3f} ' + (f'keep {float(kp):.4f} ' if a.keep > 0 else '') + f'|g| {float(cal.g.abs().mean()):.3f} |b| {float(cal.b.abs().mean()):.3f} '
                                         f'{step / el:.2f}스텝/초 남은 {(a.steps - step) / (step / el) / 60:.0f}분 종료 {time.strftime("%H:%M", time.gmtime(time.time() + 9 * 3600 + (a.steps - step) / (step / el)))} {th.tag()}', flush=True)
        if step % a.eval_every == 0 and step < a.steps: hist.append(eval_all(step))
hist.append(eval_all(step))
b0, b1 = hist[0]['hold'], hist[-1]['hold']
ok = b1['top1'] > b0['top1'] and b1['gap'] > b0['gap'] + max(0.002, 0.1 * abs(b0['gap'])) and b1['top1'] >= 2 * b1['chance']   # 11-17: 0 근처 값끼리 작은 변화는 정렬로 안 침
note = '' if b0['top1'] >= 2 * b0['chance'] else ' | 보정 전 피험자 간 같은 물체 신호가 우연 수준'
summary = f"hold top1 {b0['top1']:.3f}→{b1['top1']:.3f} (우연 {b1['chance']:.3f}) gap {b0['gap']:.4f}→{b1['gap']:.4f} ({'정렬됨' if ok else '개선 없음'}){note}"
torch.save({'g': cal.g.detach().cpu(), 'b': cal.b.detach().cpu(), 'bound': a.bound, 'renorm': a.renorm, 'lab': cal.lab.cpu(), 'subs': SUBS, 'args': vars(a), 'hist': hist, 'summary': summary}, f'{a.out_dir}/calib.pt')
json.dump(hist, open(f'{a.out_dir}/calib_hist.json', 'w'), ensure_ascii=False, indent=1)
print('CALIB_DONE', summary, flush=True)
