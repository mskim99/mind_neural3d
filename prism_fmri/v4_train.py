# V-4 1단계 (11-24, Pro6000 GPU 1): TRELLIS 형상 단계(sparse structure flow, 이미지판 L)를 우리 데이터로 미세조정 — Variational을 prior의 조건으로
#  meshgen_tools venv, 실행 위치 /home/oem/jhj/meshgen_tools. 설계 근거 `.보고 자료/261124_결과모음/V-4_V를_형상prior조건으로.md` 0단계 절
#  encz : 정답 64³ 점유(<tgt>/voxels/<uid>.npy) → 좌표 (x, y, z) → (x, −z, y) (0단계: 위아래 = ShapeNet y, 거울상 아닌 회전)
#         → 형상 VAE 인코더 평균 → <out>/<uid>.npy [8, 16, 16, 16] fp32
#  train: 학습 범위 = 각 블록 cross-attention + 그 앞 LayerNorm(norm2)만, 나머지 고정. fp32 가중치 + bf16 계산, AdamW
#         --cond tok  : (A) 조건 = 예측 토큰 [1374, 1024] (원래와 같음)
#         --cond tokv : (B) + 블록마다 V 전용 cross-attention(LayerNorm → attention(key · value = V 특징 [390, 768] 투영)) × tanh(g), g = 0에서 시작
#                       → 학습 시작점이 (A)와 정확히 같음. 새 모듈은 --lr_new (문 g가 1e-5로는 거의 안 움직여서)
#         학습 물체 토큰 과신 대책: 토큰마다 가우스 잡음 (크기 = --tok_noise × 토큰 노름 / √1024, 매 step 새로).
#           0.36 = 피험자 1 실측 cos (train 0.828 → test 0.779)에서 √((0.828 / 0.779)² − 1)
#         손실 · t · 조건 버림 = TRELLIS ImageConditionedFlowMatchingCFGTrainer 그대로 (logitNormal(1, 1), 10 %는 토큰 · V 모두 0)
#         → <out>/ft_<step>.pt (학습한 파라미터만), <out>/train_log.csv
#  genz : 미세조정한 형상 단계로 test 물체 형상 잠재 → <out>/<uid>.npy. sampler = TRELLIS 기본(25 step · cfg 5.0 · 구간 [0.5, 1] · rescale 3), neg = 토큰 · V 모두 0, seed 42
#         이후 외형은 trellis_tools.py gen --ss_z <out> --slat_text (지금 Ours와 같은 텍스트판)
# 사용: python v4_train.py encz --tgt $TGT --out $V/zgt
#       python v4_train.py train --cond tok|tokv --data <infer_train/sub-0001_keep,infer_train/sub-0006_keep> --z $V/zgt --out $V/ft/A
#       (V-4b) --ckpt를 빼면 미세조정 없는 원래 가중치로 같은 경로 · 같은 sampler
#       python v4_train.py genz --cond tok|tokv --ckpt $V/ft/A/ft_4000.pt --data <infer_test/sub-0001_keep> --out $V/gen/A_s01/z
import os, sys, glob, json, time, argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['encz', 'train', 'genz'])
p.add_argument('--tgt', default='/home/oem/jhj/MinD-3D++/data/shape_targets'); p.add_argument('--out', required=True)
p.add_argument('--cond', default='tok', choices=['tok', 'tokv']); p.add_argument('--data', default=''); p.add_argument('--z', default='')
p.add_argument('--steps', type=int, default=4000); p.add_argument('--bs', type=int, default=8); p.add_argument('--lr', type=float, default=1e-5); p.add_argument('--lr_new', type=float, default=1e-4)
p.add_argument('--save', default='2000,4000'); p.add_argument('--tok_noise', type=float, default=0.36); p.add_argument('--p_uncond', type=float, default=0.1)
p.add_argument('--seed', type=int, default=1); p.add_argument('--ckpt', default=''); p.add_argument('--limit', type=int, default=0)   # genz: 앞 N개만 (동작 확인용)
p.add_argument('--trellis_repo', default='/home/oem/jhj/meshgen_tools/repo/third_party/TRELLIS')
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
os.environ.setdefault('SPCONV_ALGO', 'native'); os.environ.setdefault('ATTN_BACKEND', 'xformers'); os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
import torch, torch.nn as nn, torch.nn.functional as F
import xformers.ops as xops
from xformers.ops import fmha
_mea = xops.memory_efficient_attention
def _mea_cutlass(q, k, v, attn_bias=None, p=0.0, scale=None, *, op=None, **kw):   # trellis_tools.py와 같은 처리 (sm_120 FA3 오류 회피)
    return _mea(q, k, v, attn_bias, p, scale, op=op if op is not None else (fmha.cutlass.FwOp, fmha.cutlass.BwOp), **kw)
xops.memory_efficient_attention = _mea_cutlass; fmha.memory_efficient_attention = _mea_cutlass
sys.path.insert(0, a.trellis_repo)
from trellis import models as TM
snaps = sorted(glob.glob(os.path.join(os.environ.get('HF_HOME', ''), 'hub', 'models--microsoft--TRELLIS-image-large', 'snapshots', '*', 'pipeline.json')))
assert snaps, 'HF 캐시에 TRELLIS-image-large 없음'
CK = os.path.join(os.path.dirname(snaps[-1]), 'ckpts'); dev = 'cuda'; NT, DV = 1374, 768
kst = lambda s: time.strftime('%H:%M', time.gmtime(s + 9 * 3600))

if a.mode == 'encz':
    enc = TM.from_pretrained(f'{CK}/ss_enc_conv3d_16l8_fp16').to(dev).eval()
    U = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{a.tgt}/voxels/*.npy')); n = 0
    for i in range(0, len(U), 32):
        ub = [u for u in U[i:i + 32] if not os.path.exists(f'{a.out}/{u}.npy')]
        if not ub: continue
        g = torch.from_numpy(np.stack([np.unpackbits(np.load(f'{a.tgt}/voxels/{u}.npy'))[: 64 ** 3].reshape(64, 64, 64) for u in ub]).astype(np.float32)).to(dev)
        g = g.permute(0, 1, 3, 2).flip(2)                     # (x, y, z) → (x, −z, y): 새 축 1 = 옛 z 뒤집음, 새 축 2 = 옛 y (v4_probe.py tf((0,2,1), (F,T,F))와 같음)
        with torch.no_grad(): z = enc(g[:, None], sample_posterior=False)
        for u, zz in zip(ub, z): np.save(f'{a.out}/{u}.npy', zz.float().cpu().numpy()); n += 1
    print(f'ENCZ_DONE {n}개 새로 · 전체 {len(glob.glob(a.out + "/*.npy"))}', flush=True); sys.exit()


class VAdapter(nn.Module):   # 블록 출력 h에 tanh(g) · Attn(LN(h), V) 더함, g = 0이면 원래 블록과 같음
    def __init__(s, d=1024, dv=DV, heads=16):
        super().__init__(); s.norm = nn.LayerNorm(d); s.kv = nn.Linear(dv, d); s.attn = nn.MultiheadAttention(d, heads, batch_first=True); s.g = nn.Parameter(torch.zeros(1))
    def forward(s, h, v):
        kv = s.kv(v); return h + torch.tanh(s.g).to(h.dtype) * s.attn(s.norm(h), kv, kv, need_weights=False)[0].to(h.dtype)

class Wrap(nn.Module):
    def __init__(s, blk):
        super().__init__(); s.blk = blk; s.ad = VAdapter(); s.v = None
    def forward(s, h, t_emb, cond):
        h = s.blk(h, t_emb, cond); return s.ad(h, s.v) if s.v is not None else h

class Flow(nn.Module):   # cond = [B, 1374 (+ 390), 1024] — tokv면 뒤 390개의 앞 768차원이 V (sampler의 neg_cond = 0이면 V도 0)
    def __init__(s, fm, tokv):
        super().__init__(); s.fm = fm; s.tokv = tokv
        if tokv: fm.blocks = nn.ModuleList([Wrap(b) for b in fm.blocks])
        for k in ('resolution', 'in_channels', 'out_channels'): setattr(s, k, getattr(fm, k))
    def forward(s, x, t, cond):
        if s.tokv:
            v = cond[:, NT:, :DV]
            for b in s.fm.blocks: b.v = v
        return s.fm(x, t, cond[:, :NT])

def build(tokv):
    fm = TM.from_pretrained(f'{CK}/ss_flow_img_dit_L_16l8_fp16'); fm.convert_to_fp32(); fm.use_fp16 = False; fm.dtype = torch.float32
    m = Flow(fm, tokv).to(dev)
    for n_, q in m.named_parameters(): q.requires_grad = ('cross_attn' in n_ or 'norm2' in n_ or '.ad.' in n_)
    return m

def load_cond(d, u, tokv):
    c = torch.from_numpy(np.load(f'{d}/tokens/{u}.npy')).float()
    if tokv:
        v = torch.from_numpy(np.load(f'{d}/var/{u}.npy')).float(); c = torch.cat([c, F.pad(v, (0, 1024 - v.shape[1]))], 0)
    return c

if a.mode == 'train':
    torch.manual_seed(a.seed); np.random.seed(a.seed); tokv = a.cond == 'tokv'
    D = [d for d in a.data.split(',') if d]
    items = [(d, os.path.basename(f)[:-4]) for d in D for f in sorted(glob.glob(f'{d}/tokens/*.npy')) if os.path.exists(f'{a.z}/{os.path.basename(f)}')]
    if tokv: items = [(d, u) for d, u in items if os.path.exists(f'{d}/var/{u}.npy')]
    m = build(tokv); m.train()
    new = [q for n_, q in m.named_parameters() if q.requires_grad and '.ad.' in n_]; old = [q for n_, q in m.named_parameters() if q.requires_grad and '.ad.' not in n_]
    groups = [{'params': old, 'lr': a.lr}] + ([{'params': new, 'lr': a.lr_new}] if new else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.0); params = old + new
    print(f'[train] cond {a.cond} | 학습 쌍 {len(items)} (폴더 {len(D)}) | 학습 파라미터 기존 {sum(q.numel() for q in old) / 1e6:.1f}M · 새 {sum(q.numel() for q in new) / 1e6:.1f}M | bs {a.bs} · step {a.steps} · lr {a.lr} / {a.lr_new} · 토큰 잡음 {a.tok_noise} · 조건 버림 {a.p_uncond}', flush=True)
    SAVE = {int(x) for x in a.save.split(',') if x}; log = open(f'{a.out}/train_log.csv', 'a'); log.write('step,loss,gate_mean_abs,sec_per_step\n')
    perm, pi, t0, run = np.random.permutation(len(items)), 0, time.time(), []
    for step in range(1, a.steps + 1):
        if pi + a.bs > len(perm): perm, pi = np.random.permutation(len(items)), 0
        bt = [items[k] for k in perm[pi:pi + a.bs]]; pi += a.bs
        c = torch.stack([load_cond(d, u, tokv) for d, u in bt]).to(dev); x0 = torch.stack([torch.from_numpy(np.load(f'{a.z}/{u}.npy')) for d, u in bt]).float().to(dev)
        tk = c[:, :NT]; c[:, :NT] = tk + torch.randn_like(tk) * (a.tok_noise * tk.norm(dim=-1, keepdim=True) / 32.0)
        drop = torch.rand(len(bt), device=dev) < a.p_uncond; c[drop] = 0
        t = torch.sigmoid(torch.randn(len(bt), device=dev) + 1.0); e = torch.randn_like(x0); tt = t.view(-1, 1, 1, 1, 1)
        xt = (1 - tt) * x0 + (1e-5 + (1 - 1e-5) * tt) * e
        with torch.autocast('cuda', dtype=torch.bfloat16): pr = m(xt, t * 1000, c)
        loss = F.mse_loss(pr.float(), (1 - 1e-5) * e - x0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); run.append(loss.item())
        if step % 50 == 0 or step == a.steps:
            sp = (time.time() - t0) / step; gm = float(np.mean([abs(float(b.ad.g)) for b in m.fm.blocks])) if tokv else 0.0
            log.write(f'{step},{np.mean(run):.5f},{gm:.5f},{sp:.3f}\n'); log.flush()
            print(f'[train {a.cond}] step {step}/{a.steps} loss {np.mean(run):.4f} |g| {gm:.4f} {sp:.2f}s/step 끝 {kst(t0 + sp * a.steps)} KST', flush=True); run = []
        if step in SAVE:
            torch.save({n_: q.detach().cpu() for n_, q in m.named_parameters() if q.requires_grad}, f'{a.out}/ft_{step}.pt'); print('저장', f'{a.out}/ft_{step}.pt', flush=True)
    print('TRAIN_DONE', flush=True); sys.exit()

if a.mode == 'genz':
    tokv = a.cond == 'tokv'; m = build(tokv).eval()
    if a.ckpt:   # --ckpt 없으면 원래 TRELLIS 형상 단계 그대로 (V-4b 0 step 대조행, --cond tok만)
        sd = torch.load(a.ckpt, map_location='cpu'); miss = [n_ for n_, q in m.named_parameters() if q.requires_grad and n_ not in sd]
        assert not miss, f'체크포인트에 없는 파라미터 {miss[:3]}'
        m.load_state_dict(sd, strict=False)
    else: assert not tokv, '0 step은 --cond tok만 (V 어댑터는 학습 전 문이 0이라 tok와 같음)'
    from trellis.pipelines import samplers
    SJ = json.load(open(snaps[-1]))['args']['sparse_structure_sampler']; smp = getattr(samplers, SJ['name'])(**SJ['args']); sp = SJ['params']
    U = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{a.data}/tokens/*.npy'))[: a.limit or None]; n = 0
    for u in U:
        if os.path.exists(f'{a.out}/{u}.npy'): continue
        c = load_cond(a.data, u, tokv).to(dev)[None]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.manual_seed(42); noise = torch.randn(1, 8, 16, 16, 16, device=dev)
            z = smp.sample(m, noise, cond=c, neg_cond=torch.zeros_like(c), **sp, verbose=False).samples
        np.save(f'{a.out}/{u}.npy', z[0].float().cpu().numpy()); n += 1
    print(f'GENZ_DONE {n}개 · {a.ckpt} · {a.data}', flush=True)
