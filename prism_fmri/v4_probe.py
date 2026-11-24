# V-4 0단계 (11-24, Pro6000 GPU 1): Variational을 TRELLIS 형상 단계(sparse structure flow)의 조건으로 미세조정할 수 있나 — 학습 없는 확인
#  meshgen_tools venv, 실행 위치 /home/oem/jhj/meshgen_tools (trellis_tools.py와 같은 환경 · 같은 가중치)
#  axis  : test 물체 정답 64³ 점유(<tgt>/voxels/<uid>.npy, packbits) ↔ TRELLIS 좌표 맞추기
#          ① 정답 DINO 토큰(<tgt>/dino, 자극 프레임 0)으로 TRELLIS 1단계 flow → 점유 (seed 42)
#          ② 정답 점유를 축 바꿈 · 뒤집기 48가지로 돌려 ①과 IoU → 가장 잘 맞는 변환 (물체 대다수가 고른 것)
#          ③ 그 변환으로 정답 점유 → 형상 VAE 인코더(평균) → 디코더 → 다시 만든 IoU (정답 잠재가 제대로 만들어지나)
#          → <out>/axis.json
#  bench : 미세조정 메모리 · 속도. 조건 = (a) 토큰 [1374, 1024] / (b) 토큰 + V 갈래 특징 [390, 768] → 새 투영(0 초기화) → 뒤에 붙임
#          범위 = full(전체) / xattn(cross-attention · 새 투영만). 무작위 입력, 배치 여러 개, 각 --n step (앞 5 step은 빼고 잼)
#          손실 = TRELLIS FlowMatching 그대로 (x_t = (1 − t)·x0 + (σmin + (1 − σmin)·t)·ε, 목표 (1 − σmin)·ε − x0, t ~ logitNormal(1, 1), 모델에 t·1000)
#          → <out>/bench.json
#  tokcos: 예측 토큰(<pred>/<uid>.npy) 대 정답 DINO 토큰의 토큰별 cos 평균, train · test 따로 (학습 물체 토큰 과신 정도) → <out>/tokcos.json
# 사용: python v4_probe.py axis  --tgt $TGT --list $DR/annotations/core_test_list.txt --out $W/prism_v4/probe
#       python v4_probe.py bench --out $W/prism_v4/probe [--bs 8,16,32] [--n 25]
#       python v4_probe.py tokcos --tgt $TGT --pred <tokens 폴더> --train $DR/annotations/core_train_list.txt --test $DR/annotations/core_test_list.txt --out …
import os, sys, glob, json, time, argparse, itertools
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['axis', 'bench', 'tokcos'])
p.add_argument('--tgt', default='/home/oem/jhj/MinD-3D++/data/shape_targets'); p.add_argument('--list', default=''); p.add_argument('--out', required=True)
p.add_argument('--pred', default=''); p.add_argument('--train', default=''); p.add_argument('--test', default='')
p.add_argument('--bs', default='8,16,32'); p.add_argument('--n', type=int, default=25); p.add_argument('--limit', type=int, default=0)
p.add_argument('--trellis_repo', default='/home/oem/jhj/meshgen_tools/repo/third_party/TRELLIS')
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
names = lambda f: [l.strip().split('/')[-1] for l in open(f) if l.strip()]

if a.mode == 'tokcos':   # GPU 불필요
    R = {}
    for sp, f in (('train', a.train), ('test', a.test)):
        cs = []
        for u in names(f):
            fp, fg = f'{a.pred}/{u}.npy', f'{a.tgt}/dino/{u}.npy'
            if not (os.path.exists(fp) and os.path.exists(fg)): continue
            x, y = np.load(fp).astype(np.float32), np.load(fg).astype(np.float32)
            cs.append(float(((x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-8)).mean()))
        R[sp] = {'n': len(cs), 'mean': float(np.mean(cs)) if cs else None, 'p10': float(np.percentile(cs, 10)) if cs else None, 'p90': float(np.percentile(cs, 90)) if cs else None}
    print('TOKCOS', json.dumps(R), flush=True); json.dump(R, open(f'{a.out}/tokcos.json', 'w'), indent=1); sys.exit()

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
CK = os.path.join(os.path.dirname(snaps[-1]), 'ckpts'); print('ckpts', CK, flush=True)
dev = 'cuda'

if a.mode == 'axis':
    enc = TM.from_pretrained(f'{CK}/ss_enc_conv3d_16l8_fp16').to(dev).eval()
    dec = TM.from_pretrained(f'{CK}/ss_dec_conv3d_16l8_fp16').to(dev).eval()
    fm = TM.from_pretrained(f'{CK}/ss_flow_img_dit_L_16l8_fp16').to(dev).eval()
    from trellis.pipelines import samplers
    SJ = json.load(open(snaps[-1]))['args']['sparse_structure_sampler']; smp = getattr(samplers, SJ['name'])(**SJ['args']); sp = SJ['params']   # TRELLIS 기본 sampler 그대로
    print('sampler', SJ, flush=True)
    perms = [(pm, fl) for pm in itertools.permutations(range(3)) for fl in itertools.product([False, True], repeat=3)]
    def tf(o, pm, fl):   # o [64,64,64] bool tensor
        o = o.permute(*pm)
        d = [i for i in range(3) if fl[i]]
        return o.flip(d) if d else o
    iou = lambda x, y: float((x & y).sum()) / max(float((x | y).sum()), 1.0)
    U = [u for u in names(a.list) if os.path.exists(f'{a.tgt}/voxels/{u}.npy')][: a.limit or None]
    print('물체', len(U), '/', len(names(a.list)), flush=True)
    votes, rows = {}, {}
    for i, u in enumerate(U):
        g = torch.from_numpy(np.unpackbits(np.load(f'{a.tgt}/voxels/{u}.npy'))[: 64 ** 3].reshape(64, 64, 64).astype(bool)).to(dev)
        c = torch.from_numpy(np.load(f'{a.tgt}/dino/{u}.npy')).float().to(dev)[None]
        with torch.no_grad():
            torch.manual_seed(42); noise = torch.randn(1, fm.in_channels, 16, 16, 16, device=dev)
            z = smp.sample(fm, noise, cond=c, neg_cond=torch.zeros_like(c), **sp, verbose=False).samples
            o = (dec(z) > 0)[0, 0]
        sc = [iou(tf(g, pm, fl), o) for pm, fl in perms]; k = int(np.argmax(sc))
        votes[k] = votes.get(k, 0) + 1; rows[u] = {'best': k, 'iou_best': sc[k], 'iou_identity': sc[0]}
        if i % 20 == 0: print(f'[axis {i + 1}/{len(U)}] {u} 최고 {perms[k]} IoU {sc[k]:.3f} (그대로 {sc[0]:.3f}) 점유 정답 {int(g.sum())} 생성 {int(o.sum())}', flush=True)
    kb = max(votes, key=votes.get); pm, fl = perms[kb]
    rec = []
    for u in U:   # ③ 정해진 변환으로 정답 잠재 → 디코더 IoU
        g = torch.from_numpy(np.unpackbits(np.load(f'{a.tgt}/voxels/{u}.npy'))[: 64 ** 3].reshape(64, 64, 64).astype(bool)).to(dev)
        gt = tf(g, pm, fl)
        with torch.no_grad():
            z = enc(gt[None, None].float(), sample_posterior=False); o = (dec(z) > 0)[0, 0]
        rec.append(iou(gt, o))
    R = {'perm': list(pm), 'flip': list(fl), 'votes': {str(perms[k]): v for k, v in sorted(votes.items(), key=lambda x: -x[1])},
         'iou_gen_vs_gt_best_mean': float(np.mean([r['iou_best'] for r in rows.values()])), 'iou_gen_vs_gt_identity_mean': float(np.mean([r['iou_identity'] for r in rows.values()])),
         'iou_gen_vs_gt_chosen_mean': None, 'vae_recon_iou_mean': float(np.mean(rec)), 'vae_recon_iou_p10': float(np.percentile(rec, 10)), 'n': len(U), 'z_shape': [8, 16, 16, 16], 'rows': rows}
    print('AXIS', json.dumps({k: v for k, v in R.items() if k != 'rows'}, ensure_ascii=False), flush=True)
    json.dump(R, open(f'{a.out}/axis.json', 'w'), indent=1, ensure_ascii=False); print('AXIS_DONE', flush=True)

if a.mode == 'bench':
    R = []
    for cond_mode, scope in (('tok', 'full'), ('tok+v', 'full'), ('tok', 'xattn'), ('tok+v', 'xattn')):
        for bs in [int(x) for x in a.bs.split(',')]:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            fm = TM.from_pretrained(f'{CK}/ss_flow_img_dit_L_16l8_fp16').to(dev); fm.convert_to_fp32(); fm.use_fp16 = False; fm.dtype = torch.float32; fm.train()   # 학습은 fp32 가중치 + bf16 autocast
            vproj = nn.Linear(768, 1024).to(dev); nn.init.zeros_(vproj.weight); nn.init.zeros_(vproj.bias)
            if scope == 'xattn':
                for n_, q in fm.named_parameters(): q.requires_grad = 'cross_attn' in n_ or 'norm2' in n_
            params = [q for q in fm.parameters() if q.requires_grad] + (list(vproj.parameters()) if cond_mode == 'tok+v' else [])
            opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.0)
            ntrain = sum(q.numel() for q in params)
            r = {'cond': cond_mode, 'scope': scope, 'bs': bs, 'trainable_M': round(ntrain / 1e6, 1), 'precision': 'fp32 가중치 + bf16 autocast', 'params_M': round(sum(q.numel() for q in fm.parameters()) / 1e6, 1)}
            try:
                ts = []
                for s in range(a.n):
                    x0 = torch.randn(bs, 8, 16, 16, 16, device=dev); c = torch.randn(bs, 1374, 1024, device=dev)
                    if cond_mode == 'tok+v': c = torch.cat([c, vproj(torch.randn(bs, 390, 768, device=dev))], 1)
                    t = torch.sigmoid(torch.randn(bs, device=dev) + 1.0); e = torch.randn_like(x0); tt = t.view(-1, 1, 1, 1, 1)
                    xt = (1 - tt) * x0 + (1e-5 + (1 - 1e-5) * tt) * e
                    torch.cuda.synchronize(); t0 = time.time()
                    with torch.autocast('cuda', dtype=torch.bfloat16): out_ = fm(xt, t * 1000, c)
                    loss = F.mse_loss(out_.float(), (1 - 1e-5) * e - x0)
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
                    torch.cuda.synchronize(); ts.append(time.time() - t0)
                st = float(np.mean(ts[5:])); r.update({'ok': True, 'step_s': round(st, 3), 'step_per_s': round(1 / st, 2), 'obj_per_s': round(bs / st, 1),
                                                       'peak_GB': round(torch.cuda.max_memory_allocated() / 2 ** 30, 1), 'loss_finite': bool(torch.isfinite(loss).item())})
            except torch.cuda.OutOfMemoryError:
                r.update({'ok': False, 'oom': True})
            print('BENCH', json.dumps(r), flush=True); R.append(r)
            del fm, opt, vproj, params
    json.dump(R, open(f'{a.out}/bench.json', 'w'), indent=1); print('BENCH_DONE', flush=True)
