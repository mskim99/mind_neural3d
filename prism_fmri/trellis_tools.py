# PRISM TRELLIS 도구 (meshgen_tools venv, Pro6000 GPU 1). 실행 위치 /home/oem/jhj/meshgen_tools
#  dino : 자극 영상 프레임 → TRELLIS 자체 전처리(preprocess_image: alpha bbox × 1.2 crop, 518², 검은 배경) + get_cond
#         → 조건 토큰 [1374, 1024] (DINOv2 ViT-L/14-reg x_prenorm + layer_norm) → <tgt>/dino/<uid>.npy (fp16)
#         흰 배경 영상이라 rembg 대신 min(RGB) < 235 를 alpha로 넣음
#  gen  : 예측 조건 토큰 <pred>/<name>.npy → TRELLIS 1·2단계 flow (이미지 인코딩 대신 예측 토큰) → GLB
#       sampler는 TRELLIS 기본값 그대로 (25 steps, cfg 5.0, neg_cond = 0 토큰)
#  gscheck: gen과 같은 생성(같은 seed) 후 TRELLIS 내부 Gaussian 렌더 색 vs to_glb가 구운 텍스처 색 비교 → <out>/gscheck.jsonl + <out>/<name>_gs.png
#         (GLB 굽는 단계에서 색이 어두워지는지 확인용, 10-30)
#  --slat_pred <폴더>: 외형 단계 분리 (11-09). 1단계(형상, SS)는 --pred 토큰, 2단계(외형, SLat)는 이 폴더의 같은 이름 토큰을 조건으로 씀 (gen · gscheck)
#  --slat_t0 t (0 < t < 1, --slat_pred와 같이): 외형 단계 SDEdit (11-09, prior 0단계). 예측 토큰으로 만든 SLat을 정규화 공간에서 시점 t까지 노이즈를 넣고
#       (x_t = (1 − t)·x0 + (σmin + (1 − σmin)·t)·ε, TRELLIS FlowEulerSampler와 같은 식) --slat_pred 조건으로 t → 0까지 같은 sampler(sample_once, 같은 step 표 · cfg)로 되돌림
#       t = 1이면 보통 외형 분리와 같음. t가 작을수록 예측 SLat을 더 남김
#  --slat_text <json> (11-09, 외형 = 텍스트판): 1단계(형상, SS)는 --pred 토큰, 2단계(외형, SLat)는 TRELLIS 텍스트판(--text_model)의 SLat flow가
#       json의 문장({이름: "a chair"})을 CLIP 텍스트 조건으로 생성. 디코더는 이미지판 것 (텍스트판도 image-large의 VAE를 씀 → 시작할 때 정규화 · 디코더 일치 확인)
#  sm_120에서 xformers 기본 커널(FA3)이 오류 → op 미지정 호출만 CUTLASS로 보냄 (run_trellis.py와 같은 처리, 계산식 동일)
# 사용: python trellis_tools.py dino --frames <tgt>/frames --out <tgt>/dino [--view 0]
#       python trellis_tools.py gen --pred <tokens 폴더> --out <glb 폴더> [--seed 42]
#       python trellis_tools.py gscheck --pred <tokens 폴더> --out <폴더> [--names <uid 목록 txt>]
import os, sys, glob, time, argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['dino', 'gen', 'gscheck'])
p.add_argument('--frames'); p.add_argument('--pred'); p.add_argument('--out', required=True)
p.add_argument('--view', type=int, default=0); p.add_argument('--seed', type=int, default=42); p.add_argument('--limit', type=int, default=0)
p.add_argument('--trellis_repo', default='/home/oem/jhj/meshgen_tools/repo/third_party/TRELLIS')
p.add_argument('--slat_pred', default='')   # 비우면 두 단계 모두 --pred 토큰
p.add_argument('--slat_t0', type=float, default=1.0)   # < 1이면 외형 단계 SDEdit
p.add_argument('--names', default='')   # gscheck: 이 목록의 이름만
p.add_argument('--slat_text', default='')   # 외형 = 텍스트판, {이름: 문장} json
p.add_argument('--text_model', default='microsoft/TRELLIS-text-base')
p.add_argument('--model', default='')   # 비우면 HF 캐시의 TRELLIS-image-large 스냅샷 폴더를 직접 씀 (오프라인)
a = p.parse_args()
os.environ.setdefault('SPCONV_ALGO', 'native')
# 서버 meshgen venv에는 flash_attn이 없음 → TRELLIS attention 백엔드를 xformers로 (기본값 flash_attn이면 모델 로드 실패)
os.environ.setdefault('ATTN_BACKEND', 'xformers'); os.environ.setdefault('SPARSE_ATTN_BACKEND', 'xformers')
import torch
import xformers.ops as xops
from xformers.ops import fmha
_mea = xops.memory_efficient_attention
def _mea_cutlass(q, k, v, attn_bias=None, p=0.0, scale=None, *, op=None, **kw):
    return _mea(q, k, v, attn_bias, p, scale, op=op if op is not None else (fmha.cutlass.FwOp, fmha.cutlass.BwOp), **kw)
xops.memory_efficient_attention = _mea_cutlass; fmha.memory_efficient_attention = _mea_cutlass
sys.path.insert(0, a.trellis_repo)
# torch.hub.load('facebookresearch/dinov2', …)는 캐시가 있어도 GitHub에 브랜치를 물어봄 → 가끔 HTTP 504로 GLB 0개 (11-09 sub-0008)
#  → 캐시 폴더가 있으면 source='local'로 (같은 코드 · 같은 가중치 캐시, 네트워크 안 씀)
_hub_load = torch.hub.load
def _hub_load_local(repo, model, *args, **kw):
    d = os.path.join(torch.hub.get_dir(), repo.replace('/', '_') + '_main')
    if isinstance(repo, str) and not os.path.isdir(repo) and os.path.isdir(d) and kw.get('source', 'github') == 'github':
        kw.pop('trust_repo', None); kw.pop('force_reload', None); kw.pop('skip_validation', None); kw['source'] = 'local'
        return _hub_load(d, model, *args, **kw)
    return _hub_load(repo, model, *args, **kw)
torch.hub.load = _hub_load_local
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

if not a.model:
    snaps = sorted(glob.glob(os.path.join(os.environ.get('HF_HOME', ''), 'hub', 'models--microsoft--TRELLIS-image-large', 'snapshots', '*', 'pipeline.json')))
    a.model = os.path.dirname(snaps[-1]) if snaps else 'microsoft/TRELLIS-image-large'
print('model', a.model, flush=True)
pipe = TrellisImageTo3DPipeline.from_pretrained(a.model); pipe.cuda()
for f in ('preprocess_image', 'get_cond', 'sample_sparse_structure', 'sample_slat', 'decode_slat'):
    assert hasattr(pipe, f), f'TRELLIS API에 {f} 없음'
print('기본 sampler', pipe.sparse_structure_sampler_params, pipe.slat_sampler_params, type(pipe.slat_sampler).__name__, flush=True)
TP, PROMPT = None, {}
if a.slat_text:
    import json
    from trellis.pipelines import TrellisTextTo3DPipeline
    PROMPT = json.load(open(a.slat_text, encoding='utf-8'))
    TP = TrellisTextTo3DPipeline.from_pretrained(a.text_model); TP.cuda()
    same = {k: bool(np.allclose(np.array(pipe.slat_normalization[k]), np.array(TP.slat_normalization[k]))) for k in ('mean', 'std')}
    dec = {k: bool(all(torch.equal(x, y) for x, y in zip(pipe.models[k].state_dict().values(), TP.models[k].state_dict().values()))) for k in pipe.models if k.startswith('slat_decoder') and k in TP.models}
    print('텍스트판', a.text_model, '| SLat 정규화 같음', same, '| 디코더 같음', dec, '| sampler', TP.slat_sampler_params, '| 문장', len(PROMPT), flush=True)
    assert all(same.values()) and dec and all(dec.values()), '텍스트판과 이미지판의 SLat 공간이 다름 → 외형만 바꿔 끼울 수 없음'
if a.slat_t0 < 1: assert a.slat_pred and hasattr(pipe.slat_sampler, 'sample_once') and hasattr(pipe.slat_sampler, 'sigma_min'), 'SDEdit은 --slat_pred와 FlowEuler sampler 필요'
def make_cond(c):
    return {'cond': c, 'neg_cond': torch.zeros_like(c)}      # TRELLIS get_cond와 같은 형식 (neg_cond = 0)


def sdedit(slat0, cond2, t0):   # 정규화 공간에서 t0까지 노이즈 → cond2로 t0 → 0 Euler (TRELLIS sampler 그대로)
    fm, smp = pipe.models['slat_flow_model'], pipe.slat_sampler; prm = dict(pipe.slat_sampler_params)
    std = torch.tensor(pipe.slat_normalization['std'])[None].to(slat0.feats.device); mean = torch.tensor(pipe.slat_normalization['mean'])[None].to(slat0.feats.device)
    s = smp.sigma_min; x0 = (slat0.feats - mean) / std
    x = slat0.replace((1 - t0) * x0 + (s + (1 - s) * t0) * torch.randn_like(x0))
    steps, rt = prm.pop('steps', 50), prm.pop('rescale_t', 1.0)          # 없으면 FlowEulerSampler.sample 기본값
    ts = np.linspace(1, 0, steps + 1); ts = rt * ts / (1 + (rt - 1) * ts); ts = [t0] + [float(t) for t in ts if t < t0]
    for t, tp in zip(ts[:-1], ts[1:]):
        x = smp.sample_once(fm, x, t, tp, **cond2, **prm).pred_x_prev
    return x.replace(x.feats * std + mean)


def generate(cond, cond2):
    coords = pipe.sample_sparse_structure(cond, 1)
    if TP is not None:
        slat = TP.sample_slat(TP.get_cond([cond2]), coords)                   # cond2 = 문장
    elif a.slat_pred and a.slat_t0 < 1:
        slat = sdedit(pipe.sample_slat(cond, coords), cond2, a.slat_t0)
    else:
        slat = pipe.sample_slat(cond2, coords)
    return pipe.decode_slat(slat, ['mesh', 'gaussian'])


def load_conds(name):   # (형상 조건, 외형 조건). --slat_pred 없으면 같은 것, 외형 토큰이 없으면 None
    ld = lambda f: make_cond(torch.from_numpy(np.load(f)).float().cuda()[None])
    c1 = ld(f'{a.pred}/{name}.npy')
    if a.slat_text: return c1, PROMPT.get(name)
    if not a.slat_pred: return c1, c1
    f2 = f'{a.slat_pred}/{name}.npy'
    return (c1, ld(f2)) if os.path.exists(f2) else (c1, None)


os.makedirs(a.out, exist_ok=True); t0 = time.time()

if a.mode == 'dino':
    dirs = sorted(d for d in os.listdir(a.frames) if os.path.exists(f'{a.frames}/{d}/{a.view}.png'))[: a.limit or None]
    for i, uid in enumerate(dirs):
        o = f'{a.out}/{uid}.npy'
        if os.path.exists(o): continue
        rgb = np.asarray(Image.open(f'{a.frames}/{uid}/{a.view}.png').convert('RGB'))
        alpha = ((rgb.min(2) < 235) * 255).astype(np.uint8)
        img = pipe.preprocess_image(Image.fromarray(np.dstack([rgb, alpha]), 'RGBA'))
        if i == 0: img.save(f'{a.out}/_check_preprocessed.png')
        cond = pipe.get_cond([img])['cond'][0]
        if i == 0: print('cond shape', tuple(cond.shape), cond.dtype, flush=True)
        np.save(o, cond.half().cpu().numpy())
        if (i + 1) % 100 == 0: print(f'[dino {i + 1}/{len(dirs)}] {uid} {time.time() - t0:.0f}s', flush=True)
    print('DINO_DONE', flush=True)
elif a.mode == 'gscheck':
    import json
    from trellis.utils import postprocessing_utils, render_utils
    names = [l.split()[-1] for l in open(a.names) if l.strip()] if a.names else [os.path.basename(f)[:-4] for f in sorted(glob.glob(f'{a.pred}/*.npy'))]
    def fg_stats(px):   # px: N×3 (0~255) → 평균 밝기, 평균 채도(max-min)
        return round(float(px.mean()), 1), round(float((px.max(1) - px.min(1)).mean()), 1)
    log = open(f'{a.out}/gscheck.jsonl', 'a')
    for i, name in enumerate(names[: a.limit or None]):
        f = f'{a.pred}/{name}.npy'
        if not os.path.exists(f): print('없음', f, flush=True); continue
        cond, cond2 = load_conds(name)
        if cond2 is None: print('외형 토큰 없음', name, flush=True); continue
        torch.manual_seed(a.seed)
        with torch.no_grad():
            outs = generate(cond, cond2)
        g = outs['gaussian'][0]
        k = render_utils.render_snapshot(g, resolution=512, bg_color=(0, 0, 0), verbose=False)['color']
        w = render_utils.render_snapshot(g, resolution=512, bg_color=(1, 1, 1), verbose=False)['color']
        k = np.stack(k).astype(np.float32); w = np.stack(w).astype(np.float32)
        al = 1 - (w - k).mean(-1) / 255                           # 검은·흰 배경 두 번 렌더로 알파 추정
        m = al > 0.95
        gs = fg_stats(k[m])                                        # 불투명한 곳의 Gaussian 색
        glb = postprocessing_utils.to_glb(g, outs['mesh'][0], simplify=0.95, texture_size=1024, verbose=False)
        tex = np.asarray(glb.visual.material.baseColorTexture.convert('RGB')).reshape(-1, 3).astype(np.float32)
        tx = fg_stats(tex[tex.sum(1) > 0])
        r = {'name': name, 'gauss_mean_chroma': gs, 'tex_mean_chroma': tx, 'opaque_px': int(m.sum())}
        log.write(json.dumps(r) + chr(10)); log.flush(); print(f'[gscheck {i + 1}] {r}', flush=True)
        Image.fromarray(np.concatenate(list(w.astype(np.uint8)), 1)).save(f'{a.out}/{name}_gs.png')
    print('GSCHECK_DONE', flush=True)
else:
    from trellis.utils import postprocessing_utils
    files = sorted(glob.glob(f'{a.pred}/*.npy'))[: a.limit or None]
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-4]; od = f'{a.out}/{name}'
        if os.path.exists(f'{od}/{name}.glb'): continue
        t1 = time.time()
        cond, cond2 = load_conds(name)
        if cond2 is None: print('외형 토큰 없음', name, flush=True); continue
        torch.manual_seed(a.seed)
        with torch.no_grad():
            outs = generate(cond, cond2)
        os.makedirs(od, exist_ok=True)
        glb = postprocessing_utils.to_glb(outs['gaussian'][0], outs['mesh'][0], simplify=0.95, texture_size=1024, verbose=False)
        glb.export(f'{od}/{name}.glb')
        print(f'[gen {i + 1}/{len(files)}] {name} {time.time() - t1:.1f}s', flush=True)
    print('GEN_DONE', flush=True)
