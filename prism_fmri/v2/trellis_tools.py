# PRISM v2 TRELLIS 도구 (meshgen_tools venv, Pro6000 GPU 1). 실행 위치 /home/oem/jhj/meshgen_tools
#  dino : 자극 영상 프레임 → TRELLIS 자체 전처리(preprocess_image: alpha bbox × 1.2 crop, 518², 검은 배경) + get_cond
#         → 조건 토큰 [1374, 1024] (DINOv2 ViT-L/14-reg x_prenorm + layer_norm) → <tgt>/dino/<uid>.npy (fp16)
#         흰 배경 영상이라 rembg 대신 min(RGB) < 235 를 alpha로 넣음
#  gen  : 예측 조건 토큰 <pred>/<name>.npy → TRELLIS 1·2단계 flow (이미지 인코딩 대신 예측 토큰) → GLB
#  sm_120에서 xformers 기본 커널(FA3)이 오류 → op 미지정 호출만 CUTLASS로 보냄 (run_trellis.py와 같은 처리, 계산식 동일)
# 사용: python trellis_tools.py dino --frames <tgt>/frames --out <tgt>/dino [--view 0]
#       python trellis_tools.py gen --pred <tokens 폴더> --out <glb 폴더> [--seed 42]
import os, sys, glob, time, argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['dino', 'gen'])
p.add_argument('--frames'); p.add_argument('--pred'); p.add_argument('--out', required=True)
p.add_argument('--view', type=int, default=0); p.add_argument('--seed', type=int, default=42); p.add_argument('--limit', type=int, default=0)
p.add_argument('--trellis_repo', default='/home/oem/jhj/meshgen_tools/repo/third_party/TRELLIS')
p.add_argument('--model', default='microsoft/TRELLIS-image-large')
a = p.parse_args()
os.environ.setdefault('SPCONV_ALGO', 'native')
import torch
import xformers.ops as xops
from xformers.ops import fmha
_mea = xops.memory_efficient_attention
def _mea_cutlass(q, k, v, attn_bias=None, p=0.0, scale=None, *, op=None, **kw):
    return _mea(q, k, v, attn_bias, p, scale, op=op if op is not None else (fmha.cutlass.FwOp, fmha.cutlass.BwOp), **kw)
xops.memory_efficient_attention = _mea_cutlass; fmha.memory_efficient_attention = _mea_cutlass
sys.path.insert(0, a.trellis_repo)
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline

pipe = TrellisImageTo3DPipeline.from_pretrained(a.model); pipe.cuda()
for f in ('preprocess_image', 'get_cond', 'sample_sparse_structure', 'sample_slat', 'decode_slat'):
    assert hasattr(pipe, f), f'TRELLIS API에 {f} 없음'
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
else:
    from trellis.utils import postprocessing_utils
    files = sorted(glob.glob(f'{a.pred}/*.npy'))[: a.limit or None]
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-4]; od = f'{a.out}/{name}'
        if os.path.exists(f'{od}/{name}.glb'): continue
        t1 = time.time()
        cond = torch.from_numpy(np.load(f)).float().cuda()[None]
        cond = {'cond': cond, 'neg_cond': torch.zeros_like(cond)}
        torch.manual_seed(a.seed)
        with torch.no_grad():
            coords = pipe.sample_sparse_structure(cond, 1, {})
            slat = pipe.sample_slat(cond, coords, {})
            outs = pipe.decode_slat(slat, ['mesh', 'gaussian'])
        os.makedirs(od, exist_ok=True)
        glb = postprocessing_utils.to_glb(outs['gaussian'][0], outs['mesh'][0], simplify=0.95, texture_size=1024, verbose=False)
        glb.export(f'{od}/{name}.glb')
        print(f'[gen {i + 1}/{len(files)}] {name} {time.time() - t1:.1f}s', flush=True)
    print('GEN_DONE', flush=True)
