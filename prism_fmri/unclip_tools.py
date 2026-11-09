# PRISM 외형 A안 (11-09, 5090): CLIP 이미지 임베딩 → SD 2.1 unCLIP(사전학습, 그대로) → 이미지 → (trellis_tools dino --rembg 1) → 외형(SLat) 조건
#  venv: MinD-3D++/venv_unclip (diffusers 0.32.2). 가중치: sd2-community/stable-diffusion-2-1-unclip (stabilityai 원본은 11-09 접근 불가, 제3자 미러)
#  embcheck : 자극 프레임에서 unCLIP 이미지 인코더(HF CLIPVisionModelWithProjection) 임베딩 vs OpenCLIP ViT-H-14 laion2b_s32b_b79k(Semantic이 쓰는 것)
#             임베딩의 코사인 → 같은 공간인지 확인 (PRISM이 예측할 대상을 정하려고)
#  embed    : --frames의 모든 물체 프레임 → <out>/<name>.npy (1024, fp32) — A안 2단계 학습 정답
#  gen      : 임베딩 → unCLIP 이미지 <out>/<name>/0.png. 임베딩은 --frames 정답 프레임에서 계산(학습 없는 상한) 또는 --embeds npz {name: [1024]}
# 사용: python unclip_tools.py embcheck --frames <tgt>/frames --names <uid 목록>
#       python unclip_tools.py gen --frames <tgt>/frames --names <uid 목록> --out <폴더> [--embeds <npz>] [--seed 42] [--noise_level 0]
import os, glob, json, argparse, time
import numpy as np, torch
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument('mode', choices=['embcheck', 'gen', 'embed'])
p.add_argument('--frames', default=''); p.add_argument('--view', type=int, default=0); p.add_argument('--names', default='')
p.add_argument('--out', default=''); p.add_argument('--embeds', default=''); p.add_argument('--norm', type=float, default=0)   # gen: > 0이면 임베딩 길이를 이 값으로 (예측은 방향만 믿음)
p.add_argument('--model', default='sd2-community/stable-diffusion-2-1-unclip')
p.add_argument('--seed', type=int, default=42); p.add_argument('--noise_level', type=int, default=0)
p.add_argument('--steps', type=int, default=20); p.add_argument('--cfg', type=float, default=10.0)   # diffusers StableUnCLIPImg2ImgPipeline 기본값
p.add_argument('--prompt', default='')
p.add_argument('--gamma', type=float, default=1.0)   # gen: < 1이면 저장 이미지를 밝게 (x^γ). 11-10 진단: unCLIP 물체가 어두워 외형이 어두움 → 0.5가 가장 나음
a = p.parse_args()
names = [l.strip() for l in open(a.names) if l.strip()] if a.names else sorted(os.listdir(a.frames))
frame = lambda n: Image.open(f'{a.frames}/{n}/{a.view}.png').convert('RGB')
dev = 'cuda'

from diffusers import StableUnCLIPImg2ImgPipeline
pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(a.model, torch_dtype=torch.float16, variant='fp16').to(dev)
pipe.set_progress_bar_config(disable=True)


@torch.no_grad()
def hf_embed(img):   # unCLIP 파이프라인이 image=로 받을 때와 같은 경로 (feature_extractor → image_encoder.image_embeds)
    x = pipe.feature_extractor(images=img, return_tensors='pt').pixel_values.to(dev, torch.float16)
    return pipe.image_encoder(x).image_embeds.float()[0]


if a.mode == 'embed':
    os.makedirs(a.out, exist_ok=True); n0 = 0
    for i, n in enumerate(names):
        o = f'{a.out}/{n}.npy'
        if os.path.exists(o) or not os.path.exists(f'{a.frames}/{n}/{a.view}.png'): continue
        np.save(o, hf_embed(frame(n)).cpu().numpy().astype(np.float32)); n0 += 1
    E = np.stack([np.load(f'{a.out}/{n}.npy') for n in names if os.path.exists(f'{a.out}/{n}.npy')])
    print(f'EMBED_DONE 새로 {n0} · 전체 {len(E)} | 길이 평균 {np.linalg.norm(E, axis=1).mean():.3f} 표준편차 {np.linalg.norm(E, axis=1).std():.3f}', flush=True)
elif a.mode == 'embcheck':
    import open_clip
    oc, _, pre = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k'); oc = oc.to(dev).eval()
    cs, cs_raw = [], []
    for n in names:
        img = frame(n); e1 = hf_embed(img)
        with torch.no_grad(): e2 = oc.encode_image(pre(img)[None].to(dev)).float()[0]
        cs.append(torch.nn.functional.cosine_similarity(e1, e2, 0).item())
    # 같은 물체끼리 vs 다른 물체끼리 (공간이 같아도 전처리가 다르면 1이 아님 → 다른 물체 쌍보다 확실히 높은지)
    print(f'EMBCHECK n {len(cs)} | HF unCLIP 인코더 vs OpenCLIP ViT-H-14 laion2b: 같은 이미지 cos 평균 {np.mean(cs):.4f} 최소 {np.min(cs):.4f}', flush=True)
    print('HF 인코더 출력 차원', pipe.image_encoder.config.projection_dim, '| 은닉', pipe.image_encoder.config.hidden_size, '| 층', pipe.image_encoder.config.num_hidden_layers, flush=True)
else:
    os.makedirs(a.out, exist_ok=True)
    E = dict(np.load(a.embeds)) if a.embeds else {}
    saved = {}
    for i, n in enumerate(names):
        o = f'{a.out}/{n}/0.png'
        if os.path.exists(o): continue
        t1 = time.time()
        e = torch.from_numpy(E[n]).float().to(dev) if a.embeds else hf_embed(frame(n))
        if a.norm > 0: e = e / e.norm() * a.norm
        saved[n] = e.cpu().numpy()
        g = torch.Generator(dev).manual_seed(a.seed)
        img = pipe(image_embeds=e[None].half(), prompt=a.prompt, noise_level=a.noise_level, num_inference_steps=a.steps,
                   guidance_scale=a.cfg, generator=g).images[0]
        if a.gamma != 1.0: img = Image.fromarray((np.clip((np.asarray(img).astype(np.float32) / 255) ** a.gamma, 0, 1) * 255).astype(np.uint8))
        os.makedirs(os.path.dirname(o), exist_ok=True); img.save(o)
        print(f'[unclip {i + 1}/{len(names)}] {n} {img.size} {time.time() - t1:.1f}s', flush=True)
    if saved and not a.embeds: np.savez(f'{a.out}/embeds_gt.npz', **saved)
    print('UNCLIP_DONE', flush=True)
