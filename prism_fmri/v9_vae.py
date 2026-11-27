# V-9 ② (11-28, 5090 GPU 0, venv_unclip): 자극 프레임 0 → SD 2.1 VAE latent (4×64×64, 분포 평균 × scaling_factor) — 학습 없음
#  프레임: data/shape_targets/frames/<uid>/0.png (영상 프레임 0, 1024², 흰 배경) → 512² (LANCZOS) → [-1, 1]
#  VAE: 서버에 이미 있는 sd2-community/stable-diffusion-2-1-unclip 의 vae (새로 받지 않음, HF_HUB_OFFLINE)
# 출력: <out> npz — keys [N] (범주/uid), lat [N, 4, 64, 64] fp16
# 사용: HF_HOME=$W/cache/hf HF_HUB_OFFLINE=1 $W/venv_unclip/bin/python v9_vae.py --frames $TGT/frames --ann $DR/annotations --out v9_vae.npz
import os, argparse, numpy as np, torch
from PIL import Image
from diffusers import AutoencoderKL
p = argparse.ArgumentParser(); p.add_argument('--frames', required=True); p.add_argument('--ann', required=True); p.add_argument('--out', required=True)
p.add_argument('--model', default='sd2-community/stable-diffusion-2-1-unclip'); p.add_argument('--bs', type=int, default=16)
a = p.parse_args()
keys = [l.strip() for f in ('core_train_list.txt', 'core_test_list.txt') for l in open(os.path.join(a.ann, f)) if l.strip()]
keys = [k for k in keys if os.path.exists(f'{a.frames}/{k.split("/")[-1]}/0.png')]
vae = AutoencoderKL.from_pretrained(a.model, subfolder='vae', torch_dtype=torch.float16).cuda().eval(); sf = vae.config.scaling_factor
print('VAE', a.model, 'scaling', sf, '물체', len(keys), flush=True); out = []
with torch.no_grad():
    for i in range(0, len(keys), a.bs):
        x = np.stack([np.asarray(Image.open(f'{a.frames}/{k.split("/")[-1]}/0.png').convert('RGB').resize((512, 512), Image.LANCZOS)) for k in keys[i:i + a.bs]])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).half().cuda() / 127.5 - 1
        out.append((vae.encode(x).latent_dist.mean * sf).cpu().numpy().astype(np.float16))
        if i % 320 == 0: print(f'[{i + len(x)}/{len(keys)}]', flush=True)
np.savez(a.out, keys=np.array(keys), lat=np.concatenate(out)); print('V9_VAE_DONE', len(keys), a.out, flush=True)
