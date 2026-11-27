# V-8 (11-28): 평가 렌더의 전경 밝기 · 채도 (A안 때 이미지 조건 외형이 어둡게 나온 문제 확인용, 학습 없음)
#  렌더 out/<이름>/render_R2/<uid>/{0..5}.png (RGBA) → 알파 > 200 픽셀의 RGB 평균(0 ~ 255) · 채도(max − min) 평균 → 물체 평균 → 세트 평균
#  같은 식으로 정답 텍스처 렌더(gt_render/R2)도 잼
# 사용: python v8_bright.py --fe <fair_eval 번들> --sets 이름1,이름2,… --out <json>
import os, json, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser(); p.add_argument('--fe', required=True); p.add_argument('--sets', required=True); p.add_argument('--out', required=True)
a = p.parse_args()
keys = [l.strip().split('/')[-1] for l in open(f'{a.fe}/core_test_list.txt') if l.strip()]
def one(d):
    b, c = [], []
    for v in range(6):
        f = f'{d}/{v}.png'
        if not os.path.exists(f): continue
        im = np.asarray(Image.open(f).convert('RGBA')).astype(np.float32); m = im[..., 3] > 200
        if m.sum() < 50: continue
        px = im[..., :3][m]; b.append(px.mean()); c.append((px.max(1) - px.min(1)).mean())
    return (float(np.mean(b)), float(np.mean(c))) if b else None
R = {}
for s in ['GT'] + a.sets.split(','):
    root = f'{a.fe}/gt_render/R2' if s == 'GT' else f'{a.fe}/out/{s}/render_R2'
    v = {u: one(f'{root}/{u}') for u in keys}; v = {u: x for u, x in v.items() if x}
    R[s] = {'bright': round(float(np.mean([x[0] for x in v.values()])), 2), 'chroma': round(float(np.mean([x[1] for x in v.values()])), 2), 'n': len(v), 'per': v}
    print(f"BRIGHT {s:<28} 밝기 {R[s]['bright']:.1f} · 채도 {R[s]['chroma']:.1f} (n {len(v)})", flush=True)
json.dump(R, open(a.out, 'w'), indent=1)
