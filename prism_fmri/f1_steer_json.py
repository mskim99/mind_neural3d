# 11-28 F-1: 모델 안 Statistic Regressor 예측(components.npz app[:, 0] = ŝ_PC1) → 조향 α json (V-10과 같은 문장 쌍1 · α = k × clip(ŝ, ±2))
# 사용: python f1_steer_json.py --comp <infer>/<sub>_keep/components.npz --base <text>/prompt.json --out <json> [--k 0.5]
import json, argparse, numpy as np
p = argparse.ArgumentParser(); p.add_argument('--comp', required=True); p.add_argument('--base', required=True); p.add_argument('--out', required=True); p.add_argument('--k', type=float, default=0.5)
p.add_argument('--plus', default='{an} with fine surface details and patterns'); p.add_argument('--minus', default='{an} with a smooth plain surface')
a = p.parse_args(); C = np.load(a.comp); B = json.load(open(a.base, encoding='utf-8')); out = {}
for k, v in zip(C['name'], C['app'][:, 0]):
    u = str(k).split('/')[-1]; out[u] = {'plus': a.plus.format(an=B[u]), 'minus': a.minus.format(an=B[u]), 'alpha': round(a.k * float(np.clip(v, -2, 2)), 4), 'z': round(float(v), 4)}
json.dump(out, open(a.out, 'w'), indent=1); al = np.array([v['alpha'] for v in out.values()]); print(f'F1_STEER {len(out)} α {al.min():.2f} ~ {al.max():.2f}', flush=True)
