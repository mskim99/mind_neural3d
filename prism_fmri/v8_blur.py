# V-8 ㉣ (11-28): 흐릿한 이미지 조건 — 16×16 RGB(프레임 0)를 1024²로 키운 흰 배경 그림 → <out>/<uid>/0.png (이후 trellis_tools.py dino)
#  --src gt  : v9_feat.npz blur16_f0 (자극 영상 프레임 0을 INTER_AREA로 16×16)
#  --src pred: v9_read.py pred_<피험자>.npz의 pred_blur16_f0(범주 뺀 값) + 예측 범주(문장 명사 → 범주 id)의 train 평균, 0 ~ 1로 자름
#  키우기: PIL BILINEAR 16 → 1024
# 사용: python v8_blur.py --v9 v9_feat.npz --ann <annotations> --src gt|pred [--pred pred_sub-0001.npz --base prompt.json] --out <폴더>
import os, json, argparse, numpy as np
from PIL import Image
p = argparse.ArgumentParser()
p.add_argument('--v9', required=True); p.add_argument('--ann', required=True); p.add_argument('--src', default='gt', choices=['gt', 'pred'])
p.add_argument('--pred', default=''); p.add_argument('--base', default=''); p.add_argument('--out', required=True)
a = p.parse_args()
lst = lambda s: [l.strip() for l in open(os.path.join(a.ann, f'core_{s}_list.txt')) if l.strip()]
TR, TE = lst('train'), lst('test'); cat = lambda k: k.split('/')[0]; uid = lambda k: k.split('/')[-1]
Z = np.load(a.v9); K = {str(k): i for i, k in enumerate(Z['keys'])}; B = Z['blur16_f0'].astype(np.float64)
if a.src == 'gt':
    IM = {uid(k): B[K[k]] for k in TE}
else:
    _t = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'text_prompts.py'), encoding='utf-8').read()
    NAME = eval(_t[_t.index('NAME = ') + 7: _t.index('}', _t.index('NAME = ')) + 1]); N2C = {n: c for c, n in NAME.items()}
    base = json.load(open(a.base, encoding='utf-8')); cm = {c: B[[K[k] for k in TR if cat(k) == c]].mean(0) for c in NAME}
    P = np.load(a.pred); ok = P['ok_blur16_f0']; IM = {}
    for k, r in zip(P['name'], P['pred_blur16_f0']):
        u = uid(str(k)); full = np.zeros(ok.size); full[ok] = r
        IM[u] = np.clip(cm[N2C[base[u].split(' ', 1)[1]]] + full.reshape(16, 16, 3), 0, 1)
for u, x in IM.items():
    os.makedirs(f'{a.out}/{u}', exist_ok=True)
    Image.fromarray((np.clip(x, 0, 1) * 255).round().astype(np.uint8)).resize((1024, 1024), Image.BILINEAR).save(f'{a.out}/{u}/0.png')
print('V8_BLUR_DONE', a.src, len(IM), a.out, flush=True)
