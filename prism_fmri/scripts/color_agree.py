# 렌더 색 일치 (11-08 색 코드 실험): 평가 렌더(물체 · 뷰)의 색 코드가 자극 색 코드와 얼마나 맞나
#  색 코드 = 정규채도(픽셀마다 (max−min)/max 평균) · rg_n = 3(R−G)/(R+G+B) · by_n = 3((R+G)/2−B)/(R+G+B), 음영에 덜 민감한 지표 (probe_color_raw.py와 같은 정의)
#  렌더: 물체 영역 = (alpha > 200, RGBA일 때) 그리고 min RGB < 235, 5×5로 깎음 — 자극 코드(probe_color_raw.py)와 같은 규칙, 뷰 전부 합침. 자극 코드 = data/stim_color.npz (영상 48프레임)
#  출력 (이름마다): 물체 104개에 걸친 Pearson r(렌더 코드, 자극 코드) · 평균 절대 오차 · 렌더 정규채도 평균
#  참고 줄: 범주 기준선 = train 같은 범주 평균 코드를 렌더 대신 넣었을 때 r · 오차 (범주만 알 때의 값), 자극 정규채도 평균
# 사용: python color_agree.py --npz <stim_color.npz> --ids <core_test_list.txt> --train_ids <core_train_list.txt> 이름=렌더폴더 [...]  [--json 출력]
import os, json, argparse, numpy as np
from PIL import Image, ImageFilter
p = argparse.ArgumentParser()
p.add_argument('--npz', required=True); p.add_argument('--ids', required=True); p.add_argument('--train_ids', required=True)
p.add_argument('--json', default=''); p.add_argument('runs', nargs='+')
a = p.parse_args()
CN = ['정규채도', 'rg_n', 'by_n']
z = np.load(a.npz); names = list(z['names']); Y = z['Y'][:, [names.index(c) for c in CN]].astype(np.float64); CODE = dict(zip(list(z['keys']), Y))
te = [l.strip() for l in open(a.ids) if l.strip()]; tr = [l.strip() for l in open(a.train_ids) if l.strip()]
G = np.stack([CODE[k] for k in te])


def code(px):
    mx, mn = px.max(1), px.min(1); R, Gc, B = px.mean(0); S = R + Gc + B + 1e-6
    return np.array([((mx - mn) / np.maximum(mx, 1)).mean(), 3 * (R - Gc) / S, 3 * ((R + Gc) / 2 - B) / S])


def obj_px(f):
    im = np.asarray(Image.open(f)); rgb = im[..., :3].astype(np.float32)
    m = rgb.min(2) < 235
    if im.shape[-1] == 4: m &= im[..., 3] > 200
    m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 0   # 5×5 깎기 (cv2.erode와 같음)
    return rgb[m] if m.sum() > 50 else None


def score(P, idx):
    Gs = G[idx]
    r = [float(np.corrcoef(P[:, j], Gs[:, j])[0, 1]) if P[:, j].std() > 1e-9 else 0.0 for j in range(3)]
    return {'r': dict(zip(CN, r)), 'mae': dict(zip(CN, np.abs(P - Gs).mean(0).tolist())), 'n': int(len(idx)), 'render_nchroma': float(P[:, 0].mean())}


show = lambda n, s: print(f"CAGREE {n:<22} r " + ' '.join(f"{c} {s['r'][c]:+.3f}" for c in CN) + ' | 오차 ' + ' '.join(f"{c} {s['mae'][c]:.3f}" for c in CN)
                          + f" | 정규채도 평균 {s['render_nchroma']:.3f} (n {s['n']})", flush=True)
cm = {c: np.mean([CODE[k] for k in tr if k.startswith(c + '/')], 0) for c in {k.split('/')[0] for k in te}}
R = {'category_baseline': score(np.stack([cm[k.split('/')[0]] for k in te]), np.arange(len(te))), 'stim_nchroma': float(G[:, 0].mean())}
show('범주 기준선', R['category_baseline']); print(f"CAGREE 자극 정규채도 평균 {R['stim_nchroma']:.3f}", flush=True)
uids = [k.split('/')[1] for k in te]
for run in a.runs:
    name, root = run.split('=', 1); dirs = {}
    for d, _, fs in os.walk(root):
        if '0.png' in fs:
            b = os.path.basename(d); i = next((j for j, u in enumerate(uids) if b.startswith(u)), None)
            if i is not None: dirs.setdefault(i, d)
    P, idx = [], []
    for i, d in sorted(dirs.items()):
        px = [x for x in (obj_px(f'{d}/{f}') for f in sorted(os.listdir(d)) if f.endswith('.png')) if x is not None]
        if px: P.append(code(np.concatenate(px))); idx.append(i)
    if not P: print(f'CAGREE {name} 렌더 없음 {root}', flush=True); continue
    R[name] = score(np.stack(P), np.array(idx)); show(name, R[name])
if a.json: json.dump(R, open(a.json, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('CAGREE_DONE', flush=True)
