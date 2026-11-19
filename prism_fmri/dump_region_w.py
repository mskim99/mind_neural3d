# -*- coding: utf-8 -*-
# 11-20 (b) 학습된 영역 가중(prism.RegionWeight)을 정성 · 정량으로 저장 (사용자 요구: 학습이 끝날 때마다 자동) — train_prism.py가 끝에 부름, 따로도 실행 가능
#  출력 <out>/ : region_w_map.png (S · V 각각 공통 + 피험자별 가중을 평면 지도에), region_w.csv (영역별 초기 · 공통 · 피험자별 가중),
#               region_w.md (가중 위 · 아래 영역, 피험자별 가중과 공통 가중의 차이, 피험자 간 가중 상관)
# 사용: python dump_region_w.py --ckpt $W/prism_2s/<이름>/model_final.pt [--out 폴더(기본 체크포인트 폴더/region_w)]
import os, json, argparse, itertools, numpy as np, torch


def dump(ckpt, out=None):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False) if isinstance(ckpt, str) else ckpt
    sd = ck['model']; subs = list(ck['subs']); out = out or os.path.join(os.path.dirname(os.path.abspath(ckpt)), 'region_w')
    if 'rw.a_s' not in sd: print('영역 가중 없음 (region_w 0)'); return
    os.makedirs(out, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    lab = np.load(os.path.join(here, 'assets', 'mmp_256_regions.npy')).astype(np.int64)
    NAME = {int(k): v for k, v in json.load(open(os.path.join(here, 'assets', 'mmp_256_regions.json'), encoding='utf-8')).items()}
    R = int(lab.max()) + 1; px = np.bincount(lab.flatten(), minlength=R); ok = (np.arange(R) > 0) & (px >= 40)
    sig = lambda t: 1 / (1 + np.exp(-t))
    W = {}
    for b in ('s', 'v'):
        a_ = sd[f'rw.a_{b}'].float().numpy(); d_ = sd[f'rw.d_{b}'].float().numpy()
        W[b] = {'init': sig(sd['rw.a_init'].float().numpy()), 'common': sig(a_), 'sub': sig(a_[None] + d_)}   # [R], [R], [K, R]
    # 표 (csv)
    with open(os.path.join(out, 'region_w.csv'), 'w', encoding='utf-8') as f:
        f.write('region_id,name,px,' + ','.join(f'{b}_{k}' for b in 'sv' for k in ['init', 'common'] + [s[-2:] for s in subs]) + '\n')
        for r in np.where(ok)[0]:
            f.write(f'{r},{NAME.get(int(r), r)},{px[r]},' + ','.join(f'{x:.4f}' for b in 'sv' for x in [W[b]['init'][r], W[b]['common'][r]] + list(W[b]['sub'][:, r])) + '\n')
    # 요약 (md)
    L = [f'# 영역 가중 — {os.path.abspath(ckpt) if isinstance(ckpt, str) else ""}', '', f'피험자 {", ".join(subs)} · 영역 {int(ok.sum())}개 (픽셀 40개 이상) · step {ck.get("step")}', '']
    for b, nm in (('s', 'Semantic'), ('v', 'Variational')):
        c = W[b]['common'][ok]; i0 = W[b]['init'][ok]; S = W[b]['sub'][:, ok]; names = [NAME.get(int(r), str(r)) for r in np.where(ok)[0]]
        o = np.argsort(-c)
        L += [f'## {nm}', '', f'- 공통 가중 범위 {c.min():.3f} ~ {c.max():.3f} (초기 {i0.min():.3f} ~ {i0.max():.3f}), 초기와 상관 {np.corrcoef(c, i0)[0, 1]:.3f}' if i0.std() > 1e-6 else f'- 공통 가중 범위 {c.min():.3f} ~ {c.max():.3f} (초기 모두 {i0.mean():.3f})',
              '- 위 10: ' + ' · '.join(f'{names[j]} {c[j]:.2f}' for j in o[:10]), '- 아래 10: ' + ' · '.join(f'{names[j]} {c[j]:.2f}' for j in o[-10:]), '',
              '| 피험자 | 공통과의 평균 절대 차 | 공통과 상관 | 가장 올린 영역 | 가장 내린 영역 |', '|---|---|---|---|---|']
        for k, s in enumerate(subs):
            dd = S[k] - c; L.append(f'| {s} | {np.abs(dd).mean():.4f} | {np.corrcoef(S[k], c)[0, 1]:.3f} | {names[int(dd.argmax())]} {dd.max():+.3f} | {names[int(dd.argmin())]} {dd.min():+.3f} |')
        if len(subs) > 1:
            cc = [np.corrcoef(S[i] - c, S[j] - c)[0, 1] for i, j in itertools.combinations(range(len(subs)), 2)]
            L += ['', f'- 피험자 간 가중 상관(가중 자체) {np.mean([np.corrcoef(S[i], S[j])[0, 1] for i, j in itertools.combinations(range(len(subs)), 2)]):.3f}, 조정값끼리 상관 {np.mean(cc):.3f}']
        L.append('')
    open(os.path.join(out, 'region_w.md'), 'w', encoding='utf-8').write('\n'.join(L))
    # 그림: 행 = S · V, 열 = 초기 · 공통 · 피험자별
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        cols = ['init', 'common'] + subs; n = len(cols)   # 그림 글은 영어 (서버에 한글 글꼴이 없을 수 있음)
        fig, ax = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
        for i, (b, nm) in enumerate((('s', 'Semantic'), ('v', 'Variational'))):
            for j, col in enumerate(cols):
                w = W[b]['init'] if j == 0 else W[b]['common'] if j == 1 else W[b]['sub'][j - 2]
                img = np.where(lab > 0, w[lab], np.nan); im = ax[i, j].imshow(img, cmap='viridis', vmin=0, vmax=1)
                ax[i, j].set_title(f'{nm} · {col}', fontsize=9); ax[i, j].axis('off')
        fig.colorbar(im, ax=ax, shrink=0.6, label='weight')
        fig.savefig(os.path.join(out, 'region_w_map.png'), dpi=130, bbox_inches='tight'); plt.close(fig)
    except Exception as e:
        print('그림 실패:', e)
    print(f'REGION_W_DUMP {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--out', default=None); a = p.parse_args()
    dump(a.ckpt, a.out)
