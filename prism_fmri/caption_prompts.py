# 캡션 → TRELLIS 텍스트판 문장 json (11-10, CPU, MinD-3D++ venv) — {uid: 문장}, trellis_tools.py gen --slat_text
#  oracle_{color,nocolor}: test 물체 자기 캡션 = 상한 확인용 (방법 아님). 풍부한 문장이 "a {범주}"보다 나은지부터 봄
#  retr_{color,nocolor}  : 방법 후보 진단 — 예측 범주(S) 안 train 물체 중 V 예측(개체 형상 [cy, w, h, area])이 정답 형상과 가장 가까운 물체의 캡션
#                          (test 캡션은 은행에 없음). retr_rand_*: 같은 범주 train 캡션 무작위 = 대조 (뇌가 범주 말고도 고르는지)
#  진단: 고른 train 물체와 test 물체의 정답 형상 거리 (V 고름 vs 무작위) — V가 정보를 주면 앞이 작아야 함
#  색 뺀 판 = "a {범주}, {모양}" (모양에 남은 색 단어도 지움), 색 넣은 판 = "a {범주}, {모양}, {색}, {재질}"
# 사용: python caption_prompts.py --cap <captions.jsonl> --comp <test components.npz (vt 포함)> --vt <view_targets_shape.npz> --train_list <core_train_list> --out <폴더>
import os, re, json, argparse, numpy as np
NAME = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor',
        '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone',
        '04530566': 'boat'}   # caption_vlm.py · text_prompts.py와 같음

p = argparse.ArgumentParser()
p.add_argument('--cap', required=True); p.add_argument('--comp', required=True); p.add_argument('--vt', required=True)
p.add_argument('--train_list', required=True); p.add_argument('--out', required=True); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.RandomState(a.seed)
COLORS = r'\b(red|green|blue|yellow|orange|purple|pink|magenta|violet|brown|black|white|gr[ae]y|silver|gold(en)?|beige|tan|cream|dark|light|bright|pale|colou?red|wooden|metallic|glossy|transparent)\b'
cap = {}
for l in open(a.cap, encoding='utf-8'):
    d = json.loads(l)
    if d.get('shape'): cap[d['uid']] = d
art = lambda w: ('an ' if w[0] in 'aeiou' else 'a ') + w
clean = lambda t: re.sub(r'\s+', ' ', re.sub(r'\s+,', ',', re.sub(COLORS, '', t, flags=re.I))).strip(' ,')


def text(d, color):
    parts = [art(NAME[d['syn']]), d['shape'] if color else clean(d['shape'])] + ([d.get('color', ''), d.get('material', '')] if color else [])
    return ', '.join(x for x in parts if x)


zv = np.load(a.vt); GT = {str(k).split('/')[-1]: zv['shape'][i].mean(0) for i, k in enumerate(zv['keys']) if zv['shape_ok'][i]}
train = [l.strip() for l in open(a.train_list) if l.strip()]
bank = {}
for k in train:
    syn, uid = k.split('/')
    if uid in cap and uid in GT: bank.setdefault(syn, []).append(uid)
z = np.load(a.comp, allow_pickle=True); classes = [str(c) for c in z['classes']]
assert 'vt' in z.files, 'components.npz에 vt 없음 → 11-10 이후 infer_prism.py로 다시 추론'
O = {n: {} for n in ('oracle_color', 'oracle_nocolor', 'retr_color', 'retr_nocolor', 'retr_rand_color', 'retr_rand_nocolor')}
det, dv, dr, hit, nofb = [], [], [], 0, 0
for name, cp, vt in zip(z['name'], z['cls_pred'], z['vt']):
    syn_t, uid = str(name).split('/'); syn_p = classes[int(cp)]; hit += syn_p == syn_t
    if uid in cap: O['oracle_color'][uid] = text(cap[uid], 1); O['oracle_nocolor'][uid] = text(cap[uid], 0)
    else: nofb += 1; O['oracle_color'][uid] = O['oracle_nocolor'][uid] = art(NAME[syn_t])   # 캡션 실패 test 물체 → 범주 문장 (104개 유지, 개수 출력)
    cand = bank.get(syn_p, [])
    if not cand:
        for n in ('retr', 'retr_rand'): O[f'{n}_color'][uid] = O[f'{n}_nocolor'][uid] = art(NAME[syn_p])
        continue
    d = np.array([np.linalg.norm(GT[c] - vt) for c in cand]); j = cand[int(d.argmin())]; r = cand[rng.randint(len(cand))]
    for n, u in (('retr', j), ('retr_rand', r)):
        O[f'{n}_color'][uid] = text(cap[u], 1); O[f'{n}_nocolor'][uid] = text(cap[u], 0)
    if uid in GT: dv.append(np.linalg.norm(GT[j] - GT[uid])); dr.append(np.mean([np.linalg.norm(GT[c] - GT[uid]) for c in cand]))   # 무작위 = 후보 전체 평균 (기댓값)
    det.append({'uid': uid, 'syn_true': syn_t, 'syn_pred': syn_p, 'retr': j, 'retr_dist_pred': float(d.min()), 'rand': r})
for n, P in O.items(): json.dump(P, open(f'{a.out}/{n}.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
json.dump(det, open(f'{a.out}/retr_detail.json', 'w'), indent=1)
print(f'[prompt] 캡션 {len(cap)} · 은행 {sum(map(len, bank.values()))} (범주 {len(bank)}) · test {len(z["name"])} · 예측 범주 맞음 {hit} · 캡션 없어 범주 문장으로 둔 test {nofb}', flush=True)
print(f'[prompt] 정답 형상 거리 (고른 train ↔ test, 같은 예측 범주 안): V로 고름 {np.mean(dv):.3f} vs 무작위 {np.mean(dr):.3f} (n {len(dv)}), V가 정보를 주면 앞이 작음', flush=True)
for n, P in O.items(): print(f'  {n} {len(P)}개 | 예: {next(iter(P.values()), "")[:150]}', flush=True)
print('PROMPT_DONE', a.out, flush=True)
