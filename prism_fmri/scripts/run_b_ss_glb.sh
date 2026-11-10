#!/bin/bash
# B안 3단계 확인 (11-10, 5090): 형상 잠재 prior 생성 z → TRELLIS 2단계(외형 = 텍스트판 "a {가장 가까운 학습 범주}") → glb + 스냅샷
#  생성 32 (T 1.0, gen_0000~0031) + 실제 검증 물체 z 8 (같은 경로 상한, 문장 = 정답 범주)
# G=<생성 폴더(check/occ_check.json 포함)> D=<출력> 로 바꿀 수 있음
# 사용: GPU=0 nohup bash run_b_ss_glb.sh > $W/logs/prism_b_ss_glb.log 2>&1 < /dev/null & disown
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_b; S=$O/ss; G=${G:-$O/ss_prior/gen_t1.0}; D=${D:-$O/ss_glb}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
step() { echo "=== $1 $(date +%H:%M:%S)"; }
mkdir -p $D/z
step "0 z · 문장 준비"
$PY - <<EOF
import json, os, numpy as np
NAME = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'monitor',
        '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone', '04530566': 'boat'}
art = lambda w: ('an ' if w[0] in 'aeiou' else 'a ') + w
R = json.load(open('$G/check/occ_check.json')); P = {}
GC = json.load(open('$G/cls.json')) if os.path.exists('$G/cls.json') else None   # 범주 조건 생성이면 문장 = 조건 범주
for k in range(32):
    n = f'gen_{k:04d}'; os.symlink('$G/' + n + '.npy', '$D/z/' + n + '.npy') if not os.path.exists('$D/z/' + n + '.npy') else None
    P[n] = art(NAME[GC[n] if GC else R['gen_nn_uid'][k].split('_')[0]])
va = [l.strip() for l in open('$O/prior_u/val_objs.txt') if l.strip() and os.path.exists('$S/z/' + l.strip() + '.npy')]
for u in [va[i] for i in sorted(np.random.RandomState(0).choice(len(va), 8, replace=False))]:
    n = 'real_' + u; os.symlink('$S/z/' + u + '.npy', '$D/z/' + n + '.npy') if not os.path.exists('$D/z/' + n + '.npy') else None
    P[n] = art(NAME[u.split('_')[0]])
json.dump(P, open('$D/prompt.json', 'w'), indent=1); print('z', len(P), '| 예', list(P.items())[:3], list(P.items())[-2:])
EOF
step "1 glb (형상 = z, 외형 = 텍스트판)"
trel gen --ss_z $D/z --slat_text $D/prompt.json --out $D/glb | grep -E "\[gen|GEN_DONE|Error|텍스트판"
step "2 모음 그림"
$PY - <<EOF
import json, glob, os
from PIL import Image, ImageDraw
P = json.load(open('$D/prompt.json')); ns = sorted(P, key=lambda n: (n.startswith('real'), n)); W, H, C = 768, 384, 4
cv = Image.new('RGB', (W * C, (H + 24) * ((len(ns) + C - 1) // C)), 'white'); d = ImageDraw.Draw(cv)
for i, n in enumerate(ns):
    f = f'$D/glb/{n}/{n}_snap.png'
    if not os.path.exists(f): continue
    r, c = divmod(i, C); cv.paste(Image.open(f).convert('RGB'), (c * W, r * (H + 24) + 24))
    d.text((c * W + 6, r * (H + 24) + 4), f"{'REAL ' if n.startswith('real') else ''}{n[-8:] if n.startswith('real') else n} | {P[n]}", fill='green' if n.startswith('real') else 'blue')
cv.save('$D/ss_glb_montage.jpg', quality=85); print('MONTAGE $D/ss_glb_montage.jpg')
EOF
step "B_SS_GLB_DONE"
