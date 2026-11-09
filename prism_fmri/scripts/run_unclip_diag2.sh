#!/bin/bash
# A안 어두움 진단 2 (11-10, 5090 GPU 1, 정성 14물체): unCLIP 상한 이미지 전처리 바꾸기 (정답 없이 쓸 수 있는 것만)
#  uc_white : rembg로 물체만 → 흰 배경 합성 → 자극과 같은 문턱값 경로(--rembg 0)  (회색 배경 · 바닥 그림자 제거)
#  uc_lvl   : uc_white + 물체 영역 평균 밝기를 125(자극 프레임 0 전경 평균 근처, B 보정 측정)로 맞춤 (배율, 255에서 자름)
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_unclip; D=$O/diag; I=$O/infer/sub-0001_keep; Q=$PR/scripts/qual14_uids.txt
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run
(cd $MG && source $MG/mg_guard.sh >/dev/null && python - <<PYEOF
import os, numpy as np, rembg
from PIL import Image
D='$D'; O='$O'; Q=[l.strip() for l in open('$Q') if l.strip()]; ses = rembg.new_session('u2net')
for u in Q:
    im = Image.open(f'{O}/img_gt/{u}/0.png').convert('RGB'); a = np.asarray(rembg.remove(im, session=ses))[..., 3:4].astype(np.float32) / 255
    x = np.asarray(im).astype(np.float32)
    w = x * a + 255 * (1 - a); fg = a[..., 0] > 0.5
    l = np.clip(x * (125 / max(x[fg].mean(), 1)), 0, 255) * a + 255 * (1 - a)
    for v, y in (('uc_white', w), ('uc_lvl', l)):
        os.makedirs(f'{D}/frames_{v}/{u}', exist_ok=True); Image.fromarray(y.astype(np.uint8)).save(f'{D}/frames_{v}/{u}/0.png')
    print(u, '전경 평균 %.1f' % x[fg].mean(), flush=True)
PYEOF
)
for v in uc_white uc_lvl; do
  trel dino --frames $D/frames_$v --out $D/tok_$v --rembg 0 | grep -E "DINO_DONE|Error"
  [ $(cat $D/quick/$v/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] || trel gscheck --pred $I/tokens --slat_pred $D/tok_$v --out $D/quick/$v --names $Q | grep -E "GSCHECK_DONE|Error"
  echo "  $v 색: $($PY $PR/scripts/gs_summary.py $D/quick/$v/gscheck.jsonl)"
done
echo DIAG2_DONE
