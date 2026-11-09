#!/bin/bash
# A안 어두움 진단 (11-10, 5090 GPU 1, 학습 없음, 정성 14물체): 외형(SLat)이 어두운 원인 나누기
#  gt_rembg : 자극 프레임 0을 rembg 경로로 → 어두우면 rembg 경로 탓
#  uc_g07 · uc_g05 : unCLIP 상한 이미지(정답 임베딩, 문장 없음 · nl 0 · cfg 10)에 감마 0.7 · 0.5 (밝게) → 밝아지면 이미지 단계 보정으로 해결
#  비교: prism_unclip/quick/unclip_gt (밝기 80.4), gt (정답 토큰, 173.6)
GPU=${GPU:-1} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; O=$W/prism_unclip; D=$O/diag; I=$O/infer/sub-0001_keep; Q=$PR/scripts/qual14_uids.txt
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $D
$PY - <<PYEOF
import os, numpy as np
from PIL import Image
D='$D'; Q=[l.strip() for l in open('$Q') if l.strip()]
for u in Q:
    for v, src, g in (('gt_rembg', '$TGT/frames/%s/0.png' % u, 1.0), ('uc_g07', '$O/img_gt/%s/0.png' % u, 0.7), ('uc_g05', '$O/img_gt/%s/0.png' % u, 0.5)):
        os.makedirs(f'{D}/frames_{v}/{u}', exist_ok=True)
        x = np.asarray(Image.open(src).convert('RGB')).astype(np.float32) / 255
        Image.fromarray((np.clip(x ** g, 0, 1) * 255).astype(np.uint8)).save(f'{D}/frames_{v}/{u}/0.png')
print('frames ok')
PYEOF
for v in gt_rembg uc_g07 uc_g05; do
  trel dino --frames $D/frames_$v --out $D/tok_$v --rembg 1 | grep -E "DINO_DONE|Error"
  [ $(cat $D/quick/$v/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] || trel gscheck --pred $I/tokens --slat_pred $D/tok_$v --out $D/quick/$v --names $Q | grep -E "GSCHECK_DONE|Error"
  echo "  $v 색: $($PY $PR/scripts/gs_summary.py $D/quick/$v/gscheck.jsonl)"
done
echo DIAG_DONE
