#!/bin/bash
# A안 밝기 보정 (11-10, 5090 GPU 0, 학습 없음): 정답 프레임 임베딩 → unCLIP 이미지의 문장 · noise_level · cfg를 바꿔 14물체
#  이미지 단계 숫자(전경 밝기 · 배경 흰 정도)로 고르고 → 상위 설정만 TRELLIS 외형(SLat)까지 (gscheck)
GPU=${GPU:-0} source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; UPY=$W/venv_unclip/bin/python; O=$W/prism_unclip; I=$O/infer/sub-0001_keep; Q=$PR/scripts/qual14_uids.txt
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; export HF_HUB_OFFLINE=1
declare -A PRM=([p0]="" [pw]="a 3D render of an object on a plain white background, bright studio lighting" [pp]="a product photo, white background")
for p in p0 pw pp; do for nl in 0 100; do for cfg in 10 5; do
  v=b_${p}_n${nl}_c${cfg}; [ -f $O/bright/$v/done ] && continue
  $UPY $PR/unclip_tools.py gen --frames $TGT/frames --names $Q --out $O/bright/$v --prompt "${PRM[$p]}" --noise_level $nl --cfg $cfg | grep -E "Error|DONE" && touch $O/bright/$v/done
done; done; done
$UPY - <<PYEOF
import os, glob, numpy as np
from PIL import Image
O='$O/bright'; rows=[]
for d in sorted(glob.glob(O+'/b_*')):
    br, bg, ch = [], [], []
    for f in glob.glob(d+'/*/0.png'):
        x = np.asarray(Image.open(f).convert('RGB').resize((256, 256))).astype(np.float32)
        b = np.concatenate([x[:16].reshape(-1,3), x[-16:].reshape(-1,3), x[:, :16].reshape(-1,3), x[:, -16:].reshape(-1,3)])   # 가장자리 = 배경
        bgm = b.mean(0); m = np.abs(x - bgm).sum(2) > 40          # 배경과 다른 곳 = 전경
        px = x[m] if m.sum() > 50 else x.reshape(-1, 3)
        br.append(px.mean()); bg.append(b.mean()); ch.append((px.max(1) - px.min(1)).mean())
    rows.append((os.path.basename(d), np.mean(br), np.mean(bg), np.mean(ch)))
for r in sorted(rows, key=lambda r: -r[2]): print('%-14s 전경 밝기 %.1f | 배경 %.1f | 전경 채도 %.1f' % r, flush=True)
PYEOF
echo BRIGHT_IMG_DONE
