#!/bin/bash
# TRELLIS 텍스트판 외형 빠른 확인 (11-09, Pro6000 GPU 1, 추론만): 형상(SS) = fig 예측 토큰, 외형(SLat) = 텍스트판이 "a {예측 범주}"로 생성
#  정성 14물체(qual14) × sub-0001 → Gaussian 렌더 → text_quick.tar (TEXT_QUICK_DONE)
#  비교: 예측 그대로(fig) · 정답 외형 = 이미 있는 $W/prism_slat/quick/{gt} 와 새로 만드는 pred
#  TEXT_MODEL로 바꿀 수 있음 (기본 microsoft/TRELLIS-text-base)
# 사용: nohup bash run_text_quick.sh > $W/logs/prism_text_quick.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; I=$W/prism_renorm/fig/infer/sub-0001_keep; O=$W/prism_text; Q=$PR/scripts/qual14_uids.txt; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; mkdir -p $O; [ -f $I/components.npz ] || { echo "fig sub-0001 예측 없음 $I"; exit 1; }

step "1 문장 (예측 범주)"
$PY $PR/text_prompts.py --comp $I/components.npz --out $O/prompt_s1.json

step "2 빠른 확인 (14물체)"
run() {   # 이름 · trellis_tools 인자
  v=$1; shift; q=$O/quick/$v; [ $(cat $q/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && return
  [ -d $q ] && mkdir -p $O/quick_bad && mv $q $O/quick_bad/${v}_$(date +%s)   # 끊긴 결과는 옆으로 치움 (rm 안 씀)
  trel gscheck --pred $I/tokens --out $q --names $Q "$@" | grep -E "텍스트판|GSCHECK_DONE|Error|error|없음"
}
run pred
run text_s1 --slat_text $O/prompt_s1.json --text_model $TM
[ -d $W/prism_slat/quick/gt ] && ln -sfn $W/prism_slat/quick/gt $O/quick/gt
for d in $O/quick/*/; do echo "  $(basename $d) 색: $($PY $PR/scripts/gs_summary.py ${d}gscheck.jsonl)"; done
cd $O && tar chf text_quick.tar prompt_s1.json quick/*/*.png quick/*/gscheck.jsonl && echo "PACK $(tar tf text_quick.tar | grep -c png)개 (예상 42)"
step "TEXT_QUICK_DONE"
