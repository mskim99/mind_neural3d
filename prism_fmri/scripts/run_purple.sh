#!/bin/bash
# 보라 치우침 출처 확인 (11-08, Pro6000 GPU 1, 추론만). 색 실험 1(run_renorm.sh)의 fig sub-0001 예측 토큰을 씀
#  1 purple_check.py: 평균 토큰 M · LN(M) · 예측 평균 P̄ · LN(M + 1.37(P̄ − M)) + 공통 치우침 뺀 예측 LN(M + 1.37(P − P̄))
#  2 Gaussian 렌더: 단일 토큰 4개 + 공통 치우침 뺀 예측 14물체 → purple.tar (PURPLE_DONE)
# 사용: nohup bash run_purple.sh > $W/logs/prism_purple.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; T=$W/prism_renorm/fig/infer/sub-0001_keep/tokens; O=$W/prism_renorm/fig/purple
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -d $T ] || { echo "토큰 없음 $T (run_renorm.sh 먼저)"; exit 1; }
step "1 토큰 (purple_check.py)"
$PY $PR/purple_check.py --pred $T --tgt $TGT --data_root $DR --out $O | grep -E "train|PURPLE|Error"
step "2 Gaussian 렌더"
trel gscheck --pred $O/single --out $O/quick/single | grep -E "gscheck|DONE|Error"
trel gscheck --pred $O/debias --out $O/quick/debias --names $PR/scripts/qual14_uids.txt | grep -E "DONE|Error|없음"
cd $O && tar cf purple.tar quick/*/*.png quick/*/gscheck.jsonl purple_stats.json && echo "PACK $(ls quick/*/*.png | wc -l)개 (예상 18)"
step "PURPLE_DONE"
