#!/bin/bash
# V-8 분석 (11-28, 5090 CPU, 학습 없음): 평가가 끝난 세트마다 렌더 기준 진단
#  ① 전경 밝기 · 채도 (v8_bright.py)  ② 렌더 색 코드와 자극 색 코드의 일치 (color_agree.py)
#  ③ 생성 렌더 고주파 등 특징의 범주 안 상관 + 범주 안 고르기 (v6_ident.py --sets, V-7과 같은 스크립트)
#  기준 ㉠ = yaw_fullausvp_s01 (기존 Ours 피험자 1번)
# 사용: SETS="gt_detail gt_color …" bash run_v8_an.sh > log
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/common.sh" || exit 1
PY=$W/venv/bin/python; FPY=/home/oem/jhj/fair_eval/venv/bin/python; V=/home/oem/jhj/fair_eval/fair_eval; O=$W/prism_v8/an; mkdir -p $O
P0=$W/prism_full/eval_ausvp/text/s01/prompt.json
OK=base; ID="base:yaw_fullausvp_s01:$P0"; CA="base=$V/out/yaw_fullausvp_s01/render_R2"; BR=yaw_fullausvp_s01
for n in ${SETS:?}; do
  E=yaw_v8g_$n; grep -q "^CD" $V/out/$E/values.txt 2>/dev/null || { echo "평가 아직: $n"; continue; }
  pj=$W/prism_v8/prompts/$n.json; [ -f $pj ] || pj=$P0          # 이미지 조건 세트는 범주 판정에 기준 문장
  ID="$ID,$n:$E:$pj"; CA="$CA $n=$V/out/$E/render_R2"; BR="$BR,$E"
done
echo "=== 밝기"; $PY $PR/v8_bright.py --fe $V --sets $BR --out $O/bright.json
echo "=== 색 일치"; $PY $PR/scripts/color_agree.py --npz $W/data/stim_color.npz --ids $V/core_test_list.txt --train_ids $DR/annotations/core_train_list.txt --json $O/color_agree.json $CA
echo "=== 범주 안 고르기 · 렌더 특징"; $PY -I $PR/v6_ident.py --fe $V --m3 $W --out $O/ident --workers 6 --sets "$ID" | grep -E "IDENT|Error|Traceback"
echo "=== V8_AN_DONE $(date +%H:%M:%S)"
