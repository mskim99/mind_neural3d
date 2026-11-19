#!/bin/bash
# 11-19 학습 + 평가 + 자원을 한 화면에 (읽기만 함): run_branch.sh로 학습 → run_branch_subs.sh로 평가하는 체인용
#  ① 학습: status_reg.sh 표 (진행 · 속도 · 남은 시간 · 종료 KST)  ② 평가: run마다 status_branch_subs.sh 표 (GLB · 방향 렌더 · 지표)
#  ③ 자원: GPU별 사용률 · 메모리, CPU 부하, 우리 CPU 합, 다른 사람 GPU 프로세스  ④ 체인 로그 마지막 줄
# 사용: RUNS="ausvp_csr_dropS ausvp_csr_noframeemb_dropR" SS="1 6" OUT=$W/prism_2s LP=prism_2s CHAIN=$W/logs/csr2_pro6000.log watch -n 1 bash status_all.sh
W=/home/oem/jhj/MinD-3D++; D=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: ${RUNS:?RUNS="이름 …"}; SS=${SS:-"1 6"}; OUT=${OUT:-$W/prism_2s}; LP=${LP:-prism_2s}
echo "■ 학습"
SET=branch OUT=$OUT LP=$LP RUNS="$RUNS" bash $D/status_reg.sh 2>/dev/null | sed -n '1,/^요약/p' | grep -v '^요약'
echo; echo "■ 평가 (피험자 $SS)"
first=1
for n in $RUNS; do
  NAME=$n SS="$SS" bash $D/status_branch_subs.sh 2>/dev/null | grep -v '^지금' | { if [ $first = 1 ]; then cat; else grep -v '^세트'; fi; }
  first=0
done
echo; echo "■ 자원"
bash $D/polite_watch.sh 2>/dev/null | grep -v '^$'
[ -n "${CHAIN:-}" ] && [ -f "$CHAIN" ] && { echo; echo "■ 체인 로그 ($(basename $CHAIN))"; grep -avE '^\s*$' $CHAIN | tail -3 | cut -c1-150; }
true
