#!/bin/bash
# 형태 지표만 (CD · EMD · FPD) — 지금 평가 기준 run_eval.sh와 같은 계산 (glb 좌표축 +x,+y,+z 고정, dist.py · fpd.py · collect.py), 물체 목록을 따로 받음 (AP 104 · APAC 220)
#  11-15: fe_shape.sh는 예전 번들 인자(--frame 없음)라 지금 번들에서 값이 비었음 → 이걸로 대신
# 사용: bash shape_eval_list.sh <이름> <GLB 폴더(uid/uid.glb)> <목록(syn/uid)> [기다릴 개수]
N=$1; PD=$2; L=$3; WAITN=${4:-0}
R=/home/oem/jhj/fair_eval/fair_eval; PY=/home/oem/jhj/fair_eval/venv/bin/python; O=$R/out/$N; F="+x,+y,+z"
[ $WAITN -gt 0 ] && until [ $(ls $PD/*/*.glb 2>/dev/null | wc -l) -ge $WAITN ]; do sleep 60; done
mkdir -p $O/logs
awk -F/ -v d="$PD" 'NF{u=$2; sub(/\r$/,"",u); print u"\t"d"/"u"/"u".glb"}' $L > $O/pairs.tsv
echo "[$N] 물체 $(wc -l < $O/pairs.tsv) | 예측 없음 $(cut -f2 $O/pairs.tsv | while read p; do [ -f "$p" ] || echo x; done | wc -l) | GT 없음 $(cut -f1 $O/pairs.tsv | while read u; do [ -f $R/gt_pc/P2/$u.npy ] || echo x; done | wc -l)"
echo "$F" > $O/frame.txt
(cd $R/scripts && $PY -I dist.py --pairs $O/pairs.tsv --root $R --frame="$F" --out $O --workers 8 > $O/logs/dist.log 2>&1)
$PY $R/scripts/fpd.py --pairs $O/pairs.tsv --root $R --frame="$F" --out $O/fpd.json > $O/logs/fpd.log 2>&1
$PY $R/scripts/collect.py --out $O > $O/values.txt
echo "SHAPE_DONE $N | $(grep -E '^(CD|EMD|FPD)	' $O/values.txt | tr '\n' ' ') $(date +%H:%M:%S)"
