#!/bin/bash
# 공정평가 잠정 기준의 형태 지표만 (FPD · CD · EMD, 렌더 없음) — 번들 run_temp.sh · run_shape_4090.sh와 같은 계산 (dist.py P2 · fpd.py · collect.py 그대로)
# source 해서 씀: fe_shape <이름> <예측 폴더> <glb|obj> <물체 목록 파일(syn/uid)>
#   glb = <폴더>/<uid>/<uid>.glb, obj = <폴더>/<uid>.obj. GT = $FE/gt_pc/P2/<uid>.npy (APAC 220은 apac_gt_p2.tar를 같은 폴더에 풀어 둠)
#   결과 $FE/out/<이름>/values.txt (P2|CD · P2|EMD · FPD)
# 필요한 변수: FE(번들 폴더), FE_PY, 선택 FE_WORKERS
fe_shape() {
  local N=$1 PD=$2 FMT=$3 L=$4 O=$FE/out/$1
  [ -f $O/values.txt ] && { echo "이미 있음 $N"; return; }
  mkdir -p $O/logs $O/gtpc
  awk -F/ -v d="$PD" -v f=$FMT 'NF{u=$2; sub(/\r$/,"",u); p=(f=="glb")? d"/"u"/"u".glb" : d"/"u".obj"; print u"\t"p}' $L > $O/pairs_shape.tsv
  local MISS=$(cut -f2 $O/pairs_shape.tsv | while read p; do [ -f "$p" ] || echo x; done | wc -l)
  local GMISS=$(cut -f1 $O/pairs_shape.tsv | while read u; do [ -f $FE/gt_pc/P2/$u.npy ] || echo x; done | wc -l)
  echo "[$N] 물체 $(wc -l < $O/pairs_shape.tsv) | 예측 없음 $MISS | GT 없음 $GMISS"
  [ $GMISS -eq 0 ] || { echo "[$N] GT 점구름 부족 → 멈춤"; return 1; }
  ln -sfn $FE/gt_pc/P2 $O/gtpc/P2
  (cd $FE/scripts && $FE_PY -I dist.py --pairs $O/pairs_shape.tsv --gt_pc $O/gtpc --out $O --workers ${FE_WORKERS:-6} > $O/logs/dist.log 2>&1)
  $FE_PY $FE/scripts/fpd.py --pairs $O/pairs_shape.tsv --root $FE --out $O/fpd.json > $O/logs/fpd.log 2>&1
  $FE_PY $FE/scripts/collect.py --out $O > $O/values.txt 2>&1
  echo "SHAPE_DONE $N | $(grep -E '^P2\|(CD|EMD)\s|^FPD\s' $O/values.txt | tr '\n' ' ') $(date +%H:%M:%S)"
}
