#!/bin/bash
# prior 0단계 (11-09, Pro6000 GPU 1, 학습 없음 · 정성 14물체): 형상(SS) = fig 예측 토큰 그대로, 외형(SLat)만 바꿔 봄 (토큰prior_설계.md 0단계)
#  (b) pca_k64 · k256 · k1024 : 외형 = 정답 토큰의 PCA-k 복원 (tok_pca.py) → 압축 공간 prior가 색을 낼 수 있나
#  (c) sde_t04 · t06 · t08    : 외형 = 예측 SLat을 시점 t까지 노이즈 → 검색 train 토큰(slat_retrieve.py) 조건으로 되돌림 (trellis_tools.py --slat_t0)
#      sde_gt_t06             : 같은 SDEdit, 조건 = test 정답 토큰 (상한)
#  비교: 외형 = 예측(fig) · 검색(t = 1) · 정답은 prism_slat/quick · prism_renorm/fig/quick에 이미 있음
#  → prior0_quick.tar (PRIOR0_DONE)
# 사용: nohup bash run_prior0.sh > $W/logs/prism_prior0.log 2>&1 < /dev/null & disown
source $(dirname $0)/common.sh || exit 1
PY=$W/venv/bin/python; I=$W/prism_renorm/fig/infer/sub-0001_keep; O=$W/prism_prior0; RT=$W/prism_slat/slat_retr
Q=$PR/scripts/qual14_uids.txt
step() { echo "=== $1 $(date +%H:%M:%S)"; }
trel() { (cd $MG && source $MG/mg_guard.sh && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
cd $W/run; [ -d $I/tokens ] && [ -f $RT/slat_map.json ] || { echo "fig 토큰 또는 검색 결과 없음 ($I, $RT: run_renorm.sh · run_slatsplit.sh 먼저)"; exit 1; }

step "1 PCA 복원 (tok_pca.py)"
[ -f $O/pca_stats.json ] || $PY $PR/tok_pca.py --tgt $TGT --data_root $DR --out $O --names $Q | grep -E "PCA|Error"

step "2 빠른 확인 (정성 14물체)"
run() {   # 이름 · trellis_tools 인자
  v=$1; shift; q=$O/quick/$v; [ $(cat $q/gscheck.jsonl 2>/dev/null | wc -l) -ge 14 ] && return
  [ -d $q ] && mkdir -p $O/quick_bad && mv $q $O/quick_bad/${v}_$(date +%s)   # 끊긴 결과는 옆으로 치움 (rm 안 씀)
  trel gscheck --pred $I/tokens --out $q --names $Q "$@" | grep -E "기본 sampler|GSCHECK_DONE|Error|없음"
}
for k in 64 256 1024; do run pca_k$k --slat_pred $O/pca_k$k; done
for t in 0.4 0.6 0.8; do run sde_t${t/./} --slat_pred $RT --slat_t0 $t; done
run sde_gt_t06 --slat_pred $TGT/dino --slat_t0 0.6
for d in $O/quick/*/; do
  echo "  $(basename $d) 색: $($PY -c "
import json; r=[json.loads(l) for l in open('${d}gscheck.jsonl')]
print('Gaussian 밝기 %.1f 채도 %.1f | 텍스처 밝기 %.1f 채도 %.1f (n %d)' % tuple([sum(x[k][i] for x in r)/len(r) for k in ('gauss_mean_chroma','tex_mean_chroma') for i in (0,1)] + [len(r)]))")"
done
cd $O && tar cf prior0_quick.tar quick/*/*.png quick/*/gscheck.jsonl pca_stats.json && echo "PACK $(tar tf prior0_quick.tar | grep -c png)개 (예상 98)"
step "PRIOR0_DONE"
