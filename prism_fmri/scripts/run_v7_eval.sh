#!/bin/bash
# V-7 평가 (11-26 사용자 확인 12:32, Pro6000 GPU 1): 실험 ausvp_vtgt_bandenergy_s1 · 대조 ausvp_s1 (피험자 1번, test 104)
#  ① 두 모델 학습 끝(model_final.pt)을 기다림 → run_branch_subs.sh(추론 → 문장 → 텍스트판 GLB → 방향 보정 · 8지표, 기준 번들 fair_eval_ref) 둘 동시
#  ② 범주 안 고르기(CD · 실루엣) · 생성 렌더 특징 = v6_ident.py (V-6 ① ②와 같은 스크립트, CPU)
#  ③ 묶음 $W/prism_v7/v7_pack.tar: 값 · 물체별 거리 · 외형 · 정성 9물체 렌더 · components.npz(head 예측 vt) · ident 출력 · 학습 로그 끝부분
#  head test 상관 · 생성 렌더 고주파와 자극의 상관은 로컬에서 (v5_feat.npz · v7_bandenergy.npz와 비교)
# 사용: GPU=1 nohup bash run_v7_eval.sh > $W/logs/v7_eval.log 2>&1 < /dev/null & disown     진행: watch -n 1 bash status_v7.sh
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
O=$W/prism_v7; FER=$W/fair_eval_ref; N1=ausvp_vtgt_bandenergy_s1; N0=ausvp_s1; G=$CUDA_VISIBLE_DEVICES
QUAL="17072ff9ec47bfadc3fd2392bc923b2f 1b77f2fbf17f28e726cd2499b0c05f5a 336fcd767dadec9fb611871e114769d9 4dc9f86110ea40aaa570c6c691c987a8 5695a98b66b2b9a9c81ddeca50aa3117 6e91124a4ecaa1a365564d9c2ed3b493 969f830a531682cef155d75bbf62b80 a51410ca51a40c2529412f6cb23454eb b5d0ae4f723bce81f119374ee5d5f944"
echo "=== V-7 평가 시작 $(date '+%m-%d %H:%M') | 코드 $(cd $PR && git log --oneline -1 | cut -c1-50) | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
[ "$(find $FER/gt_pc -type f | wc -l) $(find $FER/gt_render -type f | wc -l)" = "326 624" ] || { echo "평가 번들 정답 개수가 다름 — 멈춤"; exit 1; }
for n in $N1 $N0; do until [ -f $O/$n/model_final.pt ]; do sleep 60; done; echo "학습 끝 확인 $n $(date +%H:%M:%S)"; done
J=""; for n in $N1 $N0; do
  CK=$O/$n/model_final.pt NAME=$n SS=1 GENPAR=1 PAR=${PAR:-3} FE2=$FER FE_RUN=run_eval.sh GPU=$G bash $SD/run_branch_subs.sh > $W/logs/v7_eval_$n.log 2>&1 & J="$J $!"; sleep 5; done
wait $J; for n in $N1 $N0; do grep -aE "^GLB\||^YV2\||BRANCH_SUBS_DONE|실패" $W/logs/v7_eval_$n.log; done
echo "=== 8지표 끝 $(date +%H:%M:%S)"
I=$W/prism_branch/subs
nice -n 10 $W/venv/bin/python -I $PR/v6_ident.py --fe $FER --m3 $W --out $O/ident --workers 6 \
  --sets "exp_s01:yaw_br_${N1}_s1:$I/$N1/sub-0001_keep/prompt_text.json,ctl_s01:yaw_br_${N0}_s1:$I/$N0/sub-0001_keep/prompt_text.json" > $W/logs/v7_ident.log 2>&1
grep -aE "IDENT\||Error|Traceback|V6_IDENT_DONE" $W/logs/v7_ident.log
F=""; for n in $N1 $N0; do R=$FER/out/yaw_br_${n}_s1; d=$I/$n/sub-0001_keep
  for f in values.txt frame.txt dist_perobj.csv tex_R2.json; do [ -f $R/$f ] && F="$F ${R#$W/}/$f"; done
  for u in $QUAL; do [ -d $R/render_R2/$u ] && F="$F ${R#$W/}/render_R2/$u"; done
  for f in components.npz prompt_text.json; do [ -f $d/$f ] && F="$F ${d#$W/}/$f"; done
  tail -c 20000 $W/logs/prism_v7_$n.log > $O/trainlog_tail_$n.txt; F="$F prism_v7/trainlog_tail_$n.txt"; done
(cd $W && tar cf $O/v7_pack.tar $F prism_v7/ident data/shape_targets/v7_bandenergy.npz logs/v7_eval.log logs/v7_ident.log) && echo "PACK $O/v7_pack.tar $(tar tf $O/v7_pack.tar | wc -l)개 · $(du -h $O/v7_pack.tar | cut -f1)"
echo "=== V7_DONE $(date +%H:%M:%S)"
