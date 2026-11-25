#!/bin/bash
# V-4b (11-25 사용자 확인 15:18, Pro6000 GPU 1, 학습 없음): V-4 (A)의 CD 개선 · 외형 악화가 미세조정 몫인가 생성 경로 몫인가
#  0 step = 원래 TRELLIS 형상 단계(미세조정 없음)로 V-4와 같은 경로: v4_train.py genz --cond tok (sampler 25 step · cfg 5.0 · seed 42, test 토큰 그대로)
#   → trellis_tools.py gen --ss_z (텍스트판 외형) → 방향 보정 · 8지표 (기준 번들 fair_eval_ref, 축 고정). 세트 v4b_Z0_s01 · v4b_Z0_s06
#  V-4 출력(prism_v4/infer_test)을 그대로 씀. 결과 $W/prism_v4/gen/Z0_s{01,06}, 묶음 $W/prism_v4/v4b_pack.tar, 로그 $W/logs/v4b.log · v4b_yaw.log
# 사용: GPU=1 nohup bash run_v4b.sh > $W/logs/v4b.log 2>&1 < /dev/null & disown   (진행: watch -n 1 bash status_v4b.sh)
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
V=$W/prism_v4; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; PAR=${PAR:-4}; FER=$W/fair_eval_ref
QUAL="17072ff9ec47bfadc3fd2392bc923b2f 1b77f2fbf17f28e726cd2499b0c05f5a 336fcd767dadec9fb611871e114769d9 4dc9f86110ea40aaa570c6c691c987a8 5695a98b66b2b9a9c81ddeca50aa3117 6e91124a4ecaa1a365564d9c2ed3b493 969f830a531682cef155d75bbf62b80 a51410ca51a40c2529412f6cb23454eb b5d0ae4f723bce81f119374ee5d5f944"
mgpy() { (cd $MG && source $MG/mg_guard.sh > /dev/null && export HF_HUB_OFFLINE=1 && python "$@"); }
echo "=== V-4b 시작 $(date '+%m-%d %H:%M') | 코드 $(cd $PR && git log --oneline -1 | cut -c1-50) | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
for s in 01 06; do for f in $V/infer_test/sub-00${s}_keep/tokens $V/infer_test/sub-00${s}_keep/prompt_text.json; do [ -e $f ] || { echo "없음 $f — 멈춤"; exit 1; }; done; done
[ "$(find $FER/gt_pc -type f | wc -l) $(find $FER/gt_render -type f | wc -l)" = "326 624" ] || { echo "평가 번들 정답 개수가 다름 — 멈춤"; exit 1; }

# ① 0 step 형상 잠재
for s in 01 06; do wait_gpu_free "V-4b 형상 s$s"
  mgpy $PR/v4_train.py genz --cond tok --data $V/infer_test/sub-00${s}_keep --out $V/gen/Z0_s$s/z | grep -aE "GENZ_DONE|Error|Traceback|line [0-9]+"
done

# ② GLB (2갈래) + 방향 보정 · 평가 (GLB가 나오는 대로)
SETL="Z0_s01 Z0_s06"; ES=""; for g in $SETL; do [ -d $V/gen/$g/z ] && ES="$ES v4b_$g:$V/gen/$g/glb:$V/infer_test/sub-00${g##*_s}_keep/tokens:full:v2"; done
SETS="$ES" PAR=$PAR WAIT=1 FE2=$FER FE_RUN=run_eval.sh GPU=$CUDA_VISIBLE_DEVICES bash $SD/run_yaw_v2.sh > $W/logs/v4b_yaw.log 2>&1 & EV=$!
glb_one() { local g=$1 d=$V/gen/$1; [ -d $d/z ] || return; local k=$(ls $d/z | wc -l)
  [ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge $k ] || { wait_gpu_free "GLB $g"; mgpy $PR/trellis_tools.py gen --ss_z $d/z --slat_text $V/infer_test/sub-00${g##*_s}_keep/prompt_text.json --text_model $TM --out $d/glb 2>&1 | grep -aE "GEN_DONE|Error|없음"; }
  ls $d/glb/*/*.glb 2>/dev/null | wc -l > $d/NEED; echo "GLB|v4b_$g|$(cat $d/NEED)/$k|$(date +%H:%M:%S)"; }
J=""; for g in $SETL; do glb_one $g & J="$J $!"; sleep 5; done; wait $J
echo "=== GLB 끝 $(date +%H:%M:%S)"; wait $EV

# ③ 요약 · 묶음
echo "=== 요약"; grep -aE "^YV2\|v4b_" $W/logs/v4b_yaw.log
F=""; for g in $SETL; do O=$FER/out/yaw_v4b_$g; r=$(find $O/render_R2 -name '*.png' 2>/dev/null | wc -l); echo "RENDER|v4b_$g|$r/624$([ $r -lt 624 ] && echo ' 부족')"
  for f in values.txt frame.txt logs/frame.log; do [ -f $O/$f ] && F="$F ${O#$W/}/$f"; done
  for u in $QUAL; do [ -d $O/render_R2/$u ] && F="$F ${O#$W/}/render_R2/$u"; done; done
(cd $W && tar cf $V/v4b_pack.tar $F logs/v4b.log logs/v4b_yaw.log) && echo "PACK $V/v4b_pack.tar $(tar tf $V/v4b_pack.tar | wc -l)개 · $(du -h $V/v4b_pack.tar | cut -f1)"
echo "=== V4B_DONE $(date +%H:%M:%S)"
