#!/bin/bash
# V-4c (11-25 사용자 확인 18:44, Pro6000 GPU 1, 학습 없음): 외형 지표 악화(PSNR −1.15)가 생성 경로 탓인가, 서버 · 패키지 탓인가
#  ① orig = 기존 Ours(7명 모델) 피험자 1 · 6을 Pro6000에서 원래 경로로 다시 생성 (trellis_tools gen --pred 토큰 --slat_text, seed 42) → 방향 보정 · 8지표
#     5090 값 PSNR 14.65 · 14.39와 비교. 입력이 5090과 같은지 지문으로 먼저 확인 (토큰 cat md5 · 문장 정렬 json md5)
#  ② match = (①의 평균 PSNR이 5090과 0.3 안쪽일 때만, 읽는 법 첫 줄) --ss_z 경로에서 무작위 수 순서 · 정밀도를 --pred와 맞춘 0 step:
#     trellis_tools ss (파이프라인 fp16 형상 단계, CPU noise, seed 42, bs 1, z fp32 저장) → gen --ss_z --ss_rng 1
#     FORCE_MATCH=1이면 ①과 상관없이 ②도 함
#  토큰 · 문장은 V-4에서 만든 prism_v4/infer_test/sub-00{01,06}_keep (같은 7명 모델 b50dee0d). 결과 $W/prism_v4c, 묶음 $W/prism_v4c/v4c_pack.tar
# 사용: GPU=1 nohup bash run_v4c.sh > $W/logs/v4c.log 2>&1 < /dev/null & disown   (진행: watch -n 1 bash status_v4c.sh)
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
V=$W/prism_v4; O=$W/prism_v4c; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}; PAR=${PAR:-4}; FER=$W/fair_eval_ref; mkdir -p $O
QUAL="17072ff9ec47bfadc3fd2392bc923b2f 1b77f2fbf17f28e726cd2499b0c05f5a 336fcd767dadec9fb611871e114769d9 4dc9f86110ea40aaa570c6c691c987a8 5695a98b66b2b9a9c81ddeca50aa3117 6e91124a4ecaa1a365564d9c2ed3b493 969f830a531682cef155d75bbf62b80 a51410ca51a40c2529412f6cb23454eb b5d0ae4f723bce81f119374ee5d5f944"
mgpy() { (cd $MG && source $MG/mg_guard.sh > /dev/null && export HF_HUB_OFFLINE=1 && python "$@"); }
echo "=== V-4c 시작 $(date '+%m-%d %H:%M') | 코드 $(cd $PR && git log --oneline -1 | cut -c1-50) | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음"}')"
[ "$(find $FER/gt_pc -type f | wc -l) $(find $FER/gt_render -type f | wc -l)" = "326 624" ] || { echo "평가 번들 정답 개수가 다름 — 멈춤"; exit 1; }
# 입력 지문 (5090 기존 Ours: s01 토큰 a1278d9a 문장 3a0aa64d / s06 토큰 63c8088c 문장 aa3b556a)
declare -A ET=([01]=a1278d9a [06]=63c8088c) EP=([01]=3a0aa64d [06]=aa3b556a)
for s in 01 06; do d=$V/infer_test/sub-00${s}_keep
  t=$(cat $d/tokens/*.npy | md5sum | cut -c1-8); p=$(python3 -c "import json,hashlib,sys;d=json.load(open(sys.argv[1]));print(hashlib.md5(json.dumps(d,sort_keys=True).encode()).hexdigest()[:8])" $d/prompt_text.json)
  echo "FP|s$s|토큰 $(ls $d/tokens | wc -l)개 $t (5090 ${ET[$s]}) | 문장 $p (5090 ${EP[$s]}) | $([ $t = ${ET[$s]} ] && [ $p = ${EP[$s]} ] && echo 같음 || echo 다름)"
done
# 환경 지문 (5090: TRELLIS 442aa1e, image-large 25e0d31f, text-base f8e8cf00, torch 2.7.1+cu128, spconv-cu126 2.3.8, xformers 0.0.31.post1, kaolin 0.18.0, venv_freeze md5 97d3923f)
echo "ENV|TRELLIS $(cd $MG/repo/third_party/TRELLIS && git log --oneline -1 | cut -c1-7) | 스냅샷 $(for m in image-large text-base; do ls $MG/cache/hf/hub/models--microsoft--TRELLIS-$m/snapshots 2>/dev/null | cut -c1-8; done | tr '\n' ' ')| freeze $(md5sum < $MG/venv_freeze.txt 2>/dev/null | cut -c1-8)"
echo "ENV|$(cd $MG && grep -iE '^(torch|spconv|xformers|kaolin|transformers|trimesh|open3d|numpy|pillow|xatlas|pymeshfix|pyvista|utils3d)' venv_freeze.txt 2>/dev/null | tr '\n' ' ')"
echo "ENV|GPU $(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | grep "^$CUDA_VISIBLE_DEVICES" | tr -d '\n')"

glb() { local g=$1 d=$O/$1; shift; local n=$(ls $V/infer_test/sub-00${g##*_s}_keep/tokens | wc -l)
  [ $(ls $d/glb/*/*.glb 2>/dev/null | wc -l) -ge $n ] || { wait_gpu_free "GLB $g"; mgpy $PR/trellis_tools.py gen "$@" --slat_text $V/infer_test/sub-00${g##*_s}_keep/prompt_text.json --text_model $TM --out $d/glb 2>&1 | grep -aE "GEN_DONE|Error|없음|빈 형상"; }
  ls $d/glb/*/*.glb 2>/dev/null | wc -l > $d/NEED; echo "GLB|v4c_$g|$(cat $d/NEED)/$n|$(date +%H:%M:%S)"; }
evals() { local ES=""; for g in "$@"; do ES="$ES v4c_$g:$O/$g/glb:$V/infer_test/sub-00${g##*_s}_keep/tokens:full:v2"; done
  SETS="$ES" PAR=$PAR WAIT=1 FE2=$FER FE_RUN=run_eval.sh GPU=$CUDA_VISIBLE_DEVICES bash $SD/run_yaw_v2.sh >> $W/logs/v4c_yaw.log 2>&1; }
psnr() { grep -E '^R2\|PSNR' $FER/out/yaw_v4c_$1/values.txt 2>/dev/null | awk '{print $2}'; }

# ---- ① 원래 경로로 다시 생성 (2갈래)
evals orig_s01 orig_s06 & EV=$!
J=""; for s in 01 06; do mkdir -p $O/orig_s$s; glb orig_s$s --pred $V/infer_test/sub-00${s}_keep/tokens & J="$J $!"; sleep 5; done; wait $J
echo "=== ① GLB 끝 $(date +%H:%M:%S)"; wait $EV
P1=$(psnr orig_s01); P6=$(psnr orig_s06); D=$(python3 -c "print(round(($P1+$P6)/2-(14.65+14.39)/2,3))" 2>/dev/null)
echo "ORIG|PSNR s01 $P1 (5090 14.65) · s06 $P6 (5090 14.39) | 평균 차 $D"

# ---- ② 맞춘 0 step (첫 줄일 때만)
if [ "${FORCE_MATCH:-0}" = 1 ] || python3 -c "import sys;sys.exit(0 if abs($D)<=0.3 else 1)" 2>/dev/null; then
  echo "=== ② 맞춘 0 step 시작 $(date +%H:%M:%S)"
  for s in 01 06; do wait_gpu_free "V-4c 형상 s$s"
    mgpy $PR/trellis_tools.py ss --pred $V/infer_test/sub-00${s}_keep/tokens --out $O/match_s$s --bs 1 --ss_noise cpu --z32 1 --seed 42 2>&1 | grep -aE "SS_DONE|Error|Traceback"; done
  evals match_s01 match_s06 & EV=$!
  J=""; for s in 01 06; do glb match_s$s --ss_z $O/match_s$s/z --ss_rng 1 & J="$J $!"; sleep 5; done; wait $J
  echo "=== ② GLB 끝 $(date +%H:%M:%S)"; wait $EV
  echo "MATCH|PSNR s01 $(psnr match_s01) · s06 $(psnr match_s06)"
else echo "=== ② 건너뜀 (평균 차 $D, 0.3 밖 — 읽는 법 첫 줄 아님)"; fi

# ---- 요약 · 묶음
echo "=== 요약"; grep -aE "^YV2\|v4c_" $W/logs/v4c_yaw.log
F=""; for g in orig_s01 orig_s06 match_s01 match_s06; do R=$FER/out/yaw_v4c_$g; [ -d $R ] || continue
  for f in values.txt frame.txt logs/frame.log dist_perobj.csv tex_R2.json; do [ -f $R/$f ] && F="$F ${R#$W/}/$f"; done
  for u in $QUAL; do [ -d $R/render_R2/$u ] && F="$F ${R#$W/}/render_R2/$u"; done; done
(cd $W && tar cf $O/v4c_pack.tar $F logs/v4c.log logs/v4c_yaw.log) && echo "PACK $O/v4c_pack.tar $(tar tf $O/v4c_pack.tar | wc -l)개 · $(du -h $O/v4c_pack.tar | cut -f1)"
echo "=== V4C_DONE $(date +%H:%M:%S)"
