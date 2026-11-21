#!/bin/bash
# P-2 (11-20 총괄 인계, 사용자 확인): 영역 가중이 처음 보는 피험자(4번) · AP(9번) · APAC(11번)에서 도움이 되나 — Pro6000 GPU 1
#  학습 3개 → 추론 14세트 → 텍스트판 GLB → 방향 보정 → 기준 번들(5090 fair_eval과 같은 파일, 좌표축 형식별 고정) 8지표 · 정성 렌더, 학습이 끝나는 대로 이어서
#  모델 ($W/prism_p2/<이름>, 공통: train 1,300개 전부(HOLD=0) · 2만 스텝 · 시드 1 · bs 14)
#    ㉮ ausvp_sharedsv          최종 Ours(ausvp) + S · V 피험자 공통 (FiLM · 피험자 분류 끔), 7명 1 2 3 5 6 7 8
#    ㉯ ausvp_sharedsv_regionw  ㉮ + 공통 영역 가중 (사전값 $W/prism_calib/region_prior.npz = 지난번과 같은 7명 train 물체, 초기 0.3 ~ 0.7)
#    ㉰ ausvp_8sub              최종 Ours(ausvp) 설정 그대로, 4번 포함 8명
#  세트 이름 p2_<모델>_<조건>  (조건 문법 — 세트를 더할 때는 ADD="모델:조건 …")
#    s0K             학습한 피험자 K 그대로 (infer_prism.py)
#    s04_nopair      4번 fMRI를 짝 없이: feat_norm 통계만 4번 자기 train fMRI로(라벨 · 짝 없음), 영역 가중은 공통만 (피험자별 조정 0)
#    AP_nopair       9번 test 104개, 통계는 그 104개 자기 fMRI로 · APAC220_nopair  11번 220개 (apac_sub0011_list.txt), 같은 방식
#    s04_pair<N>as<AA>   4번 train 앞 N개 짝으로 기준 피험자 AA에 선형 사상(align_infer.py 기본 = ViT 토큰 전체 · 프레임 평균, 5090 2088a68과 같음), 영역 가중은 공통만
#  평가: run_yaw_v2.sh (방향 보정 = GT 없이 yawsel) → $W/fair_eval_ref/run_eval.sh (축 고정, 세트마다 고르지 않음). AP는 full(8지표 중 형태 3지표를 씀 + 정성 렌더),
#        APAC는 shape(GT 렌더 없음) + 정성 9물체를 같은 카메라로 따로 렌더. 번들 GT 내용이 5090 지문과 다르면 평가는 띄우지 않음 (학습 · 추론 · GLB는 계속)
#  GPU 1에 다른 사람 프로세스(명령줄에 /home/oem/jhj가 없는 것)가 있으면 단계마다 없어질 때까지 기다림 (사용자 11-17)
# 사용: GPU=1 nohup bash run_p2.sh > $W/logs/p2.log 2>&1 < /dev/null & disown        진행: watch -n 1 bash status_p2.sh
#       STAGE=check bash run_p2.sh  (번들 지문 · 데이터 · GPU 확인만, 아무것도 안 띄움)
#       STAGE=post  … (학습을 건너뛰고 추론부터 — 코드를 고친 뒤 이어 갈 때, 이미 있는 결과는 건너뜀) · STAGE=train (학습만)
#       세트 더하기(예: 짝 100 · 300개, 사용자 확인 뒤): ADD="ausvp_sharedsv_regionw:s04_pair100as01 …" STAGE=post …
SD=$(cd "$(dirname "$0")" && pwd)   # 11-21: 가드가 작업 위치를 바꾸므로 source 전에 절대 경로로 (상대 경로로 불려도 PR · SD가 맞게)
source $SD/common.sh || exit 1
PY=$W/venv/bin/python; P=$W/prism_p2; G=$CUDA_VISIBLE_DEVICES; STAGE=${STAGE:-all}
M1=ausvp_sharedsv; M2=ausvp_sharedsv_regionw; M3=ausvp_8sub
MAXJ=${MAXJ:-3}; GENPAR=${GENPAR:-2}; PAR=${PAR:-4}; TM=${TEXT_MODEL:-microsoft/TRELLIS-text-base}
FER=$W/fair_eval_ref; FERA=$W/fair_eval_ref_apac; V2=$W/fair_eval_v2; RP=$W/prism_calib/region_prior.npz
QUAL="17072ff9ec47bfadc3fd2392bc923b2f 1b77f2fbf17f28e726cd2499b0c05f5a 336fcd767dadec9fb611871e114769d9 4dc9f86110ea40aaa570c6c691c987a8 5695a98b66b2b9a9c81ddeca50aa3117 6e91124a4ecaa1a365564d9c2ed3b493 969f830a531682cef155d75bbf62b80 a51410ca51a40c2529412f6cb23454eb b5d0ae4f723bce81f119374ee5d5f944"   # 정성 9물체 (이전 세트와 같음)
SETS0="$M1:s04_nopair $M2:s04_nopair $M1:s04_pair1300as01 $M2:s04_pair1300as01 $M2:s04_pair1300as02 $M3:s04 \
$M1:AP_nopair $M2:AP_nopair $M1:APAC220_nopair $M2:APAC220_nopair $M1:s01 $M1:s06 $M2:s01 $M2:s06"   # A 6 · B 4 · C 4 = 14
ALL="$SETS0 ${ADD:-}"
mkdir -p $P $W/logs; cd $W/run

# ---- GPU 1에 다른 사람 작업이 있으면 기다림
others() { others_on_gpu; }; wait_free() { wait_gpu_free "$@"; }   # common.sh (세트마다도 부름: infer_one · glb_one · run_yaw_v2 · run_branch)

# ---- 기준 번들: 5090 fair_eval과 같은 파일인지 지문으로 확인한 뒤 링크 (계산 코드는 안 고침)
fp() { find -L $1 -type f -exec md5sum {} + | awk '{print $1}' | sort | md5sum | cut -c1-12; }
ref_ok() {
  local bad=""
  [ -f $FER/run_eval.sh ] || { echo "기준 번들 없음 $FER (fair_eval_ref.tar 풀기)"; return 1; }
  [ "$(md5sum < $FER/run_eval.sh | cut -c1-32)" = 3700cf4ce057605b9d6cf3b5ebad2a78 ] || bad="$bad run_eval.sh"
  for x in collect:3b83a40d diag_frame:a1df3f2d dist:e2b150d2 fpd:4511ce5c frame_util:d78352 official_pc:bc21ec render:4e8610 tex:e5052d; do
    [[ "$(md5sum < $FER/scripts/${x%%:*}.py)" == ${x#*:}* ]] || bad="$bad ${x%%:*}.py"; done
  [ "$(find -L $V2/gt_pc_y -type f | wc -l) $(fp $V2/gt_pc_y)" = "326 9e0d0e7d0c55" ] || bad="$bad gt_pc($(fp $V2/gt_pc_y))"
  [ "$(find -L $V2/gt_render -type f | wc -l) $(fp $V2/gt_render)" = "624 23dc218ebb43" ] || bad="$bad gt_render($(fp $V2/gt_render))"
  [ "$(md5sum < $V2/fpd/pointnet.py | cut -c1-32)" = 31d543b9cded8180d95f63a1dd0946f6 ] || bad="$bad fpd/pointnet.py"
  [ "$(md5sum < $V2/fpd/pointnet_fpd.pth | cut -c1-32)" = 3ae69b4911ad47869ad90e9b61a0341a ] || bad="$bad fpd/pointnet_fpd.pth"
  [[ "$(md5sum < $V2/core_test_list.txt)" == 159ff9e8* ]] || bad="$bad core_test_list"
  [ -z "$bad" ] || { echo "기준 번들 지문이 5090과 다름:$bad → 평가 안 띄움"; return 1; }
  ln -sfn $(readlink -f $V2/gt_pc_y) $FER/gt_pc; ln -sfn $(readlink -f $V2/gt_render) $FER/gt_render; ln -sfn $(readlink -f $V2/fpd) $FER/fpd; cp $V2/core_test_list.txt $FER/; mkdir -p $FER/out
  local m=$(while IFS=/ read s u; do u=${u%$'\r'}; [ -f $V2/gt_pc_y/P2/$u.npy ] || echo x; done < $DR/annotations/apac_sub0011_list.txt | wc -l)
  if [ $m = 0 ]; then   # APAC 옆 번들: 같은 스크립트 · 같은 GT 폴더, 목록만 sub-0011 실제 220개
    mkdir -p $FERA/out; ln -sfn $FER/scripts $FERA/scripts; ln -sfn $(readlink -f $V2/fpd) $FERA/fpd; ln -sfn $(readlink -f $V2/gt_pc_y) $FERA/gt_pc
    cp $DR/annotations/apac_sub0011_list.txt $FERA/core_test_list.txt; cp $FER/run_eval.sh $FERA/
  else echo "APAC GT 점구름 $m개 없음 → APAC 평가 안 띄움"; fi
  echo "기준 번들 OK: 스크립트 9개 · gt_pc 326 · gt_render 624 · fpd 2 · test 목록이 5090과 같음 → $FER (APAC $FERA)"; return 0
}

# ---- 세트 → 폴더 · 필요 개수 · 평가 방식
need() { case $1 in APAC220*) echo 220;; *) echo 104;; esac; }
setdir() { local m=$1 c=$2
  case $c in s0[0-9]) echo $P/infer/$m/sub-00${c#s}_keep;; s04_nopair) echo $P/infer/$m/sub-0004_new_keep;; AP_nopair) echo $P/infer/$m/sub-0009_new_keep;;
    APAC220_nopair) echo $P/infer/$m/sub-0011_new_keep;; s04_pair*as*) local n=${c#s04_pair}; echo $P/align/$m/as${n#*as}_n${n%as*}/n${n%as*}/sub-0004_as${n#*as}_keep;; *) echo BAD;; esac; }
evmode() { case $1 in APAC220*) echo shape:apac;; *) echo full:v2;; esac; }
rwc() { case $1 in *_regionw) echo "--rw_common 1";; esac; }

infer_one() {   # 모델 조건 → 토큰 · components.npz, 범주 정확도 CAT 줄
  local m=$1 c=$2 d=$(setdir $1 $2) k=$(need $2) CK=$P/$1/model_final.pt L=$W/logs/p2_infer_${1}_$2.log
  [ -f $d/components.npz ] && [ $(ls $d/tokens/*.npy 2>/dev/null | wc -l) -ge $k ] && return 0
  [ -f $CK ] || { echo "체크포인트 없음 $CK — 건너뜀 p2_${m}_$c"; return 1; }
  wait_free "추론 p2_${m}_$c"
  local IP="$PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --out $P/infer/$m"
  case $c in
    s0[0-9]) $IP --subs sub-00${c#s} > $L 2>&1;;
    s04_nopair) $IP --subs sub-0001 --fmri_sub sub-0004 --own_stats train $(rwc $m) > $L 2>&1;;
    AP_nopair) $IP --subs sub-0001 --fmri_sub sub-0009 --own_stats list $(rwc $m) > $L 2>&1;;
    APAC220_nopair) $IP --subs sub-0001 --fmri_sub sub-0011 --own_stats list --list $DR/annotations/apac_sub0011_list.txt $(rwc $m) > $L 2>&1;;
    s04_pair*as*) local n=${c#s04_pair}; local N=${n%as*} A=${n#*as}
      $PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --subs sub-00$A --fmri_subs sub-0004 --ns $N --save $N $(rwc $m) --out $P/align/$m/as${A}_n$N > $L 2>&1;;
  esac
  echo "CAT|p2_${m}_$c|$(grep -aoE '범주 정확도 [0-9.na]+|test 범주 [0-9.]+|own_stats.*sd [0-9.]+|고른 λ [^|]*' $L | tr '\n' ' ')|$(date +%H:%M:%S)"
  [ -f $d/components.npz ] || { echo "추론 실패 p2_${m}_$c ($L)"; grep -aE 'Error|Traceback' $L | tail -2; return 1; }
}
trel() { (cd $MG && source $MG/mg_guard.sh >/dev/null && export HF_HUB_OFFLINE=1 && python $PR/trellis_tools.py "$@"); }
gcnt() { ls $1/*/*.glb 2>/dev/null | wc -l; }
glb_one() { local m=$1 c=$2 d=$(setdir $1 $2) k=$(need $2) t0=$(date +%s)
  [ -f $d/components.npz ] || return 1; [ $(gcnt $d/glb_text) -ge $k ] || wait_free "GLB p2_${m}_$c"
  [ -f $d/prompt_text.json ] || $PY $PR/text_prompts.py --comp $d/components.npz --out $d/prompt_text.json > /dev/null
  [ $(gcnt $d/glb_text) -ge $k ] || trel gen --pred $d/tokens --slat_text $d/prompt_text.json --text_model $TM --out $d/glb_text 2>&1 | grep -aE "GEN_DONE|Error|없음"
  echo "GLB|p2_${m}_$c|$(gcnt $d/glb_text)/$k개|$(( $(date +%s) - t0 ))s|$(date +%H:%M:%S)"; }
lane() { local i=0; for s in $ALL; do [ $(( i % GENPAR )) = $1 ] && glb_one ${s%%:*} ${s#*:}; i=$(( i + 1 )); done; }

# ---- 0. 확인
echo "=== P-2 시작 $(date '+%m-%d %H:%M') | STAGE=$STAGE | 세트 $(echo $ALL | wc -w)개 | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
for s in sub-0001 sub-0002 sub-0003 sub-0004 sub-0005 sub-0006 sub-0007 sub-0008 sub-0009 sub-0011; do echo -n "$s $(find -L $DR/$s/npy_data -name '*.npy' | wc -l) · "; done; echo
for s in $ALL; do [ "$(setdir ${s%%:*} ${s#*:})" = BAD ] && { echo "모르는 조건 $s"; exit 1; }; done
[ -f $RP ] || { echo "영역 사전값 없음 $RP"; exit 1; }
for f in $PR/infer_prism.py $PR/align_infer.py $PR/text_prompts.py $PR/trellis_tools.py $SD/run_branch.sh $SD/run_yaw_v2.sh; do [ -f $f ] || { echo "코드 없음 $f (PR=$PR SD=$SD) — 멈춤"; exit 1; }; done; echo "코드 위치 PR=$PR"
REF=1; ref_ok || REF=0
o=$(others); echo "GPU $G 다른 사람 작업: ${o:-없음}"
[ $STAGE = check ] && exit 0

# ---- 0-1. 새 추론 경로 짧은 동작 확인 (학습 전, 몇 분): 지난 2명 모델 ausvp_sharedsv_regionw로 4번 8물체 — 짝 없이(자기 통계 · 공통 가중) / 짝 50개
SC=$W/prism_2s/ausvp_sharedsv_regionw/model_final.pt; SM=$P/smoke
if [ $STAGE = all ] && [ -f $SC ] && [ ! -f $SM/OK ]; then
  wait_free 동작확인; mkdir -p $SM; head -8 $DR/annotations/core_test_list.txt > $SM/names.txt
  $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $SC --out $SM --subs sub-0001 --fmri_sub sub-0004 --own_stats train --rw_common 1 --names $SM/names.txt > $SM/infer.log 2>&1
  $PY $PR/align_infer.py --repo $W/MinD-3D --data_root $DR --ckpt $SC --subs sub-0001 --fmri_subs sub-0004 --ns 50 --save 50 --rw_common 1 --lams 1 --out $SM/align > $SM/align.log 2>&1
  if [ $(ls $SM/sub-0004_new_keep/tokens/*.npy 2>/dev/null | wc -l) = 8 ] && [ -f $SM/align/n50/sub-0004_as01_keep/components.npz ]; then touch $SM/OK; echo "SMOKE OK (새 추론 경로 2가지) $(date +%H:%M:%S)"
  else echo "SMOKE 실패 — 학습은 계속, 추론 전에 코드 수정 필요 ($SM/infer.log · align.log)"; grep -aE 'Error|Traceback' $SM/infer.log $SM/align.log | tail -3; fi
fi

# ---- 1. 학습 (3개 동시, run_branch.sh의 gate: 동시 MAXJ개 · GPU 남은 메모리 · CPU 부하)
if [ $STAGE = all ] || [ $STAGE = train ]; then
  wait_free 학습
  RUNS="$M1 $M2 $M3" OUT=$P LP=prism_p2 HOLD=0 SAVE=10000 MAXJ=$MAXJ RP=$RP bash $SD/run_branch.sh
  for m in $M1 $M2; do [ -f $P/$m/model_final.pt ] && echo "TRAIN|$m|$(grep -aE '^\[val\]' $W/logs/prism_p2_$m.log | tail -1 | cut -c1-160)"; done
  [ -f $P/$M3/model_final.pt ] && echo "TRAIN|$M3|$(grep -aE '^\[val\]' $W/logs/prism_p2_$M3.log | tail -1 | cut -c1-160)"
  ls $P/$M2/region_w/ 2>/dev/null | tr '\n' ' '; echo
fi
[ $STAGE = train ] && exit 0

# ---- 2. 추론 (차례로, 범주 정확도 CAT 줄)
wait_free 추론
for s in $ALL; do infer_one ${s%%:*} ${s#*:}; done
echo "=== 추론 끝 $(date +%H:%M:%S)"

# ---- 3. GLB (GENPAR줄) + 4. 방향 보정 · 평가 (GLB가 차는 대로, PAR개씩)
wait_free GLB · 평가
EV=""
if [ $REF = 1 ]; then
  ES=""; for s in $ALL; do m=${s%%:*} c=${s#*:}; d=$(setdir $m $c); [ -f $d/components.npz ] || continue; ES="$ES p2_${m}_$c:$d/glb_text:$d/tokens:$(evmode $c | cut -d: -f1):$(evmode $c | cut -d: -f2)"; done
  SETS="$ES" PAR=$PAR WAIT=1 FE2=$FER FEA=$FERA FE_RUN=run_eval.sh GPU=$G bash $SD/run_yaw_v2.sh > $W/logs/p2_yaw.log 2>&1 & EV=$!
fi
J=""; for g in $(seq 0 $(( GENPAR - 1 ))); do lane $g & J="$J $!"; sleep 5; done; wait $J
echo "=== GLB 끝 $(date +%H:%M:%S)"
[ -n "$EV" ] && wait $EV
[ $REF = 1 ] || { echo "=== 평가 안 함 (기준 번들 확인 실패) — GLB까지 끝"; echo "=== P2_DONE $(date +%H:%M:%S)"; exit 0; }

# ---- 5. 정성: APAC는 GT 렌더가 없어 shape만 → 방향 보정한 GLB 9개를 같은 카메라(R2)로 따로 렌더
for s in $ALL; do m=${s%%:*} c=${s#*:}; case $c in APAC220*) ;; *) continue;; esac
  n=p2_${m}_$c; Q=$W/prism_yaw/$n/qual_R2; [ -d $W/prism_yaw/$n/glb ] || continue
  awk 'NR % 24 == 1' $FERA/core_test_list.txt | head -9 | while IFS=/ read cl u; do u=${u%$'\r'}; ls $W/prism_yaw/$n/glb/$u/$u.glb; done > $W/prism_yaw/$n/qual_list.txt
  $W/tools/blender-4.2.3-linux-x64/blender -b -t 4 -P $FER/scripts/render.py -- --list $W/prism_yaw/$n/qual_list.txt --out_root $Q --res 1024 --light 8000 \
    --camera_dist 1.5 --fov 56 --views 0:30,60:30,120:30,180:30,240:30,300:30 --metallic0 2>&1 | grep -E "FAIL" | head -3
done

# ---- 6. 요약 · 렌더 개수 검사 · 묶음 (받을 것 하나: $P/p2_pack.tar)
echo "=== 요약"; grep -aE '^CAT\||^GLB\|' $W/logs/p2.log 2>/dev/null | sort -u
grep -aE "^YV2\|p2_" $W/logs/p2_yaw.log
F=""; for s in $ALL; do m=${s%%:*} c=${s#*:}; n=p2_${m}_$c; R=$FER; [ "$(evmode $c)" = shape:apac ] && R=$FERA; O=$R/out/yaw_$n
  if [ "$(evmode $c)" = full:v2 ]; then r=$(find $O/render_R2 -name '*.png' 2>/dev/null | wc -l); echo "RENDER|$n|$r/$(( $(need $c) * 6 ))$([ $r -lt $(( $(need $c) * 6 )) ] && echo ' 부족')"; fi
  for f in values.txt frame.txt logs/frame.log; do [ -f $O/$f ] && F="$F ${O#$W/}/$f"; done
  for u in $QUAL; do [ -d $O/render_R2/$u ] && F="$F ${O#$W/}/render_R2/$u"; done
  [ -d $W/prism_yaw/$n/qual_R2 ] && F="$F prism_yaw/$n/qual_R2 prism_yaw/$n/qual_list.txt"
  d=$(setdir $m $c); for f in components.npz prompt_text.json; do [ -f $d/$f ] && F="$F ${d#$W/}/$f"; done
done
for m in $M1 $M2 $M3; do for f in train_log.csv region_w; do [ -e $P/$m/$f ] && F="$F prism_p2/$m/$f"; done; done
F="$F $(cd $W && ls -d prism_p2/align/*/*/align.json logs/p2.log logs/p2_yaw.log logs/p2_infer_*.log logs/prism_p2_*.log 2>/dev/null | tr '\n' ' ')"
(cd $W && tar cf $P/p2_pack.tar $F) && echo "PACK $P/p2_pack.tar $(tar tf $P/p2_pack.tar | wc -l)개 · $(du -h $P/p2_pack.tar | cut -f1)"
echo "=== P2_DONE $(date +%H:%M:%S)"
