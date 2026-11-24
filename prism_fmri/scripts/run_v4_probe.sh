#!/bin/bash
# V-4 0단계 (11-24, Pro6000 GPU 1): 학습 없는 확인 세 가지 — 결과 $W/prism_v4/probe/{axis,bench,tokcos}.json, 로그 $W/logs/v4_probe.log
#  ① axis  : 정답 64³ 점유 ↔ TRELLIS 형상 좌표 (축 바꿈 · 뒤집기 48가지 중 고름) + 형상 VAE로 정답 잠재를 만들었다 되돌린 IoU (test 104)
#  ② bench : 형상 단계(sparse structure flow, 이미지판 L) 미세조정 메모리 · 속도 — 조건 토큰만 / 토큰 + V 특징, 전체 / cross-attention만, 배치 8 · 16 · 32
#  ③ tokcos: 최종 Ours 7명 모델(prism_full/ausvp)의 피험자 1 예측 토큰 대 정답 토큰 cos, train 1,300 대 test 104 (학습 물체 토큰 과신 정도)
#  STAGE=check: 재료 확인만 (아무것도 안 띄움)
# 사용: STAGE=check GPU=1 bash run_v4_probe.sh / GPU=1 nohup bash run_v4_probe.sh > $W/logs/v4_probe.log 2>&1 < /dev/null & disown
source "$(cd "$(dirname "$0")" && pwd)/common.sh" || exit 1
STAGE=${STAGE:-all}; V=$W/prism_v4; O=$V/probe; PY=$W/venv/bin/python; CK=$W/prism_full/ausvp/model_final.pt
TR=$MG/repo/third_party/TRELLIS; HC=$MG/cache/hf/hub/models--microsoft--TRELLIS-image-large/snapshots
mgpy() { (cd $MG && source $MG/mg_guard.sh > /dev/null && export HF_HUB_OFFLINE=1 && python "$@"); }
echo "=== V-4 0단계 $(date '+%m-%d %H:%M') | STAGE=$STAGE | 디스크 $(df -h $W | tail -1 | awk '{print $4" 남음 ("$5")"}')"
echo "TRELLIS $(cd $TR && git log --oneline -1 | cut -c1-40) | train.py $([ -f $TR/train.py ] && echo 있음 || echo 없음) | flow 학습 코드 $(ls $TR/trellis/trainers/flow_matching/*.py 2>/dev/null | wc -l)개"
for f in ss_flow_img_dit_L_16l8_fp16 ss_enc_conv3d_16l8_fp16 ss_dec_conv3d_16l8_fp16; do echo "가중치 $f $(ls $HC/*/ckpts/$f.safetensors 2>/dev/null | wc -l)"; done
echo "정답 점유 voxels $(ls $TGT/voxels 2>/dev/null | wc -l)개 · 정답 토큰 dino $(ls $TGT/dino 2>/dev/null | wc -l)개 · 최종 Ours 7명 모델 $([ -f $CK ] && md5sum < $CK | cut -c1-8 || echo 없음)"
echo "GPU $CUDA_VISIBLE_DEVICES 다른 사람 작업: $(o=$(others_on_gpu); echo ${o:-없음})"
[ $STAGE = check ] && exit 0
[ $(ls $TGT/voxels 2>/dev/null | wc -l) -ge 1400 ] || { echo "정답 점유 없음 — 멈춤"; exit 1; }
mkdir -p $O
wait_gpu_free "V-4 axis"
echo "=== ① axis $(date +%H:%M:%S)"; mgpy $PR/v4_probe.py axis --tgt $TGT --list $DR/annotations/core_test_list.txt --out $O 2>&1 | grep -aE "^\[axis|^AXIS|물체|sampler|Error|Traceback|line [0-9]+"
wait_gpu_free "V-4 bench"
echo "=== ② bench $(date +%H:%M:%S)"; mgpy $PR/v4_probe.py bench --out $O --bs 8,16,32 --n 25 2>&1 | grep -aE "^BENCH|Error|Traceback|line [0-9]+"
if [ -f $CK ]; then
  for sp in train test; do
    wait_gpu_free "V-4 추론 $sp"
    echo "=== ③ 추론 피험자 1 $sp $(date +%H:%M:%S)"
    (cd $W/run && $PY $PR/infer_prism.py --repo $W/MinD-3D --data_root $DR --ckpt $CK --subs sub-0001 --split $sp --out $V/infer_$sp > $W/logs/v4_infer_$sp.log 2>&1)
    grep -aE "범주 정확도|Error" $W/logs/v4_infer_$sp.log | tail -2
  done
  mgpy $PR/v4_probe.py tokcos --tgt $TGT --pred $V/infer_train/sub-0001_keep/tokens --train $DR/annotations/core_train_list.txt --test /dev/null --out $O | grep -a TOKCOS
  mgpy $PR/v4_probe.py tokcos --tgt $TGT --pred $V/infer_test/sub-0001_keep/tokens --train /dev/null --test $DR/annotations/core_test_list.txt --out $O/te | grep -a TOKCOS
else echo "최종 Ours 7명 모델 없음 — ③ 건너뜀"; fi
(cd $W && tar cf $V/v4_probe.tar prism_v4/probe logs/v4_probe.log $(ls logs/v4_infer_*.log 2>/dev/null)) && echo "PACK $V/v4_probe.tar $(du -h $V/v4_probe.tar | cut -f1)"
echo "=== V4_PROBE_DONE $(date +%H:%M:%S)"
