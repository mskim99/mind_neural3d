#!/bin/bash
# V-6 ③ (11-25 사용자 확인 18:44, Pro6000 GPU 1, 학습 없음): 정답 메시를 단색(재질 제거, Blender 기본 회색) · 같은 조명으로 평가 카메라 6뷰에 렌더
#  카메라 = 평가 R2 (= 자극 카메라: 거리 1.5 · FOV 56 · 방위 0 ~ 300 60° 간격 · 고도 30 · Area 8000 · 1024²) — gt_render/R2와 재질만 다름
#  특징(V-5와 같은 고주파 에너지)은 로컬에서 잰다 (v6_common.py). 결과 $W/prism_v6/gray/<uid>/{0..5}.png, 묶음 $W/prism_v6/v6_gray.tar
# 사용: GPU=1 nohup bash run_v6_gray.sh > $W/logs/v6_gray.log 2>&1 < /dev/null & disown   (진행: watch -n 1 'ls /home/oem/jhj/MinD-3D++/prism_v6/gray/*/5.png 2>/dev/null | wc -l; tail -2 /home/oem/jhj/MinD-3D++/logs/v6_gray.log')
SD=$(cd "$(dirname "$0")" && pwd); source $SD/common.sh || exit 1
O=$W/prism_v6; FER=$W/fair_eval_ref; BL=${FE_BLENDER:-$W/tools/blender-4.2.3-linux-x64/blender}; mkdir -p $O/gray
echo "=== V-6 ③ 단색 정답 렌더 시작 $(date '+%m-%d %H:%M') | 코드 $(cd $PR && git log --oneline -1 | cut -c1-50)"
[ -x $BL ] || { echo "Blender 없음 $BL — 멈춤"; exit 1; }
# 정답 메시 찾기: <루트>/<범주>/<uid>/models/model_normalized.obj
k1=$(head -1 $FER/core_test_list.txt | tr -d '\r'); SN=""
for r in ${GT_MESH:-} $W/eval_gt_shape/ShapeNetCore_sub01 $W/data/shapenet_b $W/data/ShapeNetCore.v2 $W/data/shapenet $W/data/GT; do [ -f $r/$k1/models/model_normalized.obj ] && { SN=$r; break; }; done
[ -z "$SN" ] && SN=$(dirname $(dirname $(dirname $(find $W/data -maxdepth 6 -path "*${k1#*/}/models/model_normalized.obj" 2>/dev/null | head -1))) 2>/dev/null)
[ -n "$SN" ] && [ -f $SN/$k1/models/model_normalized.obj ] || { echo "정답 메시를 찾지 못함 ($k1) — 멈춤"; exit 1; }
tr -d '\r' < $FER/core_test_list.txt | awk -v r=$SN 'NF{print r"/"$1"/models/model_normalized.obj"}' > $O/gray_list.txt
echo "정답 메시 $SN | $(while read p; do [ -f $p ] && echo x; done < $O/gray_list.txt | wc -l) / $(wc -l < $O/gray_list.txt)"
wait_gpu_free "V-6 단색 렌더"
$BL -b -t 4 -P $FER/scripts/render.py -- --list $O/gray_list.txt --out_root $O/gray --res 1024 --light 8000 \
  --camera_dist 1.5 --fov 56 --views 0:30,60:30,120:30,180:30,240:30,300:30 --obj_up Y --gray --name shapenet 2>&1 | grep -aE "FAIL|DONE|Error"
echo "렌더 $(ls $O/gray/*/5.png 2>/dev/null | wc -l) / 104"
(cd $W && tar cf $O/v6_gray.tar prism_v6/gray prism_v6/gray_list.txt logs/v6_gray.log) && echo "PACK $O/v6_gray.tar $(du -h $O/v6_gray.tar | cut -f1)"
echo "=== V6_GRAY_DONE $(date +%H:%M:%S)"
