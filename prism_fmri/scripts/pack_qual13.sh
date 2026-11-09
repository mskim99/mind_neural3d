#!/bin/bash
# 보고 정성 표(8차 중간보고 p.5~6, 13물체)의 Ours 칸용 렌더 묶기 (11-09): 평가 결과 이름마다 13물체 × 뷰 0 · 2 · 4
#  물체 목록 = 8차 pack_qual_renders.sh와 같음 (GT · MinD-3D++ 칸은 그대로 두고 Ours만 바꿈)
# 사용: bash pack_qual13.sh text_s1 [다른 이름 ...] → $W/prism_text/qual13_<이름들>.tar
source $(dirname $0)/common.sh || exit 1
R=$W/spp_eval/out; O=$W/prism_text/qual13
UIDS="68b4a03cd8d49553e98e65cfc071e8a0 ba1adc474cfa2f429ab926a7e944e795 2d638c6b6b2feb9248da169d95204ce2 bf9df04aea08b0f0f51f77a6d7299806 6e91124a4ecaa1a365564d9c2ed3b493 1b77f2fbf17f28e726cd2499b0c05f5a 5695a98b66b2b9a9c81ddeca50aa3117 336fcd767dadec9fb611871e114769d9 b5d0ae4f723bce81f119374ee5d5f944 4dc9f86110ea40aaa570c6c691c987a8 a51410ca51a40c2529412f6cb23454eb 17072ff9ec47bfadc3fd2392bc923b2f 969f830a531682cef155d75bbf62b80"
[ $# -ge 1 ] || { echo "이름을 주세요 (예: text_s1)"; exit 1; }
for n in "$@"; do
  for u in $UIDS; do
    f=$(find $R/$n/render_a_L16000 -maxdepth 3 -type d -name "${u}*" | head -1)
    [ -z "$f" ] && { echo "없음 $n $u"; continue; }
    mkdir -p $O/$n/$u; cp $f/{0,2,4}.png $O/$n/$u/
  done
done
T=qual13_$(echo "$@" | tr ' ' '_').tar
cd $W/prism_text && tar cf $T $(for n in "$@"; do echo qual13/$n; done) && echo "QUAL13_DONE $T $(tar tf $T | grep -c png)개 (예상 $((39 * $#)))"
