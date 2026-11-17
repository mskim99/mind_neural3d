#!/bin/bash
# 11-17 공용 서버 배려 감시 (읽기만 함): GPU 프로세스마다 SM 사용률 · 메모리를 "우리(jhj) / 다른 사람"으로 나눠 보여 주고, CPU 부하 · 우리 CPU 몫
#  다른 사람 학습의 SM%가 우리 작업을 띄운 뒤 크게 떨어지면(예: 90 → 50) 우리 동시 실행 수(MAXJ)를 줄일 신호
#  우리 = 명령줄에 /home/oem/jhj 또는 prism · calib · blender · trellis가 있는 프로세스
# 사용: watch -n 1 bash /home/oem/jhj/MinD-3D++/mind_neural3d/prism_fmri/scripts/polite_watch.sh
ours() { tr '\0' ' ' 2>/dev/null < /proc/$1/cmdline | grep -qE '/home/oem/jhj|prism|calib|blender|trellis|run_branch|run_yaw'; }
echo "지금 $(TZ=Asia/Seoul date +%H:%M:%S) KST | CPU 부하(1·5·15분) $(cut -d' ' -f1-3 /proc/loadavg) / $(nproc)코어 | 우리 CPU 합 $(ps -u oem -o pid=,pcpu= | while read p c; do ours $p && echo $c; done | awk '{s+=$1} END{printf "%.0f%%", s}')"
echo; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | awk -F', ' '{printf "GPU %s 전체 %s · %s / %s\n", $1, $2, $3, $4}'
echo; printf "%-4s %-8s %-5s %-6s %-9s %s\n" GPU PID 누구 SM% 메모리 명령
declare -A MEM; while IFS=', ' read pid m; do MEM[$pid]=$m; done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
nvidia-smi pmon -c 1 -s u 2>/dev/null | awk '!/^#/ && $2 ~ /^[0-9]+$/ {print $1, $2, $4}' | while read g p sm; do
  w=다른; ours $p && w=우리
  printf "%-4s %-8s %-5s %-6s %-9s %s\n" $g $p $w "$sm" "${MEM[$p]:-?}MiB" "$(tr '\0' ' ' 2>/dev/null < /proc/$p/cmdline | cut -c1-70)"
done
