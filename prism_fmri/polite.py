# -*- coding: utf-8 -*-
# 11-17 공용 GPU 배려: 같은 GPU의 다른 사람 프로세스 SM 사용률을 every 스텝마다 재서, 목표(target %)보다 낮으면 스텝 사이 쉬는 시간을 늘리고 높으면 줄임
#  (Pro6000 GPU 1에서 1단계 하나만으로 다른 사람 학습 SM 83% → 16%가 된 뒤 넣음). 다른 사람 프로세스가 없으면 쉬지 않음
#  우리 = 명령줄에 /home/oem/jhj가 있는 프로세스. GPU 번호는 CUDA_VISIBLE_DEVICES의 첫 번호(물리 번호)
# 쓰는 법: th = Throttle(target); 매 스텝 끝에 th.step()  → th.sleep(현재 쉬는 초), th.other(마지막으로 잰 다른 사람 SM%)
import os, time, subprocess


class Throttle:
    def __init__(self, target=0, every=20, max_sleep=3.0):
        self.target = float(target); self.every = every; self.max_sleep = max_sleep; self.sleep = 0.0; self.other = -1; self.n = 0
        self.gpu = (os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0] or '0')

    def _ours(self, pid):
        try:
            return '/home/oem/jhj' in open(f'/proc/{pid}/cmdline', 'rb').read().decode(errors='ignore')
        except OSError:
            return True                                                       # 사라진 프로세스는 셈에서 뺌

    def measure(self):
        try:
            out = subprocess.run(['nvidia-smi', 'pmon', '-c', '1', '-s', 'u', '-i', self.gpu], capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return -1
        sm = [int(c[3]) for c in (l.split() for l in out.splitlines() if not l.startswith('#'))
              if len(c) > 3 and c[1].isdigit() and c[3].isdigit() and not self._ours(int(c[1]))]
        return max(sm) if sm else -1                                         # -1 = 다른 사람 없음

    def step(self):
        if self.target <= 0: return
        self.n += 1
        if self.n % self.every == 0:
            self.other = self.measure()
            if self.other < 0: self.sleep = 0.0
            elif self.other < self.target: self.sleep = min(self.max_sleep, self.sleep * 1.5 + 0.05)
            else: self.sleep = max(0.0, self.sleep * 0.7 - 0.01)
        if self.sleep > 0: time.sleep(self.sleep)

    def tag(self):
        return f'배려 {self.sleep:.2f}s/스텝 (다른 사람 SM {self.other}%, 목표 {self.target:.0f}%)' if self.target > 0 else ''
