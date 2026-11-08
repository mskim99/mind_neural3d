# components.npz 여러 개를 하나로 합침 (11-09, prior 조정용: fold 0 · 1의 안 본 예측을 한 파일로). 사용: python merge_comp.py <출력> <입력1> <입력2> ...
import sys, numpy as np
zs = [np.load(f) for f in sys.argv[2:]]
keys = [k for k in zs[0].files if all(k in z.files for z in zs)]
out = {k: (zs[0][k] if k == 'classes' else np.concatenate([z[k] for z in zs])) for k in keys}
np.savez(sys.argv[1], **out); print('MERGE', len(out['name']), '개 →', sys.argv[1], flush=True)
