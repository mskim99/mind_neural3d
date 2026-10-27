# InstantMesh 영상 자세 카메라 보정용 프레임 뽑기 (로컬). fMRI-Objaverse test 앞 N개 영상에서 0·32·…·352번(30°마다) 12장 → 320²
#  짝수 번호(0,64,…,320 = 60°마다)는 InstantMesh 입력(학습 GT와 같은 프레임), 홀수 번호(32,96,…,352)는 판정용(입력에 안 넣는 각도)
# 출력: <out>/<해시>/{00..11}.png  — 사용: python -I prep_calib_frames.py --out <폴더> [--n 30]
import os, zipfile, argparse, tempfile
import numpy as np, cv2
p = argparse.ArgumentParser()
p.add_argument('--data', default=r'C:\Users\jinsw712\Desktop\Files\EEG\DataSet\MinD-3D++\fMRI-Objaverse')
p.add_argument('--out', required=True)
p.add_argument('--n', type=int, default=30)
a = p.parse_args()
FR = list(range(0, 384, 32))
ids = [l.strip().split(',')[1].split('/')[-1].replace('.glb', '') for l in open(os.path.join(a.data, 'test_list.txt')) if l.strip()][1:]
z = zipfile.ZipFile(os.path.join(a.data, 'stimuli.zip'))
names = {os.path.basename(x)[:-4]: x for x in z.namelist() if x.endswith('.mp4')}
done = 0
for k in ids:
    if done >= a.n: break
    if k not in names: print('no video', k); continue
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as t:
        t.write(z.read(names[k])); tmp = t.name
    cap = cv2.VideoCapture(tmp); got = {}; i = 0
    while i <= FR[-1]:
        ok, f = cap.read()
        if not ok: break
        if i in FR: got[i] = f
        i += 1
    cap.release(); os.remove(tmp)
    if len(got) != len(FR): print('short', k, len(got)); continue
    od = os.path.join(a.out, k); os.makedirs(od, exist_ok=True)
    for j, fi in enumerate(FR):
        cv2.imwrite(os.path.join(od, f'{j:02d}.png'), cv2.resize(got[fi], (320, 320), interpolation=cv2.INTER_AREA))
    done += 1
print('PREP_DONE', done)
