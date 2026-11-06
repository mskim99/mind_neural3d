# PRISM 추론 1단계 (MinD-3D++ venv, 실행 위치 $W/run): test fMRI → TRELLIS 조건 토큰 [1374, 1024] (fp16) + 성분 벡터
#  --perceptive keep|mean : keep = 그대로(보정), mean = 배치 평균으로 바꿔 끼움(사실상 버리기)
#  출력 <out>/<sub>_<perceptive>/tokens/<uid>.npy, <out>/<sub>_<perceptive>/components.npz (sem·var·perc 벡터, 범주·피험자 예측, 범주 점수 cls_score [N, 13] · classes)
#  2단계(GLB)는 meshgen venv의 trellis_tools.py gen
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--split', default='test')
p.add_argument('--perceptive', default='keep', choices=['keep', 'mean']); p.add_argument('--bs', type=int, default=8)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()   # 피험자별 체크포인트는 enc_paths에서 ViT, 옛 체크포인트(11-02 첫 학습)는 Temporal Stem 없음
print('load ok | per_subject', bool((ck.get('args') or {}).get('per_subject', 0)), '| subs', SUBS, flush=True)
ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
for sub in (a.subs.split(',') if a.subs else SUBS):
    od = f'{a.out}/{sub}_{a.perceptive}'; os.makedirs(f'{od}/tokens', exist_ok=True); si = SUBS.index(sub)
    comp = {k: [] for k in ('name', 'sem', 'var', 'perc', 'cls_pred', 'cls_score', 'cls', 'subj_pred')}
    for i in range(0, len(ids), a.bs):
        ks = ids[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, fi, torch.full((len(ks),), si, dtype=torch.long, device='cuda'), perceptive_override=None if a.perceptive == 'keep' else 'mean')
        for j, k in enumerate(ks):
            np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
        comp['name'] += ks; comp['cls'] += [SYN.index(k.split('/')[0]) for k in ks]
        comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu())
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); comp['cls_score'].append(sc)   # CLIP 텍스트 프로토타입과 cos
        comp['cls_pred'].append(sc.argmax(-1)); comp['subj_pred'].append(out['subj_logit'].argmax(-1).cpu())
    np.savez(f'{od}/components.npz', name=np.array(comp['name']), cls=np.array(comp['cls']), classes=np.array(SYN),
             **{k: torch.cat(comp[k]).numpy() for k in ('sem', 'var', 'perc', 'cls_pred', 'cls_score', 'subj_pred')})
    acc = float((torch.cat(comp['cls_pred']).numpy() == np.array(comp['cls'])).mean())
    print(f'{sub} {a.perceptive}: {len(ids)} tokens → {od}, test 범주 정확도 {acc:.3f}', flush=True)
print('INFER_DONE', flush=True)
