# PRISM 추론 1단계 (MinD-3D++ venv, 실행 위치 $W/run): test fMRI → TRELLIS 조건 토큰 [1374, 1024] (fp16) + 성분 벡터
#  --perceptive keep|mean : keep = 그대로(보정), mean = 배치 평균으로 바꿔 끼움(사실상 버리기)
#  출력 <out>/<sub>_<perceptive>/tokens/<uid>.npy, <out>/<sub>_<perceptive>/components.npz (sem·var·perc 벡터, 범주·피험자 예측, 범주 점수 cls_score [N, 13] · classes)
#  --fold k/n (11-09): 목록(annotations 순서)에서 i % n == k인 물체만 (train_prism.py --fold k/n이 뺀 물체 = 안 본 물체)
#  --names <uid 목록>: 그 물체만
#  --tok_pool 1: components.npz에 예측 토큰 요약 tok_pool [N, 54, 1024] fp16 (token_util.pool, 토큰 prior 조건), --save_tokens 0이면 토큰 파일은 안 씀
#  2단계(GLB)는 meshgen venv의 trellis_tools.py gen
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
from token_util import pool
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--split', default='test')
p.add_argument('--perceptive', default='keep', choices=['keep', 'mean']); p.add_argument('--bs', type=int, default=8)
p.add_argument('--fold', default=''); p.add_argument('--tok_pool', type=int, default=0); p.add_argument('--save_tokens', type=int, default=1)
p.add_argument('--names', default='')   # uid 목록 파일: 이 물체만 (fold 거른 뒤)
p.add_argument('--fmri_sub', default='')   # AP · APAC (MinD-3D++ Tab.3, "model trained on Subject 1"): 학습에 없던 피험자 fMRI(예: sub-0009)를 --subs 하나(예: sub-0001)의 갈래로 추론
p.add_argument('--list', default='')       # 물체 목록 파일 (기본 annotations/core_<split>_list.txt, APAC는 annotations/apac_list.txt)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()   # 피험자별 체크포인트는 enc_paths에서 ViT, 옛 체크포인트(11-02 첫 학습)는 Temporal Stem 없음
print('load ok | per_subject', bool((ck.get('args') or {}).get('per_subject', 0)), '| subs', SUBS, flush=True)
ids = [l.strip() for l in open(a.list or os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
assert not a.fmri_sub or (a.subs and ',' not in a.subs), '--fmri_sub는 --subs에 갈래 피험자 하나'
if a.fold: fk, fn = map(int, a.fold.split('/')); ids = [k for i, k in enumerate(ids) if i % fn == fk]
if a.names: _w = {l.split()[-1] for l in open(a.names) if l.strip()}; ids = [k for k in ids if k.split('/')[1] in _w]
for sub in (a.subs.split(',') if a.subs else SUBS):
    dsub = a.fmri_sub or sub                                    # fMRI를 읽는 피험자 폴더
    od = f'{a.out}/{dsub}_{a.perceptive}' if not a.fmri_sub else f'{a.out}/{dsub}_as{sub[-2:]}_{a.perceptive}'; os.makedirs(f'{od}/tokens', exist_ok=True); si = SUBS.index(sub)
    comp = {k: [] for k in ('name', 'sem', 'var', 'perc', 'cls_pred', 'cls_score', 'cls', 'subj_pred', 'tok_pool', 'img')}
    for i in range(0, len(ids), a.bs):
        ks = ids[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, dsub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, fi, torch.full((len(ks),), si, dtype=torch.long, device='cuda'), perceptive_override=None if a.perceptive == 'keep' else 'mean')
        if a.tok_pool: comp['tok_pool'].append(pool(out['cond_tokens'].float()).half().cpu())
        for j, k in enumerate(ks if a.save_tokens else []):
            np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
        comp['name'] += ks; comp['cls'] += [SYN.index(k.split('/')[0]) if k.split('/')[0] in SYN else -1 for k in ks]   # APAC: 학습에 없는 범주 = −1
        comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu())
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); comp['cls_score'].append(sc)   # CLIP 텍스트 프로토타입과 cos
        comp['cls_pred'].append(sc.argmax(-1)); comp['subj_pred'].append(out['subj_logit'].argmax(-1).cpu())
        if 'sem_img' in out: comp['img'].append(out['sem_img'].float().cpu())             # 11-10 A안: 예측 이미지 임베딩 (정규화 전)
    np.savez(f'{od}/components.npz', name=np.array(comp['name']), cls=np.array(comp['cls']), classes=np.array(SYN),
             **{k: torch.cat(comp[k]).numpy() for k in ('sem', 'var', 'perc', 'cls_pred', 'cls_score', 'subj_pred', 'tok_pool', 'img') if comp[k]})
    cl = np.array(comp['cls']); kn = cl >= 0
    acc = float((torch.cat(comp['cls_pred']).numpy()[kn] == cl[kn]).mean()) if kn.any() else float('nan')
    print(f'{dsub} ({sub} 갈래) {a.perceptive}: {len(ids)} 물체 → {od}, 범주 정확도 {acc:.3f} (13범주 안 {int(kn.sum())}개)', flush=True)
print('INFER_DONE', flush=True)
