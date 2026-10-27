# PRISM v2 추론 1단계 (MinD-3D++ venv, 실행 위치 $W/run): test fMRI → TRELLIS 조건 토큰 [1374, 1024] (fp16) + 성분 벡터
#  --perceptive keep|mean : keep = 그대로(보정), mean = 배치 평균으로 바꿔 끼움(사실상 버리기)
#  출력 <out>/<sub>_<perceptive>/tokens/<uid>.npy, <out>/<sub>_<perceptive>/components.npz (sem·var·perc 벡터, 범주·피험자 예측)
#  2단계(GLB)는 meshgen venv의 trellis_tools.py gen
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from src.encoder.autoencoder2 import fMRI_Encoder
from prism_v2 import PrismV2
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--split', default='test')
p.add_argument('--perceptive', default='keep', choices=['keep', 'mean']); p.add_argument('--bs', type=int, default=8)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
cfg = OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml'))
model = PrismV2(fMRI_Encoder(cfg.model.params.fmri_encoder_config), n_classes=len(SYN), n_subjects=len(SUBS))
print('load', model.load_state_dict(ck['model'], strict=True), flush=True)
model = model.cuda().eval()
ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
for sub in (a.subs.split(',') if a.subs else SUBS):
    od = f'{a.out}/{sub}_{a.perceptive}'; os.makedirs(f'{od}/tokens', exist_ok=True)
    comp = {k: [] for k in ('name', 'sem', 'var', 'perc', 'cls_pred', 'cls', 'subj_pred')}
    for i in range(0, len(ids), a.bs):
        ks = ids[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, fi, perceptive_override=None if a.perceptive == 'keep' else 'mean')
        for j, k in enumerate(ks):
            np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
        comp['name'] += ks; comp['cls'] += [SYN.index(k.split('/')[0]) for k in ks]
        comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu())
        comp['cls_pred'].append((out['sem_clip'].float() @ model.prototypes.t()).argmax(-1).cpu()); comp['subj_pred'].append(out['subj_logit'].argmax(-1).cpu())
    np.savez(f'{od}/components.npz', name=np.array(comp['name']), cls=np.array(comp['cls']),
             **{k: torch.cat(comp[k]).numpy() for k in ('sem', 'var', 'perc', 'cls_pred', 'subj_pred')})
    acc = float((torch.cat(comp['cls_pred']).numpy() == np.array(comp['cls'])).mean())
    print(f'{sub} {a.perceptive}: {len(ids)} tokens → {od}, test 범주 정확도 {acc:.3f}', flush=True)
print('INFER_DONE', flush=True)
