# PRISM 추론 1단계 (MinD-3D++ venv, 실행 위치 $W/run): test fMRI → TRELLIS 조건 토큰 [1374, 1024] (fp16) + 성분 벡터
#  --perceptive keep|mean : keep = 그대로(보정), mean = 배치 평균으로 바꿔 끼움(사실상 버리기)
#  출력 <out>/<sub>_<perceptive>/tokens/<uid>.npy, <out>/<sub>_<perceptive>/components.npz (sem·var·perc 벡터, 범주·피험자 예측, 범주 점수 cls_score [N, 13] · classes)
#  --color_mode (color 1 모델만, 11-08): 색 코드를 무엇으로 넣나. none = 없음(fMRI만) / pred = 색 머리 예측(fMRI만) / cat_sample = 예측 top-1 범주의 train 코드 하나를 무작위로
#    cat_mean = 예측 top-1 범주의 train 코드 평균 / gt = 그 물체의 자극 코드 (상한, 방법으로 쓰지 않음). none 말고는 출력 폴더 <sub>_<perceptive>_<mode>
#    components.npz에 color_pred · color_used (표준화 전 값, 정규채도 · rg_n · by_n) 추가, 로그에 색 머리 예측 r (test 정답 대비)
#  2단계(GLB)는 meshgen venv의 trellis_tools.py gen
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--split', default='test')
p.add_argument('--perceptive', default='keep', choices=['keep', 'mean']); p.add_argument('--bs', type=int, default=8)
p.add_argument('--color_mode', default='none', choices=['none', 'pred', 'cat_sample', 'cat_mean', 'gt'])
p.add_argument('--color_npz', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'stim_color.npz')); p.add_argument('--seed', type=int, default=0)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()   # 피험자별 체크포인트는 enc_paths에서 ViT, 옛 체크포인트(11-02 첫 학습)는 Temporal Stem 없음
print('load ok | per_subject', bool((ck.get('args') or {}).get('per_subject', 0)), '| subs', SUBS, flush=True)
ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
COL = getattr(model, 'color', 0)
assert COL or a.color_mode == 'none', '색 코드는 --color 1로 학습한 체크포인트만'
if COL:
    _z = np.load(a.color_npz); _names = list(_z['names']); _Y = _z['Y'][:, [_names.index(c) for c in ('정규채도', 'rg_n', 'by_n')]].astype(np.float32)
    CODE = dict(zip(list(_z['keys']), _Y)); MU, SD = model.color_mu.cpu().numpy(), model.color_sd.cpu().numpy()
    TR = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_train_list.txt')) if l.strip()]
    BYCAT = {c: np.stack([CODE[k] for k in TR if k.split('/')[0] == c]) for c in SYN}          # 범주별 train 코드 (표준화 전)
    rng = np.random.RandomState(a.seed)
for sub in (a.subs.split(',') if a.subs else SUBS):
    od = f'{a.out}/{sub}_{a.perceptive}' + ('' if a.color_mode == 'none' else f'_{a.color_mode}'); os.makedirs(f'{od}/tokens', exist_ok=True); si = SUBS.index(sub)
    comp = {k: [] for k in ('name', 'sem', 'var', 'perc', 'cls_pred', 'cls_score', 'cls', 'subj_pred', 'color_pred', 'color_used')}
    for i in range(0, len(ids), a.bs):
        ks = ids[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, sub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        sj = torch.full((len(ks),), si, dtype=torch.long, device='cuda'); po = None if a.perceptive == 'keep' else 'mean'
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, fi, sj, perceptive_override=po)                          # 색 코드 없이 (fMRI만)
        if COL:
            cp = out['color_pred'].float().cpu().numpy() * SD + MU; comp['color_pred'].append(cp)
            top1 = (out['sem_clip'].float() @ model.prototypes.t()).argmax(-1).cpu().numpy()
            if a.color_mode != 'none':
                used = {'pred': lambda j, k: cp[j], 'gt': lambda j, k: CODE[k], 'cat_mean': lambda j, k: BYCAT[SYN[top1[j]]].mean(0),
                        'cat_sample': lambda j, k: BYCAT[SYN[top1[j]]][rng.randint(len(BYCAT[SYN[top1[j]]]))]}[a.color_mode]
                cu = np.stack([used(j, k) for j, k in enumerate(ks)]).astype(np.float32); comp['color_used'].append(cu)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(x, fi, sj, perceptive_override=po, color_code=torch.from_numpy((cu - MU) / SD).cuda())
        for j, k in enumerate(ks):
            np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
        comp['name'] += ks; comp['cls'] += [SYN.index(k.split('/')[0]) for k in ks]
        comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu())
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); comp['cls_score'].append(sc)   # CLIP 텍스트 프로토타입과 cos
        comp['cls_pred'].append(sc.argmax(-1)); comp['subj_pred'].append(out['subj_logit'].argmax(-1).cpu())
    np.savez(f'{od}/components.npz', name=np.array(comp['name']), cls=np.array(comp['cls']), classes=np.array(SYN),
             **{k: torch.cat(comp[k]).numpy() for k in ('sem', 'var', 'perc', 'cls_pred', 'cls_score', 'subj_pred')},
             **{k: np.concatenate(comp[k]) for k in ('color_pred', 'color_used') if comp[k]})
    if COL:
        P = np.concatenate(comp['color_pred']); G = np.stack([CODE[k] for k in comp['name']])
        print(f'{sub} 색 머리 예측 r (test 정답 대비) ' + ' '.join(f'{n} {np.corrcoef(P[:, j], G[:, j])[0, 1]:+.3f}' for j, n in enumerate(('정규채도', 'rg_n', 'by_n'))), flush=True)
    acc = float((torch.cat(comp['cls_pred']).numpy() == np.array(comp['cls'])).mean())
    print(f'{sub} {a.perceptive}: {len(ids)} tokens → {od}, test 범주 정확도 {acc:.3f}', flush=True)
print('INFER_DONE', flush=True)
