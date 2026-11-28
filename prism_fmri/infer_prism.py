# PRISM 추론 1단계 (MinD-3D++ venv, 실행 위치 $W/run): test fMRI → TRELLIS 조건 토큰 [1374, 1024] (fp16) + 성분 벡터
#  --perceptive keep|mean : keep = 그대로(보정), mean = 배치 평균으로 바꿔 끼움(사실상 버리기)
#  출력 <out>/<sub>_<perceptive>/tokens/<uid>.npy, <out>/<sub>_<perceptive>/components.npz (sem·var·perc 벡터, 범주·피험자 예측, 범주 점수 cls_score [N, 13] · classes)
#  --fold k/n (11-09): 목록(annotations 순서)에서 i % n == k인 물체만 (train_prism.py --fold k/n이 뺀 물체 = 안 본 물체)
#  --names <uid 목록>: 그 물체만
#  --tok_pool 1: components.npz에 예측 토큰 요약 tok_pool [N, 54, 1024] fp16 (token_util.pool, 토큰 prior 조건), --save_tokens 0이면 토큰 파일은 안 씀
#  11-10: components.npz에 clip(S CLIP 벡터 [N, 1024]) · vt(V 예측 프레임 평균, var_heads sil · view · shape일 때)도 저장
#  2단계(GLB)는 meshgen venv의 trellis_tools.py gen
#  11-21 P-2 새 피험자 "짝 없이" (--fmri_sub와 같이): --own_stats train|list = feat_norm 평균 · 표준편차를 그 피험자 자기 fMRI로 다시 잼 (라벨 · 짝 없이, 학습 때와 같은 식 · 프레임 2~7 각각이 표본)
#    train = core_train 목록 (4번처럼 train이 있는 피험자), list = 추론하는 목록 그대로 (AP 9번 · APAC 11번처럼 그것밖에 없는 피험자). S · V가 피험자 공통(shared_sv)인 모델에서만 — 그러면 --subs 갈래 번호와 무관
#    --rw_common 1 = 영역 가중의 피험자별 조정 d를 0으로 (공통 가중만). 출력 폴더 <fmri_sub>_new_keep
import os, sys, argparse, numpy as np, torch
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prism import build_from_ckpt
from token_util import pool
p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--ckpt', required=True)
p.add_argument('--out', required=True); p.add_argument('--subs', default=''); p.add_argument('--split', default='test')
p.add_argument('--perceptive', default='keep', choices=['keep', 'mean']); p.add_argument('--bs', type=int, default=8)
p.add_argument('--fold', default=''); p.add_argument('--tok_pool', type=int, default=0); p.add_argument('--save_tokens', type=int, default=1)
p.add_argument('--save_var', type=int, default=0)   # 11-24 V-4: 1이면 디코더가 받는 Variational 특징 var_feat [T + patch, 768]을 <od>/var/<uid>.npy (fp16)
p.add_argument('--names', default='')   # uid 목록 파일: 이 물체만 (fold 거른 뒤)
p.add_argument('--fmri_sub', default='')   # AP · APAC (MinD-3D++ Tab.3, "model trained on Subject 1"): 학습에 없던 피험자 fMRI(예: sub-0009)를 --subs 하나(예: sub-0001)의 갈래로 추론
p.add_argument('--list', default='')       # 물체 목록 파일 (기본 annotations/core_<split>_list.txt, APAC는 annotations/apac_list.txt)
p.add_argument('--own_stats', default='', choices=['', 'train', 'list'])   # 11-21 P-2: 새 피험자 자기 fMRI로 feat_norm 통계 (위 설명)
p.add_argument('--rw_common', type=int, default=0, choices=[0, 1])         # 11-21 P-2: 영역 가중 피험자별 조정 0 (공통 가중만)
a = p.parse_args()
ck = torch.load(a.ckpt, map_location='cpu')
SUBS = ck['subs']; SYN = ck['classes']; FRAMES = list(range(2, 8))
model = build_from_ckpt(ck, a.repo).cuda().eval()   # 피험자별 체크포인트는 enc_paths에서 ViT, 옛 체크포인트(11-02 첫 학습)는 Temporal Stem 없음
OSIL = {}
if (ck.get('args') or {}).get('oracle_sil'):                 # 11-18 V-0: 정답 실루엣 지도 (진단 전용 상한, 학습 때 경로)
    _z = np.load(ck['args']['oracle_sil'])
    for _kk in _z.files:
        if _kk == 'sil': OSIL.update({('*', str(u)): _z['sil'][i].astype(np.float32).mean(0) / 255.0 for i, u in enumerate(_z['uids'])})
        elif _kk == 'map' or _kk.startswith('map_'): OSIL.update({('*' if _kk == 'map' else _kk[4:], str(u)): _z[_kk][i].astype(np.float32) for i, u in enumerate(_z['uids'])})
print('load ok | per_subject', bool((ck.get('args') or {}).get('per_subject', 0)), '| subs', SUBS, flush=True)
ids = [l.strip() for l in open(a.list or os.path.join(a.data_root, 'annotations', f'core_{a.split}_list.txt')) if l.strip()]
assert not a.fmri_sub or (a.subs and ',' not in a.subs), '--fmri_sub는 --subs에 갈래 피험자 하나'
if a.own_stats:
    assert a.fmri_sub and model.feat_norm and model.shared_sv, '--own_stats는 --fmri_sub + feat_norm + shared_sv 모델에서만'
if a.rw_common:
    assert hasattr(model, 'rw'), '--rw_common: 영역 가중 없는 모델'
    with torch.no_grad(): model.rw.d_s.zero_(); model.rw.d_v.zero_()
    print('[rw_common] 영역 가중 피험자별 조정 0 → 공통 가중만', flush=True)
if a.fold: fk, fn = map(int, a.fold.split('/')); ids = [k for i, k in enumerate(ids) if i % fn == fk]
if a.names: _w = {l.split()[-1] for l in open(a.names) if l.strip()}; ids = [k for k in ids if k.split('/')[1] in _w]
for sub in (a.subs.split(',') if a.subs else SUBS):
    dsub = a.fmri_sub or sub                                    # fMRI를 읽는 피험자 폴더
    od = f'{a.out}/{dsub}_{a.perceptive}' if not a.fmri_sub else f'{a.out}/{dsub}_new_{a.perceptive}' if a.own_stats else f'{a.out}/{dsub}_as{sub[-2:]}_{a.perceptive}'; os.makedirs(f'{od}/tokens', exist_ok=True); si = SUBS.index(sub)
    if a.own_stats:                                             # train_prism.py feat_norm 계산과 같은 식 (자기 fMRI만, 라벨 · 짝 안 씀)
        sk = ids if a.own_stats == 'list' else [l.strip() for l in open(os.path.join(a.data_root, 'annotations', 'core_train_list.txt')) if l.strip()]
        n_ = 0; s1 = torch.zeros(257, 1024, device='cuda', dtype=torch.float64); s2 = torch.zeros_like(s1)
        with torch.no_grad():
            for i in range(0, len(sk), 16):
                x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, dsub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in sk[i:i + 16]])).cuda()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    t = model.encode_frames(x, torch.full((len(x),), si, dtype=torch.long, device='cuda')).float().flatten(0, 1).double()
                s1 += t.sum(0); s2 += (t * t).sum(0); n_ += len(t)
        mu = s1 / n_; model.feat_mu[si] = mu.float(); model.feat_sd[si] = ((s2 / n_ - mu * mu).clamp_min(0).sqrt() + 1e-6).float()
        print(f'[own_stats] {dsub} 자기 fMRI ({a.own_stats}, 물체 {len(sk)} · 표본 {n_})로 feat_norm 통계 · 평균 |mu| {float(mu.abs().mean()):.3f} · 평균 sd {float(model.feat_sd[si].mean()):.3f}', flush=True)
    comp = {k: [] for k in ('name', 'sem', 'var', 'perc', 'cls_pred', 'cls_score', 'cls', 'subj_pred', 'tok_pool', 'img', 'clip', 'vt', 'zb', 'w_ent', 'w_max')}
    for i in range(0, len(ids), a.bs):
        ks = ids[i:i + a.bs]
        x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, dsub, 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks])).cuda()
        fi = torch.tensor(FRAMES).expand(len(ks), -1).cuda()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(x, fi, torch.full((len(ks),), si, dtype=torch.long, device='cuda'), perceptive_override=None if a.perceptive == 'keep' else 'mean',
                        sil=torch.from_numpy(np.stack([OSIL.get((dsub, k.split('/')[1]), OSIL.get(('*', k.split('/')[1]))) for k in ks])).cuda() if OSIL else None)
        if a.tok_pool: comp['tok_pool'].append(pool(out['cond_tokens'].float()).half().cpu())
        for j, k in enumerate(ks if a.save_tokens else []):
            np.save(f'{od}/tokens/{k.split("/")[1]}.npy', out['cond_tokens'][j].float().half().cpu().numpy())
        if a.save_var:
            os.makedirs(f'{od}/var', exist_ok=True)
            for j, k in enumerate(ks): np.save(f'{od}/var/{k.split("/")[1]}.npy', out['var_feat'][j].float().half().cpu().numpy())
        comp['name'] += ks; comp['cls'] += [SYN.index(k.split('/')[0]) if k.split('/')[0] in SYN else -1 for k in ks]   # APAC: 학습에 없는 범주 = −1
        comp['sem'].append(out['sem_vec'].float().cpu()); comp['var'].append(out['var_vec'].float().cpu()); comp['perc'].append(out['perc_vec'].float().cpu())
        sc = (out['sem_clip'].float() @ model.prototypes.t()).cpu(); comp['cls_score'].append(sc)   # CLIP 텍스트 프로토타입과 cos
        comp['cls_pred'].append(sc.argmax(-1)); comp['subj_pred'].append(out['subj_logit'].argmax(-1).cpu())
        if 'sem_img' in out: comp['img'].append(out['sem_img'].float().cpu())
        comp['clip'].append(out['sem_clip'].float().cpu())
        if 'z_B' in out: comp['zb'].append(out['z_B'].float().cpu()); comp['w_ent'].append(out['bank_w_ent_b'].float().cpu()); comp['w_max'].append(out['bank_w_max_b'].float().cpu())   # 11-28 P-5                                  # 11-10: S의 CLIP 벡터 (정규화, 프로토타입 공간)
        if 'vt' in out: comp['vt'].append(out['vt'].float().mean(1).cpu())               # 11-10: V 예측 (shape면 개체 형상 [cy, w, h, area] 표준화, 프레임 평균)             # 11-10 A안: 예측 이미지 임베딩 (정규화 전)
    np.savez(f'{od}/components.npz', name=np.array(comp['name']), cls=np.array(comp['cls']), classes=np.array(SYN),
             **{k: torch.cat(comp[k]).numpy() for k in ('sem', 'var', 'perc', 'cls_pred', 'cls_score', 'subj_pred', 'tok_pool', 'img', 'clip', 'vt', 'zb', 'w_ent', 'w_max') if comp[k]})
    cl = np.array(comp['cls']); kn = cl >= 0
    acc = float((torch.cat(comp['cls_pred']).numpy()[kn] == cl[kn]).mean()) if kn.any() else float('nan')
    print(f'{dsub} ({sub} 갈래) {a.perceptive}: {len(ids)} 물체 → {od}, 범주 정확도 {acc:.3f} (13범주 안 {int(kn.sum())}개)', flush=True)
print('INFER_DONE', flush=True)
