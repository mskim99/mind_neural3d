# PRISM 학습 (fMRI-Shape, 다피험자) — 구조는 prism.py (기본값 = 11-06 박사님 구조도: --arch fig --sem_clip front --var_heads sr --var_mode feat)
#  MinD-3D++ venv, 실행 위치 $W/run (src → MinD-3D/src). 실행은 scripts/run_*.sh
#  정답 (prep_shape_targets.py + scripts/dino.sh): <tgt>/targets/<uid>.npz (영상 방위·center·scale), <tgt>/dino/<uid>.npy (TRELLIS 조건 토큰, 영상 프레임 0),
#    <tgt>/voxels/<uid>.npy (64³, 가장 긴 변 = 1 → 크기 비율 = 축별 차 있는 범위 / 64, probe_split.py와 같음, --var_heads sr일 때만)
#  fMRI 프레임 k ↔ 영상 프레임 round(19.2k) (MinD-3D 전처리가 BOLD 지연 6초를 이미 반영). 쓰는 fMRI 프레임 2~7
#  피험자별 ViT(<enc_dir>/subNN_encoder.pt) · Semantic · Variational + 공유 Temporal Stem · SV Encoder · Perceptive · 디코더, 배치마다 피험자 같은 수, 저장에서 ViT 제외
#  손실 = λ_tok·(MSE + (1−cos)) + λ_proto·CE(범주 텍스트 프로토타입) + λ_var·(scale + ratio | center + scale + rot) + λ_subj·CE(피험자). 저장: model_latest.pt · model_final.pt
#  --var_heads sil|view (11-09): V = 변하는 정보. build_view_targets.py가 만든 --view_tgt npz(물체 평균 · 프레임별 train 평균을 빼고 train 표준편차로 나눈 값)를 프레임마다 맞춤
#  --lam_vadv λ (11-09): V에서 범주 정보 빼기 — Variational Features → GRL → 범주 분류기. 분류기 손실은 1배, 앞단으로 가는 기울기는 −λ·s(p) (s = DANN 일정 2/(1+e^(−10p))−1)
#  --fold k/n (11-09, 토큰 prior용 교차 학습): train 목록(annotations 순서)에서 i % n == k인 물체를 학습에서 뺌 → infer_prism.py --split train --fold k/n로 그 물체의 '안 본' 예측을 만듦
#  이전 기준선(prism_nop_feat · value)을 다시 학습하려면 --arch res --sem_clip back --var_heads csr
import os, sys, math, time, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from prism import Prism, prism_losses, load_encoder, check_load

CLASSES = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'display',
           '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone',
           '04530566': 'watercraft'}
SYN = sorted(CLASSES)
FRAMES = list(range(2, 8))

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--subs', default='sub-0001,sub-0002'); p.add_argument('--out_dir', required=True)
p.add_argument('--init_ckpt', default='')   # 쓰지 않음 (예전 공유 ViT용, 이전 스크립트가 넘겨도 되게 남김)
p.add_argument('--bs', type=int, default=8); p.add_argument('--max_step', type=int, default=20000); p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--warmup', type=int, default=500); p.add_argument('--workers', type=int, default=8)
p.add_argument('--lam_tok', type=float, default=1.0); p.add_argument('--lam_proto', type=float, default=1.0)
p.add_argument('--lam_var', type=float, default=1.0); p.add_argument('--lam_subj', type=float, default=0.1)
p.add_argument('--print_every', type=int, default=20); p.add_argument('--val_every', type=int, default=1000); p.add_argument('--save_every', type=int, default=2000)
p.add_argument('--limit', type=int, default=0); p.add_argument('--resume', action='store_true')
p.add_argument('--var_mode', default='feat', choices=['feat', 'value'])   # Variational → Aggregate 입력: LayerNorm 특징 / 실제 값 임베딩
p.add_argument('--sem_clip', default='front', choices=['front', 'back'])   # front: CLIP Projection이 Semantic 경로 앞 (11-06 구조도) / back: 이전 곁가지
p.add_argument('--var_heads', default='sr', choices=['sr', 'csr', 'sil', 'view', 'shape'])   # sr: Scale + 3D 크기 비율 (11-06) / csr: 이전 Center · Scale · Rotate / sil · view: 변하는 정보 (11-09)
p.add_argument('--view_tgt', default='')      # sil · view 정답 npz (build_view_targets.py)
p.add_argument('--lam_vadv', type=float, default=0.0)   # > 0이면 V 범주 GRL
p.add_argument('--arch', default='fig', choices=['fig', 'res'])   # fig: 11-06 구조도 Temporal Stem · SV Encoder / res: 이전 구현
p.add_argument('--per_subject', type=int, default=1, choices=[1])   # 피험자별 ViT · S · V만 지원 (11-04 사용자 결정). 기록용으로 args에 남김 (build_from_ckpt가 확인)
p.add_argument('--temporal', type=int, default=1, choices=[1])     # Temporal Stem 항상 사용. 기록용
p.add_argument('--fold', default='')       # 'k/n': train에서 i % n == k 물체를 뺌 (교차 학습). 비우면 전부
p.add_argument('--enc_dir', required=True)               # <enc_dir>/subNN_encoder.pt (MinD-3D++ 피험자별 공개 가중치의 인코더)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
VT, a.vdim = {}, 0
if a.var_heads in ('sil', 'view', 'shape'):                                          # uid → ([6, vdim], ok)
    _z = np.load(a.view_tgt); _y = _z[a.var_heads].astype(np.float32); _ok = _z[a.var_heads + '_ok']
    VT = {str(k).split('/')[-1]: (_y[i], bool(_ok[i])) for i, k in enumerate(_z['keys'])}; a.vdim = int(_y.shape[-1])
    print(f'[view] {a.var_heads} 정답 {len(VT)}개 (정상 {int(_ok.sum())}), vdim {a.vdim}', flush=True)
dev = 'cuda'
SUBS = a.subs.split(',')


def rot_target(az_deg, elev_deg=30.0):
    """영상 카메라 자세 → 물체 회전(카메라 기준): Rx(고도) · Rz(−방위)."""
    t, e = math.radians(-az_deg), math.radians(elev_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return (Rx @ Rz).astype(np.float32)


def ratio_target(uid):
    f = f'{a.tgt}/voxels/{uid}.npy'
    if not os.path.exists(f): return np.zeros(3, np.float32), False                  # 복셀 없는 물체 (소파 1개) → 손실에서 뺌
    occ = np.unpackbits(np.load(f))[:64 ** 3].reshape(64, 64, 64)
    ext = [(lambda nz: (nz.max() - nz.min() + 1) / 64 if len(nz) else 0.0)(np.nonzero(occ.any(axis=tuple(j for j in range(3) if j != ax)))[0]) for ax in range(3)]
    return np.array(ext, np.float32), True


def load_item(si, k):
    syn, uid = k.split('/')
    x = np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k + '.npy')).astype(np.float32)[FRAMES]
    t = np.load(f'{a.tgt}/targets/{uid}.npz'); vf = [min(191, round(19.2 * f)) for f in FRAMES]
    return {'fmri': torch.from_numpy(x), 'frame_idx': torch.tensor(FRAMES), 'cls': SYN.index(syn), 'subj': si,
            'cond_tokens': torch.from_numpy(np.load(f'{a.tgt}/dino/{uid}.npy')),
            'center': torch.from_numpy(np.stack([t['cx'][vf], t['cy'][vf]], -1)), 'scale': torch.from_numpy(t['scale'][vf]),
            'rot': torch.from_numpy(np.stack([rot_target(float(t['az_deg'][j])) for j in vf])), 'name': k,
            **(lambda r: {'ratio': torch.from_numpy(r[0]), 'ratio_ok': r[1]})(ratio_target(uid) if a.var_heads == 'sr' else (np.zeros(3, np.float32), False)),
            **(lambda v: {'vt': torch.from_numpy(v[0]), 'vt_ok': v[1]})(VT.get(uid, (np.zeros((len(FRAMES), max(a.vdim, 1)), np.float32), False)))}


class ShapeSet(Dataset):
    def __init__(self, split):
        ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{split}_list.txt')) if l.strip()]
        if split == 'train' and a.fold:
            fk, fn = map(int, a.fold.split('/')); ids = [k for i, k in enumerate(ids) if i % fn != fk]   # 목록 순서 기준 (infer_prism.py --fold와 같음)
        ids = [k for k in ids if os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy')]
        ids = ids[:a.limit] if a.limit else ids
        self.items = [(si, k) for si, s in enumerate(SUBS) for k in ids
                      if os.path.exists(os.path.join(a.data_root, s, 'npy_data', k + '.npy'))]
        print(f'[{split}] {len(ids)} objects × {len(SUBS)} subjects = {len(self.items)}', flush=True)
        have = {si for si, _ in self.items}
        assert len(have) == len(SUBS), f'[{split}] fMRI 데이터가 없는 피험자: {[s for i, s in enumerate(SUBS) if i not in have]}'
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return load_item(*self.items[i])


@torch.no_grad()
def build_prototypes():
    import open_clip
    m, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    tok = open_clip.get_tokenizer('ViT-H-14')
    m = m.to(dev).eval()
    f = m.encode_text(tok([f'a 3D object of {CLASSES[s]}' for s in SYN]).to(dev)).float()
    del m; torch.cuda.empty_cache()
    return torch.nn.functional.normalize(f, dim=-1)


cfg = OmegaConf.load(os.path.join(a.repo, 'configs', 'mind3d_pp.yaml'))
ENC_PATHS = [f'{a.enc_dir}/sub{s[-2:]}_encoder.pt' for s in SUBS]
enc = [load_encoder(pth, cfg) for pth in ENC_PATHS]
print(f'[init] 피험자별 encoder {len(enc)}개: {ENC_PATHS}', flush=True)
model = Prism(enc, n_classes=len(SYN), n_subjects=len(SUBS), var_mode=a.var_mode, arch=a.arch, sem_clip=a.sem_clip, var_heads=a.var_heads,
              vdim=a.vdim, grl_v=a.lam_vadv > 0).to(dev)
STATE = lambda: model.trainable_state_dict()
model.prototypes.copy_(build_prototypes())
params = [q for q in model.parameters() if q.requires_grad]
print(f'trainable {sum(q.numel() for q in params) / 1e6:.1f}M, subjects {SUBS}, bs {a.bs}, steps {a.max_step}', flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.max_step))))
class BalancedBatches(torch.utils.data.Sampler):
    """배치마다 피험자를 같은 수로 (bs / 피험자 수). 한 바퀴 = 가장 적은 피험자 기준."""
    def __init__(self, items, bs):
        self.by = {}
        for i, (si, _) in enumerate(items): self.by.setdefault(si, []).append(i)
        assert bs % len(self.by) == 0, f'bs {bs}는 피험자 수 {len(self.by)}의 배수여야 함'
        self.per = bs // len(self.by); self.n = min(len(v) for v in self.by.values()) // self.per
    def __len__(self): return self.n
    def __iter__(self):
        perm = {s: np.random.permutation(v) for s, v in self.by.items()}
        for b in range(self.n):
            yield [int(i) for s in sorted(perm) for i in perm[s][b * self.per:(b + 1) * self.per]]


train_set = ShapeSet('train')
tl = DataLoader(train_set, batch_sampler=BalancedBatches(train_set.items, a.bs), num_workers=a.workers, persistent_workers=True)
vl = DataLoader(ShapeSet('test'), batch_size=a.bs, shuffle=False, num_workers=a.workers)
step = 0; latest = f'{a.out_dir}/model_latest.pt'
if a.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu'); check_load(model, ck['model'])   # 구조 옵션이 다르면 멈춤
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); step = ck['step']; print('resumed', step, flush=True)
W = dict(tok=a.lam_tok, proto=a.lam_proto, var=a.lam_var, subj=a.lam_subj)
KEYS = ['tok_mse', 'tok_cos', 'proto', 'proto_acc', 'center', 'scale', 'rot', 'ratio', 'vt', 'v_adv', 'v_adv_acc', 'subj', 'subj_acc']
log = open(f'{a.out_dir}/train_log.csv', 'a')
if os.path.getsize(f'{a.out_dir}/train_log.csv') == 0:
    log.write('step,split,total,' + ','.join(KEYS) + ',lr\n')


def to_dev(b):
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    out['prototypes'] = model.prototypes; return out


TERMS = {'tok': lambda L: W['tok'] * (L['tok_mse'] + L['tok_cos']), 'sem': lambda L: W['proto'] * L['proto'],
         'var': lambda L: W['var'] * (L['center'] + L['scale'] + L['rot'] + L['ratio'] + L['vt']), 'subj': lambda L: W['subj'] * L['subj'],
         'vadv': lambda L: L['v_adv']}


def total_loss(L):
    return sum(f(L) for f in TERMS.values())


def write(split, L, tot):
    log.write(f'{step},{split},{tot:.5f},' + ','.join(f'{float(L[k]):.5f}' for k in KEYS) + f',{sched.get_last_lr()[0]:.3e}\n'); log.flush()


@torch.no_grad()
def validate():
    model.eval(); acc = {}; per = {}
    for b in vl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'], b['subj'])
        L = prism_losses(out, b)
        for k, v in L.items(): acc.setdefault(k, []).append(float(v))
        hit = ((out['sem_clip'].float() @ model.prototypes.t()).argmax(-1) == b['cls']).float()
        for si in b['subj'].unique().tolist(): per.setdefault(si, []).extend(hit[b['subj'] == si].tolist())
    L = {k: float(np.mean(v)) for k, v in acc.items()}; tot = float(total_loss(L)); write('val', L, tot)
    print(f'[val] step {step} total {tot:.4f} tok_cos {L["tok_cos"]:.4f} proto_acc {L["proto_acc"]:.3f} '
          f'rot {L["rot"]:.4f} ratio {L["ratio"]:.4f} vt {L["vt"]:.4f} v_adv_acc {L["v_adv_acc"]:.3f} subj_acc {L["subj_acc"]:.3f} | 피험자별 범주 ' + ' '.join(f'{SUBS[k][-2:]}:{np.mean(v):.3f}' for k, v in sorted(per.items())), flush=True)
    model.train(); return L


t0 = time.time(); model.train(); done = step >= a.max_step; ep = step // max(1, len(tl))
while not done:
    ep += 1
    for b in tl:
        b = to_dev(b)
        model.grl_lam = a.lam_vadv * (2 / (1 + math.exp(-10 * step / a.max_step)) - 1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'], b['subj'])
        L = prism_losses(out, b); tot = total_loss(L)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
        if step % a.print_every == 0:
            write('train', L, float(tot))
            print(f'[ep {ep}] step {step}/{a.max_step} total {float(tot):.4f} tok_mse {float(L["tok_mse"]):.4f} tok_cos {float(L["tok_cos"]):.4f} '
                  f'proto {float(L["proto"]):.3f} proto_acc {float(L["proto_acc"]):.3f} rot {float(L["rot"]):.4f} ratio {float(L["ratio"]):.4f} '
                  f'vt {float(L["vt"]):.4f} v_adv_acc {float(L["v_adv_acc"]):.3f} grl {model.grl_lam:.3f} '
                  f'subj_acc {float(L["subj_acc"]):.3f} lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s', flush=True)
        if step % a.val_every == 0: validate()
        if step % a.save_every == 0 or step >= a.max_step:
            torch.save({'model': STATE(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'args': vars(a),
                        'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, latest + '.tmp')
            os.replace(latest + '.tmp', latest); print(f'saved step {step}', flush=True)
        if step >= a.max_step: done = True; break
validate()
torch.save({'model': STATE(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, f'{a.out_dir}/model_final.pt')
print(f'TRAIN_DONE step {step} {time.time() - t0:.0f}s', flush=True)
