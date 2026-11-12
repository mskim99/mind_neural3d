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
#  --lam_tc λ · --lam_hsic λ (11-10): S · V · P 분리 — 결합-섞기 판별기(총상관) · RBF CKA 합 (sep_losses.py). 판별기는 모델 밖, latest에만 저장
#  11-12 과적합 대책 (사용자 승인, `.보고 자료/261112_9차보고/PRISM_개선/성능개선_가설·계획.md`): --holdout n (train에서 범주마다 목록 끝 n개를 떼어 [hold]로만 평가 — 설정 · 스텝 수 고르기용,
#    [val] = test는 기록만 하고 고르는 데 쓰지 않음) · --dropout · --wd · --aug_noise · --aug_mask (fMRI 증강, 학습 때만) · --lin_path (Semantic 안 원 신호 선형 경로) · --save_steps (model_step<N>.pt)
#  이전 기준선(prism_nop_feat · value)을 다시 학습하려면 --arch res --sem_clip back --var_heads csr
import os, sys, math, time, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from prism import Prism, prism_losses, load_encoder, check_load
from sep_losses import TCDisc, tc_enc, tc_disc_update, hsic_loss, share_loss, rem_losses

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
p.add_argument('--film', type=int, default=1, choices=[0, 1])   # 0: ablation — Perceptive FiLM 끔 (P 빼기 = --lam_subj 0 --film 0)
p.add_argument('--print_every', type=int, default=20); p.add_argument('--val_every', type=int, default=1000); p.add_argument('--save_every', type=int, default=2000)
p.add_argument('--limit', type=int, default=0); p.add_argument('--resume', action='store_true')
p.add_argument('--var_mode', default='feat', choices=['feat', 'value'])   # Variational → Aggregate 입력: LayerNorm 특징 / 실제 값 임베딩
p.add_argument('--sem_clip', default='front', choices=['front', 'back'])   # front: CLIP Projection이 Semantic 경로 앞 (11-06 구조도) / back: 이전 곁가지
p.add_argument('--var_heads', default='sr', choices=['sr', 'csr', 'sil', 'view', 'shape'])   # sr: Scale + 3D 크기 비율 (11-06) / csr: 이전 Center · Scale · Rotate / sil · view: 변하는 정보 (11-09)
p.add_argument('--view_tgt', default='')      # sil · view 정답 npz (build_view_targets.py)
p.add_argument('--lam_vadv', type=float, default=0.0)   # > 0이면 V 범주 GRL
p.add_argument('--lam_img', type=float, default=0.0)       # 11-10 A안: > 0이면 Semantic 이미지 임베딩 head + (1 − cos) 손실 가중치
p.add_argument('--lam_img_nce', type=float, default=0.0)   # 같은 head의 배치 InfoNCE 가중치
p.add_argument('--img_tgt', default='')                    # 정답 임베딩 폴더 <uid>.npy (비우면 <tgt>/clipimg), 자극 프레임 0의 unCLIP 이미지 인코더 임베딩
p.add_argument('--arch', default='fig', choices=['fig', 'res'])   # fig: 11-06 구조도 Temporal Stem · SV Encoder / res: 이전 구현
p.add_argument('--per_subject', type=int, default=1, choices=[1])   # 피험자별 ViT · S · V만 지원 (11-04 사용자 결정). 기록용으로 args에 남김 (build_from_ckpt가 확인)
p.add_argument('--temporal', type=int, default=1, choices=[1])     # Temporal Stem 항상 사용. 기록용
p.add_argument('--fold', default='')       # 'k/n': train에서 i % n == k 물체를 뺌 (교차 학습). 비우면 전부
p.add_argument('--enc_path', default='')                # 11-12: 피험자 공통 인코더 파일 하나 (UKB 사전학습 등). 주면 --enc_dir 대신 씀
p.add_argument('--feat_norm', type=int, default=0, choices=[0, 1])   # 인코더 출력 표준화 (피험자 · 위치 · 차원별, 학습 물체 통계)
p.add_argument('--in_norm', type=int, default=0, choices=[0, 1])   # 인코더 입력 프레임별 z-score
p.add_argument('--enc_dir', default='')               # <enc_dir>/subNN_encoder.pt (MinD-3D++ 피험자별 공개 가중치의 인코더)
p.add_argument('--lam_tc', type=float, default=0.0)     # 11-10: > 0이면 결합-섞기 판별기 총상관 손실 가중치 (sep_losses.py A)
p.add_argument('--lam_hsic', type=float, default=0.0)   # 11-10: > 0이면 갈래 쌍 RBF CKA 합 가중치 (sep_losses.py C, 대조)
p.add_argument('--tc_lr', type=float, default=1e-4)     # 판별기 학습률 (Adam β 0.5, 0.9)
p.add_argument('--lam_share', type=float, default=0.0)   # 11-11: > 0이면 피험자 간 일치 손실 (같은 물체 · 다른 피험자의 S(· V)를 가깝게, sep_losses.share_loss) → 짝 배치
p.add_argument('--share_on', default='s', choices=['s', 'sv'])   # 일치 손실을 거는 갈래: S(sem_vec)만 / S · V(var_vec) 둘 다
p.add_argument('--share_temp', type=float, default=0.07)
p.add_argument('--share_warm', type=int, default=2000)   # 일치 손실 세기를 이 스텝 동안 0 → λ로 올림 (검토: 초기 기울기가 토큰 손실보다 훨씬 큼)
p.add_argument('--paired', type=int, default=0, choices=[0, 1])   # 1: 짝 배치만 (일치 손실 없이, 배치 구성 대조군). lam_share > 0이면 자동으로 짝 배치
p.add_argument('--pair_k', type=int, default=2)           # 물체 하나를 배치에 넣는 피험자 수 (bs 14 → 물체 7 × 피험자 2)
p.add_argument('--lam_rem', type=float, default=0.0)      # 11-11: > 0이면 P = 나머지 복원 (sep_losses.rem_losses), S · V 갈래에는 기울기 안 감
p.add_argument('--holdout', type=int, default=0)        # 11-12: > 0이면 train 범주마다 목록 끝 n개를 학습에서 빼고 [hold]로 평가
p.add_argument('--dropout', type=float, default=0.0)    # Mixed Features · 디코더 입력 · 디코더 안
p.add_argument('--wd', type=float, default=0.01)        # AdamW weight decay (이전 값 0.01)
p.add_argument('--aug_noise', type=float, default=0.0)  # 학습 때 fMRI에 가우시안 잡음 (표본마다 입력 표준편차 × 이 값)
p.add_argument('--aug_mask', type=float, default=0.0)   # 학습 때 16×16 패치를 이 비율만큼 0으로 (표본마다, 프레임 공통)
p.add_argument('--lin_path', type=int, default=0, choices=[0, 1])   # Semantic 안 원 신호 선형 경로
p.add_argument('--vit_lin', type=int, default=0)   # Semantic 안 ViT 출력 선형 읽기 (위치 유지), 값 = 패치당 줄인 차원 r (0 = 끔)
p.add_argument('--enc_unfreeze', type=int, default=0)   # 11-12: > 0이면 피험자별 ViT 마지막 n개 블록 + norm을 같이 학습 (체크포인트에 같이 저장)
p.add_argument('--enc_lr', type=float, default=1e-5)     # 푼 ViT 블록 학습률
p.add_argument('--save_steps', default='')              # '5000,10000': 그 스텝에서 model_step<N>.pt (학습 가중치만)
p.add_argument('--seed', type=int, default=-1)   # 11-10: ≥ 0이면 torch · numpy · random 시드 고정 (반복 실험용), -1이면 이전처럼 고정 안 함
a = p.parse_args()
if a.seed >= 0:
    import random; random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
VT, a.vdim = {}, 0
if a.var_heads in ('sil', 'view', 'shape'):                                          # uid → ([6, vdim], ok)
    _z = np.load(a.view_tgt); _y = _z[a.var_heads].astype(np.float32); _ok = _z[a.var_heads + '_ok']
    VT = {str(k).split('/')[-1]: (_y[i], bool(_ok[i])) for i, k in enumerate(_z['keys'])}; a.vdim = int(_y.shape[-1])
    print(f'[view] {a.var_heads} 정답 {len(VT)}개 (정상 {int(_ok.sum())}), vdim {a.vdim}', flush=True)
IMG = a.lam_img > 0 or a.lam_img_nce > 0; IMG_TGT = a.img_tgt or f'{a.tgt}/clipimg'
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


OBJ = {}   # 물체 이름 → 번호 (일치 손실의 짝 찾기용, ShapeSet이 작업자를 띄우기 전에 채움)


def load_item(si, k):
    syn, uid = k.split('/')
    x = np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k + '.npy')).astype(np.float32)[FRAMES]
    t = np.load(f'{a.tgt}/targets/{uid}.npz'); vf = [min(191, round(19.2 * f)) for f in FRAMES]
    return {'fmri': torch.from_numpy(x), 'frame_idx': torch.tensor(FRAMES), 'cls': SYN.index(syn), 'subj': si, 'obj': OBJ[k],
            'cond_tokens': torch.from_numpy(np.load(f'{a.tgt}/dino/{uid}.npy')),
            'center': torch.from_numpy(np.stack([t['cx'][vf], t['cy'][vf]], -1)), 'scale': torch.from_numpy(t['scale'][vf]),
            'rot': torch.from_numpy(np.stack([rot_target(float(t['az_deg'][j])) for j in vf])), 'name': k,
            **(lambda r: {'ratio': torch.from_numpy(r[0]), 'ratio_ok': r[1]})(ratio_target(uid) if a.var_heads == 'sr' else (np.zeros(3, np.float32), False)),
            **(lambda v: {'vt': torch.from_numpy(v[0]), 'vt_ok': v[1]})(VT.get(uid, (np.zeros((len(FRAMES), max(a.vdim, 1)), np.float32), False))),
            **({'img_emb': torch.from_numpy(np.load(f'{IMG_TGT}/{uid}.npy').astype(np.float32))} if IMG else {})}


class ShapeSet(Dataset):
    def __init__(self, split):
        src = 'train' if split == 'hold' else split
        ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{src}_list.txt')) if l.strip()]
        if a.holdout > 0 and src == 'train':                                         # 범주마다 목록 끝 n개 = 검증 물체 (학습에서 뺌)
            by = {}
            for k in ids: by.setdefault(k.split('/')[0], []).append(k)
            H = {k for v in by.values() for k in v[-a.holdout:]}
            ids = [k for k in ids if (k in H) == (split == 'hold')]
        if split == 'train' and a.fold:
            fk, fn = map(int, a.fold.split('/')); ids = [k for i, k in enumerate(ids) if i % fn != fk]   # 목록 순서 기준 (infer_prism.py --fold와 같음)
        ids = [k for k in ids if os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy')]
        ids = ids[:a.limit] if a.limit else ids
        for k in ids: OBJ.setdefault(k, len(OBJ))
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
assert a.enc_path or a.enc_dir, '--enc_dir 또는 --enc_path'
ENC_PATHS = [a.enc_path] * len(SUBS) if a.enc_path else [f'{a.enc_dir}/sub{s[-2:]}_encoder.pt' for s in SUBS]
enc = [load_encoder(a.enc_path, cfg)] * len(SUBS) if a.enc_path else [load_encoder(pth, cfg) for pth in ENC_PATHS]
print(f'[init] 피험자별 encoder {len(enc)}개: {ENC_PATHS}', flush=True)
model = Prism(enc, n_classes=len(SYN), n_subjects=len(SUBS), var_mode=a.var_mode, arch=a.arch, sem_clip=a.sem_clip, var_heads=a.var_heads,
              vdim=a.vdim, grl_v=a.lam_vadv > 0, img_head=IMG, film=bool(a.film), remainder=a.lam_rem > 0, dropout=a.dropout, lin_path=bool(a.lin_path), in_norm=bool(a.in_norm), feat_norm=bool(a.feat_norm), vit_lin=a.vit_lin).to(dev)
STATE = lambda: model.trainable_state_dict()
SEP = a.lam_tc > 0 or a.lam_hsic > 0
disc = TCDisc().to(dev) if a.lam_tc > 0 else None
d_opt = torch.optim.Adam(disc.parameters(), lr=a.tc_lr, betas=(0.5, 0.9)) if disc is not None else None
PARTS = lambda o: [o['sem_vec'], o['var_vec'], o['perc_vec']]
model.prototypes.copy_(build_prototypes())
params = [q for q in model.parameters() if q.requires_grad]
enc_params = model.unfreeze_encoder(a.enc_unfreeze) if a.enc_unfreeze > 0 else []
if enc_params: print(f'[enc] ViT 마지막 {a.enc_unfreeze}블록 + norm 학습: {sum(q.numel() for q in enc_params) / 1e6:.1f}M, lr {a.enc_lr:g}', flush=True)
print(f'trainable {sum(q.numel() for q in params) / 1e6:.1f}M, subjects {SUBS}, bs {a.bs}, steps {a.max_step}', flush=True)
opt = torch.optim.AdamW([{'params': params}] + ([{'params': enc_params, 'lr': a.enc_lr}] if enc_params else []), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)
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


class PairedBatches(torch.utils.data.Sampler):
    """11-11 짝 배치: 배치 = 물체 (bs / k)개 × 피험자 k명. 물체 j는 피험자 순열의 (j + c·shift)번째 k명에게서 가져옴 →
    물체 수가 피험자 수의 배수면 배치 안 피험자 수가 같음 (bs 14 · k 2 · 7명 → 피험자마다 2개). 모든 피험자에 있는 물체만 씀."""
    def __init__(self, items, bs, k):
        self.idx = {}
        for i, (si, key) in enumerate(items): self.idx.setdefault(key, {})[si] = i
        self.S = sorted({si for si, _ in items}); self.keys = [key for key, d in self.idx.items() if len(d) == len(self.S)]
        assert 2 <= k <= len(self.S) and bs % k == 0, f'bs {bs} · pair_k {k} · 피험자 {len(self.S)}'
        self.k = k; self.m = bs // k; self.n = len(self.keys) // self.m
        print(f'[pair] 물체 {len(self.keys)} (전 피험자 공통) · 배치 = 물체 {self.m} × 피험자 {k} · 한 바퀴 {self.n} 배치', flush=True)
    def __len__(self): return self.n
    def __iter__(self):
        order = np.random.permutation(len(self.keys)); ns = len(self.S)
        for b in range(self.n):
            sp = np.random.permutation(self.S); sh = np.random.randint(1, ns) if ns > self.k else 1; out = []
            for j, oi in enumerate(order[b * self.m:(b + 1) * self.m]):
                for c in range(self.k): out.append(int(self.idx[self.keys[oi]][int(sp[(j + c * sh) % ns])]))
            yield out


PAIRED = a.paired == 1 or a.lam_share > 0
train_set = ShapeSet('train')
tl = DataLoader(train_set, batch_sampler=(PairedBatches(train_set.items, a.bs, a.pair_k) if PAIRED else BalancedBatches(train_set.items, a.bs)), num_workers=a.workers, persistent_workers=True)
vl = DataLoader(ShapeSet('test'), batch_size=a.bs, shuffle=False, num_workers=a.workers)
hl = DataLoader(ShapeSet('hold'), batch_size=a.bs, shuffle=False, num_workers=a.workers) if a.holdout > 0 else None
SAVE_STEPS = {int(x) for x in a.save_steps.split(',') if x.strip()}
step = 0; latest = f'{a.out_dir}/model_latest.pt'
if a.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu'); check_load(model, ck['model'])   # 구조 옵션이 다르면 멈춤
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); step = ck['step']; print('resumed', step, flush=True)
    if disc is not None and 'disc' in ck: disc.load_state_dict(ck['disc']); d_opt.load_state_dict(ck['d_opt'])
if a.feat_norm and step == 0:                                                        # 학습 물체(검증 물체 뺀 것)로 인코더 출력 평균 · 표준편차 (프레임 2~7 각각을 표본으로)
    with torch.no_grad():
        for si in range(len(SUBS)):
            ks = [k for s_, k in train_set.items if s_ == si]; n = 0; s1 = torch.zeros(257, 1024, device=dev, dtype=torch.float64); s2 = torch.zeros_like(s1)
            for i in range(0, len(ks), 16):
                x = torch.from_numpy(np.stack([np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k + '.npy')).astype(np.float32)[FRAMES] for k in ks[i:i + 16]])).to(dev)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    t = model.encode_frames(x, torch.full((len(x),), si, dtype=torch.long, device=dev)).float().flatten(0, 1).double()
                s1 += t.sum(0); s2 += (t * t).sum(0); n += len(t)
            mu = s1 / n; model.feat_mu[si] = mu.float(); model.feat_sd[si] = ((s2 / n - mu * mu).clamp_min(0).sqrt() + 1e-6).float()
            print(f'[feat_norm] {SUBS[si]} 표본 {n} · 평균 |mu| {float(mu.abs().mean()):.3f} · 평균 sd {float(model.feat_sd[si].mean()):.3f}', flush=True)
W = dict(tok=a.lam_tok, proto=a.lam_proto, var=a.lam_var, subj=a.lam_subj)
KEYS = ['tok_mse', 'tok_cos', 'proto', 'proto_acc', 'center', 'scale', 'rot', 'ratio', 'vt', 'v_adv', 'v_adv_acc', 'subj', 'subj_acc']
if IMG: KEYS += ['img_cos', 'img_nce', 'img_r1']
if PAIRED or a.lam_rem > 0: KEYS += ['share_s', 'share_s_acc', 'share_v', 'share_v_acc', 'rem_g', 'rem', 'rem_r2_sv', 'rem_r2_svp']   # 11-11 P = 나머지
if SEP: KEYS += ['tc', 'd_loss', 'd_acc', 'hsic']                                   # 끈 학습은 csv 열 그대로
log = open(f'{a.out_dir}/train_log.csv', 'a')
if os.path.getsize(f'{a.out_dir}/train_log.csv') == 0:
    log.write('step,split,total,' + ','.join(KEYS) + ',lr\n')


def to_dev(b):
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    out['prototypes'] = model.prototypes; return out


TERMS = {'tok': lambda L: W['tok'] * (L['tok_mse'] + L['tok_cos']), 'sem': lambda L: W['proto'] * L['proto'],
         'var': lambda L: W['var'] * (L['center'] + L['scale'] + L['rot'] + L['ratio'] + L['vt']), 'subj': lambda L: W['subj'] * L['subj'],
         'vadv': lambda L: L['v_adv'], 'img': lambda L: a.lam_img * L['img_cos'] + a.lam_img_nce * L['img_nce'],
         'sep': lambda L: a.lam_tc * L.get('tc', 0.0) + a.lam_hsic * L.get('hsic', 0.0),
         'share': lambda L: a.lam_share * min(1.0, step / max(1, a.share_warm)) * (L.get('share_s', 0.0) + (L.get('share_v', 0.0) if a.share_on == 'sv' else 0.0)),
         'rem': lambda L: a.lam_rem * (L.get('rem_g', 0.0) + L.get('rem', 0.0))}


def prem_losses(out, b, L, paired):
    """11-11: 일치 손실(짝 배치일 때만. V 값은 share_on과 상관없이 기록) · 나머지 복원."""
    if paired:
        L['share_s'], L['share_s_acc'] = share_loss(out['sem_vec'], b['obj'], a.share_temp)
        L['share_v'], L['share_v_acc'] = share_loss(out['var_vec'], b['obj'], a.share_temp)
    if a.lam_rem > 0: L['rem_g'], L['rem'], L['rem_r2_sv'], L['rem_r2_svp'] = rem_losses(out)


def total_loss(L):
    return sum(f(L) for f in TERMS.values())


def write(split, L, tot):
    log.write(f'{step},{split},{tot:.5f},' + ','.join(f'{float(L.get(k, 0.0)):.5f}' for k in KEYS) + f',{sched.get_last_lr()[0]:.3e}\n'); log.flush()


@torch.no_grad()
def validate(loader=None, tag='val'):
    loader = loader or vl; model.eval(); acc = {}; per = {}
    G = {k: [] for k in ('sem_vec', 'var_vec', 'obj', 'subj', 'enc_vec', 'rem_sv', 'rem_p')}   # 11-11: val은 배치가 피험자 순이라 짝이 없음 → 다 모아 끝에 한 번
    for b in loader:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'], b['subj'])
        L = prism_losses(out, b)
        if PAIRED or a.lam_rem > 0:
            for k in G: G[k].append((out[k] if k in out else b[k]).float().cpu()) if (k in out or k in b) else None
        if SEP and len(b['cls']) > 2:
            pt = PARTS(out); L['hsic'] = hsic_loss(pt)
            if disc is not None: L['tc'] = tc_enc(disc, pt); L['d_loss'], L['d_acc'] = tc_disc_update(disc, d_opt, pt, train=False)
        for k, v in L.items(): acc.setdefault(k, []).append(float(v))
        hit = ((out['sem_clip'].float() @ model.prototypes.t()).argmax(-1) == b['cls']).float()
        for si in b['subj'].unique().tolist(): per.setdefault(si, []).extend(hit[b['subj'] == si].tolist())
    L = {k: float(np.mean(v)) for k, v in acc.items()}
    if G['obj']:                                                                 # 같은 물체 · 다른 피험자 검색 top-1 (다른 피험자 표본 중에서), 전체 표본 분산 기준 설명 비율
        o_ = torch.cat(G['obj']); s_ = torch.cat(G['subj'])
        for nm, key in (('share_s_acc', 'sem_vec'), ('share_v_acc', 'var_vec')):
            z = torch.nn.functional.normalize(torch.cat(G[key]), dim=-1); sim = (z @ z.t()).masked_fill(s_[:, None] == s_[None], -2.0)
            L[nm] = float((o_[sim.argmax(-1)] == o_).float().mean())
        if G['enc_vec']:
            h = torch.cat(G['enc_vec']); h = torch.nn.functional.layer_norm(h, h.shape[-1:]); g = torch.cat(G['rem_sv']); r = torch.cat(G['rem_p']); var = h.var(0).mean()
            L['rem_g'] = float(((g - h) ** 2).mean()); L['rem'] = float(((g + r - h) ** 2).mean()); L['rem_r2_sv'] = float(1 - L['rem_g'] / var); L['rem_r2_svp'] = float(1 - L['rem'] / var)
    tot = float(total_loss(L)); write(tag, L, tot)
    print(f'[{tag}] step {step} total {tot:.4f} tok_cos {L["tok_cos"]:.4f} proto_acc {L["proto_acc"]:.3f} '
          f'rot {L["rot"]:.4f} ratio {L["ratio"]:.4f} vt {L["vt"]:.4f} v_adv_acc {L["v_adv_acc"]:.3f} subj_acc {L["subj_acc"]:.3f} '
          + (f'img_cos {L["img_cos"]:.4f} img_r1 {L["img_r1"]:.3f} ' if IMG else '')
          + (f'tc {L.get("tc", 0):.4f} d_acc {L.get("d_acc", 0):.3f} hsic {L.get("hsic", 0):.4f} ' if SEP else '') + (f'검색 S {L.get("share_s_acc", 0):.3f} V {L.get("share_v_acc", 0):.3f} rem R2 {L.get("rem_r2_sv", 0):.2f}>{L.get("rem_r2_svp", 0):.2f} ' if PAIRED or a.lam_rem > 0 else '') + '| 피험자별 범주 ' + ' '.join(f'{SUBS[k][-2:]}:{np.mean(v):.3f}' for k, v in sorted(per.items())), flush=True)
    model.train(); return L


def validate_all():
    validate()
    if hl is not None: validate(hl, 'hold')


def augment(x):
    # 11-12 fMRI 증강 (학습 때만): 표본마다 표준편차 × aug_noise 가우시안 잡음, 16×16 패치를 aug_mask 비율만큼 0으로 (프레임 공통)
    if a.aug_noise > 0: x = x + torch.randn_like(x) * x.std(dim=(1, 2, 3), keepdim=True) * a.aug_noise
    if a.aug_mask > 0:
        keep = (torch.rand(x.shape[0], 1, 16, 16, device=x.device) >= a.aug_mask).to(x.dtype)
        x = x * keep.repeat_interleave(16, 2).repeat_interleave(16, 3)
    return x


t0 = time.time(); model.train(); done = step >= a.max_step; ep = step // max(1, len(tl))
while not done:
    ep += 1
    for b in tl:
        b = to_dev(b)
        model.grl_lam = a.lam_vadv * (2 / (1 + math.exp(-10 * step / a.max_step)) - 1)
        if a.aug_noise > 0 or a.aug_mask > 0: b['fmri'] = augment(b['fmri'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'], b['subj'])
        L = prism_losses(out, b); prem_losses(out, b, L, PAIRED)
        if SEP:
            pt = PARTS(out)
            if disc is not None: L['tc'] = tc_enc(disc, pt)
            if a.lam_hsic > 0: L['hsic'] = hsic_loss(pt)
        tot = total_loss(L)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); (torch.nn.utils.clip_grad_norm_(enc_params, 1.0) if enc_params else None); opt.step(); sched.step(); step += 1
        if disc is not None: L['d_loss'], L['d_acc'] = tc_disc_update(disc, d_opt, pt)   # 본 backward 뒤 (판별기를 먼저 바꾸면 그래프가 깨짐)
        if step % a.print_every == 0:
            write('train', L, float(tot))
            print(f'[ep {ep}] step {step}/{a.max_step} total {float(tot):.4f} tok_mse {float(L["tok_mse"]):.4f} tok_cos {float(L["tok_cos"]):.4f} '
                  f'proto {float(L["proto"]):.3f} proto_acc {float(L["proto_acc"]):.3f} rot {float(L["rot"]):.4f} ratio {float(L["ratio"]):.4f} '
                  f'vt {float(L["vt"]):.4f} v_adv_acc {float(L["v_adv_acc"]):.3f} grl {model.grl_lam:.3f} '
                  f'subj_acc {float(L["subj_acc"]):.3f} ' + (f'share S {float(L.get("share_s", 0)):.3f}/{float(L.get("share_s_acc", 0)):.2f} V {float(L.get("share_v", 0)):.3f}/{float(L.get("share_v_acc", 0)):.2f} rem R2 {float(L.get("rem_r2_sv", 0)):.2f}>{float(L.get("rem_r2_svp", 0)):.2f} ' if PAIRED or a.lam_rem > 0 else '') + (f'tc {float(L.get("tc", 0)):.3f} d_acc {float(L.get("d_acc", 0)):.3f} hsic {float(L.get("hsic", 0)):.3f} ' if SEP else '') + f'lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s', flush=True)
        if step % a.val_every == 0: validate_all()
        if step in SAVE_STEPS:
            torch.save({'model': STATE(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, f'{a.out_dir}/model_step{step}.pt'); print(f'saved model_step{step}.pt', flush=True)
        if step % a.save_every == 0 or step >= a.max_step:
            torch.save({'model': STATE(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'args': vars(a),
                        'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS,
                        **({'disc': disc.state_dict(), 'd_opt': d_opt.state_dict()} if disc is not None else {})}, latest + '.tmp')
            os.replace(latest + '.tmp', latest); print(f'saved step {step}', flush=True)
        if step >= a.max_step: done = True; break
validate_all()
torch.save({'model': STATE(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, f'{a.out_dir}/model_final.pt')
print(f'TRAIN_DONE step {step} {time.time() - t0:.0f}s', flush=True)
