# PRISM 학습 (fMRI-Shape, 다피험자) — 10-27 구조도 분리부 그대로 (prism.py). MinD-3D++ venv, 실행 위치 $W/run (src → MinD-3D/src)
#  데이터·정답 (prep_shape_targets.py + scripts/dino.sh): <tgt>/targets/<uid>.npz (영상 방위·center·scale), <tgt>/dino/<uid>.npy (TRELLIS 조건 토큰, 영상 프레임 0). 복셀은 안 씀
#  fMRI 프레임 k ↔ 영상 프레임 round(19.2k) (MinD-3D 전처리가 BOLD 지연 6초를 이미 반영). 쓰는 fMRI 프레임 2~7
#  --per_subject 1 (11-04): 피험자별 ViT · Semantic · Variational + 공유 SV 인코더 · Perceptive · 디코더, 배치마다 피험자 같은 수, 저장에서 ViT 제외
#  손실 = λ_tok·(MSE + (1−cos)) + λ_proto·CE(범주 텍스트 프로토타입) + λ_var·(center + scale + rot) + λ_subj·CE(피험자)
#  --dev_loss 1 (11-05 정성 진단): (1−cos)를 train 평균 토큰을 뺀 편차끼리의 (1−cos)로 바꿈 + λ_nce·InfoNCE(물체 단위). val dev_cos 최저일 때 model_best.pt
import os, sys, math, time, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from src.encoder.autoencoder2 import fMRI_Encoder
from prism import Prism, prism_losses, load_encoder

CLASSES = {'02691156': 'airplane', '02828884': 'bench', '02933112': 'cabinet', '02958343': 'car', '03001627': 'chair', '03211117': 'display',
           '03636649': 'lamp', '03691459': 'loudspeaker', '04090263': 'rifle', '04256520': 'sofa', '04379243': 'table', '04401088': 'telephone',
           '04530566': 'watercraft'}
SYN = sorted(CLASSES)
FRAMES = list(range(2, 8))

p = argparse.ArgumentParser()
p.add_argument('--repo', required=True); p.add_argument('--data_root', required=True); p.add_argument('--tgt', required=True)
p.add_argument('--subs', default='sub-0001,sub-0002'); p.add_argument('--init_ckpt', required=True); p.add_argument('--out_dir', required=True)
p.add_argument('--bs', type=int, default=8); p.add_argument('--max_step', type=int, default=20000); p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--warmup', type=int, default=500); p.add_argument('--workers', type=int, default=8)
p.add_argument('--lam_tok', type=float, default=1.0); p.add_argument('--lam_proto', type=float, default=1.0)
p.add_argument('--lam_var', type=float, default=1.0); p.add_argument('--lam_subj', type=float, default=0.1)
p.add_argument('--print_every', type=int, default=20); p.add_argument('--val_every', type=int, default=1000); p.add_argument('--save_every', type=int, default=2000)
p.add_argument('--limit', type=int, default=0); p.add_argument('--resume', action='store_true')
p.add_argument('--var_mode', default='feat', choices=['feat', 'value'])   # Variational → Aggregate 입력: LayerNorm 특징 / 실제 값 임베딩
p.add_argument('--temporal', type=int, default=1)    # 1: Temporal Stem 사용 (구조도 Spatio-Temporal), 0: 없음
p.add_argument('--probe_grad', type=int, default=0)     # >0: 학습 대신 loss별 gradient 크기만 재고 끝 (배치 수)
p.add_argument('--probe_ckpt', default='')               # 비우면 초기값만, 주면 초기값 + 이 체크포인트 둘 다
p.add_argument('--per_subject', type=int, default=0)    # 1: 피험자별 ViT · Semantic · Variational, 공유 SV 인코더 · Perceptive · 디코더 (11-04 사용자 결정)
p.add_argument('--enc_dir', default='')                  # per_subject=1: <enc_dir>/subNN_encoder.pt (MinD-3D++ 피험자별 공개 가중치의 인코더)
p.add_argument('--dev_loss', type=int, default=0)       # 1: 토큰 (1−cos)를 평균 뺀 편차 (1−cos)로
p.add_argument('--lam_nce', type=float, default=0.0); p.add_argument('--nce_temp', type=float, default=0.07)
p.add_argument('--eval_ckpt', default='')               # 주면 학습 없이 이 체크포인트로 val 한 번만 (기준선 재평가)
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
dev = 'cuda'
SUBS = a.subs.split(',')


def rot_target(az_deg, elev_deg=30.0):
    """영상 카메라 자세 → 물체 회전(카메라 기준): Rx(고도) · Rz(−방위)."""
    t, e = math.radians(-az_deg), math.radians(elev_deg)
    Rz = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])
    return (Rx @ Rz).astype(np.float32)


def load_item(si, k):
    syn, uid = k.split('/')
    x = np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k + '.npy')).astype(np.float32)[FRAMES]
    t = np.load(f'{a.tgt}/targets/{uid}.npz'); vf = [min(191, round(19.2 * f)) for f in FRAMES]
    return {'fmri': torch.from_numpy(x), 'frame_idx': torch.tensor(FRAMES), 'cls': SYN.index(syn), 'subj': si,
            'cond_tokens': torch.from_numpy(np.load(f'{a.tgt}/dino/{uid}.npy')),
            'center': torch.from_numpy(np.stack([t['cx'][vf], t['cy'][vf]], -1)), 'scale': torch.from_numpy(t['scale'][vf]),
            'rot': torch.from_numpy(np.stack([rot_target(float(t['az_deg'][j])) for j in vf])), 'name': k}


class ShapeSet(Dataset):
    def __init__(self, split):
        ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{split}_list.txt')) if l.strip()]
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
if a.per_subject:
    ENC_PATHS = [f'{a.enc_dir}/sub{s[-2:]}_encoder.pt' for s in SUBS]
    enc = [load_encoder(pth, cfg) for pth in ENC_PATHS]
    print(f'[init] 피험자별 encoder {len(enc)}개: {ENC_PATHS}', flush=True)
else:
    ENC_PATHS = [a.init_ckpt]
    enc = fMRI_Encoder(cfg.model.params.fmri_encoder_config)
    sd = torch.load(a.init_ckpt, map_location='cpu'); sd = sd.get('model', sd)
    r = enc.load_state_dict({k[len('fmri_encoder.'):]: v for k, v in sd.items() if k.startswith('fmri_encoder.')}, strict=False)
    print(f'[init] encoder from {a.init_ckpt}: missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}', flush=True)
    del sd
model = Prism(enc, n_classes=len(SYN), n_subjects=len(SUBS), temporal=bool(a.temporal), per_subject_sv=bool(a.per_subject), var_mode=a.var_mode).to(dev)
STATE = (lambda: model.trainable_state_dict()) if a.per_subject else (lambda: model.state_dict())
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
if a.per_subject:
    tl = DataLoader(train_set, batch_sampler=BalancedBatches(train_set.items, a.bs), num_workers=a.workers, persistent_workers=True)
else:
    tl = DataLoader(train_set, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
vl = DataLoader(ShapeSet('test'), batch_size=a.bs, shuffle=False, num_workers=a.workers)
MT_PATH = f'{a.tgt}/mean_tok_train.npy'   # train 물체 GT 토큰 평균 (편차 손실·val dev_cos용, 한 번 계산해 둠)
if not os.path.exists(MT_PATH):
    uids = sorted({k.split('/')[1] for _, k in train_set.items}); acc_m = np.zeros((1374, 1024), np.float64)
    for u in uids: acc_m += np.load(f'{a.tgt}/dino/{u}.npy').astype(np.float64)
    np.save(MT_PATH + '.tmp.npy', (acc_m / len(uids)).astype(np.float32)); os.replace(MT_PATH + '.tmp.npy', MT_PATH); print(f'[mean_tok] train {len(uids)}개 평균 저장', flush=True)
MT = torch.from_numpy(np.load(MT_PATH)).to(dev)
step = 0; latest = f'{a.out_dir}/model_latest.pt'
if a.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu'); model.load_state_dict(ck['model'], strict=not a.per_subject)
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); step = ck['step']; print('resumed', step, flush=True)
W = dict(tok=a.lam_tok, proto=a.lam_proto, var=a.lam_var, subj=a.lam_subj)
KEYS = ['tok_mse', 'tok_cos', 'dev_cos', 'nce', 'proto', 'proto_acc', 'center', 'scale', 'rot', 'subj', 'subj_acc']
log = open(f'{a.out_dir}/train_log.csv', 'a')
if os.path.getsize(f'{a.out_dir}/train_log.csv') == 0:
    log.write('step,split,total,' + ','.join(KEYS) + ',lr\n')


def to_dev(b):
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    out['prototypes'] = model.prototypes; return out


TERMS = {'tok': lambda L: W['tok'] * (L['tok_mse'] + (L['dev_cos'] if a.dev_loss else L['tok_cos'])), 'nce': lambda L: a.lam_nce * L['nce'], 'sem': lambda L: W['proto'] * L['proto'],
         'var': lambda L: W['var'] * (L['center'] + L['scale'] + L['rot']), 'subj': lambda L: W['subj'] * L['subj']}


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
        L = prism_losses(out, b, mean_tok=MT, nce_temp=a.nce_temp)
        for k, v in L.items(): acc.setdefault(k, []).append(float(v))
        hit = ((out['sem_clip'].float() @ model.prototypes.t()).argmax(-1) == b['cls']).float()
        for si in b['subj'].unique().tolist(): per.setdefault(si, []).extend(hit[b['subj'] == si].tolist())
    L = {k: float(np.mean(v)) for k, v in acc.items()}; tot = float(total_loss(L)); write('val', L, tot)
    print(f'[val] step {step} total {tot:.4f} tok_cos {L["tok_cos"]:.4f} dev_cos {L["dev_cos"]:.4f} nce {L["nce"]:.3f} proto_acc {L["proto_acc"]:.3f} '
          f'rot {L["rot"]:.4f} subj_acc {L["subj_acc"]:.3f} | 피험자별 범주 ' + ' '.join(f'{SUBS[k][-2:]}:{np.mean(v):.3f}' for k, v in sorted(per.items())), flush=True)
    model.train(); return L


BEST = [float('inf')]
def save_best(L):   # val dev_cos(편차 방향 오차) 최저 체크포인트
    if L['dev_cos'] < BEST[0]:
        BEST[0] = L['dev_cos']
        torch.save({'model': STATE(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, f'{a.out_dir}/model_best.pt.tmp')
        os.replace(f'{a.out_dir}/model_best.pt.tmp', f'{a.out_dir}/model_best.pt'); print(f'saved best step {step} dev_cos {BEST[0]:.4f}', flush=True)


if a.eval_ckpt:
    ck = torch.load(a.eval_ckpt, map_location='cpu'); r = model.load_state_dict(ck['model'], strict=not a.per_subject); step = ck['step']
    print(f'[eval] {a.eval_ckpt} step {step}', flush=True); validate(); print('EVAL_DONE', flush=True); sys.exit(0)


if a.probe_grad:   # loss별로 SV Encoder·P에 도착하는 gradient 크기 (λ 곱한 값 기준, 학습 안 함)
    import json
    groups = {'sv_encoder': list(model.shared.parameters()) + list(model.frame_emb.parameters())}
    res = {}
    for tag in ['init'] + (['trained'] if a.probe_ckpt else []):
        if tag == 'trained':
            model.load_state_dict(torch.load(a.probe_ckpt, map_location='cpu')['model'], strict=not a.per_subject); model.to(dev)
        model.train(); acc = {}; it = iter(tl)
        for i in range(a.probe_grad):
            b = to_dev(next(it))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(b['fmri'], b['frame_idx'], b['subj'])
            L = prism_losses(out, b, mean_tok=MT, nce_temp=a.nce_temp)
            for name, f in TERMS.items():
                v = f(L); acc.setdefault((name, 'value'), []).append(float(v))
                for gname, ps in groups.items():
                    gr = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
                    n = math.sqrt(sum(float((x.float() ** 2).sum()) for x in gr if x is not None))
                    acc.setdefault((name, gname), []).append(n)
        res[tag] = {f'{k[0]}/{k[1]}': float(np.mean(v)) for k, v in acc.items()}
        print(f'[probe {tag}] {a.probe_grad} batches  (loss value · |grad| at SV Encoder · |grad| at P)', flush=True)
        for name in TERMS:
            print(f'  {name:5s}  value {res[tag][name + "/value"]:.4f}  sv_encoder {res[tag][name + "/sv_encoder"]:.4e}  P {res[tag][name + "/P"]:.4e}', flush=True)
    json.dump(res, open(f'{a.out_dir}/grad_probe.json', 'w'), indent=1); print('PROBE_DONE', flush=True); sys.exit(0)

t0 = time.time(); model.train(); done = step >= a.max_step; ep = step // max(1, len(tl))
while not done:
    ep += 1
    for b in tl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'], b['subj'])
        L = prism_losses(out, b, mean_tok=MT, nce_temp=a.nce_temp); tot = total_loss(L)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
        if step % a.print_every == 0:
            write('train', L, float(tot))
            print(f'[ep {ep}] step {step}/{a.max_step} total {float(tot):.4f} tok_mse {float(L["tok_mse"]):.4f} tok_cos {float(L["tok_cos"]):.4f} '
                  f'proto {float(L["proto"]):.3f} proto_acc {float(L["proto_acc"]):.3f} rot {float(L["rot"]):.4f} '
                  f'subj_acc {float(L["subj_acc"]):.3f} lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s', flush=True)
        if step % a.val_every == 0: save_best(validate())
        if step % a.save_every == 0 or step >= a.max_step:
            torch.save({'model': STATE(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'args': vars(a),
                        'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, latest + '.tmp')
            os.replace(latest + '.tmp', latest); print(f'saved step {step}', flush=True)
        if step >= a.max_step: done = True; break
save_best(validate())
torch.save({'model': STATE(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN, 'enc_paths': ENC_PATHS}, f'{a.out_dir}/model_final.pt')
print(f'TRAIN_DONE step {step} {time.time() - t0:.0f}s', flush=True)
