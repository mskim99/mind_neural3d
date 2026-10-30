# PRISM v2 학습 (fMRI-Shape, 다피험자) — diffusion 없이 정답 맞히기 loss만. MinD-3D++ venv, 실행 위치 $W/run (src → MinD-3D/src)
#  데이터: fMRI-Shape <data_root>/<sub>/npy_data/<synset>/<uid>.npy (10, 256, 256), core_train/test_list
#  정답 (prep_shape_targets.py + dino_targets.py): <tgt>/targets/<uid>.npz (영상 192프레임 방위·실루엣), <tgt>/voxels/<uid>.npy (64³ packbits),
#        <tgt>/dino/<uid>.npy (TRELLIS 조건 토큰 [1374, 1024] fp16, 영상 프레임 0)
#  fMRI 프레임 k ↔ 영상 프레임 round(19.2k) (8초 영상, fMRI 0.8초 간격 10프레임. MinD-3D 전처리가 BOLD 지연 6초를 이미 반영 → 추가 지연 없음)
#  쓰는 fMRI 프레임: 2~7 (MinD-3D++와 같음)
#  손실 = λ_tok·(MSE + (1−cos)) + λ_proto·CE + λ_geo·(BCE + Dice) + λ_var·(center + scale + rot) + λ_subj·CE(피험자)
import os, sys, math, time, glob, argparse, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, '.'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omegaconf import OmegaConf
from src.encoder.autoencoder2 import fMRI_Encoder
from prism_v2 import PrismV2, prism_v2_losses

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
p.add_argument('--lam_tok', type=float, default=1.0); p.add_argument('--lam_proto', type=float, default=1.0); p.add_argument('--lam_geo', type=float, default=1.0)
p.add_argument('--lam_var', type=float, default=1.0); p.add_argument('--lam_subj', type=float, default=0.1)
p.add_argument('--print_every', type=int, default=20); p.add_argument('--val_every', type=int, default=1000); p.add_argument('--save_every', type=int, default=2000)
p.add_argument('--limit', type=int, default=0); p.add_argument('--resume', action='store_true')
p.add_argument('--probe_grad', type=int, default=0)     # >0: 학습 대신 loss별 gradient 크기만 재고 끝 (배치 수)
p.add_argument('--probe_ckpt', default='')               # 비우면 초기값만, 주면 초기값 + 이 체크포인트 둘 다
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


class ShapeSet(Dataset):
    def __init__(self, split):
        ids = [l.strip() for l in open(os.path.join(a.data_root, 'annotations', f'core_{split}_list.txt')) if l.strip()]
        ids = [k for k in ids if os.path.exists(f'{a.tgt}/dino/{k.split("/")[1]}.npy') and os.path.exists(f'{a.tgt}/voxels/{k.split("/")[1]}.npy')]
        if a.limit: ids = ids[:a.limit]
        self.items = [(si, k) for si, s in enumerate(SUBS) for k in ids
                      if os.path.exists(os.path.join(a.data_root, s, 'npy_data', k + '.npy'))]
        print(f'[{split}] {len(ids)} objects × {len(SUBS)} subjects = {len(self.items)}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        si, k = self.items[i]; syn, uid = k.split('/')
        x = np.load(os.path.join(a.data_root, SUBS[si], 'npy_data', k + '.npy')).astype(np.float32)[FRAMES]
        t = np.load(f'{a.tgt}/targets/{uid}.npz'); vf = [min(191, round(19.2 * f)) for f in FRAMES]
        vox = np.unpackbits(np.load(f'{a.tgt}/voxels/{uid}.npy'))[:64 ** 3].reshape(64, 64, 64)
        return {'fmri': torch.from_numpy(x), 'frame_idx': torch.tensor(FRAMES), 'cls': SYN.index(syn), 'subj': si,
                'cond_tokens': torch.from_numpy(np.load(f'{a.tgt}/dino/{uid}.npy')),
                'voxels': torch.from_numpy(vox.astype(np.uint8)),
                'center': torch.from_numpy(np.stack([t['cx'][vf], t['cy'][vf]], -1)), 'scale': torch.from_numpy(t['scale'][vf]),
                'rot': torch.from_numpy(np.stack([rot_target(float(t['az_deg'][j])) for j in vf])), 'name': k}


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
enc = fMRI_Encoder(cfg.model.params.fmri_encoder_config)
sd = torch.load(a.init_ckpt, map_location='cpu'); sd = sd.get('model', sd)
r = enc.load_state_dict({k[len('fmri_encoder.'):]: v for k, v in sd.items() if k.startswith('fmri_encoder.')}, strict=False)
print(f'[init] encoder from {a.init_ckpt}: missing {len(r.missing_keys)} unexpected {len(r.unexpected_keys)}', flush=True)
del sd
model = PrismV2(enc, n_classes=len(SYN), n_subjects=len(SUBS)).to(dev)
model.prototypes.copy_(build_prototypes())
params = [q for q in model.parameters() if q.requires_grad]
print(f'trainable {sum(q.numel() for q in params) / 1e6:.1f}M, subjects {SUBS}, bs {a.bs}, steps {a.max_step}', flush=True)
opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.max_step))))

tl = DataLoader(ShapeSet('train'), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
vl = DataLoader(ShapeSet('test'), batch_size=a.bs, shuffle=False, num_workers=a.workers)
step = 0; latest = f'{a.out_dir}/model_latest.pt'
if a.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu'); model.load_state_dict(ck['model'], strict=False)
    opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); step = ck['step']; print('resumed', step, flush=True)
W = dict(tok=a.lam_tok, proto=a.lam_proto, geo=a.lam_geo, var=a.lam_var, subj=a.lam_subj)
log = open(f'{a.out_dir}/train_log.csv', 'a')
if os.path.getsize(f'{a.out_dir}/train_log.csv') == 0:
    log.write('step,split,total,tok_mse,tok_cos,proto,proto_acc,geo,center,scale,rot,subj,subj_acc,lr\n')


def to_dev(b):
    out = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
    out['prototypes'] = model.prototypes; return out


def total_loss(L):
    return W['tok'] * (L['tok_mse'] + L['tok_cos']) + W['proto'] * L['proto'] + W['geo'] * L['geo'] + \
        W['var'] * (L['center'] + L['scale'] + L['rot']) + W['subj'] * L['subj']


def write(split, L, tot):
    keys = ['tok_mse', 'tok_cos', 'proto', 'proto_acc', 'geo', 'center', 'scale', 'rot', 'subj', 'subj_acc']
    log.write(f'{step},{split},{tot:.5f},' + ','.join(f'{float(L[k]):.5f}' for k in keys) + f',{sched.get_last_lr()[0]:.3e}\n'); log.flush()


@torch.no_grad()
def validate():
    model.eval(); acc = {}
    for b in vl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'])
        L = prism_v2_losses(out, b)
        for k, v in L.items(): acc.setdefault(k, []).append(float(v))
    L = {k: float(np.mean(v)) for k, v in acc.items()}; tot = float(total_loss(L)); write('val', L, tot)
    print(f'[val] step {step} total {tot:.4f} tok_cos {L["tok_cos"]:.4f} proto_acc {L["proto_acc"]:.3f} geo {L["geo"]:.4f} '
          f'rot {L["rot"]:.4f} subj_acc {L["subj_acc"]:.3f}', flush=True)
    model.train()



if a.probe_grad:   # loss별로 SV Encoder·P에 도착하는 gradient 크기 (가중치 λ 곱한 값 기준, 학습 안 함)
    import json
    groups = {'sv_encoder': list(model.shared.parameters()) + list(model.frame_emb.parameters()), 'P': list(model.P.parameters())}
    terms = {'tok': lambda L: W['tok'] * (L['tok_mse'] + L['tok_cos']), 'sem': lambda L: W['proto'] * L['proto'],
             'geo': lambda L: W['geo'] * L['geo'], 'var': lambda L: W['var'] * (L['center'] + L['scale'] + L['rot']),
             'subj': lambda L: W['subj'] * L['subj']}
    res = {}
    for tag in ['init'] + (['trained'] if a.probe_ckpt else []):
        if tag == 'trained':
            model.load_state_dict(torch.load(a.probe_ckpt, map_location='cpu')['model'], strict=False); model.to(dev)
        model.train(); acc = {}; it = iter(tl)
        for i in range(a.probe_grad):
            b = to_dev(next(it))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = model(b['fmri'], b['frame_idx'])
            L = prism_v2_losses(out, b)
            for name, f in terms.items():
                v = f(L); acc.setdefault((name, 'value'), []).append(float(v))
                for gname, ps in groups.items():
                    gr = torch.autograd.grad(v, ps, retain_graph=True, allow_unused=True)
                    n = math.sqrt(sum(float((x.float() ** 2).sum()) for x in gr if x is not None))
                    acc.setdefault((name, gname), []).append(n)
        res[tag] = {f'{k[0]}/{k[1]}': float(np.mean(v)) for k, v in acc.items()}
        print(f'[probe {tag}] {a.probe_grad} batches  (loss value · |grad| at SV Encoder · |grad| at P)', flush=True)
        for name in terms:
            print(f'  {name:5s}  value {res[tag][name + "/value"]:.4f}  sv_encoder {res[tag][name + "/sv_encoder"]:.4e}  P {res[tag][name + "/P"]:.4e}', flush=True)
    json.dump(res, open(f'{a.out_dir}/grad_probe.json', 'w'), indent=1); print('PROBE_DONE', flush=True); sys.exit(0)

t0 = time.time(); model.train(); done = step >= a.max_step; ep = step // max(1, len(tl))
while not done:
    ep += 1
    for b in tl:
        b = to_dev(b)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = model(b['fmri'], b['frame_idx'])
        L = prism_v2_losses(out, b); tot = total_loss(L)
        opt.zero_grad(set_to_none=True); tot.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); step += 1
        if step % a.print_every == 0:
            write('train', L, float(tot))
            print(f'[ep {ep}] step {step}/{a.max_step} total {float(tot):.4f} tok_mse {float(L["tok_mse"]):.4f} tok_cos {float(L["tok_cos"]):.4f} '
                  f'proto {float(L["proto"]):.3f} proto_acc {float(L["proto_acc"]):.3f} geo {float(L["geo"]):.4f} rot {float(L["rot"]):.4f} '
                  f'subj_acc {float(L["subj_acc"]):.3f} lr {sched.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB {time.time() - t0:.0f}s', flush=True)
        if step % a.val_every == 0: validate()
        if step % a.save_every == 0 or step >= a.max_step:
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'step': step, 'args': vars(a)}, latest + '.tmp')
            os.replace(latest + '.tmp', latest); print(f'saved step {step}', flush=True)
        if step >= a.max_step: done = True; break
validate()
torch.save({'model': model.state_dict(), 'step': step, 'args': vars(a), 'subs': SUBS, 'classes': SYN}, f'{a.out_dir}/model_final.pt')
print(f'TRAIN_DONE step {step} {time.time() - t0:.0f}s', flush=True)
