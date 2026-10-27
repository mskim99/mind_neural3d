# PRISM fMRI판 학습 (피험자별, GPU 1장) — Pro6000 MinD-3D++ 재학습 스크립트(train_mind3dpp_ours.py, 10-12 판)를 바탕으로 바꾼 것만:
#  ① 모델: PrismMVDiffusion (prism_model.py) — 의미/변형/피험자 헤드, 의미 ProtoCE, 의미 경로 확산 gradient 차단
#  ② 초기값: --init_ckpt (MinD-3D++ 공개 가중치 sub01.pt 등)에서 인코더·LoRA·logit scale을 불러옴. 새 헤드는 항등으로 시작
#  ③ 데이터에 범주 번호 'cls' 추가 (학습 리스트의 범주를 정렬한 순서), 범주 이름 → OpenCLIP 텍스트 프로토타입
#  ④ 손실 = 확산 + λ_clip·CLIP 대조(공식) + λ_proto·ProtoCE + λ_ortho·직교(EEG판과 같음, 기본 0 = 끔)
#  ④-2 --stage1_steps N: EEG판 2단계 학습. 처음 N 스텝은 확산 없이 ProtoCE + 직교만(의미·변형·피험자 헤드, CLIP projection),
#      그다음 joint(확산 포함). 피험자 헤드는 joint에서 고정 (EEG판 bias_filtering과 같음). 0이면 joint만
#  ⑤ 기본 스텝 수를 짧게(--max_step 기본 5000): 공개 가중치에서 시작하는 미세조정이라
#  ⑥ --freeze_encoder: 인코더 ViT 본체 고정 (GPU 1을 다른 학습과 나눠 쓸 때, 4090 24 GB용)
#  학습 GT: 자극 영상 프레임 (Pro6000 data/GTviews_video/obj, 로컬 DataSet/MinD-3D++/GT_views_train) — 학습 정답은 피험자가 본 자극 안에서만
# 실행 (cwd = pretrained_models/ 와 src → MinD-3D/src 링크가 있는 작업 폴더, 예: /home/oem/jhj/MinD-3D++/run):
#   torchrun --nproc_per_node=1 <이 파일> --dataset obj --repo .../MinD-3D --data_root ... --image_root ... --z123 ... --init_ckpt .../sub01.pt --out_dir ...
import os, sys, time, random, argparse
import numpy as np, torch
from torch import distributed as dist
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode

p = argparse.ArgumentParser()
p.add_argument('--dataset', choices=['shape', 'obj'], required=True)
p.add_argument('--repo', required=True)                 # MinD-3D 저장소 (configs/mind3d_pp.yaml)
p.add_argument('--data_root', required=True)            # shape: fMRI-Shape 폴더 / obj: fMRI-Objaverse 폴더
p.add_argument('--image_root', required=True)           # 학습 GT 6뷰: shape <cls>/<uid>/{0..5}.png, obj <hash>.glb/{0..5}.png (RGBA)
p.add_argument('--sub_id', default='sub-0001')
p.add_argument('--z123', required=True)                 # zero123plus-v1.2 로컬 폴더
p.add_argument('--init_ckpt', default='')               # MinD-3D++ 공개 가중치 (비우면 LEA 사전학습 + 흰 배경 UNet에서 시작 = 처음부터)
p.add_argument('--out_dir', required=True)
p.add_argument('--bs', type=int, default=7)
p.add_argument('--accum', type=int, default=4)
p.add_argument('--max_step', type=int, default=5000)    # optimizer 스텝
p.add_argument('--lr', type=float, default=1e-4)
p.add_argument('--warmup', type=int, default=300)
p.add_argument('--lambda_clip', type=float, default=1.0)
p.add_argument('--lambda_proto', type=float, default=1.0)
p.add_argument('--proto_temp', type=float, default=0.07)
p.add_argument('--lambda_ortho', type=float, default=0.0)
p.add_argument('--stage1_steps', type=int, default=0)
p.add_argument('--freeze_encoder', action='store_true')  # fMRI 인코더 ViT 본체 고정 (공개 가중치 그대로) → 헤드·aggregation·LoRA만 학습, 메모리 절약
p.add_argument('--no_sem_detach', action='store_true')  # 켜면 확산 손실 gradient가 의미 경로로도 흐름 (EEG판은 차단)
p.add_argument('--save_every', type=int, default=1000)
p.add_argument('--print_every', type=int, default=10)
p.add_argument('--workers', type=int, default=8)
p.add_argument('--resume', action='store_true')
p.add_argument('--grad_ckpt', action='store_true')
p.add_argument('--check_data', type=int, default=0)     # >0이면 데이터 N개만 읽어 형상 출력 후 종료
a = p.parse_args()

dist.init_process_group('nccl'); rank = dist.get_rank(); torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))
random.seed(42 + rank); np.random.seed(42 + rank); torch.manual_seed(42 + rank)
os.makedirs(a.out_dir, exist_ok=True)

SHAPENET_NAME = {"02691156": "Airplane.", "02828884": "Bench.", "02933112": "Cabinet.", "02958343": "Car.", "03001627": "Chair.",
                 "03211117": "Display (Monitor).", "03636649": "Lamp.", "03691459": "Loudspeaker.", "04090263": "Rifle.",
                 "04256520": "Sofa.", "04379243": "Table.", "04401088": "Telephone.", "04530566": "Watercraft (Boat.)"}
CLIP_TF = Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224),
                   Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

def load_im(path, color=(1., 1., 1.)):
    im = np.asarray(Image.open(path), dtype=np.float32) / 255.
    alpha = im[:, :, 3:]; im = im[:, :, :3] * alpha + np.array(color, dtype=np.float32) * (1 - alpha)
    return torch.from_numpy(im).permute(2, 0, 1).contiguous().float()

class Sub(Dataset):
    def __init__(self, mode, classes=None):
        self.mode = mode
        if a.dataset == 'shape':
            lst = [l.strip() for l in open(f'{a.data_root}/annotations/core_{mode}_list.txt') if l.strip()]
            self.items = [(x.split('/')[0], x.split('/')[1]) for x in lst]
            self.frames_train, self.frames_test = list(range(10)), [2, 3, 4, 5, 6, 7]
        else:
            lst = [l.strip().split(',') for l in open(f'{a.data_root}/{mode}_list.txt') if l.strip()][1:]
            self.items = [(c, u.split('/')[-1].replace('.glb', '')) for c, u in lst]
            self.frames_train, self.frames_test = list(range(8)), [1, 2, 3, 4, 5, 6]
            self.caption = {}
            for line in list(open(f'{a.data_root}/Object_caption_Cap3D.csv', encoding='utf-8'))[1:]:
                parts = line.strip().split(',', 2)
                if len(parts) == 3:
                    self.caption[parts[1].split('/')[-1].replace('.glb', '')] = parts[2].strip().strip('"')
        n0 = len(self.items)
        self.items = [(c, u) for c, u in self.items if os.path.exists(self.view(c, u, 5))]
        self.classes = classes or sorted({c for c, _ in self.items})          # 범주 번호 = 학습 리스트 범주 정렬 순서
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        print(f'[{mode}] {len(self.items)} objects (GT 렌더 없는 {n0 - len(self.items)}개 제외), {len(self.classes)} classes', flush=True)
    def __len__(self):
        return len(self.items)
    def npy(self, c, u):
        return f'{a.data_root}/{a.sub_id}/npy_data/{c}/{u}.npy' if a.dataset == 'shape' else f'{a.data_root}/{a.sub_id}/npy_data/{u}.npy'
    def view(self, c, u, i):
        return f'{a.image_root}/{c}/{u}/{i}.png' if a.dataset == 'shape' else f'{a.image_root}/{u}.glb/{i}.png'
    def __getitem__(self, k):
        c, u = self.items[k]
        f = np.load(self.npy(c, u))
        s = np.array(sorted(random.sample(self.frames_train, 6))) if self.mode == 'train' else np.array(self.frames_test)
        f = torch.from_numpy(f[s].astype(np.float32))
        imgs = torch.stack([load_im(self.view(c, u, i)) for i in range(6)])
        clip_img = CLIP_TF(imgs[random.randrange(6)])
        text = SHAPENET_NAME[c] if a.dataset == 'shape' else self.caption.get(u, c.replace('_', ' '))
        return {'target_imgs': imgs, 'fmri': f, 'img': clip_img, 'text': text, 'cls': self.cidx[c]}

def class_text(c):
    return SHAPENET_NAME[c].rstrip('.') if a.dataset == 'shape' else c.replace('_', ' ')

tr = Sub('train')
if a.check_data:
    for k in range(a.check_data):
        d = tr[k]; print({x: (tuple(v.shape), str(v.dtype), round(float(v.mean()), 3)) if torch.is_tensor(v) else v for x, v in d.items()}, flush=True)
    print('classes:', [class_text(c) for c in tr.classes[:10]], '...', flush=True)
    sys.exit(0)
from torch.utils.data.distributed import DistributedSampler
samp = DistributedSampler(tr, shuffle=True, seed=42)
tl = DataLoader(tr, batch_size=a.bs, num_workers=a.workers, drop_last=True, sampler=samp)

sys.path.insert(0, os.getcwd())                                     # 공식 src (cwd/src → MinD-3D/src)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))     # prism_encoder.py, prism_model.py
from prism_model import PrismMVDiffusion
cfg = OmegaConf.load(f'{a.repo}/configs/mind3d_pp.yaml')
sdc = dict(cfg.model.params.stable_diffusion_config); sdc['pretrained_model_name_or_path'] = a.z123
model = PrismMVDiffusion(cfg, sdc, cfg.model.params.fmri_encoder_config, a.out_dir, [class_text(c) for c in tr.classes],
                         lr=a.lr, warmup=a.warmup, total_steps=a.max_step, sem_detach=not a.no_sem_detach, proto_temp=a.proto_temp).cuda()
if a.init_ckpt:
    model.load_init(a.init_ckpt)
if a.freeze_encoder:
    model.set_freeze_encoder(True)
model.build_prototypes()
if a.grad_ckpt:                                                      # train_mind3dpp_ours.py와 같은 패치 (계산 결과 동일)
    from diffusers import UNet2DConditionModel
    pcls = {type(m) for m in model.unet.modules() if type(m).__name__ == 'ReferenceOnlyAttnProc'}
    for c in pcls:
        def _call(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, mode='w', ref_dict=None, is_cfg_guidance=False, _orig=c.__call__):
            return _orig(self, attn, hidden_states, encoder_hidden_states, attention_mask, 'm' if mode == 'r' else mode, ref_dict, is_cfg_guidance)
        c.__call__ = _call
    n = 0
    for m in model.unet.modules():
        if isinstance(m, UNet2DConditionModel): m.enable_gradient_checkpointing(); n += 1
    print(f'grad_ckpt: UNet {n}개 checkpointing, ReferenceOnlyAttnProc 클래스 {len(pcls)}개 패치', flush=True)

def trainable_state():
    return {k: v for k, v in model.state_dict().items()
            if k.startswith(('fmri_encoder.', 'semantic_to_clip.')) or 'lora_' in k or k in ('i_logit_scale', 't_logit_scale', 'cls_prototypes')}

step, it, epoch = 0, 0, 0
latest = f'{a.out_dir}/model_latest.pt'
if a.resume and os.path.exists(latest):
    ck = torch.load(latest, map_location='cpu')
    r = model.load_state_dict(ck['model'], strict=False)
    model.opt.load_state_dict(ck['opt']); model.sche.load_state_dict(ck['sche'])
    step, it, epoch = ck['step'], ck['it'], ck['epoch']
    print(f'resumed step {step} epoch {epoch}; unexpected {len(r.unexpected_keys)}', flush=True)

def save_latest():
    if rank != 0: return
    tmp = latest + '.tmp'
    torch.save({'model': trainable_state(), 'opt': model.opt.state_dict(), 'sche': model.sche.state_dict(), 'step': step, 'it': it, 'epoch': epoch,
                'class_names': model.class_names}, tmp)
    os.replace(tmp, latest); print(f'saved model_latest.pt (step {step})', flush=True)

ntr = sum(x.numel() for x in model.parameters() if x.requires_grad)
if rank == 0:
    print(f'{a.dataset}: trainable {ntr/1e6:.1f}M, world {dist.get_world_size()} x bs {a.bs} x accum {a.accum} = {dist.get_world_size()*a.bs*a.accum}/step, '
          f'stop at step {a.max_step}, init {a.init_ckpt or "LEA(처음부터)"}, sem_detach {not a.no_sem_detach}, freeze_encoder {a.freeze_encoder}, lambda_ortho {a.lambda_ortho}, stage1_steps {a.stage1_steps}', flush=True)
t0 = time.time(); done = step >= a.max_step; stage = None
while not done:
    epoch += 1; samp.set_epoch(epoch)
    for batch in tl:
        model.train(); it += 1
        if step < a.stage1_steps:                                          # 1단계: 확산 없음
            stage = 'S1'; pl, pa, ol = model.semantic_stage_loss(batch); dl = cl = torch.zeros((), device=pl.device)
            ((a.lambda_proto * pl + a.lambda_ortho * ol) / a.accum).backward()
        else:
            if stage != 'J':                                               # joint 시작: 피험자 헤드 고정 (EEG판)
                model.fmri_encoder.subj_head.requires_grad_(False); stage = 'J'
            dl, cl, pl, pa, ol = model.get_loss(batch)
            ((dl + a.lambda_clip * cl + a.lambda_proto * pl + a.lambda_ortho * ol) / a.accum).backward()
        if it % a.accum != 0:
            continue
        model.opt.step(); model.opt.zero_grad(); model.sche.step(); step += 1
        if rank == 0 and step % a.print_every == 0:
            print(f'[ep {epoch}] step {step}/{a.max_step} diff {float(dl):.4f} clip {float(cl):.4f} proto {float(pl):.4f} proto_acc {float(pa):.3f} ortho {float(ol):.4f} {stage} '
                  f'lr {model.sche.get_last_lr()[0]:.2e} mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB {time.time()-t0:.0f}s', flush=True)
        if a.save_every and step % a.save_every == 0 and step < a.max_step:
            save_latest()
        if step >= a.max_step:
            done = True; break
if rank == 0:
    torch.save({'epoch_it': epoch, 'it': step, 'model': model.state_dict(), 'class_names': model.class_names}, f'{a.out_dir}/model_final.pt')
    save_latest()
    print(f'TRAIN_DONE {a.dataset} step {step} epoch {epoch} {time.time()-t0:.0f}s', flush=True)
