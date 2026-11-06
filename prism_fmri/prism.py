# PRISM 모델 — 박사님 10-27 구조도(.참고자료/261027_구조도.png)의 분리부를 그림대로 구현 (11-02)
#  형태 감독(Geometry head) 없음. 각 성분 head의 출력(Semantic·Variational Features)이 그대로 재구성 입력
#  fMRI 6프레임 → [Spatio-Temporal] MinD-3D++ fMRI ViT(고정, 프레임마다 공간) + Temporal Stem(학습, 패치 위치마다 프레임 축 1D conv)
#    → Semantic-Variation Encoder(학습, + 프레임 임베딩) → Mixed Features
#    → 채널 분할 S 384 / V 384 / P 256 (11-05, 학습 P 없음) → [Semantic Routing | Variation Routing | Perceptive Filtering]
#      Semantic    : 패치 16×16 격자에서 ConvNeXt식 residual encoder (Down Stage → ConvNeXt Block×N → Residual Encoder) → Semantic Features
#                    (+ 학습 감독: 풀링 → CLIP projection → 범주 텍스트 프로토타입 CE)
#      Variational : 프레임마다 Center / Scale / Rotate head (MLP×3 → LayerNorm → Linear → tanh / softplus / 6D) → Aggregate → Variational Features
#      Perceptive  : Token Pooling → Subject Projector (Linear-BN-GELU) → Perceptive Features (+ 피험자 분류 보조)
#    → 재구성: 구조도의 Tri-plane 대신 TRELLIS — [Semantic ; Variational] Features를 Perceptive(FiLM)로 보정 → 조건 토큰 디코더 → TRELLIS 조건 토큰 [1374, 1024]
#  구조도에 없어 정한 것: ConvNeXt 블록 수(2)·채널, head 은닉 256, Aggregate = 세 head 특징(LayerNorm 출력) 이어 붙여 Linear, Perceptive 입력 방식 FiLM,
#    Temporal Stem — 구조도엔 이름만 있어 박사님 EEG 코드(src/mvdiffusion_var_semantic_cls_sg.py TemporalStem: 채널마다 시간축 1D conv + BN + GELU)를
#    fMRI에 옮김: 채널마다(depthwise) 프레임 축 conv(커널 3, 6프레임 유지) + GELU + Linear, 잔차. 마지막 Linear 0 초기화(처음엔 ViT 특징 그대로).
#    EEG는 시간 → 공간 순서지만, 여기선 공간(ViT)이 고정·사전학습이라 ViT 뒤에 둠. temporal=False면 이전 체크포인트와 같은 구조
#  성분 분할 (11-05): 학습 직교 P를 뺌. Mixed Features를 채널 384 / 384 / 256으로 바로 자름 (박사님 11월 그림, 회전은 SV Encoder가 흡수).
#    P가 있던 이전 체크포인트(7명 기준선 prism_shape_ps7 등)는 이 코드로 읽지 않음 (지표는 이미 있음)
#  Variational 입력 (11-05 비교 실험): var_mode='feat' = head의 LayerNorm 특징 3×256 → Aggregate (기본, 이전과 같음)
#    var_mode='value' = head가 낸 실제 값 [center 2, scale 1, 회전 행렬 9] = 12 → MLP 임베딩 → 768 (박사님 구조도처럼 예측 값을 씀)
#  피험자 (11-04): per_subject_sv=True면 ViT · Semantic · Variational은 피험자별(자기 피험자 데이터로만 학습),
#    Temporal Stem · SV Encoder · P · Perceptive(FiLM) · 디코더는 전 피험자 공유. False면 이전 구조(전부 공유, ViT 하나)
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n = nn.LayerNorm(d); self.f1 = nn.Linear(d, h); self.f2 = nn.Linear(h, d)
    def forward(self, x):
        return x + self.f2(F.gelu(self.f1(self.n(x))))


def rot6d_to_matrix(x):
    """[..., 6] → 회전 행렬 [..., 3, 3] (Zhou et al. 2019, 앞 두 열을 Gram-Schmidt)."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-1)


class LN2d(nn.Module):
    """채널 LayerNorm ([B, C, H, W])."""
    def __init__(self, c):
        super().__init__(); self.n = nn.LayerNorm(c)
    def forward(self, x):
        return self.n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """구조도 ConvNeXt Block: SimConv(깊이별 7×7) → LayerNorm → FFN(4배, GELU) + 잔차."""
    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c); self.n = nn.LayerNorm(c)
        self.f1 = nn.Linear(c, 4 * c); self.f2 = nn.Linear(4 * c, c)
    def forward(self, x):
        h = self.n(self.dw(x).permute(0, 2, 3, 1))
        return x + self.f2(F.gelu(self.f1(h))).permute(0, 3, 1, 2)


class SemanticEncoder(nn.Module):
    """구조도 Semantic: Down Stage(16×16 → 8×8) → ConvNeXt Block × N → Residual Encoder(8×8 → 16×16, 입력 잔차)."""
    def __init__(self, cin, c=512, n_blocks=2, cout=768):
        super().__init__()
        self.down = nn.Sequential(nn.Conv2d(cin, c, 2, stride=2), LN2d(c))
        self.blocks = nn.Sequential(*[ConvNeXtBlock(c) for _ in range(n_blocks)])
        self.up = nn.Sequential(nn.ConvTranspose2d(c, cout, 2, stride=2), LN2d(cout))
        self.skip = nn.Conv2d(cin, cout, 1)
    def forward(self, x):                                   # [B, cin, 16, 16] → [B, cout, 16, 16]
        return self.up(self.blocks(self.down(x))) + self.skip(x)


class PoseHead(nn.Module):
    """구조도 Center/Scale/Rotate head: MLP×3 → LayerNorm → Linear (+ 활성은 밖에서). 특징(LayerNorm 출력)도 돌려줌."""
    def __init__(self, d, k, h=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.n = nn.LayerNorm(h); self.out = nn.Linear(h, k)
    def forward(self, x):
        f = self.n(self.mlp(x)); return self.out(f), f


class TemporalStem(nn.Module):
    """[B, T, N, D] → 같은 크기. 패치 위치(N)·채널(D)마다 따로 프레임 축(T) 1D conv (박사님 EEG TemporalStem의 fMRI판), 잔차."""
    def __init__(self, d, k=3):
        super().__init__()
        self.n = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.out = nn.Linear(d, d); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, x):
        B, T, N, D = x.shape
        h = self.n(x).permute(0, 2, 3, 1).reshape(B * N, D, T)                  # 패치 위치마다 [D, T]
        h = F.gelu(self.conv(h)).reshape(B, N, D, T).permute(0, 3, 1, 2)        # [B, T, N, D]
        return x + self.out(h)


SUBJECT_KEYS = ('sem_enc', 'sem_proj', 'center', 'scale', 'rotate', 'aggregate')   # 피험자별로 따로 두는 부품 (Semantic · Variational)


def remap_old_keys(sd):
    """피험자별 구조 이전 체크포인트(encoder 하나, S·V 한 벌) → 지금 키 이름 (encoders.0. / sem_enc.0. …)."""
    if any(k.startswith('sem_enc.0.') for k in sd):             # 이미 지금 형식
        return dict(sd)
    out = {}
    for k, v in sd.items():
        if k.startswith('encoder.'):
            k = 'encoders.0.' + k[len('encoder.'):]
        else:
            for n in SUBJECT_KEYS:
                if k.startswith(n + '.'):
                    k = f'{n}.0.' + k[len(n) + 1:]; break
        out[k] = v
    return out


def _route(mods, idx, x):
    """샘플마다 자기 피험자 모듈로 보냄. mods: 모듈 리스트, idx: [B] 피험자 번호. 출력이 튜플이어도 됨."""
    if len(mods) == 1:
        return mods[0](x)
    outs, order = [], []
    for s in idx.unique().tolist():
        m = (idx == s).nonzero(as_tuple=True)[0]; order.append(m); outs.append(mods[s](x[m]))
    order = torch.cat(order); inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[t] for o in outs])[inv] for t in range(len(outs[0])))
    return torch.cat(outs)[inv]


class Prism(nn.Module):
    """encoder: fMRI ViT 하나(피험자 공통, 이전 방식) 또는 피험자별 리스트.
    per_subject_sv=True면 Semantic · Variational을 피험자마다 따로 둠 (각자 자기 피험자 데이터로만 학습).
    공유: Temporal Stem · Semantic-Variation Encoder · 직교 투영 P · Perceptive(Subject Projector · 피험자 분류 · FiLM) · 조건 토큰 디코더."""
    def __init__(self, encoder, n_classes, n_subjects, d=1024, splits=(384, 384, 256), dc=768, n_query=1374, dec_layers=2, n_frames=10,
                 sem_ch=512, sem_blocks=2, head_dim=256, temporal=True, per_subject_sv=False, var_mode='feat'):
        super().__init__()
        assert sum(splits) == d
        encs = list(encoder) if isinstance(encoder, (list, tuple, nn.ModuleList)) else [encoder]
        self.encoders = nn.ModuleList([e.requires_grad_(False).eval() for e in encs])   # MinD-3D++ fMRI ViT (패치 16, 256×256 → 16×16 + CLS), 고정
        self.enc_grad = False                                          # True: 입력까지 기울기 (성분 지도용)
        K = n_subjects if per_subject_sv else 1
        self.splits = splits
        ds, dv, dp = splits
        self.frame_emb = nn.Embedding(n_frames, d)
        self.temporal = TemporalStem(d) if temporal else None              # Spatio-Temporal의 시간 부분
        self.shared = nn.Sequential(ResMLP(d, 2 * d), ResMLP(d, 2 * d))   # Semantic-Variation Encoder
        # Semantic (피험자별 K벌)
        self.sem_enc = nn.ModuleList([SemanticEncoder(ds, sem_ch, sem_blocks, dc) for _ in range(K)])
        self.sem_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024), nn.GELU(), nn.Linear(1024, 1024)) for _ in range(K)])   # 학습 감독용
        self.register_buffer('prototypes', torch.zeros(n_classes, 1024))
        # Variational (피험자별 K벌)
        self.center = nn.ModuleList([PoseHead(2 * dv, 2, head_dim) for _ in range(K)])
        self.scale = nn.ModuleList([PoseHead(2 * dv, 1, head_dim) for _ in range(K)])
        self.rotate = nn.ModuleList([PoseHead(2 * dv, 6, head_dim) for _ in range(K)])
        assert var_mode in ('feat', 'value'); self.var_mode = var_mode
        agg_in = 3 * head_dim if var_mode == 'feat' else 12
        self.aggregate = nn.ModuleList([nn.Sequential(nn.Linear(agg_in, dc), nn.GELU(), nn.LayerNorm(dc)) if var_mode == 'feat' else
                                        nn.Sequential(nn.Linear(agg_in, head_dim), nn.GELU(), nn.Linear(head_dim, dc), nn.GELU(), nn.LayerNorm(dc)) for _ in range(K)])
        # Perceptive (공유)
        self.subj_proj = nn.Sequential(nn.Linear(dp, 256), nn.BatchNorm1d(256), nn.GELU())
        self.subj_cls = nn.Linear(256, max(n_subjects, 1))
        self.film = nn.Linear(256, 2 * dc); nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)   # 처음엔 보정 없음
        # 재구성 (TRELLIS 조건 토큰, 공유)
        self.type_emb = nn.Parameter(torch.zeros(2, dc))                 # Semantic / Variational 토큰 구분
        self.time_emb = nn.Embedding(n_frames, dc)
        self.query = nn.Parameter(torch.randn(1, n_query, dc) * 0.02)
        layer = nn.TransformerDecoderLayer(dc, 12, 4 * dc, dropout=0.0, batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, dec_layers)
        self.tok_out = nn.Sequential(nn.LayerNorm(dc), nn.Linear(dc, 1024))

    def train(self, mode=True):
        super().train(mode); self.encoders.eval(); return self

    def trainable_state_dict(self):
        """저장용: 고정 ViT(피험자마다 약 1 GB)는 빼고 저장. 불러올 때 ViT는 enc_paths에서 다시 읽음."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith('encoders.')}

    def _vit(self, e, x):
        B, T = x.shape[:2]
        return e.forward_encoder_wo_pred(x.reshape(B * T, 1, 256, 256)).view(B, T, 257, -1)

    def encode_frames(self, fmri, subj):
        """fmri [B, T, 256, 256] → ViT 토큰 [B, T, 257, 1024] (피험자마다 자기 ViT)."""
        with torch.set_grad_enabled(self.enc_grad and torch.is_grad_enabled()):
            if len(self.encoders) == 1:
                return self._vit(self.encoders[0], fmri)
            outs, order = [], []
            for s in subj.unique().tolist():
                m = (subj == s).nonzero(as_tuple=True)[0]; order.append(m); outs.append(self._vit(self.encoders[s], fmri[m]))
            order = torch.cat(order); inv = torch.empty_like(order); inv[order] = torch.arange(len(order), device=order.device)
            return torch.cat(outs)[inv]

    def forward(self, fmri, frame_idx, subj=None, perceptive_override=None):
        """fmri [B, T, 256, 256], frame_idx [B, T] (fMRI 프레임 번호, 0~9), subj [B] 피험자 번호 (없으면 0).
        perceptive_override: 평가 때 Perceptive 벡터를 바꿔 끼우기 ('mean' → 배치 평균 = 사실상 버리기, 텐서 → 그 값)."""
        if subj is None:
            subj = torch.zeros(fmri.shape[0], dtype=torch.long, device=fmri.device)
        tok = self.encode_frames(fmri, subj).float()
        B, T, N, D = tok.shape
        x = tok + self.frame_emb(frame_idx)[:, :, None, :]
        if self.temporal is not None:
            x = self.temporal(x)                                                     # 프레임 사이 정보 교환
        mixed = self.shared(x)                                                       # Mixed Features [B, T, 257, D]
        s, v, p = mixed.split(self.splits, dim=-1)                                  # 채널 분할 (학습 파라미터 없음)
        out = {}
        # Semantic (피험자별): 프레임 평균(정적 성분) → 패치 16×16 격자 → ConvNeXt식 encoder → Semantic Features [B, 256, dc]
        sg = s[:, :, 1:].mean(1).transpose(1, 2).reshape(B, -1, 16, 16)
        sem_feat = _route(self.sem_enc, subj, sg).flatten(2).transpose(1, 2)
        out['sem_vec'] = sem_feat.mean(1)
        out['sem_clip'] = F.normalize(_route(self.sem_proj, subj, out['sem_vec']), dim=-1)
        # Variational (피험자별): 프레임마다 [CLS; 패치 평균] → Center / Scale / Rotate → Aggregate → Variational Features [B, T, dc]
        vin = torch.cat([v[:, :, 0], v[:, :, 1:].mean(2)], -1)
        c, fc = _route(self.center, subj, vin); sc, fs = _route(self.scale, subj, vin); r, fr = _route(self.rotate, subj, vin)
        out['center'] = torch.tanh(c); out['scale'] = F.softplus(sc).squeeze(-1); out['rot6d'] = r
        if self.var_mode == 'feat':
            agg_in = torch.cat([fc, fs, fr], -1)                                     # LayerNorm 특징 3×256
        else:
            agg_in = torch.cat([out['center'], out['scale'][..., None], rot6d_to_matrix(r).flatten(-2)], -1)   # 실제 값 2+1+9
        var_feat = _route(self.aggregate, subj, agg_in)
        out['var_vec'] = var_feat.mean(1)
        # Perceptive (공유): Token Pooling → Subject Projector
        pv = p.mean((1, 2))
        if perceptive_override is not None:
            pv = pv.mean(0, keepdim=True).expand_as(pv) if isinstance(perceptive_override, str) else perceptive_override
        h = self.subj_proj(pv)
        out['subj_logit'] = self.subj_cls(h); out['perc_vec'] = pv
        # 재구성 (공유): [Semantic ; Variational] Features (Perceptive FiLM 보정) → 조건 토큰
        mem = torch.cat([sem_feat + self.type_emb[0], var_feat + self.type_emb[1] + self.time_emb(frame_idx)], 1)   # [B, 256 + T, dc]
        gamma, beta = self.film(h).chunk(2, -1)
        mem = mem * (1 + gamma[:, None]) + beta[:, None]
        out['cond_tokens'] = self.tok_out(self.dec(self.query.expand(B, -1, -1), mem))   # [B, 1374, 1024]
        return out


def load_encoder(path, cfg):
    """MinD-3D++ 가중치(.pt, 전체 또는 인코더만) → fMRI_Encoder (fmri_encoder.* 키). 실행 위치 $W/run (src 필요)."""
    from src.encoder.autoencoder2 import fMRI_Encoder
    e = fMRI_Encoder(cfg.model.params.fmri_encoder_config); sd = torch.load(path, map_location='cpu'); sd = sd.get('model', sd)
    r = e.load_state_dict({k[len('fmri_encoder.'):]: v for k, v in sd.items() if k.startswith('fmri_encoder.')}, strict=False)
    assert not r.missing_keys, f'{path}: 인코더 키 {len(r.missing_keys)}개 없음'
    return e


def build_from_ckpt(ck, repo):
    """체크포인트 → Prism. 피험자별 체크포인트는 enc_paths에서 ViT를 읽고, 이전 체크포인트는 키 이름만 바꿔 읽음."""
    import os
    from omegaconf import OmegaConf
    from src.encoder.autoencoder2 import fMRI_Encoder
    cfg = OmegaConf.load(os.path.join(repo, 'configs', 'mind3d_pp.yaml'))
    args = ck.get('args') or {}; per = bool(args.get('per_subject', 0))
    encs = [load_encoder(pth, cfg) for pth in ck['enc_paths']] if per else fMRI_Encoder(cfg.model.params.fmri_encoder_config)
    model = Prism(encs, n_classes=len(ck['classes']), n_subjects=len(ck['subs']), temporal=bool(args.get('temporal', 0)), per_subject_sv=per,
                  var_mode=args.get('var_mode', 'feat'))
    r = model.load_state_dict(remap_old_keys(ck['model']), strict=not per)
    if per:
        assert not r.unexpected_keys and all(k.startswith('encoders.') for k in r.missing_keys), r
    return model


def prism_losses(out, batch, proto_temp=0.07, mean_tok=None, nce_temp=0.07):
    """mean_tok [1374,1024] (train GT 평균 토큰)을 주면 편차 손실도 계산 (11-05 정성 진단):
       dev_cos = 1 − cos(예측 − 평균, GT − 평균) (토큰별), nce = 물체 단위 InfoNCE (토큰 평균 편차, 배치 안 같은 물체는 모두 정답)."""
    L = {}
    out = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}   # bf16 → fp32
    tgt = batch['cond_tokens'].float()
    L['tok_mse'] = F.mse_loss(out['cond_tokens'], tgt)
    L['tok_cos'] = (1 - F.cosine_similarity(out['cond_tokens'], tgt, dim=-1)).mean()
    if mean_tok is not None:
        dp, dg = out['cond_tokens'] - mean_tok, tgt - mean_tok
        L['dev_cos'] = (1 - F.cosine_similarity(dp, dg, dim=-1)).mean()
        zp, zg = F.normalize(dp.mean(1), dim=-1), F.normalize(dg.mean(1), dim=-1)
        names = batch['name']; same = torch.tensor([[x == y for y in names] for x in names], dtype=torch.float32, device=zp.device)
        logp = F.log_softmax(zp @ zg.t() / nce_temp, -1)
        L['nce'] = -((logp * same).sum(-1) / same.sum(-1)).mean()
    else:
        L['dev_cos'] = torch.zeros((), device=tgt.device); L['nce'] = torch.zeros((), device=tgt.device)
    logits = out['sem_clip'] @ batch['prototypes'].t() / proto_temp
    L['proto'] = F.cross_entropy(logits, batch['cls']); L['proto_acc'] = (logits.argmax(-1) == batch['cls']).float().mean()
    L['center'] = F.mse_loss(out['center'], batch['center']); L['scale'] = F.mse_loss(out['scale'], batch['scale'])
    L['rot'] = F.mse_loss(rot6d_to_matrix(out['rot6d']), batch['rot'])
    if out['subj_logit'].shape[-1] > 1:
        L['subj'] = F.cross_entropy(out['subj_logit'], batch['subj']); L['subj_acc'] = (out['subj_logit'].argmax(-1) == batch['subj']).float().mean()
    else:
        L['subj'] = out['subj_logit'].sum() * 0; L['subj_acc'] = torch.zeros(())
    return L
