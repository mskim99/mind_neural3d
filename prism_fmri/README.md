# PRISM fMRI판

박사님 10-27 구조도의 분리부를 그대로 구현 (11-02). 앞단은 MinD-3D++ fMRI ViT(고정), 재구성은 TRELLIS, 형태 감독 없음.

| 성분 | 구현 (`prism.py`) |
|---|---|
| 앞단 | ViT 토큰 + 프레임 임베딩 → Semantic-Variation Encoder (ResMLP×2) → Mixed Features → Orthogonal Projection |
| Semantic | 패치 16×16 격자에서 ConvNeXt식 residual encoder (Down Stage → ConvNeXt Block×2 → Residual Encoder) + 학습 감독(CLIP 범주 텍스트 프로토타입 CE) |
| Variational | 프레임마다 Center(tanh) / Scale(softplus) / Rotate(6D) head (MLP×3 → LayerNorm → Linear) → Aggregate |
| Perceptive | Token Pooling → Subject Projector (Linear-BN-GELU) → FiLM + 피험자 분류(보조) |
| 재구성 | [Semantic ; Variational] Features → 조건 토큰 디코더 → TRELLIS 조건 토큰 [1374, 1024] → TRELLIS |

## 실행 (Pro6000, `scripts/`)
1. `dino.sh` — 자극 영상 프레임 → TRELLIS 조건 토큰 정답 (meshgen venv)
2. `train.sh` — 학습 (`STEPS`, `BS`, `TAG`, `EXTRA`) → `$W/prism_shape$TAG`, 로그 `logs/prism_shape$TAG.log`
3. `infer.sh` — test fMRI → 조건 토큰 → TRELLIS GLB (`PERC=keep|mean`) → `$W/prism_infer$TAG`
4. `eval.sh` — Shape test 104 렌더·지표 (`M0=1` 권장: metallic 0 렌더)

전처리(로컬): `prep_shape_targets.py` (영상 방위·center·scale, 영상 프레임). 정답 폴더는 서버의 `$W/data/prism_v2_shape` 그대로 씀.
