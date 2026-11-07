# PRISM fMRI판

박사님 구조도의 분리부를 구현 (11-02 첫 구현, 11-06 박사님 새 구조도 반영). 앞단은 MinD-3D++ fMRI ViT(고정, 피험자별), 재구성은 Tri-plane 대신 TRELLIS.
구조도 그림: `EEG/.보고 자료/261112_9차보고/PRISM_구조도.png`

| 성분 | 구현 (`prism.py`, 기본값 `--arch fig --sem_clip front --var_heads sr --var_mode feat`) |
|---|---|
| 앞단 | ViT(프레임마다, 고정) → Temporal Stem (LayerNorm → + Frame Emb. → DWConv1D(프레임 축) → GELU → Linear) → SV Encoder (ResMLP → LayerNorm → Linear → GELU → Linear) → Mixed Features → 채널 분할 S 384 / V 384 / P 256 |
| Semantic | 패치 토큰 → CLIP Projection(→ 1024, 13범주 CLIP 텍스트 프로토타입 CE) → 16×16 격자 → Down Stage → ConvNeXt Block×2 → Res Encoder |
| Variational | 프레임마다 [CLS; 패치 평균] → Scale(softplus) · Ratio(3D 크기 비율, sigmoid) head (MLP×3 → LayerNorm → Linear) → Aggregate |
| Perceptive | Token Pooling → Subject Projector (Linear-BN-GELU) → FiLM γ, β + 피험자 분류(보조) |
| 재구성 | Ⓒ[Semantic ; Variational] → FiLM ⊙(1+γ)+β → Transformer Decoder (learned query 1374, Cross-Attention K·V) → TRELLIS 조건 토큰 [1374, 1024] → TRELLIS |

이전 값(`--arch res --sem_clip back --var_heads csr`)은 기준선 `prism_nop_feat` · `prism_nop_value`를 읽고 다시 학습하려고 남김 (`scripts/run_var_cmp.sh`).

## 실행 (Pro6000, 코드는 huijong 브랜치 → 서버 `git pull`)
- `scripts/common.sh` — 공통 경로 · guard (모든 스크립트가 source)
- `scripts/dino.sh` — 자극 영상 프레임 → TRELLIS 조건 토큰 정답 (meshgen venv)
- `scripts/run_fig.sh` — 지금 구조 학습(7명, bs 14, 4만 스텝) → sub-0001 GLB → 8지표 (`spp_eval/run_spp.sh`) · 진행 `scripts/status_fig.sh`
- `scripts/run_var_cmp.sh` · `run_var_cmp7.sh` — Variational feat vs value 비교 (학습 · sub-0001 / 나머지 6명 평가)
- `scripts/run_tokfix.sh` — 예측 토큰 → train 정답 토큰 공간 (kNN · PCA, 추론만)
- `scripts/run_renorm.sh` — 색 실험 1: 예측 토큰 LN 재적용(+ 편차 ×1.37 · ×1.6) 스윕 + top-3 범주 · 범주로 거른 kNN 일치 → 정성 14 · 8지표 (추론만)
- `scripts/run_purple.sh` — 색 실험 1 후속: 보라 치우침 출처 (평균 토큰 · 예측 평균 · 공통 치우침 뺀 예측 렌더)
- 분석: `eval_pose.py` (Variational 출력 vs GT), `probe_color.py` (색 선형 프로브), `probe_split.py` (분기 분리 프로브), `tokfix.py`, `renorm.py`, `purple_check.py`

전처리(로컬): `prep_shape_targets.py` (영상 방위·center·scale, 영상 프레임, 64³ 복셀). 정답 폴더는 서버의 `$W/data/shape_targets`.
