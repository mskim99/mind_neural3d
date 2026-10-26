# PRISM fMRI판 (성분 분리 → Zero123++ 직접 조건)

```
fMRI (6프레임)
→ fMRI 인코더 (ViT, MinD-3D++ 공개 가중치)
→ 분리 (Semantic 의미 / Variational 변형 / Perceptive 피험자)  ← 제안 기법의 핵심. 앞뒤는 기존 모델 그대로 (사용자 10-26)
   1. 의미   (CLS)        → zero123++ cross-attention
   2. 변형   (패치 토큰)  → zero123++ cond_lat
   3. 피험자 (CLS)        → 버림 (지금은 손실 없음, 자리만 — 아직 미구현)
→ zero123++ (공개 가중치, 고정) + LoRA (MinD-3D++ 공개 가중치에서 시작)
→ 6View → TRELLIS (제안 기법 3D는 TRELLIS만. InstantMesh는 MinD-3D++ 기준선 전용)
```

기존 fMRI판(캡션 검색 → SDXL)은 그대로 두고, 이 폴더에 새로 추가함.

## 파일
| 파일 | 내용 |
|---|---|
| `prism_encoder.py` | `PrismFMRIEncoder`: MinD-3D++ `fMRI_Encoder` 상속 + 헤드 3개. 의미·변형 헤드는 잔차 MLP(마지막 층 0 초기화) → 처음엔 항등이라 공개 가중치를 넣으면 MinD-3D++와 같은 출력 |
| `prism_model.py` | `PrismMVDiffusion`: 공식 `src/mvdiffusion.MVDiffusion` 상속. 의미 ProtoCE(범주 프로토타입), 의미 경로 확산 gradient 차단(EEG판과 같음), 공식 CLIP 대조 손실 유지, 직교 손실 없음 |
| `train_prism_fmri.py` | 학습 (Pro6000 MinD-3D++ 재학습 스크립트 기반). `--init_ckpt`로 공개 가중치에서 시작 |
| `infer_prism_fmri.py` | 추론: 6뷰까지 (`images/` 격자, `views/<이름>/00~05.png` TRELLIS 입력). 3D는 `scripts/infer_prism.sh`가 이어서 TRELLIS로 |
| `tools/instantmesh_baseline.py` | **MinD-3D++ 기준선 전용** 6뷰 → InstantMesh (`--cam v12` 공식 / `--cam video` 영상 자세: FOV 30·거리 5.06·방향 자동). PRISM에는 안 씀 |
| `tools/cam_calib.py`, `tools/prep_calib_frames.py` | 영상 자세 InstantMesh 카메라 보정 시험 (기준선용, 10-26) |
| `scripts/*.sh` | Pro6000 실행 (`common.sh` 경로, `smoke_prism.sh`, `train_prism.sh`, `infer_prism.sh`, `eval_prism.sh`) |

## 학습 손실
| 성분 | 손실 |
|---|---|
| 의미 | ProtoCE: 6프레임 평균 → `semantic_to_clip`(1024→128→1024) → 범주 이름 OpenCLIP ViT-H 텍스트 프로토타입 (temp 0.07). 확산 손실은 차단(`--no_sem_detach`로 해제) |
| 변형 · LoRA · aggregation | 확산 손실 (v-예측 MSE, 공식) |
| 인코더 12층 CLS | 공식 CLIP 대조 손실 (OpenCLIP 이미지·텍스트) |
| 피험자 | 직교 손실에만 참여 (EEG판 bias_filtering과 같음), joint 단계에서 고정 |

총 손실 = 확산 + `λ_clip`·CLIP + `λ_proto`·ProtoCE + `λ_ortho`·직교 (기본 1, 1, 0).

- **직교 손실 (EEG판 그대로, 10-26 추가):** 의미(CLS 6프레임 평균)·변형(패치 토큰 평균)·피험자(헤드 출력 6프레임 평균) 세 벡터의 쌍별 |cos| 배치 평균 합. `ORTHO=1`로 켬.
- **2단계 학습 (EEG판 그대로):** `STAGE1=N`이면 처음 N 스텝은 확산 없이 ProtoCE + 직교만 → joint. 피험자 헤드는 joint에서 고정. EEG판과 다른 점: 1단계에서도 fMRI 인코더 ViT는 고정(`FREEZE`, 공개 가중치)
- 설계대로 돌리기: `ORTHO=1 STAGE1=1000 OUT_TAG=_s1o STEPS=6000 BS=4 AC=4 bash scripts/train_prism.sh` (출력 `prism_train_obj_s1o/`)

- **학습 GT = 자극 영상 프레임** (Pro6000 `data/GTviews_video/obj`, 영상 0·64·…·320번 프레임). 학습 정답은 피험자가 본 자극 안에서만. 평가 GT·렌더는 MinD-3D++ 프로토콜 그대로.
- **`--freeze_encoder`** (스크립트 기본 켜짐, `FREEZE=`로 끔): 인코더 ViT 본체를 공개 가중치 그대로 고정하고 헤드·aggregation·LoRA·semantic_to_clip만 학습. GPU 1을 MinD-3D++ 재학습과 나눠 쓰거나 4090(24 GB)에서 돌릴 때.
- 영상 GT로 학습하면 6뷰가 영상 자세(고도 0°, 회전 방향 물체별). PRISM은 TRELLIS라 카메라 불필요. MinD-3D++ 영상 GT 재학습 기준선은 `tools/instantmesh_baseline.py --cam video`.
- TRELLIS 출력 메시 방위는 GT와 맞는다는 보장이 없음 → 뷰별 지표 해석 전에 확인.

## Pro6000 실행 환경
- 저장소 위치: `/home/oem/jhj/MinD-3D++/mind_neural3d` (`common.sh`의 `P`)
- 공식 코드 `MinD-3D/`(4b74ebe), 작업 폴더 `run/`(`src` → `MinD-3D/src`, `pretrained_models/`), venv, 데이터, `weights/zero123plus-v1.2`, `weights/InstantMesh`는 MinD-3D++ 재학습 때 세팅한 것 그대로 사용
- 초기 가중치: `weights/MinD-3D-pp/fMRI-Objaverse/sub01.pt` (HF `Fudan-fMRI/MinD-3D-pp`, 8.6 GB)
- GPU: 기본 1번 (`GPU=` 로 변경)

## 확인 순서
1. `bash scripts/smoke_prism.sh` — 데이터 2개 형상, 공개 가중치 불러오기(`[init] … lora >0`), 20스텝 손실·메모리
2. 연결 확인: `CKPT=<공개 가중치> bash scripts/infer_prism.sh --limit 5` → 헤드가 항등이라 MinD-3D++ 공개 가중치 추론과 같은 6뷰가 나와야 함
3. `STEPS=5000 bash scripts/train_prism.sh` → `bash scripts/infer_prism.sh` → `MESH=<추론 meshes 폴더> NAME=<이름> bash scripts/eval_prism.sh`
