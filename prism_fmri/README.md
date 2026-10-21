# PRISM fMRI판 (성분 분리 → Zero123++ 직접 조건)

```
fMRI (6프레임)
→ fMRI 인코더 (ViT, MinD-3D++ 공개 가중치)
→ 분리 (의미 / 변형 / 피험자)
   1. 의미   (CLS)        → zero123++ cross-attention
   2. 변형   (패치 토큰)  → zero123++ cond_lat
   3. 피험자 (CLS)        → 버림 (지금은 손실 없음, 자리만)
→ zero123++ (공개 가중치, 고정) + LoRA (MinD-3D++ 공개 가중치에서 시작)
→ 6View → InstantMesh 또는 TRELLIS
```

기존 fMRI판(캡션 검색 → SDXL)은 그대로 두고, 이 폴더에 새로 추가함.

## 파일
| 파일 | 내용 |
|---|---|
| `prism_encoder.py` | `PrismFMRIEncoder`: MinD-3D++ `fMRI_Encoder` 상속 + 헤드 3개. 의미·변형 헤드는 잔차 MLP(마지막 층 0 초기화) → 처음엔 항등이라 공개 가중치를 넣으면 MinD-3D++와 같은 출력 |
| `prism_model.py` | `PrismMVDiffusion`: 공식 `src/mvdiffusion.MVDiffusion` 상속. 의미 ProtoCE(범주 프로토타입), 의미 경로 확산 gradient 차단(EEG판과 같음), 공식 CLIP 대조 손실 유지, 직교 손실 없음 |
| `train_prism_fmri.py` | 학습 (Pro6000 MinD-3D++ 재학습 스크립트 기반). `--init_ckpt`로 공개 가중치에서 시작 |
| `infer_prism_fmri.py` | 추론: 6뷰(`images/` 격자, `views/<이름>/00~05.png` TRELLIS 입력) + InstantMesh 메시(`--no_lrm`이면 생략) |
| `scripts/*.sh` | Pro6000 실행 (`common.sh` 경로, `smoke_prism.sh`, `train_prism.sh`, `infer_prism.sh`) |

## 학습 손실
| 성분 | 손실 |
|---|---|
| 의미 | ProtoCE: 6프레임 평균 → `semantic_to_clip`(1024→128→1024) → 범주 이름 OpenCLIP ViT-H 텍스트 프로토타입 (temp 0.07). 확산 손실은 차단(`--no_sem_detach`로 해제) |
| 변형 · LoRA · aggregation | 확산 손실 (v-예측 MSE, 공식) |
| 인코더 12층 CLS | 공식 CLIP 대조 손실 (OpenCLIP 이미지·텍스트) |
| 피험자 | 없음 |

총 손실 = 확산 + `λ_clip`·CLIP + `λ_proto`·ProtoCE (기본 1, 1).

## Pro6000 실행 환경
- 저장소 위치: `/home/oem/jhj/MinD-3D++/mind_neural3d` (`common.sh`의 `P`)
- 공식 코드 `MinD-3D/`(4b74ebe), 작업 폴더 `run/`(`src` → `MinD-3D/src`, `pretrained_models/`), venv, 데이터, `weights/zero123plus-v1.2`, `weights/InstantMesh`는 MinD-3D++ 재학습 때 세팅한 것 그대로 사용
- 초기 가중치: `weights/MinD-3D-pp/fMRI-Objaverse/sub01.pt` (HF `Fudan-fMRI/MinD-3D-pp`, 8.6 GB)
- GPU: 기본 1번 (`GPU=` 로 변경)

## 확인 순서
1. `bash scripts/smoke_prism.sh` — 데이터 2개 형상, 공개 가중치 불러오기(`[init] … lora >0`), 20스텝 손실·메모리
2. 연결 확인: `CKPT=<공개 가중치> bash scripts/infer_prism.sh --limit 5` → 헤드가 항등이라 MinD-3D++ 공개 가중치 추론과 같은 6뷰가 나와야 함
3. `STEPS=5000 bash scripts/train_prism.sh` → `bash scripts/infer_prism.sh` → 평가(MinD-3D++ 평가와 같은 `eval_obj_bs28.sh` 방식)
