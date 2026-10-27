# PRISM fMRI판

| 폴더 | 내용 |
|---|---|
| `v2/` | **현재 구현** — 박사님 10-27 구조도를 ViT·TRELLIS로 옮긴 1차 구현 (fMRI-Shape, 다피험자). 설계 근거: `EEG/.보고 자료/261029_7차보고/PRISM_설계논의.md` |
| `legacy_v1/` | 이전 구현 (Zero123++ 기반, 10-22~26). 지금은 **MinD-3D++ 기준점 평가**에만 씀: `scripts/preview_eval.sh`(공개 가중치 6뷰 → InstantMesh → 평가), `tools/instantmesh_baseline.py`, `tools/cam_calib.py`(영상 자세 InstantMesh 카메라 보정) |

## v2 실행 (Pro6000, `v2/scripts/`)
1. `dino_v2.sh` — 자극 영상 프레임 → TRELLIS 조건 토큰 정답 (meshgen venv)
2. `train_v2.sh` — 학습 (`STEPS`, `BS`, `TAG`, `EXTRA`)
3. `infer_v2.sh` — test fMRI → 조건 토큰 → TRELLIS GLB (`PERC=keep|mean`)
전처리(로컬): `v2/prep_shape_targets.py` (영상 방위·실루엣, 64³ 복셀, 영상 프레임)
