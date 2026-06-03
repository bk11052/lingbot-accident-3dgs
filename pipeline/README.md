# 단안 교통사고 장면 재구성 파이프라인

lingbot-map(피드포워드 GCTStream)을 지오메트리 백본으로, 단안 대시캠 영상에서
**배경 3DGS(`.splat`)** 와 **동적 차량 3D 궤적(`.json`)** 을 생성한다. 프론트엔드(웹,
팀원 제작)가 이 둘을 읽어 장면을 재구성하고, 궤적 위에 목업 차량 3D 모델을 올린다.

> 졸업작품 주제: *"LiDAR 없는 단안 영상에서 교통사고 장면을 3DGS로 재구성"*. 기존
> 3DGS 레포의 COLMAP 기반 지오메트리 프론트엔드(포즈/depth/scale)가 대시캠에서 자주
> 실패해, 이를 lingbot-map 한 번의 추론으로 대체했다.

## 스테이지

| 스테이지 | 역할 | 핵심 출력 |
|---|---|---|
| `s01_ingest` | ffmpeg 프레임 추출 | `images_colmap/*.jpg` |
| `s02_segment_track` | YOLOv8-seg + ByteTrack + ego-motion 보정 + 동/정적 분류 | 동적 마스크, `bbox_sequence.json` |
| `s03_geometry` ★ | lingbot 추론 → 포즈+depth+world points, gravity align, **메트릭 스케일 보정** | `poses.npy`, `intrinsics.json`, `scaled_depth_maps/`, `sparse.ply` |
| `s04_background_pcd` | 동적 픽셀 제외 PC 융합 + Open3D SOR | `background.ply` |
| `s05_gaussian_train` | 3DGS 학습(동적 마스크 loss 제외) | `model/.../point_cloud.ply` |
| `s06_trajectory` | AB3DMOT 하이브리드(ByteTrack id + lingbot depth + KF) | `trajectories.json` |
| `s07_export` | `output.ply` → `output.splat`, 궤적 패스스루 | `output.splat` + `trajectories.json` |

## 실행 (A5000 서버)

> 로컬(mac)에는 CUDA가 없어 실행 불가 — 코드 편집은 로컬, 실행/검증은 A5000에서.

```bash
# 전체
python -m pipeline.run_pipeline --input clip.mp4

# 지오메트리만 검증 (가장 먼저 확인할 것)
python -m pipeline.run_pipeline --input clip.mp4 \
    --steps s01_ingest,s02_segment_track,s03_geometry

# 중간부터 재개 (이전 out_root 재사용)
python -m pipeline.run_pipeline --input clip.mp4 --steps s06_trajectory \
    --out_root pipeline/outputs/run_YYYYMMDD_HHMMSS
```

## 설치

### 1) lingbot 지오메트리 env (s01~s04, s06, s07)

```bash
# lingbot-map 본체
pip install -e .
# 파이프라인 의존성
pip install -r requirements-pipeline.txt
```

- FlashInfer는 선택. 없으면 SDPA 폴백(`geometry.use_sdpa: auto`). **A5000(CUDA 11.8)은
  FlashInfer 미설치 → SDPA 자동 사용.**
- 체크포인트: `geometry.model_path`(YAML) 또는 환경변수 `LINGBOT_MODEL_PATH`에
  로컬 `.pt` 또는 HuggingFace repo id(`robbyant/lingbot-map`) 지정.

### 2) third_party clone

```bash
mkdir -p third_party
git clone https://github.com/xinshuoweng/AB3DMOT third_party/AB3DMOT
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting third_party/gaussian-splatting
```

- AB3DMOT는 `KF` 클래스만 사용(s06). 경로는 `AB3DMOT_ROOT` 환경변수로 덮어쓸 수 있다.

### 3) 3DGS 학습 env (s05) — 별도 conda 환경 권장

lingbot(torch 2.8/cu128)과 gaussian-splatting의 CUDA 래스터라이저는 충돌할 수 있어
**별도 env**(기본 이름 `gs3dgs`, `gaussian_train.conda_env`로 변경)를 만든다. s05는 이
env를 `conda run`으로 호출하고, 산출물은 디스크(out_root)로 주고받는다.

```bash
conda create -n gs3dgs python=3.10 -y
conda activate gs3dgs
# gaussian-splatting 환경 구성 (해당 레포 안내대로):
pip install -e third_party/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install -e third_party/gaussian-splatting/submodules/simple-knn
pip install plyfile tqdm
```

### 시스템 의존성
- `ffmpeg` (s01 프레임 추출)

## 출력 계약 (프론트엔드 인터페이스)

`<out_root>/s07_export/`:
- `output.splat` — 배경 가우시안 스플랫
- `trajectories.json` — track별 월드 좌표 궤적 (메트릭, 지면 z≈0, 속도 m/s 포함)

## 좌표/단위 규약
- 포즈: 4×4 camera-to-world (c2w). gravity align 후 **지면 z=0**, +z가 위.
- 메트릭 스케일: `geometry.metric_rescale: true`면 카메라 높이 prior(video 1.5m)에 맞춰
  월드·depth를 리스케일 → 궤적 속도(m/s)가 물리적으로 의미를 가짐.

## 라이선스 주의
YOLOv8-seg(AGPL-3.0), AB3DMOT(CMU 비상업 연구용), gaussian-splatting(Inria 비상업) —
졸업작품(비상업)엔 OK, 상업화 시 재검토.
