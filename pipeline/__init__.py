"""교통사고 단안 영상 재구성 파이프라인.

lingbot-map(피드포워드 GCTStream)을 지오메트리 백본으로 삼아, 단안 대시캠
영상에서 ① 배경 3DGS(.splat) ② 동적 차량 3D 궤적(.json)을 생성한다.

스테이지 흐름:
    s01_ingest          프레임 추출 (ffmpeg)
    s02_segment_track   YOLOv8-seg + ByteTrack + ego-motion + 동/정적 분류
    s03_geometry        lingbot 추론 + gravity align + 메트릭 스케일 보정 ★핵심
    s04_background_pcd   동적 픽셀 제외 PC 융합 + Open3D SOR
    s05_gaussian_train   3DGS 학습 (동적 마스크 loss 제외) → output.ply
    s06_trajectory       AB3DMOT 하이브리드 궤적 → trajectories.json
    s07_export           output.ply → output.splat + 궤적 패스스루
"""
