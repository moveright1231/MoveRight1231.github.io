---
layout: post
title: About
---

* content
{:toc}

## Profile

---

**이동우 (Dongwoo Lee)**

세명대학교 정보통신학부

- **Email** : dlehddn1231@naver.com
- **GitHub** : [github.com/MoveRight1231](https://github.com/MoveRight1231)

---

## Research Interests

3D Computer Vision에 관심이 많으며, 특히 다음 분야를 중점적으로 공부하고 있습니다.

- **Novel View Synthesis** : NeRF, 3D Gaussian Splatting
- **3D Reconstruction** : Structure-from-Motion (COLMAP, GLOMAP), Multi-view Stereo
- **Feature Matching** : LightGlue, MASt3R, DUSt3R
- **Depth Estimation** : DepthPro
- **Object Detection & Tracking** : YOLOv8, Multiple Object Tracking
- **Multimodal Learning** : CLIP

---

## Skills

**Languages**
- Python, C

**Deep Learning**
- PyTorch, TensorFlow
- YOLOv8 (Ultralytics), OpenCV, Pillow

**3D Vision**
- COLMAP, GLOMAP, nerfstudio
- LightGlue, DUSt3R, MASt3R

**Tools & Environments**
- VSCode, Google Colab (GPU)
- Roboflow (데이터 어노테이션)
- Docker, Ubuntu, macOS

**etc**
- Tesseract OCR, Google Cloud Vision API
- Kivy (Android 앱 개발, Buildozer)
- Django

---

## Projects

### CBT 시험지 자동 채점 시스템 (캡스톤 디자인)
YOLOv8 기반 객체 탐지 + Tesseract OCR + Kivy UI를 활용하여 객관식 시험지를 자동으로 채점하는 Android 앱 개발.

- 데이터셋 직접 구축 및 Roboflow로 어노테이션 (약 20장, Pascal VOC 포맷)
- 문제 영역 검출 / 선택지 체크 여부 분류 모델 각각 학습
- 가로형 / 세로형 시험지 자동 구분 알고리즘 설계
- Buildozer로 Android APK 빌드 및 배포 (arm64-v8a / armeabi-v7a)

### 3D Gaussian Splatting 재현
nerfstudio + COLMAP을 활용하여 직접 촬영한 이미지로 3D Gaussian Splatting 파이프라인 구현.

### 식당 추천 시스템 (CLIP 기반)
OpenAI CLIP 모델을 활용한 이미지-텍스트 유사도 기반 식당 추천 시스템 구현.

### ResNet Ablation Study
Residual Connection 유무에 따른 성능 변화 실험 및 분석.

### 인테리어 추천 프로그램
사용자 취향에 맞는 인테리어 스타일 추천 시스템.

---

## Paper Reviews

| 논문 | 분야 |
|---|---|
| NeRF: Representing Scenes as Neural Radiance Fields | Novel View Synthesis |
| 3D Gaussian Splatting for Real-Time Radiance Field Rendering | Novel View Synthesis |
| MASt3R | 3D Reconstruction / Feature Matching |
| DUSt3R | 3D Reconstruction |
| GLOMAP: Global Structure-from-Motion | SfM |
| LightGlue | Feature Matching |
| DepthPro | Depth Estimation |
| AlexNet | Image Classification |
