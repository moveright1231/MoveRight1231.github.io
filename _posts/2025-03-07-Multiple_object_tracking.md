---
layout: post
title: Multiple Object Tracking
date: 2025-03-07 11:30:00 +0800
category: experiment
thumbnail: style/image/Multiple Object Tracking/4.png
icon: code
---

# Multiple Object Tracking

**프로젝트 개요**

| **항목** | **내용** |
| --- | --- |
| **프로젝트명** | Football Players & Ball Tracking with YOLOv5 + ByteTrack |
| **모델(Model)** | YOLOv5x (COCO), YOLOv5 커스텀 가중치(football-players-detection), ByteTrack MOT |
| **데이터셋(Dataset)** | Roboflow Football Players Detection, DFL Bundesliga Data Shootout 샘플 영상 |
| **환경** | RTX 3060, PyTorch 2.3.0, CUDA 12.4, Python 3.9 |

---

# **목적 (Objective)**

### **▪ 실험의 목표**

- 커스텀 축구 선수 감지 모델과 ByteTrack을 결합해 경기 영상에서 선수·공을 안정적으로 추적한다.
- COCO 사전학습 모델 대비 커스텀 가중치 활용 시 ID 지속성·검출 정밀도가 얼마나 개선되는지 평가한다.

### **▪ 얻고자 하는 인사이트**

- 프레임 단위 검출이 아닌 MOT 파이프라인에서 YOLOv5+ByteTrack 조합의 안정성.
- 공·선수 분리 감지 후 역할별 색상/마커 주석이 실시간 분석에 주는 가독성 효과.

---

# **💡 배경 및 아이디어 (Background & Motivation)**

### **▪ 실험 동기**

- 스포츠 분석 자동화를 위해 객체 검출 + 다중 객체 추적이 필수적이며, ByteTrack이 단순·강력한 SOTA 기반임.
- Roboflow 커스텀 가중치로 도메인 특화 검출력을 높이고, 공 소유자 파악 같은 다운스트림 로직을 시험.

---

# **📦 데이터셋 (Datasets)**

| **구분** | **내용** |
| --- | --- |
| **데이터셋 이름** | Roboflow Football Players Detection, DFL Bundesliga Data Shootout |
| **크기** | Players: 수천 장(roboflow), Video: 1 클립(약 수백 프레임) |
| **입력 형식** | 1280×720 RGB 영상 프레임 |
| **전처리 및 증강** | YOLOv5 기본 aug (mosaic, hsv), 추론 시 리사이즈 1280 |
| **평가 지표** | mAP@0.5 (검출), IDF1 / MOTA (추적), 주관적 시각 품질 |

---

# **⚙️ 환경 (Environment)**

- local PC RTX 3060 12VRAM
- PyTorch 2.3.0
- CUDA 12.4 / cuDNN
- Python 3.9
- 사용 패키지: ultralytics>=8.2.64, yolov5, ByteTrack, onemetric, cython_bbox, opencv-python, numpy, tqdm, matplotlib

---

# **🧠 실험 설계 (Experiment Design)**

### **▪ 실험 목적**

- COCO 사전학습 vs 커스텀 가중치 모델의 추적 성능 비교
- ByteTrack 하이퍼파라미터(탐지 임계값, track_buffer) 변화에 따른 ID 안정성 관찰
- 공-선수 근접도로 소유자 추정 로직의 신뢰도 확인

### **▪ 변수 설정**

| **항목** | **설정값** |
| --- | --- |
| Epoch | (사전학습 완료, 추가 학습 없음) |
| Batch Size | 추론용 없음 |
| Learning Rate | - |
| Optimizer | - |
| Track Threshold | 0.25 / 0.35 비교 |
| Match Threshold | 0.8 (고정) |
| Track Buffer | 30 / 60 비교 |
| Detection Conf | 0.25 |
| Input Size | 1280 |

### **▪ 비교 대상**

| **실험명** | **설명** |
| --- | --- |
| **coco-det** | YOLOv5x COCO → ByteTrack |
| **custom-det** | 커스텀 football 가중치 → ByteTrack |
| **custom-det-highthr** | 커스텀 + 높은 track_thresh(0.35) |
| **buffer-60** | 커스텀 + track_buffer 60 |

---

# **📊 결과 및 분석 (Results & Analysis)**


![녹음 2026-01-28 154843.gif](style/image/Multiple_Object_Tracking/1.gif)

![녹음 2026-01-28 154630.gif](style/image/Multiple_Object_Tracking/2.gif)

![녹음 2026-01-28 154758.gif](style/image/Multiple_Object_Tracking/3.gif)

### **▪ 그래프/표**

- mAP 및 IDF1은 소규모 샘플 기준 정성 평가 위주.
- 관찰 포인트:
    - 커스텀 가중치가 선수 검출 recall 향상 → ID 스위치 감소.
    - track_buffer 증가 시 일시적 occlusion에서 ID 유지가 개선되지만, 공·선수 혼동이 약간 증가.
    - 공 소유자 추정은 단일 공 검출 가정이 깨질 때 불안정하므로 후처리 필요.

| **실험명** | **IDF1(질적)** | **ID Switch(질적)** | **비고** |
| --- | --- | --- | --- |
| coco-det | 중간 | 높음 | 공·선수 미검 많음 |
| custom-det | 높음 | 낮음 | 기본 설정 |
| custom-det-highthr | 비슷 | 낮음 | 누락 프레임 약간 증가 |
| buffer-60 | 높음 | 더 낮음 | 오탐 지속 가능성 증가 |

### **▪ 주요 관찰**

- 커스텀 가중치 + 기본 ByteTrack 파라미터가 가장 균형적.
- 공 추적은 작은 객체 특성상 conf 조정과 NMS 튜닝이 필요.
- 소유자 판별은 공 주변 패딩 박스 IoU 기반으로 단순 구현되어, 다중 공/오탐 시 오동작.

---

# **인사이트 및 결론 (Insights & Conclusion)**

- 도메인 특화 YOLOv5 가중치가 추적 파이프라인 전체의 ID 안정성을 크게 좌우한다.
- ByteTrack의 track_buffer와 track_thresh는 occlusion 상황에서 ID 유지·오탐 지속 사이 트레이드오프를 만든다.
- 공 소유자 판단은 추가 규칙(속도 벡터, 팀 컬러 분류)이 들어가야 실제 경기 분석에 적용 가능하다.

---

# **추가 실험 / 개선 방향 (Further Work)**

- 공 전용 소형 객체 탐지기(YOLOv8n/RT-DETR) 비교
- 팀 컬러 분류를 통한 소유 팀 식별
- Multi-ball 상황 대비한 다중 공 필터링/트래커 분기
- 리플레이·슬로모션 영상에서 프레임레이트 변화 대응 실험
- 3개 이상 영상으로 일반화 검증, seed 고정

---