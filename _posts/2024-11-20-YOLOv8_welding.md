---
layout: post
title: YOLOv8 기반 용접 불량 자동 검출 시스템
date: 2024-11-20 15:00:00 +0800
category: experiment
thumbnail: /style/image/welding_project/12.png
icon: code
---


# YOLOv8 기반 용접 불량 자동 검출 시스템

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | YOLOv8 기반 용접 불량 자동 검출 및 웹 서빙 시스템 |
| 모델 | YOLOv8n (baseline), YOLOv8n + Augmentation, YOLOv8s + Augmentation |
| 데이터셋 | Kaggle - Welding Defect Object Detection v2 |
| 환경 | RTX 3060 12GB, PyTorch 2.5.1+cu121, CUDA 13.1, Python 3.10 |
| 서빙 | FastAPI + Streamlit 웹 애플리케이션 |

## 목적 (Objective)

**▪ 실험의 목표**

커스텀 데이터셋으로 YOLOv8을 파인튜닝하여 용접 불량(Bad Weld, Defect)을 자동 검출하고, 학습된 모델을 FastAPI + Streamlit으로 실제 서빙 가능한 웹 애플리케이션까지 구현한다.

모델 크기(yolov8n vs yolov8s)와 Augmentation 유무에 따른 성능 차이를 비교 분석한다.

**▪ 얻고자 하는 인사이트**

- 동일 데이터에서 Augmentation이 mAP와 일반화 성능에 미치는 실질적 영향
- 모델 크기(파라미터 수) 증가가 제조업 불량 탐지에서 가져오는 성능/속도 트레이드오프
- 학습된 모델이 실패하는 케이스(False Positive/Negative)의 패턴 분석
- 모델 학습부터 웹 서빙까지 전체 파이프라인 구현

## 💡 배경 및 아이디어 (Background & Motivation)

제조업에서 용접 품질 검사는 전통적으로 숙련된 작업자의 육안 검사에 의존한다. 이는 검사자의 피로도, 주관적 판단, 검사 속도 한계라는 문제를 가진다.

Computer Vision 기반 자동 검출 시스템은 이러한 문제를 해결할 수 있으며, 실제 산업 현장에서 수요가 높은 분야다. 특히 이번 프로젝트는 단순히 모델을 돌리는 것을 넘어, **실제 서비스 형태(웹 앱)까지 구현**하여 "모델 학습 → 서빙 파이프라인" 전체 흐름을 경험하는 것이 핵심 목표다.

**▪ 왜 YOLOv8인가**

YOLOv8은 Ultralytics에서 2023년 공개한 single-stage detector로, 이전 버전 대비 아래 3가지 핵심 변화가 있다.

첫째, **Anchor-free Detection Head**다. 기존 YOLO는 사전 정의된 anchor box를 기반으로 객체를 탐지했는데, YOLOv8은 anchor-free 방식으로 전환하여 다양한 크기와 비율의 객체에 더 유연하게 대응할 수 있다. 용접 결함처럼 크기가 일정하지 않은 객체 탐지에 유리하다.

둘째, **Decoupled Head**다. Classification과 Localization을 분리된 head에서 처리하여 각 task에 최적화된 학습이 가능하다.

셋째, **C2f(Cross Stage Partial with 2 convolutions) Backbone**이다. YOLOv5의 C3 모듈을 개선한 구조로, gradient flow를 풍부하게 하여 더 나은 특징 추출이 가능하다.

이러한 특성이 실시간 탐지가 필요한 산업 현장 적용에 적합하고, Ultralytics의 간결한 API로 파인튜닝과 실험 반복이 용이하여 선택했다.

## 📦 데이터셋 (Dataset)

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | Kaggle - Welding Defect Object Detection v2 |
| 라이선스 | CC0: Public Domain |
| 라벨 포맷 | YOLO format (.txt) |
| 해상도 | 전체 640×640 통일 |

**클래스 구성 및 분포**

| 클래스 | 정의 | Train | Valid | Test |
| --- | --- | --- | --- | --- |
| Bad Weld | 전반적인 불량 용접 | 1,089 (23.8%) | 194 (24.2%) | 95 (31.6%) |
| Good Weld | 정상 용접 | 1,896 (41.4%) | 335 (41.8%) | 117 (38.9%) |
| Defect | 특정 결함 (기포, 크랙 등) | 1,598 (34.9%) | 273 (34.0%) | 89 (29.6%) |
| **합계** |  | **4,583** | **802** | **301** |

![image.png](style/image/welding_project/1.png)

**이미지 수**

| Split | 이미지 수 |
| --- | --- |
| Train | 1,619장 |
| Valid | 283장 |
| Test | 126장 |

**EDA 주요 관찰**

- 클래스 분포가 Bad Weld 23.8% / Good Weld 41.4% / Defect 34.9%로 비교적 균형잡혀 있어 별도의 클래스 가중치 처리 없이 학습 가능
- 전체 이미지가 640×640으로 통일되어 있어 YOLOv8 기본 입력 크기와 완전히 호환, 추가 전처리 불필요
- 바운딩 박스 평균 면적 0.0568 (이미지 대비 약 5.7%)로 중간 크기 객체
- 박스 크기 편차가 큼 (std ≈ mean, 너비 기준 0.239 ± 0.228) → 아주 작은 결함부터 큰 결함까지 다양하게 분포

![image.png](style/image/welding_project/2.png)

## ⚙️ 환경 (Environment)

| 항목 | 내용 |
| --- | --- |
| OS | MacOS에서 로컬 서버 사용
(Windows 11 + WSL2 Ubuntu) |
| GPU | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| CUDA | 13.1 |
| PyTorch | 2.5.1+cu121 |
| Python | 3.10.19 |
| 주요 패키지 | ultralytics, fastapi, uvicorn, streamlit, opencv-python |

## 🧠 실험 설계 (Experiment Design)

**▪ 실험 목적**

동일 데이터셋에서 Augmentation 전략과 모델 크기 변화에 따른 탐지 성능 비교

**▪ 변수 설정**

| 항목 | exp1 baseline | exp2 augment | exp3 yolov8s |
| --- | --- | --- | --- |
| 모델 | yolov8n | yolov8n | yolov8s |
| Epochs | 100 | 100 | 100 |
| Early Stopping | patience=20 | patience=20 | patience=20 |
| Augmentation | X (기본값) | O (mosaic, hsv, flip 등) | O |
| Image Size | 640 | 640 | 640 |
| Batch Size | 16 | 16 | 16 |
| Optimizer | AdamW (auto) | AdamW (auto) | AdamW (auto) |

**▪ 모델 파라미터 비교**

| 모델 | Parameters | GFLOPs | 추론속도 |
| --- | --- | --- | --- |
| yolov8n | 3,006,233 | 8.1 | 2.7ms |
| yolov8s | 11,126,745 | 28.4 | 6.2ms |

yolov8s는 yolov8n 대비 파라미터가 약 3.7배, GFLOPs는 약 3.5배 크다.

**▪ 평가 지표**

- **mAP50**: IoU 0.5 기준 평균 정밀도 (주요 지표)
- **mAP50-95**: IoU 0.5~0.95 기준 평균 정밀도 (엄격한 지표)
- **Precision**: 탐지한 것 중 실제 불량 비율
- **Recall**: 실제 불량 중 탐지한 비율 (산업 현장에서 가장 중요한 지표)

## 📊 결과 및 분석 (Results & Analysis)

**▪ 실험 비교표 (전체)**

| 실험 | 모델 | mAP50 | mAP50-95 | Precision | Recall | 학습시간 |
| --- | --- | --- | --- | --- | --- | --- |
| exp1 baseline | yolov8n | **0.727** | **0.474** | 0.709 | 0.709 | 25.7분 |
| exp2 augment | yolov8n | 0.663 | 0.378 | 0.637 | 0.640 | 26.1분 |
| exp3 yolov8s | yolov8s | 0.705 | 0.414 | **0.713** | 0.652 | 40.5분 |

**▪ 클래스별 AP50 상세**

| 클래스 | exp1 baseline | exp2 augment | exp3 yolov8s |
| --- | --- | --- | --- |
| Bad Weld | **0.7817** | 0.7684 | 0.7675 |
| Good Weld | **0.8589** | 0.7862 | 0.8158 |
| Defect | **0.5398** | 0.4334 | 0.5313 |

## exp1 baseline

![image.png](style/image/welding_project/3.png)

![image.png](style/image/welding_project/4.png)

![image.png](style/image/welding_project/5.png)

## exp2 augment

![image.png](style/image/welding_project/6.png)

![image.png](style/image/welding_project/7.png)

![image.png](style/image/welding_project/8.png)

## exp3 YOLOv8s

![image.png](style/image/welding_project/9.png)

![image.png](style/image/welding_project/10.png)

![image.png](style/image/welding_project/11.png)

**▪ 주요 관찰 1: Augmentation이 오히려 성능을 떨어뜨렸다**

exp2(augment)가 exp1(baseline)보다 mAP50 기준 0.064 낮은 결과가 나왔다. 직관적으로는 Augmentation이 일반화 성능을 높여야 하지만, 이 데이터셋에서는 역효과가 났다.

원인은 데이터셋의 특성에 있다. 전체 이미지가 이미 640×640으로 통일되어 있고, 동일한 산업 환경(촬영 각도, 조명, 카메라 거리)에서 수집된 데이터다. 즉, **데이터 분포가 이미 매우 균일한 상황**에서 mosaic, hsv shift, random flip 같은 aggressive augmentation을 적용하면 실제 데이터에 존재하지 않는 분포를 만들어내 오히려 모델을 혼란스럽게 한다.

특히 Defect AP가 0.5398 → 0.4334로 가장 크게 떨어진 것이 이를 방증한다. Defect는 기포, 크랙처럼 시각적으로 미세한 특징을 가지는데, 색상 왜곡이나 모자이크 변환이 이 미세한 패턴을 훼손했을 가능성이 높다.

**▪ 주요 관찰 2: 더 큰 모델(yolov8s)이 baseline보다 낮은 성능**

yolov8s는 파라미터가 3.7배 많음에도 mAP50이 0.705로 baseline(0.727)보다 낮았다. 이는 **데이터 수 대비 모델 용량 과잉** 문제다.

Train 이미지 1,619장은 yolov8n을 충분히 학습시킬 수 있는 양이지만, yolov8s의 11M 파라미터를 일반화까지 학습시키기에는 부족하다. 작은 데이터셋에서는 모델 용량이 클수록 더 많은 데이터를 요구하기 때문에, 오히려 작은 모델이 더 잘 일반화된 결과가 나왔다.

추론 속도 측면에서도 yolov8n이 2.7ms, yolov8s가 6.2ms로 2.3배 차이가 나는데, 성능은 더 낮으면서 속도도 느리므로 이 데이터셋에서 yolov8s는 적합하지 않은 선택이었다.

**▪ 주요 관찰 3: Defect 클래스가 전체 성능의 병목**

3개 클래스 중 Defect AP50이 0.5398로 가장 낮고, Recall이 0.509로 실제 결함의 절반 가까이를 탐지하지 못하고 있다. Good Weld(0.8589)와 비교하면 0.32 차이가 나는 상당한 격차다.

Bad Weld는 전반적인 불량 용접을 의미하고, Defect는 기포·크랙·기공 등 특정 결함을 의미한다. 두 클래스는 시각적으로 유사한 경우가 많아 클래스 간 경계가 모호하다. Confusion Matrix를 보면 Bad Weld와 Defect 간의 혼동이 다른 클래스 조합 대비 가장 높게 나타난다.

**▪ 검출 성공 케이스**

![image.png](style/image/welding_project/12.png)

![image.png](style/image/welding_project/13.png)

기포(porosity)와 스패터(spatter)가 명확히 보이는 전형적인 Bad Weld 이미지에서 confidence 0.85로 정확히 탐지했다. 이처럼 훈련 데이터와 시각적으로 유사한 가로 구도의 이미지에서는 높은 신뢰도로 검출이 이루어진다.

**▪ 검출 실패 케이스 (False Negative)**

![image.png](style/image/welding_project/14.png)

육안으로도 불균일한 비드와 거친 표면이 보이는 불량 이미지를 Good Weld 0.76으로 오분류했다. 이 케이스의 원인을 분석하면 다음과 같다.

- **촬영 구도 차이**: 훈련 데이터는 대부분 가로 구도인데, 이 이미지는 세로 구도로 촬영되어 모델이 학습한 패턴과 다르게 인식했을 가능성이 높다.
- **배경 차이**: 밝은 회색 배경 + 금속판 위의 용접 구도로, 훈련 데이터의 전형적인 어두운 배경과 차이가 있다.
- **라벨 모호성**: 이 정도 불량이 훈련 데이터에서 어떤 클래스로 라벨링되었는지가 불명확하다. Bad Weld와 Defect의 경계가 모호한 케이스가 훈련 데이터에도 포함되어 있을 가능성이 있다.

이 실패 케이스는 Defect Recall 0.509라는 수치가 단순한 숫자가 아니라, 실제로 모델이 애매한 케이스에서 얼마나 불안정한지를 보여주는 사례다.

## 🌐 서빙 구현 (Serving)

**▪ 시스템 구조**

```
사용자 (브라우저)
    ↓ 이미지 업로드
Streamlit Frontend (포트 8501)
    ↓ HTTP POST /predict
FastAPI Backend (포트 8000)
    ↓ 추론
YOLOv8 모델 (best.pt)
    ↓ 탐지 결과 + 시각화 이미지 (base64)
Streamlit Frontend
    ↓ 결과 표시
사용자
```

**▪ FastAPI 백엔드 핵심 구조**

```python
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np    = np.array(img_pil)

    # 추론
    results = model(img_np, conf=0.25, iou=0.45)[0]

    # 탐지 결과 수집
    detections = []
    for box in results.boxes:
        cls_name = CLASSES[int(box.cls)]
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({
            "class": cls_name,
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2],
        })

    return {
        "detections": detections,
        "image_base64": img_base64,
        "inference_time_ms": inference_time,
        "is_defective": any(d["class"] != "Good Weld" for d in detections)
    }
```

**▪ 웹 앱 주요 기능**

- 이미지 업로드 → 원본 / 검출 결과 2열 나란히 비교
- 판정: 정상(OK) / 불량(NG) 배너 표시
- 검출 객체 수, 추론 시간(28.9ms), Confidence Threshold 메트릭 표시
- 검출 상세 테이블 (클래스, 신뢰도, bbox 좌표)
- 사이드바 Confidence Threshold 슬라이더 (0.10 ~ 0.90)

웹 결과 추론 시간이 22 ~ 38.6ms로 초당 약 30프레임 이상으로 처리가 가능한 속도다. RTX 3060 환경에서 실시간 처리에 충분한 성능이다.

## 🔍 인사이트 및 결론 (Insights & Conclusion)

**1. 데이터 분포가 균일할수록 Augmentation은 독이 될 수 있다**

일반적으로 Augmentation은 데이터 다양성을 높여 일반화 성능을 향상시키는 것으로 알려져 있다. 하지만 이번 실험처럼 데이터가 이미 동일한 촬영 환경에서 수집되어 분포가 균일한 경우, aggressive augmentation이 실제 데이터에 없는 분포를 만들어 오히려 성능을 떨어뜨릴 수 있다. Augmentation 전략은 데이터의 특성을 먼저 분석하고 선택해야 한다.

**2. 소규모 데이터셋에서는 모델 크기보다 모델 적합성이 중요하다**

yolov8s(11M params)가 yolov8n(3M params)보다 낮은 성능을 보인 것은, 모델 용량이 데이터 양에 맞지 않을 때 일반화 성능이 오히려 하락할 수 있음을 보여준다. 1,619장의 훈련 데이터에서는 작고 가벼운 yolov8n이 더 적합한 선택이었다.

**3. 클래스 정의의 모호성이 탐지 성능의 상한을 결정한다**

Bad Weld와 Defect의 Recall이 각각 0.763, 0.509로 큰 차이를 보이는 이유는 두 클래스의 시각적 경계가 모호하기 때문이다. 모델 구조나 학습 전략을 아무리 개선해도, 라벨 자체가 일관성 없이 부여되어 있다면 성능 향상에는 한계가 있다. 실제 산업 데이터셋 구축 시 클래스 정의와 라벨링 가이드라인을 명확히 하는 것이 모델 개선만큼 중요하다.

**4. 모델 학습에서 서빙까지의 파이프라인**

이번 프로젝트를 통해 단순히 모델을 학습시키는 것을 넘어, FastAPI로 추론 API를 구성하고 Streamlit으로 사용자 인터페이스까지 구현하는 전체 파이프라인을 경험했다. 28.9ms의 추론 속도는 실시간 처리에 충분하며, Confidence Threshold 조절을 통해 Precision과 Recall 간의 트레이드오프를 사용자가 직접 조정할 수 있도록 설계했다.

## 🚀 추가 실험 / 개선 방향 (Further Work)

**단기 개선 (모델 성능)**

- Confidence threshold 0.25 → 0.15로 낮춰 Defect Recall 향상 시도 (Precision 희생)
- Soft Augmentation (약한 hsv shift, 수평 flip만) 적용하여 augment 재실험
- YOLOv8n → YOLOv8m으로 중간 크기 모델 시도 (데이터 추가 없이 상한선 확인)

**장기 확장 (도메인 심화)**

- **DepthPro 연결**: 탐지된 불량 위치에 단일 이미지 깊이 추정 적용 → 2D 탐지 + 3D 위치 추정 파이프라인 구성. 불량의 깊이 정보를 추가하면 표면 결함인지 내부 결함인지 구분하는 데 활용 가능
- **YOLOv8-seg 확장**: Bounding Box → Instance Segmentation으로 불량 영역을 픽셀 단위로 분리하여 불량 면적 정량화 가능
- **실시간 영상 처리**: 웹캠/산업 카메라 스트림에서 실시간 불량 탐지
- **Docker 컨테이너화**: 실제 서버 환경 배포를 위한 컨테이너 구성