---
layout: post
title: PatchCore 자동감지 기능 업데이트
date: 2025-03-03 17:00:00 +0800
category: experiment
thumbnail: /style/image/patchcore/5.png
icon: code
---

# PatchCore 서빙 고도화 — YOLOv8-cls 카테고리 자동 감지 추가

> 이전 포스트: [PatchCore 기반 비지도 이상 탐지 시스템](https://moveright1231.github.io/2026-04-02-PatchCore_AnomalyDetection)
> 

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 목표 | 카테고리 수동 선택 → 이미지 업로드만으로 자동 추론 |
| 추가 모델 | YOLOv8n-cls (15개 카테고리 분류기) |
| 학습 데이터 | MVTec AD 정상 이미지 (train/good/) |
| 변경 범위 | FastAPI + Streamlit 서빙 업데이트 |

## 💡 배경 및 아이디어 (Background & Motivation)

**▪ 기존 시스템의 한계**

이전 PatchCore 서빙에서는 사용자가 이미지를 업로드하기 전에 카테고리(bottle, cable, carpet 등)를 직접 드롭다운에서 선택해야 했다.

```
기존 흐름:
사용자 → 카테고리 선택 → 이미지 업로드 → PatchCore 추론
```

실무에서 검사할 제품 종류가 고정된 라인이라면 괜찮지만, 여러 제품을 다루는 환경에서는 사용자가 매번 카테고리를 알고 선택해야 하는 불편함이 있다.

**▪ 해결책: YOLOv8-cls로 카테고리 자동 감지**

이미지만 넣으면 어떤 카테고리인지 자동으로 분류한 뒤, 해당 카테고리의 Memory Bank로 PatchCore 추론까지 자동으로 실행되는 파이프라인을 구축했다.

```
개선된 흐름:
사용자 → 이미지 업로드
    ↓
YOLOv8-cls → 카테고리 자동 감지 ("bottle", 97.3%)
    ↓
해당 Memory Bank 자동 로드
    ↓
PatchCore 추론 → 정상/불량 판정 + 히트맵
```

**▪ 왜 Detection이 아니라 Classification인가**

MVTec 데이터셋은 이미지 전체가 하나의 카테고리라서 bbox 라벨이 없다. 카테고리 판별에는 "이게 bottle인가 cable인가"를 맞히면 충분하므로 Classification이 적합하다.

## ⚙️ 환경 (Environment)

기존 PatchCore 환경과 동일 (`anomaly-detect` conda 환경)에 ultralytics 추가.

```bash
pip install ultralytics
pip install "numpy<2" --force-reinstall
pip install --force-reinstall "opencv-python-headless==4.8.1.78"
```

## 🔧 구현 상세

**▪ 추가된 파일 구조**

```
anomaly-detection/
├── prepare_cls_data.py     ← NEW: MVTec → YOLOv8-cls 데이터 구조 변환
├── train_cls.py            ← NEW: yolov8n-cls 분류기 학습
├── app/
│   ├── main.py             ← 업데이트: /predict/auto, /cls/status 추가
│   └── streamlit_app.py    ← 업데이트: 자동/수동 모드 UI
├── data/
│   ├── MVTec/              ← 기존
│   └── MVTec_cls/          ← NEW: train/{category}/, val/{category}/
└── runs/
    ├── {category}_resnet50_exp1/  ← 기존 Memory Bank
    └── cls/
        └── weights/
            └── best.pt     ← NEW: YOLOv8-cls 분류기
```

### 1단계 — 데이터 준비 (prepare_cls_data.py)

MVTec 15개 카테고리의 정상 이미지를 YOLOv8-cls가 요구하는 폴더 구조로 변환한다.

```
data/MVTec_cls/
├── train/{category}/   ← data/MVTec/{category}/train/good/ 복사
└── val/{category}/     ← data/MVTec/{category}/test/good/ 복사
```

불량 이미지는 사용하지 않는다. 카테고리 분류기는 "이게 bottle인가 cable인가"만 판단하면 되기 때문에 정상 이미지만으로 충분하다.

### 2단계 — YOLOv8-cls 학습 (train_cls.py)

```python
model = YOLO("yolov8n-cls.pt")
model.train(
    data="data/MVTec_cls",
    epochs=30,
    imgsz=224,
    batch=32,
    project="runs/cls",
    name=".",       # runs/cls/weights/best.pt 에 고정 저장
    exist_ok=True,
)
```

- Backbone: yolov8n-cls (경량, 빠른 추론)
- 저장 경로: `runs/cls/weights/best.pt` (FastAPI 하드코딩 참조)
- 15개 카테고리 분류: bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper

### 3단계 — FastAPI 업데이트 (app/main.py)

**추가된 엔드포인트**

| 엔드포인트 | 설명 |
| --- | --- |
| `GET /cls/status` | 분류기(best.pt) 존재 여부 반환 |
| `POST /predict/auto` | 이미지만 입력 → 카테고리 자동 감지 → PatchCore 추론 |

**설계 포인트**

기존 `/predict` 엔드포인트의 PatchCore 추론 코드를 `_run_inference()` 함수로 추출하여, `/predict`와 `/predict/auto` 두 엔드포인트가 공유하도록 리팩토링했다.

YOLOv8-cls 모델은 `_cls_model` 전역 변수에 첫 요청 시 1회만 로드하는 방식으로 기존 PatchCore 캐시 전략과 일관성을 유지했다.

```python
# /predict/auto 응답 예시
{
    "is_anomaly": true,
    "anomaly_score": 1.2345,
    "threshold": 0.5,
    "inference_time_ms": 120.3,
    "heatmap_base64": "...",
    "overlay_base64": "...",
    "detected_category": "bottle",
    "category_confidence": 0.9731
}
```

### 4단계 — Streamlit UI 업데이트 (app/streamlit_app.py)

`GET /cls/status` 응답으로 분류기 준비 여부를 체크하여 UI를 동적으로 구성한다.

```
분류기(best.pt) 없음 → "수동 선택" 모드만 표시 + 학습 안내 메시지
분류기(best.pt) 있음 → "자동 감지" / "수동 선택" 라디오 버튼 표시
```
![분류기 준비](style/image/patchcore/a.png)


자동 감지 모드에서는 카테고리 드롭다운을 숨기고, 추론 결과 아래에 감지된 카테고리와 신뢰도를 배너로 표시한다.

```
감지된 카테고리: bottle (97.3%)
```

## 🔧 이슈 해결 — NumPy 버전 충돌 재발

### 문제

`ultralytics` 패키지 설치 시 NumPy가 자동으로 2.2.6으로 업그레이드됐다. 기존 `faiss-gpu 1.7.2`는 NumPy 1.x 기준으로 컴파일되어 있어 `numpy.core.multiarray failed to import` 에러가 발생했다.

```
ultralytics 설치
    → numpy 2.2.6으로 자동 업그레이드
    → faiss-gpu (NumPy 1.x 빌드) import 실패
    → numpy.core.multiarray failed to import
```

이전 PatchCore 구축 시에도 동일한 충돌을 경험했는데, ultralytics 추가로 다시 재발한 케이스다.

### 해결

```bash
pip install "numpy<2" --force-reinstall
pip uninstall opencv-python -y                        # ultralytics가 설치한 4.13 제거
pip install --force-reinstall "opencv-python-headless==4.8.1.78"
pip install "numpy<2" --force-reinstall               # opencv 재설치 후 numpy 다시 고정
```

최종 확인:

```
numpy 1.26.4 / cv2 4.8.1 / faiss OK
```

ultralytics는 `opencv-python>=4.6.0` 의존성 경고를 출력하지만, `opencv-python-headless`가 동일한 역할을 하므로 실제 동작에는 영향이 없다.

## 📊 결과 — 전체 파이프라인 데모

![자동 감지 파이프라인 데모](style/image/patchcore/b.png)

![자동 감지 파이프라인 데모](style/image/patchcore/PatchCore.gif)

이미지를 업로드하는 순간부터 카테고리 자동 감지 → PatchCore 추론 → 히트맵 시각화까지 한 번에 실행된다.

**전체 파이프라인 흐름**

```
이미지 업로드
    ↓ (YOLOv8-cls, ~30ms)
카테고리: bottle (confidence: 97.3%)
    ↓ (Memory Bank 로드 + FAISS KNN)
Anomaly Score: 1.2345
    ↓ (히트맵 생성)
판정: 불량 (Anomaly)
원본 / 히트맵 / 오버레이 3열 표시
```

## 🔍 인사이트 및 결론 (Insights & Conclusion)

**1. 2-stage 파이프라인으로 사용성 대폭 향상**

YOLOv8-cls → PatchCore 2단계 파이프라인은 단순한 UX 개선을 넘어, "어떤 제품이든 이미지 하나로 이상 탐지"가 가능한 범용 검사 시스템을 만들었다. 실무에서 여러 제품 라인을 하나의 시스템으로 관리하는 시나리오에서 특히 유용하다.

**2. 분류기 학습 데이터로 정상 이미지만 사용해도 충분하다**

불량 이미지 없이 정상 이미지만으로 YOLOv8-cls를 학습했는데, 카테고리 분류는 정상/불량 여부가 아니라 "이게 어떤 종류의 물체인가"를 판단하는 것이므로 정상 이미지만으로 충분하다. 불량 데이터가 거의 없는 실무 상황에서 이 접근법이 유효하다.

**3. 의존성 관리가 중요하다**

NumPy 버전 충돌이 PatchCore 구축 시와 ultralytics 추가 시 두 번 재발했다. faiss-gpu와 ultralytics처럼 서로 다른 NumPy 버전을 요구하는 패키지를 같은 환경에서 사용할 때는 버전을 명시적으로 고정하는 습관이 중요하다.

## 🚀 개선 방향 (Further Work)

- **분류 신뢰도 임계값 설정**: confidence가 낮을 때 (예: < 80%) 자동 감지 대신 수동 선택 유도
- **분류 실패 케이스 분석**: 어떤 이미지에서 카테고리를 잘못 분류하는지 패턴 파악
- **Jetson Nano 온디바이스 배포**: YOLOv8-cls + PatchCore 전체 파이프라인을 ONNX로 변환하여 엣지 디바이스에서 실행
- **실시간 카메라 스트림 연동**: 웹캠 또는 산업 카메라에서 실시간으로 카테고리 감지 + 이상 탐지