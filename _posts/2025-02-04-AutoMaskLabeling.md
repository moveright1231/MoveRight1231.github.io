---
layout: post
title: 용접 불량 자동 검출 시스템 + auto mask labeling(SAM2)
date: 2025-02-04 12:00:00 +0800
category: experiment
thumbnail: /style/image/AutoMaskLeabeling/12.png
icon: code
---

# welding defect (seg)


# SAM2 자동 라벨링으로 YOLOv8 Detection을 Segmentation으로 확장하기

> 이전 포스트: [YOLOv8 기반 용접 불량 자동 검출 시스템](https://moveright1231.github.io/2024-11-20-YOLOv8_welding)
> 

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | SAM2 자동 라벨링 파이프라인 + YOLOv8-seg 용접 불량 Segmentation |
| 모델 | SAM2 (sam2_hiera_large), YOLOv8n-seg, YOLOv8s-seg |
| 데이터셋 | Kaggle - Welding Defect Object Detection v2 (이전 프로젝트와 동일) |
| 환경 | RTX 3060 12GB, PyTorch 2.5.1+cu121, CUDA 13.1, Python 3.10 |
| 서빙 | FastAPI + Streamlit (Detection/Segmentation 모드 전환) |

## 목적 (Objective)

**▪ 실험의 목표**

이전 프로젝트에서 YOLOv8n Detection으로 mAP50 0.727을 달성했지만, Bounding Box는 객체를 사각형으로만 감싸기 때문에 실제 불량 부위의 정확한 형태를 파악하기 어렵다.

```
Detection (bbox)          Segmentation (mask)
┌──────────────┐          ╭──────────╮
│  ████ ░░░░   │    →     │  ████    │  ← 불량 영역만 정확히
│  ████ ░░░░   │          ╰──────────╯     픽셀 단위로 표현
└──────────────┘
```

Segmentation 마스크를 출력하면 불량 면적/형태 정량화, 복잡한 경계(크랙, 용접 결함 형태) 정확한 표현, 더 직관적인 검사 결과 시각화가 가능하다.

**▪ 핵심 문제: 마스크 라벨이 없다**

Segmentation 학습에는 픽셀 단위 마스크 라벨이 필요한데, 기존 Welding Defect 데이터셋은 bbox 라벨만 제공한다. 직접 마스킹하면 1,619장 × 평균 2.7개 객체 = 약 4,300개 마스크를 수작업으로 그려야 한다.

**▪ 해결책: SAM2 Box Prompt 방식 자동 라벨링**

Meta의 SAM2(Segment Anything Model 2)는 bbox를 프롬프트로 받아 마스크를 자동 생성한다. 기존 bbox 라벨을 그대로 프롬프트로 활용하면 별도의 수작업 없이 마스크 라벨을 자동 생성할 수 있다.

## 💡 배경 및 아이디어 (Background & Motivation)

**▪ 왜 SAM2인가**

SAM2는 Meta가 2024년 공개한 범용 분할 모델로, 이미지와 비디오 모두에서 동작한다. 핵심은 어떤 객체든 Point, Box, Mask를 프롬프트로 받아 마스크를 생성한다는 점이다.

이번 프로젝트에서는 **Box Prompt 방식**을 활용했다. 기존 YOLO bbox 라벨 (cx, cy, w, h)을 SAM2가 이해하는 (x1, y1, x2, y2) 픽셀 좌표로 변환하여 프롬프트로 입력하면 SAM2가 해당 영역의 마스크를 자동으로 생성한다.

이 접근의 핵심 장점은 **4,300개의 마스크를 코드 한 번 실행으로 생성**할 수 있다는 것이다. 수작업 라벨링 대비 시간과 비용을 대폭 절감할 수 있어, 실무에서 Segmentation 데이터셋 구축 비용 문제를 해결하는 현실적인 방법이다.

## 📦 데이터셋 (Dataset)

이전 프로젝트와 동일한 Kaggle Welding Defect Object Detection v2 데이터셋을 사용한다.

| Split | 이미지 수 | bbox 객체 수 |
| --- | --- | --- |
| Train | 1,619장 | 4,235개 |
| Valid | 283장 | 779개 |
| Test | 126장 | 295개 |

클래스: Bad Weld (0), Good Weld (1), Defect (2) — 이전 프로젝트와 동일

## ⚙️ 환경 (Environment)

| 항목 | 내용 |
| --- | --- |
| OS | Windows 11 + WSL2 Ubuntu |
| GPU | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| CUDA | 13.1 / PyTorch 2.5.1+cu121 |
| Python | 3.10.19 |
| SAM2 | sam2_hiera_large.pt |
| 주요 패키지 | sam-2, ultralytics, fastapi, streamlit |

## 🧠 전체 파이프라인

```
기존 데이터셋 (bbox 라벨만)
        │
        ▼
  [ auto_label.py ]
  bbox → SAM2 box prompt → 마스크 생성 → polygon 변환
        │
        ▼
  마스크 품질 필터링 (면적 비율 기반)
        │
        ▼
  labels_seg/ (YOLO-seg polygon 포맷)
        │
        ▼
  [ train_seg.py ] YOLOv8n-seg 학습 (3 실험)
        │
        ▼
  Detection vs Segmentation 성능 비교
        │
        ▼
  FastAPI + Streamlit 서빙 업데이트
```

## 🔧 SAM2 자동 라벨링 구현

**▪ bbox → SAM2 프롬프트 변환 핵심 로직**

```python
# bbox (cx, cy, w, h) normalized → SAM2 box prompt (x1, y1, x2, y2) pixel
x1 = (cx - w/2) * img_width
y1 = (cy - h/2) * img_height
x2 = (cx + w/2) * img_width
y2 = (cy + h/2) * img_height

# SAM2 추론
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    box=np.array([x1, y1, x2, y2]),
    multimask_output=False
)

# 마스크 → YOLO-seg polygon 변환
contours, _ = cv2.findContours(mask.astype(np.uint8),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
polygon = contours[0].reshape(-1, 2) / [img_width, img_height]
# 저장 포맷: class_id x1 y1 x2 y2 x3 y3 ...
```

**▪ 데이터 구조 처리 (하드 링크)**

YOLO는 `images/` 경로에서 자동으로 `labels/`를 찾는다. `labels_seg/`를 `labels/`로 인식시키기 위해 심볼릭 링크 대신 **하드 링크 구조**를 생성했다. (심볼릭 링크는 YOLO가 resolve해 원본 `labels/`로 돌아가는 문제 발생)

```
data_seg/
├── train/
│   ├── images/  ← 원본 이미지 하드 링크 (용량 추가 없음)
│   └── labels/  ← labels_seg 파일 하드 링크
├── valid/
└── test/
```

**▪ SAM2 자동 라벨링 결과 통계**

| 데이터 분할 | 전체 객체 | 성공 | 스킵 | 성공률 |
| --- | --- | --- | --- | --- |
| train | 4,235 | 4,178 | 57 | **98.7%** |
| valid | 779 | 760 | 19 | **97.6%** |
| test | 295 | 290 | 5 | **98.3%** |

### 학습 데이터 샘플

![image.png](style/image/AutoMaskLeabeling/1.png)

- Detection 학습 데이터 — only bbox

![image.png](style/image/AutoMaskLeabeling/2.png)

- Segmentation 학습 데이터 — SAM2로 auto mask labeling

---

스킵 기준: 생성된 마스크 면적이 bbox 대비 지나치게 작거나 큰 경우 (품질 필터링)

전체 98% 이상의 높은 성공률로 자동 라벨링이 완료됐다.

![image.png](style/image/AutoMaskLeabeling/3.png)

## 🧠 실험 설계 (Experiment Design)

이전 Detection 실험과 **1:1 대응**되는 동일 구조로 3개 실험을 설계했다.

|  | Detection (이전) | Segmentation (이번) |
| --- | --- | --- |
| exp1 | yolov8n, Augment OFF | yolov8n-seg, Augment OFF |
| exp2 | yolov8n, Augment ON | yolov8n-seg, Augment ON |
| exp3 | yolov8s, Augment ON | yolov8s-seg, Augment ON |

공통 설정: Epochs 100, Early Stopping patience=20, Image Size 640, Batch 16

## 📊 결과 및 분석 (Results & Analysis)

**▪ Validation 세트 결과 (학습 중 최고값)**

| 실험 | 모델 | Box mAP50 | Mask mAP50 | 학습시간 |
| --- | --- | --- | --- | --- |
| Det exp1 | yolov8n | 0.7270 | - | 25.7분 |
| Det exp2 | yolov8n | 0.6617 | - | 26.1분 |
| Det exp3 | yolov8s | 0.7049 | - | 40.5분 |
| **Seg exp1** | yolov8n-seg | **0.7402** | 0.6434 | 36.0분 |
| Seg exp2 | yolov8n-seg | 0.6954 | 0.5998 | ~60분 |
| Seg exp3 | yolov8s-seg | 0.7190 | 0.6344 | ~70분 |

**▪ Test 세트 최종 성능 (best.pt 기준)**

| 지표 | Detection | Seg (Box) | Seg (Mask) | Det 대비 |
| --- | --- | --- | --- | --- |
| **mAP50** | 0.6960 | **0.7196** | 0.6243 | **+0.024 ↑** |
| mAP50-95 | 0.4533 | **0.5039** | 0.3982 | **+0.051 ↑** |
| Precision | 0.7132 | **0.7443** | 0.6917 | +0.031 ↑ |
| Recall | **0.6561** | 0.6539 | 0.6010 | -0.002 |

![image.png](style/image/AutoMaskLeabeling/4.png)

**▪ 클래스별 AP50 상세**

| 클래스 | Detection | Seg (Box) | Seg (Mask) |
| --- | --- | --- | --- |
| Bad Weld | 0.8910 | **0.8973** | 0.6563 |
| Good Weld | 0.7873 | 0.8245 | **0.8390** |
| **Defect** | 0.4099 | **0.4372** | 0.3775 |

![image.png](style/image/AutoMaskLeabeling/5.png)

**▪ 학습 곡선**

![image.png](style/image/AutoMaskLeabeling/6.png)

## Validation 정답 vs 예측 비교

### SAM2

![image.png](style/image/AutoMaskLeabeling/7.png)

![image.png](style/image/AutoMaskLeabeling/8.png)

### Detect

![image.png](style/image/AutoMaskLeabeling/9.png)

![image.png](style/image/AutoMaskLeabeling/10.png)

## 🔍 주요 발견 및 분석 (Insights & Analysis)

**▪ 발견 1: Seg Box mAP가 Detection보다 일관되게 우세**

3개 실험 모두에서 Seg 모델의 Box mAP가 Detection보다 높게 나왔다. 이는 마스크를 함께 학습하면 마스크 경계의 역전파 신호가 bbox localization에도 기여하기 때문이다. 즉, SAM2로 자동 생성한 마스크 라벨이 **공짜 정보로 작용해 bbox 정확도까지 끌어올린 셈**이다.

**▪ 발견 2: Augmentation이 이번에도 역효과**

이전 Detection 실험과 동일하게 Augmentation을 강화한 exp2, exp3이 exp1보다 낮은 성능을 보였다. 원인은 동일하다. 1,619장의 학습 데이터는 mosaic + mixup + copy_paste를 동시에 적용하기에 부족하고, 과도한 augmentation이 소규모 데이터셋에서 노이즈로 작용한다.

두 번의 실험(Detection, Segmentation)에서 동일한 결론이 반복됐다는 것이 이 가설을 더욱 강하게 뒷받침한다.

**▪ 발견 3: Mask mAP가 Box mAP보다 낮은 이유**

Mask mAP50(0.6243) < Box mAP50(0.7196)인 차이는 **라벨 품질 차이**에서 비롯된다.

```
bbox 라벨: 사람이 직접 그린 정확한 사각형
마스크 라벨: SAM2가 자동 생성 → 일부 경계 부정확

특히 Defect 클래스:
- 불규칙한 형태의 미세 결함
- SAM2가 용접 비드와 결함 경계를 혼동하는 경우 多
- Mask AP50 0.3775 (가장 낮음)
```

SAM2가 자동 생성한 마스크는 박스 내부 객체를 잘 잡지만, 미세한 결함 경계는 부정확할 수 있다. Defect 클래스의 낮은 Mask AP는 수작업 라벨과 자동 라벨의 품질 차이를 반영한다.

**▪ 발견 4: Good Weld는 Mask AP > Box AP**

Good Weld의 경우 Mask AP50(0.8390) > Box AP50(0.8245)로 역전됐다. 정상 용접 비드는 형태가 규칙적이어서 SAM2 마스크 품질이 높고, 마스크 경계가 명확해 모델이 더 쉽게 학습했기 때문이다.

## 🖼️ 시각적 비교

![image.png](style/image/AutoMaskLeabeling/11.png)

![image.png](style/image/AutoMaskLeabeling/12.png)

| 케이스 | 설명 |
| --- | --- |
| Bad Weld | 마스크가 불량 경계를 잘 잡은 성공 케이스 |
| Defect | bbox vs mask 차이가 명확한 케이스, SAM2 경계 불정확 |

## 🌐 서빙 구조 업데이트 (Serving)

FastAPI에 `mode` 파라미터를 추가해 **단일 서버에서 두 모델을 동시에 서빙**한다.

```
POST /predict?mode=det   → bbox JSON 반환
POST /predict?mode=seg   → bbox + 마스크 면적 JSON 반환
POST /predict/image?mode=det  → bbox 시각화 이미지
POST /predict/image?mode=seg  → 마스크 오버레이 이미지
```

Streamlit 사이드바에서 **Detection / Segmentation 라디오 버튼**으로 실시간 모드 전환이 가능하다.

![스크린샷 2026-03-12 오후 7.20.42.png](style/image/AutoMaskLeabeling/13.png)

![스크린샷 2026-03-12 오후 7.21.09.png](style/image/AutoMaskLeabeling/14.png)

![스크린샷 2026-03-12 오후 5.15.33.png](style/image/AutoMaskLeabeling/15.png)

### 검출 모드 선택 기능

![스크린샷 2026-03-12 오후 7.21.30.png](style/image/AutoMaskLeabeling/16.png)

## 🔍 인사이트 및 결론 (Insights & Conclusion)

**1. SAM2 자동 라벨링은 실용적인 Segmentation 데이터 구축 방법이다**

4,300개 마스크를 수작업 없이 코드 한 번으로 생성했다. 성공률 98% 이상으로 안정적이었으며, Good Weld처럼 형태가 규칙적인 클래스에서는 사람이 그린 것과 유사한 품질의 마스크를 얻을 수 있었다.

**2. 마스크 학습이 Box localization도 향상시킨다**

Seg 모델의 Box mAP가 Detection 모델보다 높게 나온 것은 예상 밖의 결과였다. 마스크 경계 학습이 bbox regression에 추가적인 지도 신호를 제공한다는 점에서, SAM2 자동 라벨링이 단순히 Segmentation을 위한 것이 아니라 Detection 성능 향상에도 기여할 수 있다.

**3. 자동 라벨 품질의 한계는 클래스 특성에 따라 다르다**

Defect처럼 불규칙하고 미세한 결함은 SAM2도 경계를 정확히 잡기 어렵다. Mask AP가 Box AP보다 크게 낮은 클래스는 자동 라벨 품질이 낮을 가능성이 높으며, 해당 클래스는 수동 검수를 추가하는 것이 효과적이다.

**4. 두 번의 실험에서 동일한 Augmentation 결론**

Detection과 Segmentation 실험 모두에서 baseline(Augment OFF)이 가장 좋은 성능을 보였다. 소규모 균일 데이터셋에서는 Augmentation 전략을 신중하게 선택해야 한다는 결론이 반복적으로 검증됐다.

## 🚀 추가 실험 / 개선 방향 (Further Work)

- **SAM2 파인튜닝**: 용접 도메인 데이터로 SAM2 자체를 파인튜닝하면 Defect 클래스 마스크 품질 향상 가능
- **능동 학습**: 모델이 틀린 케이스만 골라 수동 보정 후 재학습 → 최소한의 수작업으로 성능 극대화
- **YOLOv8m-seg**: 더 큰 모델로 데이터를 늘리지 않고 Defect 클래스 성능 개선 시도
- **마스크 면적 기반 불량 정량화**: 픽셀 단위 마스크로 불량 면적을 수치화하여 품질 등급화 가능
- **DepthPro 연결**: 2D Segmentation + 단일 이미지 깊이 추정으로 불량의 3D 위치 및 깊이 추정