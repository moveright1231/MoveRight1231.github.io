---
layout: post
title: 콜라캔 불량 탐지 — 데이터 수집부터 YOLOv8 학습까지
date: 2026-02-11 12:00:00 +0800
category: experiment
thumbnail: /style/image/jet_anomaly/class_samples.png
icon: code
---

# 젯슨 나노 온디바이스 이상탐지 — 1편: 데이터 수집부터 학습까지

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 목표 | 콜라캔 불량 탐지 모델을 직접 만들어 Jetson Nano에 실시간 배포 |
| 모델 | YOLOv8n-cls (분류, 1.44M params) |
| 클래스 | crack / good / line / open (4종) |
| 데이터 | 직접 촬영, 총 294장 → 증강 후 600장 |
| 배포 | ONNX → Jetson Nano B01 onnxruntime 추론 |

## 💡 배경 및 아이디어 (Background & Motivation)

**▪ 왜 콜라캔인가**

이상탐지 프로젝트를 하려면 데이터가 필요하다. MVTec처럼 공개 데이터셋을 쓰는 방법도 있지만, 처음부터 끝까지 직접 데이터를 모아서 모델을 만들어보고 싶었다. 집 냉장고에 콜라캔이 쌓여 있었고, 찌그러뜨리고 선을 그어서 불량품을 직접 만들기로 했다.

**▪ 클래스 설계**

불량 유형을 현실적으로 정의했다.

| 클래스 | 설명 | 제조 방법 |
| --- | --- | --- |
| `good` | 정상 캔 | 그냥 찍음 |
| `crack` | 균열/찌그러짐 | 캔을 손으로 눌러서 변형 |
| `line` | 선 불량 | 네임펜으로 캔 표면에 선 표시 |
| `open` | 뚜껑 열림 | 탭을 따서 열린 상태 |

Detection(검출)이 아닌 Classification(분류)을 선택한 이유는, 캔 전체 이미지에서 "이 캔이 어떤 상태인가"를 맞히는 게 목적이기 때문이다. bbox 라벨 작업 없이도 폴더 구조만으로 학습이 가능한 YOLOv8n-cls가 가장 적합했다.

**▪ 전체 파이프라인**

```
데이터 수집 (직접 촬영, 294장)
    ↓
전처리 (HEIC 변환, 방향 통일)
    ↓
오프라인 증강 (albumentations, 각 150장)
    ↓
train/val 분리 (8:2, stratify, seed=42)
    ↓
YOLOv8n-cls 학습 (GPU, epochs=100, patience=10)
    ↓
평가 + GradCAM 시각화
    ↓
ONNX export (opset=11)
    ↓
Jetson Nano 실시간 추론 (2편)
```

## 📦 데이터 수집

**▪ 원본 데이터 현황**

| 클래스 | 장수 | 포맷 | 해상도 |
| --- | --- | --- | --- |
| crack | 72장 | JPG | 3024×4032 (세로) |
| good | 102장 | HEIC | 3024×4032 (세로) |
| line | 59장 | JPG | 3024×4032 (세로) |
| open | 61장 | JPG | 4032×3024 (가로) |
| **합계** | **294장** | | |

촬영 환경은 스마트폰 카메라 하나다. 배경, 조명, 각도를 다양하게 바꿔가며 촬영했다. 클래스당 60~100장 수준으로, 딥러닝 학습 데이터로는 매우 적은 양이다.

![클래스별 샘플 이미지](style/image/jet_anomaly/class_samples.png)

## ⚙️ 전처리 과정

### 이슈 1 — HEIC 포맷

good 폴더 102장이 iPhone으로 촬영한 HEIC 파일이었다. OpenCV는 HEIC를 직접 읽지 못한다.

```bash
# ImageMagick으로 일괄 변환
for f in data/good/*.HEIC; do
    convert "$f" "${f%.HEIC}.jpg"
done
```

변환 후 원본 HEIC 파일 삭제. 이후 모든 클래스가 JPG로 통일됐다.

### 이슈 2 — 방향 불일치

crack/good/line은 세로(3024×4032)로 촬영됐는데, open만 가로(4032×3024)였다. 아마 뚜껑을 따는 자세가 달라서 자동으로 방향이 바뀐 것 같다.

```python
from PIL import Image
import os

for f in sorted(Path("data/open").glob("*.jpg")):
    img = Image.open(f)
    img = img.rotate(-90, expand=True)   # 시계 반대 방향 90° → 세로
    img.save(f)
```

원본 61장 전부 세로 방향으로 통일했다. 방향을 맞추지 않으면 증강 시 RandomCrop 비율이 클래스마다 달라져서 학습에 혼선이 생긴다.

### 이슈 3 — 클래스 불균형

| 클래스 | 장수 | 비율 |
| --- | --- | --- |
| crack | 72장 | 1.0x (최소) |
| good | 102장 | 1.42x |
| line | 59장 | 0.82x |
| open | 61장 | 0.85x |

최대 1.7배 차이가 났다. 추가 촬영 대신 오프라인 증강으로 전 클래스를 150장으로 맞추기로 했다.

## 🔧 증강 파이프라인

albumentations 2.x 기반으로 클래스 특성에 맞는 파이프라인을 따로 설계했다.

**▪ 핵심 설계 결정: line 클래스는 별도 파이프라인**

`line` 클래스는 캔 표면에 네임펜으로 직접 그은 선이 특징이다. 상하반전이나 큰 회전을 적용하면 선의 방향이 뒤집혀서 "선이 있다"는 특징보다 "방향이 이상하다"는 노이즈가 더 강해진다. 또한 어둡게 하는 증강을 강하게 적용하면 선이 배경에 묻혀버린다.

```python
# line 클래스 — 방향성 결함 특성상 제약
line_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.0),           # 90° 회전 제외
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.03,
                       rotate_limit=10, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.05, 0.2),  # 어둡게는 거의 안 함
        contrast_limit=0.15, p=0.7),
    A.GaussNoise(std_range=(0.02, 0.1), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomResizedCrop(size=(640, 640), scale=(0.85, 1.0), p=0.5),
    A.Resize(640, 640),
])

# 나머지 클래스 — 방향 제약 없음
general_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                       rotate_limit=15, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=(-0.1, 0.3),
        contrast_limit=0.25, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10,
                         sat_shift_limit=20,
                         val_shift_limit=8, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.12), p=0.4),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomResizedCrop(size=(640, 640), scale=(0.85, 1.0), p=0.5),
    A.Resize(640, 640),
])
```

**▪ 증강 결과**

| 클래스 | 원본 | 증강 후 | 증강 생성 |
| --- | --- | --- | --- |
| crack | 72장 | 150장 | 78장 |
| good | 102장 | 150장 | 48장 |
| line | 59장 | 150장 | 91장 |
| open | 61장 | 150장 | 89장 |
| **합계** | **294장** | **600장** | |

원본 이미지는 640×640으로 리사이즈 후 복사, 부족한 수량만큼 증강 이미지를 추가 생성했다.

![증강 전후 비교](style/image/jet_anomaly/augmentation_comparison.png)

**▪ albumentations 2.x API 변경 주의사항**

```python
# 1.x (구버전)
A.GaussNoise(var_limit=(10.0, 50.0))   # 픽셀값 기준 분산

# 2.x (신버전) — 0~1 정규화 기준 표준편차로 변경
A.GaussNoise(std_range=(0.02, 0.12))

# RandomResizedCrop도 변경
# 1.x: A.RandomResizedCrop(height=640, width=640, scale=(0.85, 1.0))
# 2.x: A.RandomResizedCrop(size=(640, 640), scale=(0.85, 1.0))
```

## 🏋️ YOLOv8n-cls 학습

**▪ train/val 분리**

```python
from sklearn.model_selection import train_test_split

train_files, val_files = train_test_split(
    files, test_size=0.2, random_state=42
)
```

stratify 없이 단순 8:2 분리. 클래스당 150장 → train 120장 / val 30장.

```
dataset/
├── train/  (480장: crack 120 / good 120 / line 120 / open 120)
└── val/    (120장: crack 30  / good 30  / line 30  / open 30)
```

**▪ 학습 설정**

```python
model = YOLO("yolov8n-cls.pt")     # 사전학습 가중치 사용
model.train(
    data="dataset/",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,                       # RTX 3060
    patience=10,                    # EarlyStopping
    workers=0,                      # WSL2 multiprocessing 오류 방지
)
```

`workers=8`로 설정했다가 WSL2 환경에서 `ConnectionResetError`가 발생하며 GPU 점유율이 0%로 떨어지는 문제가 있었다. `workers=0` (단일 스레드)으로 변경하니 해결됐다.

**▪ 1차 학습 결과**

| 항목 | 값 |
| --- | --- |
| 학습 epoch | 42 (best @ 32, EarlyStopping 발동) |
| 소요 시간 | 약 15분 |
| **best val top1_acc** | **0.992 (99.2%)** |
| best.pt 크기 | 3.0 MB |

99.2% 정확도가 나왔지만 Confusion Matrix를 보니 `line → crack` 오분류가 1건 있었다.

![학습 곡선](style/image/jet_anomaly/results.png)

**▪ 1차 실패 원인 분석 및 2차 학습**

실제로 Jetson에서 추론해보니 `good` 캔도 `crack`으로 분류하는 케이스가 발생했다. 원인을 분석했다.

```
원인 1: crack 원본 데이터 퀄리티 문제
    → crack 72장 중 일부가 흐릿하거나 찌그러짐이 미미한 이미지
    → 44장 추가 촬영 (더 명확한 찌그러짐)

원인 2: 어둡게 하는 증강이 너무 강함
    → brightness_limit=(-0.3, 0.3) → (-0.1, 0.3)으로 완화
    → line 클래스는 (-0.05, 0.2)로 별도 관리

원인 3: scale 증강이 캔 형태 왜곡
    → scale_limit=0.15 → 0.05로 대폭 축소
```

44장 추가 후 data_aug 재생성 → train/val 재분리 → 재학습:

| 항목 | 1차 | 2차 (최종) |
| --- | --- | --- |
| crack 원본 | 72장 | **116장 (+44)** |
| 학습 epoch | 42 (best @ 32) | 37 (best @ 27) |
| **best val top1_acc** | 0.992 | **1.000 (100%)** |
| 오분류 | 1건 (line→crack) | **0건** |

**▪ 클래스별 최종 평가 (val 120장)**

| 클래스 | Precision | Recall | F1 |
| --- | --- | --- | --- |
| crack | 1.000 | 1.000 | 1.000 |
| good | 1.000 | 1.000 | 1.000 |
| line | 1.000 | 1.000 | 1.000 |
| open | 1.000 | 1.000 | 1.000 |
| **전체** | | | **1.000** |

![Confusion Matrix](style/image/jet_anomaly/confusion_matrix_normalized.png)

## 🔍 GradCAM 시각화

모델이 어느 부분을 보고 판단하는지 확인하기 위해 GradCAM을 적용했다. YOLOv8n-cls의 마지막 C2f 레이어(index 8)를 타겟으로 설정했다.

```python
from pytorch_grad_cam import GradCAM

# YOLOv8은 (logits, logits) tuple 반환 → 래퍼 필요
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

target_layer = [pt_model.model[8]]   # 마지막 C2f
cam = GradCAM(model=YOLOWrapper(pt_model), target_layers=target_layer)
```

**▪ GradCAM 이슈 — fused 모델 gradient 비활성화**

YOLOv8은 추론 시 레이어를 fusion하는데, fused 모델은 기본적으로 `requires_grad=False`다. GradCAM이 gradient를 역전파할 수 없어 `AttributeError`가 발생했다.

```python
# 해결: fused 모델 파라미터에 gradient 활성화
for p in pt_model.parameters():
    p.requires_grad_(True)
```

![GradCAM crack](style/image/jet_anomaly/crack.png)
![GradCAM good](style/image/jet_anomaly/good.png)
![GradCAM line](style/image/jet_anomaly/line.png)
![GradCAM open](style/image/jet_anomaly/open.png)

crack 클래스에서는 찌그러진 부분, line 클래스에서는 선이 그어진 영역, open 클래스에서는 열린 탭 부분에 히트맵이 집중됐다. 모델이 실제로 의미 있는 특징을 학습했음을 확인했다.

## 🔧 ONNX Export

Jetson Nano는 Python 3.6 환경이라 ultralytics를 사용할 수 없다. onnxruntime으로 추론하기 위해 ONNX로 변환했다.

```python
model = YOLO("runs/cola_cls/weights/best.pt")
model.export(
    format="onnx",
    opset=11,       # Jetson Nano onnxruntime 호환
    imgsz=640,
    dynamic=False,  # 고정 입력 크기
    simplify=True,  # onnx-simplifier 적용
)
```

| 항목 | 값 |
| --- | --- |
| 입력 shape | (1, 3, 640, 640) |
| 출력 shape | (1, 4) |
| 파일 크기 | 5.52 MB |
| opset | 11 |

onnxruntime으로 추론 검증까지 완료했다:

```python
sess = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)
output = sess.run(None, {sess.get_inputs()[0].name: dummy})
# 출력 shape: (1, 4) ✅
```

## 🔍 인사이트 및 결론 (Insights & Conclusion)

**1. 직접 만든 소규모 데이터셋으로도 100% val accuracy 달성 가능하다**

클래스당 60~100장의 스마트폰 사진으로 시작했다. 증강으로 150장을 만들고, 오분류 원인을 분석해서 데이터와 파이프라인을 개선하는 과정을 2차례 반복했다. 데이터 양보다 **데이터 품질과 증강 설계**가 더 중요했다.

**2. 클래스 특성에 맞는 증강 파이프라인이 필요하다**

모든 클래스에 동일한 증강을 적용하는 건 단순하지만 최적이 아니다. `line` 클래스처럼 방향성이 있는 결함은 상하반전/큰 회전을 제외하고, 어두운 증강도 최소화해야 특징이 유지된다. 클래스별로 생각하고 설계하는 과정이 정확도 향상에 실질적인 도움이 됐다.

**3. val 100%가 실환경에서의 100%를 보장하지 않는다**

val 셋이 아이폰으로 촬영한 정지 이미지와 동일한 도메인이라서 100%가 나온 것이다. 실제 Jetson에 Pi 카메라를 달고 추론하면 도메인 갭(블러, 노이즈, 색온도 차이)으로 인해 성능이 달라진다. 이 문제는 2편에서 다룬다.

**4. WSL2 환경 특이사항**

YOLOv8의 `workers` 파라미터 기본값은 멀티프로세싱을 사용하는데, WSL2에서 소켓 기반 데이터 로더가 충돌한다. `workers=0`으로 설정하면 단일 스레드로 동작해서 문제가 없다. WSL2 사용자라면 반드시 체크해야 할 설정이다.

## 🚀 다음 편 예고

2편에서는 학습된 모델을 Jetson Nano에 실제로 올리는 과정을 다룬다.

```
2편 주요 내용:
├── Jetson Nano 환경 세팅 (Python 3.6, onnxruntime, OpenCV)
├── 실시간 추론 스크립트 작성
├── 도메인 갭 문제 발견 및 분석
│   └── 아이폰 학습 모델 → Pi 카메라 실환경 오분류
├── 도메인 증강 재학습 (Pi 카메라 시뮬레이션)
│   └── GaussianBlur, ImageCompression, RandomShadow 등
└── HQ 카메라 교체 삽질기
```

val 100% 모델이 실환경에서 틀리는 이유, 그 갭을 데이터로 메우는 방법을 정리할 예정이다.
