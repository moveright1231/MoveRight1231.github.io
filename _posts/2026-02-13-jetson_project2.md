---
layout: post
title: 콜라캔 불량 탐지 — Jetson Nano 배포 삽질기
date: 2026-02-13 18:00:00 +0800
category: experiment
thumbnail: /style/image/jet_anomaly/title.JPG
icon: code
---

# 젯슨 나노 온디바이스 이상탐지 — 2편: Jetson 배포 삽질기

> 이전 포스트: [콜라캔 불량 탐지 — 데이터 수집부터 YOLOv8 학습까지](#)

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 목표 | 1편에서 학습한 모델을 Jetson Nano에서 실시간 추론 |
| 배포 방식 | ONNX × 2 → onnxruntime (Two-stage) |
| 카메라 | Logitech C270 USB 웹캠 (1280×720) |
| 추론 FPS | 5~7 FPS (Two-stage, CPU) |
| 주요 삽질 | onnxruntime 설치, HQ 카메라 드라이버, 도메인 갭, Two-stage 아키텍처 |

> 📸 **[사진 A]** Jetson Nano + C270 웹캠 + 콜라캔 전체 세팅 샷
![A](style/image/jet_anomaly/title.JPG)

![B](style/image/jet_anomaly/B.JPG)

## 💡 배경

1편에서 val accuracy 100%짜리 YOLOv8n-cls 모델을 만들었다. 이제 Jetson Nano B01에 올려서 캔을 카메라 앞에 가져다 대면 실시간으로 불량 여부를 판별하는 시스템을 만들 차례다.

결론부터 말하면, **val 100% 모델을 배포하는 건 끝이 아니라 시작이었다.** 이 편에서는 배포 과정에서 마주친 삽질과 그 해결 과정을 기록한다.

```
삽질 1: onnxruntime 설치 (Python 3.6 + aarch64 환경)
삽질 2: HQ 카메라 드라이버 (IMX477 I2C 오류)
삽질 3: 도메인 갭 — val 100% 모델이 실환경에서 틀리는 문제
삽질 4: Two-stage 아키텍처로 재설계
```

## ⚙️ Jetson Nano 환경

**▪ 스펙**

| 항목 | 내용 |
| --- | --- |
| 보드 | Jetson Nano B01 |
| JetPack | 4.6.6 (R32.7.6) |
| Python | 3.6 (ultralytics 사용 불가) |
| CUDA | 10.2 |
| GPU | Maxwell 128 CUDA cores |
| RAM | 4GB (모델 + 런타임 공유) |
| 카메라 | Logitech C270 USB 웹캠 |

![C](style/image/jet_anomaly/C.JPG)

Python 3.6이라는 게 핵심 제약이다. ultralytics는 Python 3.8 이상이 필요해서 Jetson에서는 쓸 수 없다. 그래서 1편에서 ONNX로 변환해두고, Jetson에서는 onnxruntime으로만 추론하는 방식을 택했다.

**▪ onnxruntime 설치**

`pip3 install onnxruntime`을 치면 x86 버전이 설치돼서 Jetson에서 실행이 안 된다. aarch64용 wheel 파일을 따로 받아야 한다.

```bash
# Jetson Nano aarch64 + Python 3.6용 wheel
pip3 install onnxruntime-1.11.0-cp36-cp36m-linux_aarch64.whl
```

## 🔧 삽질 1 — HQ 카메라 드라이버

처음에는 Pi 카메라(IMX219)를 쓰다가 화질이 아쉬워서 Raspberry Pi HQ Camera(IMX477)로 교체를 시도했다.

![E](style/image/jet_anomaly/E.JPG)

**▪ 증상: No cameras available**

카메라 교체 후 GStreamer 파이프라인에서 `No cameras available` 에러가 발생했다.

```bash
$ dmesg | grep -i "imx"
imx477 7-001a: imx477_board_setup: error during i2c read probe (-121)
imx477: probe of 7-001a failed with error -121
```

에러 코드 -121은 I2C 원격 통신 실패(EREMOTEIO)다. 드라이버는 올바르게 설치됐는데 카메라가 I2C 신호 자체에 응답하지 않는 것이다.

Arducam 드라이버를 설치해도 같은 에러. 원인을 파고들었더니 케이블 호환성 문제로 추정됐다. Jetson Nano는 15핀 CSI 커넥터를 사용하는데 HQ 카메라에 동봉된 케이블이 RPi4용 22핀이라 물리적으로 호환이 안 된다.

**▪ 드라이버 설치 후 기존 카메라도 먹통**

설상가상으로 IMX477 드라이버가 기존 IMX219도 인식 못하게 됐다. Arducam 드라이버가 IMX477 I2C 주소(0x1a)만 탐색하면서 IMX219(0x10)를 무시하기 때문이다.

```bash
# 복구 방법
sudo dpkg -r arducam-nvidia-l4t-kernel
sudo apt-get install --reinstall nvidia-l4t-kernel nvidia-l4t-kernel-dtbs
sudo reboot
```

**▪ 결론: USB 웹캠으로 전환**

케이블 문제를 해결하는 데 시간이 너무 소요돼서 일단 USB 웹캠(Logitech C270)으로 전환했다. USB 웹캠은 드라이버 설치 없이 `/dev/video0`으로 바로 잡힌다.

![D](style/image/jet_anomaly/D.JPG)

## 🔧 삽질 2 — 도메인 갭

**이것이 이번 편의 핵심 삽질이다.** val 100%인데 실환경에서 틀린다.

**▪ 1차 증상: 모든 캔이 crack**

모델을 Jetson에 올리고 USB 웹캠으로 캔을 찍었더니 클래스에 관계없이 전부 crack으로 판별됐다.

```
good 캔  → crack 47%
open 캔  → crack 47%
line 캔  → crack 47%
```

confidence까지 같다. 모델이 입력을 무시하고 crack을 기본값으로 출력하는 것이다.

**▪ 원인: 학습 데이터(아이폰)와 추론 환경(C270 웹캠)의 도메인 차이**

| 항목 | 아이폰 | C270 웹캠 |
| --- | --- | --- |
| 해상도 | 3024×4032 | 1280×720 |
| 선명도 | 매우 선명 | 보통 |
| 화각 | 일정 거리 클로즈업 | 고정 거리 |
| 색온도 | 자동 보정 | 고정 |

**▪ 1차 시도: 도메인 증강 재학습**

Pi 카메라 특성을 시뮬레이션하는 도메인 증강(블러/노이즈/압축 등)을 적용해 재학습했다. 하지만 C270은 Pi 카메라가 아니라 이 접근법은 오히려 역효과가 났다.

![도메인 증강 비교](style/image/jet_anomaly/domain_aug_comparison.JPG)
*Pi 카메라 시뮬레이션 도메인 증강 — C270에는 맞지 않았다*

**▪ 2차 시도: 웹캠 이미지로 재학습**

C270으로 각 클래스 30장씩 직접 촬영해서 기존 학습 데이터에 추가했다. val accuracy 100% 달성. 그런데도 실환경에서 good이 crack/line으로 오분류됐다.

```python
# 4-class 단일 모델 결과
good 캔  → crack 47%  ← 여전히 틀림
open 캔  → open 41%   ← 정확
crack 캔 → crack 46%  ← 정확
```

데이터를 더 추가해도 같은 패턴이 반복됐다. val은 100%인데 실환경에서는 good이 잡히지 않는다.

**▪ 원인 분석: val 100%는 무의미했다**

학습/val 세트의 웹캠 이미지가 동일한 원본에서 나왔다(증강만 달랐다). 모델이 실제 일반화 능력을 검증받지 못한 것이다. val 100%는 같은 이미지를 다른 방식으로 보여줬을 뿐이었다.

## 🔧 삽질 3 — Two-stage 아키텍처로 재설계

4-class 단일 모델의 한계를 인정하고 아키텍처를 바꿨다.

**▪ 설계 아이디어**

```
기존: 4-class 단일 모델
    good / crack / line / open → 한번에 분류
    ↓ 문제: good이 다른 불량 클래스와 혼동됨

변경: Two-stage 파이프라인
    Stage 1: 이진 분류 → good? or defect?
    Stage 2: 결함 종류 분류 → crack? line? open?
```

"정상인가 아닌가"와 "어떤 결함인가"를 분리하면 각각 더 단순한 문제가 된다.

**▪ Two-stage 추론 흐름**

```
카메라 프레임 (1280×720 BGR)
    ↓
HoughCircles → 캔 윗면 크롭 (실패 시 전체 프레임 폴백)
    ↓
preprocess: resize 640×640 → BGR→RGB → /255.0 → CHW → (1,3,640,640)
    ↓
Stage 1: stage1.onnx  [input: (1,3,640,640) → output: (1,2) logits]
    ↓ softmax → [p_defect, p_good]
    ↓ EMA 스무딩 적용
    │
    ├── p_defect < 0.50 → NORMAL 표시 (Stage 2 생략)
    │
    └── p_defect ≥ 0.50
            ↓
        Stage 2: stage2.onnx  [input: (1,3,640,640) → output: (1,3) logits]
            ↓ softmax → [p_crack, p_line, p_open]
            ↓ argmax → 결함 종류
            ↓
        DEFECT + type(crack/line/open) 표시
```

![G](style/image/jet_anomaly/G.JPG)

**▪ 모델 아키텍처 — YOLOv8n-cls**

두 모델 모두 YOLOv8n-cls(nano classification)를 베이스로 한다. 학습은 ImageNet pretrained 가중치에서 시작하는 **전이학습(Transfer Learning)** 방식이다.

| 항목 | 내용 |
| --- | --- |
| 베이스 모델 | YOLOv8n-cls (ImageNet pretrained) |
| 파라미터 수 | 1,440,004 |
| 연산량 | 3.3 GFLOPs |
| 입력 | (1, 3, 640, 640) float32 |
| Stage 1 출력 | (1, 2) — [defect, good] |
| Stage 2 출력 | (1, 3) — [crack, line, open] |
| ONNX opset | 11 (Jetson onnxruntime 1.11 호환) |
| 모델 크기 | 5.5MB × 2 |

YOLOv8n은 분류 헤드 앞단에 CSP(Cross Stage Partial) 기반 백본을 사용한다. Pretrained 백본이 이미 엣지/텍스처 등 저수준 특징을 학습하고 있어서, 소량의 새 데이터로도 빠르게 수렴한다.

**▪ 학습 설정**

```python
model.train(
    data     = "dataset_twostage/binary",  # Stage 1
    epochs   = 50,
    imgsz    = 640,
    batch    = 16,
    device   = "0",       # GPU
    patience = 10,        # EarlyStopping
)
```

| 하이퍼파라미터 | 값 | 이유 |
| --- | --- | --- |
| imgsz | 640 | 학습/추론 일관성 |
| batch | 16 | RTX 3060 12GB 기준 |
| patience | 10 | 과적합 방지, 조기 종료 |
| optimizer | AdamW (YOLOv8 기본) | 소량 데이터에 안정적 |

**▪ 데이터 분리 — 누수 없는 train/val 설계**

이전 시도에서 val 100%가 나왔지만 실환경에서 틀린 이유가 **데이터 누수** 때문이었다. 같은 원본 이미지의 증강본이 train에도, val에도 들어간 것이다.

```
수정 전 (누수 있음):
  data_webcam/good/img_001.JPG → 증강 → train/
  data_webcam/good/img_001.JPG → val/   ← 동일 원본!

수정 후 (누수 없음):
  data_webcam/good/ 90장 → 셔플 → 84장(train) + 6장(val) 완전 분리
                                      ↓
                              augmentation 적용 → 120장
```

val 세트에는 증강을 적용하지 않은 **원본 이미지만** 사용한다. 이것이 실환경에 가장 가까운 평가 조건이다.

**▪ 전처리 파이프라인**

추론 시 입력 이미지는 다음 단계를 거쳐 ONNX 모델 입력 텐서로 변환된다.

```python
def preprocess(img):
    img = cv2.resize(img, (640, 640))           # 1. 640×640 리사이즈
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 2. BGR → RGB (OpenCV 기본이 BGR)
    img = img.astype(np.float32) / 255.0        # 3. [0,255] → [0.0,1.0] 정규화
    img = np.transpose(img, (2, 0, 1))          # 4. HWC → CHW (PyTorch 텐서 포맷)
    img = np.expand_dims(img, axis=0)           # 5. CHW → BCHW (배치 차원 추가)
    return img  # shape: (1, 3, 640, 640)
```

모델 출력은 raw logit이므로 softmax를 수동 적용해 확률로 변환한다.

```python
def softmax(x):
    e = np.exp(x - np.max(x))  # overflow 방지를 위해 max 빼기
    return e / e.sum()

logits = sess.run([output_name], {input_name: inp})[0][0]  # shape: (2,) or (3,)
probs  = softmax(logits)  # 합이 1인 확률 분포
```

**▪ 데이터 설계**

Stage 1 학습 데이터에서 클래스 균형이 핵심이었다.

```
Stage 1 (good vs defect):
  good   train: 120장
  defect train: 120장 (crack 40 + line 40 + open 40)  ← 균형 맞춤

Stage 2 (crack/line/open):
  crack  train: 120장
  line   train: 120장
  open   train: 120장
```

처음에 defect train을 360장(각 120장)으로 했더니 good:defect = 1:3 불균형으로 모델이 defect 쪽으로 편향됐다. 균형을 맞추고 나서 해결됐다.

**▪ good 이미지 부족 문제**

Stage 1에서 good 30장으로는 모델이 계속 붕괴했다. 모든 입력에 동일한 확률(0.73 defect)을 출력하는 완전 붕괴 상태였다.

```
[진단] 모든 클래스에서 동일한 s1_defect=0.73 출력
  → 원인: 24장(30장 중 val 6장 제외)만으로는 의미 있는 특징 학습 불가
  → 해결: good 이미지 90장으로 보강 (거리/각도/조명/캔 방향 다양화)
```

90장으로 늘린 후 Stage 1이 정상 작동했다.

| 클래스 | s1_defect | 판정 |
| --- | --- | --- |
| good | 0.27 | → good ✅ |
| open | 0.73 | → defect ✅ |
| crack | 0.61 | → defect ✅ |
| line | 0.72 | → defect ✅ |

**▪ 시간축 스무딩 (EMA)**

Stage 1 결과가 프레임마다 조금씩 흔들려서 good/defect가 번갈아 나오는 flickering이 발생했다. 지수이동평균(EMA)을 적용해 해결했다.

```python
s1_smooth = 0.4 * probs1 + 0.6 * s1_smooth_prev
is_defect = s1_smooth[defect_idx] >= 0.50
```

이전 프레임들의 평균이 유지되기 때문에 한두 프레임 흔들려도 결과가 바뀌지 않는다.

## ⚡ 최종 추론 성능


![H](style/image/jet_anomaly/H.JPG)

![I](style/image/jet_anomaly/I.JPG)

**▪ 성능 측정 (USB 웹캠, CPU 추론)**

| 항목 | 값 |
| --- | --- |
| 해상도 | 1280×720 |
| 모델 | stage1.onnx (5.5MB) + stage2.onnx (5.5MB) |
| 추론 디바이스 | CPU (onnxruntime) |
| FPS (good 판정) | **~7 FPS** (Stage 1만 실행) |
| FPS (defect 판정) | **~5 FPS** (Stage 1 + Stage 2) |
| 클래스 정확도 | good ✅  crack ✅  line ✅  open ✅ |

**▪ 클래스별 추론 데모**

![gif1](style/image/jet_anomaly/good.gif)

![gif2](style/image/jet_anomaly/crack.gif)

![gif3](style/image/jet_anomaly/line.gif)

![gif4](style/image/jet_anomaly/open.gif)
**▪ 실행 방법**

```bash
# 기본 실행 (화면 출력)
python3 inference_jetson.py \
    --stage1 stage1.onnx --stage2 stage2.onnx --usb

# 헤드리스 (모니터 없을 때)
python3 inference_jetson.py \
    --stage1 stage1.onnx --stage2 stage2.onnx --usb --no-show

# 크롭 비활성화 (FPS 우선)
python3 inference_jetson.py \
    --stage1 stage1.onnx --stage2 stage2.onnx --usb --no-crop
```

**▪ HoughCircles 캔 크롭**

1280×720 전체 프레임에서 추론하면 배경(테이블, 손 등)이 포함돼 노이즈가 된다. 캔 윗면이 원형이라는 특성을 이용해 Hough Circle Transform으로 원을 검출하고 해당 영역만 크롭해서 분류한다.

```python
def detect_can_crop(frame, padding=30):
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9,9), 2)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,                      # 누산기 해상도 (1.0=원본, 클수록 빠르지만 부정확)
        minDist=min(h,w)*0.3,        # 원 중심 간 최소 거리 (다중 검출 방지)
        param1=100,                  # Canny 엣지 검출 상한 임계값
        param2=40,                   # 원 검출 누산기 임계값 (낮을수록 더 많이 검출)
        minRadius=int(min(h,w)*0.05),
        maxRadius=int(min(h,w)*0.40),
    )
    if circles is None:
        return None, None   # 실패 시 전체 프레임 폴백
    x, y, r = np.round(circles[0][0]).astype(int)
    return frame[y-r-padding:y+r+padding, x-r-padding:x+r+padding], (x, y, r)
```

검출된 원의 중심과 반지름으로 bounding square를 만들고 padding을 더해 크롭한다. 감지에 실패해도 전체 프레임으로 폴백하기 때문에 추론이 중단되지 않는다.

![F](style/image/jet_anomaly/F.JPG)

## 🔍 인사이트

**1. val accuracy와 실환경 성능은 다르다**

val 100% 모델이 실환경에서 틀렸다. 학습/추론 도메인이 다르면 val 점수는 무의미하다. val 세트에 실환경 데이터가 포함돼야 의미 있는 지표가 된다.

**2. 데이터 도메인이 맞지 않으면 아키텍처 변경보다 데이터 수집이 먼저다**

모델 구조를 바꾸기 전에, 실제 배포 환경과 같은 카메라/조명/거리 조건으로 찍은 데이터가 먼저 있어야 한다. 아무리 좋은 아키텍처도 도메인이 다른 데이터로는 작동하지 않는다.

**3. 어려운 문제는 분해하면 쉬워진다**

"4 클래스 중 하나를 맞춰라"보다 "정상인가 아닌가 → 어떤 결함인가" 두 단계로 나누는 것이 훨씬 쉽다. Two-stage 접근은 모델에게 한 번에 너무 많은 것을 요구하지 않는다는 점에서 실용적이다.

**4. 엣지 배포는 환경 파악이 먼저다**

Python 버전, 드라이버, 패키지 설치 방법이 PC와 전혀 다르다. Jetson에서 ultralytics 불가, `pip install onnxruntime` 불가, CSI 카메라는 GStreamer 없이 불가. 환경 제약을 먼저 파악하고 거기에 맞춰 설계해야 삽질을 줄일 수 있다.

## 🚀 마무리

직접 찍은 콜라캔 사진 294장으로 시작해서, val 100%짜리 모델을 만들고, Jetson에 올렸다가 실환경에서 틀리는 걸 발견하고, 도메인 갭을 분석하고, 웹캠 데이터를 직접 수집하고, Two-stage 아키텍처로 재설계해서 결국 4개 클래스 모두 정상 작동하는 시스템을 완성했다.

```
전체 작업 흐름:
1편 — 데이터 294장 직접 수집
    → 증강 600장 (albumentations)
    → YOLOv8n-cls 학습 (val 100%)
    → GradCAM 시각화
    → ONNX export

2편 — Jetson 배포
    → onnxruntime aarch64 설치
    → HQ 카메라 드라이버 삽질 → USB 웹캠으로 전환
    → 도메인 갭 발견 → 웹캠 데이터 수집 (120장)
    → Two-stage 재설계 (good vs defect → 결함 종류)
    → EMA 스무딩으로 flickering 제거
    → 최종 5~7 FPS 실시간 추론 완성
```

val accuracy가 높다고 끝이 아니다. 실환경 데이터로 검증하고, 도메인 갭을 확인하고, 필요하면 구조를 바꿀 준비가 돼 있어야 한다. 이걸 처음부터 끝까지 직접 겪은 게 이번 프로젝트의 핵심 경험이다.
