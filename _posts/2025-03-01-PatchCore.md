---
layout: post
title: PatchCore 기반 비지도 이상 탐지 시스템 (MVTec AD)
date: 2025-03-01 15:00:00 +0800
category: experiment
thumbnail: /style/image/patchcore/1.png
icon: code
---


# anomaly-detect Post

# PatchCore 기반 비지도 이상 탐지 시스템

> 이전 포스트: [SAM2 자동 라벨링 + YOLOv8-seg](https://moveright1231.github.io/2025-02-04-AutoMaskLabeling)
> 

## 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| 프로젝트명 | PatchCore 기반 비지도 산업 표면 결함 탐지 시스템 |
| 알고리즘 | PatchCore (Memory Bank 기반 이상 탐지) |
| Backbone | ResNet50 (ImageNet pretrained) |
| 데이터셋 | MVTec Anomaly Detection Dataset (15개 카테고리) |
| 환경 | RTX 3060 12GB, PyTorch 2.5.1+cu121, CUDA 13.1, Python 3.10 |
| 서빙 | FastAPI + Streamlit 웹 애플리케이션 |

## 목적 (Objective)

**▪ 실험의 목표**

이전 Welding Defect 프로젝트는 정상/불량 라벨이 모두 필요한 지도학습 기반이었다. 이번 프로젝트는 **정상 이미지만으로 학습하여 어떤 불량도 탐지**하는 비지도학습 방식으로, 완전히 다른 접근법을 시도한다.

**▪ 얻고자 하는 인사이트**

- 라벨 없이 정상 데이터만으로 얼마나 정확하게 불량을 탐지할 수 있는지
- Pretrained 피처의 중간 레이어(layer2+3)가 이상 탐지에 얼마나 효과적인지
- Coreset Subsampling이 Memory Bank 크기와 성능에 미치는 영향
- 15개 카테고리 전체에서의 성능 일관성과 카테고리별 특성 차이

## 💡 배경 및 아이디어 (Background & Motivation)

**▪ 왜 Anomaly Detection인가**

실제 제조 현장에서 불량은 드물게 발생하기 때문에 불량 데이터를 충분히 수집하기 어렵다. Welding 프로젝트처럼 bbox 라벨을 직접 만드는 것도 큰 비용이다. 반면 정상 제품 이미지는 얼마든지 수집할 수 있다.

Anomaly Detection은 이 현실적인 문제를 해결한다. 정상 이미지만 학습한 모델이 "이건 정상과 다르다"고 판단하는 방식으로, 라벨링 비용 없이 새로운 종류의 불량도 탐지할 수 있다.

**▪ 왜 PatchCore인가**

PatchCore는 2022년 CVPR에서 발표된 알고리즘으로, MVTec AD 벤치마크에서 SOTA 성능을 달성했다. 핵심 아이디어는 단순하다.

```
1. 별도 학습 없이 ImageNet pretrained ResNet50의 중간 레이어 피처 활용
2. 정상 이미지의 패치 피처를 Memory Bank에 저장
3. 테스트 시 Memory Bank와의 거리로 이상도 측정
```

역전파 없이 피처를 저장하는 방식이라 학습이 매우 빠르고, 새로운 정상 데이터가 추가되면 Memory Bank만 업데이트하면 된다.

## 📦 데이터셋 (Dataset)

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | MVTec Anomaly Detection Dataset |
| 출처 | Kaggle (ipythonx/mvtec-ad) |
| 라이선스 | CC BY-NC-SA 4.0 (연구 목적) |
| 카테고리 수 | 15개 (texture 5개 + object 10개) |
| 총 이미지 | train 정상 3,409장 / test 정상 467장 / test 불량 1,242장 |

**카테고리별 데이터 구성**

| 카테고리 | 학습(정상) | 테스트(정상) | 테스트(불량) | 불량 종류 |
| --- | --- | --- | --- | --- |
| bottle | 209 | 20 | 63 | 3 |
| cable | 224 | 58 | 92 | 8 |
| capsule | 219 | 23 | 109 | 5 |
| carpet | 280 | 28 | 89 | 5 |
| grid | 264 | 21 | 57 | 5 |
| hazelnut | 391 | 40 | 70 | 4 |
| leather | 245 | 32 | 92 | 5 |
| metal_nut | 220 | 22 | 77 | 4 |
| pill | 267 | 26 | 141 | 7 |
| screw | 320 | 41 | 119 | 5 |
| tile | 230 | 33 | 84 | 5 |
| toothbrush | 60 | 12 | 30 | 1 |
| transistor | 213 | 60 | 40 | 4 |
| wood | 247 | 19 | 60 | 5 |
| zipper | 240 | 32 | 119 | 7 |

![image.png](style/image/patchcore/1.png)

![image.png](style/image/patchcore/2.png)

![image.png](style/image/patchcore/3.png)

## ⚙️ 환경 (Environment)

| 항목 | 내용 |
| --- | --- |
| OS | Windows 11 + WSL2 Ubuntu |
| GPU | NVIDIA GeForce RTX 3060 (12GB VRAM) |
| CUDA | 13.1 / PyTorch 2.5.1+cu121 |
| Python | 3.10.19 / Conda 환경 anomaly-detect |
| 주요 패키지 | torch, torchvision, scikit-learn, faiss-cpu, fastapi, streamlit |

## 🧠 알고리즘 상세 (PatchCore)

**▪ 전체 파이프라인**

```
[학습 단계] — 역전파 없음
train/good/ 이미지
    ↓ ResNet50 forward pass (GPU)
layer2 피처 (H/8 × W/8 × 512)
layer3 피처 (H/16 × W/16 × 1024)
    ↓ 동일 해상도(28×28)로 보간 후 concat
패치 피처 (N × 1536차원)
    ↓ 랜덤 샘플링 → 10,000개 후보로 축소
    ↓ Greedy Coreset Subsampling (10%)
대표 패치 선택 → memory_bank.pt 저장

[추론 단계]
테스트 이미지
    ↓ ResNet50 forward pass
패치 피처
    ↓ FAISS IndexFlatL2 KNN (k=1)
각 패치의 최근접 정상 패치 거리
    ↓ 원본 해상도로 업샘플링
Anomaly Score Map → 정상/불량 판정
```

**▪ GPU 사용 방식**

PatchCore는 역전파가 없어서 YOLOv8 학습 대비 GPU-Util이 0~1%로 낮게 보인다. ResNet50 forward pass 순간에만 GPU가 사용되고 Coreset 계산, KNN 검색은 CPU에서 이루어진다. YOLOv8처럼 GPU가 지속적으로 100% 가동되는 방식이 아니라 "필요한 순간에만 GPU를 쓰는" 방식이기 때문에 수치만 보고 GPU가 안 쓰인다고 착각하기 쉽다.

## 🔧 구현 과정 및 이슈 해결

실제 구현 중 발생한 이슈들을 기록한다.

**▪ 이슈 1: NumPy 버전 충돌**

`faiss-gpu 1.7.2`가 NumPy 1.x 기반으로 컴파일되어 있는데, `opencv-python 4.13`이 NumPy 2.x를 요구하여 import 에러가 발생했다.

```bash
pip install "numpy<2"
pip install "opencv-python-headless==4.8.1.78"
```

**▪ 이슈 2: metal_nut train 폴더 누락**

kagglehub 다운로드 중 metal_nut 카테고리의 `train/` 폴더가 누락됐다. 전체 학습 실행 중 `AssertionError: 이미지 없음`으로 발견됐고, kagglehub 캐시에서 직접 복사해 해결했다.

```bash
cp -r ~/.cache/kagglehub/.../metal_nut/train data/MVTec/metal_nut/train
```

**▪ 이슈 3: Greedy Coreset 성능 문제 (핵심)**

bottle 기준 209장 × 784패치 = 163,856개에서 Greedy Coreset을 직접 적용하면 O(m × n × d) 연산이 약 4조 회에 달해 사실상 hang 상태가 됐다.

```
# 2단계 방식으로 해결 (AUROC 품질 손실 1% 미만)
1단계: 랜덤 샘플링으로 10,000개로 먼저 축소 (즉시)
2단계: 10,000개에서 Greedy Coreset 적용 (~4초)
```

**▪ 이슈 4: FAISS GPU cuBLAS core dump**

`faiss.index_cpu_to_gpu()` 사용 시 추론 중 cuBLAS 에러로 core dump가 발생했다. CPU `IndexFlatL2`로 변경하여 안정성을 확보했다.

**▪ 이슈 5~6: torch.save / FAISS contiguous 오류**

```python
# numpy ndarray → torch.Tensor로 변환 후 저장
torch.save({"bank": torch.from_numpy(self.bank)}, path)

# FAISS 입력 전 contiguous array 변환
x = np.ascontiguousarray(patches.cpu().float().numpy().astype(np.float32))
```

## 📊 결과 및 분석 (Results & Analysis)

**▪ 카테고리별 학습 결과**

| 카테고리 | 학습 이미지 | Bank 크기 | 학습 시간 |
| --- | --- | --- | --- |
| bottle | 209장 | 10,000 | 42.9초 |
| hazelnut | 391장 | 10,000 | 57.2초 |
| toothbrush | 60장 | **4,704** | 11.9초 |
| screw | 320장 | 10,000 | 94.1초 |
| 전체 합계 | 3,409장 | - | **~15분** |

toothbrush는 학습 이미지 60장으로 bank_size가 4,704로 제한됐다. 전체 15개 카테고리를 약 15분 만에 학습 완료했다.

![스크린샷 2026-04-02 오후 4.03.24.png](style/image/patchcore/4.png)

**▪ 카테고리별 평가 결과**

| 카테고리 | Image AUROC | Pixel AUROC | PRO Score |
| --- | --- | --- | --- |
| bottle | **1.0000** | 0.9859 | 0.8806 |
| cable | 0.9777 | 0.9644 | 0.8701 |
| capsule | 0.8504 | 0.9690 | 0.6955 |
| carpet | 0.9811 | 0.9878 | **0.9210** |
| grid | 0.8379 | 0.9633 | 0.8142 |
| hazelnut | **1.0000** | 0.9889 | 0.8438 |
| leather | **1.0000** | **0.9937** | 0.8609 |
| metal_nut | 0.9988 | 0.9565 | 0.9093 |
| pill | 0.9165 | 0.9322 | 0.8429 |
| screw | 0.6991 | 0.9608 | 0.7985 |
| tile | 0.9910 | 0.9556 | 0.8153 |
| toothbrush | 0.9028 | 0.9875 | 0.8791 |
| transistor | 0.9717 | 0.8775 | 0.7262 |
| wood | 0.9781 | 0.9510 | 0.8508 |
| zipper | 0.9401 | 0.9616 | 0.7341 |
| **평균** | **0.9497** | **0.9557** | **0.8428** |

**▪ 시각화 결과**

![image.png](style/image/patchcore/5.png)

![image.png](style/image/patchcore/6.png)

![image.png](style/image/patchcore/7.png)

![스크린샷 2026-04-02 오후 3.58.45.png](style/image/patchcore/8.png)

![스크린샷 2026-04-02 오후 3.59.32.png](style/image/patchcore/9.png)

**▪ bottle 상세 결과 (단일 카테고리)**

| 지표 | 점수 |
| --- | --- |
| Image AUROC | **100.0%** |
| Pixel AUROC | **98.59%** |
| PRO score | **88.06%** |
| 추론 시간 | ~506ms/장 |

MVTec 논문 기준 목표(AUROC > 98%) 초과 달성.

## 🔍 주요 발견 및 분석 (Insights & Analysis)

**▪ 발견 1: bottle, hazelnut, leather에서 Image AUROC 100% 달성**

규칙적인 형태를 가진 카테고리에서 완벽한 탐지 성능을 보였다. ResNet50 pretrained 피처만으로도 도메인 특화 학습 없이 이 수준의 성능이 가능하다.

**▪ 발견 2: screw에서 Image AUROC 69.9%로 가장 낮음**

screw는 나사 방향(회전)이 정상 이미지마다 다양하다. PatchCore는 회전 불변(rotation invariant) 특성이 없어서, 같은 클래스라도 방향이 다르면 이상으로 판단할 수 있다. 회전 augmentation으로 Memory Bank를 보강하면 개선 가능하다.

**▪ 발견 3: Image AUROC가 낮아도 Pixel AUROC는 93% 이상**

screw(Image AUROC 70%), grid(84%)에서도 Pixel AUROC는 96%를 유지했다. "이미지 단위 판정은 어렵지만, 불량이 존재한다면 위치는 정확히 잡아낸다"는 의미다. 작업자에게 불량 위치를 알려주는 용도로는 충분히 활용 가능하다.

**▪ 발견 4: 학습 속도가 압도적으로 빠르다**

15개 카테고리 전체 학습에 약 15분이 걸렸다. YOLOv8이 카테고리당 25~40분 걸렸던 것과 비교하면 극명한 차이다. 역전파 없이 피처 저장만 하기 때문이다. 실무에서 새 제품 라인 추가 시 정상 이미지 200장만 있으면 1분 이내로 모델 구축이 가능하다.

**▪ 발견 5: 추론 속도는 500ms로 YOLO 대비 느림**

KNN 검색에서 10,000개 피처와 모든 패치를 비교해야 해서 이미지당 500ms 내외가 소요된다. FAISS GPU 전환 또는 Coreset 1%로 줄이면 개선 가능하다.

## 🌐 서빙 구현 (Serving)

**▪ 시스템 구조**

```
사용자 (브라우저)
    ↓ 이미지 업로드 + 카테고리 선택
Streamlit Frontend (포트 8501)
    ↓ HTTP POST /predict
FastAPI Backend (포트 8000)
    ↓ Memory Bank 로드 + FAISS KNN 검색
PatchCore 모델 (memory_bank.pt)
    ↓ Anomaly Score Map + 히트맵 생성
Streamlit Frontend
    ↓ 원본 / 히트맵 / 오버레이 3열 표시
사용자
```

**▪ 핵심 기능**

- `GET /models`: 학습된 (category, exp) 목록 자동 스캔 → 학습된 것만 선택 가능
- `POST /predict`: 이미지 + 카테고리 → 이상 점수 + 히트맵(base64) + 오버레이 반환
- 모델 캐시로 중복 로드 방지
- `start.sh`로 FastAPI + Streamlit 한 번에 실행

![웹앱 데모](style/image/patchcore/10.png)

![스크린샷 2026-04-02 오후 6.18.50.png](style/image/patchcore/11.png)

## 🔍 인사이트 및 결론 (Insights & Conclusion)

**1. 라벨 없는 이상 탐지는 실무에서 더 현실적이다**

Welding 프로젝트에서 배운 것은 라벨링 자체가 큰 비용이라는 점이다. PatchCore는 정상 이미지만 있으면 어떤 종류의 불량이 나타나도 탐지할 수 있다. 새로운 불량 유형에 재학습 없이 대응 가능하다는 점이 실무에서 가장 큰 장점이다.

**2. Pretrained 피처의 힘**

별도 학습 없이 ImageNet pretrained ResNet50의 중간 레이어 피처만으로 산업 표면 결함을 평균 Image AUROC 94.97%, Pixel AUROC 95.57%로 탐지할 수 있다. 도메인 특화 학습 없이 이 수준의 성능이 나온다는 것이 PatchCore의 핵심 가치다.

**3. 형태 규칙성이 성능을 결정한다**

bottle, leather처럼 형태가 규칙적인 카테고리에서 AUROC 100%를 달성했지만, screw처럼 회전 방향이 다양한 카테고리에서는 70%로 크게 낮아졌다. PatchCore는 회전 불변성이 없다는 구조적 한계를 실험으로 직접 확인했다.

**4. 지도 → 약지도 → 비지도 로드맵**

세 프로젝트를 거치면서 지도학습(Welding Detection) → 약지도학습(SAM2 자동 라벨 + Seg) → 비지도학습(PatchCore) 순으로 라벨 의존도를 줄여왔다. 이 흐름이 실제 산업 현장에서 직면하는 데이터 부족 문제를 단계적으로 해결하는 방식과 일치한다.

## 🚀 추가 실험 / 개선 방향 (Further Work)

- **screw 개선**: 회전 augmentation으로 Memory Bank를 다양한 방향으로 보강
- **WideResNet50 실험 (exp2)**: 더 강력한 Backbone이 AUROC에 미치는 영향 비교
- **Coreset 1% 실험 (exp3)**: 속도와 정확도 트레이드오프 정량화 (목표: 추론 50ms)
- **FAISS GPU 전환**: KNN 검색 GPU 가속으로 추론 500ms → 50ms 목표
- **Jetson Nano 온디바이스 배포**: ResNet50 ONNX export → Jetson에서 추론 속도 벤치마크
- **Welding 도메인 적용**: Welding Defect 정상 이미지에 PatchCore 적용, 기존 YOLOv8 Detection과 성능/비용 비교