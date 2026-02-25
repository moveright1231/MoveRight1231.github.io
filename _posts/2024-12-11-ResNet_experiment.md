---
layout: post
title: Residual Connection Ablation Study on ResNet
date: 2024-12-11 01:00:00 +0800
category: experiment
thumbnail: /style/image/Resnet_image3.png
icon: code
---

# Residual Connection Ablation Study on ResNet

# 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| **프로젝트명** | Residual Connection Ablation Study on ResNet |
| **모델(Model)** | ResNet-18 (CIFAR-10),
ResNet-50 (CIFAR-100) |
| **데이터셋(Dataset)** | CIFAR-10 / CIFAR-100 |
| **환경** | Google Colab (T4 GPU), PyTorch 2.2, CUDA 12.4, Python 3.11 |

---

# 목적 (Objective)

### ▪ 실험의 목표

- CNN에서 **Residual Connection(skip connection)**이
학습 안정성, 수렴 속도, 일반화 능력(Validation 성능)에
미치는 영향을 비교 분석한다.

### ▪ 얻고자 하는 인사이트

- 계산 복잡도가 낮거나 깊은 네트워크에서 **Residual Path**가 없는 경우 gradient 흐름,
학습 중 Loss와 정확도의 변화.
- 반대로 skip connection이 있을 때 **Loss의 안정적 감소와 Accuracy의 꾸준한 상승**이 나타나는지 관찰한다.

---

# 💡 배경 및 아이디어 (Background & Motivation)

### ▪ 실험 동기

- ResNet 논문(He et al., 2015)에서 제안된 **skip connection**은
“깊은 네트워크가 오히려 학습이 어렵다는 문제(Vanishing Gradient)”를 해결하는 핵심 아이디어이다.
- 그러나 **단순히 블록을 쌓는 것보다 skip을 제거했을 때 정확히 어떤 현상이 발생하는지**
실험적으로 확인하고 싶었다.
- 그래서 깊지 않은 네트워크와 깊은 네트워크에서의 차이와 Skip Connection의 성능을 확인하고자 함.
- *본 실험은 “skip 연결의 유무”만을 변수로 두고 나머지 설정을 동일하게 하여
학습 안정성과 수렴 특성을 직접 관찰한다.

---

# 📦 데이터셋 (Datasets)

| 항목 | CIFAR-10 | CIFAR-100 |
| --- | --- | --- |
| **Train** | 50,000 | 50,000 |
| **Test** | 10,000 | 10,000 |
| **Input** | 32×32 RGB | 32×32 RGB |
| **전처리/증강** | RandomCrop, HorizontalFlip, Normalize | RandomCrop, AutoAugment, Normalize |
| **평가 지표** | Accuracy, Loss | Top-1 Accuracy, Loss |

---

# ⚙️ 환경 (Environment)

- Google Colab Pro (T4 GPU)
- PyTorch 2.2
- CUDA 12.4 / cuDNN
- Python 3.11
- Auto Mixed Precision (AMP) 활성화
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4)
- Scheduler: CosineAnnealingLR
- Criterion: CrossEntropyLoss

---

# 🧠 실험 설계 (Experiment Design)

### ▪ 실험 목적

- 동일한 구조(ResNet)에서 **Residual Connection의 유무**가
학습 및 일반화에 미치는 영향을 정량적으로 분석.

### ▪ 변수 설정

| 항목 | ResNet-18 | ResNet-50 |
| --- | --- | --- |
| Dataset | CIFAR-10 | CIFAR-100 |
| Epoch | 20 | 30 |
| Batch Size | 128 | 128 |
| Learning Rate | 0.1 | 0.2 |
| Optimizer | SGD | SGD |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Weight Decay | 5e-4 | 5e-4 |
| Data Augmentation | RandomCrop, Flip | AutoAugment 포함 |

### ▪ 비교 대상

| 실험명 | 설명 |
| --- | --- |
| **with-skip** | Residual Block 내 skip connection 유지 |
| **no-skip** | Skip connection 제거 |
| (추가) | BN On/Off 실험으로 정규화 영향 확인 예정 |

---

# 📊 결과 및 분석 (Results & Analysis)

## ✅ ResNet-18 (CIFAR-10)

![image.png](style/image/Resnet_image.png)

![image.png](style/image/Resnet_image1.png)

- **Loss 곡선**:
    
    skip을 적용한 모델이 오히려 느리게 수렴했으며,
    epoch 5~7에서 skip 적용 valid가 Loss가 치솟음.
    
    최종 epoch에서는 적용을 한 모델과 적용하지 않은 모델이 Loss가 비슷하게 수렴함.
    
- **Accuracy 곡선**:
epoch 5~7에서 skip 적용 train이 accuracy가 확 떨어짐.
    
    최종 epoch에서는 적용을 한 모델과 적용하지 않은 모델이 accuracy가 비슷하게 측정됨.
    
- **결론**:
    
    낮은 네트워크에서 skip connection이 있을 때 오히려 노이즈가 되어 적은 epoch에서는 방해가 될 수 있음을 확인
    
    일정 이상의 epoch에서는 동일한 성능을 보임.
    

---

## ✅ ResNet-50 (CIFAR-100)

![image.png](style/image/Resnet_image2.png)

![image.png](style/image/Resnet_image3.png)

- **Loss 곡선**
    - 초반 학습(1~3 epoch)에서는 skip 유무에 따른 큰 차이는 없었음.
    - 7~8 epoch에서 **no-skip validation loss가 약 60까지 폭등함**
    이는 BN 통계 불안정(batch norm running mean/var 붕괴)로 추정됨.
    - 이후 다시 안정화되었지만 전체적으로 **no-skip이 불안정한 학습 패턴**을 보임.
- **Accuracy 곡선**
    - with-skip 모델은 train/val 모두 **꾸준히 상승**하며 최종 val acc ≈ 0.78 수준 도달.
    - no-skip 모델은 중간 epoch(7~8)에서 accuracy가 급락했다가 회복되었으나,
    최종 val acc는 **약 10~15% 낮음**.
    - no-skip 모델에서 한번의 큰 정확도 하락 이후 오르락내리락 불안정한 패턴을 보임.
- **분석**
    1. Residual path가 없는 경우, gradient가 깊은 층까지 전파되지 못해
    weight 업데이트가 불균형하게 일어남.
    2. BN이 불안정해져 특정 배치에서 running stats가 깨지고,
    validation 시 손실 폭증 발생.
    3. skip 연결은 이러한 현상을 완화시켜
    학습이 훨씬 **안정적이고 일반화 성능이 우수**함을 확인.
- **그래프 해석 요약**
    - **with-skip train**: Loss 낮고 Accuracy 꾸준히 상승 → 안정적 학습
    - **with-skip val**: 일정한 향상 곡선 → 일반화 잘됨
    - **no-skip val**: 손실 급등 후 회복 → BN 통계 붕괴 가능성
    - **no-skip train**: 안정적이지만 validation에서 과적합 성향

---

# 🔍 인사이트 및 결론 (Insights & Conclusion)

- **Residual Connection은 깊은 네트워크에서 학습 안정성과 일반화 향상에 결정적 역할을 한다.**
- no-skip 구조는 gradient 흐름이 끊겨 BN 통계가 쉽게 불안정해지고,
validation 손실 폭증(spike)과 정확도 급락을 유발한다.
- skip connection이 있는 경우 Loss 곡선이 부드럽고,
Accuracy는 점진적으로 상승하며 수렴 속도 또한 빠르다.
- 하지만 계산 복잡도가 낮은 네트워크에서는 (적은 epoch에서는 더 크게) 오히려 방해가 되는 경우도 있음을 확인.

> 결론:
> 
> 
> "Residual block은 깊은 CNN의 학습을 안정화하고, 일반화 성능을 향상시킨다."
> 

---

# 💾

---

- **데이터셋**: `torchvision.datasets.CIFAR10`, `CIFAR100`
- **모델 구현**: Custom ResNet class (skip toggleable)
- **Optimizer**: SGD + CosineAnnealingLR
- **결과 그래프**:
    - Loss vs Epochs (with-skip / no-skip)
    - Accuracy vs Epochs (with-skip / no-skip)
- **참고 논문**:
    - He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.*