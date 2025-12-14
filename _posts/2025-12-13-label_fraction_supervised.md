---
layout: post
title: 라벨 제한 Supervised Learning (ResNet16)
date: 2025-12-13 15:00:00 +0800
category: experiment
thumbnail: /style/image/thumbnail.png
icon: code
---

# 라벨제한 ResNet16

## 환경 세팅

| 항목 | 내용 |
| --- | --- |
| **프로젝트명** | 라벨이 제한된 환경에서 
Supervised Learning  |
| **모델(Model)** | ResNet18 |
| **데이터셋(Dataset)** | CIFAR-10 |

가상환경 사용 : sslcv
버전 : python=3.10

```python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install tqdm numpy pandas matplotlib scikit-learn umap-learn pyyaml

```

```python
# configs/supervised.yaml

seed: 42
data_dir: "data"
out_dir: "outputs/supervised"
epochs: 50
batch_size: 256
lr: 0.1
weight_decay: 0.0005
num_workers: 4
label_fraction: 0.10   # 0.01 / 0.05 / 0.10 / 1.0 로 바꿔가며 실행
```

라벨된 데이터의 비율을 바꿔가며 실험하기 위한 yaml 파일

```python
opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
criterion = nn.CrossEntropyLoss()
```

### 얻고자하는 인사이트

- 일반화 한계 측정: 라벨이 거의 없을 때(0.01, 0.05) 얼마나 성능이 떨어지고, 어느 지점부터 추가 라벨이 효율이 낮아지는지(0.10→1.0) 확인.
- 라벨 효율 곡선: label_fraction 대비 valid acc를 그려보면 “라벨 1% → acc 급락, 5%→가파른 상승, 10%→완만, 100%→상한” 같은 비선형 곡선을 얻어 라벨 추가의 ROI를 정량화.
- 오버피팅 패턴: 적은 라벨(1%, 5%)에서는 train acc가 급상승하고 valid acc가 일찍 plateau/하락하며, 100%에서는 train/valid가 더 오래 동행하는 양상을 비교.
- 데이터 증강/모델 용량 영향: 같은 증강/모델에서 라벨 부족시 어디까지 버티는지 확인해, semi-supervised나 강한 증강(CTA, MixMatch 등)을 도입했을 때 개선 폭을 예측하는 베이스라인 구축.
- 리소스/시간 대비 성능: 라벨 10%로 58%대, 100%로는 ~90% 근처(예상)라면, 원하는 목표 정확도에 필요한 라벨 비율을 추정.
- 하이퍼파라미터 민감도: 라벨이 적을수록 learning rate/정규화에 더 민감함. 각 fraction에서 최적 설정이 달라지는지 관찰.

- label_fraction : 0.01 의 결과

![image.png](style/image/supervised_resnet16/1.png)

![image.png](style/image/supervised_resnet16/2.png)

```python
Best valid acc : 0.3451
```

## label_fraction 0.01 결과 분석

### 성능 지표

- **Best Valid Accuracy:** 34.51% - CIFAR-10의 랜덤 추측(10%)보다는 높지만 실용적 수준과는 거리가 멀음
- **Train Accuracy:** 그래프 상 매우 높은 수준까지 상승 - 모델이 소수의 라벨 데이터를 완벽히 암기
- **Train-Valid 격차:** 극심한 과적합 발생, 약 40~50%p 이상의 격차 추정

### 학습 곡선 특징

- **Train Loss:** 안정적으로 0에 가깝게 감소 - 학습 데이터 완벽 피팅
- **Valid Loss:** 초반 급락 후 2.0 이상에서 정체 또는 상승 - 일반화 실패
- **Valid Accuracy:** 초반 급상승 후 epoch 10~15 이후 plateau, 이후 개선 없음

### 핵심 문제점

- **데이터 부족:** 전체 데이터의 1%(약 500개)만 라벨링되어 클래스당 평균 50개 샘플만 사용
- **표현 학습 실패:** 모델이 데이터의 본질적 특징을 학습하지 못하고 소수 샘플만 암기
- **조기 수렴:** Valid 성능이 빠르게 한계에 도달하여 추가 학습이 무의미

### 시사점

1% 라벨 환경에서는 Supervised Learning만으로는 의미 있는 성능 달성이 불가능하며, Self-Supervised Pre-training 또는 Semi-Supervised Learning 기법이 필수적임을 보여줌.

- abel_fraction : 0.05 의 결과

![image.png](style/image/supervised_resnet16/3.png)

![image.png](style/image/supervised_resnet16/4.png)

```python
Best valid acc : 0.4936
```

## label_fraction 0.05 결과 분석

### 성능 지표

- **Best Valid Accuracy:** 49.36% - 1%에서 14.85%p 상승하여 가장 높은 개선률 기록
- **Train Accuracy:** 그래프 상 매우 높은 수준 - 여전히 강한 암기 경향 존재
- **Train-Valid 격차:** 심각한 과적합 지속, 약 30~40%p의 격차 추정

### 학습 곡선 특징

- **Train Loss:** 0에 가깝게 안정적으로 감소
- **Valid Loss:** 초반 급락 후 약 1.5~2.0 사이에서 정체
- **Valid Accuracy:** 초반 급상승 후 epoch 15~20 근처에서 수렴 시작

### 핵심 특징

- **라벨 효율성:** 1%에서 5%로 5배 증가 시 성능이 43% 향상(34.51%→49.36%)되어 가장 높은 ROI 구간
- **데이터 규모:** 약 2,500개 샘플 사용(클래스당 평균 250개) - 1% 대비 5배 증가로 표현 학습 개선
- **여전한 한계:** 랜덤 추측(10%) 대비 4.9배 성능이지만 실용적 수준(70~80%)과는 여전히 큰 격차

### 시사점

5% 라벨 환경은 1% 대비 큰 개선을 보이나, 여전히 과적합이 심각하고 일반화 성능이 제한적. 이 구간에서 Semi-Supervised Learning이나 Self-Supervised Pre-training 도입 시 가장 큰 성능 향상 효과를 기대할 수 있음.

- label_fraction : 0.10 의 결과

![image.png](style/image/supervised_resnet16/5.png)

![image.png](style/image/supervised_resnet16/6.png)

```python
Best valid acc: 0.5836
```

## label_fraction 0.10 결과 분석

### 성능 지표

- **Best Valid Accuracy:** 58.36% - 5%에서 9%p 상승하여 개선 폭이 둔화되기 시작
- **Train Accuracy:** 그래프 상 꾸준히 상승하며 높은 수준 도달
- **Train-Valid 격차:** 여전히 과적합 존재하나 1%, 5% 대비 격차 감소, 약 20~30%p 추정

### 학습 곡선 특징

- **Train Loss:** 안정적으로 0에 가깝게 감소하며 이상 없음
- **Valid Loss:** 초반 급락 후 1.4~2.0 사이에서 정체 및 스파이크 발생
- **Valid Accuracy:** 초반 급상승 후 epoch 20 근처에서 수렴, 이후 개선 미미

### 핵심 특징

- **라벨 효율성 둔화:** 5%에서 10%로 2배 증가 시 성능은 18.2% 향상(49.36%→58.36%)으로 1%→5% 구간 대비 개선률 감소
- **데이터 규모:** 약 5,000개 샘플 사용(클래스당 평균 500개) - 5% 대비 2배 증가
- **일반화 한계 도달:** Valid accuracy가 수렴하여 추가 학습의 효과가 거의 없음. Valid loss의 스파이크는 모델이 일반화 한계에 근접했음을 시사
- **강한 과적합 지속:** Train accuracy는 계속 상승하나 Valid는 정체되어 모델이 학습 데이터 암기에 치중

### 시사점

10% 라벨 환경에서는 Supervised Learning의 성능이 약 58%로 수렴하며 추가 epoch 학습이 무의미함. 이 지점에서 더 많은 라벨 데이터 확보 또는 Semi-Supervised/Self-Supervised Learning 기법 도입이 필수적. 특히 라벨 없는 나머지 90% 데이터를 활용할 수 있는 기법이 큰 효과를 발휘할 것으로 예상됨.

- label_fraction : 1.0의 결과

![image.png](style/image/supervised_resnet16/7.png)

![image.png](style/image/supervised_resnet16/8.png)

```python
Best valid acc : 0.8379
```

## label_fraction 1.0 (100%) 결과 분석

### 성능 지표

- **Best Valid Accuracy:** 83.79% - 10%에서 25.43%p 상승하여 실용적 성능 수준 달성
- **Train Accuracy:** 안정적으로 높은 수준까지 상승하며 건강한 학습 패턴
- **Train-Valid 격차:** 과적합이 크게 완화되어 약 10~15%p 수준으로 감소

### 학습 곡선 특징

- **Train Loss:** 매우 안정적으로 0에 가깝게 감소하며 정상적인 학습 진행
- **Valid Loss:** Train Loss와 함께 안정적으로 감소하며 스파이크 없이 수렴
- **Valid Accuracy:** 전 구간에서 꾸준히 상승하며 epoch 40~50 근처에서 안정화

### 핵심 특징

- **충분한 데이터 규모:** 전체 50,000개 샘플 사용(클래스당 5,000개) - 모델이 데이터 분포를 충분히 학습
- **건강한 학습:** Train과 Valid 성능이 함께 향상되어 과적합 최소화. 모델이 실제 특징을 학습하여 일반화 능력 확보
- **실용적 성능:** 83.79%의 Valid accuracy는 CIFAR-10에서 ResNet18 기준 실용적 수준. 추가 개선을 위해서는 모델 확장, 고급 증강 기법 등 필요
- **라벨 효율성:** 10%에서 100%로 10배 증가 시 성능은 43.6% 향상(58.36%→83.79%)으로 충분한 라벨 데이터의 중요성 입증

### 시사점

100% 라벨 환경에서는 Supervised Learning이 안정적이고 효과적으로 작동하며, 충분한 라벨 데이터가 모델의 일반화 능력과 실용적 성능 달성에 필수적임을 확인. 라벨이 제한된 환경(1~10%)과의 극명한 성능 차이는 Self-Supervised 또는 Semi-Supervised Learning 기법의 필요성을 강조함.

## 실험 결과 요약

### 실험 개요

CIFAR-10 데이터셋에서 ResNet18 모델을 사용하여 라벨 비율(label_fraction)을 0.01, 0.05, 0.10, 1.0으로 변화시키며 Supervised Learning의 성능을 측정하였다. 총 50 epoch 동안 학습을 진행하였으며, 배치 크기는 256, 학습률은 0.1, weight decay는 0.0005로 설정하였다.

### 정량적 결과

| Label Fraction | Best Valid Accuracy | Train Accuracy (Final) | 주요 특징 |
| --- | --- | --- | --- |
| 0.01 (1%) | 0.3451 | 높음 (그래프 상) | 심각한 과적합, 일반화 능력 부족 |
| 0.05 (5%) | 0.4936 | 높음 (그래프 상) | 과적합 지속, 성능 개선 제한적 |
| 0.10 (10%) | 0.5836 | 지속 상승 | Valid 수렴, 추가 학습 무의미 |
| 1.0 (100%) | 0.8379 | 안정적 상승 | Train/Valid 동행, 건강한 학습 |

### 정성적 분석

### 1. 라벨 비율에 따른 성능 변화

- **1% → 5%:** Valid accuracy가 34.51%에서 49.36%로 약 14.85%p 상승. 라벨 추가의 효과가 가장 크게 나타남.
- **5% → 10%:** 49.36%에서 58.36%로 약 9%p 상승. 개선 폭이 둔화되기 시작.
- **10% → 100%:** 58.36%에서 83.79%로 약 25.43%p 상승. 라벨 10배 증가로 큰 성능 향상 달성.

### 2. 과적합(Overfitting) 패턴

- **1%, 5% 구간:** Train accuracy는 높게 상승하나 Valid accuracy는 낮은 수준에 머물러 심각한 과적합 현상 관찰. Train과 Valid 간 격차가 매우 큼.
- **10% 구간:** Train accuracy는 꾸준히 상승하나 Valid accuracy는 약 epoch 20 이후 plateau 도달. Valid loss는 1.4~2.0 사이에서 진동하며 추가 학습의 효과 미미.
- **100% 구간:** Train과 Valid accuracy가 비교적 함께 상승하며 건강한 학습 곡선 형성. 과적합이 상대적으로 완화됨.

### 3. Loss 곡선 분석

- **적은 라벨 비율(1%, 5%, 10%):** Train loss는 안정적으로 감소하나, Valid loss는 초반 급락 후 수렴 또는 상승. 특히 10% 구간에서 Valid loss가 1.4에서 2.0 사이에서 스파이크 발생.
- **100% 라벨:** Train과 Valid loss가 모두 안정적으로 감소하며 수렴하는 경향.

### 4. 일반화 한계

라벨이 10% 이하일 때 모델은 학습 데이터에 대해서는 높은 정확도를 보이나, 검증 데이터에 대한 일반화 능력이 현저히 떨어진다. 이는 모델이 제한된 라벨 정보만으로 데이터의 전반적인 분포를 학습하지 못했음을 시사한다.

### 주요 인사이트

- **라벨 효율 곡선:** 라벨 1%에서 5%로 증가 시 가장 높은 ROI(약 43% 성능 향상)를 보이며, 이후 점차 수확 체감 효과 발생.
- **임계점 도달:** 라벨 10%에서 Valid accuracy는 수렴하여 추가 epoch 학습이 무의미한 지점에 도달. 더 많은 라벨 데이터 확보 또는 Semi-Supervised/Self-Supervised 기법 도입 필요.
- **라벨 부족 환경의 한계:** 적은 라벨(1~10%)로는 약 35~58%의 제한적인 성능만 달성 가능. 실용적 수준(80% 이상)을 위해서는 충분한 라벨 데이터(본 실험에서는 100%) 필수.
- **Self-Supervised Learning 필요성 제기:** 라벨이 제한된 환경에서 Supervised Learning만으로는 일반화 성능 확보가 어려우므로, 라벨 없는 데이터를 활용한 Self-Supervised Representation Learning 도입 시 큰 성능 개선 기대.

### 향후 실험 방향

- 동일한 라벨 비율(1%, 5%, 10%)에서 Self-Supervised Learning(SimCLR, MoCo 등)을 적용하여 Supervised Learning 대비 성능 개선 폭 측정
- 강한 데이터 증강(CutMix, MixUp, CutOut 등) 도입을 통한 적은 라벨 환경에서의 성능 향상 검증
- 학습률, 정규화 강도 등 하이퍼파라미터 튜닝을 통한 각 라벨 비율별 최적 설정 탐색
- Semi-Supervised Learning 기법(Pseudo-Labeling, FixMatch 등) 적용 실험