---
layout: post
title: LightGlue Feature Matching Comparison
date: 2025-04-23 11:30:00 +0800
category: experiment
thumbnail: style/image/lightglue_image_matching/image1.png
icon: code
---

# lightglue image matching

# 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| **프로젝트명** | LightGlue Feature Matching Comparison (SuperPoint / DISK / SIFT / ALIKED / DoGHardNet) |
| **모델(Model)** | LightGlue + SuperPoint / DISK / SIFT / ALIKED / DoGHardNet |
| **데이터셋(Dataset)** | 직접 촬영 이미지 페어 (EFEM 장비: IMG_1868.JPG, IMG_1869.JPG) |
| **환경** | 로컬 서버 / RTX 3060 12GB / PyTorch 2.9.1 / CUDA 13.1 / Python 3.11.14 |

---

# 목적 (Objective)

### ▪ 실험의 목표

- LightGlue 파이프라인을 실습하고, 서로 다른 로컬 특징 추출기(5종)의 매칭 특성과 포즈 안정성을 비교한다.

### ▪ 얻고자 하는 인사이트

- 매칭 수/인라이어 비율/포즈 에러의 상관
- 디테일 구조(장비 패널, 라벨, 케이블)에서 각 특징의 강점

---

# 📦 데이터셋 (Datasets)

| 구분 | 내용 |
| --- | --- |
| **이미지 페어** | EFEM 장비 서로 다른 시점 2장 |
| **입력 형식** | RGB |
| **전처리** | 없음 |
| **평가 지표** | 매칭 수, 인라이어 수/비율, 포즈 평균 재투영 오차 |

![image.png](style/image/lightglue_image_matching/image.png)

---

# 🧠 실험 설계 (Experiment Design)

| 항목 | 설정값 |
| --- | --- |
| Feature Extractor | SuperPoint / DISK / SIFT / ALIKED / DoGHardNet |
| max_kpts | 1024 |
| filter_threshold | 0.1 |
| Inlier 계산 | Essential Matrix (RANSAC) |
| Pose 평가 | 평균 재투영 오차 |

---

# 📊 결과 및 분석 (Results & Analysis)

## 1) 정량 결과 (logs 기반)

| Feature | Match | Inlier | Inlier Ratio | Pose Error |
| --- | --- | --- | --- | --- |
| **SuperPoint** | 334 | 273 | **0.817** | **0.272** |
| **DISK** | 339 | 278 | **0.820** | 0.512 |
| **SIFT** | 194 | 142 | 0.732 | 1.377 |
| **ALIKED** | **345** | **286** | **0.829** | 0.448 |
| **DoGHardNet** | 228 | 167 | 0.732 | **0.264** |

※ Pose Error는 낮을수록 좋음

### 정량 해석

- **ALIKED**: 매칭 수/인라이어 비율 최고 → 매칭은 매우 잘 잡힘. 포즈 에러는 중간.
- **SuperPoint**: 매칭 수는 충분하고 포즈 에러가 낮아 균형이 가장 좋음.
- **DoGHardNet**: 매칭 수는 적지만 포즈 에러가 가장 낮음 → 강한 정합만 남는 경향.
- **DISK**: 매칭 수/인라이어 비율은 높으나 포즈 에러가 비교적 큼 → 강한 지역 집중/편향 가능성.
- **SIFT**: 매칭 수/비율/포즈 모두 가장 낮아 상대적으로 약세.

---

## 2) 시각화 결과 (images 기반)

### *매칭 점수에 따라 green → red 순으로 높음

### SuperPoint

![image.png](style/image/lightglue_image_matching/image1.png)

- 장비 전역에 비교적 고르게 매칭 분포.
- 상단 패널(검정 장치), 라벨, 케이블 구역 모두 골고루 잡힘.
- **전역 분산 + 안정적**인 특징 분포.

### DISK

![image.png](style/image/lightglue_image_matching/image%202.png)

- 강한 텍스처/라벨 영역에 매칭이 집중됨.
- 긴 수평 라인 매칭이 많고 빨간(고신뢰) 라인 비중이 큼.
- **매칭 수는 많지만 분포 편중** 경향.

### SIFT

![image.png](style/image/lightglue_image_matching/image%203.png)

- 매칭 수 자체가 적고 특정 영역(라벨, 패널) 중심.
- 전역 커버리지 부족 → 장면 전체 제약력이 떨어짐.

### ALIKED

![image.png](style/image/lightglue_image_matching/image%204.png)

- 매칭 수가 가장 많고, 강한 영역에 매우 잘 붙음.
- 빨간 고신뢰 매칭이 넓게 퍼져 있지만, 일부 영역 과집중도 보임.
- **매칭 양과 인라이어 비율 측면 최고**.

### DoGHardNet

![image.png](style/image/lightglue_image_matching/image%205.png)

- 매칭 수는 중간 이하이나 비교적 깔끔한 라인 구성.
- 낮은 noise + 강한 매칭 중심 → **포즈 안정성은 좋게 나옴**.

---

# 인사이트 및 결론 (Insights & Conclusion)

- **매칭 수가 많다고 포즈 성능이 항상 좋은 것은 아님**.
(DISK/ALIKED는 매칭 수 많지만 포즈 에러는 SP/DoGHardNet보다 높음)
- **SuperPoint는 전역 분산 + 낮은 포즈 에러**로 가장 균형형.
- **ALIKED는 매칭량/인라이어 비율 최강**이지만, 포즈 에러가 약간 높아 매칭의 “분포 편향”이 존재.
- **DoGHardNet은 매칭 수가 적어도 포즈 안정성이 우수**.
→ 강한 매칭만 남기는 성향.
- **SIFT는 상대적으로 약세**.

> 결론:
> 
> 
> “EFEM 장비와 같은 설비 task에서는 SuperPoint가 전역적으로 안정적이고 균형 잡힌 결과를 보였고, ALIKED는 매칭 수와 인라이어 비율이 가장 높았으나 포즈 오차는 다소 증가.
> 
> DoGHardNet은 매칭 수가 적지만 포즈 안정성에서 가장 높음을 확인함.”
> 

---

# 코드 및 리소스 (Appendix)

- 카메라 내참수 K(예시) 계산

```python
w, h = 1440, 1080
    fov_deg = 60  # 대략적인 광각
    fx = fy = (w / 2) / math.tan(math.radians(fov_deg / 2))
    k = np.array(
        [
            [fx, 0, w / 2],
            [0, fy, h / 2],
            [0, 0, 1],
        ]
    )
```