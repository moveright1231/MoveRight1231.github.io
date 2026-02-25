---
layout: post
title: MASt3R 논문리뷰
date: 2025-02-18 18:30:00 +0800
category: paper
thumbnail: /style/image/dust3r_paper/1.png
icon: code
---

# MASt3R 논문리뷰

# MASt3R: Faster & Better Dense Image Matching for 3D Reconstruction

## 개요

MASt3R는 DUSt3R의 후속 연구로,

**dense image matching + 3D reconstruction**을

더 빠르고, 더 정확하게 수행하기 위한 모델임.

DUSt3R가

> “정답 3D 없이도, 두 이미지 간 3D 일관성만으로 학습 가능하다”
> 
> 
> 는 걸 보여줬다면,
> 

MASt3R는

> “그걸 실제 대규모 3D 재구성 파이프라인에서
> 
> 
> 더 빠르고 안정적으로 쓰게 만들자”
> 
> 에 초점을 둠.
> 

핵심은 다음 세 가지임.

- dense matching 정확도 향상
- 계산량 감소 (속도 개선)
- 고해상도 이미지 & 많은 view 처리 가능

---

## 1. 기존 SfM / COLMAP의 근본적인 한계

기존 Structure-from-Motion(SfM) 파이프라인은

대략 아래와 같은 흐름을 가짐.

```python
SIFT → Feature Matching → RANSAC
→ Essential / Fundamental Matrix
→ PnP → Triangulation → Bundle Adjustment
```

이 구조의 문제는 명확한 문제는

- feature matching이 깨지면 전체 파이프라인이 붕괴 (의미가 없어짐)
- 반복 패턴(벽, 바닥, 창문)에 극도로 취약 (제대로 된 depth를 파악하지 못함)
- reflection(빛의 반사), glass, texture-less 영역에 약함
- calibration 오차에 민감
- indoor / object-level reconstruction에 부적합

즉, **matching이 모든 걸 좌우하는 구조**임.

---

## 2. DUSt3R의 핵심 아이디어

MASt3R를 이해하려면 DUSt3R부터 짚고 가야 함.

### SfM 문제의 본질 재정의

DUSt3R의 출발점은 이 질문으로 부터 나옴.

> 우리는 인터넷에 실제 이미지에 맞는 3D 데이터는 없지만
> 
> 
> 같은 물체를 찍은 이미지 수십억 장은 가지고 있음
> 
> 그럼 3D를 어떻게 학습할 수 있을까?
> 

DUSt3R의 답은 다음과 같음.

> **정답 3D 데이터 대신,
두 이미지의 3D 일관성만 맞추자 (image matching)**
> 

---

### DUSt3R의 Self-supervised 학습 방식

![image.png](style/image/dust3r_paper/1.png)

DUSt3R는 이미지 pair를 입력으로 받아

아래를 동시에 예측함.

- Image 1의 pointmap (픽셀 → 3D)
- Image 2의 pointmap
- 두 이미지 간 dense correspondence
- 각 예측의 confidence (신뢰도)

학습 신호는 다음 제약 하나임.

```python
P1(x1) ≈ R · P2(x2) + t
```

하지만 중요한 점:

- R, t 모름 (회전, 이동)
- correspondence도 모름 (일치 하는지 안하는지)

→ 전부 네트워크가 **스스로 찾아야 함, end to end를 추구함**

---

### 3D Consistency Constraint

DUSt3R 및 MASt3R의 학습은

두 이미지의 3D pointmap이 **같은 물체를 본 경우 일관되도록** 강제함.

개념적으로는 다음 조건을 만족해야 함.

```python
P1(x1) ≈ R · P2(x2) + t
```

의미는 다음과 같음.

- P1(x1): Image 1의 픽셀 x1에서 예측한 3D 좌표
- P2(x2): Image 2의 픽셀 x2에서 예측한 3D 좌표
- R, t: 두 이미지 사이의 회전과 이동
- x1, x2: 같은 물체를 가리키는 대응 픽셀

중요한 점은,

- R, t는 주어지지 않음
- correspondence (x1 ↔ x2)도 주어지지 않음

→ 네트워크가 **3D, 대응, 포즈를 동시에 추론**해야 함.

---

### Confidence-weighted loss의 의미

Loss는 단순히 3D가 같아지도록 강제하지 않음.

- confidence가 높은 점들만 loss에 크게 반영
- 가려진 점, reflection, 잘못된 매칭은 자동 무시

→ 이 구조는 사실상

**RANSAC을 신경망으로 구현한 것**과 같음.

---

## 3. DUSt3R vs COLMAP (개념 비교)

| COLMAP | DUSt3R |
| --- | --- |
| SIFT | ViT feature |
| Feature Matching | Cross-attention |
| RANSAC | Confidence |
| Triangulation | Pointmap regression |
| PnP | Procrustes alignment |
| Epipolar constraint | Learned consistency |

DUSt3R는

COLMAP의 핵심 불안정 요소들을 **통째로 제거**함.

---

## 4. MASt3R의 등장 배경

DUSt3R는 강력하지만, 한계도 명확함.

- dense matching 비용이 큼
- 고해상도 이미지에서 느림
- image pair 수가 늘면 계산량 폭증

MASt3R는 이를 해결하기 위해 설계됨.

---

## 5. MASt3R의 핵심 목표

MASt3R의 목표는 명확함.

- DUSt3R 수준의 robustness 유지
- 더 빠른 matching
- 더 많은 이미지, 더 높은 해상도 처리
- 실전 SfM / MVS 파이프라인에 적합

---

## 6. MASt3R 전체 구조 개요

![image.png](style/image/dust3r_paper/2.png)

> **[Figure] MASt3R 전체 파이프라인 다이어그램**
> 

이미지 설명:

- ViT backbone 기반 feature 추출
- DUSt3R-style Head3D 유지
- 추가적인 Matching Prediction Head
- coarse-to-fine matching 구조

---

### MASt3R 네트워크 출력 정의

MASt3R는 이미지 pair (I1, I2)를 입력으로 받아

다음과 같은 값을 예측함.

- Pointmap:
    - P1: H×W 픽셀 → 3D 좌표
    - P2: H×W 픽셀 → 3D 좌표
- Correspondence:
    - 각 픽셀 쌍이 같은 3D 점인지에 대한 매칭 확률
- Confidence:
    - 각 예측이 얼마나 신뢰할 수 있는지 나타내는 값

이를 함수 형태로 쓰면 다음과 같음.

(P1, P2, C, W) = fθ(I1, I2)

- C: correspondence score
- W: confidence weight

이 confidence는 이후 모든 loss와 정렬 과정에서

가중치로 사용됨.

---

## 7. MASt3R의 4가지 핵심 알고리즘

### 7.1 DUSt3R Framework 유지

- scale normalization만 제거
- 나머지는 DUSt3R 구조 그대로 사용
- pointmap + confidence 출력 유지

이유:

![image.png](style/image/dust3r_paper/3.png)

(예시)

- 극단적인 viewpoint 변화에서도
pointmap 품질이 매우 안정적이기 때문

---

### 7.2 Matching Prediction Head

![image.png](style/image/dust3r_paper/4.png)

> **[Figure] DUSt3R of architecture**
> 

![image.png](style/image/dust3r_paper/5.png)

> **[Figure] Matching Prediction Head 구조**
> 

MASt3R는 DUSt3R에

**추가적인 matching 전용 head**를 붙임.

- correspondence 정확도 향상
- matching 실패율 감소
- SfM 초기화 안정성 증가

---

### Confidence-weighted Matching Loss

MASt3R의 matching loss는

모든 픽셀을 동일하게 취급하지 않음.

각 correspondence는 confidence w를 가짐.

Loss 개념은 다음과 같음.

```python
L_match = Σ w(x1, x2) · || P1(x1) − (R · P2(x2) + t) ||²
```

의미:

- confidence가 높은 대응점일수록 loss에 크게 기여
- 잘못된 매칭, 가려진 점, reflection은 자연스럽게 무시됨

이 구조는 전통적인 SfM의

RANSAC + outlier rejection을

신경망 내부로 흡수한 형태라고 볼 수 있음.

---

### 7.3 Fast Reciprocal Matching

![image.png](style/image/dust3r_paper/6.png)

기존 dense matching은

모든 픽셀 쌍을 고려해야 해서 느림.

MASt3R는 다음 전략 사용.

- 이미지 일부 픽셀만 사용
- reciprocal matching (서로 가리키는 경우만 유지)
- confidence 기반 필터링

→ 속도 대폭 개선

---

### 7.4 Coarse-to-Fine Matching

과정은 다음과 같음.

1. 저해상도에서 coarse matching
2. 관련 영역(window) 선택
3. 고해상도에서 해당 영역만 fine matching

→ 불필요한 연산 제거

→ 고해상도에서도 효율적

---

## 8. Intrinsics Recovery (카메라 내부 파라미터)

MASt3R / DUSt3R는

카메라 intrinsic이 없어도 작동함.

아이디어:

- pointmap이 image 좌표계에 정의됨
- 이를 활용해 focal length 추정 가능

Weiszfeld algorithm을 사용해

robust하게 intrinsics 복구함.

---

## 9. Global Alignment

여러 image pair에서 얻은 결과를

하나의 전역 좌표계로 정렬해야 함.

방법:

- 각 pair는 local coordinate
- Procrustes alignment로 R, t 추정
- confidence 높은 점에 더 큰 가중치

결과:

- 모든 카메라 pose
- 모든 3D point
→ 하나의 일관된 global space로 정렬됨

이후 COLMAP의 BA로 refinement 가능.

### Procrustes Alignment (전역 정렬 수식)

각 image pair는 자신만의 local 좌표계를 가지므로

전역 좌표계로 정렬해야 함.

이를 위해 Procrustes alignment를 사용함.

목표는 다음 최적화 문제를 푸는 것임.

```python
min_{R, t} Σ w_i · || R · P_i + t − Q_i ||²
```

- P_i: local pointmap의 3D 점
- Q_i: 전역 좌표계 기준 3D 점
- R: 회전 행렬
- t: 이동 벡터
- w_i: confidence 기반 가중치

confidence가 높은 점일수록

전역 정렬에 더 큰 영향을 미침.

이후 COLMAP의 Bundle Adjustment에서

reprojection error를 직접 최소화함.

---

## 10. COLMAP + MASt3R 파이프라인

실전 활용 예시는 다음과 같음.

1. 비디오 → frame 추출
2. MASt3R로 pointmap + pose graph 생성
3. COLMAP에 입력
4. Bundle Adjustment
5. Dense reconstruction / mesh 생성

---

## 11. 장점 요약

- 반복 패턴에 매우 강함
- calibration 변화에 둔감
- registration 실패 거의 없음
- 실내 / object scan에 강력
- NeRF 이전 단계로도 활용 가능

---

## 12. DUSt3R vs MASt3R 정리

| 항목 | DUSt3R | MASt3R |
| --- | --- | --- |
| Matching | 정확하지만 느림 | 더 빠르고 안정적 |
| 해상도 | 제한적 | 고해상도 가능 |
| Pair 수 | 증가 시 부담 | 대규모 처리 가능 |
| 실전성 | 연구 중심 | 실사용 지향 |

## 수식 관점에서 본 MASt3R의 의미

MASt3R는 기존 SfM의 수식을 버린 것이 아니라,

- epipolar constraint
- triangulation
- RANSAC
- PnP

를 **명시적 알고리즘 → 미분 가능한 학습 문제**로 변환함.

즉,

기하 수식은 유지하되

해결 방식을 최적화 문제로 바꾼 접근임.

이 점에서 MASt3R는

순수 딥러닝 모델이라기보다

"학습 가능한 SfM"에 가깝다고 볼 수 있음.

---

## 13. 마무리

MASt3R는

“SfM을 신경망으로 대체할 수 있는가?”라는 질문에

가장 현실적인 답을 제시한 모델 중 하나임.

- NeRF 이전 단계의 geometry 확보
- 기존 SfM의 취약점 제거
- 대규모 3D reconstruction 가능

앞으로는

**DUSt3R / MASt3R + NeRF / Gaussian Splatting**

조합이 사실상 표준 파이프라인이 될 가능성이 큼.