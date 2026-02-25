---
layout: post
title: NeRF 논문리뷰
date: 2025-03-15 11:30:00 +0800
category: paper
thumbnail: /style/image/nerf_paper/1.png
icon: code
---

# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

# NeRF (Neural Radiance Fields) 논문 리뷰

## 📌 개요

NeRF - Neural Radiance Fields. 여러 시점의 **2D 이미지 + 카메라 포즈**를 입력으로 받아 장면의 **연속적인 3D 표현을 Neural Network로 학습**하고, **새로운 시점에서 사실적인 이미지 생성(Novel View Synthesis)**이 가능한 모델임.  [oai_citation:0‡arXiv](https://arxiv.org/abs/2003.08934?utm_source=chatgpt.com)

---

## 🧠 핵심 아이디어

NeRF의 핵심은 **신경망을 이용한 5D 장면 표현**임.

임의의 위치 (x, y, z)와 방향 (θ, φ)를 입력으로 받아

해당 위치의 **색(RGB)과 밀도(σ)**를 예측함.

(NeRF 원 논문, ECCV 2020)

### NeRF의 장면 함수

NeRF는 장면을 다음과 같은 함수로 표현함:

F_Θ(x, y, z, θ, φ) → (c_r, c_g, c_b, σ)

의미는 다음과 같음.

- (x, y, z): 3D 공간 상의 위치 좌표
- (θ, φ): 해당 지점을 바라보는 시선 방향(view direction)
- (c_r, c_g, c_b): 예측된 RGB 색상 값
- σ (sigma): volume density
→ 해당 지점이 얼마나 불투명한지를 나타내는 값

즉, NeRF는 **공간의 한 점 + 바라보는 방향**이 주어졌을 때

그 지점에서 관측되는 색과 밀도를 신경망으로 예측함.

---

## 📊 작동 원리 (Volume Rendering)

NeRF는 **differentiable volume rendering**을 기반으로 학습함.

카메라 위치에서 픽셀마다 하나의 ray를 쏘고,

그 ray를 따라 여러 지점을 샘플링한 뒤 색과 밀도를 누적하여

최종 픽셀 색을 계산함.

### Volume Rendering 개념 정리

- 하나의 픽셀 = 하나의 ray
- ray를 따라 N개의 샘플 포인트를 선택
- 각 포인트에서
    - 색(color)
    - 밀도(density)
    를 NeRF MLP로 예측
- 앞쪽 점일수록 더 큰 영향
- 밀도가 높을수록 뒤쪽은 가려짐

픽셀 색 C(r)는 다음 개념을 따름:

- 앞에 있는 점이 불투명하면 뒤쪽 기여 감소
- 모든 샘플의 색을 **가중합**으로 누적

이를 수식 기반 적분으로 정의했지만,

실제 구현에서는 **유한 개 샘플을 이용한 합(sum)**으로 근사함.

즉,

ray를 따라 N개의 점을 뽑고

각 점의 색 × 투과 확률을 누적하는 방식임.

---

## 📌 Positional Encoding (좌표 인코딩)

NeRF는 일반적인 MLP가

고주파 공간 정보를 잘 학습하지 못하는 문제를 해결하기 위해

**Fourier 기반 Positional Encoding**을 사용함.

핵심 아이디어는 다음과 같음.

- 원래 좌표 p (예: x, y, z)
- 이를 sin, cos 함수로 여러 주파수 대역으로 변환
- 저주파 + 고주파 정보를 함께 입력

개념적으로는 아래와 같음:

γ(p) =

[ sin(π·p), cos(π·p),

sin(2π·p), cos(2π·p),

sin(4π·p), cos(4π·p), … ]

이렇게 변환된 벡터를 MLP에 입력하면,

- 평면, 곡면 같은 저주파 구조
- 모서리, 질감 같은 고주파 디테일

을 동시에 잘 표현할 수 있음.

Positional Encoding이 없으면

NeRF 결과가 **흐릿하고 디테일이 사라지는 현상**이 발생함.

---

## 🛠 NeRF 학습 파이프라인

1. 여러 시점에서 이미지 캡처 + 카메라 포즈 확보
2. 각 이미지의 픽셀마다 ray 생성
3. ray를 따라 다수 샘플 생성
4. MLP로 샘플 위치/방향을 입력 → 색/밀도 출력
5. volume rendering으로 예측 이미지 생성
6. 예측 이미지와 실제 이미지 차이로 **MSE 최적화**
7. 반복을 통해 네트워크 파라미터 최적화

이 과정이 **end-to-end 미분 최적화**로 진행됨.

---

## 📈 NeRF 모델 구조

![image.png](style/image/nerf_paper/1.png)

입력 이미지 (Input Images): 장면의 다양한 각도에서 촬영된 여러 장의 이미지가 입력으로 사용. 입력 데이터는 실제 이미지가 아닌 3D 모델링 된 드럼 세트임.
NeRF 최적화 (Optimize NeRF): 입력 이미지와 해당 카메라 포즈를 사용하여 신경망(MLP)을 학습시켜 장면의 연속적인 5D 표현(3D 위치와 시점 방향에 따른 색상 및 밀도)을 생성. 이 단계는 3D 공간에 퍼져 있는 점들로 표현되는 잠재적인 장면을 나타냄.
새로운 뷰 렌더링 (Render new views): 최적화된 NeRF 모델을 사용하여 장면의 새로운 시점에서 이미지를 렌더링.

아래는 NeRF 모델의 기본 구조 다이어그램임:

```python
(x, y, z) ── Γ PE ──▶ MLP ──▶ (σ, c)
(direction) ────────▶ MLP ─▶ (c view-dep)
```

- PE: Positional Encoding
- MLP: 완전 연결 신경망
- view-dependent color 처리로 **방향성 표현 강화**

---

## 📷 결과와 특성

### 📌 장점

- **사실적인 Novel View 생성**이 가능함. [oai_citation:6‡위키백과](https://en.wikipedia.org/wiki/Neural_radiance_field?utm_source=chatgpt.com)
- 3D 메쉬/포인트 없이도 **연속적 scene 표현**을 달성함. [oai_citation:7‡위키백과](https://en.wikipedia.org/wiki/Neural_radiance_field?utm_source=chatgpt.com)
- 밀도 기반 표현으로 **반사/투명 효과**까지 묘사 가능함. [oai_citation:8‡Medium](https://medium.com/klleon/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-neural-radience-fields-nerf-2a32b817ca38?utm_source=chatgpt.com)

### ⚠ 단점

- 학습 및 렌더링이 **매우 느림 (수시간~수일)**
- 많은 view가 필요함
- dynamic scene에는 적용 어려움

## Neural Radiance Field Scene Representation

![image.png](style/image/nerf_paper/2.png)

(a) View 1 과 (b) View 2: 이는 두 개의 다른 시점에서 렌더링된 동일한 장면(배)을 보여줍니다. 확대된 영역을 보면, 특정 3D 지점(다이아몬드와 사각형으로 표시)이 다른 시점에서 어떻게 다르게 보이는지 확인할 수 있습니다. 특히, 배 표면의 금속 부분이나 물결의 질감이 보는 각도에 따라 빛나는 정도나 색상이 달라지는 것을 시사합니다.
(c) Radiance Distributions: 이 부분은 각 3D 지점(다이아몬드, 사각형, 삼각형, 원으로 표시)에서 방출되는 빛의 색상과 강도가 시야 방향에 따라 어떻게 달라지는지를 나타냅니다.
오렌지색 원 안에 있는 다이아몬드와 사각형은 서로 다른 시점에서 본 배 표면의 특정 지점을 나타냅니다. 이 지점들에서 방출되는 빛의 색상 분포가 시야 방향에 따라 변하는 것을 보여줍니다. 예를 들어, 보는 각도에 따라 밝게 빛나는 부분(specularity)이 나타날 수 있습니다.
청록색 원 안에 있는 삼각형과 원은 물 표면의 특정 지점을 나타냅니다. 이 지점에서도 시야 방향에 따라 빛의 표현이 달라짐을 보여줍니다.

핵심 아이디어는 NeRF 모델이 단순히 3D 위치(x, y, z)뿐만 아니라 **보는 방향(θ, φ)**까지 고려하여 각 지점에서 방출되는 색상(RGB)을 결정한다는 것입니다. 이렇게 하면 금속의 반짝임과 같은 시야 의존적인 재질 표현이 가능해져 더욱 사실적인 이미지를 생성할 수 있습니다.

![image.png](style/image/nerf_paper/3.png)

Ground Truth (원본): 실제 장면 또는 가장 정확한 렌더링을 나타냅니다. 이 이미지에서는 레고로 만들어진 굴삭기 모습이 선명하게 보입니다.
Complete Model (완전 모델): NeRF의 모든 기술(위치 인코딩, 시야 의존적 방출 휘도, 계층적 샘플링 등)을 적용했을 때의 결과입니다. 원본 이미지와 매우 유사한 결과물을 보여주며, 레고 블록의 질감과 굴삭기 트랙의 디테일을 잘 표현하고 있습니다.
No View Dependence (시야 의존성 없음): NeRF 모델에서 시야 방향에 따른 색상 변화를 고려하지 않았을 때의 결과입니다. 이를 제거하면 굴삭기 트랙의 금속성 반사와 같은 시야 의존적인 효과가 제대로 재현되지 않아, 다소 밋밋하고 평평한 질감으로 나타납니다.
No Positional Encoding (위치 인코딩 없음): NeRF 모델에서 입력 좌표에 대한 위치 인코딩을 적용하지 않았을 때의 결과입니다. 위치 인코딩은 고주파수 정보(세밀한 디테일)를 표현하는 데 도움을 주는데, 이를 제거하면 장면이 전반적으로 흐릿하고 세부 묘사가 부족한 모습을 보입니다. 특히 레고 블록의 날카로운 모서리나 트랙의 패턴이 뭉개져 보입니다.

이 이미지는 NeRF의 핵심 요소인 위치 인코딩과 시야 의존적 방출 휘도가 사실적인 3D 장면 렌더링에 얼마나 중요한 역할을 하는지를 명확하게 보여줍니다.

---

![image.png](style/image/nerf_paper/4.png)

NeRF는 복잡한 3D 장면을 연속적인 볼륨 함수로 표현하여 새로운 시점의 이미지를 합성하는 방법을 제시함.

(a) 5D 입력 (위치 + 방향):

장면 내의 특정 3D 공간 위치 (x, y, z)와 해당 지점에서 바라보는 2D 시야 방향 (θ, φ)를 입력으로 받습니다.
이 5D 좌표는 장면을 표현하는 딥러닝 모델 F_Θ (MLP)에 입력됩니다.

(b) 출력 (색상 + 밀도):

F_Θ 모델은 입력된 5D 좌표에 대해 해당 지점의 부피 밀도 σ (얼마나 불투명한지)와 시야 방향에 따른 RGB 색상 c를 출력합니다.
이러한 색상과 밀도 값들은 장면을 통과하는 카메라 광선(Ray 1, Ray 2 등)을 따라 샘플링됩니다.

(c) 볼륨 렌더링:

볼륨 렌더링 기법을 사용하여 광선을 따라 샘플링된 각 지점의 색상과 밀도를 최종 2D 이미지 픽셀 값으로 통합합니다.
그래프는 광선 경로를 따라 밀도 σ가 어떻게 분포하는지를 보여줍니다. 밀도가 높은 부분은 더 불투명하여 더 많은 색상 정보를 기여합니다.

(d) 렌더링 손실:

NeRF 모델이 생성한 렌더링 이미지와 실제 관측된 이미지(g.t.는 ground truth, 즉 실제 이미지) 간의 차이를 계산하여 손실(loss)을 구합니다.
이 손실 함수(||.||_2^2)는 렌더링된 이미지와 실제 이미지 간의 제곱 오차 합입니다.
이 손실을 최소화하는 방향으로 F_Θ 모델의 가중치 Θ를 학습시킵니다. 미분 가능한 렌더링 과정을 통해 이 손실이 네트워크에 역전파되어 학습이 이루어집니다.

### 요약

이 그림은 NeRF가 5D 좌표를 입력으로 받아 각 지점의 색상과 밀도를 예측하고, 이를 볼륨 렌더링을 통해 이미지로 합성한 뒤, 실제 이미지와의 차이를 최소화하도록 학습하는 전체적인 과정을 보여줍니다.

---

## 📌 네트워크 시각화 (MLP 구조)

```python
Input: 5D coordinate (x,y,z,θ,φ)
↓ Positional Encoding
↓ Fully-connected layers (MLP)
Output: Volume density (σ), RGB color (c)
```

---

## 🧠 응용 및 확장

NeRF 이후 다양한 파생 연구가 등장함:

- **LLFF / IBRNet**: 포즈/뷰 종속 일반화 강화
- **FastNeRF / PlenOctrees / Instant NGP**: 빠른 트레이닝/렌더링
- **NeRF in the Wild**: 조명/카메라 변동 대응
- **Gaussian Splatting**: NeRF보다 빠른 실시간 렌더링

이러한 연구들은 NeRF의 **연속적 implicit scene 표현**을 기반으로 확장됨.  [oai_citation:9‡LG AI Research](https://www.lgresearch.ai/blog/view?seq=237&utm_source=chatgpt.com)

---

## 📎 참고 논문 정보

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**

- ECCV 2020
- NeRF는 신경망 기반의 연속 장면 표현으로 novel view synthesis 문제를 해결함. [oai_citation:10‡arXiv](https://arxiv.org/abs/2003.08934?utm_source=chatgpt.com)