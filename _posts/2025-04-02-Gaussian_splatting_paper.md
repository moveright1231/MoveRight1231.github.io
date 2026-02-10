---
layout: post
title: 3D Gaussian Splatting 논문리뷰
date: 2025-04-02 21:30:00 +0800
category: paper
thumbnail: style/image/gs_paper/1.png
icon: code
---

# Gaussian Splatting 논문리뷰

# 3D Gaussian Splatting: 실시간 Radiance Field Rendering 리뷰

## 1. 기존 방법의 한계

- *메쉬(Mesh)와 포인트 클라우드(Point Cloud)**는 명시적 3D 구조이므로 GPU 기반 래스터화에 적합하지만, 복잡한 표면에서는 찢어짐이나 구멍이 발생해 사실적인 장면 재현에 어려움이 있다.
- **NeRF** 같은 연속적 Radiance Field 모델은 높은 시각적 품질을 제공하지만, 모든 픽셀마다 MLP를 쿼리하고 레이트레이싱을 해야 하므로 매우 느리고, Mip‑NeRF360과 같은 SOTA 모델은 학습에 최대 48시간이 걸린다.

이러한 한계 때문에 실시간 렌더링이 가능한 새로운 장면 표현이 필요하다.

![image.png](style/image/gs_paper/1.png)

## 2. 3D Gaussian Splatting의 핵심 아이디어

3D Gaussian Splatting(3DGS)은 장면을 **수천~수백만 개의 3D Gaussian 분포**로 표현하고, 이를 2D 화면으로 프로젝트하여 실시간으로 렌더링하는 방법이다.

- 각 **3D Gaussian**은 위치(μ), 크기와 형태(공분산 Σ), 회전(R, quaternion), 색상(구면 조화 계수), 불투명도(α) 등 5개의 파라미터를 가진다. 이러한 파라미터는 모두 최적화(학습해야할) 대상이다.
- Gaussian을 2D 이미지 상으로 투영하면 타원 형태의 ‘스플랫(splat)’이 되며, 스플랫의 색과 불투명도를 **α‑블렌딩** 방식으로 누적해 픽셀 색을 계산한다.
- NeRF처럼 MLP를 사용하지 않고 명시적 3D 구조를 유지하면서도, 수식은 NeRF와 동일한 이미지 생성 모델을 사용하기 때문에 NeRF의 이점을 부분적으로 계승한다.

이렇게 작은 반투명 안개 같은 Gaussian들을 수백만 개 쌓아 올려 한 장면을 표현하는 것이 3DGS의 기본 아이디어다.

## 3. 3D Gaussian 모델링

### 3.1 초기화

- 입력은 **Structure‑from‑Motion(SfM)** 과정에서 추출한 sparse point cloud와 캘리브레이션된 카메라 파라미터다. 대부분의 포인트 기반 방법과 달리 Multi‑View Stereo 데이터가 필요 없으며, NeRF‑synthetic 데이터셋에서는 무작위 초기화만으로도 높은 품질을 얻을 수 있다.
- 초기 Gaussian의 위치는 SfM 점의 좌표에서 가져오며, 초기 공분산은 이웃 점까지의 평균 거리로 추정된 등방성 Gaussian으로 설정한다.

### 3.2 공분산 표현

- 공분산 행렬 Σ는 **스케일 행렬 S**와 **회전 행렬 R**의 곱으로 표현되어 양의 준정부호성을 유지한다.
- 스케일은 3D 벡터 s로, 회전은 quaternion q로 저장하여 각각 독립적으로 최적화하고 q를 정규화하여 유효한 회전을 보장한다.
- 이러한 분해 덕분에 gradient descent로 업데이트해도 Σ가 유효하지 않은 값을 갖는 문제를 피할 수 있으며, 이방성 Gaussian이 장면의 다양한 형태를 효과적으로 모델링한다.

### 3.3 색상과 불투명도

- 각 Gaussian의 색상은 **Spherical Harmonics (SH) 계수**로 표현되어 시점 변화에 따른 색 변화를 저비용으로 모델링한다.
- 불투명도 α는 sigmoid 함수를 통해 0~1 범위로 제한되어 부드럽게 변화하며, 투명한 Gaussian은 학습 과정 중 삭제된다.

## 4. Differentiable Rendering: 타일 기반 Rasterization

### 4.1 2D 투영과 키 생성

- 각 Gaussian은 외부 카메라 변환 W와 내부 카메라 행렬 K를 통해 3D에서 2D로 투영된다. 공분산은 projection의 Jacobian J를 곱해 2×2 분산 행렬로 변환한다.
- 3D 스플랫들은 **view frustum** 내부에 있는지 확인하는 culling을 거친 뒤, **16×16 타일**로 분할된 화면에서 어느 타일에 속하는지에 따라 키(key)가 부여된다.
- 모든 Gaussian 키를 GPU Radix Sort로 정렬해, 타일별로 깊이 우선 순서를 갖는 리스트를 생성한다.

### 4.2 α‑블렌딩과 병렬 처리

- 각 타일에 대해 하나의 스레드 블록이 할당되고, 정렬된 Gaussian 리스트를 순회하면서 α‑블렌딩을 수행한다.
- 픽셀의 누적 α가 1에 가까워지면 해당 스레드는 조기에 종료되어 불필요한 계산을 줄인다.
- 그 결과 전체 장면의 forward pass와 backward pass 모두가 빠르게 수행되며, 각 pixel에서 혼합된 Gaussian 수를 제한하지 않아도 된다.

이 타일 기반 rasterization 덕분에 3DGS는 1080p 해상도에서도 **30fps 이상의 실시간 렌더링**을 달성한다.

## 5. 최적화와 Gaussian Densification

![image.png](style/image/gs_paper/2.png)

Operation Flow (검은색 화살표):

데이터의 흐름 및 연산 과정.
SfM 포인트에서 초기화, 3D 가우시안의 생성, 투영, 래스터라이저를 통한 이미지 생성 등의 순서로 진행.

Gradient Flow (파란색 화살표):

역전파 과정에서 gradient가 흐르는 방향.
래스터라이저에서 생성된 이미지와 실제 이미지 간의 오차를 기반으로 계산된 gradient가 3D 가우시안의 속성(위치, 공분산, 색상, 불투명도)과 적응형 밀도 제어 과정에 영향을 미침.

![image.png](style/image/gs_paper/3.png)

이 피규어에서 저자는 3D 가우시안 표현이 복잡한 지오메트리를 어떻게 표현하는지 보여줌.

- original 이미지는 원본으로 빨간 bbox의 환기구 부분을 사용
- Shrunken Gaussians 이미지는 bbox 영역을 가우시안을 축소 시킨 후에 렌더링 한 결과
- Original 이미지와 비교했을 때, Shrunken Gaussians 이미지의 환기구는 더 어둡고 뭉쳐 보이는 경향이 보임.
이는 3D 가우시안이 원래 이미지의 디테일을 얼마나 정확하게 포착하고 재현하는지를 보여주는 예시로.
가우시안을 축소시켰을 때, 얇은 격자선과 같은 미세한 구조에서 정보 손실이 발생하거나 표현이 뭉개질 수 있음을 보여줌.
이러한 결과는 3D 가우시안이 장면을 표현하는 방식의 특성을 보여주며, 특히 매우 얇거나 복잡한 구조를 표현할 때 가우시안의 크기, 모양(공분산), 그리고 밀도(opacity) 최적화가 얼마나 중요한지를 저자들은 강조함.

### 5.1 최적화 목표

- 학습은 예측된 렌더링 이미지와 실제 학습 뷰 이미지 간의 L1 손실과 D‑SSIM 손실을 결합한 loss 함수로 수행된다. 논문에서는 L1 가중치를 0.8, SSIM 가중치를 0.2로 설정한다.
- 최적화는 각 Gaussian의 위치, 공분산, SH 계수, α를 SGD로 업데이트하며, 일부 연산에는 커스텀 CUDA 커널이 사용된다.

### 5.2 Adaptive Density Control

3DGS는 학습 과정에서 Gaussian의 수와 분포를 **적응적으로 제어**한다.

- **Remove**: 불투명도 α가 0.005 미만이거나 크기가 너무 큰 Gaussian은 삭제한다.
- **Split**: 크기가 커서 배경을 제대로 표현하지 못하는 Gaussian은 두 개의 더 작은 Gaussian으로 분할하고 크기를 1/1.6로 줄인다.
- **Clone**: Under‑reconstruction 영역의 Gaussian은 동일한 크기의 Gaussian을 복제한 후 gradient 방향으로 이동시켜 더 많은 세부 구조를 포착한다.

이러한 densification 단계는 100번의 iteration마다 수행되어 모델이 필요 이상으로 커지는 것을 방지하고, floaters나 over‑reconstruction 문제를 해결한다.

![image.png](style/image/gs_paper/4.png)

- 너무 작거나 큰 분포에대해 면적 사이즈에 맞춰주는 Optimization 진행

Under-Reconstruction (과소 재구성):

상단 행은 장면의 일부가 불충분하게 재구성되었을 때(즉, 빈 공간이 많거나 세부 정보가 부족할 때) 발생하는 상황을 보여줍니다.
이 경우, 기존의 3D 가우시안을 **'Clone' (복제)**하여 동일한 크기와 모양으로 하나 더 만듭니다.
이 복제된 가우시안은 종종 원래 가우시안의 위치에서 약간 이동하여, 재구성되지 않은 공간을 더 잘 채우도록 합니다.
이후 **'Optimization Continues' (최적화 계속)**되면서, 이 새로운 가우시안들이 장면의 부족한 부분을 채워나가도록 학습됩니다.

Over-Reconstruction (과대 재구성):

하단 행은 장면의 특정 부분이 너무 많은 가우시안으로 표현되었거나, 하나의 큰 가우시안이 넓은 영역을 덮어 세부 묘사가 부족할 때(즉, 과도하게 표현되었을 때) 발생하는 상황을 보여줍니다.
이 경우, 기존의 크고 넓은 3D 가우시안을 **'Split' (분할)**하여 더 작고 세밀한 두 개의 가우시안으로 나눕니다.
이 분할 과정은 원래 가우시안의 스케일(크기)을 줄이고, 원래 가우시안이 차지했던 영역을 더 정교하게 표현하도록 합니다.
이후 **'Optimization Continues' (최적화 계속)**되면서, 이 더 작은 가우시안들이 장면의 해당 부분을 더 정확하게 표현하도록 학습됩니다.

## 6. 결과 및 장점

- 3D Gaussian Splatting은 Mip‑NeRF360 등 SOTA NeRF 방법과 **동등하거나 더 나은 시각적 품질**을 보여주며, 학습 속도는 InstantNGP 수준으로 빠르고, 1080p 해상도에서도 **실시간 렌더링**을 지원한다.
- 기존 NeRF 기반 방법과 달리 **implicit 뉴럴 네트워크가 렌더링에 필요하지 않기 때문에 모바일 또는 웹 환경(WebGL, AR)에서 실행하기 쉽다**.
- 명시적 3D Gaussian은 하드웨어 친화적이며, tile‑based GPU rasterization 덕분에 Backpropagation까지 빠른 속도로 수행된다.

## 7. 결론 및 전망

3D Gaussian Splatting은 **SfM에서 얻은 sparse 포인트를 기반으로 3D Gaussian들을 초기화**하고, **파라미터 최적화와 적응형 densification**, **타일 기반 differentiable rasterization**을 결합하여 **실시간 고품질 novel‑view synthesis**를 가능하게 한다.

이 방법은 explicit하게 3D 장면을 표현하면서도 NeRF 수준의 품질을 유지하고, 실시간 응용(AR/VR, 게임, 모바일)에서 즉시 활용될 수 있다. 향후 연구에서는 Gaussian 수를 더욱 줄이거나 dynamic scene 및 lighting 변화에 대한 확장, 메모리 최적화, Gaussians 간의 feature 학습 등 다양한 방향으로 발전할 것으로 기대된다.

![image.png](style/image/gs_paper/5.png)

전반적인 품질:
대부분의 장면에서 'Ours' (제안하는 방법)는 Ground Truth와 매우 유사한 품질을 보여줍니다.
Mip-NeRF360, InstantNGP, Plenoxels 역시 특정 장면에서는 좋은 결과를 보여주지만, 때때로 미묘한 차이점이나 아티팩트(artifact)를 보임.

주요 차이점 (화살표 및 확대 이미지로 표시된 부분):
-자전거 장면: Mip-NeRF360과 Plenoxels에서 자전거에 미묘한 왜곡이나 반사광 표현이 다르게 나타납니다.
-테이블 장면: Mip-NeRF360과 InstantNGP에서 확대된 부분(창문)의 디테일이 Ground Truth와 다릅니다.
-주방 식기 장면: Mip-NeRF360에서 식기에 빨간색 화살표가 가리키는 부분에 약간의 연기나 흐릿함이 보입니다.
-거실 장면: Plenoxels에서 벽면이나 가구의 디테일이 약간 뭉개지거나 흐릿하게 보입니다.
-장난감 장면: InstantNGP와 Plenoxels에서 장난감의 디테일이 약간 흐릿하게 보이며, 특히 'A' 글자 주변의 녹색 테두리가 InstantNGP와 Plenoxels에서 더 선명하게 표현된 반면, 'Ours'는 Ground Truth와 유사하게 표현되었습니다.
-방 장면: Plenoxels에서 벽면의 디테일이 흐릿하게 보이며, 연기가 낀 듯한 아티팩트가 나타납니다.
-자동차 장면: Plenoxels에서 자동차 뒷부분에 연기 같은 아티팩트가 보이며, Mip-NeRF360에서 사람 형상에 빨간색 화살표가 가리키는 부분에 미묘한 차이가 있습니다.
기차 장면: Plenoxels에서 기차 앞부분에 연기 같은 아티팩트가 두드러집니다.

→결론적으로, 이 그림은 3D Gaussian Splatting이 다양한 장면에서 기존 방법들과 동등하거나 때로는 더 나은 시각적 품질을 제공하며, 특히 아티팩트를 줄이는 데 효과적임을 보여줍니다.

# **3D Gaussian Splatting – 실시간 Radiance Field 렌더링 기술 자세히 살펴보기**

## **1. 배경: 기존 3D 표현과 NeRF의 한계**

- **메시와 포인트 클라우드**는 명시적(explicit) 3D 표현이라서 GPU 래스터화에 적합하지만 복잡한 표면에서는 삼각형이 찢어지거나 구멍이 생기기 쉽다 .
- *NeRF (Neural Radiance Fields)**는 다수의 이미지를 사용해 MLP를 최적화하고 레이 마칭을 통해 새로운 시점의 색을 예측한다. NeRF는 고품질 이미지를 얻을 수 있지만, 모든 픽셀에서 MLP를 질의해야 하기 때문에 렌더링이 매우 느리며 대규모 장면을 실시간으로 합성하기 어렵다 .
- 최근 Plenoxels, InstantNGP 같은 가속형 radiance field가 등장했지만 품질‑속도 트레이드오프가 있어 Mip‑NeRF360 수준의 품질을 얻기 위해서는 수십 시간 이상의 훈련이 필요하다 .

이러한 제약 때문에 **높은 품질을 유지하면서도 실시간으로 novel‑view synthesis가 가능한 새로운 3D 표현**이 요구된다.

## **2. 3D Gaussian Splatting의 핵심 아이디어**

논문 “3D Gaussian Splatting for Real‑Time Radiance Field Rendering”에서는 NeRF와 포인트 기반 방법의 장점을 결합한 새로운 표현을 제안한다. 핵심 아이디어는 다음과 같다.

1. **3D 장면을 수천 개의 3D Gaussian 분포**로 표현한다. 각 Gaussian은 위치(mu), **이방성 공분산**(스케일 S와 회전 R으로 분해), **색상**(구면 조화 계수), **불투명도** alpha를 포함한다 .
2. Gaussian을 2D 영상으로 투영해 **스플랫(splat)**으로 만들어 픽셀 위에 누적한다. NeRF처럼 MLP를 사용하지 않고도 연속적 Radiance Field와 동등한 이미지 생성 모델을 제공하므로 표준 alpha 블렌딩으로 빠르게 렌더링할 수 있다 .
3. *SfM (Structure‑from‑Motion)**으로 추출한 sparse 포인트 클라우드와 카메라 파라미터만으로 초기 Gaussian을 만들고, **인터리브된 최적화/밀도 제어** 및 **빠른 타일 기반 래스터화**를 통해 높은 품질과 실시간 속도를 달성한다 .

이렇게 하면 연속적 Radiance Field의 품질을 유지하면서도 explicit 3D 프리미티브를 사용해 GPU에서 실시간 렌더링할 수 있다.

## **3. 3D Gaussian 표현 세부 구조**

### **3.1 Gaussian 분포와 이미지 투영**

각 3D Gaussian은 **평균 mu와 공분산 Sigma**로 정의된다. 공간상의 한 점 x에 대한 Gaussian 값은 다음과 같다 .

```python
G(x) = exp(-0.5 * (x - mu)^T * inverse(Sigma) * (x - mu))
```

렌더링을 위해서는 3D Gaussian을 카메라 뷰로 투영해야 한다. **외부 변환** W와 **내부 카메라 행렬** K가 주어질 때, Gaussian의 평균과 공분산은 다음과 같이 변환된다 .

- **평균 투영:** mu’ = K * W * mu
- **공분산 투영:** Sigma’ = J * W * Sigma * transpose(W) * transpose(J), 여기서 J는 원근 투영의 아핀 근사의 Jacobian이다 .

투영된 Gaussian은 화면에서 2D 타원 형태의 스플랫이 된다. 모든 Gaussian의 스플랫을 alpha 블렌딩으로 누적하면 최종 픽셀 색을 얻을 수 있다 .

### **3.2 공분산의 이방성 표현**

공분산 행렬 Sigma는 3D Gaussian의 형태를 나타내지만, 경사 하강법으로 직접 최적화하면 양의 준정부호(positive semi‑definite) 조건을 유지하기 어렵다 . 이를 해결하기 위해 저자들은 **스케일링 행렬 S**과 **회전 행렬 R**로 분해한다:

```python
Sigma = R * S * transpose(S) * transpose(R)
```

- **스케일링 S** – 대각 성분 s를 갖는 행렬로 Gaussian의 크기를 결정한다. 초기화 시 가장 가까운 세 포인트까지의 평균 거리를 이용하여 등방성 Gaussian을 설정한다 .
- **회전 R** – 회전을 나타내는 quaternion q로 저장하여 정규화 후 행렬로 변환한다 .

이 분해는 Sigma가 항상 유효한 공분산 행렬이 되도록 하며, 스케일과 회전을 서로 독립적으로 최적화할 수 있다 . 또한 자동 미분 오버헤드를 줄이기 위해 각 파라미터의 기울기를 명시적으로 유도한다 .

### **3.3 색상과 불투명도 파라미터**

- **색상:** 각 Gaussian의 색상은 **구면 조화 (Spherical Harmonics, SH)** 계수로 표현된다 . SH는 시점 방향에 따라 색이 변하는 물체의 반사 특성을 효율적으로 모델링하며, 뷰 의존 색을 MLP 없이 계산할 수 있다 .
- **불투명도 alpha:** Gaussian의 투명도는 0~1 사이에 존재해야 하므로 **sigmoid 활성화 함수**를 사용해 alpha = sigmoid(a)로 모델링한다 . alpha가 작아지면 해당 Gaussian은 학습 과정에서 삭제된다 .

### **3.4 초기화: SfM 기반 포인트에서 시작**

- 입력은 **SfM**을 통해 추정된 sparse point cloud와 보정된 카메라 파라미터다. NeRF와 달리 MVS 깊이 맵을 필요로 하지 않으며, 포인트 클라우드의 점 위치를 Gaussian의 평균으로 사용한다 .
- 공분산의 초기 크기는 인접한 포인트까지의 평균 거리로 정하며, 회전은 단위 quaternion으로 초기화한다 .
- Synthetic NeRF 데이터셋처럼 SfM 포인트가 없는 경우에는 Gaussian을 임의로 초기화하고도 좋은 품질을 얻을 수 있다 .

## **4. 최적화와 적응형 밀도 제어**

3DGS의 학습은 **렌더링과 이미지 비교를 반복하면서** Gaussian 파라미터를 최적화하는 과정이다 . 주요 요소는 다음과 같다.

### **4.1 손실 함수**

논문은 **L1 손실과 D‑SSIM 손실**을 결합한 손실을 사용한다. 전체 손실은 다음과 같다  :

```python
L = (1 - lambda) * L1 + lambda * L_DSSIM,  where lambda = 0.2
```

L1 손실은 색상 차이를, D‑SSIM은 구조적 유사도(SSIM) 기반 차이를 측정해 디테일 보존에 도움을 준다.

### **4.2 최적화 과정**

1. **Warm‑up:** 초기 단계에서는 낮은 해상도로 렌더링하여 안정적으로 학습을 시작한다. 250, 500 iteration 후 이미지 해상도를 두 번 업샘플한다 .
2. **파라미터 업데이트:** 위치, 스케일, 회전, SH 계수, alpha를 확률적 경사 하강법(SGD)으로 업데이트한다 . alpha는 sigmoid, 스케일은 지수(exponential) 활성화로 양의 값을 보장한다 .
3. **L1+D‑SSIM 손실을 통한 역전파**는 CUDA 커널로 최적화된 타일 기반 rasterization에 의해 가속된다 .

### **4.3 적응형 Gaussian 밀도 제어 (Densification)**

NeRF와 달리 Gaussian 개수는 학습 중에 증가하거나 감소한다. 저자들은 **두 가지 상황**에서 Gaussian을 추가하거나 분할하고, 불필요한 Gaussian을 제거한다 :

- **Under reconstruction:** 작은 Gaussian이 있는 영역에서 시점별 위치 기울기의 평균 크기가 일정 threshold tau_pos 이상일 때, **해당 Gaussian을 복제하고 기울기 방향으로 이동**시켜 더 많은 세부 구조를 생성한다 .
- **Over reconstruction:** 큰 Gaussian이 큰 영역을 덮고 있어 세부 묘사가 부족한 경우, **Gaussian을 두 개로 분할**하고 스케일을 실험적으로 설정된 phi=1.6으로 나눈다 .

100 iteration마다 이러한 densification을 수행하며, 일정 threshold 이하의 alpha를 가진 Gaussian은 삭제한다 . 또한 3000번마다 alpha 값을 0에 가깝게 재설정해 밀도가 과도하게 증가하는 것을 방지한다 . 이렇게 하면 **floaters** (카메라 근처에 떠 있는 Gaussian) 문제를 줄일 수 있고, 모델 크기를 효과적으로 제어할 수 있다 .

## **5. 빠른 미분 가능 래스터라이저**

실시간 렌더링을 위해서는 많은 Gaussian을 빠르게 화면에 투영하고 정렬해야 한다. 3DGS는 **타일 기반 rasterization**을 도입해 이 문제를 해결한다 .

1. **타일 분할:** 화면을 16×16 타일로 나누고, 각 Gaussian의 99% 신뢰 구간이 view frustum과 교차하는지 검사해 **frustum culling**을 수행한다 .
2. **키(key) 생성:** 각 Gaussian 스플랫의 view space depth와 타일 ID를 결합한 키를 생성하고, **GPU Radix sort**로 한 번에 정렬한다 . 픽셀별 정렬은 하지 않는다.
3. **타일별 alpha 블렌딩:** 정렬된 Gaussian 리스트를 타일 단위로 순회하며 alpha 블렌딩을 수행한다. 각 타일은 스레드 블록에 할당되고, pixels의 누적 alpha가 1에 가까워지면 조기 종료해 연산을 절약한다 . 이 기법은 역전파에 필요한 중간 정보를 효율적으로 저장해 빠른 backward pass를 지원한다 .

이 rasterizer는 **무제한 개수의 Gaussian이 혼합되어도** gradient를 정확히 계산할 수 있으며, 1080p 해상도에서 **30 fps 이상의 실시간 렌더링**을 달성한다 .

## **6. 알고리즘 요약**

아래는 3DGS의 전체 파이프라인을 요약한 흐름도이다.

1. **입력 준비:** 여러 시점의 사진과 카메라 포즈, SfM에서 추출한 sparse point cloud.
2. **초기 Gaussian 생성:** SfM 포인트를 Gaussian의 평균으로, 가장 가까운 세 포인트까지의 평균 거리로 스케일을 초기화하고, 회전은 단위 quaternion으로 초기화한다 .
3. **파라미터 최적화:** 렌더링과 이미지 비교를 반복하면서 위치, 스케일, 회전, SH 계수, alpha를 SGD로 업데이트한다. 손실은 L1와 D‑SSIM을 결합한 손실이다 .
4. **적응형 밀도 제어:** 100 iteration마다 alpha threshold 이하의 Gaussian을 제거하고, view space 위치 기울기에 따라 Gaussian을 **복제하거나 분할**한다 .
5. **타일 기반 rasterization:** 학습 중에도 빠르게 Gaussian을 rasterize하여 alpha 블렌딩과 gradient 계산을 수행한다 .
6. **학습 완료 후:** trained Gaussian set을 사용해 **실시간 novel view synthesis**를 수행한다. 이 때 복잡한 MLP 질의가 필요 없으므로 1080p에서도 30 fps 이상을 달성한다 .

## **7. 실험과 결과**

### **7.1 데이터셋과 설정**

저자들은 Mip‑NeRF360이 사용한 **13개 실세계 장면**과 **Synthetic NeRF (Blender)** 데이터셋에서 평가를 수행했다. SfM 포인트만으로 초기화하거나 임의 초기화를 사용하는 설정 모두를 테스트했다 .

- **Warm‑up:** 초기 1000 iteration 동안 낮은 해상도로 학습한 뒤 해상도를 증가시켰다 .
- **Evaluation metrics:** PSNR, SSIM, LPIPS를 사용했다 .

### **7.2 성능**

- **시각적 품질:** 3DGS는 Mip‑NeRF360 등 최신 NeRF 방법과 **동등하거나 더 나은 품질**을 달성했다. Synthesized scenes에서 3DGS (30K iterations)는 Mip‑NeRF와 거의 동일한 PSNR을 기록하며, InstantNGP Base나 Plenoxels보다 높은 PSNR을 얻었다 .
- **속도:** 3DGS는 **실시간 novel‑view synthesis**가 가능하다. 논문 Figure 1에서 InstantNGP가 9.2 fps, Plenoxels가 8.2 fps를 달성하는 반면, 3DGS는 **1080p에서 93~135 fps**를 기록했다 . Mip‑NeRF360은 0.071 fps에 불과하다 .
- **학습 시간:** Mip‑NeRF360은 최대 48 시간이 필요한 반면, 3DGS는 **6 분 (7K iteration 설정)** 또는 **51 분 (30K iteration 설정)** 만에 학습해 SOTA 품질을 달성했다 . InstantNGP Base와 Plenoxels는 약 7~26 분의 학습 시간이 필요하지만 PSNR이 더 낮다 .
- **메모리:** 3DGS 모델의 평균 크기는 약 3.8 MB로, InstantNGP Big의 9 MB보다 작다 .
- **Ablation:** **이방성 공분산** 최적화, **SH 계수**, **적응형 densification**, **무제한 splat의 gradient 계산**이 PSNR 향상에 기여함을 ablation study로 보여준다 .

### **7.3 장점과 비교**

| **특징** | **NeRF 기반 방법** | **3D Gaussian Splatting** |
| --- | --- | --- |
| **렌더링 속도** | MLP 쿼리가 필요해 느림; 실시간 불가 | GPU rasterization만으로 1080p에서 30 fps 이상 달성 |
| **표현** | implicit continuous radiance field | explicit 3D Gaussian 프리미티브 |
| **학습 데이터** | 다수의 이미지 + 종종 depth supervision 필요 | SfM 포인트와 카메라만 필요; MVS 깊이 없이도 초기화 가능 |
| **학습 시간** | 수 시간~수십 시간 (SOTA) | 수분~수십 분; 실시간 학습도 가능 |
| **모바일/웹 구현** | MLP 추론을 위해 GPU/Neural Engine 필요 | WebGL/AR 환경에서도 실행 가능 |
| **메모리/모델 크기** | NeRF grid 또는 hash structures로 수십 MB | Gaussian 파라미터만 저장해 수 MB 수준 |

## **8. 결론 및 전망**

**3D Gaussian Splatting**은 SfM으로 추정한 sparse points만으로 초기화한 뒤 **이방성 Gaussian 분포**를 최적화하여 장면을 묘사하는 방법이다. NeRF처럼 고품질의 novel view synthesis를 제공하면서도 MLP 쿼리를 완전히 제거하고, **tile‑based rasterization**과 **adaptive densification**을 통해 **실시간(≥30 fps) 렌더링**을 달성한다 .

저자들은 이 접근법이 **모바일/AR/VR 애플리케이션**에서 즉시 사용할 수 있을 만큼 빠르고 메모리 효율적이라는 점을 강조한다 . 향후 연구에서는 Gaussian 개수를 추가적으로 줄이거나, **dynamic scene**, **조명 변화**, **반사·산란 재질**을 모델링하는 확장, 클러스터링 기반 메모리 최적화 등이 주요 과제로 남는다. 또한 신경망 기반 표현과 Gaussian Splatting을 결합하는 하이브리드 접근법도 흥미로운 연구 방향으로 제안된다.