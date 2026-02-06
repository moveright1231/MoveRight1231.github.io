---
layout: post
title: Global Structure-from-Motion 논문리뷰
date: 2025-03-04 11:30:00 +0800
category: paper
thumbnail: style/image/blog_colmap&glomap/1.png
icon: code
---


# Structure-from-Motion (SfM) 리뷰 — COLMAP & GLOMAP 기반 정리

## 🧠 1. SfM 개요

- *Structure-from-Motion (SfM)**은 여러 장의 2D 이미지로부터 **카메라 포즈(Camera pose)**와 **3차원 구조(3D geometry)**를 동시에 복원하는 컴퓨터 비전 문제다. SfM은 epipolar geometry, feature correspondence, triangulation, bundle adjustment 등의 기하학적 최적화 기반으로 구성된다. [oai_citation:0‡위키백과](https://en.wikipedia.org/wiki/Structure_from_motion?utm_source=chatgpt.com)

일반적인 SfM 파이프라인은 다음과 같다:

1. **Correspondence Search**
    
    입력 이미지에서 feature를 추출하고 매칭함.
    
2. **Epipolar Geometry & Verification**
    
    두 뷰 간 geometry를 RANSAC으로 검증해 inlier인 매칭만 필터링함.
    
3. **Pose Estimation & Triangulation**
    
    두 뷰 또는 다수 뷰로부터 카메라 포즈를 추정하고 3D 점을 생성함.
    
4. **Bundle Adjustment**
    
    전체 카메라 포즈와 3D 점을 joint로 최적화함.
    

---

## 🏗 2. COLMAP: Structure-from-Motion Revisited (CVPR 2016)

**COLMAP**은 *Structure-from-Motion Revisited* 논문으로 정립된 범용 SfM 구현이다. SfM 알고리즘 그 자체뿐 아니라, 실제 엔드-투-엔드 3D 재구성 파이프라인으로 설계되었다.  [oai_citation:1‡벡터로그](https://velog.io/%40shj4901/Structure-from-Motion-Revisited?utm_source=chatgpt.com)

![image.png](blog_colmap&glomap/1.png)

**그림 1. GLOMAP 재구성 예시**

(a) 예시 재구성:

왼쪽 상단과 오른쪽 상단 이미지는 건물과 야외 공간을 3D 포인트 클라우드로 재구성한 모습임. 붉은색 선은 카메라의 이동 경로를 나타냄.
왼쪽 하단과 오른쪽 하단 이미지는 또 다른 건물들을 3D 포인트 클라우드로 재구성한 결과임. 이 역시 붉은색 선으로 카메라의 움직임을 보여줌.

(b) LaMAR [60] LIN 재구성:

이 부분은 LaMAR 데이터셋의 LIN 시나리오에 대한 여러 시스템의 재구성 결과를 비교함.
맵 위에 붉은색 선으로 카메라 궤적이 표시되어 있으며, 이를 통해 각 시스템이 얼마나 정확하게 카메라의 움직임을 추적하고 3D 환경을 재구성했는지 시각적으로 확인할 수 있음.
본 논문의 GLOMAP 시스템은 이 이미지에서 이전의 다른 방법들(Theia [71], COLMAP [62] 등)에 비해 더 정확하고 안정적인 재구성 결과를 보여준다고 주장함. (참고: 제공된 이미지에는 Theia, COLMAP, GLOMAP의 결과가 명확히 구분되어 표시되지 않았으나, 논문의 설명에 따르면 GLOMAP이 더 나은 성능을 보인다고 함.)

### 🧩 특징

- **Incremental Reconstruction**
    
    초기 두 이미지로 시작하여 점진적으로 이미지와 3D 포인트를 확장함.
    
- **Scene Graph Augmentation**
    
    view graph에서 homography / essential / fundamental inliers 기반으로 안정적인 연결을 선정함.  [oai_citation:2‡xoft](https://xoft.tistory.com/118?utm_source=chatgpt.com)
    
- **Next Best View Selection**
    
    재구성 확장 시 **좋은 baseline**과 **많은 관측 포인트**를 가진 시점을 우선함.
    
- **Robust Triangulation**
    
    RANSAC-based sampling으로 robust한 3D 포인트 추정과 **positive depth** 검증을 포함함.  [oai_citation:3‡xoft](https://xoft.tistory.com/118?utm_source=chatgpt.com)
    
- **Bundle Adjustment**
    
    local 및 global BA를 반복적으로 수행하여 전체 구조를 최적화함.  [oai_citation:4‡xoft](https://xoft.tistory.com/118?utm_source=chatgpt.com)
    

### ▶ 특징적 구현 세부

- RANSAC 기반 geometric verification
- Scene graph를 이용한 correspondence filtering
- Sparse direct solver + iterative solver 결합 최적화

COLMAP은 incremental SfM의 **정확도와 robustness**를 유지하면서 다양한 이미지 세트에 적용 가능하도록 설계되었다.  [oai_citation:5‡벡터로그](https://velog.io/%40shj4901/Structure-from-Motion-Revisited?utm_source=chatgpt.com)

---

## 🌐 3. Global SfM 과 *Global Structure-from-Motion Revisited* (GLOMAP, ECCV 2024)

기존의 SfM 접근 방식은 크게 두 가지로 나뉜다:

| 방식 | 특징 |
| --- | --- |
| **Incremental SfM** | 정확하고 견고하지만 느리고, bundle adjustment 반복이 많음 |
| **Global SfM** | 모든 카메라를 동시에 추정하므로 빠르고 확장성이 높음 |

### 🧠 GLOMAP 논문 개요

*Global Structure-from-Motion Revisited*는 global SfM 접근 자체를 **다시 정의한 실용적인 SfM 파이프라인**으로, incremental 방식 수준의 정확도와 robustness를 유지하면서 **현저하게 빠른 성능**을 제공한다.  [oai_citation:7‡arXiv](https://arxiv.org/abs/2407.20219?utm_source=chatgpt.com)

**핵심 차이점**은 global SfM에서 흔히 사용하는 *translation averaging*을 **배제하고**, 카메라 위치와 3D 포인트 위치를 **joint로 최적화**하는 새로운 글로벌 포지셔닝 전략에 있다.

![image.png](blog_colmap&glomap/image2.png)

**그림 2. GLOMAP 파이프라인 개요**

위 그림은 GLOMAP 시스템의 전체 파이프라인을 보여줌. SfM 문제를 해결하기 위한 주요 단계를 네 개의 상자로 나누어 설명함.

### 📥 Input Images (입력 이미지)

3D 구조 복원에 사용될 일련의 입력 이미지들을 나타냄. 건물이나 랜드마크를 다양한 각도에서 촬영한 사진들이 예시로 제시됨.

### 🔍 Correspondence Search (대응 검색)

입력 이미지들 간의 특징점을 찾고 관계를 설정하는 과정임.

- **Feature Extraction (특징 추출)**: 각 이미지에서 독특하고 식별 가능한 특징점(코너, 엣지 등)을 감지하고 추출함
- **Matching (매칭)**: 추출된 특징점들 중 이미지 간에 동일한 3D 점에 해당하는 특징점들을 찾아 연결함
- **Two-view Estimation (2-뷰 추정)**: 매칭된 특징점 쌍으로 두 카메라 간의 상대적 기하학적 관계(Fundamental Matrix, Essential Matrix)를 추정함
- **View Graph Calibration (뷰 그래프 보정)**: 2-뷰 추정 결과를 바탕으로 이미지 간의 연결성과 기하학적 제약을 나타내는 뷰 그래프를 구축하고 보정함
- **Relative Pose Decomposition (상대 포즈 분해)**: 뷰 그래프의 엣지로부터 각 카메라 쌍에 대한 상대적 회전(Rotation)과 이동(Translation) 값을 계산함

### 🌐 Global Estimation (전역 추정)

전체 이미지 집합에 대한 카메라들의 전역적 위치와 3D 구조를 동시에 복구하는 핵심 단계임.

- **Rotation Averaging (회전 평균)**: 모든 카메라의 절대적 회전(Orientation)을 뷰 그래프의 상대적 회전 정보들을 종합하여 추정함
- **Global Positioning (전역 포지셔닝)**: 본 논문의 핵심 기여로, 별도의 변환 평균화와 삼각측량 단계를 통합하여 카메라의 전역적 위치(Translation)와 3D 점의 위치를 동시에 추정함. 이 방법은 알려지지 않은 카메라 내부 파라미터나 거의 공선형인 움직임에도 강건함
- **Bundle Adjustment (번들 조정)**: 추정된 카메라 포즈와 3D 구조로 재투영 오차(Reprojection Error)를 최소화하여 전체 재구성의 정확도를 향상시킴
- **Structure Refinement (구조 개선)**: 번들 조정 이후 3D 구조를 더욱 정밀하게 만들기 위해 포인트를 재삼각측량하거나 추가적인 최적화를 수행함

### 📤 Output Reconstruction (출력 재구성)

최종적으로 복구된 3D 구조와 카메라 포즈를 나타냄. 점 구름(Point Cloud) 형태로 3D 모델이 생성됨.

이 파이프라인은 이미지에서 3D 장면을 복원하는 과정을 단계별로 보여줌. 특히 GLOMAP이 기존 글로벌 SfM 방식과 달리 'Global Positioning' 단계에서 카메라 위치와 3D 구조 추정을 통합하여 효율성과 정확성을 높였다는 점이 핵심임.

### 📌 GLOMAP 주요 기여

### 1) Global Positioning

과거 global SfM은 rotation averaging → translation averaging → triangulation을 분리하여 수행했다.

하지만 이 방식은 translation averaging 단계에서 다음과 같은 문제가 있었다:

- **스케일 모호성**
- **카메라 intrinsic 정보가 없을 때 오류 증가**
- **co-linear motion에서의 추정 불안정성**

**GLOMAP은 이 문제를 해결하기 위해**:

- 카메라 위치와 3D 포인트를 **joint optimization**함으로써 translation averaging의 불안정성을 제거했다.

![image.png](blog_colmap&glomap/image3.png)

**왼쪽 이미지 (초기 상태)**

- 여러 대의 카메라(녹색 사각형)와 씬 내의 3D 포인트들(색깔 있는 원)이 임의로 초기화된 상태임
- 점선은 카메라와 3D 포인트 간의 관측 연결을 나타냄
- 아직 카메라 위치도, 3D 포인트 위치도 정확하게 정해지지 않은 상태임

**오른쪽 이미지 (글로벌 포지셔닝 후)**

- 카메라의 위치(c_i)와 방향(회전)이 정확히 결정됨
- 3D 포인트(X_k)의 정확한 위치가 추정됨
- 각 카메라에서 특정 3D 포인트를 볼 때 생성되는 이미지 상의 광선(v_ik)과 3D 포인트에서 카메라 중심을 잇는 벡터(X_k - c_i) 사이의 각도(θ)를 최소화하는 방식으로 최적화가 이루어짐

**수식 설명**

- d_ik: 정규화 계수(normalizing factor)임
- v_ik: 이미지 상에서 포인트 X_k를 관찰하는 카메라 광선임
- sin θ: 두 벡터 간의 각도를 나타내는 항으로, 이 각도를 줄이는 게 목표임

**핵심 포인트**

기존 글로벌 SfM은 '변환 평균화'와 '삼각측량' 단계를 분리했음. 하지만 GLOMAP은 카메라 위치와 3D 포인트 위치를 **한 번에 공동으로 추정**하는 방식을 사용함. 이게 바로 GLOMAP이 빠르면서도 정확한 이유임.

### 2) Correspondence Search

COLMAP의 correspondence 검색 모듈을 기반으로 RootSIFT와 bag-of-words를 이용하여 robust한 매칭을 구성한다.  

### 3) Global Refinement

joint estimation 후 global bundle adjustment를 수행하여 구조와 포즈를 동시에 정제한다.  [oai_citation:12‡Linfei's world](https://lpanaf.github.io/eccv24_glomap/?utm_source=chatgpt.com)

### 🧪 성능 및 결과

- GLOMAP은 **global SfM baseline보다 우수**한 성능을 보이며,
- **COLMAP과 비슷하거나 더 우수한 정확성**을 유지하면서 **수배 빠른 처리 속도**를 달성했다. [oai_citation:13‡Linfei's world](https://lpanaf.github.io/eccv24_glomap/?utm_source=chatgpt.com)

### 🔗 구현 및 오픈소스

GLOMAP은 COLMAP DB를 입력으로 받아 **sparse reconstruction** 결과를 출력하는 global SfM 라이브러리로 공개되어 있다.  [oai_citation:14‡GitHub](https://github.com/colmap/glomap?utm_source=chatgpt.com)

---

## 📊 4. COLMAP vs GLOMAP 비교

| 항목 | COLMAP (Incremental SfM) | GLOMAP (Global SfM) |
| --- | --- | --- |
| Pose Estimation 방식 | 순차적, incremental | 전체 동시추정, global |
| 정확도 | 매우 높음 | ≈ COLMAP 또는 상회 |
| Robustness | 매우 높음 | 증분방법에 근접 |
| 속도 | 느림 | 빠름 |
| 확장성 | 제한적 | 우수 |
| Translation Averaging | 사용 | joint optimization으로 대체 |

---

## 🚀 5. 정리

- **SfM은 2D 이미지로부터 3D 구조를 복원하는 핵심 CV 분야 문제**이며, geometric verification, triangulation, pose estimation, bundle adjustment를 중심으로 구성된다. [oai_citation:16‡위키백과](https://en.wikipedia.org/wiki/Structure_from_motion?utm_source=chatgpt.com)
- **COLMAP 논문**은 incremental SfM의 범용적이고 실용적인 구현으로 SfM 커뮤니티에서 표준으로 자리매김했다. [oai_citation:17‡벡터로그](https://velog.io/%40shj4901/Structure-from-Motion-Revisited?utm_source=chatgpt.com)
- **GLOMAP은 global SfM을 재정의한 논문**으로, 기존 global SfM의 한계를 극복하고 incremental 수준의 성능을 global pipeline에서도 달성했다. [oai_citation:18‡arXiv](https://arxiv.org/abs/2407.20219?utm_source=chatgpt.com)

---

## 📎 참고 논문

- Schönberger & Frahm, *Structure-from-Motion Revisited*, CVPR 2016. [oai_citation:19‡벡터로그](https://velog.io/%40shj4901/Structure-from-Motion-Revisited?utm_source=chatgpt.com)
- Pan et al., *Global Structure-from-Motion Revisited (GLOMAP)*, ECCV 2024. [oai_citation:20‡arXiv](https://arxiv.org/abs/2407.20219?utm_source=chatgpt.com)