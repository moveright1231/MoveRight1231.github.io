---
layout: post
title: 3D Gaussian Splatting (nerfstudio, colmap)
date: 2025-10-02 11:30:00 +0800
category: experiment
thumbnail: style/image/PROJECT_REPORT/t.png
icon: code
---

# PROJECT_REPORT

# 3D Gaussian Splatting 프로젝트 보고서

---

# 프로젝트 개요

| 항목 | 내용 |
| --- | --- |
| **프로젝트명** | 3D Gaussian Splatting을 활용한 실물 객체 3D 재구성 |
| **모델(Model)** | Nerfstudio Splatfacto (3D Gaussian Splatting) |
| **데이터셋(Dataset)** | 커스텀 이미지 (인형 촬영) - 30장 |
| **환경** | WSL2 Ubuntu 24.04, RTX 3060 (12GB), CUDA 12.1, PyTorch 2.2.2 |

---

# 목적 (Objective)

### ▪ 실험의 목표

- **3D Gaussian Splatting 기술을 활용한 실물 객체의 고품질 3D 재구성**
- Multi-view 이미지로부터 학습하여 실시간 렌더링이 가능한 3D 모델 생성
- COLMAP 기반 카메라 포즈 추정과 Gaussian Splatting의 통합 워크플로우 검증

### ▪ 얻고자 하는 인사이트

- Iteration에 따른 렌더링 품질 변화 추이 관찰
- 복잡도가 다른 객체(인형 vs 배경)의 재구성 품질 차이 분석
- Downscale factor가 학습 속도 및 메모리 사용량에 미치는 영향 파악
- 실제 응용 가능성 평가 (e-commerce, 가상 전시, AR/VR 등)

---

# 💡 배경 및 아이디어 (Background & Motivation)

### ▪ 실험 동기

**3D Gaussian Splatting**은 2023년 발표된 혁신적인 3D 재구성 기술로, NeRF(Neural Radiance Fields)의 느린 렌더링 속도를 극복하고 실시간 렌더링을 가능하게 합니다.

- **참고 논문**: “3D Gaussian Splatting for Real-Time Radiance Field Rendering” (Kerbl et al., SIGGRAPH 2023)
- **핵심 아이디어**:
    - 3D 공간을 수십만 개의 3D 가우시안으로 표현
    - 각 가우시안은 위치, 색상, 크기, 회전, 불투명도를 가짐
    - Differentiable rasterization으로 빠른 학습 및 렌더링
    - NeRF 대비 1000배 이상 빠른 렌더링 속도

**Nerfstudio**의 Splatfacto 구현을 선택한 이유:
- COLMAP과의 통합으로 쉬운 데이터 준비
- 학습 중 실시간 뷰어 제공
- 최적화된 하이퍼파라미터
- PLY 포맷으로 export 가능

---

# 📦 데이터셋 (Datasets)

| 구분 | 내용 |
| --- | --- |
| **데이터셋 이름** | Custom Doll Dataset |
| **촬영 대상** | 책상 위 인형 (복잡한 텍스처와 형태) |
| **이미지 수** | 총 30장 (학습에 25장 사용) |
| **원본 해상도** | ~3000×4000 (iPhone 촬영) |
| **학습 해상도** | Downscale factor 4 적용 (750×1000) |
| **카메라 포즈 추정** | COLMAP (Structure-from-Motion) |
| **전처리** | 자동 리사이징, undistortion |
| **평가 방식** | 시각적 품질 평가, 실시간 뷰어 검증 |

### ▪ 데이터 특성

- **장점**:
    - 고해상도 원본 이미지
    - 다양한 각도에서 촬영 (360도 커버리지)
    - 충분한 오버랩 (COLMAP 재구성 성공)
- **도전 과제**:
    - 책상 표면의 반사 및 단순한 텍스처
    - 배경의 복잡도 부족 (단색 벽면)
    - 조명 변화 (창문 빛의 영향)

---

# ⚙️ 환경 (Environment)

### ▪ 하드웨어

- **GPU**: NVIDIA GeForce RTX 3060 (12GB VRAM)
- **CPU**: 8 cores
- **RAM**: 10GB (WSL2 할당)
- **Swap**: 16GB
- **Storage**: E: drive (1.9TB SSD)

### ▪ 소프트웨어

| 구성요소 | 버전 |
| --- | --- |
| OS | Ubuntu 24.04 LTS (WSL2) |
| Python | 3.9.23 |
| PyTorch | 2.2.2+cu121 |
| CUDA | 12.1 |
| Nerfstudio | 1.1.5 |
| COLMAP | Latest |
| GCC | 11.5.0 (CUDA 호환성) |

### ▪ WSL 최적화 설정

```
[wsl2]
memory=16GB          # 메모리 할당 증가
processors=8
swap=16GB
```

**환경변수**:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export DISPLAY=  # X11 문제 회피
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
```

---

# 🧠 실험 설계 (Experiment Design)

### ▪ 실험 1: 전체 장면 학습 (배경 포함)

**목적**: 인형과 배경을 포함한 전체 장면의 3D 재구성

![image.png](style/image/PROJECT_REPORT/gsply.png)

### ▪ 하이퍼파라미터 설정

| 항목 | 설정값 | 비고 |
| --- | --- | --- |
| **Max Iterations** | 30,000 | 표준 설정 |
| **Downscale Factor** | 4 | 메모리 절약 (원본 1/4 크기) |
| **Batch Size** | Auto | Nerfstudio 자동 설정 |
| **Learning Rate** | - | Adaptive (optimizer별 차등) |
| **Optimizer** | Adam | 각 속성별 다른 LR |
| **Background Color** | Black | 기본 배경 |
| **Viewer** | Enabled | 실시간 모니터링 |

### ▪ Optimizer 상세 설정

Gaussian Splatting은 각 속성별로 다른 학습률을 사용:

| 속성 | Learning Rate | Optimizer |
| --- | --- | --- |
| Means (위치) | 0.00016 | Adam |
| Features (색상) | 0.0025 | Adam |
| Opacities (불투명도) | 0.05 | Adam |
| Scales (크기) | 0.005 | Adam |
| Quats (회전) | 0.001 | Adam |
| Camera Optimization | 0.0001 → 5e-7 | Cosine Decay |

### ▪ 학습 스케줄

- **Total Steps**: 30,000
- **Checkpoint 저장**: 매 2,000 step
- **Evaluation**: 매 100 step
- **GIF 생성**: 1,000 / 7,000 / 15,000 / 30,000 step

### ▪ 비교 항목

| 측면 | 관찰 내용 |
| --- | --- |
| **수렴 속도** | Iteration에 따른 Loss 감소 |
| **렌더링 품질** | 객체별 디테일 재현도 |
| **가우시안 분포** | 공간적 밀도 분포 |
| **메모리 사용량** | GPU/RAM 점유율 |

---

# 📊 결과 및 분석 (Results & Analysis)

### ▪ 학습 결과 요약

| 지표 | 값 |
| --- | --- |
| **최종 Checkpoint** | step-000029999.ckpt (149MB) |
| **가우시안 개수** | 205,821개 |
| **PLY 파일 크기** | 49MB |
| **학습 시간** | ~30분 (RTX 3060) |
| **평균 Iteration 시간** | ~0.044초/step |

### ▪ 결과 PLY (cloudcompare)

![ply_cc.gif](style/image/PROJECT_REPORT/ply_cc.gif)

### ▪ Iteration별 렌더링 품질 변화

### 📹 2**,000 Iterations**

![1000 iterations](style/image/PROJECT_REPORT/nerf_gs.gif)

1000 iterations

**관찰**:
- 기본적인 형태와 색상이 나타나기 시작
- 가우시안이 공간에 분산되며 초기 구조 형성
- 디테일 부족, 뿌옇게 보이는 영역 다수
- 인형의 대략적인 실루엣 인식 가능

---

### 📹 **7,000 Iterations**

![7000 iterations](PROJECT_REPORT/nerf_gs2.gif)

7000 iterations

**관찰**:
- 인형의 세부 텍스처가 명확해짐
- 색상 재현도 향상
- 책상 표면의 질감이 드러나기 시작
- 배경과 전경의 분리가 뚜렷해짐
- 일부 view에서 artifact 존재

---

### 📹 **18,000 Iterations**

![15000 iterations](style/image/PROJECT_REPORT/nerf_gs3.gif)

15000 iterations

**관찰**:
- 인형의 디테일이 대부분 재현됨
- 얼굴 표정, 옷의 주름 등 세밀한 특징 포착
- 책상의 반사 특성이 부분적으로 표현됨
- 배경의 단순함으로 인한 over-smoothing 현상

---

### 📹 **30,000 Iterations (최종)**

![30000 iterations](style/image/PROJECT_REPORT/4.gif)

30000 iterations

**관찰**:
- 인형의 최종 품질: **매우 우수**
- 고주파 디테일 (텍스처, 패턴) 정확히 재현
- 다양한 각도에서 일관된 품질
- 색상 및 조명 효과 자연스러움

- 책상 및 배경의 품질: **보통**
    - 단순한 텍스처로 인한 디테일 부족
    - 반사 영역에서 가우시안 분포 불균형
    - 배경 벽면이 over-smoothed
    - 일부 시점에서 floater artifacts 관찰

---

### ▪ 객체별 품질 분석

### ✅ **잘 재구성된 부분: 인형**

**성공 요인**:
1. **풍부한 텍스처**: 몸의 형태, 얼굴 디테일 등 고주파 정보가 많음
2. **명확한 기하학적 구조**: 팔, 다리, 머리 등 distinct한 형태
3. **충분한 멀티뷰 커버리지**: 360도 모든 각도에서 촬영
4. **좋은 조명**: 인형 표면에 균일한 조명

**가우시안 분포**:
- 인형 영역에 가우시안이 밀집
- 작은 스케일의 가우시안이 디테일 표현
- RGB Spherical Harmonics로 각도별 색상 변화 정확히 모델링

---

### ⚠️ **품질이 아쉬운 부분: 책상 & 배경**

**제한 요인**:
1. **단순한 텍스처**:
- 책상 표면의 단색 영역이 많음
- 특징점(feature points) 부족으로 COLMAP 재구성 어려움

1. **반사 및 Specular 특성**:
    - 책상 표면의 반사가 view-dependent하게 변함
    - Gaussian Splatting의 Spherical Harmonics가 완벽히 표현 못함
    - 일부 각도에서 색상 inconsistency
2. **배경의 복잡도 부족**:
    - 단색 벽면은 3D 정보가 거의 없음
    - 가우시안이 부정확한 위치에 배치됨
    - Over-smoothing으로 인한 디테일 손실
3. **가우시안 분포 불균형**:
    - 인형에 비해 배경 영역의 가우시안 밀도 낮음
    - 일부 floater (공중에 떠있는 가우시안) 발생

---

### ▪ 정량적 분석

### 가우시안 속성 분포

| 속성 | 통계 |
| --- | --- |
| **총 가우시안 수** | 205,821개 |
| **위치 (xyz)** | 3D 공간에 분포 |
| **색상 (SH)** | DC + 26차 계수 (고급 색상 표현) |
| **불투명도** | 평균: ~0.7 (인형), ~0.3 (배경) |
| **스케일** | 인형: 작음 (세밀), 배경: 큼 (단순) |

### GPU 사용률

| 단계 | GPU 사용률 | GPU 메모리 | 상태 |
| --- | --- | --- | --- |
| **이미지 전처리** | 10-20% | ~1GB | CPU 집중 |
| **학습 중** | 80-95% | ~3-4GB | GPU 집중 |
| **뷰어 로딩** | 30-50% | ~1.5GB | 렌더링 |

---

# 🔍 인사이트 및 결론 (Insights & Conclusion)

### ▪ 핵심 발견

1. **3D Gaussian Splatting은 텍스처가 풍부한 객체 재구성에 매우 효과적**
    - 인형과 같은 복잡한 객체는 고품질로 재현
    - 30,000 iterations로 상업적 품질 달성
2. **단순한 텍스처 영역은 재구성 품질이 제한적**
    - 책상, 배경 등 feature가 적은 영역은 over-smoothing
    - COLMAP 단계에서 이미 재구성 정확도 낮음
3. **Downscale factor는 메모리-품질 trade-off의 핵심**
    - Factor 4로도 충분한 품질 (WSL 환경에서 안정적)
    - 메모리 제약이 없다면 factor 2 권장
4. **실시간 뷰어의 가치**
    - 학습 중 품질을 즉시 확인 가능
    - 조기 중단 또는 하이퍼파라미터 조정 판단에 유용

### ▪ 기술적 통찰

**가우시안 분포의 자기 조직화**:
- 학습 초기: 균일하게 분포된 가우시안
- 학습 중기: 특징이 많은 영역에 densification
- 학습 후기: 불필요한 가우시안 pruning

**Spherical Harmonics의 한계**:
- View-dependent effect (반사, 투명도) 표현 제한
- 26차 SH로도 복잡한 BRDF 완벽히 모델링 불가
- 향후 Neural Shader 등으로 개선 가능

### ▪ 결론

> 3D Gaussian Splatting은 복잡한 형태와 텍스처를 가진 객체의 3D 재구성에 탁월하며, WSL 환경에서도 안정적으로 학습 가능함을 확인하였다. 다만, 단순 텍스처 영역의 품질 향상을 위해서는 추가적인 데이터 수집 전략 또는 마스크 기반 학습이 필요하다.
> 

---

# 🚀 추가 실험 / 개선 방향 (Further Work)

### ▪ 진행 중인 실험

### 1️⃣ **마스크 기반 학습 (객체 분리)**

**목적**: 인형만 집중 학습하여 배경의 노이즈 제거

**방법**:
- Rembg를 활용한 자동 배경 제거
- 마스크 30개 생성 완료 (`masks_rembg/`)
- `--masks-path` 옵션으로 학습 진행 중

**기대 효과**:
- 인형 영역에 가우시안 집중 배치
- 배경을 흰색 또는 투명하게 처리
- PLY 파일 크기 감소 (인형만 포함)
- 다른 배경에 합성 가능

---

### ▪ 향후 개선 방향

### 2️⃣ **데이터 수집 개선**

- **배경 텍스처 추가**: 책상에 패턴 있는 천 깔기
- **조명 통제**: 균일한 조명 환경 구성
- **이미지 수 증가**: 30장 → 50-100장으로 밀집 촬영
- **다양한 객체**: 다른 형태/재질의 객체로 실험

### 3️⃣ **하이퍼파라미터 튜닝**

| 파라미터 | 현재 | 개선안 | 기대 효과 |
| --- | --- | --- | --- |
| Downscale Factor | 4 | 2 | 더 높은 해상도 |
| Iterations | 30,000 | 50,000 | 미세 디테일 개선 |
| Densification | Auto | Manual control | 배경 품질 향상 |

### 4️⃣ **Advanced Techniques**

- **Depth Supervision**: Depth 센서 데이터 추가
- **Appearance Embedding**: 조명 변화 모델링
- **Anti-aliasing**: Mip-Splatting 적용
- **Dynamic Scene**: 움직이는 객체 재구성

### 5️⃣ **응용 연구**

- **E-commerce**: 제품 360도 뷰어 개발
- **AR/VR**: Unity/Unreal Engine 통합
- **Digital Twin**: 실제 공간 가상화
- **Animation**: 가우시안 애니메이션 기법

---

# 📁 코드 및 리소스 (Appendix)

### ▪ 디렉토리 구조

```
gs-project/
├── images_resized/              # 원본 이미지 (30장)
├── nerfstudio_data/             # COLMAP 재구성 결과
│   ├── colmap/                  # 카메라 파라미터
│   ├── images -> ../images_resized
│   ├── images_2/                # Downscale 2
│   └── images_4/                # Downscale 4 (사용)
├── masks_rembg/                 # Rembg 마스크 (30장)
├── masked_images/               # 배경 제거 이미지
├── mask_visualizations_rembg/   # 마스크 시각화
├── outputs/                     # 학습 결과
│   └── nerfstudio_data/splatfacto/2026-01-29_151204/
│       ├── config.yml           # 학습 설정
│       ├── nerfstudio_models/
│       │   └── step-000029999.ckpt  # 최종 체크포인트 (149MB)
│       └── dataparser_transforms.json
├── exports/                     # Export 결과
│   └── splat/
│       └── splat.ply            # 가우시안 PLY (49MB, 205,821개)
├── run_gs.sh                    # 전체 학습 스크립트
├── run_gs_object_only.sh        # 마스크 학습 스크립트
├── view_result.sh               # 결과 뷰어
└── export_ply.sh                # PLY export
```

### ▪ 주요 스크립트

### `run_gs.sh` - 메인 학습 스크립트

```bash
#!/bin/bash
# 최적화된 설정:
# - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# - DISPLAY= (X11 비활성화)
# - Downscale factor 4
# - Background 실행 (nohup)

ns-train splatfacto \
  --data nerfstudio_data \
  --max-num-iterations 30000 \
  --viewer.quit-on-train-completion True \
  colmap \
  --downscale-factor 4
```

### `view_result.sh` - 결과 확인

```bash
ns-viewer --load-config \
  outputs/nerfstudio_data/splatfacto/2026-01-29_151204/config.yml
```

### `export_ply.sh` - PLY 파일 생성

```bash
ns-export gaussian-splat \
  --load-config outputs/.../config.yml \
  --output-dir exports/
```

### ▪ 환경 재현

```bash
# WSL2 Ubuntu 24.04
# 1. Miniforge 설치
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# 2. 환경 생성
conda create -n tracking python=3.9
conda activate tracking

# 3. PyTorch 설치
pip install torch==2.2.2 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Nerfstudio 설치
pip install nerfstudio

# 5. GCC-11 설치
sudo apt install gcc-11 g++-11

# 6. Rembg 설치 (마스크 생성용)
pip install rembg[gpu]
```

### ▪ 참고 자료

**논문**:
- Kerbl et al., “3D Gaussian Splatting for Real-Time Radiance Field Rendering”, SIGGRAPH 2023
- Schönberger et al., “Structure-from-Motion Revisited”, CVPR 2016 (COLMAP)

**웹 리소스**:
- [Nerfstudio 공식 문서](https://docs.nerf.studio/)
- [3D Gaussian Splatting GitHub](https://github.com/graphdeco-inria/gaussian-splatting)
- [SuperSplat 뷰어](https://playcanvas.com/supersplat)

**시각화 도구**:
- Nerfstudio Viewer (http://localhost:7007)
- SuperSplat (웹 기반 PLY 뷰어)
- CloudCompare (3D 시각화)

---

# 📸 결과 샘플

### ▪ 최종 렌더링 결과

| 시점 | 품질 | 비고 |
| --- | --- | --- |
| 정면 | ★★★★★ | 인형 디테일 완벽 |
| 측면 | ★★★★★ | 옷의 주름 재현 우수 |
| 상단 | ★★★☆☆ | 책상 반사 아쉬움 |
| 배경 | ★★☆☆☆ | Over-smoothing |

### ▪ PLY 파일 정보

**헤더 정보**:

```
ply
format binary_little_endian 1.0
comment Generated by Nerstudio 1.1.5
element vertex 205821
property float x, y, z          # 위치
property float nx, ny, nz       # 법선
property float f_dc_0~2         # 기본 색상
property float f_rest_0~25      # SH 계수 (각도별 색상)
property float opacity          # 불투명도
property float scale_0~2        # 크기 (xyz)
property float rot_0~3          # 회전 (quaternion)
```

---

# 🎓 학습 내용 및 소감

### ▪ 기술적 학습

1. **3D 재구성 파이프라인 이해**
    - COLMAP SfM → Gaussian Splatting 학습 → Rendering
    - 각 단계의 역할과 한계점 파악
2. **WSL 환경에서의 GPU 학습**
    - 메모리 관리의 중요성
    - CUDA 호환성 이슈 해결 경험
3. **하이퍼파라미터의 영향**
    - Downscale factor가 메모리/품질에 미치는 영향
    - Iteration 수에 따른 수렴 양상

### ▪ 개선 포인트

- **데이터 수집 단계부터 3D 재구성을 고려한 촬영 필요**
- **텍스처가 풍부한 환경 구성이 품질 향상의 핵심**
- **실시간 모니터링으로 조기에 문제 발견 가능**

### ▪ 향후 계획

마스크 기반 학습을 완료하여 배경 없는 깨끗한 3D 모델을 생성하고, 이를 웹 뷰어에 임베드하여 인터랙티브한 3D 갤러리를 구축할 예정.

---
