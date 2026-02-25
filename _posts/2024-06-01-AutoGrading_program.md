---
layout: post
title: (캡스톤) CBT 시험지 자동 체점
date: 2024-06-01 18:30:00 +0800
category: project
thumbnail: /style/image/autograding/4.png
icon: code
---

# AutoGrading Program

# 자동 채점 시스템 프로젝트 보고서

## 1. 프로젝트 개요

본 프로젝트는 컴퓨터 비전과 딥러닝 기술을 활용하여 객관식 시험지를 자동으로 채점하는 시스템을 개발하는 것을 목표로 정보처리기사(정처기) CBT 시험지 이미지를 입력받아, 문제를 자동으로 검출하고 수험생이 선택한 답안을 인식한 뒤, 정답 키와 비교하여 점수를 산출한다.

- **개발 언어**: Python 3.10
- **핵심 기술**: YOLOv8 (객체 탐지), OCR (문자 인식), Kivy (모바일 UI)
- **배포 환경**: Android (Buildozer)

---

## 2. 데이터셋 구축

### 2.1 데이터 수집 및 구성

![스크린샷 2026-02-26 오전 1.27.53.png](style/image/autograding/1.png)

시험지에서 객관식 항목의 체크 표시 유무를 학습시키기 위해 직접 데이터셋을 구축하였다. 시험지 이미지를 촬영하거나 스캔하여 각 보기 항목(원형 마킹 영역)에 대해 다음 두 가지 클래스로 레이블링하였다.

| 클래스 | 설명 |
| --- | --- |
| `choice` | 수험생이 선택(체크)한 항목 |
| `notchoice` | 선택되지 않은 항목 |

문제 영역 검출을 위한 별도 데이터셋도 함께 구축하였다.

| 클래스 | 설명 |
| --- | --- |
| `question` | 시험지 내 개별 문제 영역 |

### 2.2 어노테이션 도구 및 포맷

- **어노테이션 플랫폼**: Roboflow
- **어노테이션 포맷**: Pascal VOC (XML)
- **이미지 전처리**: 자동 방향 보정(Auto-orient), 640×640 리사이즈
- **데이터 분할**: train / valid / test
- **총 레이블링 이미지 수**: 약 20장

```
image_sagementation.v1i.voc/
├── train/   ← 학습 이미지 + XML 어노테이션
├── valid/   ← 검증 데이터
└── test/    ← 테스트 데이터
```

---

## 3. 모델 학습

### 3.1 학습 환경

Google Colab의 GPU 환경에서 Ultralytics YOLOv8 프레임워크를 사용하여 두 가지 모델을 개별적으로 학습하였다.

### 3.2 모델 1 — 문제 영역 검출 모델

- **목적**: 시험지 전체 이미지에서 개별 문제(question) 영역의 바운딩 박스를 검출
- **학습 클래스**: `question`
- **입력 이미지 크기**: 1216×1216
- **최종 산출물**:
    - `last(5).pt` (최신 버전, 실제 추론에 사용)
    - `best.pt` (검증 손실 기준 최적 가중치)
    - `last.onnx` (ONNX 변환 버전, 43MB)
- **시행착오**: 여러 버전(v1~v5)에 걸쳐 반복 학습하며 성능을 개선하였고, 현재 가장 안정적인 v5 버전(`last(5).pt`)을 채택하였다. 문제 영역이 길 경우 잘리는 문제가 발생하여 데이터셋 보강이 필요한 것으로 파악되었다.

### 3.3 모델 2 — 선택지 체크 검출 모델

- **목적**: 문제 이미지 내에서 체크된 항목과 체크되지 않은 항목을 검출
- **학습 클래스**: `choice`, `notchoice`
- **최종 산출물**: `choice(1).pt`

---

## 4. 시스템 파이프라인

전체 채점 과정은 3단계 파이프라인으로 구성된다.

```
[입력] 시험지 JPG 이미지
         ↓
  ┌─────────────────────────────┐
  │  Stage 1: 문제 영역 검출    │  ← YOLO (last (5).pt)
  │  'question' 바운딩박스 추출 │
  └──────────────┬──────────────┘
                 ↓ 문제별 크롭 이미지
  ┌─────────────────────────────┐
  │  Stage 2: 문제 번호 인식    │  ← Tesseract OCR
  │  파일명을 문제 번호로 변환  │
  └──────────────┬──────────────┘
                 ↓ 번호가 붙은 문제 이미지
  ┌─────────────────────────────┐
  │  Stage 3: 선택지 인식 및    │  ← YOLO (choice(1).pt)
  │  정답 비교 (채점)           │  ← 좌표 기반 알고리즘
  └──────────────┬──────────────┘
                 ↓
[출력] 오답 이미지 + 최종 점수 (X / 100점)
```

---

## 5. 선택 번호 판별 알고리즘

### 5.1 문제의 배경

YOLO 모델이 `choice`(선택한 항목)와 `notchoice`(선택하지 않은 항목)의 바운딩 박스를 각각 검출한 이후, 이 좌표 정보만으로 수험생이 몇 번 보기를 골랐는지 역산해야 했다. 이 과정에서 다양한 시행착오를 겪었다.

### 5.2 시험지 유형 구분

시험지마다 객관식 보기의 배열 방향이 다르다는 것을 발견하였다. 검출된 모든 박스(choice + notchoice)를 포함하는 전체 영역의 가로/세로 비율로 유형을 자동 구분한다.

![image.png](style/image/autograding/2.png)

![image.png](style/image/autograding/3.png)

```python
rectangle_width = x2 - x1
rectangle_height = y2 - y1

if rectangle_width > rectangle_height:
    shape_type = 'A'  # 가로형: 보기가 좌우로 나열됨
else:
    shape_type = 'B'  # 세로형: 보기가 위아래로 나열됨
```

| 유형 | 보기 배열 방향 | 판별 기준 축 |
| --- | --- | --- |
| A형 (가로형) | ① ② ③ ④ 가 좌→우 순서 | X 좌표 |
| B형 (세로형) | ① ② ③ ④ 가 위→아래 순서 | Y 좌표 |

### 5.3 A형 (가로형) 판별 로직

X좌표 기준으로 `notchoice` 박스들을 정렬한 뒤, `choice` 박스의 X좌표와 비교하여 몇 번째 위치인지 파악한다.

```python
sorted_notchoices = sorted(notchoice_boxes, key=lambda box: (box[0] + box[2]) / 2)

choice_x = (sorted_choices[0][0] + sorted_choices[0][2]) / 2
choice_y = (sorted_choices[0][1] + sorted_choices[0][3]) / 2

for idx, notchoice in enumerate(sorted_notchoices):
    notchoice_x = (notchoice[0] + notchoice[2]) / 2

    if choice_y < notchoice_y + error_range:  # 같은 행에 있을 경우
        if choice_x < notchoice_x:
            tmp2 = 1   # choice가 첫 번째 notchoice보다 왼쪽
        elif choice_x > notchoice_x:
            tmp2 = 2
    else:              # choice가 아래 행에 있을 경우
        if choice_x < notchoice_x:
            tmp2 = 3
        elif choice_x > notchoice_x:
            tmp2 = 4
```

- `error_range = 10` (픽셀): 좌표 오차 허용 범위

### 5.4 B형 (세로형) 판별 로직

Y좌표 기준으로 `choice`가 `notchoice`보다 아래에 위치할수록 더 높은 번호(2, 3, 4번)임을 이용한다.

```python
if choice[1] > notchoice[1]:
    tmp += 1  # 더 아래에 있을수록 번호가 증가
    if tmp >= 5:
        tmp = 4
```

### 5.5 시행착오 과정

- 처음에는 바운딩 박스의 절대 좌표만으로 번호를 매기려 했으나, 시험지 촬영 각도나 스캔 품질에 따라 좌표가 달라지는 문제가 있었다.
- 이를 해결하기 위해 `notchoice` 박스들과의 **상대적 위치 관계**를 이용하는 방식으로 전환하였다.
- 가로형/세로형 구분 없이 단일 로직으로 처리하려다 오답이 많이 발생하여, 유형을 자동 감지하는 분기 처리를 추가하였다.
- 픽셀 단위의 좌표 오차를 흡수하기 위해 `error_range` 파라미터를 도입하였다.

---

## 6. 정답 처리

### 6.1 내장 정답 키

자주 사용하는 시험 회차의 정답을 코드 내에 배열로 하드코딩하였다.

| 회차 | 배열 변수 | 문항 수 |
| --- | --- | --- |
| 2020년 8월 22일 | `answer1` | 100문항 |
| 2020년 6월 6일 | `answer` | 90문항 |

### 6.2 외부 정답 파일 지원

직접 정답 이미지를 업로드하면 Google Cloud Vision API의 OCR로 정답 번호를 추출하여 채점에 활용한다.

```python
def detect_text(path):
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    response = client.text_detection(image=image)
    return text_list
```

---

## 7. 모바일 앱 구성 (Kivy)

Android 배포를 위해 Kivy 프레임워크로 UI를 구성하였다.

| 화면 | 주요 기능 |
| --- | --- |
| **MainScreen** | 프로그램 초기화(Reset), 시험지 업로드, 정답 키 선택, 채점 시작, 결과 확인 |
| **FileChooserScreen** | 기기 내 시험지 이미지 파일 탐색 및 선택 |
| **ResultsScreen** | 오답 문제 이미지 그리드 표시, 최종 점수(`X / 100점 맞았습니다`) 출력 |

Android APK 빌드는 Buildozer를 통해 수행하며, arm64-v8a / armeabi-v7a 아키텍처를 지원한다.

### 7.2 프로그램 시행

![image.png](style/image/autograding/4.png)

메인화면

- file choice : 시험지를 선택 및 추가
- Choose Answer key : 정답지를 선택 및 추가
- grading start : 체점 시작
- result : 체점 결과 및 오답

![스크린샷 2026-02-26 오전 1.23.01.png](style/image/autograding/5.png)

V 표시로 푼 시험지를 입력

![스크린샷 2026-02-26 오전 1.23.41.png](style/image/autograding/6.png)

등록된 정답을 사용하거나 추가

![스크린샷 2026-02-26 오전 1.26.55.png](style/image/autograding/7.png)

체점이 완료되면 result 창에서 점수와 오답들을 확인

---

## 8. 기술 스택 요약

| 구분 | 기술 |
| --- | --- |
| 객체 탐지 | YOLOv8 (Ultralytics) |
| 로컬 OCR | Tesseract (pytesseract) |
| 클라우드 OCR | Google Cloud Vision API |
| 이미지 처리 | OpenCV, Pillow |
| 딥러닝 백엔드 | PyTorch, TensorFlow |
| 모바일 UI | Kivy |
| 학습 환경 | Google Colab (GPU) |
| 빌드/배포 | Buildozer (Android APK) |

---

## 9. 향후 개선 사항

- 문제 영역이 긴 경우 잘리는 문제 → 데이터셋 보강 및 재학습 필요
- 파일 경로 하드코딩 → 상대 경로 또는 설정 파일 기반으로 개선
- A형/B형 외 다양한 시험지 레이아웃 대응
- 문제 번호 OCR 정확도 향상 (누락된 번호는 현재 이전 이미지로 대체 처리)
- Google Cloud Vision API 인증 키를 환경 변수로 분리하여 보안 강화