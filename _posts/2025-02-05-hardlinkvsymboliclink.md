---
layout: post
title: "하드 링크 vs 심볼릭 링크 — YOLO 학습 환경에서 생긴 경로 문제와 해결"
date: 2025-02-05 21:00:00 +0800
category: experiment
thumbnail: /style/image/thumbnail.png
icon: code
---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 문제 상황 | YOLO 학습 시 심볼릭 링크로 구성한 라벨 경로를 찾지 못하는 오류 |
| 해결 방법 | 심볼릭 링크 → 하드 링크로 전환 |
| 환경 | Windows 11 + WSL2 Ubuntu, Python 3.10 |
| 핵심 개념 | inode, 심볼릭 링크, 하드 링크, Path.resolve() |

---

## 💡 배경 — 왜 이 문제가 생겼나

SAM2 auto labeling 파이프라인을 구성하면서
기존 `labels/` 와 새로 생성한 `labels_seg/` 를 함께 관리해야 하는 상황이 생겼다.

디스크 공간을 아끼고 싶어서 `labels_seg/` 를 심볼릭 링크로 연결했는데,
YOLO 학습 시 라벨을 찾지 못하는 오류가 발생했다.

이 포스트는 그 원인을 파악하는 과정에서 공부한
**하드 링크와 심볼릭 링크의 차이**, 그리고 **YOLO 내부 경로 처리 방식**을 정리한 글이다.

---

## 🧠 개념 정리 — inode부터 이해하기

### inode란?

리눅스 파일 시스템에서 파일은 사실 두 가지로 구성된다.

- **inode**: 파일의 실제 데이터와 메타정보 (크기, 권한, 데이터 블록 위치 등)
- **파일명**: inode를 가리키는 이름, 즉 포인터

```
우리가 "파일"이라고 부르는 것
= 파일명(포인터) + inode(실제 데이터)
```

이 구조를 이해하면 하드 링크와 심볼릭 링크의 차이가 명확해진다.

---

### 심볼릭 링크 (Symbolic Link)

```
inode #1234 (실제 데이터)
    ↑
원본.txt          ← 원본 파일명, inode를 직접 가리킴

inode #9999 (내용: "원본.txt 경로")
    ↑
symlink.txt       ← 심볼릭 링크, "저 경로로 가세요"라는 포인터
```

심볼릭 링크는 **별도의 inode를 가진 독립적인 파일**이다.
내용물이 원본 파일의 경로 문자열일 뿐이다.

쉽게 비유하면 **바로가기 아이콘**이다.
바로가기를 열면 원본 경로로 이동하는 것처럼, 심볼릭 링크에 접근하면 OS가 원본 경로로 추적한다.

---

### 하드 링크 (Hard Link)

```
inode #1234 (실제 데이터)
    ↑           ↑
원본.txt       hardlink.txt   ← 둘 다 같은 inode를 직접 가리킴
```

하드 링크는 **동일한 inode를 가리키는 또 다른 이름**이다.
원본과 하드 링크는 완전히 동등하다. "원본"과 "복사본"이 아니라 **같은 파일의 두 이름**이다.

```bash
ls -li 원본.txt hardlink.txt

# inode 번호가 동일하게 출력됨
1234 ... 원본.txt
1234 ... hardlink.txt
```

---

### 핵심 차이 한눈에 보기

| | 심볼릭 링크 | 하드 링크 |
|---|---|---|
| 실체 | 경로를 담은 별도 파일 | 동일 inode를 가리키는 이름 |
| inode | 별도 inode | 동일 inode 공유 |
| 원본 삭제 시 | 링크가 깨짐 (dangling link) | 나머지 이름으로 여전히 접근 가능 |
| 디스크 사용량 | 경로 문자열 크기만 추가 | 0 (데이터 복사 없음) |
| 파티션 제한 | 없음 | 같은 파티션 내에서만 가능 |
| `resolve()` 결과 | 원본 실제 경로로 추적됨 | 현재 경로 유지 |

---

## 🐛 문제 발생 — YOLO와 심볼릭 링크

### YOLO의 라벨 경로 탐색 방식

YOLO는 이미지 경로에서 라벨을 이렇게 찾는다.

```
이미지 경로: .../data_seg/train/images/weld_001.jpg
                                ↓ 'images' → 'labels' 치환
라벨 경로:  .../data_seg/train/labels/weld_001.txt
```

단순한 문자열 치환처럼 보이지만, 내부적으로 `Path.resolve()` 를 거친다.

---

### 심볼릭 링크가 문제가 된 이유

처음에 구성한 디렉토리 구조:

```bash
# 시도했던 방식
ln -s /mnt/e/.../train/labels_seg /mnt/e/.../data_seg/train/labels
```

```
data_seg/train/labels → (symlink) → .../train/labels_seg/
```

YOLO 내부에서 `Path.resolve()` 를 호출하는 순간:

```python
# YOLO 내부 동작 (대략)
label_path = "data_seg/train/labels/weld_001.txt"
label_path = label_path.resolve()
# resolve()가 심볼릭 링크를 추적해버림
# → "data/The Welding Defect Dataset/train/labels_seg/weld_001.txt"
# data_seg/ 구조를 완전히 벗어난 경로로 이동
```

`resolve()` 는 심볼릭 링크를 따라가서 **실제 경로**로 변환한다.
그 결과 `data_seg/` 구조를 벗어나 원본 데이터셋 폴더를 가리키게 되어 경로를 찾지 못했다.

---

## ✅ 해결 — 하드 링크로 전환

### 핵심 코드

```python
for lbl in src_lbls.glob("*.txt"):       # labels_seg/ 의 파일들
    dst = seg_lbl_dir / lbl.name          # data_seg/train/labels/ 경로
    try:
        os.link(lbl, dst)                 # 하드 링크 생성
    except OSError:
        shutil.copy2(lbl, dst)            # 파티션이 다르면 실제 복사로 fallback
```

### 하드 링크 이후 구조

```
[inode #5521] ← labels_seg/weld_001.txt        (원본)
              ← data_seg/train/labels/weld_001.txt  (하드 링크)
```

두 경로가 완전히 동일한 inode를 공유하므로:

```python
label_path = "data_seg/train/labels/weld_001.txt"
label_path.resolve()
# → "data_seg/train/labels/weld_001.txt"  그대로
# 심볼릭 링크가 없으니 추적할 포인터 자체가 없음
```

YOLO 입장에서는 `data_seg/train/labels/` 에 진짜 파일이 있는 것처럼 동작한다.

---

## 💾 디스크 사용량은?

하드 링크는 데이터를 복사하지 않는다. 같은 inode를 공유하므로 **디스크 추가 사용량은 0**이다.

```bash
ls -li labels_seg/weld_001.txt data_seg/train/labels/weld_001.txt

5521 ... labels_seg/weld_001.txt
5521 ... data_seg/train/labels/weld_001.txt
# inode 번호 동일 = 실제로 같은 파일
```

단, 파티션이 다르면 `os.link()` 가 실패하므로 `shutil.copy2()` 로 fallback 처리를 해뒀다.

---

## 🔍 인사이트 및 결론

**1. `resolve()` 는 심볼릭 링크를 끝까지 추적한다**

단순히 경로를 절대경로로 바꾸는 것이 아니라, 심볼릭 링크를 따라가서 실제 inode가 있는 경로까지 풀어낸다. 라이브러리 내부에서 `resolve()` 를 쓰는 경우 심볼릭 링크 구조가 예상치 않게 깨질 수 있다.

**2. 하드 링크는 "같은 파일의 두 이름"이다**

원본과 하드 링크 중 어느 쪽을 삭제해도 나머지 이름으로 파일에 접근할 수 있다. inode의 reference count가 0이 되는 순간에야 실제 데이터가 삭제된다.

**3. 디스크를 아끼면서 구조를 유지하고 싶다면 하드 링크가 유리하다**

단, 같은 파티션 내에서만 동작한다는 제약이 있으므로 fallback 처리를 항상 넣어두는 것이 좋다.