---
layout: post
title: CLIP_Test
date: 2025-01-11 04:00:00 +0800
category: experiment
thumbnail: /style/image/thumbnail.png
icon: code
---


## CLIP이란?

CLIP(Contrastive Language-Image Pre-training)은 OpenAI에서 개발한 멀티모달 딥러닝 모델.
텍스트와 이미지를 동시에 이해하고 연결까지만 가능.

### 주요 특징

- **멀티모달 학습:** 4억 개 이상의 이미지-텍스트 쌍으로 학습되어 이미지와 텍스트 간의 관계를 매칭.
- **제로샷 학습:** 특정 작업에 대한 추가 학습 없이도 다양한 시각적 작업을 수행 가능.
- **자연어 기반 분류:** 자연어 설명만으로 이미지를 분류할 수 있어 여러 분야에 적용 가능.

### 작동 원리

CLIP은 두 개의 인코더로 구성됩니다:

1. **이미지 인코더:** Vision Transformer(ViT) 또는 ResNet을 사용하여 이미지를 벡터로 변환합니다.
2. **텍스트 인코더:** Transformer를 사용하여 텍스트를 벡터로 변환합니다.

학습 과정에서 올바른 이미지-텍스트 쌍의 유사도는 높이고, 잘못된 쌍의 유사도는 낮추는 대조 학습(Contrastive Learning) 방식을 사용합니다.

### 이론은 알아봤으니 Code 레벨에서 확인

- openai 프리트레인된 Vit-B-32 모델을 사용

```java
tokenizer = open_clip.get_tokenizer("ViT-B-32")
image = Image.open(image_path).convert("RGB")
image_input = preprocess(image).unsqueeze(0).to(DEVICE)
```

1. 토크나이저를 통해 텍스트를 CLIP 모델에 fit될 수 있게 토큰화
2. 이미지는 우선 행열채 → 채행열 → 개채행열 텐서로 변환 (이렇게 해야 인코더에 들어감)

```java
food_texts = [
    "a photo of ramen",
    "a photo of pizza",
    "a photo of fried chicken",
    "a photo of bibimbap",
    "a photo of kimchi stew",
    "a plate of sushi",
    "a photo of cola",
]
with torch.no_grad():
    text_tokens = tokenizer(food_texts).to(DEVICE)
```

- 지금은 귀여운 양의 텍스트만 데이터로 사용
- 텍스트 토큰화를 진행

```java
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_tokens)  
```

![스크린샷 2025-12-08 오후 7.09.57.png](style/image/clip/1.png)

![스크린샷 2025-12-08 오후 7.11.06.png](style/image/clip/2.png)

앞 뒤 CLS 토큰이 붙고, 단어에 맞게 a = 320, photo = 1125 등 임베딩 된 모습
0번과 1번의 경우 마지막 단어만 다르기에 18892, 4474로 다르게 벡터화 되고 나머지는 동일함 확인

![스크린샷 2025-12-08 오후 7.17.16.png](style/image/clip/3.png)

![스크린샷 2025-12-08 오후 7.15.16.png](style/image/clip/4.png)

preprocess를 통과하여 전처리 된 이미지를 텍스트 벡터와 같은 디멘젼으로 맞춰줌

```java
similarities = (image_features @ text_features.T).squeeze(0)  # shape: (N,)
```

코사인 유사도 계산

![image.png](style/image/clip/5.png)

input image를 콜라로 넣어둔 상태여서 “a photo of cola”와 가장 유사도가 높음을 확인

![스크린샷 2025-12-08 오후 7.19.16.png](style/image/clip/6.png)