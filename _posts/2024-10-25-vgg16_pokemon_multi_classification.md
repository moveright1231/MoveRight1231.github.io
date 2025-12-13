---
layout: post
title: Multi classification (vgg16, pokemon 데이터셋)
date: 2024-10-25 11:00:00 +0800
category: experiment
thumbnail: /style/image/thumbnail.png
icon: code
---

# Multi Classification

## 데이터셋 분리 및 전처리

[포켓몬 데이터셋](https://www.kaggle.com/datasets/hongdcs/pokemon-gen1-151-classes-classification)

![image.png](style/image/2024-10-25-vgg16_pokemon_multi_classification/a.png)

class 폴더마다 데이터들이 들어가있음, 학습에 사용하기 위해 train, valid, test로 나누는 과정이 필요함

```java
for folder in [train_root, valid_root, test_root]:
    if not os.path.exists(folder):
        os.makedirs(folder)
    for cls in cls_list:
        cls_folder = f"{folder}/{cls}"
        if not os.path.exists(cls_folder):
            os.makedirs(cls_folder)
            
for cls in cls_list:
    file_list = os.listdir(f"{file_root}/{cls}")
    random.shuffle(file_list)
    test_ratio = 0.1
    num_file = len(file_list)
    
    test_list = file_list[:int(num_file*test_ratio)]
    valid_list = file_list[int(num_file*test_ratio):int(num_file*test_ratio)*2]
    train_list = file_list[int(num_file*test_ratio)*2:]

    for i in test_list:
        shutil.copyfile(f"{file_root}/{cls}/{i}", f"{test_root}/{cls}/{i}" )
        
    for i in valid_list:
        shutil.copyfile(f"{file_root}/{cls}/{i}", f"{valid_root}/{cls}/{i}" )
    
    for i in train_list:
        shutil.copyfile(f"{file_root}/{cls}/{i}", f"{train_root}/{cls}/{i}" )
```

train, valid, test 폴더를 생성한 후
train 80%, valid 10%, test 10%로 분류하여 저장

VGG 모델을 사용할 것이므로 224x224 사이즈로 수정
데이터셋이 이미 여러 각도와 다양한 자세, 배경으로 이루어져 있기에 Data Augmentation을 주지 않음

```java
print(dataset_sizes)
print(class_names[:10])
print(len(class_names))

result
{'train': 14984, 'valid': 4062, 'test': 4080}
['Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 'Blastoise', 'Bulbasaur']
151
```

- 폴더 별로 데이터들이 잘 들어갔는지 확인
- 151개의 클래스

![image.png](style/image/2024-10-25-vgg16_pokemon_multi_classification/b.png)

클래스 명(폴더명)과 이미지가 잘 들어갔는지 확인.

## 학습

early stopping을 적용하여 valid Loss가 5번정도 개선이 없으면 중단되는 로직 구현

```java
def train_model(
    model, criterion, optimizer, dataloaders, dataset_sizes, device,
    model_dir, model_name, num_epochs=25,
    early_stopping=True,
    monitor="val_loss",   # "val_loss" or "val_acc"
    patience=5,           # 개선 없을 때 몇 epoch 기다릴지
    min_delta=1e-4,       # 이만큼 이상 개선되어야 "개선"으로 인정
    save_best=True
):
    print(device)
    since = time()

    # best 기록 초기화
    if monitor == "val_loss":
        best_metric = float("inf")   # 작을수록 좋음
        mode = "min"
    elif monitor == "val_acc":
        best_metric = 0.0            # 클수록 좋음
        mode = "max"
    else:
        raise ValueError("monitor must be 'val_loss' or 'val_acc'")

    best_acc = 0.0  # 참고용(출력용)
    epochs_no_improve = 0

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            epoch_val_loss = None
            epoch_val_acc = None

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'train':
                    train_loss.append(epoch_loss)
                    train_acc.append(epoch_acc.item())
                else:
                    epoch_val_loss = epoch_loss
                    epoch_val_acc = epoch_acc.item()
                    valid_loss.append(epoch_loss)
                    valid_acc.append(epoch_acc.item())

                    # best val acc 저장(원래 네 로직 유지)
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc

                    # 모델 저장(옵션)
                    if save_best:
                        # early stopping의 monitor 기준으로 best 판단해서 저장하도록 할 수도 있음
                        pass

            # --- Early Stopping 체크는 valid phase 끝난 뒤(여기서) ---
            if early_stopping:
                if monitor == "val_loss":
                    current = epoch_val_loss
                    improved = (best_metric - current) > min_delta
                else:  # val_acc
                    current = epoch_val_acc
                    improved = (current - best_metric) > min_delta

                if improved:
                    best_metric = current
                    epochs_no_improve = 0

                    # best 모델 저장
                    if save_best:
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        model_save_path = os.path.join(model_dir, f'{model_name}.pth')
                        torch.save(model.state_dict(), model_save_path)

                else:
                    epochs_no_improve += 1
                    print(f"[EarlyStop] no improvement: {epochs_no_improve}/{patience}")

                    if epochs_no_improve >= patience:
                        print(f"[EarlyStop] Stop training at epoch {epoch+1}. Best {monitor} = {best_metric:.6f}")
                        break

            print()

        time_elapsed = time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

    return [train_loss, valid_loss, train_acc, valid_acc]
```

pretrained된 VGG16 모델을 사용하고 데이터셋 클래스 수의 맞게 수정 (151개의 포켓몬 종류)

```java
vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

for param in vgg_model.features.parameters():
    param.requires_grad = False

in_features = vgg_model.classifier[6].in_features   # 4096
vgg_model.classifier[6] = nn.Linear(in_features, num_class)

vgg_model = vgg_model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg_model.parameters(), lr=0.0005, momentum=0.9)
```

- 옵티마이저는 SGD + momentum 방식을 사용
- Multi_class_classification 문제이기에 크로스엔트로피 손실 함수를 사용

```java
Epoch 37/50
----------
100%|██████████| 469/469 [01:17<00:00,  6.05it/s]
train Loss: 0.0265 Acc: 0.9969
100%|██████████| 127/127 [00:29<00:00,  4.23it/s]
valid Loss: 0.0176 Acc: 0.9966
```

train, valid의 Loss와 Acc가 충분히 높기에 직접 중단함.
(early stopping이 작은 상향 업데이트에도 작동을 하지 않아 제대로 활용이 안됨, 로직 변경 필요)

## Test

test 데이터셋도 train과 valid와 동일한 전처리를 거치고 평가 시작

```python
def test_model(model, criterion, dataloader, dataset_size, device):
    print(device)
    since = time()

    model.eval()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    time_elapsed = time() - since
    print(f'Test complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Acc: {epoch_acc:.4f} Loss: {epoch_loss:.4f}')
```

eval모드로 test 구현

학습때 저장된 best model 가중치 파일을 사용

```python
cuda:0
100%|██████████| 255/255 [00:35<00:00,  7.18it/s]
Test complete in 0m 36s
Acc: 0.9941 Loss: 0.0336
```

- Test에서도 높은 정확도와 낮은 Loss를 확인함

직접 이미지를 넣어서 test

```python
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

def predict_one(img_path, model, device, transform, class_names, topk=3):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = probs.topk(topk)
    pred_class = class_names[top_idxs[0].item()]
    print(f"Top-1: {pred_class} ({top_probs[0].item():.3f})")
    for p, idx in zip(top_probs, top_idxs):
        print(f"- {class_names[idx.item()]}: {p.item():.3f}")
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Predicted: {pred_class}")
    plt.show()

img_path = r"D:/ai_projects/datasets/pkm_datasets/test/Weepinbell/img_40.png"  # 원하는 이미지 경로로 변경
predict_one(img_path, vgg_model, DEVICE, data_transforms["test"], class_names, topk=3)

```

![image.png](style/image/2024-10-25-vgg16_pokemon_multi_classification/c.png)

## 🔗 참고 자료

- 프로젝트 코드: (https://github.com/moveright1231/vgg16_multi_classification)