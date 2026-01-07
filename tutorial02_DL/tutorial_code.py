import os
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 데이터 폴더 경로 판단
cur_dir = os.getcwd()
dir_name = os.path.basename(cur_dir)
if dir_name == 'tutorial02_DL':
    img_root = os.path.join(cur_dir, 'hotdog_dataset') 
elif dir_name.startswith('lecture-tukorea-20260109'):
    img_root = os.path.join(cur_dir, 'tutorial02_DL', 'hotdog_dataset')
else: 
    raise ValueError('Please run this script from the proper directory.')

# GPU 가용 여부 판단
if torch.cuda.is_available(): # NVIDIA GPU
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available(): # Apple Metal GPU
    device = torch.device('mps:0')
elif torch.xpu.is_available(): # Intel GPU. 단 pytorch XPU 빌드 필요
    device = torch.device('xpu:0')
else:   # GPU 없음. CPU 사용
    device = torch.device('cpu')

print ("Pytorch device:", device)

img_size = (224, 224)
batch_size = 128

# 이미지 로딩 및 전처리 구성
train_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# 데이터 로딩 및 데이터셋 생성
train_imgs = datasets.ImageFolder(root=os.path.join(img_root, 'train'), transform=train_transform)
test_imgs = datasets.ImageFolder(root=os.path.join(img_root, 'test'), transform=test_transform)
train_dataloader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_imgs, batch_size=batch_size)

# 이미지 데이터 확인
# plt.imshow(train_imgs[0][0].permute(1, 2, 0))
# plt.show()

# CNN 모델 정의 
cnn = nn.Sequential(
    nn.Conv2d(3, 96, kernel_size=11, padding=1, stride=4), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(), 
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), 
    nn.MaxPool2d(kernel_size=3, stride=2), 
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(0.5), 
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5), 
    nn.Linear(4096, 2)
)

# 가중치 초기화
for layer in cnn:
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

# 모델을 GPU 또는 CPU에 위치
cnn = cnn.to(device)

# 학습 설정
optimizer = optim.SGD(cnn.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss() 

# 모델 학습 
print("학습 시작...")
for epoch in range(50): 
    loss_sum, cnt = 0, 0

    cnn.train()    # 학습 모드로 전환 (Dropout 등 활성화)
    for feature, label in train_dataloader: # 모든 학습 데이터에 대해
        optimizer.zero_grad() # Gradient 초기화
        pred = cnn(feature.to(device)) # 순전파 계산
        loss = loss_fn(pred, label.to(device)) # 손실 함숫값 계산
        loss.backward() # 역전파 계산
        optimizer.step() # 가중치 갱신
        loss_sum += loss.item()
        cnt += 1

    print("Epoch #", epoch, " - training loss=", (loss_sum / cnt))

    if epoch > 0 and epoch % 5 == 4: # 5 epoch마다 테스트 데이터로 평가
        cnn.eval() # 평가 모드로 전환 (Dropout 등 비활성화)
        with torch.no_grad(): # 평가시 역전파 계산 비활성화
            total, hit = 0, 0
            
            for feature, label in test_dataloader: # 모든 테스트 데이터에 대해
                pred_test = F.softmax(cnn(feature.to(device)), dim=1) # 예측 결과 도출
                total += pred_test.shape[0] # 테스트 데이터수 합산
                hit += (pred_test.argmax(axis=1) == label.to(device)).sum().item() # 정답수 합산

            print("예측 정확도 (Accuracy) = ", hit, "/", total, " = ", round(hit/total, 2))

print("학습 완료")

# 학습한 모델 저장
torch.save(cnn.state_dict(), "hotdog_cnn.pt")