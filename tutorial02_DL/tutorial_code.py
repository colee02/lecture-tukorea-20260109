import os
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
dir_name = os.path.basename(cur_dir)
if dir_name == 'tutorial02_DL':
    img_root = os.path.join(cur_dir, 'hotdog_dataset') 
elif dir_name.startswith('lecture-tukorea-20260109'):
    img_root = os.path.join(cur_dir, 'tutorial02_DL', 'hotdog_dataset')
else: 
    raise ValueError('Please run this script from the proper directory.')

if torch.cuda.is_available(): # NVIDIA GPU
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available(): # Apple Metal GPU
    device = torch.device('mps:0')
elif torch.xpu.is_available(): # Intel GPU. 단 pytorch XPU 빌드 필요
    device = torch.device('xpu:0')
else:   # GPU 없음. CPU 사용
    device = torch.device('cpu')

print ("Using device:", device)

img_size = (224, 224)
batch_size = 128

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

train_imgs = datasets.ImageFolder(root=os.path.join(img_root, 'train'), transform=train_transform)
test_imgs = datasets.ImageFolder(root=os.path.join(img_root, 'test'), transform=test_transform)
train_dataloader = DataLoader(train_imgs, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_imgs, batch_size=batch_size)

# plt.imshow(train_imgs[0][0].permute(1, 2, 0))
# plt.show()

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

for layer in cnn:
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)

print(cnn(next(iter(train_dataloader))[0]).shape)
cnn = cnn.to(device)

optimizer = optim.SGD(cnn.parameters(), lr=0.01, weight_decay=0.001)
loss_fn = nn.CrossEntropyLoss() 

for epoch in range(50): 
    loss_sum = 0
    cnt = 0

    cnn.train()    
    for feature, label in train_dataloader: 
        optimizer.zero_grad()
        pred = cnn(feature.to(device))
        loss = loss_fn(pred, label.to(device))
        # print(pred, label, loss)
        
        loss.backward()
        optimizer.step() 
        loss_sum += loss.item()
        cnt += 1

    print(epoch, (loss_sum / cnt))
    if epoch > 0 and epoch % 5 == 4: 
        cnn.eval()
        with torch.no_grad():
            total = 0
            hit = 0
            
            for feature, label in test_dataloader: 
                pred_test = F.softmax(cnn(feature.to(device)), dim=1)
                total += pred_test.shape[0]
                hit += (pred_test.argmax(axis=1) == label.to(device)).sum().item()
                # print(pred_test[:, 0].std().item())
            
            print(hit, "/", total, " = ", round(hit/total, 2))

torch.save(cnn.state_dict(), "hotdog_cnn.pt")