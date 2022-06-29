import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sys import path
from zipfile import ZipFile

from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torch.optim import Adam
from torchvision import models
from torch.autograd import Variable
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_PATH = "./data/train/"
TEST_PATH = "./data/test1/"

print(DEVICE)

#--------------------------------------------------------
# un-zip the data
#--------------------------------------------------------

from pathlib import Path

path = Path(TRAIN_PATH)
if not (path).exists():
    print("-- un-zip datasets --")
    with ZipFile('./data/train.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

    with ZipFile('./data/test1.zip', 'r') as zip_ref:
        zip_ref.extractall('./data')

#--------------------------------------------------------
# custom dataset
#--------------------------------------------------------

class CatandDogSet(Dataset): # 확인용
    def __init__(self, path):
        self.path = path
        self.img_name = os.listdir(self.path)
                
    def __len__(self):
        return len(self.img_name)
    
    def __getitem__(self, idx):
        img_idx = self.img_name[idx]
        img_item_path = os.path.join(self.path, img_idx)
        img = Image.open(img_item_path)
        label = img_idx.split(".")[0] # 파일명에서 해당 사진의 label 추출
        return img, label, img_item_path

class CustomSet(Dataset): # 학습에 사용될 것
    def __init__(self, imgs, class_to_int, path, mode, transforms):
        self.imgs = imgs
        self.class_to_int = class_to_int
        self.mode = mode
        self.transforms = transforms
        self.path = path
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(self.path + self.imgs[idx])
        if self.mode  in ["train" , "test"]:
            label = self.class_to_int[self.imgs[idx].split(".")[0]]
            img = self.transforms(img)
            return img, label
        elif self.mode == "pred":
            pred_id = self.imgs[idx].split(".")[0]
            img = self.transforms(img)
            return img, pred_id      

#--------------------------------------------------------
# extract datasets
#--------------------------------------------------------

imgs = os.listdir(TRAIN_PATH)
train_imgs = np.random.choice(imgs , 20000, replace=False) # 비복원 추출
test_imgs = np.setdiff1d(imgs , train_imgs) # imgs과 train_imgs의 차집합
pred_imgs = [f"{path}.jpg" for path in range(1,len(os.listdir(TEST_PATH))+1)]

#--------------------------------------------------------
# data transformation
#--------------------------------------------------------

class_to_int = {"cat" : 0 , "dog" : 1}
train_transforms = transforms.Compose([
    transforms.Resize((360 , 360)) , 
    transforms.RandomHorizontalFlip(p = 0.5) , # 0.5 확률로 좌우 반전
    transforms.ToTensor() , 
    transforms.Normalize((0, 0, 0) , (1, 1, 1)) # 각 채널별로 mean과 std로 정규화
])

test_pred_transforms = transforms.Compose([
    transforms.Resize((360 , 360)) , 
    transforms.ToTensor() , 
    transforms.Normalize((0 , 0 , 0) , (1 , 1, 1))
])

dataset = CatandDogSet(TRAIN_PATH)

EPOCHS = 5
BATCH_SIZE = 20

train = DataLoader(CustomSet(train_imgs , class_to_int , TRAIN_PATH , "train" , train_transforms) , batch_size=BATCH_SIZE , shuffle = True)
test = DataLoader(CustomSet(test_imgs , class_to_int , TRAIN_PATH , "test" , test_pred_transforms) , batch_size=BATCH_SIZE , shuffle = True)
pred = DataLoader(CustomSet(pred_imgs , class_to_int , TEST_PATH , "pred" , test_pred_transforms))

img, label , path = dataset[np.random.randint(len(dataset))]
train_feature, train_label = next(iter(train))
check_img = np.transpose(train_feature[0], (1,2,0))
# plt.imshow(check_img)
# plt.show()
# plt.imshow(img)
# plt.show()

#--------------------------------------------------------
# get pre-trained model
#--------------------------------------------------------

resnet = models.resnet18(pretrained=True).to(DEVICE)
num_classes = 2 # 이진 분류이므로
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, num_classes) # 출력층 노드 수 조절

# print(resnet)

#--------------------------------------------------------
# train model
#--------------------------------------------------------

loss_function = nn.CrossEntropyLoss()

# model(신경망) 파라미터를 optimizer에 전달해줄 때 nn.Module의 parameters() 메소드를 사용
# Karpathy's learning rate 사용 (3e-4)
optimizer = torch.optim.Adam(resnet.parameters(), lr=3e-4)

batch_num = len(train)
test_batch_num = len(test)
losses = []

for e in range(1):
    total_loss = 0

    progress = tqdm(enumerate(train), desc="Loss: ", total=batch_num)
    resnet.train()
    for i, data in progress:
        features, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        resnet.zero_grad() # 모든 모델의 파라미터 미분값을 0으로 초기화
        outputs = resnet(features)

        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step() # step() : 파라미터를 업데이트함

        # training data 가져오기
        current_loss = loss.item() # item() : 키, 값 반환
        total_loss += current_loss

        # set_description : 진행률 프로세스바 업데이트
        progress.set_description("Loss: {:.4f}".format(total_loss/(i+1)))
    
    # ----------------- VALIDATION  ----------------- 
    val_losses = 0
    precision, recall, f1, accuracy = [], [], [], []
    
    # set model to evaluating (testing)
    resnet.eval()
    with torch.no_grad():
        for i, data in enumerate(test):
            features, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            outputs = resnet(features) # 네트워크로부터 예측값 가져오기

            val_losses += loss_function(outputs, labels)

            predicted_classes = torch.max(outputs, 1)[1] # 네트워크의 예측값으로부터 class 값(범주) 가져오기
          
    print(f"Epoch {e+1}/{EPOCHS}, training loss: {total_loss/batch_num}, validation loss: {val_losses/test_batch_num}")
    losses.append(total_loss/batch_num) # 학습률을 위한 작업
print("done")
