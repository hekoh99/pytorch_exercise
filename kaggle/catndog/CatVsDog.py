import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from sys import path
from zipfile import ZipFile

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from torch.optim import Adam
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

class CatandDogSet(Dataset):
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

class CustomSet(Dataset):
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
    transforms.RandomHorizontalFlip(p = 0.5) , 
    transforms.ToTensor() , 
    transforms.Normalize((0 , 0 , 0) , (1 , 1 , 1))
])

test_pred_transforms = transforms.Compose([
    transforms.Resize((360 , 360)) , 
    transforms.ToTensor() , 
    transforms.Normalize((0 , 0 , 0) , (1 , 1 , 1))
])

dataset = CatandDogSet(TRAIN_PATH)

BATCH_SIZE = 20

train = DataLoader(CustomSet(train_imgs , class_to_int , TRAIN_PATH , "train" , train_transforms) , batch_size = BATCH_SIZE , shuffle = True)
test = DataLoader(CustomSet(test_imgs , class_to_int , TRAIN_PATH , "test" , test_pred_transforms) , batch_size = BATCH_SIZE , shuffle = True)
pred = DataLoader(CustomSet(pred_imgs , class_to_int , TEST_PATH , "pred" , test_pred_transforms))

img, label , path = dataset[np.random.randint(len(dataset))]
plt.imshow(img)
plt.show()