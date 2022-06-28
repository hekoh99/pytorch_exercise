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
        label = img_idx.split(".")[0]
        return img, label, img_item_path

imgs = os.listdir(TRAIN_PATH)