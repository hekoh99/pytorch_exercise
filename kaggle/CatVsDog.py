import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline 

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
TRAIN_PATH = "./train/"
TEST_PATH = "./test1/"

print(DEVICE)