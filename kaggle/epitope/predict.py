import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# --------------------------------------------------------
#   Hyper parameter setting
# --------------------------------------------------------

CFG = {
    'NUM_WORKERS':4,
    'ANTIGEN_WINDOW':128, # antigen max len으로 antigen window 설정
    'ANTIGEN_MAX_LEN':128, 
    'EPITOPE_MAX_LEN':256,
    'EPOCHS':10,
    'LEARNING_RATE':1e-3,
    'BATCH_SIZE':2048,
    'THRESHOLD':0.5,
    'SEED':41
}