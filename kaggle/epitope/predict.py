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
print(device)

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

# --------------------------------------------------------
#   Set Random seed
# --------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED'])

# --------------------------------------------------------
#   Data pre-processing
# --------------------------------------------------------

def get_preprocessing(data_type, new_df):
    alpha_map = {
                'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5,
                'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L':11,
                'M':12, 'N':13, 'O':14, 'P':15, 'Q':16, 'R':17,
                'S':18, 'T':19, 'U':20, 'V':21, 'W':22, 'X':23,
                'Y':24, 'Z':25, '<PAD>':26,
            }
    epitope_list = []
    left_antigen_list = []
    right_antigen_list = []
    label_list = []

    for epitope, antigen, s_p, e_p in tqdm(zip(new_df['epitope_seq'], new_df['antigen_seq'], new_df['start_position'], new_df['end_position'])):
        epitope_pad = [26 for _ in range(CFG['EPITOPE_MAX_LEN'])] # 길이 맞춰주기 위한 패딩. 모두 같은 feature를 가져야 함
        left_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]
        right_antigen_pad = [26 for _ in range(CFG['ANTIGEN_MAX_LEN'])]
        
        epitope = [alpha_map[x] for x in epitope]
    
    return epitope_list, left_antigen_list, right_antigen_list, label_list