from preprocess import *
import sys
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

H, W = 500, 500
FULL_CHANNEL = False

tr3 = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [1, 1, 1])
tr4 = transforms.Normalize(mean = [0.5, 0.5, 0.5, 0.5], std = [1, 1, 1, 1])

class dataset(Dataset):
    def __init__(self, sentences, label):
        self.sen = sentences
        self.label = label
    def __getitem__(self, idx):
        return torch.Tensor([self.sen[idx]]), self.label[idx]
    def __len__(self):
        return len(self.sen)

if __name__ == '__main__':
    tr_dt, tr_lb, te_dt, te_lb = genconfig()
    tr_set = dataset(tr_dt, tr_lb)
    tr_loader = DataLoader(tr_set, batch_size = 16, shuffle = True)
    te_set = dataset(te_dt, te_lb)
    te_loader = DataLoader(te_set, batch_size = 16)
    print('checking data...')
    for s, label in tr_loader:
        pass
    for s, label in te_loader:
        pass
    print('finished!')
