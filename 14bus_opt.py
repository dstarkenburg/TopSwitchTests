import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

data = pd.read_csv('loads.csv')
x1_1 = data.get('load')
x1_2 = data.get('qd')
x1_3 = data.get('pd')
y1 = data.get('status')

data = pd.read_csv('branches.csv')
alpha = data.get('alpha')
x2_1 = data.get('branch')
x2_2 = data.get('prisk')
y2 = data.get('status')