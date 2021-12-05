import torch, torchvision
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models 


from pathlib import Path
from collections import defaultdict 

import numpy as np
import pandas as pd
from tqdm import tqdm 
import PIL.Image as Image
import cv2

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob
import shutil

# config
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F0\
0FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

# SEED 
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Recognize traffic signs

