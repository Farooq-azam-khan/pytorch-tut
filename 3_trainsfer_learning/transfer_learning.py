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
train_folders = sorted(glob('GTSRB/Final_Training/Images/*'))
print(f'Total Training Folders: {len(train_folders)}')

def load_images(img_path, resize=True):
    # why does cv2 read it as BGR?
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    if resize:
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
    return img 

def show_image(img_path):
    img = load_images(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def show_sign_grid(image_paths):
    images = [load_images(img) for img in image_paths]
    images = torch.as_tensor(images)
    # permute: https://pytorch.org/docs/stable/generated/torch.permute.html
    # changes the ordering, but why? and what are these magic numbers?
    images = images.permute(0, 3, 1, 2)
    grid_img = torchvision.utils.make_grid(images, nrow=11)
    plt.figure(figsize=(24,12))
    plt.imshow(grid_img.permute(1,2,0))
    plt.axis('off')
    plt.show()

'''
print('Showing Sample Images')
sample_images = [np.random.choice(glob(f'{tf}/*ppm')) for tf in train_folders]
show_sign_grid(sample_images)
img_path = glob(f'{train_folders[16]}/*ppm')[1]
show_image(img_path)
'''

class_names = ['priority_road', 'give_way', 'stop', 'no_entry']
class_indicies = [12, 13, 14, 17]

DATA_DIR = Path('data')
DATASETS = ['train', 'val', 'test']

# make directory for each type of dataset and each class index
for ds in DATASETS:
    for cls in class_names:
        (DATA_DIR / ds / cls).mkdir(parents=True, exist_ok=True)

# 80% training, 10% validation, 10% testing
for i, cls_index in enumerate(class_indicies):
    image_paths = np.array(glob(f'{train_folders[cls_index]}/*ppm'))
    class_name = class_names[i]
    print(f'{class_name}: {len(image_paths)}')
    np.random.shuffle(image_paths)

    ds_split = np.split(
            image_paths, 
            indices_or_sections=[int(.8*len(image_paths)), int(.9*len(image_paths))]
    )
    dataset_data = zip(DATASETS, ds_split)
    for ds, images in dataset_data:
        for img_path in images: 
            shutil.copy(img_path, f'{DATA_DIR}/{ds}/{class_name}/')







