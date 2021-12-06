import torch, torchvision
from torch import optim
from torch.optim import lr_scheduler
from torch import nn
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



# image augmentation

# Normalization required by pre-trained model 
mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

# image transformations (learned in comp vision class)
transforms = { 
    'train': T.Compose([
        T.RandomResizedCrop(size=256),
        T.RandomRotation(degrees=15), 
        T.RandomHorizontalFlip(), 
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ])
    , 'val': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(), 
        T.Normalize(mean_nums, std_nums)
    ])
    , 'test': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ])
}

image_datasets = {
        d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS
}

data_loaders = {
        d: DataLoader(image_datasets[d], batch_size=4, shuffle=True)#, num_workers=4)
        for d in DATASETS
}

dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}
class_names = image_datasets['train'].classes 
print(dataset_sizes)

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([mean_nums])
    std = np.array([std_nums])
    inp = std*inp+mean
    inp = np.clip(inp, 0,1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    plt.show()
'''
inputs, classes = next(iter(data_loaders['train']))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
'''

def create_model(n_classes):
    model = models.resnet34(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, n_classes)
    return model.to(device)

base_model = create_model(len(class_names))
def train_epoch(
        model, 
        data_loader, 
        loss_fn,
        optimizer, 
        device,
        scheduler,
        n_examples
        ):
    
    model = model.train()
    losses = []
    correct_predictions = 0
    
    for inputs, labels in tqdm(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)

        correct_predictions += torch.sum(preds==labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step() # why do you step?
        optimizer.zero_grad() # why zero out the grad?

    scheduler.step()
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(
        model, 
        data_loader, 
        loss_fn,
        device, 
        n_examples
        ):
    model = model.eval()
    losses = []
    correct_predictions = 0 
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)

def train_model(
        model, 
        data_loaders, 
        dataset_sizes, 
        device, 
        n_epochs=3
        ):
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1} / {n_epochs}')
        print('-'*10)
    
        train_acc, train_loss = train_epoch(
                model, 
                data_loaders['train'],
                loss_fn,
                optimizer,
                device, 
                scheduler,
                dataset_sizes['train']
        )
        print(f'Train Loss {train_loss:3f}, accuracy {train_acc:2f}')

        val_acc, val_loss = eval_model(
            model, 
            data_loaders['val'],
            loss_fn,
            device, 
            dataset_sizes['val']
        )

        print(f'Val loss {val_loss:3f} accuracy {val_acc:2f}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    print(f'Best val accuracy: {best_accuracy}')
    model.load_state_dict(torch.load('best_model_state.bin'))
    return model, history
print(f'Training on {device}...')

import os 
if os.path.exists('history.npy') and os.path.exists('best_model_state.bin'):
    print('Loading History and Model from save file')
    base_model_sd = torch.load('best_model_state.bin')
    base_model.load_state_dict(base_model_sd)
    history = np.load('history.npy', allow_pickle=True)
    history = history.tolist()
    history['train_acc'] = [x.item() for x in history['train_acc']]
    history['train_loss'] = [x.item() for x in history['train_loss']]
    history['val_acc'] = [x.item() for x in history['val_acc']]
    history['val_loss'] = [x.item() for x in history['val_loss']]


else:
    base_model, history = train_model(base_model, data_loaders, dataset_sizes, device)
    np.save('history.npy', arr=history)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['train_loss'], label='train loss')
    ax1.plot(history['val_loss'], label='validation loss')

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend()
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(history['train_acc'], label='train accuracy')
    ax2.plot(history['val_acc'], label='validation accuracy')

    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend()
    ax2.set_xlabel('Epoch')
    fig.suptitle('Training History')
    plt.show()
'''
plot_training_history(history)
'''

def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)
            real_values.extend(labels)

    predictions = torch.as_tensor(predictions).cpu()
    real_values = torch.as_tensor(real_values).cpu()
    return predictions, real_values
print(type(base_model))
y_pred, y_test = get_predictions(base_model, data_loaders['test'])
cr = classification_report(y_test, y_pred, target_names=class_names)
print(type(cr), cr)

def show_confusion_matrix(confusion_matrix, class_names):
    cm = confusion_matrix.copy()
    cell_counts = cm.flatten()
    cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]
    row_percentages = [f'{value:.2f}' for value in cm_row_norm.flatten()]

    cell_labels = [f'{cnt}\n{per}' for cnt, per in zip(cell_counts, row_percentages)]
    cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])
    df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)
    

    # plot with sns
    hmap = sns.heatmap(df_cm, annot=cell_labels, fmt='', cmap='Blues')
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True Sign')
    plt.xlabel('Predicted Sign')
    plt.show()

cm = confusion_matrix(y_test, y_pred)
print(cm)
show_confusion_matrix(cm, class_names)

def predict_proba(model, image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')
    img = transforms['test'](img).unsqueeze(0)

    pred = model(img.to(device))
    pred = F.softmax(pred, dim=1)
    return pred.detach().cpu().numpy().flatten()

def show_prediction_confidence(predictions, class_names):
    pred_df = pd.DataFrame({'class_names': class_names, 'values': predictions})
    sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
    plt.xlim([0,1])
    plt.show()

show_image('stop-sign.jpg')
pred = predict_proba(base_model, 'stop-sign.jpg')
print(pred)
show_prediction_confidence(pred, class_names)

# Unknown Model 
show_image('unknown-sign.jpg')
pred = predict_proba(base_model,'unknown-sign.jpg')
print(pred)
show_prediction_confidence(pred, class_names)


