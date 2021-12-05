import os
import numpy as np 
import pandas as pd
from tqdm import tqdm 
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import torch 
from torch import nn, optim
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device=}')

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F0\
0FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Will it rain Tomorrow?
df = pd.read_csv('weatherAUS.csv')
print(df.shape)
print(df.columns)

# simplify problem 
cols = ['Rainfall', 'Humidity3pm', 'Pressure9am', 'RainToday', 'RainTomorrow']
df = df[cols]

df['RainToday'] = df['RainToday'].replace({'No':0, 'Yes': 1})
df['RainTomorrow'] = df['RainTomorrow'].replace({'No':0, 'Yes': 1})
df = df.dropna(how='any')

print(f'After Preprocessing: {df.shape}')
print('How Balanced is the Dataset?')
print('Rain Tomorrow Value Count')
print(df['RainTomorrow'].value_counts() / df.shape[0])
#sns.countplot(x='RainTomorrow', data=df)
#plt.show()

X = df[['Rainfall', 'Humidity3pm', 'RainToday', 'Pressure9am']]
y = df[['RainTomorrow']]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=0.2, 
            random_state=RANDOM_SEED
)

X_train = torch.from_numpy(X_train.to_numpy()).float().to(device)
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float()).to(device)

X_test = torch.from_numpy(X_test.to_numpy()).float().to(device)
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float()).to(device)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Build the NN
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

net = Net(X_train.shape[1]).to(device)
print(net)
'''
ax = plt.gca()
plt.plot(
        np.linspace(-1, 1, 5),
        F.relu(torch.linspace(-1, 1, steps=5)).numpy()
)
ax.set_ylim([-1.5, 1.5])
plt.show()
'''
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)

def calculate_accuracy(y_true, y_pred):
    # ge: greater than or equal to
    predicted = y_pred.ge(0.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)

print('Started Training')
for epoch in range(1_000):
   
    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    
    train_loss = criterion(y_pred, y_train)

    if epoch % 100 == 0:
        print(f'Epoch {epoch}')
        # train accuracy + loss 
        train_acc = calculate_accuracy(y_train, y_pred)

        # test loss + acc
        y_test_pred = net(X_test)
        y_test_pred = torch.squeeze(y_test_pred)
        test_acc = calculate_accuracy(y_test, y_test_pred)
        test_loss = criterion(y_test_pred, y_test) 
        print(
f'''
Train set - loss {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
Test set - loss {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
''')

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()



# save the model 
MODEL_PATH = 'model.pth'
torch.save(net, MODEL_PATH)

# load the model 
net = torch.load(MODEL_PATH)

# Evaluation
classes = ['No Rain', 'Raining']
y_pred = net(X_test)
y_pred = y_pred.ge(0.5).view(-1).cpu()
y_test = y_test.cpu()
print(classification_report(y_test, y_pred, target_names=classes))

print('Confusion Matrix')
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
hmap = sns.heatmap(df_cm, annot=True, fmt='d')
hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')


plt.show()
