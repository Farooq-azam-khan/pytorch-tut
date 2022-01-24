import torch 
import os 
import numpy as np
import tqdm as tqdm 
import seaborn  as sns 
from pylab import rcParams 
from matplotlib import rc 
from sklearn.preprocessing import MinMaxScaler 
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim 
import pandas as pd 

import matplotlib.pyplot as plt 


print('Setting Params')
sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F0210FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print('Loading Data')
df = pd.read_csv('./data/time_series_covid19_confirmed_global.csv')

print('Data Processing Step')
# Get rid of Province, Country, Latitude, and Longitude
df = df.iloc[:,4:]
print('Number of missing values:', df.isnull().sum().sum())

daily_cases = df.sum(axis=0) # sum across the columns
daily_cases.index = pd.to_datetime(daily_cases.index)

# plt.plot(daily_cases)
# plt.title('Cummulative daily cases')
# plt.show()

print('Removing Cummulation with pandas diff()')
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.diff.html
daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

# plt.plot(daily_cases)
# plt.title('Daily Cases')
# plt.show()

print(f'Daily Cases shape: {daily_cases.shape}')

print('Splitting Data')
test_data_size = 14
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]
print(f'Train Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}')

print('Scalling Data between 0 and 1 with sklearn\'s MinMaxScaler')
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))
print(f'Train Data Shape: {train_data.shape}, Test Data Shape: {test_data.shape}')