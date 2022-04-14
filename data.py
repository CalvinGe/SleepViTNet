import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# import smote_variants as sv
from random import shuffle
import random


def extract(data_file, anno_file):
    # read files:
    path1 = './data/sleep_edf_npy/'
    x_data = np.load(path1 + data_file + '.npy')
    y_data = np.load(path1 + anno_file + '.npy')
    

    sleepStart = np.min(np.argwhere(y_data != 0))
    sleepEnd = len(y_data) - np.min(np.argwhere(y_data[::-1] != 0))
    W_reserve = 120
    print(sleepStart)
    print(len(y_data) - sleepEnd)
    if sleepStart > W_reserve and len(y_data) > sleepEnd + W_reserve:
        x_data = x_data[sleepStart-W_reserve:sleepEnd+W_reserve, :, :]
        y_data = y_data[sleepStart-W_reserve:sleepEnd+W_reserve]
    
    pad = 1000
    d1 = len(y_data)
    x_prev = x_data[:d1-2]
    x_behind = x_data[2:]
    x_new = np.concatenate((x_prev[:, :, 3000-pad:], x_data[1:d1-1], x_behind[:, :, :pad]), axis=2)

    y_new = y_data[1: d1-1]
    print("x_new: ", x_new.shape)
    print("y_new: ", y_new.shape)
    return x_new, y_new



def data_preparation(BATCH_SIZE, num_id):

    data_file = pd.read_table('data_file.txt', header=None)
    anno_file = pd.read_table('anno_file.txt', header=None)

    r=random.random
    random.seed(5)
    a = list(range(len(data_file)))
    shuffle(a, random=r)

    for j in range(6):
        for i in range(j*30, j*30+30):
            x_tmp, y_tmp = extract(data_file.iloc[a[i], 0], anno_file.iloc[a[i], 0])
            if i == j*30:
                x_30 = x_tmp
                y_30 = y_tmp
            else:
                x_30 = np.concatenate((x_30, x_tmp), axis=0)
                y_30 = np.concatenate((y_30, y_tmp), axis=0)
        if j == 0:
            x_60 = x_30
            y_60 = y_30
        else:
            x_60 = np.concatenate((x_60, x_30), axis=0)
            y_60 = np.concatenate((y_60, y_30), axis=0)
    
    for i in range(180, num_id):
        x_tmp, y_tmp = extract(data_file.iloc[a[i], 0], anno_file.iloc[a[i], 0])
        if i == 180:
            x_17 = x_tmp
            y_17 = y_tmp
        else:
            x_17 = np.concatenate((x_17, x_tmp), axis=0)
            y_17 = np.concatenate((y_17, y_tmp), axis=0)


    class EEGdataset(Dataset):
        def __init__(self, x, y):
            self.batch_size = BATCH_SIZE

            num_batch = int(x.shape[0] / self.batch_size)
            self.x_data = torch.from_numpy(x[:num_batch * self.batch_size, :, :])
            self.x_data = self.x_data.type(torch.FloatTensor)

            self.y_data = torch.LongTensor(y[:num_batch * self.batch_size])
            self.len = self.y_data.shape[0]

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.len

    train_data = EEGdataset(x_60, y_60)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_data = EEGdataset(x_17, y_17)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader
