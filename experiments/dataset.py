import torch
from torch.utils.data import Dataset
import numpy as np
import random
import glob
import pandas as pd


class myDataset(Dataset):
    def __init__(self, path, look_back=100, train=True, future_look = 1):
        self.root_dir = path
        self.look_back = look_back
        self.future_look = future_look
        self.train = train
        self.size = None
        self.data = self.load_stock_data()

        length = self.data.shape[1]
        self.labels = self.data[:,length - future_look:length+1,1].astype(float) #get label - the last tickers of loopback
        self.prices = self.data[:,:length - future_look,1].astype(float) # get only prices
        self.datetime = self.data[:, :-1, 0].astype(str).tolist()

    def __getitem__(self, idx):
        lab = torch.tensor(np.expand_dims(self.labels[idx],axis=-1))
        item = torch.from_numpy(np.expand_dims(self.prices[idx],axis=-1))

        return {"label" :  lab, "x_series" : item,"date": self.datetime[idx]}

    def __len__(self):
        return self.size

    def load_stock_data(self):
        stock_df = pd.read_csv(self.root_dir)
        if len(stock_df) < self.look_back:
            raise Exception("The stock time-series length is lower than look back.")

        data = []
        for index in range(len(stock_df) - self.look_back - self.future_look):
            data.append(stock_df[index:index + self.look_back + self.future_look])
        data = np.array(data)
        val_set_size = int(np.round(0.1 * data.shape[0]))
        train_set_size = data.shape[0] - (val_set_size)

        if self.train:
            data = data[:train_set_size, ...]
            self.size = train_set_size
        else:
            data = data[train_set_size:, ...]
            self.size = val_set_size
        return data
