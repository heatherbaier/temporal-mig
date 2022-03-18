from torchvision import transforms
import netCDF4 as nc
import numpy as np
import random
import torch
import json
import ast
import os

from lstm_config import get_config

config, _ = get_config()



def get_munis(features_list, rank, worker_map):
    index = worker_map[rank]
    return features_list[index]


class Dataloader():

    def __init__(self, features, rank):

        self.features = features
        self.rank = rank
        self.split = config.tv_split
        self.num_train = int(72 * self.split)
        self.load_data()

    def load_data(self):
        
        xs, ys = [], []
        
        for feat in self.features:
            
#             print(feat)

            with open(feat, "r") as f:
                data = json.load(f)
            
            features = list(data.values())
            migs = [torch.tensor([ast.literal_eval(i["migrants"])]) for i in features]
            features = [torch.tensor([ast.literal_eval(i["features: "])]) for i in features]

            num_steps = 12
#             x, y = [], []
            for i in range(len(features)):
                xs.append(torch.cat(features[i:i+num_steps]))
                ys.append(torch.cat(migs[i:i+num_steps]))

        x = torch.cat([i.unsqueeze(0) for i in xs if i.shape[0] == 12])
        y = torch.cat([i.unsqueeze(0) for i in ys if i.shape[0] == 12])

        train_num = int(x.shape[0] * config.tv_split)

        # Get a list of train and val indices
        train_indices = random.sample(range(x.shape[0]), train_num)
        val_indices = [i for i in range(x.shape[0]) if i not in train_indices]

        # Subset x & y into train and val
        self.x_train = torch.index_select(x, 0, torch.tensor(train_indices))#.split(32)
        self.x_val = torch.index_select(x, 0, torch.tensor(val_indices))#.split(32)
        self.y_train = torch.index_select(y, 0, torch.tensor(train_indices))#.split(32)
        self.y_val = torch.index_select(y, 0, torch.tensor(val_indices))#.split(32)

        print(self.x_train.shape)
        print(self.x_val.shape)


