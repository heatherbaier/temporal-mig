from torchvision import transforms
import netCDF4 as nc
import numpy as np
import random
import torch
import os

from config import get_config

config, _ = get_config()



def get_munis(imagery_list, rank, worker_map):
    index = worker_map[rank]
    return imagery_list[index]


class Dataloader():

    def __init__(self, muni, imagery_dir, rank):

        self.muni = muni
        self.imagery_dir = imagery_dir
        self.train_data, self.val_data = [], []
        self.rank = rank
        self.split = config.tv_split
        self.num_train = int(72 * self.split)
        self.load_data()

    def load_data(self):

        ds = nc.Dataset(self.muni, "r")

        ims, migs = ds["ims"], ds["migrants"]

        self.train_data.append((torch.tensor(np.array(ims[0:self.num_train]), dtype = torch.float32), torch.tensor(np.array(migs[0:self.num_train]), dtype = torch.float32)))
        self.val_data.append((torch.tensor(np.array(ims[self.num_train:]), dtype = torch.float32), torch.tensor(np.array(migs[self.num_train:]), dtype = torch.float32)))

        with open(config.log_name, "a") as f:
            f.write(str(self.rank) + "  NUM TRAIN: " + str(self.train_data[0][0].shape) + "  NUM VAL: " + str(self.val_data[0][0].shape) + "\n")