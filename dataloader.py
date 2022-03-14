from torchvision import transforms
import netCDF4 as nc
import numpy as np
import random
import torch
import os

from config import get_config
config, _ = get_config()

def get_munis(imagery_list, rank, worker_map):

    # munis = os.listdir(config.imagery_dir)
    # munis = [i for i in munis if i.startswith("484")]

    # # num_munis = len(munis)
    # # test = num_munis / (world_size * config.ppn)
    # # while int(test) - test != 0:
    # #     num_munis -= 1
    # #     test = num_munis / (world_size * config.ppn)
    
    # # with open(config.log_name, "a") as f:
    # #     f.write("NUM NUNIS: " + str(num_munis) + " out of " + str(len(munis)) + "\n")

    # # munis = munis[0:num_munis]

    # # divide munis into chunks based on the number of nodes
    # split_munis = np.array_split(munis, world_size - 1)
    # rank_munis = split_munis[rank - 1]

    index = worker_map[rank]

    return [imagery_list[index]]


class Dataloader():

    def __init__(self, munis, imagery_dir, rank):

        self.munis = munis
        self.imagery_dir = imagery_dir
        self.train_data, self.val_data = [], []
        self.rank = rank
        self.split = config.tv_split
        self.num_train = int(72 * self.split)
        self.load_data()

    def load_data(self):

        for ncf in self.munis:

            # print(os.path.join(self.imagery_dir, ncf))

            # ds = nc.Dataset(os.path.join(self.imagery_dir, ncf), "r")

            ds = nc.Dataset(ncf, "r")

            # ims, migs = ds["ims"][0:1], ds["migrants"][0:1]
            ims, migs = ds["ims"], ds["migrants"]

            # self.train_data.append((torch.tensor(np.array(ims), dtype = torch.float32), torch.tensor(np.array(migs), dtype = torch.float32)))

            self.train_data.append((torch.tensor(np.array(ims[0:self.num_train]), dtype = torch.float32), torch.tensor(np.array(migs[0:self.num_train]), dtype = torch.float32)))
            # self.val_data.append((torch.tensor(np.array(ims[self.num_train:]), dtype = torch.float32), torch.tensor(np.array(migs[self.num_train:]), dtype = torch.float32)))

            self.val_data.append((torch.tensor(np.array(ims[self.num_train:6]), dtype = torch.float32), torch.tensor(np.array(migs[self.num_train:6]), dtype = torch.float32)))

            # with open(config.log_name, "a") as f:
            #     f.write(str(self.rank) + " " + str(ncf) + "\n")