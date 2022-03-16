from torchvision import transforms
import netCDF4 as nc
import numpy as np
import random
import torch
import os


class ExtractLoader():

    def __init__(self, muni):

        self.muni = muni
        self.muni_name = self.muni.split("/")[-1].strip(".nc")
        self.load_data()

    def load_data(self):

        ds = nc.Dataset(self.muni, "r")

        ims, migs = ds["ims"], ds["migrants"]

        self.imagery = torch.tensor(np.array(ims), dtype = torch.float32).split(1)
        self.migs = torch.tensor(np.array(migs), dtype = torch.float32).split(1)

        # with open(config.log_name, "a") as f:
        #     f.write(str(self.rank) + "  NUM TRAIN: " + str(len(self.x_train)) + "  NUM VAL: " + str(len(self.x_val)) + "\n")
