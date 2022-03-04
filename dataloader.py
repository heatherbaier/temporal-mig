from torchvision import transforms
import netCDF4 as nc
import numpy as np
import torch
import os


def get_munis(rank, world_size):

    # read in list of munis
    # with open("/sciclone/geograd/heather_data/munis_list.txt", "r") as f:
    #     munis = f.read().splitlines()

    munis = os.listdir("/sciclone/scr-mlt/hmbaier/tm/")
    munis = [i for i in munis if i.startswith("484")]

    # divide munis into chunks based on the number of nodes
    split_munis = np.array_split(munis, world_size - 1)
    rank_munis = split_munis[rank - 1]

    return rank_munis


def list_imagery(folders):

    # for each muicipality that has extracted tifs, get file names and return a master list
    all_images = []
    for folder in folders:
        [all_images.append(folder + i) for i in os.listdir(folder)]



class Dataloader():

    def __init__(self, munis, imagery_dir, rank):

        self.munis = munis
        self.imagery_dir = imagery_dir
        self.data = []
        self.rank = rank
        self.load_data()

    def load_data(self):

        for ncf in self.munis[0:2]:
            print(os.path.join(self.imagery_dir, ncf))
            ds = nc.Dataset(os.path.join(self.imagery_dir, ncf), "r")
            ims, migs = ds["ims"][0:1], ds["migrants"][0:1]
            self.data.append((torch.tensor(np.array(ims), dtype = torch.float32), torch.tensor(np.array(migs), dtype = torch.float32)))
            # for (im, mig) in zip(ims,  migs):
                # self.data.append((torch.tensor(np.array(im.data), dtype = torch.float32), torch.tensor(np.array(mig.data), dtype = torch.float32)))

            # with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
            #     f.write(str(self.rank) + " " + str(ncf) + "\n")