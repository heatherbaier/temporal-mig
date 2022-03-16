#!/usr/bin/env python3
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import models
import datetime
import socket
import pprint
import time
import sys
import io
import os

from dataloader import *
from utils import *

from config import get_config

config, _ = get_config()


import heapq
import os



@record
def run_averager(num_workers):

    with open(config.log_name, "a") as f:
        f.write(str('Setting up averager! On rank: ') + str(dist.get_rank()) + "\n")      

    cur_epoch = 0

    while cur_epoch < config.epochs:

        # Get the number of files in the epoch folder
        epoch_folder = os.path.join(config.records_dir, "epochs", str(cur_epoch))
        num_files = len(os.listdir(epoch_folder))

        # num_workers - 1 * 2 would indicate there is both a train and val file in the folder for every rank
        if num_files == (num_workers * 2):

            train_avg, val_avg = [], []

            # For each of the files, open it and save the average to the appropriate list
            for r in os.listdir(epoch_folder):

                if "train" in r:
                    with open(epoch_folder + "/" + r, "r") as f:
                        try:
                            train_avg.append(float(f.read()))
                        except:
                            pass
    
                elif "val" in r:
                    with open(epoch_folder + "/" + r, "r") as f:
                        try:
                            val_avg.append(float(f.read()))                        
                        except:
                            pass

            with open(config.log_name, "a") as f:
                f.write("Epoch: " + str(cur_epoch) + "  Training Accuracy: " + str(np.average(train_avg)) + "\n"  + "          Validation Accuracy: " + str(np.average(val_avg)) + "\n\n")

            cur_epoch += 1

            # If we've reached the total number of epochs, end
            if cur_epoch == config.epochs:
                return

        # Otherwise, wait for a second and check again
        else:
            time.sleep(2)