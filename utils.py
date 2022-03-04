from torch.distributed.elastic.multiprocessing.errors import record

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import models
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
from threading import Lock

from dataloader import *

from config import get_config
config, _ = get_config()


# The evaluator instance.
evaluator = None

# A lock to ensure we only have one parameter server.
global_lock = Lock()


def _call_method(method, rref, *args, **kwargs):

    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def get_evaluator(placeholder):
    """
    Returns a singleton evaluator to all trainer processes
    """
    global evaluator
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not evaluator:
            # construct it once
            evaluator = Evaluator()
        return evaluator

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Evaluator:

    def __init__(self):

        self.training_loss = 0
        self.validation_loss = 0
        self.num_train_collected = 0
        self.num_val_collected = 0
        self.evaluator_rref = rpc.remote(
            "worker_0", get_evaluator, args=(1,))
        self.epoch_tracker = {}
        self.num_train = dist.get_world_size() - 1

        print("eval initialized!! num train is: ", self.num_train)

    def collect_losses(self, loss, epoch):

        # print("IN COLLECT LOSSES!")

        train = True 

        if epoch in self.epoch_tracker.keys():
            self.epoch_tracker[epoch].update(loss)
            
            # print("NUM IN EPOCH TRACKER: ", self.epoch_tracker[epoch].count, " out of ", self.num_train)
            
            if self.epoch_tracker[epoch].count == self.num_train:
                print("EPOCH: ", str(epoch), str(self.epoch_tracker[epoch].avg))
                with open(config.log_name, "a") as f:
                    f.write("EPOCH: " + str(epoch) + str(self.epoch_tracker[epoch].avg) + "\n")   

        else:
            self.epoch_tracker[epoch] = AverageMeter()
            self.epoch_tracker[epoch].update(loss)
            # print(self.epoch_tracker[epoch])

        with open(config.log_name, "a") as f:
            f.write("IN COLLECT LOSSES WITH NUM = " + str(self.epoch_tracker[epoch].count) + "\n")    

        return 0        