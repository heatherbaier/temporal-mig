from torch.distributed.elastic.multiprocessing.errors import record
from collections import OrderedDict
import torch.distributed.rpc as rpc
import torch.distributed as dist
from torchvision import models
from threading import Lock
import torch
import heapq
import os
import gc

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


@record
def load_ddp_state(state_dict):

    r18 = models.resnet18()
    r18.fc = torch.nn.Linear(512, 1)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation[key]
        new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict
    

@record
def sort_by_size(munis, imagery_dir):
    """
    Sort images by size, highest to lowest, but keep the name of the images
    """
    munis = [imagery_dir + i for i in munis]
    munis = sorted(munis, key =  lambda x: os.stat(x).st_size)
    munis.reverse()
    return munis


@record
def sublist_creator(lst, n):
    """
    Split the sorted file list into n lists of equal sums
    """
    lists = [[] for _ in range(n)]
    totals = [(0, i) for i in range(n)]
    heapq.heapify(totals)
    for value in lst:
        total, index = heapq.heappop(totals)
        lists[index].append(value)
        heapq.heappush(totals, (total + value, index))
    return lists


@record
def make_worker_list(files_lists, ppn):
    """
    Get the list of workers based on the number of files to each node
    """
    workers = []
    for c, (i) in enumerate(files_lists):
        for j in range(0, len(i)):
            workers.append(j + (ppn * c))
    return workers


@record
def reverse_size(files_lists, size_dict):
    """
    Make a new imagery list 
    """
    image_list = []
    for j in files_lists:
        for i in j:
            image_list.append(size_dict[i])
    return image_list


@record
def organize_data(base_dir, ppn, nodes):
    
    # Get a list of the municipalities
    munis = os.listdir(base_dir)
    munis = [i for i in munis if i.startswith("484")]
        
    # Sort the municipalities from biggest to smallest size
    munis = sorted(munis, key =  lambda x: os.stat(base_dir + x).st_size)
    
    # Make a dictionary with the image sizes as keys and image names as values
    size_dict = {}
    for x in munis:
        size_dict[os.stat(base_dir + x).st_size] = base_dir + x
                    
    # Change the munis list to be image sizes then reverse it
    munis = [os.stat(base_dir + x).st_size for x in munis]
    munis.reverse()
        
    files_lists = sublist_creator(munis, nodes)
    workers = make_worker_list(files_lists, ppn)
    image_list = reverse_size(files_lists, size_dict)
        
    return image_list, workers


def load_extracter_state(state_dict):

    r18 = models.resnet18()
    r18.fc = torch.nn.Linear(512, 1)

    key_transformation = {k:v for k,v in zip(state_dict.keys(), r18.state_dict().keys())}

    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if "fc." not in key:
            new_key = key_transformation[key]
            new_state_dict[new_key] = value

    del r18, key_transformation, state_dict
    gc.collect()

    return new_state_dict