#!/usr/bin/env python3
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import models
import datetime
import socket
import pprint
import sys
import io
import os

from dataloader import *
from utils import *

from config import get_config

config, _ = get_config()


@record
def main(rank, world_size, model_group):

    ######################################################
    # Load rank data
    ######################################################    
    munis = get_munis(rank, world_size)

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write("MUNIS: " + str(munis) + "  RANK: " + str(rank) + "\n")

    data = Dataloader(munis, "/sciclone/geograd/heather_data/netCDFs/", rank)

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write(str('Done with dataloader in rank: ') + str(rank) + "\n")  

    print('Done with dataloader in rank: ', rank)

    ######################################################
    # Set up DDP model and model utilities
    ######################################################    

    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    ddp_model = DDP(model, process_group = model_group)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.01)   

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write(str('Done with model setup in rank: ') + str(rank) + "\n")

    ######################################################
    # Train!
    ######################################################

    for epoch in range(0, 5):

        train_tracker = AverageMeter()

        for (input, target) in data.data:

            optimizer.zero_grad()

            input = input.permute(0,3,1,2)
            target = target.view(-1, 1)

            output = ddp_model(input)
            print("Prediction: ", output, "  True: ", target)

            loss = criterion(output, target)
            train_tracker.update(loss.item())

            loss.backward()
            optimizer.step()

            print("Done with iteration in epoch ", epoch)

        df = remote_method(Evaluator.collect_losses, eval_rref, train_tracker.avg, epoch)

    # """ whoop whoop """
    # rpc.shutdown()


if __name__ == "__main__":


    os.environ['TP_SOCKET_IFNAME'] = "ib0"
    os.environ['GLOO_SOCKET_IFNAME'] = "ib0"

    ###########################################################################
    # 1) Initialize the main process group
    # 2) Initialize a second group because Rank 0's only job is to aggregate 
    # losses and print updates, therefore, it does not particiapte in training 
    # which means we have to create a second process group that contains 
    # only the ranks above 0 that actually participate in training
    ###########################################################################
    dist.init_process_group(backend = "gloo", timeout = datetime.timedelta(0, 5000))

    model_group = dist.new_group(ranks = [i for i in range(1, int(os.environ['WORLD_SIZE']))])    

    ###########################################################################
    # Make the folder for the run's stats & log files
    ###########################################################################
    if dist.get_rank() == 0:
        os.mkdir(config.records_dir)

    ###########################################################################
    # Initialize RPC and the evaluator on all ranks
    ###########################################################################
    rpc.init_rpc(f"worker_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

    eval = Evaluator()
    eval_rref = eval.evaluator_rref

    ###########################################################################
    # Run the trainer on every rank but 0
    ###########################################################################
    if dist.get_rank() != 0:
        main(int(os.environ["RANK"]), int(os.environ['WORLD_SIZE']), model_group)

    """ whoop whoop """
    rpc.shutdown()