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


@record
def run_averager(world_size):

    cur_epoch = 0

    while cur_epoch < config.epochs:

        # Get the number of files in the epoch folder
        epoch_folder = os.path.join(config.records_dir, "epochs", str(cur_epoch))
        num_files = len(os.listdir(epoch_folder))

        # world_size - 1 * 2 would indicate there is both a train and val file in the folder for every rank
        if num_files == ((world_size - 1) * 2):

            train_avg, val_avg = []

            # For each of the files, open it and save the average to the appropriate list
            for r in os.listdir(epoch_folder):

                if "train" in r:
                    with open(epoch_folder + "/" + r, "r") as f:
                        train_avg.append(float(f.read()))
                        
                elif "val" in r:
                    with open(epoch_folder + "/" + r, "r") as f:
                        val_avg.append(float(f.read()))                        

            with open(config.log_name, "a") as f:
                f.write("Epoch: " + str(cur_epoch) + "  Training Accuracy: " + str(np.average(train_avg)) + "\n"  + "          Training Accuracy: " + str(np.average(val_avg)) + "\n\n")

            cur_epoch += 1

            # If we've reached the total number of epochs, end
            if cur_epoch == config.epochs:
                return

        # Otherwise, wait for a second and check again
        else:
            time.sleep(2)


@record
def load_ddp_state(model, ddp_model):

    # DDP Model has slightly different key name so make a dictionary map of model key names -> ddp model key names
    key_map = {}
    for m_k, ddpm_k in zip(model.state_dict().keys(), ddp_model.state_dict().keys()):
        key_map[m_k] = ddpm_k

    # Load the optimized weights into the non-ddp model
    for k in model.state_dict().keys():
        ddp_key = key_map[k]
        k = ddp_model.state_dict()[ddp_key]

@record
def main(rank, world_size, model_group):

    ######################################################
    # Load rank data
    ######################################################    
    munis = get_munis(rank, world_size)

    # with open(config.log_name, "a") as f:
    #     f.write("MUNIS: " + str(munis) + "  RANK: " + str(rank) + "\n")

    data = Dataloader(munis, "/sciclone/geograd/heather_data/netCDFs/", rank)

    with open(config.log_name, "a") as f:
        f.write(str('Done with dataloader in rank: ') + str(rank) + "\n")  

    # print('Done with dataloader in rank: ', rank)

    ######################################################
    # Set up DDP model and model utilities
    ######################################################    

    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    ddp_model = DDP(model, process_group = model_group)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.01)   

    with open(config.log_name, "a") as f:
        f.write(str('Done with model setup in rank: ') + str(rank) + "\n")

    for epoch in range(0, config.epochs):

        train_tracker, val_tracker = AverageMeter(), AverageMeter()

        ######################################################
        # Train!
        ######################################################
        for (input, target) in data.train_data:

            optimizer.zero_grad()

            input = input.permute(0,3,1,2)
            target = target.view(-1, 1)

            output = ddp_model(input)

            loss = criterion(output, target)
            train_tracker.update(loss.item())

            loss.backward()
            optimizer.step()

            # print("Done with iteration in epoch ", epoch)

        if config.use_rpc:
            df = remote_method(Evaluator.collect_losses, eval_rref, train_tracker.avg, epoch)
        else:
            epoch_folder = os.path.join(config.records_dir, "epochs", "train" + str(epoch))
            fname = f"{epoch_folder}/{str(rank)}.txt"
            with open(fname, "w") as f:
                f.write(str(train_tracker.avg))


        ######################################################
        # Valdate!
        ######################################################
        model = load_ddp_state(model)

        for (input, target) in data.val_data:

            optimizer.zero_grad()

            input = input.permute(0,3,1,2)
            target = target.view(-1, 1)

            output = ddp_model(input)

            loss = criterion(output, target)
            val_tracker.update(loss.item())

            # Currently no use_rpc option here yet!!
            epoch_folder = os.path.join(config.records_dir, "epochs", "val" + str(epoch))
            fname = f"{epoch_folder}/{str(rank)}.txt"
            with open(fname, "w") as f:
                f.write(str(train_tracker.avg))



def get_workers(rank, world_size):

    """
    - Get the number of municipalities/netCDF files in order to make another process group that only includes the number of process that we need
    - We don't want to use all of the processors available on a node in order to stay below memory limits, so we will only use the first 32 processors on a node

        Need to add one because rank 0 only averages
    """

    num_munis = len(os.listdir(config.imagery_dir))

    ppn_usable = config.ppn_usable
    ppn = config.ppn

    if ppn == ppn_usable:
        workers = [i for i in range(1, num_munis)]

    else:
        m = 1
        workers = []
        for i in range(1, world_size):
            if (i - (ppn * m)) % ppn < ppn_usable:
                workers.append(i)

        workers = workers[0:num_munis]

    if rank == 0:
        with open(config.log_name, "a") as f:
            f.write("WORkING RANKS: " + str(workers) + "\n")

    return workers


if __name__ == "__main__":

    os.environ['TP_SOCKET_IFNAME'] = "ib0"
    os.environ['GLOO_SOCKET_IFNAME'] = "ib0"

    ###########################################################################
    # 1) Initialize the main process group
    ###########################################################################
    dist.init_process_group(backend = "gloo", timeout = datetime.timedelta(0, 5000))

    ###########################################################################
    # Make the folder for the run's stats & log files
    ###########################################################################
    if dist.get_rank() == 0:
        os.mkdir(config.records_dir)
        os.mkdir(os.path.join(config.records_dir, "epochs"))
        for i in range(config.epochs):
            os.mkdir(os.path.join(config.records_dir, "epochs", str(i)))    

    ###########################################################################
    # 2) Initialize a second group because Rank 0's only job is to aggregate 
    # losses and print updates, therefore, it does not particiapte in training 
    # which means we have to create a second process group that contains 
    # only the ranks above 0 that actually participate in training
    ###########################################################################
    workers = get_workers(dist.get_rank(), dist.get_world_size())
    model_group = dist.new_group(ranks = workers)    

    # model_group = dist.new_group(ranks = [i for i in range(1, int(os.environ['WORLD_SIZE']))])    

    ###########################################################################
    # Initialize RPC and the evaluator on all ranks
    ###########################################################################
    if config.use_rpc:
        rpc.init_rpc(f"worker_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

        eval = Evaluator()
        eval_rref = eval.evaluator_rref

    ###########################################################################
    # Run the trainer on every rank but 0
    ###########################################################################
    if dist.get_rank() != 0:
        main(int(os.environ["RANK"]), int(os.environ['WORLD_SIZE']), model_group)
    elif dist.get_rank() in workers:
        run_averager(int(os.environ['WORLD_SIZE']))
    else:
        pass


    """ whoop whoop """
    if config.use_rpc:
        rpc.shutdown()