#!/usr/bin/env python3
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import models
import datetime
import argparse
import os
import gc

from dataloader import *
from averager import *
from utils import *

from config import get_config

config, _ = get_config()




@record
def main(rank, model_group, imagery_list, worker_map):

    ######################################################
    # Load rank data
    ######################################################    
    munis = get_munis(imagery_list, rank, worker_map)

    data = Dataloader(munis, config.imagery_dir, rank)

    with open(config.log_name, "a") as f:
        f.write(str('Done with dataloader in rank: ') + str(rank) + "\n")  

    model_group.barrier()

    ######################################################
    # Set up DDP model and model utilities
    ######################################################    

    model = models.resnet18(pretrained = True)
    model.fc = torch.nn.Linear(512, 1)
    ddp_model = DDP(model, process_group = model_group)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = config.lr)   

    with open(config.log_name, "a") as f:
        f.write(str('Done with model setup in rank: ') + str(rank) + "\n")

    model_group.barrier()

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

            if config.use_rpc:
                df = remote_method(Evaluator.collect_losses, eval_rref, train_tracker.avg, epoch)
            else:
                epoch_folder = os.path.join(config.records_dir, "epochs", str(epoch))
                fname = f"{epoch_folder}/train_{str(rank)}.txt"
                with open(fname, "w") as f:
                    f.write(str(train_tracker.avg))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            

        mname = config.models_dir + "model_epoch" + str(epoch) + ".torch"

        if rank == 1:

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': ddp_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                    },  mname)

        model_group.barrier()

        ######################################################
        # Valdate!
        ######################################################

        weights = load_ddp_state(ddp_model.state_dict())
        model.load_state_dict(weights)
        model.eval()

        del weights
        gc.collect()

        with torch.no_grad():

            for (input, target) in data.val_data:

                optimizer.zero_grad()

                input = input.permute(0,3,1,2)
                target = target.view(-1, 1)

                output = model(input)

                loss = criterion(output, target).item()
                val_tracker.update(loss)

                # Currently no use_rpc option here yet!!
                epoch_folder = os.path.join(config.records_dir, "epochs", str(epoch))
                fname = f"{epoch_folder}/val_{str(rank)}.txt"
                with open(fname, "w") as f:
                    f.write(str(val_tracker.avg))



if __name__ == "__main__":

    os.environ['TP_SOCKET_IFNAME'] = "ib0"
    os.environ['GLOO_SOCKET_IFNAME'] = "ib0"


    ###########################################################################
    # Parse Input Agruments
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument("ppn")
    parser.add_argument("nodes")
    args = parser.parse_args()


    ###########################################################################
    # Initialize the main process group
    ###########################################################################
    dist.init_process_group(backend = "gloo", timeout = datetime.timedelta(0, 5000))
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])


    ###########################################################################
    # Make the folder for the run's stats & log files
    ###########################################################################
    if dist.get_rank() == 0:
        os.mkdir(config.records_dir)
        os.mkdir(os.path.join(config.records_dir, "epochs"))
        for i in range(config.epochs):
            os.mkdir(os.path.join(config.records_dir, "epochs", str(i)))    
        os.mkdir(os.path.join(config.records_dir, "models"))
            
    ###########################################################################
    # Get the sorted imagery list and make the worker -> imagery list index map
    ###########################################################################
    imagery_list, workers = organize_data(config.imagery_dir, int(args.ppn), int(args.nodes))
    worker_map = {w:i for w,i in zip(workers, [i for i in range(len(workers))])}


    if dist.get_rank() == 0:
        print(imagery_list)
        print(workers)


    ###########################################################################
    # 2) Initialize a second group for only the nodes participating in training
    ###########################################################################
    model_group = dist.new_group(ranks = workers, timeout = datetime.timedelta(0, 5000))    


    ###########################################################################
    # Initialize RPC and the evaluator on all ranks (is using RPC)
    ###########################################################################
    if config.use_rpc:

        rpc.init_rpc(f"worker_{rank}", rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

        eval = Evaluator()
        eval_rref = eval.evaluator_rref


    ###########################################################################
    # Run the trainer on every rank but 0
    ########################################################################### 
    last_rank = world_size - 1

    if dist.get_rank() in workers:
        main(rank, model_group, imagery_list, worker_map)

    elif dist.get_rank() == last_rank:
        run_averager(len(workers))

    else:
        pass


    """ whoop whoop """
    if config.use_rpc:
        rpc.shutdown()


    