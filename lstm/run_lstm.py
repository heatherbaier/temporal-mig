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
# from averager import *
from utils import *
from lstm import *

from lstm_config import get_config
config, _ = get_config()


@record
def main(rank, model_group, features_list, worker_map):

    ######################################################
    # Load rank data
    ######################################################    
    features_path = get_munis(features_list, rank, worker_map)

    data = Dataloader(features_path, rank)

    with open(config.log_name, "a") as f:
        f.write(str('Done with dataloader in rank: ') + str(rank) + "\n")  

    model_group.barrier()

    model = LSTM(input_size = 512,
                 hidden_size = 128,
                 output_size = 12)  
    ddp_model = DDP(model, process_group = model_group)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.001)      

    with open(config.log_name, "a") as f:
        f.write(str('Done with model setup in rank: ') + str(rank) + "\n")

    model_group.barrier()

    train_tracker, val_tracker = AverageMeter(), AverageMeter()

    for epoch in range(0, 1):

        train_tracker.reset()
        val_tracker.reset()

        ######################################################
        # Train!
        ######################################################
        for (input, target) in zip(data.x_train, data.y_train):

            optimizer.zero_grad()

            # input = input.permute(0,3,1,2)
            # target = target.view(-1, 1)

            output = ddp_model(input)

            loss = criterion(output, target)
            train_tracker.update(loss.item())

            model_group.barrier()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_folder = os.path.join(config.records_dir, "epochs", str(epoch))
        fname = f"{epoch_folder}/train_{str(rank)}.txt"
        with open(fname, "w") as f:
            f.write(str(train_tracker.avg))


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

            for (input, target) in zip(data.x_val, data.y_val):

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
    imagery_list, workers = organize_data(config.features_dir, int(args.ppn), int(args.nodes))
    worker_map = {w:i for w,i in zip(workers, [i for i in range(len(workers))])}

    if dist.get_rank() == 0:
        print(imagery_list)
        print(workers)

    ###########################################################################
    # 2) Initialize a second group for only the nodes participating in training
    ###########################################################################
    model_group = dist.new_group(ranks = workers, timeout = datetime.timedelta(0, 5000))    

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

