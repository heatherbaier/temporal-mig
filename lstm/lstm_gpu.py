import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

from dataloader import *
from lstm import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"


def get_imagery_list(direc, rank, world_size):
    files = os.listdir(direc)
    files = [direc + i for i in files if "484" in i]
    files = np.array_split(files, world_size)
    return files[rank]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    

def main(rank, world_size):
    
    setup(rank, world_size)
        
    model = LSTM(input_size = 512,
                 hidden_size = 128,
                 output_size = 12).to(rank)
    ddp_model = DDP(model, device_ids=[rank])    
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.001)
            
    imagery = get_imagery_list("../../../../../heather_data/temporal_features/jsons/", rank, world_size)
    
    print("here! in rank: ", rank, " with world size: ", world_size, "  Num imagery: ", len(imagery))
    
    data = Dataloader(imagery, rank)
    
    train_tracker, val_tracker = AverageMeter(), AverageMeter()

    for epoch in range(0, 10):

        train_tracker.reset()
        val_tracker.reset()

        ######################################################
        # Train!
        ######################################################
        for (input, target) in zip(data.x_train, data.y_train):

            optimizer.zero_grad()
            
            input, target = input.to(rank), target.to(rank)
    
            output = ddp_model(input.to(rank), rank)

            loss = criterion(output, target)
                        
            train_tracker.update(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        with open(f"./records/{str(rank)}.txt", "w") as f:
            f.write(str(train_tracker.avg))
                
#         print("done with epoch in rank: ", rank)
            
        dist.barrier()
        
        if rank == 0:
            
            mean = []
            for i in os.listdir("./records/"):
                if "ipynb" not in i:
                    with open("./records/" + i, "r") as f:
                        mean.append(float(f.read()))
            
            print("EPOCH MEAN: ", np.average(mean))
            
            

#         epoch_folder = os.path.join(config.records_dir, "epochs", str(epoch))
#         fname = f"{epoch_folder}/train_{str(rank)}.txt"
#         with open(fname, "w") as f:
#             f.write(str(train_tracker.avg))


#         mname = config.models_dir + "model_epoch" + str(epoch) + ".torch"

#         if rank == 1:

#             torch.save({
#                         'epoch': epoch,
#                         'model_state_dict': ddp_model.state_dict(),
#                         'optimizer_state_dict': optimizer.state_dict(),
#                         'loss': criterion,
#                     },  mname)

#         model_group.barrier()

#         ######################################################
#         # Valdate!
#         ######################################################

#         weights = load_ddp_state(ddp_model.state_dict())
#         model.load_state_dict(weights)
#         model.eval()

#         del weights
#         gc.collect()

#         with torch.no_grad():

#             for (input, target) in zip(data.x_val, data.y_val):

#                 optimizer.zero_grad()

#                 input = input.permute(0,3,1,2)
#                 target = target.view(-1, 1)

#                 output = model(input)

#                 loss = criterion(output, target).item()
#                 val_tracker.update(loss)

#             # Currently no use_rpc option here yet!!
#             epoch_folder = os.path.join(config.records_dir, "epochs", str(epoch))
#             fname = f"{epoch_folder}/val_{str(rank)}.txt"
#             with open(fname, "w") as f:
#                 f.write(str(val_tracker.avg))
    
    
    
    


if __name__ == "__main__":
    
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    mp.spawn(main,
             args=(world_size,),
             nprocs=world_size,
             join=True)