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
def setup_model(model_group):

    print("About to setup model!")

    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 1)
    ddp_model = DDP(model, process_group = model_group)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr = 0.01)   

    return ddp_model, criterion, optimizer



@record
def setup_eval(rank, world_size):

    # pass

    # rpc.init_rpc(name = "evaluator", 
    #              rank = rank, 
    #              world_size = world_size,
    #              rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

    eval = Evaluator()
    eval_rref = eval.evaluator_rref


@record
def main(rank, world_size, model_group):

    # rpc.init_rpc(f"worker_{rank}", 
    #             rank = rank, 
    #             world_size = world_size,
    #             rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

    munis = get_munis(rank, world_size)

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write("MUNIS: " + str(munis) + "  RANK: " + str(rank) + "\n")

    data = Dataloader(munis, "/sciclone/geograd/heather_data/netCDFs/", rank)

    # model_group.barrier()

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write(str('Done with dataloader in rank: ') + str(rank) + "\n")  

    print('Done with dataloader in rank: ', rank)

    ddp_model, criterion, optimizer = setup_model(model_group)

    with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
        f.write(str('Done with model setup in rank: ') + str(rank) + "\n")

    for epoch in range(0, 5):

        for (input, target) in data.data:

            optimizer.zero_grad()

            input = input.permute(0,3,1,2)
            target = target.view(-1, 1)

            output = ddp_model(input)

    #         if str(rank) == "15":
    #             with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
    #                 f.write(str("15 IS DONE TO HERE") + "\n")

    #         output = ddp_model(input)

    #         # wait until all workers are done predicting so as not to trigger a RendexvousTimeoutError
    #         dist.barrier()

            loss = criterion(output, target)

            coords = remote_method(Evaluator.collect_losses, eval_rref, loss, epoch)

            print("LOSS: ", loss)

    #         # with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
    #         #     f.write("HERE \n")

    #         with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
    #             f.write("RANK: " + str(rank) + "  LOSS: " + str(loss) + "\n")

            loss.backward()
            optimizer.step()

            print("Done with iteration in epoch ", epoch)



    #         with open("/sciclone/home20/hmbaier/tm/log.txt", "a") as f:
    #             f.write(str(output) + " " + str(target) + "\n")



    # print("RANK: ", rank, munis)


if __name__ == "__main__":

    env_dict = {
        k: os.environ[k]
        for k in (
            "LOCAL_RANK",
            "RANK",
            "GROUP_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        )
    }


    dist.init_process_group(backend = "gloo", timeout = datetime.timedelta(0, 5000))

    # Rank 0's only job is to aggregate losses and print updates,
    # therefore, it does not particiapte in training whcih means we have to create a second process group
    # that contains only the ranks above 0 that actually participate in training
    model_group = dist.new_group(ranks = [i for i in range(1, int(os.environ['WORLD_SIZE']))])    
    print(model_group)

    print(
        (
            f"On PID {os.getpid()}, after init process group, "
            f"rank={dist.get_rank()}, world_size = {dist.get_world_size()}\n"
        )
    )

    if dist.get_rank() == 0:
        os.mkdir(config.records_dir)

    os.environ['TP_SOCKET_IFNAME'] = "ib0"
    os.environ['GLOO_SOCKET_IFNAME'] = "ib0"


    # Initialize RPC on all ranks
    rpc.init_rpc(f"worker_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

    # Set up the evaluator on all ranks
    eval = Evaluator()
    eval_rref = eval.evaluator_rref

    if dist.get_rank() != 0:

        main(int(os.environ["RANK"]), int(os.environ['WORLD_SIZE']), model_group)

    rpc.shutdown()

    # print(rdzv_handler)

