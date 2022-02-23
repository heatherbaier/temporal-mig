#!/usr/bin/env python3
import torch.distributed as dist
import socket
import pprint
import sys
import io
import os


if __name__ == "__main__":


    # hostname = socket.gethostname()
    # host_ip = socket.gethostbyname(hostname) 

    # os.environ['rdzv_endpoint'] = str(host_ip)


    # print(os.environ['MASTER_ADDR'])

    # os.system("qsub /sciclone/home20/hmbaier/tm/ipjob.sh")

    # print('we here yo!!')

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

    with io.StringIO() as buff:
        print("======================================================", file=buff)
        print(
            f"Environment variables set by the agent on PID {os.getpid()}:", file=buff
        )
        pprint.pprint(env_dict, stream=buff)
        print("======================================================", file=buff)
        print(buff.getvalue())
        sys.stdout.flush()

    dist.init_process_group(backend="gloo")
    dist.barrier()

    print(
        (
            f"On PID {os.getpid()}, after init process group, "
            f"rank={dist.get_rank()}, world_size = {dist.get_world_size()}\n"
        )
    )