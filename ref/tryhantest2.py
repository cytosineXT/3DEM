# torchrun --nproc_per_node=2 NNtrain_mulDDP2.py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from tqdm import tqdm
from net.jxtnet_upConv4 import MeshAutoencoder
import torch.utils.data.dataloader as DataLoader
import os
import sys
import re
import matplotlib.pyplot as plt
from pathlib import Path
from net.utils import increment_path, meshRCSDataset, get_logger, get_model_memory, psnr, ssim, find_matching_files, process_files
from NNvalfast import plotRCS2, plot2DRCS
import signal
import datetime

def setup(rank, world_size):
    # nprocs=world_size
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12356'
    print(f"Initializing process group for rank {rank}, worldsize{world_size}")
    dist.init_process_group(backend="nccl",init_method="file:///tmp/torch_sharedfile" ,rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=1)) #草 卡在这步了
    # dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=20)) #草 卡在这步了
    # dist.init_process_group(backend="mpi", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=20)) #草 卡在这步了
    print(f"Process group initialized for rank {rank}")
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    print("Cleaning up process group")

def signal_handler(sig, frame):
    print(f"Received signal {sig}, exiting...")
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main(rank, world_size):
    print(f"Starting main for rank {rank}")
# def main():
    # rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    setup(rank, world_size)
    
    print(1)
    
    print(f"Process {rank} completed training")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
