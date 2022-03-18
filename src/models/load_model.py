# OMP_NUM_THREADS=2 python -m torch.distributed.run --nproc_per_node 4 90plus.py

import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from subprocess import call

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ..settings import configs, model
from ..utils import Timer, find_best3, eval_total

def ini_model():
    local_rank = configs.LOCAL_RANK
    global model
    # Load model to gpu
    device = torch.device("cuda", local_rank)
    configs.DEVICE = device
    # Check if load specific model or load best model in model folder
    if configs.LOAD_MODEL:
        if configs.LOAD_BEST:
            configs.MODEL_NAME = find_best3(local_rank)
        try:
            print(configs.MODEL_DIR + configs.MODEL_NAME)
            model.load_state_dict(torch.load(configs.MODEL_DIR + configs.MODEL_NAME))

        except FileNotFoundError or IsADirectoryError:
            print(f"{configs.MODEL_NAME} Model not found!")
    
    # Move loaded model with parameters to gpus
    # Then warp with DDP, reducer will be constructed too.
    model.to(device)
    if configs.DDP_ON:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    

    return model
    