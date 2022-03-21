import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ..settings import configs, model
from ..utils import find_best_n_model

def ini_model():
    global model
    # Load model to gpu
    # Check if load specific model or load best model in model folder
    if configs.LOAD_MODEL:
        if configs.LOAD_BEST:
            configs.MODEL_NAME = find_best_n_model(configs._LOCAL_RANK)
        try:
            model.load_state_dict(torch.load(configs._MODEL_DIR + configs.MODEL_NAME, map_location=configs._DEVICE))
            configs._LOAD_SUCCESS = True

        except FileNotFoundError:
            if configs._LOCAL_RANK == 0:
                print(f"[\"{configs.MODEL_NAME}\"] Model not found! Fall back to untrained model.\n")
            configs._LOAD_SUCCESS = False
        except IsADirectoryError:
            if configs._LOCAL_RANK == 0:
                print(f"IsADirectoryError! Fall back to untrained model.\n")
            configs._LOAD_SUCCESS = False
            
    # Move loaded model with parameters to gpus
    # Then warp with DDP, reducer will be constructed too.
    model.to(configs._DEVICE)
    if configs.DDP_ON:
        model = DDP(model, device_ids=[configs._LOCAL_RANK], output_device=configs._LOCAL_RANK)
    
    return model
    