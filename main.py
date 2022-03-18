from src.models import ini_model
from src.preprocess import Preprocessor
from src.settings import configs
from src.utils import Timer, find_best3, eval_total

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



def train():
    if configs.DDP_ON:
        # DDP backend initialization
        configs.LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(configs.LOCAL_RANK)
        dist.init_process_group(backend='nccl')
    else:
        configs.LOCAL_RANK = 0

    model = ini_model()
    trainloader, testloader = Preprocessor().get_loader()
    
    # Start timer from here
    timer = Timer()
    timer.timeit()
    if configs.LOAD_MODEL and configs.LOCAL_RANK == 0:
        print(f"\nVerifying loaded model ({configs.MODEL_NAME})'s accuracy as its name suggested...")
        eval_total(model, testloader, timer)
        
    if configs.LOCAL_RANK == 0:
        print(f"Start training! Total {configs.TOTAL_EPOCHS} epochs.\n")
    
    return
    # Define loss function and optimizer for the following training process
    criterion = nn.CrossEntropyLoss()
    opt1 = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
    opt2 = optim.SGD(model.parameters(), lr=configs.LEARNING_RATE, momentum=0.90)
    opts = [opt2, opt1]
    opt_use_adam = configs.OPT_USE_ADAM
    
    # Mixed precision for speed up
    # https://zhuanlan.zhihu.com/p/165152789
    scalar = torch.cuda.amp.GradScaler()
    
    # ========================== Train =============================
    for epoch in range(configs.TOTAL_EPOCHS):
        if epoch%configs.LEARNING_RATE_UPDATE_EPOCH == configs.LEARNING_RATE_UPDATE_EPOCH - 1:
            configs.LEARNING_RATE *= configs.LEARNING_RATE_UPDATE_RATE
            if configs.LEARNING_RATE <= configs.LEARNING_RATE_END:
                configs.LEARNING_RATE = configs.LEARNING_RATE_END
            print(f"Learning rate updated to {configs.LEARNING_RATE}\n")
            opt1 = optim.Adam(model.parameters(), lr=configs.LEARNING_RATE)
            opt2 = optim.SGD(model.parameters(), lr=configs.LEARNING_RATE, momentum=0.90)
        
        # To avoid duplicated data sent to multi-gpu
        trainloader.sampler.set_epoch(epoch)
        
        # Just for removing worst models
        if epoch % configs.EPOCH_TO_LOAD_BEST == 0:
            remove_bad_models()

        # By my stategy, chose optimizer dynamically
        optimizer = opts[int(opt_use_adam)]
        
        # Counter for printing information during training
        count_log = 0 if configs.N_LOGS_PER_EPOCH == 0 else int(len(trainloader) / configs.N_LOGS_PER_EPOCH)
        
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # Speed up with half precision
            with torch.cuda.amp.autocast():
                # forward + backward + optimize
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
            
            # Scale the gradient
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            
            # print statistics
            running_loss += loss.item() * inputs.shape[0]
            
            if count_log != 0 and local_rank == 0 and i % count_log == count_log - 1:
                print(f'[{epoch + 1}(Epochs), {i + 1:5d}(batches)] loss: {running_loss / count_log:.3f}')
                running_loss = 0.0
                
        # Switch to another optimizer after some epochs
        if configs.ADAM_SGD_SWITCH:
            if epoch % configs.EPOCHS_PER_SWITCH == configs.EPOCHS_PER_SWITCH - 1:
                opt_use_adam = not opt_use_adam
                print(f"Epoch {epoch + 1}: Opt switched to {'Adam' if opt_use_adam else 'SGD'}")
        
        # Evaluate model on main GPU after some epochs
        if local_rank == 0 and epoch % configs.EPOCHS_PER_EVAL == configs.EPOCHS_PER_EVAL - 1:
            eval_total(model, testloader, timer, device, epoch)

    print(f'Training Finished! ({str(datetime.timedelta(seconds=int(timer.timeit())))})')


if __name__ == '__main__':
    try:
        # gc.collect()
        torch.cuda.empty_cache()
        configs.reset_working_dir(__file__)
        train()
    except KeyboardInterrupt:
        print("Exit!")

