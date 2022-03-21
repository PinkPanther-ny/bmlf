import torch.optim as optim
import torch.nn as nn


class LossSelector:
    
    cross_entropy_param = (nn.CrossEntropyLoss, [])


    basic_loss = {
        "CrossEntropy": cross_entropy_param,
        
    }

    
    def __init__(self, loss_name, cfg) -> None:
        self.loss_name = loss_name
        self.LOCAL_RANK = getattr(cfg, "_LOCAL_RANK")
        pass
    
    def get_loss(self):
        loss_info = self.basic_loss[self.loss_name]
        loss = loss_info[0](*loss_info[1])
        if self.LOCAL_RANK == 0:
            print(f"Loss function [ {self.loss_name} ] loaded!")
        
        return loss