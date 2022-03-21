import torch.nn as nn

CrossEntropy_param = (nn.CrossEntropyLoss, [])

basic_loss = {
    "CrossEntropy": CrossEntropy_param,
    
}
