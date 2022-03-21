import torch.optim as optim

class OptSelector:
    
    # lr, momentum, dampening, weight_decay, nesterov
    sgd_param = (optim.SGD, [1e-5, 0.9, 0, 1e-4, True])
    # lr, lambd, alpha, t0, weight_decay
    asgd_param = (optim.ASGD, [1e-5, 1e-4, 0.75, 1e6, 1e-4])
    # lr
    adagrad_param = (optim.Adagrad, [1e-5])
    adam_param = (optim.Adam, [1e-5])
    adamMax_param = (optim.Adamax, [1e-5])

    basic_opts = {
        "SGD": sgd_param,
        "ASGD": asgd_param,
        "Adagrad": adagrad_param,
        "Adam": adam_param,
        "Adamax": adamMax_param
        
    }
    
    def __init__(self, params, opt_name, cfg) -> None:
        self.params = params
        self.opt_name = opt_name
        self.LR = getattr(cfg, "LEARNING_RATE")
        self.LOCAL_RANK = getattr(cfg, "_LOCAL_RANK")
        pass
    
    def get_optim(self):
        opt_info = self.basic_opts[self.opt_name]
        opt = opt_info[0](self.params, *opt_info[1])
        if self.LOCAL_RANK == 0:
            print(f"Optimizer [ {self.opt_name} ] loaded!")
        
        return opt