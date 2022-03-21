import os
import json

import torch


class Config:
    def __init__(self, *dict_config) -> None:
        # ==============================================
        # GLOBAL SETTINGS
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3, 4, 5, 6, 7]))
        
        self.DDP_ON: bool = True
        self.MIX_PRECISION: bool = True

        self.BATCH_SIZE: int = 512
        self.LEARNING_RATE: float = 1e-5
        self.TOTAL_EPOCHS: int = 5000

        self.LOAD_MODEL: bool = True
        self.MODEL_NAME: str = "92_35.pth"
        self.LOAD_BEST: bool = True
        self.N_LOGS_PER_EPOCH: int = 0

        # ==============================================
        # SPECIAL SETTINGS
        
        # Select in optim/load_opt and loss/load_loss
        self.OPT = "SGD"
        self.LOSS = "CrossEntropy"
        
        self.EPOCHS_PER_EVAL: int = 1
        self.NUM_WORKERS: int = 4
        self.MODEL_DIR_NAME: str = "/models_v100/"
        
        # ==============================================
        # Private
        self._WORKING_DIR: str = os.path.dirname(os.path.realpath(__file__))
        self._MODEL_DIR: str = self._WORKING_DIR + self.MODEL_DIR_NAME
        self._DATA_DIR: str = self._WORKING_DIR + '/data/'
        self._CLASSES: tuple = ('plane', 'car', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck')

        self._DEVICE = None
        self._LOCAL_RANK = None
        self._LOAD_SUCCESS: bool = False
        
        if self.DDP_ON:
            self._LOCAL_RANK = int(os.environ["LOCAL_RANK"])
        else:
            self._LOCAL_RANK = 0
        
        self._DEVICE = torch.device("cuda", self._LOCAL_RANK)
        
        if len(dict_config) != 0:
            d = eval(dict_config[0])
            for k in dict(d):
                setattr(self, k, d[k])

    def reset_working_dir(self, main_dir):
        self._WORKING_DIR: str = os.path.dirname(os.path.realpath(main_dir))
        self._MODEL_DIR: str = self._WORKING_DIR + self.MODEL_DIR_NAME
        self._DATA_DIR: str = self._WORKING_DIR + '/data/'
                
        if not os.path.exists(self._MODEL_DIR):
            os.makedirs(self._MODEL_DIR)

    def save(self, fn='/config.json'):
        with open(self._WORKING_DIR + fn, 'w') as fp:
            json.dump(str(self.__dict__), fp, indent=4)

    def load(self, fn='/config.json'):
        try:

            with open(self._WORKING_DIR + fn, 'r') as fp:
                dict_config = json.load(fp)
                d = eval(dict_config)
                for k in dict(d):
                    setattr(self, k, d[k])
            print("Config file loaded successfully!")
        except:
            print("Config file does not exits, use default value instead!")


configs = Config()
