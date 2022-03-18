import os
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3, 4, 5, 6, 7]))


class Config:
    def __init__(self, *dict_config) -> None:
        # ==============================================
        # GLOBAL SETTINGS
        self.DDP_ON: bool = True

        self.BATCH_SIZE: int = 512
        self.LEARNING_RATE: float = 1e-3
        self.LEARNING_RATE_UPDATE_EPOCH: int = 30
        self.LEARNING_RATE_UPDATE_RATE: float = 0.12
        self.LEARNING_RATE_END: float = 1e-5
        self.TOTAL_EPOCHS: int = 5000

        self.OPT_USE_ADAM: bool = True

        self.LOAD_MODEL: bool = True
        self.MODEL_NAME: str = "10X92.pth"
        self.LOAD_BEST: bool = False
        self.EPOCH_TO_LOAD_BEST: int = 15

        self.MODEL_SAVE_THRESHOLD: float = 0

        self.NUM_WORKERS: int = 4
        self.N_LOGS_PER_EPOCH: int = 0

        # ==============================================
        # SPECIAL SETTINGS
        self.EPOCHS_PER_EVAL: int = 2

        self.ADAM_SGD_SWITCH: bool = True
        self.EPOCHS_PER_SWITCH: int = 30

        # ==============================================
        # NOT SUPPOSED TO BE CHANGED OFTEN

        self.WORKING_DIR: str = os.path.dirname(os.path.realpath(__file__))
        self.MODEL_DIR: str = self.WORKING_DIR + "/models_v100/"
        self.DATA_DIR: str = self.WORKING_DIR + '/data/'
        self.CLASSES: tuple = ('plane', 'car', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck')

        self.DEVICE = None
        self.LOCAL_RANK = None
        
        if len(dict_config) != 0:
            d = eval(dict_config[0])
            for k in dict(d):
                setattr(self, k, d[k])

    def reset_working_dir(self, main_dir):
        self.WORKING_DIR: str = os.path.dirname(os.path.realpath(main_dir))
        self.MODEL_DIR: str = self.WORKING_DIR + "/models_v100/"
        self.DATA_DIR: str = self.WORKING_DIR + '/data/'
                
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)

    def save(self, fn='/config.json'):
        with open(self.WORKING_DIR + fn, 'w') as fp:
            json.dump(str(self.__dict__), fp, indent=4)

    def load(self, fn='/config.json'):
        try:

            with open(self.WORKING_DIR + fn, 'r') as fp:
                dict_config = json.load(fp)
                d = eval(dict_config)
                for k in dict(d):
                    setattr(self, k, d[k])
            print("Config file loaded successfully!")
        except:
            print("Config file does not exits, use default value instead!")


configs = Config()
# configs.load()
# configs.save()
