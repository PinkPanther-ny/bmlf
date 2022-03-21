import random
from typing import Tuple

import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from ..settings.configs import configs
import math


class Preprocessor:
    
    # Official data augmentation for CIFAR10 dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode="constant"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self,
                 trans_train=transform_train,
                 trans_test=transform_test) -> None:
        self.trans_train = trans_train
        self.trans_test = trans_test
        self.loader = None

    def get_loader(self)->Tuple[DataLoader, DataLoader]:

        if self.loader is not None:
            return self.loader

        data_dir = configs._DATA_DIR
        batch_size = configs.BATCH_SIZE
        n_workers = configs.NUM_WORKERS

        train_set = CIFAR10(root=data_dir, train=True,
                            download=False, transform=self.transform_train)
        test_set = CIFAR10(root=data_dir, train=False,
                           download=False, transform=self.transform_test)
        if configs.DDP_ON:
            train_sampler = DistributedSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      sampler=train_sampler)
        else:
            train_loader = DataLoader(train_set, batch_size=batch_size,
                                      shuffle=True, num_workers=n_workers)

        # Test with whole test set, no need for distributed sampler
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=n_workers)

        # Return two iterables which contain data in blocks, block size equals to batch size
        return train_loader, test_loader

    def visualize_data(self, n=9, train=True, rand=True)->None:
        loader = self.get_loader()[int(not train)]
        wid = int(math.floor(math.sqrt(n)))
        if wid * wid < n:
            wid += 1
        fig = plt.figure(figsize=(2 * wid, 2 * wid))
        print(wid)

        for i in range(n):
            if rand:
                index = random.randint(0, len(loader.dataset) - 1)
            else:
                index = i

            # Add subplot to corresponding position
            fig.add_subplot(wid, wid, i + 1)
            plt.imshow((np.transpose(loader.dataset[index][0].numpy(), (1, 2, 0))))
            plt.axis('off')
            plt.title(configs._CLASSES[loader.dataset[index][1]])

        fig.show()
