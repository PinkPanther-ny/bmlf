import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        def make_sequence(in_channels, increase_channels=False):

            middle_channels = in_channels*2 if increase_channels else in_channels
            first_stride = 2 if increase_channels else 1
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=middle_channels, dilation=1, kernel_size=3, padding=1, stride=first_stride),
                nn.BatchNorm2d(num_features=middle_channels, eps=0.000001, momentum=0.9),
                nn.ReLU(),
                nn.Conv2d(in_channels=middle_channels, out_channels=middle_channels, dilation=1, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(num_features=middle_channels, eps=0.000001, momentum=0.9),
                )

        def make_sequence_left(in_channels):

            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, dilation=1, kernel_size=1, padding=0, stride=2),
                nn.BatchNorm2d(num_features=in_channels*2, eps=0.000001, momentum=0.9),
                )

        self.activation_func = F.leaky_relu

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=64, eps=0.000001, momentum=0.9)
        # self.pool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.seq1_r = make_sequence(in_channels=64)

        self.seq2_l = make_sequence_left(64)
        self.seq2_r = make_sequence(in_channels=64, increase_channels=True)

        self.seq3_r = make_sequence(in_channels=128)

        self.seq4_l = make_sequence_left(128)
        self.seq4_r = make_sequence(in_channels=128, increase_channels=True)

        self.seq5_r = make_sequence(in_channels=256)

        self.seq6_l = make_sequence_left(256)
        self.seq6_r = make_sequence(in_channels=256, increase_channels=True)

        self.seq7_r = make_sequence(in_channels=512)

        self.seq_end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        # self.resnet18 = models.resnet50(num_classes=10)
        # self.resnet18._modules['conv1'] = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1)
        # self.resnet18._modules['maxpool'] = nn.MaxPool2d(kernel_size=1)

    def forward(self, x):
        # return self.resnet18(x)

        # Before the first block
        x = self.activation_func(self.bn1(self.conv1(x)))
        # x = self.pool(x)

        # block 1_1
        x = x + self.seq1_r(x)
        x = self.activation_func(x)
        # block 1_2
        x = x + self.seq1_r(x)
        x = self.activation_func(x)

        # block 2_1
        x = self.seq2_l(x) + self.seq2_r(x)
        x = self.activation_func(x)

        # block 3_1
        x = x + self.seq3_r(x)
        x = self.activation_func(x)

        # block 4
        x = self.seq4_l(x) + self.seq4_r(x)
        x = self.activation_func(x)

        # block 5
        x = x + self.seq5_r(x)
        x = self.activation_func(x)

        # block 6
        x = self.seq6_l(x) + self.seq6_r(x)
        x = self.activation_func(x)

        # block 7
        x = x + self.seq7_r(x)
        x = self.activation_func(x)

        # end
        x = self.seq_end(x)

        return x