import torch.nn as nn
from .mainsubsidary import MainSubConv2d

class NIN(nn.Module):
    def __init__(self, prune_layer):
        super(NIN, self).__init__()
        if prune_layer is None:
            self.main_or_sub_string = [True for i in range(7)]
        else:
            self.main_or_sub_string = [False if i<=prune_layer else True for i in range(7)]
        self.xnor = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(192, 160, kernel_size=1, stride=1, padding=0, main_or_sub=self.main_or_sub_string[0]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(160, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(160,  96, kernel_size=1, stride=1, padding=0,main_or_sub=self.main_or_sub_string[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5, main_or_sub=self.main_or_sub_string[2]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(192, 192, kernel_size=1, stride=1, padding=0, main_or_sub=self.main_or_sub_string[3]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(192, 192, kernel_size=1, stride=1, padding=0, main_or_sub=self.main_or_sub_string[4]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5, main_or_sub=self.main_or_sub_string[5]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MainSubConv2d(192, 192, kernel_size=1, stride=1, padding=0, main_or_sub=self.main_or_sub_string[6]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
            )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x