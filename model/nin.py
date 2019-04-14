import torch.nn as nn
from .mainsubsidary import MaskBinActiveConv2d

class NIN(nn.Module):
    def __init__(self, binact='0000000'):
        super(NIN, self).__init__()
        self.binact = binact
        self.xnor = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(192, 160, kernel_size=1, stride=1, padding=0, binact=self.binact[0]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(160, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(160,  96, kernel_size=1, stride=1, padding=0,binact=self.binact[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(96, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5, binact=self.binact[2]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0, binact=self.binact[3]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0, binact=self.binact[4]),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5, binact=self.binact[5]),
            nn.ReLU(inplace=True),

            nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=True),
            MaskBinActiveConv2d(192, 192, kernel_size=1, stride=1, padding=0, binact=self.binact[6]),
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