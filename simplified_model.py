import torch
from torch import nn

from model import Model


class SimplifiedModel(Model):
    """
    Simplified colorization model with half the amount of parameters, derived from the default model
    """
    def __init__(self, q_values: int, h: callable, h_inv: callable, criterion: callable):
        super(SimplifiedModel, self).__init__("simplified", q_values, h, h_inv, criterion)

        # Reduce the maximum depth of a layer to 2 and reduce the maximum number of channels to 416
        self.conv1 = self.conv_layer(2, 1, 64, stride=2)
        self.conv2 = self.conv_layer(2, 64, 128, stride=2)
        self.conv3 = self.conv_layer(2, 128, 256, stride=2)
        self.conv4 = self.conv_layer(2, 256, 416)
        self.conv5 = self.conv_layer(2, 416, 416, dilation=2)
        self.conv6 = self.conv_layer(2, 416, 416, dilation=2)
        self.conv7 = self.conv_layer(2, 416, 416)
        self.conv8 = self.conv_layer(2, 416, 256, stride=0.5, normalise=False)

        self.distribution = nn.Conv2d(256, q_values, 1)

        self.optimiser = torch.optim.Adam(self.parameters(), lr=3.16e-5, betas=(0.9, 0.99), weight_decay=1e-3)