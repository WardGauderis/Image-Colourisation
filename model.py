from torch import nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO normalize

        self.conv1 = self.conv_layer(2, 1, 64, stride=2)
        self.conv2 = self.conv_layer(2, 64, 128, stride=2)
        self.conv3 = self.conv_layer(3, 128, 256, stride=2)
        self.conv4 = self.conv_layer(3, 256, 512)
        self.conv5 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv6 = self.conv_layer(3, 512, 512, dilation=2)
        self.conv7 = self.conv_layer(3, 512, 512)
        self.conv8 = self.conv_layer(3, 512, 256, stride=0.5, normalize=False)

        self.distribution = nn.Conv2d(256, 313, 1)

        self.output = nn.Sequential(
            nn.Softmax(),  # TODO annealed_softmax, maybe dim=1
            nn.Conv2d(313, 2, 1),  # TODO argmax or whatever depending on training or inference
            nn.Upsample(scale_factor=4.0, mode="bilinear")  # TODO denormalize
        )

    @staticmethod
    def conv_layer(depth: int, in_channels: int, out_channels: int, stride: int | float = 1,
                   dilation: int = 1, normalize: bool = True) -> nn.Sequential:
        layers = []
        for i in range(depth):
            if stride == 0.5 and i == 0:
                layers.append(nn.ConvTranspose2d(in_channels,
                                                 out_channels,
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1))
            else:
                layers.append(nn.Conv2d(in_channels if i == 0 else out_channels,
                                        out_channels,
                                        3,
                                        padding=dilation,
                                        stride=stride if i == depth - 1 and stride != 0.5 else 1,
                                        dilation=dilation))
            layers.append(nn.ReLU())

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = Model()
    print(model)
