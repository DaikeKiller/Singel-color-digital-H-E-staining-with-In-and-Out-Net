import torch
import torch.nn as nn


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SubDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layer = CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2)
            layers.append(layer)
            in_channels = feature

        layers.append(
            nn.Conv2d(features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        temp = self.initial(input)
        out = self.model(temp)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.D1 = SubDiscriminator(in_channels=in_channels, features=features)
        self.D2 = SubDiscriminator(in_channels=in_channels, features=features)
        self.D3 = SubDiscriminator(in_channels=in_channels, features=features)

    def forward(self, x, y):
        im_H = y[0]
        im_E = y[1]
        im_HE = y[2]
        out1 = self.D1(x, im_H)
        out2 = self.D2(x, im_E)
        out3 = self.D3(x, im_HE)
        return out1, out2, out3


def test():
    x = torch.randn([1, 3, 256, 256])
    y1 = torch.randn([1, 3, 256, 256])
    y2 = torch.randn([1, 3, 256, 256])
    y3 = torch.randn([1, 3, 256, 256])
    model = Discriminator()
    out = model(x, [y1, y2, y3])
    print(model)
    print(out[0].shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


if __name__ == "__main__":
    test()