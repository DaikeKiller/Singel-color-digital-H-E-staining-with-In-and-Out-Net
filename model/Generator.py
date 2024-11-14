import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu",
                 use_dropout=False):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,
                      padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2,
                                    padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act=="relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class SubGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,
                      padding=1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.down1 = Block(out_channels, out_channels*2, down=True, act="leaky",
                           use_dropout=False)
        self.down2 = Block(out_channels*2, out_channels * 4, down=True, act="leaky",
                           use_dropout=False)
        self.down3 = Block(out_channels*4, out_channels * 8, down=True, act="leaky",
                           use_dropout=False)
        self.down4 = Block(out_channels*8, out_channels * 8, down=True, act="leaky",
                           use_dropout=False)
        self.down5 = Block(out_channels*8, out_channels * 8, down=True, act="leaky",
                           use_dropout=False)
        self.down6 = Block(out_channels*8, out_channels * 8, down=True, act="leaky",
                           use_dropout=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels*8, out_channels*8, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        self.up1 = Block(out_channels*8, out_channels*8, down=False, act="relu",
                         use_dropout=True)
        self.up2 = Block(out_channels * 8*2, out_channels * 8, down=False, act="relu",
                         use_dropout=True)
        self.up3 = Block(out_channels * 8*2, out_channels * 8, down=False, act="relu",
                         use_dropout=True)
        self.up4 = Block(out_channels * 8*2, out_channels * 8, down=False, act="relu",
                         use_dropout=False)
        self.up5 = Block(out_channels * 8*2, out_channels * 4, down=False, act="relu",
                         use_dropout=False)
        self.up6 = Block(out_channels * 4*2, out_channels * 2, down=False, act="relu",
                         use_dropout=False)
        self.up7 = Block(out_channels * 2*2, out_channels, down=False, act="relu",
                         use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(out_channels*2, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # print(x.shape)
        d1 = self.initial_down(x)
        # print(d1.shape)
        d2 = self.down1(d1)
        # print(d2.shape)
        d3 = self.down2(d2)
        # print(d3.shape)
        d4 = self.down3(d3)
        # print(d4.shape)
        d5 = self.down4(d4)
        # print(d5.shape)
        d6 = self.down5(d5)
        # print(d6.shape)
        d7 = self.down6(d6)
        # print(d7.shape)
        bottleneck = self.bottleneck(d7)
        # print(bottleneck.shape)
        up1 = self.up1(bottleneck)
        # print(up1.shape)
        up2 = self.up2(torch.cat([up1, d7], dim=1))
        # print(up2.shape)
        up3 = self.up3(torch.cat([up2, d6], dim=1))
        # print(up3.shape)
        up4 = self.up4(torch.cat([up3, d5], dim=1))
        # print(up4.shape)
        up5 = self.up5(torch.cat([up4, d4], dim=1))
        # print(up5.shape)
        up6 = self.up6(torch.cat([up5, d3], dim=1))
        # print(up6.shape)
        up7 = self.up7(torch.cat([up6, d2], dim=1))
        # print(up7.shape)
        return self.final_up(torch.cat([up7, d1], dim=1))


class ConnectLayer(nn.Module):
    def __init__(self):
        super().__init__()
        V = torch.ones((3, 2)) / 2

        # self.R_H = torch.tanh(nn.Parameter(V[0, 0]))
        # self.G_H = torch.tanh(nn.Parameter(V[1, 0]))
        # self.B_H = torch.tanh(nn.Parameter(V[2, 0]))
        # self.R_E = torch.tanh(nn.Parameter(V[0, 1]))
        # self.G_E = torch.tanh(nn.Parameter(V[1, 1]))
        # self.B_E = torch.tanh(nn.Parameter(V[2, 1]))
        self.R_H = nn.Parameter(V[0, 0])
        self.G_H = nn.Parameter(V[1, 0])
        self.B_H = nn.Parameter(V[2, 0])
        self.R_E = nn.Parameter(V[0, 1])
        self.G_E = nn.Parameter(V[1, 1])
        self.B_E = nn.Parameter(V[2, 1])
        # self.R_H = 0.5
        # self.G_H = 0.36
        # self.B_H = 0
        # self.R_E = 0
        # self.G_E = 0.55
        # self.B_E = 0.12

    def forward(self, x):
        im_H = torch.mean(x[0][:, :, :, :], dim=1)
        im_E = torch.mean(x[1][:, :, :, :], dim=1)
        im_H_syth = im_H * 0.5 + 0.5
        im_E_syth = im_E * 0.5 + 0.5
        R = 1 - torch.sigmoid(self.R_H) * im_H_syth - torch.sigmoid(self.R_E) * im_E_syth
        G = 1 - torch.sigmoid(self.G_H) * im_H_syth - torch.sigmoid(self.G_E) * im_E_syth
        B = 1 - torch.sigmoid(self.B_H) * im_H_syth - torch.sigmoid(self.B_E) * im_E_syth
        im_HE_syth = torch.stack((R, G, B), dim=1)
        return (im_HE_syth - 0.5) / 0.5


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        self.G1 = SubGenerator(in_channels=in_channels, out_channels=out_channels)
        self.G2 = SubGenerator(in_channels=in_channels, out_channels=out_channels)
        self.CL = ConnectLayer()
        # self.DeBlur = MedNet2()

    def forward(self, x):
        im_H = self.G1(x)
        im_E = self.G2(x)
        im_HE_mid = self.CL([im_H, im_E])
        # im_HE_list = self.DeBlur(im_HE_mid)
        return im_H, im_E, im_HE_mid


def test():
    x = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=3, out_channels=64)
    preds = model(x)
    print(model)
    print(preds[-1].shape)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)


if __name__ == "__main__":
    test()
