import torch
import torch.nn as nn


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=not normalize
            )
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 64):
        super().__init__()
        f = num_features

        self.down1 = UNetDownBlock(in_channels, f, normalize=False)
        self.down2 = UNetDownBlock(f, f * 2, normalize=True)
        self.down3 = UNetDownBlock(f * 2, f * 4, normalize=True)
        self.down4 = UNetDownBlock(f * 4, f * 8, normalize=True)
        self.down5 = UNetDownBlock(f * 8, f * 8, normalize=True)
        self.down6 = UNetDownBlock(f * 8, f * 8, normalize=True)
        self.down7 = UNetDownBlock(f * 8, f * 8, normalize=True)
        self.down8 = nn.Sequential(
            nn.Conv2d(f * 8, f * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.up1 = UNetUpBlock(f * 8, f * 8, dropout=0.5)
        self.up2 = UNetUpBlock(f * 16, f * 8, dropout=0.5)
        self.up3 = UNetUpBlock(f * 16, f * 8, dropout=0.5)
        self.up4 = UNetUpBlock(f * 16, f * 8, dropout=0.0)
        self.up5 = UNetUpBlock(f * 16, f * 4, dropout=0.0)
        self.up6 = UNetUpBlock(f * 8, f * 2, dropout=0.0)
        self.up7 = UNetUpBlock(f * 4, f, dropout=0.0)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                f * 2,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True
            ),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        e6 = self.down6(e5)
        e7 = self.down7(e6)
        e8 = self.down8(e7)

        d1 = self.up1(e8)
        d1 = torch.cat([d1, e7], dim=1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e6], dim=1)

        d3 = self.up3(d2)
        d3 = torch.cat([d3, e5], dim=1)

        d4 = self.up4(d3)
        d4 = torch.cat([d4, e4], dim=1)

        d5 = self.up5(d4)
        d5 = torch.cat([d5, e3], dim=1)

        d6 = self.up6(d5)
        d6 = torch.cat([d6, e2], dim=1)

        d7 = self.up7(d6)
        d7 = torch.cat([d7, e1], dim=1)

        out = self.final(d7)
        return out


def init_weights(net: nn.Module, init_type: str = "normal", init_gain: float = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, "weight") and classname.find("BatchNorm2d") != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)
    return net


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator().to(device)
    init_weights(generator)
    dummy_input = torch.randn(2, 3, 256, 256, device=device)
    output = generator(dummy_input)
    print(f"[TEST] UNetGenerator: Input={dummy_input.shape} -> Output={output.shape}")
    num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"[TEST] Generator trainable parameters: {num_params:,}")
