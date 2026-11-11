import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 6, num_features: int = 64):
        super().__init__()
        f = num_features

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(f, f * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(f * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(f * 4, f * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(f * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1, bias=True)

    def forward(self, sketch: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sketch, target], dim=1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        logits = self.final(x)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator = PatchGANDiscriminator().to(device)

    dummy_sketch = torch.randn(2, 3, 256, 256, device=device)
    dummy_real = torch.randn(2, 3, 256, 256, device=device)

    output = discriminator(dummy_sketch, dummy_real)
    print(f"[TEST] PatchGANDiscriminator: Inputs {dummy_sketch.shape} -> Output logits {output.shape}")
    num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"[TEST] Discriminator trainable parameters: {num_params:,}")
