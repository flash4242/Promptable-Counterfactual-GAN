import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """Very small residual block with BN and LeakyReLU.
    The residual path is scaled before adding to the identity to stabilize early training.
    """
    def __init__(self, channels, activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = activation
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        #out=self.act(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # scale the residual to keep updates conservative at first
        return x + 0.1 * out


class ResidualGenerator(nn.Module):
    """Residual generator for counterfactuals.
    - Returns `residual` (do `x_cf = torch.clamp(x + residual, -1, 1)` in training/eval).
    - Default is conservative: base_ch=32, n_resblocks=2, residual_scaling=0.1.
    - No tanh on the output (keeps gradients stable). Clamping happens externally.
    """

    def __init__(self, img_shape=(1, 28, 28), num_classes=10, base_ch=64, n_resblocks=6,
                 residual_scaling=0.1):
        super().__init__()
        C, H, W = img_shape
        self.embed = nn.Embedding(num_classes, H * W)

        # entry
        self.conv_in = nn.Conv2d(C + 2, base_ch, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # small stack of residual blocks (cheap but effective)
        blocks = []
        for _ in range(n_resblocks):
            blocks.append(_ResBlock(base_ch, activation=self.act))
        self.resblocks = nn.Sequential(*blocks)

        # a small bottleneck conv and the final output conv
        self.conv_mid = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)

        # global scaling (keeps outputs conservative; adjust during experiments)
        self.residual_scaling = residual_scaling

        self._init_weights()


    def _init_weights(self):
        # Kaiming init for convs, normal small init for embedding
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, x, target, mask=None):
        B, C, H, W = x.shape
        y_map = self.embed(target).view(B, 1, H, W).to(x.dtype).to(x.device)
        inp = torch.cat([x, y_map, mask], dim=1)

        h = self.act(self.conv_in(inp))
        h = self.resblocks(h)
        h = self.act(self.conv_mid(h))

        raw_residual = self.conv_out(h) * self.residual_scaling
        if mask is not None:
            masked_residual = raw_residual * mask  # only modify allowed pixels
        else:
            print("WARNING: no mask provided to generator, modifying entire image")
            masked_residual = raw_residual
        return raw_residual, masked_residual
