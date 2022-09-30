import torch
from torch import nn
from torchvision.ops import StochasticDepth


class SqueezeExciteNd(nn.Module):
    def __init__(self, num_channels, r=16):
        """num_channels: No of input channels
           r: By how much the num_channels should be reduced
        """
        super().__init__()
        num_channels_reduced = num_channels // r
        assert r <= num_channels, (r, num_channels)

        self.r = r
        # nn.AdaptiveAvgPool2d
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        batch_size, num_channels, *spatial = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1
                                           ).mean(dim=-1)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(
            squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        view_shape = (batch_size, num_channels) + (1,) * len(spatial)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(*view_shape))

        return output_tensor

class MBConvN(nn.Module):
  """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
  def __init__(self, n_in, n_out, expansion_factor,
               kernel_size=3, stride=1, r=24, p=0):
    super().__init__()

    padding = (kernel_size - 1) // 2
    expanded = expansion_factor * n_in
    self.skip_connection = (n_in == n_out) and (stride == 1)

    self.expand_pw = (nn.Identity() if (expansion_factor == 1) else
                      ConvNormActivation(n_in, expanded, kernel_size=1))
    self.depthwise = ConvNormActivation(expanded, expanded,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=expanded)
    self.se = SqueezeExciteNd(expanded, r=r)
    self.reduce_pw = ConvNormActivation(expanded, n_out, kernel_size=1, act=False)
    self.dropsample = StochasticDepth(p, mode='row')

  def forward(self, x):
    residual = x

    x = self.expand_pw(x)
    x = self.depthwise(x)
    x = self.se(x)
    x = self.reduce_pw(x)

    if self.skip_connection:
      x = self.dropsample(x)
      x = x + residual

    return x


class ConvNormActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, groups=1, act=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class MBConv1(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3, stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=1,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)


class MBConv6(MBConvN):
  def __init__(self, n_in, n_out, kernel_size=3, stride=1, r=24, p=0):
    super().__init__(n_in, n_out, expansion_factor=6,
                     kernel_size=kernel_size, stride=stride,
                     r=r, p=p)

class EfficientNet1d(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation(in_channels, 32, kernel_size=3, stride=2),
            MBConv1(32, 16, kernel_size=3, r=4, p=0.0),
            MBConv6(16, 24, kernel_size=3, stride=2, r=24, p=0.0125),
            MBConv6(24, 24, kernel_size=3, stride=1, r=24, p=0.025),
            MBConv6(24, 40, kernel_size=5, stride=2, r=24, p=0.0375),
            MBConv6(40, 240, kernel_size=5, stride=1, r=24, p=0.05),
            MBConv6(240, 80, kernel_size=3, stride=2, r=24, p=0.0625),
            MBConv6(80, 80, kernel_size=3, stride=1, r=24, p=0.075),
            MBConv6(80, 80, kernel_size=3, stride=1, r=24, p=0.0875),
            MBConv6(80, 112, kernel_size=5, stride=1, r=24, p=0.1),
            MBConv6(112, 112, kernel_size=5, stride=1, r=24, p=0.1125),
            MBConv6(112, 112, kernel_size=5, stride=1, r=24, p=0.125),
            MBConv6(112, 192, kernel_size=5, stride=2, r=24, p=0.1375),
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.15),
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.1625),
            MBConv6(192, 192, kernel_size=5, stride=1, r=24, p=0.175),
            MBConv6(192, 320, kernel_size=3, stride=1, r=24, p=0.1875),
            ConvNormActivation(320, 1280, kernel_size=1, stride=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x).squeeze(-1)
        y = self.classifier(x)
        return y


class ConvNormActivation2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1,
                 padding=0, groups=1, act=True):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,
                   out_channels,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   groups=groups,
                   bias=False),
            nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.SiLU() if act else nn.Identity()
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class wav2shape(nn.Module):
    def __init__(self, in_channels, bin_per_oct,outdim):
        super().__init__()
        self.features = nn.Sequential(
            ConvNormActivation2d(in_channels, 128, kernel_size=(bin_per_oct,8), padding="same"),
            nn.AvgPool2d(kernel_size=(1,8),padding="valid"),
            ConvNormActivation2d(128, 64, kernel_size=(bin_per_oct,4), padding="same"),
            ConvNormActivation2d(64, 64, kernel_size=(bin_per_oct,4), padding="same"),
            nn.AvgPool2d(kernel_size=(1,8),padding="valid"),
            ConvNormActivation2d(64, 8, kernel_size=(bin_per_oct,1), padding="same"),
            nn.Flatten(),
            nn.Linear(16,64),#not sure in channel
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True,
                           track_running_stats=True),
            nn.SiLU(),
            nn.Linear(64,outdim),#add nonneg kernel regulation
            #LINEAR ACTIVATION??
        )
    def forward(self, input_tensor):
        return self.block(input_tensor)