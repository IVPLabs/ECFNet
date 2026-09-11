import torch
import torch.nn as nn
from typing import Tuple
import math
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class PConv(nn.Module):
    """Pinwheel-shaped convolution implemented with asymmetric zero padding."""

    def __init__(self, c1, c2, k, s=1):
        super().__init__()
        if c2 % 4 != 0:
            raise ValueError(f"PConv output channels must be divisible by 4, got {c2}")

        # Four asymmetric padding patterns capture the four pinwheel directions.
        paddings = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList(nn.ZeroPad2d(padding) for padding in paddings)
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.fuse = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        horizontal_0 = self.cw(self.pad[0](x))
        horizontal_1 = self.cw(self.pad[1](x))
        vertical_0 = self.ch(self.pad[2](x))
        vertical_1 = self.ch(self.pad[3](x))
        return self.fuse(torch.cat((horizontal_0, horizontal_1, vertical_0, vertical_1), dim=1))


class PixelUnshuffleDownsample(nn.Module):
    r"""Pixel-unshuffle (space-to-depth) followed by normalization and 1x1 channel reduction.

    The spatial resolution is reduced by 2 while the channels are rearranged
    from C to 4C, then projected to 2C.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv2d(in_channels=4*dim, out_channels=2*dim, kernel_size=1)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2


        x = self.norm(x)
        x = self.reduction(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2fPBasicBlock(nn.Module):
    """C2f-style block with a two-convolution residual path and a PConv path."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.25):
        super().__init__()
        hidden_channels = int(c2 * e)
        if hidden_channels % 4 != 0:
            raise ValueError(
                "C2fP hidden channels must be divisible by 4 for the PConv branches, "
                f"got {hidden_channels}"
            )

        # Two base branches are retained and separately refined by the residual
        # convolution path and the pinwheel-shaped convolution path.
        self.cv1 = Conv(c1, hidden_channels, 1, 1)
        self.cv2 = Conv(c1, hidden_channels, 1, 1)
        self.residual_path = Bottleneck(
            hidden_channels,
            hidden_channels,
            shortcut=shortcut,
            g=g,
            k=((3, 3), (3, 3)),
            e=1.0,
        )
        self.pconv_path = nn.Sequential(
            PConv(hidden_channels, hidden_channels, k=3, s=1),
            PConv(hidden_channels, hidden_channels, k=4, s=1),
        )
        self.fuse = Conv(4 * hidden_channels, c2, 1, 1)

    def forward(self, x):
        branch_1 = self.cv1(x)
        branch_2 = self.cv2(x)
        residual_features = self.residual_path(branch_1)
        pinwheel_features = self.pconv_path(branch_2)
        return self.fuse(
            torch.cat((branch_1, branch_2, residual_features, pinwheel_features), dim=1)
        )


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        # c2*0.25 c2*0.25 1 (3,3)

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )



class APKD(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        # APKD is not part of RBCN. Import lazily so training RBCN does not
        # require APKD-only optional dependencies such as einops.
        from model.apkd import APKDBlock

        self.apkd = APKDBlock(
            dim=dim*2,
            is_first=True,
            is_last=True,
            **kwargs
        )

        self.attn_process = nn.Sequential(
            nn.BatchNorm2d(2*dim),
            nn.Conv2d(2*dim, dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, teacher_feat, student_feat):
        feat = self.apkd(student_feat, teacher_feat)
        attn = self.attn_process(feat)
        return attn


class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.conv2(out)  #It will not significantly affect the performance of the model.
        # out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out

class PatchEmbedWithResNet(nn.Module):
    def __init__(self, img_size=1024, in_chans=3, embed_dim=96, norm_layer=None):
        """
        A ResNet-based implementation to downsample the image from 1024x1024 to 128x128.

        Args:
            img_size (int): Input image size. Default: 1024.
            in_chans (int): Number of input image channels. Default: 3.
            embed_dim (int): Number of output channels after embedding. Default: 96.
            norm_layer (nn.Module, optional): Normalization layer. Default: None.
        """
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.net = nn.Sequential(
            ResidualBlock(in_chans, 32, stride=1),  # 640 -> 640
            ResidualBlock(32, 64, stride=2),  # 640 -> 320
            ResidualBlock(64, 128, stride=2),  # 320 -> 160
            nn.BatchNorm2d(embed_dim),
        )

        # Normalization layer
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match expected size ({self.img_size}*{self.img_size})."

        # Downsample using the network
        x = self.net(x)  # [B, embed_dim, 128, 128]

        if self.norm is not None:
            x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        depth,
        input_resolution,
        num_layers=4,
        downsample=True,
        block_type="c2fp",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.block_type = block_type.lower()

        if self.block_type == "c2fp":
            block_factory = lambda: C2fPBasicBlock(
                c1=dim, c2=dim, shortcut=True, g=1, e=0.25
            )
        elif self.block_type == "c3k2":
            # Retain the previous C3k2 implementation as an ablation option.
            block_factory = lambda: C3k2(
                c1=dim, c2=dim, n=1, shortcut=True, g=1, e=0.25
            )
        else:
            raise ValueError(
                f"Unsupported RBCN block type '{block_type}'. Choose 'c2fp' or 'c3k2'."
            )

        self.blocks = nn.ModuleList(block_factory() for _ in range(depth))

        if downsample :
            self.downsample = PixelUnshuffleDownsample(input_resolution, dim=dim, norm_layer=nn.BatchNorm2d)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

