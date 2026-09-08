"""
resnet_cifar.py

Standard CIFAR ResNet-32 backbone (He et al. 2016's CIFAR variant: 3 stages
of 5 BasicBlocks each, stem + no bottleneck, 31 conv layers + 1 FC = 32
weighted layers total), plus NormedLinear (cosine classifier, standard in
long-tailed recognition since Kang et al. 2020) and BCLModelCIFAR, the
two-branch BCL model adapted to this backbone.

NOTE ON RECONSTRUCTION: the BasicBlock/ResNet_Cifar backbone itself is the
standard, widely-used CIFAR ResNet-32 implementation (same structure as in
the BCL/ConCutMix public repos and the broader long-tailed-recognition
codebase lineage) -- only NormedLinear and BCLModelCIFAR were recovered
verbatim from past chat turns.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standard CIFAR ResNet-32 backbone
# ---------------------------------------------------------------------------
def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _PadShortcut(nn.Module):
    """
    Option-A shortcut (He et al.'s original CIFAR-ResNet): 2x2 stride
    average-pool for spatial downsampling, then zero-pad the channel
    dimension -- no extra learnable conv layer, so the total weighted-layer
    count stays exactly 6n+2 (32 for n=5) instead of gaining a 1x1 conv per
    downsampling stage.
    """

    def __init__(self, in_planes, planes, stride):
        super().__init__()
        self.stride = stride
        self.pad_channels = planes - in_planes

    def forward(self, x):
        if self.stride > 1:
            x = F.avg_pool2d(x, kernel_size=1, stride=self.stride)
        if self.pad_channels > 0:
            pad_before = self.pad_channels // 2
            pad_after = self.pad_channels - pad_before
            x = F.pad(x, (0, 0, 0, 0, pad_before, pad_after))
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet_Cifar(nn.Module):
    """3 stages x 5 BasicBlocks (n=5 -> 6n+2 = 32 layers), channels 16-32-64."""

    def __init__(self, block=BasicBlock, num_blocks=(5, 5, 5)):
        super().__init__()
        self.in_planes = 16
        self.conv1 = _conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = _PadShortcut(self.in_planes, planes * block.expansion, stride)
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)


def resnet32():
    return ResNet_Cifar(BasicBlock, (5, 5, 5))


# ---------------------------------------------------------------------------
class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 30

    def forward(self, x):
        return self.s * F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


# ---------------------------------------------------------------------------
# Two-branch BCL model for CIFAR -- same interface as resnext.BCLModel:
# forward(x) -> (feat_mlp, logits, centers_logits)
# ---------------------------------------------------------------------------
class BCLModelCIFAR(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = 128, use_norm: bool = True):
        super().__init__()
        self.encoder = resnet32()
        dim_in = self.encoder.feat_dim  # 64

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.head_fc = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.BatchNorm1d(dim_in), nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
        )
        self.fc = NormedLinear(dim_in, num_classes) if use_norm else nn.Linear(dim_in, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        feat_mlp = F.normalize(self.head(feat), dim=1)
        logits = self.fc(feat)
        centers_logits = F.normalize(self.head_fc(self.fc.weight.T), dim=1)
        return feat_mlp, logits, centers_logits


if __name__ == "__main__":
    # Quick structural check: 32 layers, correct output shapes.
    m = BCLModelCIFAR(num_classes=100)
    n_conv_layers = sum(1 for mod in m.encoder.modules() if isinstance(mod, nn.Conv2d))
    print(f"Conv2d layers in encoder: {n_conv_layers} (expect 1 stem + 3*5*2 = 31 -> 32 total incl. stem)")

    x = torch.randn(8, 3, 32, 32)
    feat_mlp, logits, centers = m(x)
    print("feat_mlp:", feat_mlp.shape, "logits:", logits.shape, "centers:", centers.shape)
    assert feat_mlp.shape == (8, 128)
    assert logits.shape == (8, 100)
    assert centers.shape == (100, 128)
    print("OK: shapes match expected BCLModel interface")
