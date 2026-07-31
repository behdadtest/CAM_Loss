from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from src.conv_adapter import ConvAdapter


class BasicBlockResidualParallelAdapter(nn.Module):

    _REQUIRED_ATTRS = ("conv1", "bn1", "relu", "conv2", "bn2", "downsample", "stride")

    def __init__(
        self,
        block: nn.Module,
        gamma: int = 4,
        kernel_size: int = 3,
    ):
        super().__init__()

        missing = [a for a in self._REQUIRED_ATTRS if not hasattr(block, a)]
        if missing:
            raise TypeError(
                f"block is missing expected BasicBlock attributes: {missing}. "
                f"BasicBlockResidualParallelAdapter only supports the standard "
                f"torchvision ResNet BasicBlock."
            )

        self.block = block

        in_channels = block.conv1.in_channels
        out_channels = block.conv2.out_channels
        stride = self._normalize_stride(block.stride)

        if in_channels % gamma != 0:
            raise ValueError(
                f"in_channels={in_channels} must be divisible by gamma={gamma}"
            )
        width = in_channels // gamma

        self.adapter = ConvAdapter(
            inplanes=in_channels,
            outplanes=out_channels,
            width=width,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            stride=stride,
            groups=width,
            dilation=1,
            norm_layer=None,
            act_layer=nn.ReLU,
        )

    @staticmethod
    def _normalize_stride(stride: Union[int, tuple]) -> int:
        if isinstance(stride, (tuple, list)):
            return int(stride[0])
        return int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)

        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        delta = self.adapter(x)

        out = out + identity + delta
        out = self.block.relu(out)

        return out


class ResNet18CAMWithAdapter(nn.Module):
    """
    ResNet18 backbone (optionally frozen, pretrained) with a Conv-Adapter
    attached in parallel to every residual block, plus a CAM head
    (1x1 conv -> num_classes, then GAP) — همون رفتار ResNet18CAM ساده،
    ولی با adapter. خروجی: (logits, cams) درست مثل مدل قبلی.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        gamma: int = 4,
        adapter_kernel_size: int = 3,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for parameter in base.parameters():
                parameter.requires_grad = False

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = self._add_adapters(base.layer1, gamma, adapter_kernel_size)
        self.layer2 = self._add_adapters(base.layer2, gamma, adapter_kernel_size)
        self.layer3 = self._add_adapters(base.layer3, gamma, adapter_kernel_size)
        self.layer4 = self._add_adapters(base.layer4, gamma, adapter_kernel_size)

        feat_channels = base.fc.in_features  # 512 for resnet18

        # CAM head: همیشه trainable، چون کلاس‌های جدید داریم و از صفر شروع می‌شه.
        self.cam_conv = nn.Conv2d(
            in_channels=feat_channels,
            out_channels=num_classes,
            kernel_size=1,
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    @staticmethod
    def _add_adapters(
        layer: nn.Sequential,
        gamma: int,
        kernel_size: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            *[
                BasicBlockResidualParallelAdapter(
                    block=block,
                    gamma=gamma,
                    kernel_size=kernel_size,
                )
                for block in layer
            ]
        )

    def train(self, mode: bool = True) -> "ResNet18CAMWithAdapter":
        super().train(mode)

        if mode and self.freeze_backbone:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    is_frozen = all(
                        not parameter.requires_grad
                        for parameter in module.parameters()
                    )
                    if is_frozen:
                        module.eval()

        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        features = self.layer4(x)

        cams = self.cam_conv(features)
        cams = F.relu(cams)

        logits = self.gap(cams).flatten(1)

        return logits, cams