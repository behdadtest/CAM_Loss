import torch.nn as nn
from torchvision import models


def convert_stride_to_dilation(layer: nn.Sequential, dilation: int = 2) -> None:
    """Turn a stride-2 ResNet stage into a dilated stride-1 stage, in place.

    torchvision's ``replace_stride_with_dilation`` raises NotImplementedError for
    BasicBlock (resnet18/34), but nothing about the maths prevents it: both convs
    in a BasicBlock are 3x3, so dilation and padding only need setting on the
    modules themselves. This mirrors what ``_make_layer(dilate=True)`` does for
    Bottleneck -- the first block loses its stride and keeps dilation 1, every
    later block picks the dilation up.

    Doubles the CAM resolution (7x7 -> 14x14 at 224px) for roughly 4x the cost of
    this one stage.
    """
    first = layer[0]
    first.conv1.stride = (1, 1)
    first.stride = 1
    if getattr(first, "downsample", None) is not None:
        first.downsample[0].stride = (1, 1)

    for block in list(layer)[1:]:
        for conv in (block.conv1, block.conv2):
            if conv.kernel_size == (3, 3):
                conv.dilation = (dilation, dilation)
                conv.padding = (dilation, dilation)


class ResNet18CAM(nn.Module):
    """ResNet18 with a 1x1 CAM head.

    ``logits = GAP(cam_conv(features))`` with **no** ReLU in between. This is the
    standard CAM formulation (Zhou et al., 2016): the logits are then exactly a
    linear classifier on the globally pooled features, and the CAM is the spatial
    decomposition of that same score.

    ReLU belongs in the loss and in the visualisation, never here. With a ReLU in
    front of the pooling, a class whose map goes fully negative is clamped to 0
    *and* receives exactly zero gradient -- it can never be predicted again. The
    background penalty pushes every class map negative over most of the image, so
    that state is not hypothetical.

    ``dilate_last_block`` replaces layer4's stride with dilation, doubling the CAM
    resolution (14x14 instead of 7x7 at 224px input) for roughly 4x the cost of
    the last block alone. Pretrained weights transfer unchanged.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dilate_last_block: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if dilate_last_block:
            convert_stride_to_dilation(base.layer4, dilation=2)

        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )

        self.cam_conv = nn.Conv2d(
            in_channels=base.fc.in_features,
            out_channels=num_classes,
            kernel_size=1,
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        features = self.backbone(x)

        cams = self.cam_conv(features)          # raw, signed -- ReLU lives downstream
        logits = self.gap(cams).flatten(1)

        return logits, cams
