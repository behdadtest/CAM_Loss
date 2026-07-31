import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18CAM(nn.Module):
 
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

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

        cams = self.cam_conv(features)       
        cams = F.relu(cams)                    

        logits = self.gap(cams).flatten(1)     

        return logits, cams