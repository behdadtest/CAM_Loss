import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# def build_all_cams(features: torch.Tensor, classifier_weight: torch.Tensor) -> torch.Tensor:
#     """
#         cams: [B, num_classes, H, W]
#     """

#     cams = torch.einsum("kc,bchw->bkhw", classifier_weight, features)
#     cams = F.relu(cams)

#     return cams


class CAMMaskLoss(nn.Module):

    def __init__(
        self,
        outside_weight=1.0,
        inside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.outside_weight = outside_weight
        self.inside_weight = inside_weight  # unused; kept for call-site compatibility
        self.eps = eps

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:

        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            return cams.sum() * 0.0

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        # cams:       [B, K, Hc, Wc]
        # target_cam: [B, Hc, Wc]
        target_cam = cams[batch_idx, labels]

        # target_cam: [B, 1, Hc, Wc]
        target_cam = target_cam.unsqueeze(1)

        # Mask resizing
        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        inside = mask
        outside = 1.0 - mask

        # Only positive activation counts as "energy": negative logit
        # contribution isn't evidence for the class, so it shouldn't be
        # able to dilute or inflate the containment ratio either direction.
        positive = F.relu(target_cam)

        energy_outside = (positive * outside).sum(dim=(1, 2, 3))
        energy_total = positive.sum(dim=(1, 2, 3)) + self.eps

        containment = energy_outside / energy_total

        loss = self.outside_weight * containment

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)