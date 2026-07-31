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
        self.inside_weight = inside_weight
        self.eps = eps

    def normalize_cams(self, cams: torch.Tensor) -> torch.Tensor:

        b, k, h, w = cams.shape

        flat = cams.reshape(b, k, -1)

        max_abs = flat.abs().max(dim=2)[0].view(b, k, 1, 1)

        cams = cams / (max_abs + self.eps)

        return cams

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

        # cams = self.normalize_cams(cams)

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        # cams:       [B, K, Hc, Wc]
        # target_cam: [B, Hc, Wc]
        target_cam = cams[batch_idx, labels]

        # target_cam: [B, 1, Hc, Wc]
        target_cam = target_cam.unsqueeze(1)

        # Mask resizing
        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(
                mask.float(),
                size=target_cam.shape[-2:],
                mode="nearest",
            )

        # Binary
        # mask = (mask > 0.5).float()

        outside = 1.0 - mask
        inside = mask

        outside_area = outside.sum(dim=(1, 2, 3)) + self.eps
        inside_area = inside.sum(dim=(1, 2, 3)) + self.eps

        # --- Outside loss for ALL classes (target + non-target) together, one shared weight ---
        # outside broadcast over all K classes: [B, K, Hc, Wc]
        outside_k = outside.expand(b, k, hc, wc)

        cams_zero = torch.zeros_like(cams)

        # mse between every class CAM and zero: [B, K, Hc, Wc]
        outside_mse_map = F.mse_loss(
            cams,
            cams_zero,
            reduction="none",
        )

        # keep only outside-the-mask pixels, for all K classes
        outside_mse_map = outside_mse_map * outside_k

        # normalize by (outside pixels * number of classes)
        outside_all_area = outside_area * k

        outside_mse_all = outside_mse_map.sum(dim=(1, 2, 3)) / outside_all_area

        loss = self.outside_weight * outside_mse_all

        # --- Inside loss: target class CAM should be pushed UP (toward 1) inside the mask ---
        target_one = torch.ones_like(target_cam)

        inside_mse_map = F.mse_loss(
            target_cam,
            target_one,
            reduction="none",
        )

        inside_mse = (inside_mse_map * inside).sum(dim=(1, 2, 3)) / inside_area

        loss = loss + self.inside_weight * inside_mse

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)