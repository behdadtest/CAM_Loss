import torch
import torch.nn as nn
import torch.nn.functional as F


def build_all_cams(features: torch.Tensor, classifier_weight: torch.Tensor) -> torch.Tensor:
    """
        cams: [B, num_classes, H, W]
    """

    cams = torch.einsum("kc,bchw->bkhw", classifier_weight, features)
    # cams = F.relu(cams)

    return cams


class CAMMaskLoss(nn.Module):

    def __init__(
        self,
        outside_weight=1.0,
        non_target_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        self.outside_weight = outside_weight
        self.non_target_weight = non_target_weight
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

        cams = self.normalize_cams(cams)

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
        mask = (mask > 0.5).float()

        # Out side base
        outside = 1.0 - mask

        # number of pixels in outside
        outside_area = outside.sum(dim=(1, 2, 3)) + self.eps

        target_zero = torch.zeros_like(target_cam)

        # MSE
        # mse_map: [B, 1, Hc, Wc]
        mse_map = F.mse_loss(
            target_cam,
            target_zero,
            reduction="none",
        )

        # just mse of outside remain
        outside_mse = (mse_map * outside).sum(dim=(1, 2, 3)) / outside_area

        loss = self.outside_weight * outside_mse # not importent for now

        # --- Non-target classes: their CAMs should also stay quiet outside the mask ---
        # one_hot: [B, K] -> 1 at the target class, 0 elsewhere
        one_hot = F.one_hot(labels, num_classes=k).float()

        # non_target_mask: [B, K, 1, 1] -> 1 for every class except the target one
        non_target_mask = (1.0 - one_hot).view(b, k, 1, 1)

        # outside region broadcast over all K classes: [B, K, Hc, Wc]
        outside_k = outside.expand(b, k, hc, wc)

        cams_zero = torch.zeros_like(cams)

        # mse between every class CAM and zero: [B, K, Hc, Wc]
        non_target_mse_map = F.mse_loss(
            cams,
            cams_zero,
            reduction="none",
        )

        # only keep: (a) outside-the-mask pixels, (b) non-target classes
        non_target_mse_map = non_target_mse_map * outside_k * non_target_mask

        # normalize by (outside pixels * number of non-target classes)
        non_target_area = outside_area * max(k - 1, 1)

        non_target_outside_mse = non_target_mse_map.sum(dim=(1, 2, 3)) / non_target_area

        loss = loss + self.non_target_weight * non_target_outside_mse

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)