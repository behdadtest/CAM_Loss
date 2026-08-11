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


class CAMSoftmaxMaskLoss(nn.Module):
    """
    Spatial-softmax containment.

        p    = softmax(cam[label].flatten() / T)
        loss = sum(p * outside)

    Reads the CAM as a distribution over *where* the evidence sits, and asks
    how much of that distribution lands on the background.

    Why this instead of the relu-gated energy ratio in CAMMaskLoss:

    * **No degenerate minimum.** `p` sums to 1 by construction, so there is no
      way to empty it. The ratio loss returns a perfect 0 whenever the whole
      CAM is negative -- both sides of the ratio vanish under relu -- and that
      state is reachable just by lowering the map. A GAP+CE classifier over
      many classes drifts into it on its own, because softmax cross-entropy
      only constrains differences between class maps and never their level.

    * **Gradient everywhere.** d(loss)/d(z_j) = (p_j / T) * (o_j - loss): every
      cell keeps a gradient, pushing cells that are more background than
      average down and cells that are more object than average up. Under relu
      a cell that crosses zero stops receiving any signal at all, so the ratio
      loss surrenders the map one cell at a time as the level sinks.

    * **Level-invariant.** Adding a constant to the CAM leaves softmax
      unchanged, so this loss neither rewards nor even notices the absolute
      drift that cross-entropy imposes on the logits.

    The two reference values are unchanged, which keeps the diagnostics
    directly comparable with the ratio loss: a uniform `p` scores
    mean(outside) -- the flat-CAM baseline -- and a `p` concentrated on the
    most-inside cell scores min(outside) -- the floor.

    `temperature` divides the CAM before the softmax, so it sets how sharply
    the loss reads the map: T -> 0 makes `p` one-hot at the CAM's peak (the
    loss becomes a soft pointing game), T -> inf makes `p` uniform (the loss
    becomes the constant mean(outside) and stops teaching anything). Note
    softmax is invariant to shifts but not to scale, so a CAM whose spread
    grows will produce a peakier `p`; `softmax_entropy_norm` in the
    diagnostics tracks that.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        outside_weight=1.0,
        inside_weight=1.0,
        eps=1e-6,
    ):
        super().__init__()

        temperature = float(temperature)
        if not temperature > 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.temperature = temperature
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
        # float32 even under autocast: softmax over a small map in fp16 loses
        # resolution exactly where the CAM values are close together.
        target_cam = cams[batch_idx, labels].float()

        # Mask resizing (same "area" downsample the ratio loss uses, so the
        # two losses see an identical notion of inside/outside).
        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="area")

        outside = (1.0 - mask.float()).reshape(b, -1)

        # p: [B, Hc*Wc], sums to 1 along the spatial axis.
        p = F.softmax(target_cam.reshape(b, -1) / self.temperature, dim=1)

        containment = (p * outside).sum(dim=1)

        loss = self.outside_weight * containment

        # for those which have mask we have to calculate CAMLoss
        loss = loss * has_mask

        # return mean
        return loss.sum() / (has_mask.sum() + self.eps)


CAM_LOSS_REGISTRY = {
    "ratio": CAMMaskLoss,
    "softmax": CAMSoftmaxMaskLoss,
}


def build_cam_loss(config: dict) -> nn.Module:
    """
    Pick the CAM loss from config.

    Defaults to "ratio" so a config written before this option existed keeps
    reproducing its original run.
    """
    loss_type = str(config.get("cam_loss_type", "ratio")).lower()

    if loss_type not in CAM_LOSS_REGISTRY:
        raise ValueError(
            f"Unknown cam_loss_type='{loss_type}'. "
            f"Available options: {sorted(CAM_LOSS_REGISTRY)}"
        )

    outside_weight = float(config.get("outside_weight", 1.0))
    inside_weight = float(config.get("inside_weight", 1.0))

    if loss_type == "softmax":
        return CAMSoftmaxMaskLoss(
            temperature=float(config.get("cam_softmax_temperature", 1.0)),
            outside_weight=outside_weight,
            inside_weight=inside_weight,
        )

    return CAMMaskLoss(
        outside_weight=outside_weight,
        inside_weight=inside_weight,
    )