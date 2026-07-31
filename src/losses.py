"""CAM losses that tie a classifier's activation maps to SAM pseudo-masks.

Three losses live here:

``CAMBackgroundLoss``      recommended. Scale-free "how much of the CAM's energy
                          landed in the background" objective.
``NormalizedCAMMaskLoss``  the original MSE formulation, but on a max-normalised
                          CAM so it cannot be satisfied by dimming the map.
``LegacyCAMMaskLoss``      the original implementation, kept only so old runs can
                          be reproduced. See its docstring for why not to use it.

All of them expect **raw, pre-ReLU** CAMs (see ``src/model.py``): ReLU is applied
inside the loss, not in the path that produces the logits.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_cams(cams: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    if tuple(cams.shape[-2:]) == tuple(size):
        return cams
    return F.interpolate(cams, size=size, mode="bilinear", align_corners=False)


def _resize_mask(mask: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Resize a binary mask, preserving the *fraction* of foreground per cell.

    ``mode="area"`` averages every source pixel falling inside a cell, so a cell
    that is 40% object comes out as 0.4. ``mode="nearest"`` samples one single
    pixel per cell and discards the other ~1000 of them, which is why it used to
    disagree with the correct answer on more than half the foreground cells.
    """
    mask = mask.float()
    if tuple(mask.shape[-2:]) == tuple(size):
        return mask
    if size[0] <= mask.shape[-2] and size[1] <= mask.shape[-1]:
        return F.interpolate(mask, size=size, mode="area")
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False)


class _MaskedCAMLoss(nn.Module):
    """Shared geometry: resolution, foreground/background split, ignore band.

    After every forward pass, ``last_components`` holds the (detached) mean of
    each individual term and ``last_num_valid`` holds how many samples in the
    batch actually contributed. Training code uses the latter to weight the
    epoch average correctly.
    """

    def __init__(
        self,
        ignore_band: int = 8,
        fg_threshold: float = 0.5,
        loss_size=None,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.ignore_band = int(ignore_band)
        self.fg_threshold = float(fg_threshold)
        self.loss_size = int(loss_size) if loss_size else None
        self.eps = float(eps)

        self.last_components: Dict[str, float] = {}
        self.last_num_valid: int = 0

    def _prepare(self, cams: torch.Tensor, mask: torch.Tensor):
        """Return ``(cams, foreground, background)`` at the loss resolution.

        The CAM is *upsampled* to the mask instead of the mask being downsampled
        to the CAM, so no mask detail is thrown away. ``loss_size`` caps that
        resolution if the memory cost matters (many classes, large batches).
        """
        mask = mask.float()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        full_h = int(mask.shape[-2])
        size = (self.loss_size, self.loss_size) if self.loss_size else tuple(mask.shape[-2:])

        cams = _resize_cams(cams, size)
        soft_mask = _resize_mask(mask, size)

        foreground = (soft_mask >= self.fg_threshold).float()

        # Each CAM cell sees far more than its own stride, so a cell just outside
        # the object legitimately contains object evidence. Dilating the mask
        # turns that ring into an "ignore" region rather than a penalised one.
        band = int(round(self.ignore_band * size[0] / float(full_h)))
        if band > 0:
            foreground_dilated = F.max_pool2d(
                foreground, 2 * band + 1, stride=1, padding=band
            )
        else:
            foreground_dilated = foreground

        return cams, foreground, 1.0 - foreground_dilated

    def _empty(self, cams: torch.Tensor) -> torch.Tensor:
        self.last_num_valid = 0
        self.last_components = {}
        return cams.sum() * 0.0

    def _reduce(self, per_sample, parts, valid) -> torch.Tensor:
        count = float(valid.sum().item())
        self.last_num_valid = int(count)
        self.last_components = {
            name: float((value.detach() * valid).sum().item()) / max(count, 1.0)
            for name, value in parts.items()
        }
        return (per_sample * valid).sum() / (valid.sum() + self.eps)

    def _valid_samples(self, has_mask, foreground, background):
        """A sample is only usable if it has *both* object and background pixels.

        An all-foreground mask (SAM segmenting the whole photo) carries no
        background signal at all, and would otherwise contribute a pure
        "light up everywhere" gradient.
        """
        fg_area = foreground.sum(dim=(1, 2, 3))
        bg_area = background.sum(dim=(1, 2, 3))
        valid = has_mask * (fg_area > 0).float() * (bg_area > 0).float()
        return valid, fg_area, bg_area


class CAMBackgroundLoss(_MaskedCAMLoss):
    """Scale-free background suppression.

    The main term is *containment*: what fraction of the target CAM's positive
    energy landed outside the object.

    .. math::
        L = \\frac{\\sum \\mathrm{ReLU}(cam_t)\\cdot bg}{\\sum \\mathrm{ReLU}(cam_t)}

    It lives in [0, 1], so ``lambda_cam`` means something stable; it cannot be
    minimised by shrinking the whole map (numerator and denominator shrink
    together); and it needs no target constant, so nothing pins the CAM's scale
    and the classifier stays free to become confident.

    Two optional terms:

    ``coverage_weight``   keeps the map from collapsing onto a single
                          discriminative pixel (e.g. only the dog's nose).
    ``nontarget_weight``  pixel-wise class competition inside the object: at every
                          pixel on the animal, the true class should win. Without
                          it, a wrong class may sit on the object for free.

    All three terms are bounded in [0, 1], so ``lambda_cam`` keeps the same
    meaning as the weights change.
    """

    def __init__(
        self,
        outside_weight: float = 1.0,
        coverage_weight: float = 0.0,
        nontarget_weight: float = 0.0,
        ignore_band: int = 8,
        fg_threshold: float = 0.5,
        loss_size=None,
        eps: float = 1e-6,
    ):
        super().__init__(
            ignore_band=ignore_band,
            fg_threshold=fg_threshold,
            loss_size=loss_size,
            eps=eps,
        )
        self.outside_weight = float(outside_weight)
        self.coverage_weight = float(coverage_weight)
        self.nontarget_weight = float(nontarget_weight)

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:

        has_mask = has_mask.float().view(-1)
        if has_mask.sum() == 0:
            return self._empty(cams)

        b, k = cams.shape[0], cams.shape[1]
        idx = torch.arange(b, device=cams.device)

        cams, foreground, background = self._prepare(cams, mask)
        valid, fg_area, _ = self._valid_samples(has_mask, foreground, background)
        if valid.sum() == 0:
            return self._empty(cams)

        positive = F.relu(cams)
        target = positive[idx, labels].unsqueeze(1)

        energy_outside = (target * background).sum(dim=(1, 2, 3))
        energy_total = target.sum(dim=(1, 2, 3)) + self.eps
        containment = energy_outside / energy_total

        loss = self.outside_weight * containment
        parts = {"outside": containment}

        if self.coverage_weight > 0:
            # Detached peak: normalising by a live maximum would let the model
            # lower this term by inflating its own peak instead of spreading out.
            peak = target.amax(dim=(1, 2, 3), keepdim=True).detach() + self.eps
            covered = ((target / peak) * foreground).sum(dim=(1, 2, 3)) / (fg_area + self.eps)
            coverage = 1.0 - covered
            loss = loss + self.coverage_weight * coverage
            parts["coverage"] = coverage

        if self.nontarget_weight > 0 and k > 1:
            # Bounded in [0, 1] on purpose. A pixel-wise cross-entropy would say
            # the same thing but is unbounded, so one badly-scaled batch lets this
            # term dwarf the other two and lambda_cam stops meaning anything.
            prob_target = F.softmax(cams, dim=1)[idx, labels].unsqueeze(1)
            nontarget = 1.0 - (prob_target * foreground).sum(dim=(1, 2, 3)) / (fg_area + self.eps)
            loss = loss + self.nontarget_weight * nontarget
            parts["nontarget"] = nontarget

        return self._reduce(loss, parts, valid)


class NormalizedCAMMaskLoss(_MaskedCAMLoss):
    """The original MSE formulation, repaired.

    Same idea as before -- push the CAM toward 0 outside the mask and toward 1
    inside it -- but computed on ``relu(cam) / peak`` with a **detached** peak.

    That single change removes both of the original failure modes: the loss no
    longer cares about the absolute size of the CAM (so it cannot be satisfied by
    turning the map off), and the "equal 1 inside" target constrains only the
    map's *shape*, not its scale, so it no longer caps the logits.
    """

    def __init__(
        self,
        outside_weight: float = 1.0,
        inside_weight: float = 1.0,
        ignore_band: int = 8,
        fg_threshold: float = 0.5,
        loss_size=None,
        eps: float = 1e-6,
    ):
        super().__init__(
            ignore_band=ignore_band,
            fg_threshold=fg_threshold,
            loss_size=loss_size,
            eps=eps,
        )
        self.outside_weight = float(outside_weight)
        self.inside_weight = float(inside_weight)

    def forward(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
    ) -> torch.Tensor:

        has_mask = has_mask.float().view(-1)
        if has_mask.sum() == 0:
            return self._empty(cams)

        b, k = cams.shape[0], cams.shape[1]
        idx = torch.arange(b, device=cams.device)

        cams, foreground, background = self._prepare(cams, mask)
        valid, fg_area, bg_area = self._valid_samples(has_mask, foreground, background)
        if valid.sum() == 0:
            return self._empty(cams)

        positive = F.relu(cams)
        peak = positive.amax(dim=(2, 3), keepdim=True).detach() + self.eps
        normalized = positive / peak

        # Outside: every class toward 0, averaged over background pixels and classes.
        outside = ((normalized ** 2) * background).sum(dim=(1, 2, 3)) / (bg_area * k + self.eps)

        # Inside: the target class toward 1.
        target = normalized[idx, labels].unsqueeze(1)
        inside = (((target - 1.0) ** 2) * foreground).sum(dim=(1, 2, 3)) / (fg_area + self.eps)

        loss = self.outside_weight * outside + self.inside_weight * inside
        return self._reduce(loss, {"outside": outside, "inside": inside}, valid)


class LegacyCAMMaskLoss(nn.Module):
    """The original loss, preserved verbatim for reproducing old runs.

    Do not use it for new experiments. Three reasons, all measurable:

    1. It targets ``cam == 1`` inside the mask on **un-normalised** values. Since
       the logit is the spatial mean of the map, that pins the target logit to
       roughly the foreground area fraction (~0.2) and caps softmax confidence
       near 60% on a two-class problem.
    2. The outside term is not scale-invariant, so scaling the CAM down by 100x
       reduces it by 10000x -- a free minimum that teaches no localisation.
    3. It downsamples the mask with ``mode="nearest"``, which agrees with a
       correct area-based downsample only about 45% of the time.
    """

    def __init__(self, outside_weight=1.0, inside_weight=1.0, eps=1e-6):
        super().__init__()
        self.outside_weight = outside_weight
        self.inside_weight = inside_weight
        self.eps = eps
        self.last_components: Dict[str, float] = {}
        self.last_num_valid: int = 0

    def forward(self, cams, mask, labels, has_mask):
        has_mask = has_mask.float()

        if has_mask.sum() == 0:
            self.last_num_valid = 0
            return cams.sum() * 0.0

        b, k, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        target_cam = cams[batch_idx, labels].unsqueeze(1)

        if mask.shape[-2:] != target_cam.shape[-2:]:
            mask = F.interpolate(mask.float(), size=target_cam.shape[-2:], mode="nearest")

        outside = 1.0 - mask
        inside = mask

        outside_area = outside.sum(dim=(1, 2, 3)) + self.eps
        inside_area = inside.sum(dim=(1, 2, 3)) + self.eps

        outside_mse_map = F.mse_loss(cams, torch.zeros_like(cams), reduction="none")
        outside_mse_map = outside_mse_map * outside.expand(b, k, hc, wc)
        outside_mse_all = outside_mse_map.sum(dim=(1, 2, 3)) / (outside_area * k)

        inside_mse_map = F.mse_loss(target_cam, torch.ones_like(target_cam), reduction="none")
        inside_mse = (inside_mse_map * inside).sum(dim=(1, 2, 3)) / inside_area

        loss = self.outside_weight * outside_mse_all + self.inside_weight * inside_mse
        loss = loss * has_mask

        self.last_num_valid = int(has_mask.sum().item())
        self.last_components = {
            "outside": float((outside_mse_all.detach() * has_mask).sum().item())
            / max(self.last_num_valid, 1),
            "inside": float((inside_mse.detach() * has_mask).sum().item())
            / max(self.last_num_valid, 1),
        }
        return loss.sum() / (has_mask.sum() + self.eps)


def build_cam_loss(config: dict) -> nn.Module:
    """Construct the CAM loss named by ``config["cam_loss_type"]``."""
    name = str(config.get("cam_loss_type", "containment")).strip().lower()

    if name == "containment":
        return CAMBackgroundLoss(
            outside_weight=float(config.get("outside_weight", 1.0)),
            coverage_weight=float(config.get("coverage_weight", 0.0)),
            nontarget_weight=float(config.get("nontarget_weight", 0.0)),
            ignore_band=int(config.get("cam_ignore_band", 8)),
            fg_threshold=float(config.get("cam_fg_threshold", 0.5)),
            loss_size=config.get("cam_loss_size", None),
        )

    if name in ("normalized_mse", "normalised_mse"):
        return NormalizedCAMMaskLoss(
            outside_weight=float(config.get("outside_weight", 1.0)),
            inside_weight=float(config.get("inside_weight", 1.0)),
            ignore_band=int(config.get("cam_ignore_band", 8)),
            fg_threshold=float(config.get("cam_fg_threshold", 0.5)),
            loss_size=config.get("cam_loss_size", None),
        )

    if name == "legacy_mse":
        return LegacyCAMMaskLoss(
            outside_weight=float(config.get("outside_weight", 1.0)),
            inside_weight=float(config.get("inside_weight", 1.0)),
        )

    raise ValueError(
        f"Unknown cam_loss_type={name!r}. "
        f"Expected one of: containment, normalized_mse, legacy_mse."
    )
