"""
Per-epoch statistics of the target-class CAM against the pseudo-mask.

`CAMMaskLoss` optimises a *ratio*:

    containment = sum(relu(cam) * outside) / sum(relu(cam))

which is invariant to the scale of the CAM. That has a consequence worth
measuring rather than assuming: the model can drive the ratio down either by
suppressing background activation (what we want) or by inflating foreground
activation while leaving the background exactly where it was (what we do not
want, and what a plain "keep the CAM near zero in the background" reading of
the objective would forbid).

So every ratio metric here is paired with an absolute one, and the ratio is
reported next to the two baselines that make it readable:

    floor      containment of a perfect CAM  (all energy in the best cell)
    uniform    containment of a flat CAM     (no localisation at all)

A `containment` sitting at `uniform` means the CAM loss has achieved nothing,
no matter how small the number looks.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.reports.common import EPS, downsample_mask, summarize

# Metrics reported with mean + spread; everything else gets a mean only.
_FULL_SPREAD = (
    "containment",
    "relative_containment",
    "containment_active",
    "softmax_containment",
    "softmax_relative_containment",
    "mean_pos_outside",
    "cam_inside_minus_outside",
)


class CAMStatsAccumulator:
    """Accumulates per-sample CAM/mask statistics over an epoch."""

    def __init__(self, eps: float = EPS, softmax_temperature: float = 1.0):
        self.eps = eps
        # The spatial-softmax containment is recorded on every run, whichever
        # loss is actually training, so a ratio run and a softmax run can be
        # compared on the same axis.
        self.softmax_temperature = float(softmax_temperature)
        self._values: Dict[str, List[float]] = {}
        self.n_samples_seen = 0
        self.n_samples_with_mask = 0
        self.n_batches = 0

    # ------------------------------------------------------------------
    # collection
    # ------------------------------------------------------------------

    def _add(self, name: str, tensor: torch.Tensor) -> None:
        self._values.setdefault(name, []).extend(
            tensor.detach().float().cpu().tolist()
        )

    @torch.no_grad()
    def update(
        self,
        cams: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor,
        has_mask: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> None:
        self.n_batches += 1
        self.n_samples_seen += int(cams.shape[0])

        keep = has_mask.detach().float().flatten() > 0.5
        if not bool(keep.any()):
            return

        cams = cams.detach().float()
        b, _, hc, wc = cams.shape
        batch_idx = torch.arange(b, device=cams.device)

        target = cams[batch_idx, labels][keep].unsqueeze(1)     # [N, 1, Hc, Wc]
        mask_cam = downsample_mask(mask.detach()[keep], (hc, wc))

        n = int(target.shape[0])
        self.n_samples_with_mask += n

        inside = mask_cam
        outside = 1.0 - mask_cam
        area_in = inside.sum(dim=(1, 2, 3))
        area_out = outside.sum(dim=(1, 2, 3))
        n_cells = float(hc * wc)

        positive = F.relu(target)
        energy_in = (positive * inside).sum(dim=(1, 2, 3))
        energy_out = (positive * outside).sum(dim=(1, 2, 3))
        energy_total = energy_in + energy_out + self.eps

        # --- the loss itself, and the two baselines that frame it ----------
        containment = energy_out / energy_total
        uniform = area_out / n_cells               # flat positive CAM
        floor = torch.amin(outside, dim=(1, 2, 3))  # single best cell

        self._add("containment", containment)
        self._add("uniform_containment", uniform)
        self._add("floor_containment", floor)
        # < 1 beats a flat CAM, ~1 means the CAM carries no localisation,
        # > 1 means it is actively anti-localised.
        relative = containment / (uniform + self.eps)
        self._add("relative_containment", relative)
        # 0 = perfect, 1 = no better than flat. Normalises the floor away.
        headroom = (uniform - floor).clamp_min(self.eps)
        self._add("normalized_containment", ((containment - floor) / headroom).clamp(-1.0, 3.0))

        # --- absolute activation, in raw CAM units -------------------------
        # These are the numbers that answer "is the background CAM near zero?".
        self._add("mean_cam_inside", (target * inside).sum(dim=(1, 2, 3)) / (area_in + self.eps))
        self._add("mean_cam_outside", (target * outside).sum(dim=(1, 2, 3)) / (area_out + self.eps))
        self._add("mean_pos_inside", energy_in / (area_in + self.eps))
        self._add("mean_pos_outside", energy_out / (area_out + self.eps))
        self._add("energy_inside", energy_in)
        self._add("energy_outside", energy_out)

        # The degenerate minimum. `containment` is 0/(0+eps) = 0 whenever the
        # whole CAM is negative, which is a perfect loss from a map that says
        # nothing. Logits are GAP(cam) and cross-entropy only reads differences
        # between classes, so an all-negative CAM costs the classifier nothing
        # and no other term in the objective pushes back against it.
        total_positive = energy_in + energy_out
        self._add("total_positive_energy", total_positive)
        active = total_positive > 1e-6
        self._add("cam_all_negative", (~active).float())

        # Containment over the samples the loss can still act on. The plain
        # mean is dragged toward 0 by every collapsed sample, which makes a
        # dead run look like a well-localised one; this is the honest number.
        if bool(active.any()):
            self._add("containment_active", containment[active])
            self._add("relative_containment_active", relative[active])

        # --- spatial-softmax containment ------------------------------------
        # What CAMSoftmaxMaskLoss optimises. Shares the flat/floor baselines
        # with the ratio loss, and unlike the ratio it cannot be zeroed by
        # lowering the map, so it stays honest through a collapse.
        p = F.softmax(target.reshape(n, -1) / self.softmax_temperature, dim=1)
        outside_flat = outside.reshape(n, -1)
        softmax_containment = (p * outside_flat).sum(dim=1)
        softmax_relative = softmax_containment / (uniform + self.eps)
        self._add("softmax_containment", softmax_containment)
        self._add("softmax_relative_containment", softmax_relative)
        # Scored between what this sample can achieve and what a flat CAM gets.
        # A tiny mask can have a floor near 1.0, so the raw value understates
        # those samples and the plain relative one still penalises them.
        self._add(
            "softmax_normalized_containment",
            ((softmax_containment - floor) / headroom).clamp(-1.0, 3.0),
        )
        # The mean of a bimodal population describes none of it. This is the
        # size of the tail that is doing no better than a flat CAM.
        self._add("softmax_inverted_frac", (softmax_relative > 0.9).float())

        # Entropy of p over the grid, normalised to [0, 1]. 1 = uniform, i.e.
        # the CAM says nothing about where the object is. Near 0 = one-hot,
        # which localises confidently but starves every other cell of
        # gradient, since d(loss)/d(z_j) scales with p_j.
        entropy = -(p * torch.log(p + self.eps)).sum(dim=1)
        self._add("softmax_entropy_norm", entropy / float(np.log(max(n_cells, 2))))
        self._add("softmax_peak_prob", p.amax(dim=1))

        # Signed separation between object and background, averaged over
        # cells. Positive means the object side of the map really is higher.
        # Unlike containment this survives the whole map going negative, so it
        # is the metric that catches a CAM inverted in absolute terms.
        self._add(
            "cam_inside_minus_outside",
            (target * inside).sum(dim=(1, 2, 3)) / (area_in + self.eps)
            - (target * outside).sum(dim=(1, 2, 3)) / (area_out + self.eps),
        )

        # Peak background activation relative to peak object activation: the
        # ratio can look healthy while a single background cell still burns.
        big = torch.finfo(target.dtype).max
        bg_cells = (outside > 0.5)
        fg_cells = (inside > 0.5)
        has_bg = bg_cells.flatten(1).any(dim=1)
        has_fg = fg_cells.flatten(1).any(dim=1)

        zero = torch.zeros(n, device=target.device, dtype=target.dtype)
        peak_out = torch.where(
            has_bg, torch.amax(target.masked_fill(~bg_cells, -big), dim=(1, 2, 3)), zero
        )
        peak_in = torch.where(
            has_fg, torch.amax(target.masked_fill(~fg_cells, -big), dim=(1, 2, 3)), zero
        )
        self._add("peak_cam_inside", peak_in)
        self._add("peak_cam_outside", peak_out)
        # Signed margin, not a ratio: peak_in passes through zero as the CAM
        # collapses, and a ratio against it explodes exactly when the run is
        # going wrong. Positive margin = the object holds the strongest cell.
        self._add("peak_margin_inside_minus_outside", peak_in - peak_out)

        # Fraction of background-dominant cells that are still positive: the
        # most direct read of "CAM is not near zero in the background".
        bg_float = bg_cells.float()
        bg_positive = ((target > 0).float() * bg_float).sum(dim=(1, 2, 3))
        self._add("positive_background_cell_frac", bg_positive / (bg_float.sum(dim=(1, 2, 3)) + self.eps))
        self._add("positive_cell_frac", (target > 0).float().mean(dim=(1, 2, 3)))

        # --- shape of the CAM ----------------------------------------------
        self._add("cam_max", torch.amax(target, dim=(1, 2, 3)))
        self._add("cam_min", torch.amin(target, dim=(1, 2, 3)))
        self._add("cam_std", target.flatten(1).std(dim=1, unbiased=False))
        self._add("mask_coverage_cam_grid", inside.mean(dim=(1, 2, 3)))

        # --- pointing game: does the CAM peak land on the object? -----------
        flat_target = target.flatten(1)
        peak_index = flat_target.argmax(dim=1)
        peak_coverage = inside.flatten(1)[torch.arange(n, device=target.device), peak_index]
        self._add("peak_inside_coverage", peak_coverage)
        self._add("pointing_hit", (peak_coverage >= 0.5).float())

        if logits is not None:
            logits = logits.detach().float()[keep]
            kept_labels = labels[keep]
            rows = torch.arange(n, device=logits.device)
            target_logit = logits[rows, kept_labels]
            self._add("logit_target", target_logit)
            if logits.shape[1] > 1:
                other = logits.clone()
                other[rows, kept_labels] = -big
                self._add("logit_margin", target_logit - other.amax(dim=1))

            # Split localisation by whether the sample was classified right.
            # On a misclassified sample the true-class CAM has no reason to sit
            # on the object, so this separates "the CAM loss is not working"
            # from "the classifier is wrong and drags the CAM with it".
            correct = logits.argmax(dim=1) == kept_labels
            if bool(correct.any()):
                self._add("softmax_containment_correct", softmax_containment[correct])
                self._add("pointing_hit_correct", (peak_coverage >= 0.5).float()[correct])
            if bool((~correct).any()):
                self._add("softmax_containment_wrong", softmax_containment[~correct])
                self._add("pointing_hit_wrong", (peak_coverage >= 0.5).float()[~correct])

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------

    def compute(self, prefix: str = "") -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            f"{prefix}n_samples_seen": self.n_samples_seen,
            f"{prefix}n_samples_with_mask": self.n_samples_with_mask,
            f"{prefix}mask_rate": (
                self.n_samples_with_mask / self.n_samples_seen
                if self.n_samples_seen else None
            ),
        }
        for name in metric_names():
            values = self._values.get(name, [])
            summary = summarize(name, values, full=name in _FULL_SPREAD)
            for key, value in summary.items():
                out[f"{prefix}{key}"] = value

        # Does localisation quality track how big the object is? A strong
        # negative correlation means the failures are concentrated on small
        # masks, where a 7x7 grid simply cannot resolve the object -- a
        # resolution problem, not a loss problem.
        out[f"{prefix}corr_softmax_containment_vs_coverage"] = self._correlation(
            "softmax_containment", "mask_coverage_cam_grid"
        )
        return out

    def _correlation(self, a: str, b: str) -> Optional[float]:
        xs = self._values.get(a, [])
        ys = self._values.get(b, [])
        if len(xs) != len(ys) or len(xs) < 3:
            return None
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        if x.size < 3 or x.std() < 1e-12 or y.std() < 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    def per_sample(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in self._values.items()}


def metric_names() -> Sequence[str]:
    """Stable metric ordering, so CSV columns never shift between epochs."""
    return (
        "containment",
        "uniform_containment",
        "floor_containment",
        "relative_containment",
        "containment_active",
        "relative_containment_active",
        "softmax_containment",
        "softmax_relative_containment",
        "softmax_normalized_containment",
        "softmax_inverted_frac",
        "softmax_containment_correct",
        "softmax_containment_wrong",
        "pointing_hit_correct",
        "pointing_hit_wrong",
        "softmax_entropy_norm",
        "softmax_peak_prob",
        "normalized_containment",
        "mean_cam_inside",
        "mean_cam_outside",
        "cam_inside_minus_outside",
        "mean_pos_inside",
        "mean_pos_outside",
        "energy_inside",
        "energy_outside",
        "total_positive_energy",
        "cam_all_negative",
        "peak_cam_inside",
        "peak_cam_outside",
        "peak_margin_inside_minus_outside",
        "positive_background_cell_frac",
        "positive_cell_frac",
        "cam_max",
        "cam_min",
        "cam_std",
        "mask_coverage_cam_grid",
        "peak_inside_coverage",
        "pointing_hit",
        "logit_target",
        "logit_margin",
    )


def metric_keys(prefix: str = "") -> List[str]:
    """Every column `compute()` can emit, in a fixed order."""
    return list(CAMStatsAccumulator().compute(prefix=prefix).keys())


def format_epoch_line(stats: Dict[str, Optional[float]], prefix: str, label: str) -> str:
    """One compact console line per split per epoch."""
    def get(name: str) -> str:
        value = stats.get(f"{prefix}{name}")
        return "n/a" if value is None else f"{float(value):.4f}"

    return (
        f"CAM {label:<5}| masked={stats.get(f'{prefix}n_samples_with_mask')}"
        f"/{stats.get(f'{prefix}n_samples_seen')} "
        f"contain={get('containment_mean')} "
        f"(flat={get('uniform_containment_mean')} floor={get('floor_containment_mean')}) "
        f"rel={get('relative_containment_mean')} "
        f"bg_pos_cells={get('positive_background_cell_frac_mean')} "
        f"bg_act={get('mean_pos_outside_mean')} "
        f"fg_act={get('mean_pos_inside_mean')} "
        f"point={get('pointing_hit_mean')} "
        f"all_neg={get('cam_all_negative_mean')} "
        f"softmax={get('softmax_containment_mean')} "
        f"H={get('softmax_entropy_norm_mean')}"
    )
