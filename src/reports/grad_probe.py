"""
How much does the CAM term actually move the weights, and does it fight the
classifier?

`lambda_cam` is usually tuned by staring at the two loss values, but equal loss
magnitudes do not imply equal influence: what reaches the optimiser is the
gradient. This probe backprops the two terms separately (without stepping) and
reports, per parameter group:

  * `||grad(ce)||` vs `||lambda * grad(cam)||` and their ratio
    - ratio near 0  -> the CAM term is decorative, raise lambda_cam
    - ratio very large -> the CAM term is drowning the classifier
  * `cos(grad(ce), grad(cam))`
    - clearly negative -> the two objectives genuinely conflict, and a bigger
      lambda buys localisation at the direct cost of accuracy
    - near 0 -> they are roughly orthogonal, which is the comfortable regime
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def _flat_grad(grads, params) -> torch.Tensor:
    parts = []
    for grad, param in zip(grads, params):
        parts.append(
            torch.zeros_like(param).flatten() if grad is None
            else grad.detach().flatten()
        )
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def _param_groups(model) -> Dict[str, List[torch.nn.Parameter]]:
    """Split trainable params into the groups we care about separately."""
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "all": [],
        "cam_head": [],
        "adapters": [],
        "backbone": [],
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        groups["all"].append(param)
        if "cam_conv" in name:
            groups["cam_head"].append(param)
        elif "adapter" in name:
            groups["adapters"].append(param)
        else:
            groups["backbone"].append(param)
    return {k: v for k, v in groups.items() if v}


def _compare(g_ce: torch.Tensor, g_cam: torch.Tensor, lambda_cam: float) -> Dict[str, float]:
    ce_norm = float(g_ce.norm())
    cam_norm = float(g_cam.norm())
    weighted = cam_norm * abs(lambda_cam)

    if ce_norm > 0 and cam_norm > 0:
        cosine = float(
            torch.dot(g_ce, g_cam) / (g_ce.norm() * g_cam.norm())
        )
    else:
        cosine = 0.0

    return {
        "ce_grad_norm": ce_norm,
        "cam_grad_norm": cam_norm,
        "weighted_cam_grad_norm": weighted,
        "cam_over_ce_ratio": weighted / ce_norm if ce_norm > 0 else None,
        "cosine_ce_cam": cosine,
    }


def run_grad_probe(
    model,
    dataloader,
    ce_loss_fn,
    cam_loss_fn,
    device: str,
    lambda_cam: float,
    max_batches: int = 3,
) -> Optional[Dict[str, float]]:
    """
    Average the gradient comparison over the first `max_batches` batches.

    Runs in `eval()` mode for determinism and restores the model's previous
    mode afterwards. Nothing is stepped and no gradient is left on `.grad`.
    """
    if dataloader is None or max_batches <= 0:
        return None

    was_training = model.training
    model.eval()

    groups = _param_groups(model)
    if not groups:
        model.train(was_training)
        return None

    totals: Dict[str, List[float]] = {}
    n_batches = 0
    n_masked_total = 0

    try:
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break

            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)

            n_masked = int((has_mask > 0.5).sum().item())
            n_masked_total += n_masked

            model.zero_grad(set_to_none=True)

            logits, cams = model(x)
            loss_cls = ce_loss_fn(logits, y)
            loss_cam = cam_loss_fn(cams=cams, mask=masks, labels=y, has_mask=has_mask)

            for group_name, params in groups.items():
                g_ce = torch.autograd.grad(
                    loss_cls, params, retain_graph=True, allow_unused=True
                )
                g_cam = torch.autograd.grad(
                    loss_cam, params, retain_graph=True, allow_unused=True
                )

                comparison = _compare(
                    _flat_grad(g_ce, params),
                    _flat_grad(g_cam, params),
                    lambda_cam,
                )
                for key, value in comparison.items():
                    if value is None:
                        continue
                    totals.setdefault(f"{group_name}/{key}", []).append(float(value))

            totals.setdefault("loss/ce", []).append(float(loss_cls.item()))
            totals.setdefault("loss/cam", []).append(float(loss_cam.item()))
            n_batches += 1
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    if n_batches == 0:
        return None

    result: Dict[str, float] = {
        key: sum(values) / len(values) for key, values in totals.items()
    }
    result["n_batches"] = n_batches
    result["n_masked_samples"] = n_masked_total
    result["lambda_cam"] = float(lambda_cam)
    return result


@torch.no_grad()
def probe_cam_head(model) -> Dict[str, float]:
    """
    Level statistics of the CAM head itself.

    The CAM's absolute level can drift for two very different reasons, and
    they need different fixes: the per-class bias is a spatially uniform
    offset that moves the whole map without touching localisation at all,
    while the weights move it through the features. Separating them here
    turns "the CAM went negative" into a specific thing to change.
    """
    out: Dict[str, float] = {}

    for name, module in model.named_modules():
        if "cam_conv" not in name or not hasattr(module, "weight"):
            continue

        weight = module.weight.detach().float()
        out["head_weight_norm"] = float(weight.norm())
        out["head_weight_mean"] = float(weight.mean())
        out["head_weight_absmean"] = float(weight.abs().mean())

        bias = getattr(module, "bias", None)
        if bias is None:
            out["head_has_bias"] = 0.0
        else:
            bias = bias.detach().float()
            out["head_has_bias"] = 1.0
            out["head_bias_mean"] = float(bias.mean())
            out["head_bias_std"] = float(bias.std()) if bias.numel() > 1 else 0.0
            out["head_bias_min"] = float(bias.min())
            out["head_bias_max"] = float(bias.max())
        break

    return out


def format_probe_line(probe: Optional[Dict[str, float]]) -> str:
    if not probe:
        return "GRAD  | (not available)"

    def get(key: str) -> str:
        value = probe.get(key)
        return "n/a" if value is None else f"{float(value):.3e}"

    ratio = probe.get("all/cam_over_ce_ratio")
    cosine = probe.get("all/cosine_ce_cam")
    return (
        f"GRAD  | |g_ce|={get('all/ce_grad_norm')} "
        f"|lam*g_cam|={get('all/weighted_cam_grad_norm')} "
        f"ratio={'n/a' if ratio is None else f'{ratio:.4f}'} "
        f"cos={'n/a' if cosine is None else f'{cosine:+.4f}'} "
        f"masked={probe.get('n_masked_samples')}"
    )
