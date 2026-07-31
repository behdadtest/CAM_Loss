"""Post-training CAM diagnostics.

Reports three metrics per masked test sample, all computed on the **ground-truth**
class's map (that is what training supervises):

``bg_energy``   fraction of the CAM's positive energy sitting in the background.
                Scale-free, threshold-free, and exactly the quantity the loss is
                trying to minimise. This is the headline number.
``pointing``    does the CAM's single hottest pixel land on the object? The
                classic weakly-supervised-localisation sanity check.
``iou``         IoU at a fixed threshold, kept for continuity. Read it with
                suspicion: min-max normalisation stretches *any* map to the full
                range, so even a flat, useless CAM scores respectably.
"""

import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from src.data import get_dataloaders
from src.mask_config_utils import build_mask_config
from src.model import ResNet18CAM
from src.utils import load_checkpoint


def _upsample(cam: torch.Tensor, size) -> torch.Tensor:
    return F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=size, mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)


def compute_cam_metrics(cam: torch.Tensor, mask: torch.Tensor, cam_threshold: float = 0.5):
    """``cam``: raw [h, w] map for the true class. ``mask``: [1, H, W] or [H, W]."""
    if mask.dim() == 3:
        mask = mask[0]

    cam_full = _upsample(cam, mask.shape[-2:])
    mask_bin = (mask >= 0.5).float()

    positive = F.relu(cam_full)
    total_energy = float(positive.sum().item())
    if total_energy > 1e-9:
        bg_energy = float((positive * (1.0 - mask_bin)).sum().item()) / total_energy
    else:
        bg_energy = 1.0          # a dead map localises nothing

    flat_index = int(torch.argmax(cam_full).item())
    row, col = divmod(flat_index, cam_full.shape[-1])
    pointing = float(mask_bin[row, col].item())

    cam_norm = cam_full - cam_full.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-6)
    cam_bin = (cam_norm >= cam_threshold).float()

    intersection = float((cam_bin * mask_bin).sum().item())
    union = float(((cam_bin + mask_bin) >= 1).float().sum().item())
    iou = intersection / union if union > 1e-6 else 1.0

    return {"bg_energy": bg_energy, "pointing": pointing, "iou": iou}


def _denormalize(x, mean, std):
    img = x.unsqueeze(0) * std + mean
    img = img.clamp(0, 1).squeeze(0)
    return (img.permute(1, 2, 0).numpy() * 255).astype("uint8")


def run_cam_debug(
    model,
    test_loader,
    classes,
    device,
    run_dir,
    top_k: int = 20,
    cam_iou_threshold: float = 0.5,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
):

    model.eval()

    ce_none = torch.nn.CrossEntropyLoss(reduction="none")

    mean_t = torch.tensor(normalize_mean).view(1, 3, 1, 1)
    std_t = torch.tensor(normalize_std).view(1, 3, 1, 1)

    records = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            has_mask = batch["has_mask"].to(device, non_blocking=True)
            paths = batch["image_path"]

            logits, cams = model(x)
            losses = ce_none(logits, y)
            preds = logits.argmax(dim=1)

            for i in range(x.size(0)):
                sample_has_mask = bool(has_mask[i].item())

                # The ground-truth class, matching what the loss supervises.
                cam_i = cams[i, y[i]].detach().float().cpu()
                mask_i = masks[i].detach().cpu()

                metrics = (
                    compute_cam_metrics(cam_i, mask_i, cam_threshold=cam_iou_threshold)
                    if sample_has_mask
                    else None
                )

                records.append({
                    "loss": losses[i].item(),
                    "metrics": metrics,
                    "has_mask": sample_has_mask,
                    "image": x[i].detach().cpu(),
                    "mask": mask_i,
                    "true_label": y[i].item(),
                    "pred_label": preds[i].item(),
                    "cam": cam_i,
                    "image_path": paths[i],
                })

    scored = [r for r in records if r["metrics"] is not None]

    cam_dir = Path(run_dir) / "cam_debug"
    cam_dir.mkdir(parents=True, exist_ok=True)

    if not scored:
        print("cam_debug: no masked samples in the test split, nothing to report.")
        return None

    # Rank by background energy: the quantity the loss actually optimises.
    scored.sort(key=lambda r: r["metrics"]["bg_energy"], reverse=True)
    worst = scored[:top_k]
    best = list(reversed(scored[-top_k:]))

    worst_dir = cam_dir / "worst_bg_energy"
    best_dir = cam_dir / "best_bg_energy"
    worst_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    def _save_group(group, out_dir, prefix):
        summary_rows = []
        for rank, r in enumerate(group, start=1):
            img_np = _denormalize(r["image"], mean_t, std_t)
            h, w = img_np.shape[:2]

            cam_t = F.relu(_upsample(r["cam"], (h, w)))
            cam_t = cam_t / (cam_t.max() + 1e-6)
            heatmap = (cm.jet(cam_t.numpy())[:, :, :3] * 255).astype("uint8")
            overlay = (0.5 * img_np + 0.5 * heatmap).astype("uint8")

            mask_np = (r["mask"][0].numpy() * 255).astype("uint8")
            mask_rgb = np.stack([mask_np] * 3, axis=-1)

            panel = np.concatenate([img_np, mask_rgb, overlay], axis=1)

            m = r["metrics"]
            fname = (
                f"{prefix}_{rank:03d}_true-{classes[r['true_label']]}"
                f"_pred-{classes[r['pred_label']]}"
                f"_bgE-{m['bg_energy']:.3f}_iou-{m['iou']:.3f}_loss-{r['loss']:.3f}.png"
            )
            Image.fromarray(panel).save(out_dir / fname)

            summary_rows.append([
                rank, r["image_path"], classes[r["true_label"]], classes[r["pred_label"]],
                m["bg_energy"], m["pointing"], m["iou"], r["loss"],
            ])

        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "rank", "image_path", "true_label", "pred_label",
                "bg_energy", "pointing", "iou", "loss",
            ])
            writer.writerows(summary_rows)

    _save_group(worst, worst_dir, "worst")
    _save_group(best, best_dir, "best")

    summary = {
        "num_masked_samples": len(scored),
        "bg_energy_mean": statistics.mean(r["metrics"]["bg_energy"] for r in scored),
        "bg_energy_median": statistics.median(r["metrics"]["bg_energy"] for r in scored),
        "pointing_game_accuracy": statistics.mean(r["metrics"]["pointing"] for r in scored),
        "iou_mean": statistics.mean(r["metrics"]["iou"] for r in scored),
        "cam_iou_threshold": cam_iou_threshold,
    }

    with open(cam_dir / "cam_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n--- CAM localisation over {len(scored)} masked test samples ---")
    print(f"background energy fraction : {summary['bg_energy_mean']:.4f}  (lower is better, 0..1)")
    print(f"pointing game accuracy     : {summary['pointing_game_accuracy']:.4f}  (higher is better)")
    print(f"IoU @ {cam_iou_threshold}                 : {summary['iou_mean']:.4f}  (weak metric, see module docstring)")
    print(f"panels saved to: {cam_dir}")

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run CAM diagnostics on a checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="defaults to the newest under runs/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    mask_root = config.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
    else:
        mask_dirs = config.get("mask_dirs", {})
        mask_manifests = config.get("mask_manifests", {})

    _, _, test_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 8),
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 0),
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        test_ratio=config.get("test_ratio", 0.1),
        augment=False,
        strict_mask_coverage=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18CAM(
        num_classes=len(classes),
        pretrained=False,
        dilate_last_block=bool(config.get("dilate_last_block", False)),
    ).to(device)
    model = load_checkpoint(model, device=device, path=args.checkpoint, checkpoint_dir=args.runs_dir)

    run_cam_debug(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
        run_dir=args.runs_dir,
        top_k=args.top_k,
        cam_iou_threshold=float(config.get("cam_debug_iou_threshold", 0.5)),
    )


if __name__ == "__main__":
    main()

## for run : python -m src.cam_debug
