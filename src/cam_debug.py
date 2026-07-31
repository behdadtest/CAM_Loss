import csv
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from src.data import get_dataloaders
from src.model import ResNet18CAM

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image

from src.utils import load_checkpoint


def _compute_cam_mask_iou(cam: torch.Tensor, mask: torch.Tensor, cam_threshold: float = 0.5):
    if mask.dim() == 3:
        mask = mask[0]

    # Min-max normalize CAM to [0, 1].
    cam_norm = cam - cam.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-6)

    # Resize CAM to mask resolution.
    cam_resized = F.interpolate(
        cam_norm.unsqueeze(0).unsqueeze(0),
        size=mask.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    cam_bin = (cam_resized >= cam_threshold).float()
    # print(mask)
    # print(cam_bin)
    mask_bin = (mask >= 0.5).float()

    intersection = (cam_bin * mask_bin).sum().item()
    union = ((cam_bin + mask_bin) >= 1).float().sum().item()

    if union < 1e-6:
        return 1.0

    return intersection / union


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
                cam_i = cams[i, preds[i]].detach().cpu()
                mask_i = masks[i].detach().cpu()

                iou = (
                    _compute_cam_mask_iou(cam_i, mask_i, cam_threshold=cam_iou_threshold)
                    if sample_has_mask else None
                )

                records.append({
                    "loss": losses[i].item(),
                    "iou": iou,
                    "has_mask": sample_has_mask,
                    "image": x[i].detach().cpu(),
                    "true_label": y[i].item(),
                    "pred_label": preds[i].item(),
                    "cam": cam_i,
                    "image_path": paths[i],
                })

    iou_records = [r for r in records if r["has_mask"] and r["iou"] is not None]

    iou_records.sort(key=lambda r: r["iou"])
    worst = iou_records[:top_k]
    best = list(reversed(iou_records[-top_k:]))

    cam_dir = Path(run_dir) / "cam_debug"
    worst_dir = cam_dir / "worst_iou"
    best_dir = cam_dir / "best_iou"
    worst_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    def _save_group(group, out_dir, prefix):
        summary_rows = []
        for rank, r in enumerate(group, start=1):
            img = r["image"].unsqueeze(0) * std_t + mean_t
            img = img.clamp(0, 1).squeeze(0)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")

            cam_t = r["cam"]
            cam_t = cam_t - cam_t.min()
            cam_t = cam_t / (cam_t.max() + 1e-6)
            cam_np = cam_t.numpy()

            cam_img = Image.fromarray((cam_np * 255).astype("uint8")).resize(
                (img_np.shape[1], img_np.shape[0]), resample=Image.BILINEAR
            )
            cam_resized = np.array(cam_img).astype("float32") / 255.0

            heatmap = cm.jet(cam_resized)[:, :, :3]
            heatmap = (heatmap * 255).astype("uint8")

            overlay = (0.5 * img_np + 0.5 * heatmap).astype("uint8")

            true_name = classes[r["true_label"]]
            pred_name = classes[r["pred_label"]]

            fname = (
                f"{prefix}_{rank:03d}_true-{true_name}_pred-{pred_name}_"
                f"iou-{r['iou']:.3f}_loss-{r['loss']:.3f}.png"
            )

            Image.fromarray(overlay).save(out_dir / fname)

            summary_rows.append([
                rank, r["image_path"], true_name, pred_name, r["iou"], r["loss"]
            ])

        with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "image_path", "true_label", "pred_label", "iou", "loss"])
            writer.writerows(summary_rows)

    _save_group(worst, worst_dir, "worst")
    _save_group(best, best_dir, "best")

    print(f"Saved {len(worst)} worst-IoU CAM overlays to {worst_dir}")
    print(f"Saved {len(best)} best-IoU CAM overlays to {best_dir}")
    print(
        f"IoU stats over {len(iou_records)} masked samples: "
        f"min={iou_records[0]['iou']:.3f} max={iou_records[-1]['iou']:.3f} "
        f"mean={statistics.mean(r['iou'] for r in iou_records):.3f}"
    )


def main(model, test_loader, classes, device, run_dir):
    run_cam_debug(
        model=model,
        test_loader=test_loader,
        classes=classes,
        device=device,
        run_dir=run_dir,
        top_k=5,
        cam_iou_threshold=0.5,
    )


if __name__ == "__main__":
    
    config_path = Path("configs/config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    _, _, test_loader, classes = get_dataloaders(
            data_dir=config["data_dir"],
            image_size=config.get("image_size", 224),
            batch_size=config.get("batch_size", 8),
            val_ratio=config.get("val_ratio", 0.2),
            seed=config.get("seed", 42),
            num_workers=config.get("num_workers", 0),
            mask_dirs=config.get("mask_dirs", {}),
            mask_manifests=config.get("mask_manifests", None),
            test_ratio=config.get("test_ratio", 0.1),
    )
    model = ResNet18CAM(num_classes=len(classes), pretrained=False).to("cuda")
    model = load_checkpoint(model, device="cuda")
    
       
    main(model=model, test_loader=test_loader, classes=classes, device="cuda", run_dir="runs")
    
## for run : python -m src.cam_debug
