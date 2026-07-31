"""Side-by-side CAM comparison between two checkpoints (e.g. lambda_cam 0 vs >0).

Run::

    python -m src.compare_cams --baseline runs/A/best_resnet18_cam.pth \\
                               --cam_model runs/B/best_resnet18_cam.pth

Writes one panel per image (image | mask | baseline CAM | CAM-loss CAM) plus a CSV
of per-image localisation metrics and an aggregate summary.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

from src.data import get_dataloaders
from src.mask_config_utils import build_mask_config
from src.model import ResNet18CAM

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(num_classes, checkpoint_path, device, dilate_last_block=False):
    model = ResNet18CAM(
        num_classes=num_classes,
        pretrained=False,
        dilate_last_block=dilate_last_block,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()
    return model


def denormalize_image(x):
    x = x.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return (x.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


@torch.no_grad()
def get_target_cam(model, image, label, size):
    """Positive part of the true class's CAM, upsampled to ``size``, in [0, 1]."""
    _, cams = model(image)

    target_cam = cams[0, int(label)].unsqueeze(0).unsqueeze(0)
    target_cam = F.interpolate(target_cam, size=size, mode="bilinear", align_corners=False)
    target_cam = F.relu(target_cam)[0, 0].float().cpu().numpy()

    return target_cam / (target_cam.max() + 1e-6)


def make_heatmap_overlay(image_rgb, cam, alpha=0.45):
    heatmap = (cm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    overlay = image_rgb.astype(np.float32) * (1 - alpha) + heatmap.astype(np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_mask_visual(mask):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask > 0.5] = [255, 255, 255]
    return mask_rgb


def draw_title(img, title):
    pil = Image.fromarray(img)
    canvas = Image.new("RGB", (pil.width, pil.height + 28), (255, 255, 255))
    canvas.paste(pil, (0, 28))
    ImageDraw.Draw(canvas).text((8, 6), title, fill=(0, 0, 0))
    return np.array(canvas)


def concat_images(images):
    pil_images = [Image.fromarray(img) for img in images]
    out = Image.new(
        "RGB",
        (sum(im.width for im in pil_images), max(im.height for im in pil_images)),
        (255, 255, 255),
    )
    x = 0
    for im in pil_images:
        out.paste(im, (x, 0))
        x += im.width
    return out


def cam_alignment_metrics(cam, mask):
    """Mean CAM inside/outside the object, and the scale-free background energy."""
    eps = 1e-6
    inside = mask > 0.5
    outside = ~inside

    inside_score = float(cam[inside].mean()) if inside.sum() > 0 else 0.0
    outside_score = float(cam[outside].mean()) if outside.sum() > 0 else 0.0

    total = float(cam.sum())
    bg_energy = float((cam * outside).sum()) / total if total > eps else 1.0

    return inside_score, outside_score, inside_score / (outside_score + eps), bg_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", required=True, help="checkpoint trained with lambda_cam=0")
    parser.add_argument("--cam_model", required=True, help="checkpoint trained with lambda_cam>0")
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--output_dir", default="outputs_cam_comparison")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    mask_root = config.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
    else:
        mask_dirs = config.get("mask_dirs", {})
        mask_manifests = config.get("mask_manifests", {})

    _, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=1,
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=0,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        test_ratio=config.get("test_ratio", 0.1),
        augment=False,
        strict_mask_coverage=False,
    )
    loader = test_loader if args.split == "test" else val_loader

    print("Classes:", classes)

    dilate = bool(config.get("dilate_last_block", False))
    baseline_model = load_model(len(classes), args.baseline, device, dilate)
    cam_model = load_model(len(classes), args.cam_model, device, dilate)

    rows = []
    saved = 0

    for batch in tqdm(loader, desc="Comparing CAMs"):
        if batch["has_mask"].item() < 0.5:
            continue

        image = batch["image"].to(device)
        label = batch["label"].item()
        mask = batch["mask"][0, 0].detach().cpu().numpy()
        image_path = batch["image_path"][0]

        image_rgb = denormalize_image(batch["image"][0])
        size = image_rgb.shape[:2]

        baseline_cam = get_target_cam(baseline_model, image, label, size)
        cam_loss_cam = get_target_cam(cam_model, image, label, size)

        b_in, b_out, b_ratio, b_bg = cam_alignment_metrics(baseline_cam, mask)
        c_in, c_out, c_ratio, c_bg = cam_alignment_metrics(cam_loss_cam, mask)

        panels = [
            draw_title(image_rgb, f"Image | label={classes[label]}"),
            draw_title(make_mask_visual(mask), "Pseudo mask"),
            draw_title(make_heatmap_overlay(image_rgb, baseline_cam), f"Baseline | bgE={b_bg:.2f}"),
            draw_title(make_heatmap_overlay(image_rgb, cam_loss_cam), f"CAM-loss | bgE={c_bg:.2f}"),
        ]

        save_path = vis_dir / f"{saved:03d}_{Path(image_path).stem}_{classes[label]}.png"
        concat_images(panels).save(save_path)

        rows.append({
            "image_path": image_path,
            "class_name": classes[label],
            "baseline_inside": b_in,
            "baseline_outside": b_out,
            "baseline_ratio": b_ratio,
            "baseline_bg_energy": b_bg,
            "cam_inside": c_in,
            "cam_outside": c_out,
            "cam_ratio": c_ratio,
            "cam_bg_energy": c_bg,
            "bg_energy_reduction": b_bg - c_bg,
            "visualization": str(save_path),
        })

        saved += 1
        if saved >= args.num_images:
            break

    if not rows:
        print("No masked samples found in this split; nothing to compare.")
        return

    metrics_path = out_dir / "cam_alignment_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def mean_of(key):
        return sum(r[key] for r in rows) / len(rows)

    print("\nDone.")
    print("Saved visualizations:", vis_dir)
    print("Saved metrics:", metrics_path)
    print(f"\nAverages over {len(rows)} images:")
    print(f"  baseline background energy : {mean_of('baseline_bg_energy'):.4f}")
    print(f"  CAM-loss background energy : {mean_of('cam_bg_energy'):.4f}")
    print(f"  reduction                  : {mean_of('bg_energy_reduction'):+.4f}")
    print(f"  baseline inside/outside    : {mean_of('baseline_ratio'):.4f}")
    print(f"  CAM-loss inside/outside    : {mean_of('cam_ratio'):.4f}")


if __name__ == "__main__":
    main()
