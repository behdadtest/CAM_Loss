from pathlib import Path
import argparse
import csv

import cv2
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from data import get_dataloaders
from model import ResNet18CAM
from losses import build_all_cams


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()
    return model


def denormalize_image(x):
    """
    x: [3, H, W], normalized tensor
    return: uint8 RGB image [H, W, 3]
    """
    x = x.detach().cpu()
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0, 1)
    img = x.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


def normalize_cam(cam):
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-6)
    return cam


@torch.no_grad()
def get_target_cam(model, image, label, device):
    """
    image: [1, 3, H, W]
    label: scalar tensor [1]
    return: CAM [H, W] normalized 0..1
    """
    logits, features = model(image)

    cams = build_all_cams(
        features=features,
        classifier_weight=model.fc.weight,
    )

    # cams: [1, num_classes, Hc, Wc]
    target_cam = cams[0, label.item()]  # [Hc, Wc]
    target_cam = target_cam.unsqueeze(0).unsqueeze(0)  # [1, 1, Hc, Wc]

    target_cam = F.interpolate(
        target_cam,
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    target_cam = target_cam[0, 0].detach().cpu().numpy()
    target_cam = normalize_cam(target_cam)

    return target_cam


def make_heatmap_overlay(image_rgb, cam, alpha=0.45):
    """
    image_rgb: uint8 [H, W, 3]
    cam: float [H, W], 0..1
    """
    heat = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = image_rgb.astype(np.float32) * (1 - alpha) + heatmap_rgb.astype(np.float32) * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def make_mask_visual(mask):
    """
    mask: [H, W], 0/1
    """
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask > 0.5] = [255, 255, 255]
    return mask_rgb


def draw_title(img, title):
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)

    h = 28
    canvas = Image.new("RGB", (pil.width, pil.height + h), (255, 255, 255))
    canvas.paste(pil, (0, h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(0, 0, 0))

    return np.array(canvas)


def concat_images(images):
    pil_images = [Image.fromarray(img) for img in images]
    widths = [im.width for im in pil_images]
    heights = [im.height for im in pil_images]

    out = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))

    x = 0
    for im in pil_images:
        out.paste(im, (x, 0))
        x += im.width

    return out


def cam_alignment_metrics(cam, mask):
    """
    cam: [H, W], 0..1
    mask: [H, W], 0/1

    خروجی:
    inside_score  = میانگین CAM داخل object
    outside_score = میانگین CAM بیرون object
    ratio         = inside / outside
    """
    eps = 1e-6

    inside = mask > 0.5
    outside = ~inside

    inside_score = cam[inside].mean() if inside.sum() > 0 else 0.0
    outside_score = cam[outside].mean() if outside.sum() > 0 else 0.0
    ratio = inside_score / (outside_score + eps)

    return float(inside_score), float(outside_score), float(ratio)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--baseline", required=True, help="Path to baseline best_model.pth")
    parser.add_argument("--cam_model", required=True, help="Path to CAM-loss best_model.pth")
    parser.add_argument("--num_images", type=int, default=30)
    parser.add_argument("--output_dir", default="outputs_cam_comparison")
    args = parser.parse_args()

    config = load_config(args.config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=1,
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=0,
        mask_dirs=config.get("mask_dirs", {}),
        mask_manifests=config.get("mask_manifests", {}),
    )

    print("Classes:", classes)

    baseline_model = ResNet18CAM(
        num_classes=len(classes),
        pretrained=False,
    ).to(device)

    cam_model = ResNet18CAM(
        num_classes=len(classes),
        pretrained=False,
    ).to(device)

    baseline_model = load_checkpoint(baseline_model, args.baseline, device)
    cam_model = load_checkpoint(cam_model, args.cam_model, device)

    metrics_path = out_dir / "cam_alignment_metrics.csv"

    rows = []

    saved = 0

    for batch in tqdm(val_loader, desc="Comparing CAMs"):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        mask = batch["mask"][0, 0].detach().cpu().numpy()
        has_mask = batch["has_mask"].item()
        image_path = batch["image_path"][0]

        if has_mask < 0.5:
            continue

        class_name = classes[label.item()]

        image_rgb = denormalize_image(batch["image"][0])

        baseline_cam = get_target_cam(
            model=baseline_model,
            image=image,
            label=label,
            device=device,
        )

        cam_loss_cam = get_target_cam(
            model=cam_model,
            image=image,
            label=label,
            device=device,
        )

        baseline_inside, baseline_outside, baseline_ratio = cam_alignment_metrics(
            baseline_cam,
            mask,
        )

        cam_inside, cam_outside, cam_ratio = cam_alignment_metrics(
            cam_loss_cam,
            mask,
        )

        baseline_overlay = make_heatmap_overlay(image_rgb, baseline_cam)
        cam_loss_overlay = make_heatmap_overlay(image_rgb, cam_loss_cam)
        mask_vis = make_mask_visual(mask)

        panels = [
            draw_title(image_rgb, f"Image | label={class_name}"),
            draw_title(mask_vis, "Pseudo mask"),
            draw_title(baseline_overlay, f"Baseline CAM | ratio={baseline_ratio:.2f}"),
            draw_title(cam_loss_overlay, f"CAM-loss CAM | ratio={cam_ratio:.2f}"),
        ]

        grid = concat_images(panels)

        image_stem = Path(image_path).stem
        save_path = vis_dir / f"{saved:03d}_{image_stem}_{class_name}.png"
        grid.save(save_path)

        rows.append({
            "image_path": image_path,
            "class_name": class_name,

            "baseline_inside": baseline_inside,
            "baseline_outside": baseline_outside,
            "baseline_ratio": baseline_ratio,

            "cam_inside": cam_inside,
            "cam_outside": cam_outside,
            "cam_ratio": cam_ratio,

            "ratio_improvement": cam_ratio - baseline_ratio,
            "inside_improvement": cam_inside - baseline_inside,
            "outside_reduction": baseline_outside - cam_outside,
            "visualization": str(save_path),
        })

        saved += 1

        if saved >= args.num_images:
            break

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image_path",
            "class_name",
            "baseline_inside",
            "baseline_outside",
            "baseline_ratio",
            "cam_inside",
            "cam_outside",
            "cam_ratio",
            "ratio_improvement",
            "inside_improvement",
            "outside_reduction",
            "visualization",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Done.")
    print("Saved visualizations:", vis_dir)
    print("Saved metrics:", metrics_path)

    if len(rows) > 0:
        avg_baseline_ratio = sum(r["baseline_ratio"] for r in rows) / len(rows)
        avg_cam_ratio = sum(r["cam_ratio"] for r in rows) / len(rows)

        avg_baseline_inside = sum(r["baseline_inside"] for r in rows) / len(rows)
        avg_cam_inside = sum(r["cam_inside"] for r in rows) / len(rows)

        avg_baseline_outside = sum(r["baseline_outside"] for r in rows) / len(rows)
        avg_cam_outside = sum(r["cam_outside"] for r in rows) / len(rows)

        print()
        print("Average metrics:")
        print(f"Baseline inside:  {avg_baseline_inside:.4f}")
        print(f"CAM-loss inside:  {avg_cam_inside:.4f}")
        print(f"Baseline outside: {avg_baseline_outside:.4f}")
        print(f"CAM-loss outside: {avg_cam_outside:.4f}")
        print(f"Baseline ratio:   {avg_baseline_ratio:.4f}")
        print(f"CAM-loss ratio:   {avg_cam_ratio:.4f}")


if __name__ == "__main__":
    main()