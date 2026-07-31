"""Look for outlier activations in the backbone features feeding the CAM head.

Run::

    python -m src.debug_features --checkpoint runs/<timestamp>/best_resnet18_cam.pth

A very high max/median ratio in a channel means one spatial position dominates it,
which usually shows up as a hot speck in the CAM.
"""

import argparse

import torch
import yaml

from src.data import get_dataloaders
from src.mask_config_utils import build_mask_config
from src.model import ResNet18CAM
from src.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="defaults to the newest under runs/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    mask_root = config.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
    else:
        mask_dirs = config.get("mask_dirs", {})
        mask_manifests = config.get("mask_manifests", {})

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 8),
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=0,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        test_ratio=config.get("test_ratio", 0.1),
        augment=False,
        strict_mask_coverage=False,
    )
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    print(f"Classes: {classes}")

    model = ResNet18CAM(
        num_classes=len(classes),
        pretrained=False,
        dilate_last_block=bool(config.get("dilate_last_block", False)),
    )
    model = load_checkpoint(model, device=device, path=args.checkpoint, checkpoint_dir=args.runs_dir)
    model.to(device).eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["image"].to(device)
            paths = batch["image_path"]

            features = model.backbone(x)

            b, c, h, w = features.shape
            flat = features.reshape(b, c, -1).float()

            max_val = flat.abs().max(dim=2)[0]
            median_val = flat.abs().median(dim=2)[0]
            ratio = max_val / (median_val + 1e-6)

            print(f"\n===== Batch {batch_idx} =====")
            print(f"Feature map shape: {tuple(features.shape)}  (spatial pixels per map = {h * w})")
            print(f"Max ratio in batch:  {ratio.max().item():.2f}")
            print(f"Mean ratio in batch: {ratio.mean().item():.2f}")

            for i in range(b):
                fname = paths[i].replace("\\", "/").split("/")[-1]
                print(f"--- Sample {i} ({fname}) ---")
                for ch in ratio[i].topk(3).indices:
                    pos = flat[i, ch].argmax().item()
                    print(
                        f"  channel={ch.item():3d}  ratio={ratio[i, ch].item():7.2f}  "
                        f"outlier_pos=(row={pos // w}, col={pos % w})"
                    )

            if batch_idx + 1 >= args.batches:
                break


if __name__ == "__main__":
    main()
