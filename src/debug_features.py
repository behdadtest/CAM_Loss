"""
debug_features.py

بررسی وجود outlier در feature map های ResNet18CAM، با استفاده از دیتالودر واقعی پروژه.
"""

import torch

from model import ResNet18CAM        # مسیر رو مطابق پروژه‌ی خودتون اصلاح کنید
from data import get_dataloaders      # مسیر رو مطابق پروژه‌ی خودتون اصلاح کنید


# ---------------- تنظیمات ----------------
DATA_DIR = r"D:\CVLab\mainProj\cats_vs_dogs_folder"
CHECKPOINT_PATH = r"D:\CVLab\mainProj\runs\2026-07-22_09-42-36\best_resnet18_cam.pth"
IMAGE_SIZE = 224          # همون سایزی که موقع train استفاده کردید
BATCH_SIZE = 8
VAL_RATIO = 0.0
TEST_RATIO = 0.0
SEED = 42
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_BATCHES_TO_CHECK = 3
# ------------------------------------------


def main():
    print(f"Device: {DEVICE}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        mask_dirs=None,       # برای این تست به ماسک نیازی نداریم
        mask_manifests=None,
        test_ratio=TEST_RATIO,
    )

    print(f"Classes: {classes}")

    model = ResNet18CAM(num_classes=len(classes), pretrained=False)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_loader):
            x = batch["image"].to(DEVICE)
            paths = batch["image_path"]

            logits, features = model(x)

            b, c, h, w = features.shape
            flat = features.reshape(b, c, -1).float()

            max_val = flat.abs().max(dim=2)[0]         # [B, C]
            median_val = flat.abs().median(dim=2)[0]   # [B, C]
            ratio = max_val / (median_val + 1e-6)

            print(f"\n===== Batch {batch_idx} =====")
            print(f"Feature map shape: {features.shape}  (spatial pixels per map = {h * w})")
            print(f"Max ratio in batch: {ratio.max().item():.2f}")
            print(f"Mean ratio in batch: {ratio.mean().item():.2f}")

            for i in range(b):
                fname = paths[i].split("\\")[-1].split("/")[-1]
                top_channels = ratio[i].topk(3).indices
                print(f"--- Sample {i} ({fname}) ---")
                for ch in top_channels:
                    pos = flat[i, ch].argmax().item()
                    row, col = pos // w, pos % w
                    print(
                        f"  channel={ch.item():3d}  ratio={ratio[i, ch].item():7.2f}  "
                        f"outlier_pos=(row={row}, col={col})"
                    )

            if batch_idx + 1 >= NUM_BATCHES_TO_CHECK:
                break


if __name__ == "__main__":
    main()