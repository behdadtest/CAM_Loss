from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import get_dataloaders
from src.losses import CAMMaskLoss
from src.model import ResNet18CAM
from src.train import run_training
from src.utils import get_device, set_seed


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    set_seed(config.get("seed", 42))

    device = get_device()
    print(f"Using device: {device}")

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 32),
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 0),
        mask_dirs=config.get("mask_dirs", {}),
        mask_manifests=config.get("mask_manifests", {}),
        test_ratio=config.get("test_ratio", 0.1),
    )

    num_classes = len(classes)
    print(f"Detected classes ({num_classes}): {classes}")

    model = ResNet18CAM(
        num_classes=num_classes,
        pretrained=bool(config.get("pretrained", True)),
    ).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    cam_loss_fn = CAMMaskLoss(
        outside_weight=float(config.get("outside_weight", 1.0)),
        non_target_weight=float(config.get("non_target_weight", 1.0)),
    )

    lambda_cam = float(config.get("lambda_cam", 0.05))
    use_amp = bool(config.get("use_amp", True))
    epochs = int(config.get("epochs", 10))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    scheduler = None

    if str(config.get("scheduler", "none")).lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(config.get("eta_min", 0.0)),
        )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(device == "cuda" and use_amp)
    )

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classes=classes,
        ce_loss_fn=ce_loss_fn,
        cam_loss_fn=cam_loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        lambda_cam=lambda_cam,
        use_amp=use_amp,
        epochs=epochs,
        config=config,
        runs_dir=config.get("runs_dir", "runs"),
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()