from pathlib import Path

import torch
import torch.nn as nn
import yaml

from src.data import get_dataloaders
from src.losses import build_cam_loss
from src.mask_config_utils import build_mask_config
from src.model import ResNet18CAM
from src.model_with_adapter import ResNet18CAMWithAdapter
from src.train import make_grad_scaler, run_training
from src.utils import get_device, set_seed

CONFIG_PATH = Path("configs/config.yaml")


def load_config(config_path=CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    mask_root = config.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
        config["mask_dirs"] = mask_dirs
        config["mask_manifests"] = mask_manifests
    else:
        config.setdefault("mask_dirs", {})
        config.setdefault("mask_manifests", {})

    return config


def build_model(config: dict, num_classes: int) -> nn.Module:
    if bool(config.get("use_adapter", False)):
        return ResNet18CAMWithAdapter(
            num_classes=num_classes,
            pretrained=bool(config.get("pretrained", True)),
            gamma=int(config.get("adapter_gamma", 4)),
            adapter_kernel_size=int(config.get("adapter_kernel_size", 3)),
            freeze_backbone=bool(config.get("freeze_backbone", True)),
            dilate_last_block=bool(config.get("dilate_last_block", False)),
        )

    return ResNet18CAM(
        num_classes=num_classes,
        pretrained=bool(config.get("pretrained", True)),
        dilate_last_block=bool(config.get("dilate_last_block", False)),
    )


def build_dataloaders(config: dict):
    return get_dataloaders(
        data_dir=config["data_dir"],
        image_size=config.get("image_size", 224),
        batch_size=config.get("batch_size", 32),
        val_ratio=config.get("val_ratio", 0.2),
        seed=config.get("seed", 42),
        num_workers=config.get("num_workers", 0),
        mask_dirs=config.get("mask_dirs", {}),
        mask_manifests=config.get("mask_manifests", {}),
        test_ratio=config.get("test_ratio", 0.1),
        train_shots_per_class=config.get("train_shots_per_class", None),
        train_fraction=config.get("train_fraction", None),
        augment=bool(config.get("augment", True)),
        augment_kwargs=config.get("augment_kwargs", {}) or {},
        bg_randomize_prob=float(config.get("bg_randomize_prob", 0.0)),
        mask_extension=str(config.get("mask_extension", ".png")),
        mask_prefix=str(config.get("mask_prefix", "")),
        mask_suffix=str(config.get("mask_suffix", "")),
        min_fg_fraction=float(config.get("min_fg_fraction", 0.01)),
        max_fg_fraction=float(config.get("max_fg_fraction", 0.95)),
        strict_mask_coverage=bool(config.get("strict_mask_coverage", True)),
    )


def run(config: dict):
    set_seed(config.get("seed", 42), deterministic=bool(config.get("deterministic", False)))

    device = get_device()
    print(f"Using device: {device}")

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        if not bool(config.get("deterministic", False)):
            torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader, classes = build_dataloaders(config)

    num_classes = len(classes)
    print(f"Detected classes ({num_classes}): {classes}")

    model = build_model(config, num_classes).to(device)
    print(
        f"Model: {model.__class__.__name__} | "
        f"trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    ce_loss_fn = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("label_smoothing", 0.0))
    )
    cam_loss_fn = build_cam_loss(config)
    print(f"CAM loss: {cam_loss_fn.__class__.__name__}")

    lambda_cam = float(config.get("lambda_cam", 1.0))
    use_amp = bool(config.get("use_amp", False))
    epochs = int(config.get("epochs", 10))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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

    scaler = make_grad_scaler(device, use_amp)

    # output_dir is the run root; runs_dir kept as an alias for older configs.
    runs_dir = config.get("output_dir") or config.get("runs_dir") or "runs"

    return run_training(
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
        runs_dir=runs_dir,
        test_loader=test_loader,
    )


def main():
    run(load_config())


if __name__ == "__main__":
    main()
