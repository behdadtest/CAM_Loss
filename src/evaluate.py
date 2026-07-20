import torch
from tqdm import tqdm

from src.losses import build_all_cams
from src.train import accuracy_from_logits


@torch.no_grad()
def evaluate(model, dataloader, ce_loss_fn, cam_loss_fn, device, lambda_cam: float):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_cam_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc="Val", leave=False)

    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        logits, features = model(x)
        loss_cls = ce_loss_fn(logits, y)

        cams = build_all_cams(features, model.fc.weight)  # [B, num_classes, H, W]
        loss_cam = cam_loss_fn(cams=cams, mask=masks, labels=y, has_mask=has_mask)

        loss = loss_cls + lambda_cam * loss_cam
        acc = accuracy_from_logits(logits, y)

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_cam_loss += loss_cam.item()
        total_acc += acc

    n = max(1, len(dataloader))
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "cam_loss": total_cam_loss / n,
        "acc": total_acc / n,
    }
