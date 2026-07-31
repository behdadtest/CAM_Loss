import torch
from tqdm import tqdm

from src.train import _Meter, accuracy_from_logits, autocast_context


@torch.no_grad()
def evaluate(model, dataloader, ce_loss_fn, cam_loss_fn, device, lambda_cam: float, use_amp: bool = False):
    model.eval()

    loss_meter, cls_meter, cam_meter, acc_meter = _Meter(), _Meter(), _Meter(), _Meter()

    for batch in tqdm(dataloader, desc="Val", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)
        batch_size = x.size(0)

        with autocast_context(device, use_amp):
            logits, cams = model(x)

        loss_cls = ce_loss_fn(logits.float(), y)
        loss_cam = cam_loss_fn(cams=cams.float(), mask=masks, labels=y, has_mask=has_mask)
        loss = loss_cls + lambda_cam * loss_cam

        n_masked = getattr(cam_loss_fn, "last_num_valid", batch_size)

        # Weighted by sample count, not by batch: a short final batch must not
        # count the same as a full one.
        loss_meter.update(loss.item(), batch_size)
        cls_meter.update(loss_cls.item(), batch_size)
        cam_meter.update(loss_cam.item(), n_masked)
        acc_meter.update(accuracy_from_logits(logits, y), batch_size)

    return {
        "loss": loss_meter.avg,
        "cls_loss": cls_meter.avg,
        "cam_loss": cam_meter.avg,
        "acc": acc_meter.avg,
        "masked_samples": cam_meter.count,
    }
