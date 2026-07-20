import csv
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.losses import build_all_cams


def _autocast_context(device: str, use_amp: bool):
    if device == "cuda" and use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(
    model,
    dataloader,
    ce_loss_fn,
    cam_loss_fn,
    optimizer,
    scaler,
    device,
    lambda_cam: float,
    use_amp: bool,
):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_cam_loss = 0.0
    total_acc = 0.0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, use_amp):
            logits, features = model(x)
            loss_cls = ce_loss_fn(logits, y)

            # All class CAMs: [B, num_classes, H, W]
            cams = build_all_cams(features, model.fc.weight)

            # CAM loss uses the target-class CAM for each sample.
            loss_cam = cam_loss_fn(cams=cams, mask=masks, labels=y, has_mask=has_mask)

            loss = loss_cls + lambda_cam * loss_cam

        if device == "cuda" and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc = accuracy_from_logits(logits.detach(), y)

        total_loss += loss.item()
        total_cls_loss += loss_cls.item()
        total_cam_loss += loss_cam.item()
        total_acc += acc

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{loss_cls.item():.4f}",
            "cam": f"{loss_cam.item():.4f}",
            "acc": f"{acc:.3f}",
        })

    n = max(1, len(dataloader))
    return {
        "loss": total_loss / n,
        "cls_loss": total_cls_loss / n,
        "cam_loss": total_cam_loss / n,
        "acc": total_acc / n,
    }


def _plot_learning_curve(history, save_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping learning curve plot.")
        return

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_cam_debug(
    model,
    test_loader,
    classes,
    device,
    run_dir,
    top_k: int = 20,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225),
):

    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        from PIL import Image
    except ImportError:
        print("matplotlib/PIL/numpy not available, skipping CAM debug visualization.")
        return

    if test_loader is None or len(test_loader.dataset) == 0:
        print("No test set available, skipping CAM debug visualization.")
        return

    model.eval()

    ce_none = torch.nn.CrossEntropyLoss(reduction="none")

    mean_t = torch.tensor(normalize_mean).view(1, 3, 1, 1)
    std_t = torch.tensor(normalize_std).view(1, 3, 1, 1)

    records = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            paths = batch["image_path"]

            logits, features = model(x)
            losses = ce_none(logits, y)
            preds = logits.argmax(dim=1)

            # All class CAMs: [B, num_classes, Hc, Wc]
            cams = build_all_cams(features, model.fc.weight)

            for i in range(x.size(0)):
                records.append({
                    "loss": losses[i].item(),
                    "image": x[i].detach().cpu(),
                    "true_label": y[i].item(),
                    "pred_label": preds[i].item(),
                    # CAM for the class the model actually predicted.
                    "cam": cams[i, preds[i]].detach().cpu(),
                    "image_path": paths[i],
                })

    records.sort(key=lambda r: r["loss"], reverse=True)
    worst = records[:top_k]

    cam_dir = Path(run_dir) / "cam_debug"
    cam_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for rank, r in enumerate(worst, start=1):
        # Undo normalization to get a viewable RGB image.
        img = r["image"].unsqueeze(0) * std_t + mean_t
        img = img.clamp(0, 1).squeeze(0)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")

        # Min-max normalize the CAM to [0, 1] before colorizing.
        cam = r["cam"]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        cam_np = cam.numpy()

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
            f"worst_{rank:03d}_true-{true_name}_pred-{pred_name}_"
            f"loss-{r['loss']:.3f}.png"
        )

        Image.fromarray(overlay).save(cam_dir / fname)

        summary_rows.append([rank, r["image_path"], true_name, pred_name, r["loss"]])

    with open(cam_dir / "worst_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "image_path", "true_label", "pred_label", "loss"])
        writer.writerows(summary_rows)

    print(f"Saved {len(worst)} worst-case CAM overlays to {cam_dir}")


def run_training(
    model,
    train_loader,
    val_loader,
    classes,
    ce_loss_fn,
    cam_loss_fn,
    optimizer,
    scheduler,
    scaler,
    device,
    lambda_cam: float,
    use_amp: bool,
    epochs: int,
    config: dict,
    runs_dir: str = "runs",
    test_loader=None,
):
    # Local imports to avoid circular imports at module load time.
    from src.evaluate import evaluate
    from src.utils import save_checkpoint

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(runs_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the exact config used for this run.
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    # Save model / run metadata.
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        "model_class": model.__class__.__name__,
        "num_classes": len(classes),
        "classes": classes,
        "num_parameters": num_params,
        "trainable_parameters": num_trainable,
        "lambda_cam": lambda_cam,
        "use_amp": use_amp,
        "epochs": epochs,
        "device": str(device),
        "timestamp": timestamp,
        "early_stopping": bool(config.get("early_stopping", False)),
        "early_stopping_monitor": str(config.get("early_stopping_monitor", "val_acc")),
        "early_stopping_patience": int(config.get("early_stopping_patience", 10)),
        "early_stopping_min_delta": float(config.get("early_stopping_min_delta", 0.0)),
    }
    with open(run_dir / "model_info.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    log_path = run_dir / "training_log.csv"
    best_val_acc = 0.0

    early_stopping_enabled = bool(config.get("early_stopping", False))
    early_stopping_monitor = str(config.get("early_stopping_monitor", "val_acc"))
    early_stopping_patience = int(config.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    early_stopping_mode = "min" if "loss" in early_stopping_monitor else "max"

    if early_stopping_mode == "max":
        best_monitor_value = float("-inf")
    else:
        best_monitor_value = float("inf")

    epochs_no_improve = 0
    stopped_early = False
    last_epoch_ran = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch", "lr",
            "train_loss", "train_cls_loss", "train_cam_loss", "train_acc",
            "val_loss", "val_cls_loss", "val_cam_loss", "val_acc",
        ])

        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"\nEpoch [{epoch}/{epochs}] | lr={current_lr:.8f}")

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                ce_loss_fn=ce_loss_fn,
                cam_loss_fn=cam_loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                lambda_cam=lambda_cam,
                use_amp=use_amp,
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                ce_loss_fn=ce_loss_fn,
                cam_loss_fn=cam_loss_fn,
                device=device,
                lambda_cam=lambda_cam,
            )

            print(
                f"Train | loss={train_metrics['loss']:.4f} "
                f"cls={train_metrics['cls_loss']:.4f} "
                f"cam={train_metrics['cam_loss']:.4f} "
                f"acc={train_metrics['acc']:.4f}"
            )

            print(
                f"Val   | loss={val_metrics['loss']:.4f} "
                f"cls={val_metrics['cls_loss']:.4f} "
                f"cam={val_metrics['cam_loss']:.4f} "
                f"acc={val_metrics['acc']:.4f} "
                f"lr={current_lr:.8f}"
            )

            if scheduler is not None:
                scheduler.step()

            writer.writerow([
                epoch,
                current_lr,
                train_metrics["loss"],
                train_metrics["cls_loss"],
                train_metrics["cam_loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["cls_loss"],
                val_metrics["cam_loss"],
                val_metrics["acc"],
            ])
            f.flush()

            history["epoch"].append(epoch)
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_acc"].append(val_metrics["acc"])

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                classes=classes,
                path=run_dir / "last_resnet18_cam.pth",
                extra={
                    "val_acc": val_metrics["acc"],
                    "lambda_cam": lambda_cam,
                    "lr": current_lr,
                },
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    classes=classes,
                    path=run_dir / "best_resnet18_cam.pth",
                    extra={
                        "val_acc": best_val_acc,
                        "lambda_cam": lambda_cam,
                        "lr": current_lr,
                    },
                )

                print(f"Best model saved. Val Acc: {best_val_acc:.4f}")

            last_epoch_ran = epoch

            if early_stopping_enabled:
                monitor_value = val_metrics.get(
                    early_stopping_monitor.replace("val_", "", 1), None
                ) if early_stopping_monitor.startswith("val_") else None
                if monitor_value is None:
                    # Fallback: allow monitoring train metrics too (e.g. "train_loss").
                    monitor_value = train_metrics.get(
                        early_stopping_monitor.replace("train_", "", 1), None
                    ) if early_stopping_monitor.startswith("train_") else None

                if monitor_value is None:
                    print(
                        f"Warning: early_stopping_monitor='{early_stopping_monitor}' "
                        f"not recognized, disabling early stopping."
                    )
                    early_stopping_enabled = False
                else:
                    if early_stopping_mode == "max":
                        improved = monitor_value > (best_monitor_value + early_stopping_min_delta)
                    else:
                        improved = monitor_value < (best_monitor_value - early_stopping_min_delta)

                    if improved:
                        best_monitor_value = monitor_value
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    print(
                        f"EarlyStopping | monitor={early_stopping_monitor} "
                        f"value={monitor_value:.4f} best={best_monitor_value:.4f} "
                        f"no_improve={epochs_no_improve}/{early_stopping_patience}"
                    )

                    if epochs_no_improve >= early_stopping_patience:
                        stopped_early = True
                        print(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(no improvement in '{early_stopping_monitor}' "
                            f"for {early_stopping_patience} epochs)."
                        )
                        break

    _plot_learning_curve(history, run_dir / "learning_curve.png")

    if test_loader is not None and bool(config.get("cam_debug", True)):
        run_cam_debug(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=device,
            run_dir=run_dir,
            top_k=int(config.get("cam_debug_top_k", 20)),
        )

    print("Training finished.")
    print("Best val acc:", best_val_acc)
    print("Run directory:", run_dir)
    if stopped_early:
        print(f"Stopped early after epoch {last_epoch_ran}/{epochs}.")

    return {
        "run_dir": run_dir,
        "best_val_acc": best_val_acc,
        "history": history,
        "stopped_early": stopped_early,
        "last_epoch_ran": last_epoch_ran,
    }