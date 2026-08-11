import csv
import json
import statistics
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.cam_debug import run_cam_debug
from src.reports.cam_stats import CAMStatsAccumulator, format_epoch_line
from src.reports.common import safe_call
from src.reports.final_report import write_cam_metrics_csv, write_final_report
from src.reports.grad_probe import format_probe_line, probe_cam_head, run_grad_probe
from src.reports.mask_audit import run_mask_audit
from src.reports.snapshots import CAMSnapshotter


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
    cam_stats=None,
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
            logits, cams = model(x)

            loss_cls = ce_loss_fn(logits, y)

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

        if cam_stats is not None:
            cam_stats.update(
                cams=cams.detach(), mask=masks, labels=y,
                has_mask=has_mask, logits=logits.detach(),
            )

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


def analyze_loss_balance(history, run_dir):
    """
    خلاصه‌ی معنادار از لاس‌های خام (بدون ضرب در lambda_cam) برای
    کمک به انتخاب ضریب lambda_cam.
    خروجی: یک JSON با آمار کلی + نسبت هر epoch، و پرینت در کنسول.
    """
    train_cls = history.get("train_cls_loss", [])
    train_cam = history.get("train_cam_loss", [])
    val_cls = history.get("val_cls_loss", [])
    val_cam = history.get("val_cam_loss", [])

    if not train_cls or not train_cam:
        print("داده‌ی کافی برای تحلیل loss balance وجود ندارد.")
        return None

    def safe_ratio(a, b):
        return (a / b) if b > 1e-8 else None

    train_ratios = [safe_ratio(c, cam) for c, cam in zip(train_cls, train_cam)]
    val_ratios = [safe_ratio(c, cam) for c, cam in zip(val_cls, val_cam)] if val_cls else []

    finite_train_ratios = [r for r in train_ratios if r is not None]
    finite_val_ratios = [r for r in val_ratios if r is not None]

    summary = {
        "train_cls_loss_mean": statistics.mean(train_cls),
        "train_cls_loss_last": train_cls[-1],
        "train_cam_loss_mean": statistics.mean(train_cam),
        "train_cam_loss_last": train_cam[-1],
        "train_cls_to_cam_ratio_per_epoch": train_ratios,
        "val_cls_to_cam_ratio_per_epoch": val_ratios,
        # پیشنهاد: lambda_cam ~ cls_loss / cam_loss تا دو ترم هم‌مقیاس بشن
        "suggested_lambda_cam_mean_over_epochs": (
            statistics.mean(finite_train_ratios) if finite_train_ratios else None
        ),
        "suggested_lambda_cam_last_epoch": (
            train_ratios[-1] if train_ratios and train_ratios[-1] is not None else None
        ),
        "suggested_lambda_cam_from_val_mean": (
            statistics.mean(finite_val_ratios) if finite_val_ratios else None
        ),
    }

    out_path = Path(run_dir) / "loss_balance_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n--- Loss balance summary (lambda_cam) ---")
    print(f"cls_loss (train): mean={summary['train_cls_loss_mean']:.4f} last={summary['train_cls_loss_last']:.4f}")
    print(f"cam_loss (train): mean={summary['train_cam_loss_mean']:.4f} last={summary['train_cam_loss_last']:.4f}")
    if summary["suggested_lambda_cam_mean_over_epochs"] is not None:
        print(f"recommended lambda_cam ( AVG epochs، cls/cam): "
              f"{summary['suggested_lambda_cam_mean_over_epochs']:.4f}")
    if summary["suggested_lambda_cam_last_epoch"] is not None:
        print(f"recommended lambda_cam (based on last epoch): "
              f"{summary['suggested_lambda_cam_last_epoch']:.4f}")
    print(f"saved: {out_path}")

    return summary


@torch.no_grad()
def infer_cam_grid(model, image_size: int, device: str):
    """Spatial size of the CAM the model produces, straight from a dummy forward."""
    was_training = model.training
    model.eval()
    try:
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        _, cams = model(dummy)
        return tuple(int(v) for v in cams.shape[-2:])
    finally:
        model.train(was_training)


def _flatten_metrics(prefix: str, metrics) -> dict:
    if not metrics:
        return {}
    return {f"{prefix}{key}": value for key, value in metrics.items()}


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
        "cam_loss_class": cam_loss_fn.__class__.__name__,
        "cam_loss_type": str(config.get("cam_loss_type", "ratio")),
        "cam_softmax_temperature": float(config.get("cam_softmax_temperature", 1.0)),
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

    # ------------------------------------------------------------------
    # Diagnostics setup. Every call here goes through `safe_call`: a broken
    # report must never take a training run down with it.
    # ------------------------------------------------------------------
    diagnostics_enabled = bool(config.get("diagnostics", True))
    image_size = int(config.get("image_size", 224))

    cam_grid = None
    mask_audit = None
    snapshotter = None
    cam_rows = []

    if diagnostics_enabled:
        cam_grid = safe_call(
            "infer_cam_grid", run_dir, infer_cam_grid, model, image_size, device
        )
        print(f"CAM grid: {cam_grid}")

        if bool(config.get("mask_audit", True)) and cam_grid is not None:
            mask_audit = safe_call(
                "mask_audit", run_dir, run_mask_audit,
                loaders={
                    "train": train_loader,
                    "val": val_loader,
                    "test": test_loader,
                },
                run_dir=run_dir,
                image_size=image_size,
                cam_size=cam_grid,
                max_decoded=config.get("mask_audit_max_decoded", 600),
                num_examples=int(config.get("mask_audit_examples", 12)),
                seed=int(config.get("seed", 42)),
            )

        if bool(config.get("cam_snapshots", True)):
            snapshotter = safe_call(
                "snapshotter_init", run_dir, CAMSnapshotter,
                loader=val_loader,
                device=device,
                classes=classes,
                run_dir=run_dir,
                num_samples=int(config.get("cam_snapshots_count", 8)),
                seed=int(config.get("seed", 42)),
            )

    softmax_temperature = float(config.get("cam_softmax_temperature", 1.0))
    collect_cam_stats = diagnostics_enabled and bool(config.get("cam_stats", True))
    grad_probe_enabled = diagnostics_enabled and bool(config.get("grad_probe", True))
    grad_probe_batches = int(config.get("grad_probe_batches", 3))
    snapshot_every = max(1, int(config.get("cam_snapshots_every", 1)))
    cam_metrics_path = run_dir / "diagnostics" / "cam_metrics.csv"

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
        "train_cls_loss": [],
        "train_cam_loss": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_cam_loss": [],
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

            train_cam_stats = (
                CAMStatsAccumulator(softmax_temperature=softmax_temperature)
                if collect_cam_stats else None
            )
            val_cam_stats = (
                CAMStatsAccumulator(softmax_temperature=softmax_temperature)
                if collect_cam_stats else None
            )

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
                cam_stats=train_cam_stats,
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                ce_loss_fn=ce_loss_fn,
                cam_loss_fn=cam_loss_fn,
                device=device,
                lambda_cam=lambda_cam,
                cam_stats=val_cam_stats,
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

            # --- diagnostics for this epoch -----------------------------
            if diagnostics_enabled:
                cam_row = {"epoch": epoch, "lr": current_lr}

                if train_cam_stats is not None:
                    train_summary = train_cam_stats.compute(prefix="train_")
                    val_summary = val_cam_stats.compute(prefix="val_")
                    cam_row.update(train_summary)
                    cam_row.update(val_summary)
                    print(format_epoch_line(train_summary, "train_", "train"))
                    print(format_epoch_line(val_summary, "val_", "val"))

                if grad_probe_enabled:
                    probe = safe_call(
                        "grad_probe", run_dir, run_grad_probe,
                        model=model,
                        dataloader=train_loader,
                        ce_loss_fn=ce_loss_fn,
                        cam_loss_fn=cam_loss_fn,
                        device=device,
                        lambda_cam=lambda_cam,
                        max_batches=grad_probe_batches,
                    )
                    cam_row.update(_flatten_metrics("grad_", probe))
                    print(format_probe_line(probe))

                cam_row.update(
                    safe_call("cam_head_probe", run_dir, probe_cam_head, model) or {}
                )

                cam_rows.append(cam_row)
                # Rewritten every epoch so a crashed run still leaves usable data.
                safe_call(
                    "cam_metrics_csv", run_dir,
                    write_cam_metrics_csv, cam_metrics_path, cam_rows,
                )

                if snapshotter is not None and epoch % snapshot_every == 0:
                    safe_call("snapshot", run_dir, snapshotter.capture, model, epoch)

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
            history["train_cls_loss"].append(train_metrics["cls_loss"])
            history["train_cam_loss"].append(train_metrics["cam_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_cam_loss"].append(val_metrics["cam_loss"])
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
    analyze_loss_balance(history, run_dir)

    if test_loader is not None and bool(config.get("cam_debug", True)):
        run_cam_debug(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=device,
            run_dir=run_dir,
            top_k=int(config.get("cam_debug_top_k", 20)),
            cam_iou_threshold=float(config.get("cam_debug_iou_threshold", 0.5)),
        )

    if diagnostics_enabled:
        safe_call(
            "final_report", run_dir, write_final_report,
            run_dir=run_dir,
            history=history,
            cam_rows=cam_rows,
            lambda_cam=lambda_cam,
            config=config,
            mask_audit=mask_audit,
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
        "cam_rows": cam_rows,
        "mask_audit": mask_audit,
    }
