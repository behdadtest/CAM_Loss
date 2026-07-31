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


def autocast_context(device: str, use_amp: bool):
    if not (device == "cuda" and use_amp):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):       # torch < 2.4
        return torch.cuda.amp.autocast()


def make_grad_scaler(device: str, use_amp: bool):
    enabled = bool(device == "cuda" and use_amp)
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):       # torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=enabled)


def lambda_for_epoch(lambda_cam: float, epoch: int, warmup_epochs: int) -> float:
    """Ramp the CAM weight in over the first few epochs.

    At step 0 the CAM head is random. Hitting it with full background suppression
    immediately rewards the all-zero map before anything has been learned, so the
    run can settle into that solution and never leave.
    """
    if warmup_epochs <= 0:
        return lambda_cam
    return lambda_cam * min(1.0, epoch / float(warmup_epochs))


class _Meter:
    """Sum/count average, so a short final batch does not count as a full one."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        if n <= 0:
            return
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def _cam_head_parameters(model):
    head = getattr(model, "cam_conv", None)
    if head is None:
        return []
    return [p for p in head.parameters() if p.requires_grad]


def grad_norm_ratio(loss_cls, loss_cam, params):
    """||grad of CE|| / ||grad of CAM loss|| at the CAM head.

    A far better guide to ``lambda_cam`` than the ratio of the loss *values*,
    which live on unrelated scales. Cheap: the head is the last layer, so neither
    backward pass touches the backbone.

    Under AMP both gradients carry the same scale factor, so the ratio is still
    meaningful, but small values may underflow in fp16 -- treat it as indicative.
    """
    if not params:
        return None

    def norm_of(loss):
        grads = torch.autograd.grad(
            loss, params, retain_graph=True, allow_unused=True
        )
        total = sum(
            (g.detach().float() ** 2).sum() for g in grads if g is not None
        )
        return float(torch.sqrt(total).item()) if torch.is_tensor(total) else 0.0

    cam_norm = norm_of(loss_cam)
    if cam_norm <= 1e-12:
        return None
    return norm_of(loss_cls) / cam_norm


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
    grad_probe_every: int = 0,
):
    model.train()

    loss_meter, cls_meter, cam_meter, acc_meter = _Meter(), _Meter(), _Meter(), _Meter()
    probe_meter = _Meter()
    component_meters = {}

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for step, batch in enumerate(pbar):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        has_mask = batch["has_mask"].to(device, non_blocking=True)
        batch_size = x.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, use_amp):
            logits, cams = model(x)

        # Both losses in fp32: MSE-like terms on small values underflow to exactly
        # zero in fp16, which silently switches the regulariser off.
        loss_cls = ce_loss_fn(logits.float(), y)
        loss_cam = cam_loss_fn(cams=cams.float(), mask=masks, labels=y, has_mask=has_mask)
        loss = loss_cls + lambda_cam * loss_cam

        if grad_probe_every and step % grad_probe_every == 0:
            ratio = grad_norm_ratio(loss_cls, loss_cam, _cam_head_parameters(model))
            if ratio is not None:
                probe_meter.update(ratio)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        n_masked = getattr(cam_loss_fn, "last_num_valid", batch_size)

        loss_meter.update(loss.item(), batch_size)
        cls_meter.update(loss_cls.item(), batch_size)
        cam_meter.update(loss_cam.item(), n_masked)
        acc_meter.update(accuracy_from_logits(logits.detach(), y), batch_size)

        for name, value in getattr(cam_loss_fn, "last_components", {}).items():
            component_meters.setdefault(name, _Meter()).update(value, n_masked)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "cls": f"{loss_cls.item():.4f}",
            "cam": f"{loss_cam.item():.4f}",
            "acc": f"{acc_meter.avg:.3f}",
        })

    return {
        "loss": loss_meter.avg,
        "cls_loss": cls_meter.avg,
        "cam_loss": cam_meter.avg,
        "acc": acc_meter.avg,
        "masked_samples": cam_meter.count,
        "grad_norm_ratio": probe_meter.avg if probe_meter.count else None,
        "cam_components": {k: m.avg for k, m in component_meters.items()},
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

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Total loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="train_acc")
    axes[1].plot(epochs, history["val_acc"], label="val_acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_cls_loss"], label="train_cls")
    axes[2].plot(epochs, history["train_cam_loss"], label="train_cam (raw)")
    axes[2].plot(epochs, history["val_cam_loss"], label="val_cam (raw)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Loss terms (unweighted)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def analyze_loss_balance(history, run_dir):
    """Summarise the two loss terms to help choose ``lambda_cam``.

    The loss-value ratio is reported for continuity, but it compares quantities on
    unrelated scales -- prefer ``grad_norm_ratio``, which measures which loss is
    actually winning at the CAM head.
    """
    train_cls = history.get("train_cls_loss", [])
    train_cam = history.get("train_cam_loss", [])
    val_cls = history.get("val_cls_loss", [])
    val_cam = history.get("val_cam_loss", [])
    grad_ratios = [r for r in history.get("grad_norm_ratio", []) if r is not None]

    if not train_cls or not train_cam:
        print("Not enough data for a loss balance analysis.")
        return None

    def safe_ratio(a, b):
        return (a / b) if b > 1e-8 else None

    train_ratios = [safe_ratio(c, cam) for c, cam in zip(train_cls, train_cam)]
    val_ratios = [safe_ratio(c, cam) for c, cam in zip(val_cls, val_cam)] if val_cls else []

    finite_train = [r for r in train_ratios if r is not None]
    finite_val = [r for r in val_ratios if r is not None]

    summary = {
        "train_cls_loss_mean": statistics.mean(train_cls),
        "train_cls_loss_last": train_cls[-1],
        "train_cam_loss_mean": statistics.mean(train_cam),
        "train_cam_loss_last": train_cam[-1],
        "train_cls_to_cam_ratio_per_epoch": train_ratios,
        "val_cls_to_cam_ratio_per_epoch": val_ratios,
        "loss_value_ratio_mean": statistics.mean(finite_train) if finite_train else None,
        "loss_value_ratio_last": train_ratios[-1] if train_ratios else None,
        "loss_value_ratio_val_mean": statistics.mean(finite_val) if finite_val else None,
        # The one worth acting on: >1 means CE dominates at the head, <1 means the
        # CAM loss does. Aim for the same order of magnitude.
        "grad_norm_ratio_per_epoch": history.get("grad_norm_ratio", []),
        "grad_norm_ratio_mean": statistics.mean(grad_ratios) if grad_ratios else None,
        "grad_norm_ratio_last": grad_ratios[-1] if grad_ratios else None,
    }

    out_path = Path(run_dir) / "loss_balance_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n--- Loss balance summary ---")
    print(f"cls_loss (train): mean={summary['train_cls_loss_mean']:.4f} last={summary['train_cls_loss_last']:.4f}")
    print(f"cam_loss (train): mean={summary['train_cam_loss_mean']:.4f} last={summary['train_cam_loss_last']:.4f}")
    if summary["grad_norm_ratio_mean"] is not None:
        print(
            f"grad-norm ratio ||dCE|| / ||dCAM|| at the head: "
            f"mean={summary['grad_norm_ratio_mean']:.3f} "
            f"last={summary['grad_norm_ratio_last']:.3f}   "
            f"(multiply lambda_cam by this to equalise them)"
        )
    else:
        print("grad-norm ratio not collected (set grad_probe_every > 0 to enable).")
    print(f"saved: {out_path}")

    return summary


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

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    warmup_epochs = int(config.get("cam_warmup_epochs", 0))
    grad_probe_every = int(config.get("grad_probe_every", 0))

    model_info = {
        "model_class": model.__class__.__name__,
        "num_classes": len(classes),
        "classes": classes,
        "num_parameters": num_params,
        "trainable_parameters": num_trainable,
        "cam_loss_class": cam_loss_fn.__class__.__name__,
        "lambda_cam": lambda_cam,
        "cam_warmup_epochs": warmup_epochs,
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
    best_val_acc = -1.0

    early_stopping_enabled = bool(config.get("early_stopping", False))
    early_stopping_monitor = str(config.get("early_stopping_monitor", "val_acc"))
    early_stopping_patience = int(config.get("early_stopping_patience", 10))
    early_stopping_min_delta = float(config.get("early_stopping_min_delta", 0.0))
    early_stopping_mode = "min" if "loss" in early_stopping_monitor else "max"

    best_monitor_value = float("-inf") if early_stopping_mode == "max" else float("inf")

    epochs_no_improve = 0
    stopped_early = False
    last_epoch_ran = 0

    history = {
        "epoch": [],
        "lambda_eff": [],
        "train_loss": [],
        "train_cls_loss": [],
        "train_cam_loss": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_cam_loss": [],
        "train_acc": [],
        "val_acc": [],
        "grad_norm_ratio": [],
    }

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch", "lr", "lambda_eff",
            "train_loss", "train_cls_loss", "train_cam_loss", "train_acc",
            "val_loss", "val_cls_loss", "val_cam_loss", "val_acc",
            "grad_norm_ratio", "train_masked_samples",
        ])

        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]["lr"]
            lambda_eff = lambda_for_epoch(lambda_cam, epoch, warmup_epochs)

            print(f"\nEpoch [{epoch}/{epochs}] | lr={current_lr:.8f} | lambda_cam={lambda_eff:.4f}")

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                ce_loss_fn=ce_loss_fn,
                cam_loss_fn=cam_loss_fn,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                lambda_cam=lambda_eff,
                use_amp=use_amp,
                grad_probe_every=grad_probe_every,
            )

            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                ce_loss_fn=ce_loss_fn,
                cam_loss_fn=cam_loss_fn,
                device=device,
                lambda_cam=lambda_eff,
            )

            print(
                f"Train | loss={train_metrics['loss']:.4f} "
                f"cls={train_metrics['cls_loss']:.4f} "
                f"cam={train_metrics['cam_loss']:.4f} "
                f"acc={train_metrics['acc']:.4f} "
                f"(masked samples: {train_metrics['masked_samples']})"
            )
            if train_metrics["cam_components"]:
                parts = "  ".join(
                    f"{k}={v:.4f}" for k, v in sorted(train_metrics["cam_components"].items())
                )
                print(f"      | cam terms: {parts}")
            if train_metrics["grad_norm_ratio"] is not None:
                print(f"      | ||dCE|| / ||dCAM|| at head = {train_metrics['grad_norm_ratio']:.3f}")

            print(
                f"Val   | loss={val_metrics['loss']:.4f} "
                f"cls={val_metrics['cls_loss']:.4f} "
                f"cam={val_metrics['cam_loss']:.4f} "
                f"acc={val_metrics['acc']:.4f}"
            )

            if scheduler is not None:
                scheduler.step()

            writer.writerow([
                epoch,
                current_lr,
                lambda_eff,
                train_metrics["loss"],
                train_metrics["cls_loss"],
                train_metrics["cam_loss"],
                train_metrics["acc"],
                val_metrics["loss"],
                val_metrics["cls_loss"],
                val_metrics["cam_loss"],
                val_metrics["acc"],
                train_metrics["grad_norm_ratio"],
                train_metrics["masked_samples"],
            ])
            f.flush()

            history["epoch"].append(epoch)
            history["lambda_eff"].append(lambda_eff)
            history["train_loss"].append(train_metrics["loss"])
            history["train_cls_loss"].append(train_metrics["cls_loss"])
            history["train_cam_loss"].append(train_metrics["cam_loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_cls_loss"].append(val_metrics["cls_loss"])
            history["val_cam_loss"].append(val_metrics["cam_loss"])
            history["train_acc"].append(train_metrics["acc"])
            history["val_acc"].append(val_metrics["acc"])
            history["grad_norm_ratio"].append(train_metrics["grad_norm_ratio"])

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                classes=classes,
                path=run_dir / "last_resnet18_cam.pth",
                extra={"val_acc": val_metrics["acc"], "lambda_cam": lambda_cam, "lr": current_lr},
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    classes=classes,
                    path=run_dir / "best_resnet18_cam.pth",
                    extra={"val_acc": best_val_acc, "lambda_cam": lambda_cam, "lr": current_lr},
                )

                print(f"Best model saved. Val Acc: {best_val_acc:.4f}")

            last_epoch_ran = epoch

            if early_stopping_enabled:
                monitor_value = None
                if early_stopping_monitor.startswith("val_"):
                    monitor_value = val_metrics.get(early_stopping_monitor[4:])
                elif early_stopping_monitor.startswith("train_"):
                    monitor_value = train_metrics.get(early_stopping_monitor[6:])

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

    cam_summary = None
    if test_loader is not None and bool(config.get("cam_debug", True)):
        cam_summary = run_cam_debug(
            model=model,
            test_loader=test_loader,
            classes=classes,
            device=device,
            run_dir=run_dir,
            top_k=int(config.get("cam_debug_top_k", 20)),
            cam_iou_threshold=float(config.get("cam_debug_iou_threshold", 0.5)),
        )

    print("Training finished.")
    print("Best val acc:", best_val_acc)
    print("Run directory:", run_dir)
    if stopped_early:
        print(f"Stopped early after epoch {last_epoch_ran}/{epochs}.")

    return {
        "run_dir": run_dir,
        "best_val_acc": max(best_val_acc, 0.0),
        "history": history,
        "stopped_early": stopped_early,
        "last_epoch_ran": last_epoch_ran,
        "cam_summary": cam_summary,
    }
