import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cudnn.benchmark picks algorithms by timing, which makes runs differ from
        # each other. Worth turning off when comparing lambda values, since at low
        # shot counts that noise can be larger than the effect being measured.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_checkpoint(model, optimizer, epoch, classes, path: str, extra: dict = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "classes": classes,
    }

    if extra:
        payload.update(extra)

    torch.save(payload, path)


def latest_checkpoint_path(checkpoint_dir="runs", pattern="best_*.pth") -> Optional[Path]:
    """Most recently modified checkpoint under ``checkpoint_dir``.

    Searches recursively: checkpoints are saved as
    ``runs/<timestamp>/best_resnet18_cam.pth``, so a non-recursive glob on
    ``runs/`` never matched anything and every caller got ``None``.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.rglob(pattern))
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_checkpoint(model, optimizer=None, device=None, path=None, checkpoint_dir="runs"):
    """Load weights from ``path``, or from the newest checkpoint under ``runs/``."""
    if path is None:
        path = latest_checkpoint_path(checkpoint_dir)

    if path is None:
        raise FileNotFoundError(
            f"No checkpoint found under '{checkpoint_dir}'. "
            f"Pass path=... explicitly, or train a model first."
        )

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)

    if optimizer is not None and isinstance(checkpoint, dict):
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

    print(f"Loaded checkpoint: {path}")
    return model
