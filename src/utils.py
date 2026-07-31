import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    
def latest_checkpoint_path(checkpoint_dir = "runs", pattern = "best_*.pth"):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob(pattern))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint
    
def load_checkpoint(model, optimizer=None, device=None):
    path = latest_checkpoint_path()
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model
