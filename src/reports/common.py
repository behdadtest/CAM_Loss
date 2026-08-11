"""
Shared helpers for the diagnostic report modules.

Console output from this package is deliberately ASCII-only: these reports are
meant to survive being piped into a log file on Windows, where a non-UTF-8
codepage turns any non-ASCII character into a UnicodeEncodeError.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

EPS = 1e-6

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"
OK = "OK"

_LEVEL_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2, OK: 3}


# ----------------------------------------------------------------------
# Findings
# ----------------------------------------------------------------------

def finding(level: str, code: str, message: str) -> Dict[str, str]:
    return {"level": level, "code": code, "message": message}


def sort_findings(findings: Sequence[dict]) -> List[dict]:
    return sorted(findings, key=lambda f: _LEVEL_ORDER.get(f["level"], 9))


def render_findings(findings: Sequence[dict]) -> List[str]:
    if not findings:
        return ["- (no findings)"]
    return [
        f"- **[{f['level']}]** `{f['code']}` &mdash; {f['message']}"
        for f in sort_findings(findings)
    ]


def print_findings(findings: Sequence[dict], header: str) -> None:
    print(f"\n--- {header} ---")
    if not findings:
        print("  (no findings)")
        return
    for f in sort_findings(findings):
        print(f"  [{f['level']:<8}] {f['code']}: {f['message']}")


def worst_level(findings: Sequence[dict]) -> str:
    if not findings:
        return OK
    return sort_findings(findings)[0]["level"]


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

def resolve_base_dataset(dataset):
    """Unwrap nested `Subset`s into (base_dataset, indices_into_base)."""
    if isinstance(dataset, Subset):
        base, inner = resolve_base_dataset(dataset.dataset)
        return base, [inner[i] for i in dataset.indices]
    return dataset, list(range(len(dataset)))


def loader_base_and_indices(loader):
    if loader is None:
        return None, []
    return resolve_base_dataset(loader.dataset)


# ----------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------

def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def summarize(name: str, values: Sequence[float], full: bool = False) -> Dict[str, Optional[float]]:
    """Mean (and optionally spread) for one metric, with stable key names."""
    keys = [f"{name}_mean"]
    if full:
        keys += [f"{name}_std", f"{name}_p10", f"{name}_median", f"{name}_p90"]

    if len(values) == 0:
        return {k: None for k in keys}

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: None for k in keys}

    out: Dict[str, Optional[float]] = {f"{name}_mean": float(arr.mean())}
    if full:
        out[f"{name}_std"] = float(arr.std())
        out[f"{name}_p10"] = float(np.percentile(arr, 10))
        out[f"{name}_median"] = float(np.median(arr))
        out[f"{name}_p90"] = float(np.percentile(arr, 90))
    return out


def nan_series(values: Sequence[Optional[float]]) -> np.ndarray:
    """Turn a list that may contain `None` into a plottable float array."""
    return np.array(
        [np.nan if v is None else float(v) for v in values],
        dtype=np.float64,
    )


def downsample_mask(mask: torch.Tensor, size) -> torch.Tensor:
    """Match exactly what `CAMMaskLoss` does to the mask before using it."""
    if mask.shape[-2:] != tuple(size):
        return F.interpolate(mask.float(), size=tuple(size), mode="area")
    return mask.float()


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return to_serializable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_serializable(obj.detach().cpu().tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def get_pyplot():
    """Return `matplotlib.pyplot` with the Agg backend, or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[reports] matplotlib not available, skipping plots.")
        return None


def safe_call(label: str, run_dir, fn, /, *args, **kwargs):
    """
    Run a diagnostic and never let it kill the training run.

    A broken report is an annoyance; a training run that dies at epoch 19
    because of a broken report is a lost afternoon.

    The first three parameters are positional-only so that `run_dir=...` can be
    forwarded to `fn` without colliding with this function's own arguments.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - diagnostics must not be fatal
        print(f"[reports] '{label}' failed: {type(exc).__name__}: {exc}")
        try:
            errors_path = Path(run_dir) / "diagnostics" / "report_errors.log"
            errors_path.parent.mkdir(parents=True, exist_ok=True)
            with open(errors_path, "a", encoding="utf-8") as f:
                f.write(f"===== {label} =====\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception:  # noqa: BLE001
            pass
        return None
