"""
Per-epoch CAM snapshots on a fixed set of validation images.

The end-of-run `cam_debug` overlays show where the CAM landed; they cannot show
how it got there. These snapshots pin the same images across every epoch, so
the run produces a flip-book: mask drifting away from the object, the CAM
collapsing to a single cell, the background quietly staying positive.

Each row is:

    image | mask | CAM (min-max) | sign map | overlay + mask outline

The sign map matters more than it looks. `CAMMaskLoss` applies `relu` before
measuring energy, so any cell that is already negative contributes nothing and
receives exactly zero gradient from the CAM term, while GAP still pulls it back
up through the classification loss. Red/blue tells you at a glance which parts
of the background the CAM loss can still influence.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_CAPTION_HEIGHT = 46
_HEADER_HEIGHT = 22


def _get_font():
    try:
        from PIL import ImageFont
        try:
            return ImageFont.load_default(size=14)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def _denormalize(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _jet(values01: np.ndarray) -> np.ndarray:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        return (cm.jet(values01)[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        gray = (values01 * 255).astype(np.uint8)
        return np.stack([gray, np.zeros_like(gray), 255 - gray], axis=2)


def _upsample(cam: torch.Tensor, size: int) -> np.ndarray:
    resized = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy()


def _sign_map(cam_full: np.ndarray) -> np.ndarray:
    """Red where the CAM is positive, blue where it is negative."""
    scale = float(np.abs(cam_full).max()) + 1e-6
    normalized = np.clip(cam_full / scale, -1.0, 1.0)

    out = np.zeros((*cam_full.shape, 3), dtype=np.float32)
    positive = np.clip(normalized, 0, 1)
    negative = np.clip(-normalized, 0, 1)
    out[:, :, 0] = 40 + 215 * positive
    out[:, :, 2] = 40 + 215 * negative
    out[:, :, 1] = 40 * (1 - positive - negative)
    return out.clip(0, 255).astype(np.uint8)


def _mask_boundary(binary: np.ndarray) -> np.ndarray:
    m = binary > 0.5
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def _with_header(tile: np.ndarray, title: str, font) -> np.ndarray:
    header = Image.new("RGB", (tile.shape[1], _HEADER_HEIGHT), (18, 18, 18))
    ImageDraw.Draw(header).text((4, 4), title, fill=(235, 235, 235), font=font)
    return np.concatenate([np.asarray(header), tile], axis=0)


class CAMSnapshotter:
    """Holds a fixed sample of images and re-renders their CAMs every epoch."""

    def __init__(
        self,
        loader,
        device: str,
        classes,
        run_dir,
        num_samples: int = 8,
        seed: int = 42,
        max_scan_batches: int = 50,
    ):
        self.device = device
        self.classes = list(classes)
        self.out_dir = Path(run_dir) / "diagnostics" / "snapshots"
        self.font = _get_font()
        self.samples: List[dict] = []

        if loader is None or num_samples <= 0:
            return

        pool: List[dict] = []
        fallback: List[dict] = []

        for batch_index, batch in enumerate(loader):
            if batch_index >= max_scan_batches or len(pool) >= num_samples * 4:
                break
            for i in range(batch["image"].shape[0]):
                entry = {
                    "image": batch["image"][i].clone(),
                    "mask": batch["mask"][i].clone(),
                    "label": int(batch["label"][i].item()),
                    "path": str(batch["image_path"][i]),
                }
                if float(batch["has_mask"][i].item()) > 0.5:
                    pool.append(entry)
                elif len(fallback) < num_samples:
                    fallback.append(entry)

        if not pool:
            print(
                "[reports] snapshots: no masked validation samples found; "
                "falling back to unmasked images (mask panels will be empty)."
            )
            pool = fallback

        if not pool:
            return

        # Spread the picks over the scanned pool instead of taking a
        # contiguous prefix, which would be one or two batches only.
        step = max(1, len(pool) // num_samples)
        self.samples = pool[::step][:num_samples]

    def available(self) -> bool:
        return bool(self.samples)

    @torch.no_grad()
    def capture(self, model, epoch: int) -> Optional[str]:
        if not self.samples:
            return None

        was_training = model.training
        model.eval()

        try:
            images = torch.stack([s["image"] for s in self.samples]).to(self.device)
            labels = torch.tensor(
                [s["label"] for s in self.samples], device=self.device, dtype=torch.long
            )
            _, cams = model(images)
            cams = cams.detach().float()
            target_cams = cams[torch.arange(len(self.samples), device=cams.device), labels]
        finally:
            model.train(was_training)

        rows = [self._render_row(sample, target_cams[i].cpu())
                for i, sample in enumerate(self.samples)]

        width = max(row.shape[1] for row in rows)
        padded = [
            np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0)), constant_values=18)
            for row in rows
        ]
        grid = np.concatenate(padded, axis=0)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"epoch_{epoch:03d}.png"
        Image.fromarray(grid).save(path)
        return str(path)

    def _render_row(self, sample: dict, cam: torch.Tensor) -> np.ndarray:
        image_np = _denormalize(sample["image"])
        size = image_np.shape[0]

        mask_np = sample["mask"][0].cpu().numpy()
        binary = (mask_np > 0.5).astype(np.float32)

        cam_full = _upsample(cam, size)

        span = cam_full.max() - cam_full.min()
        cam_norm = (cam_full - cam_full.min()) / (span + 1e-6)
        heat = _jet(cam_norm)

        overlay = (0.55 * image_np + 0.45 * heat).astype(np.uint8)
        boundary = _mask_boundary(binary)
        overlay[boundary] = np.array([0, 255, 0], dtype=np.uint8)

        mask_tile = np.repeat((binary * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        tiles = [
            _with_header(image_np, "input", self.font),
            _with_header(mask_tile, "mask (white = object)", self.font),
            _with_header(heat, "CAM (min-max)", self.font),
            _with_header(_sign_map(cam_full), "sign (red +, blue -)", self.font),
            _with_header(overlay, "overlay + mask outline", self.font),
        ]
        row = np.concatenate(tiles, axis=1)
        return np.concatenate([row, self._caption(sample, cam, binary, row.shape[1])], axis=0)

    def _caption(
        self,
        sample: dict,
        cam: torch.Tensor,
        binary: np.ndarray,
        width: int,
    ) -> np.ndarray:
        hc, wc = cam.shape
        mask_cam = F.interpolate(
            torch.from_numpy(binary).unsqueeze(0).unsqueeze(0).float(),
            size=(hc, wc),
            mode="area",
        )[0, 0]

        inside = mask_cam
        outside = 1.0 - mask_cam
        positive = F.relu(cam)

        energy_out = float((positive * outside).sum())
        energy_in = float((positive * inside).sum())
        containment = energy_out / (energy_in + energy_out + 1e-6)
        uniform = float(outside.mean())

        mean_pos_out = energy_out / (float(outside.sum()) + 1e-6)
        mean_pos_in = energy_in / (float(inside.sum()) + 1e-6)

        class_name = self.classes[sample["label"]] if sample["label"] < len(self.classes) else "?"

        # A CAM that is negative everywhere scores containment = 0/eps = 0: a
        # perfect loss from an empty map. Say so, rather than letting the
        # min-max panel make it look like a localised heatmap.
        collapsed = float(cam.max()) <= 0.0
        warning = "  <<< CAM ENTIRELY NEGATIVE: loss is 0 for free" if collapsed else ""

        line_1 = (
            f"{Path(sample['path']).name}  |  class={class_name}  |  "
            f"containment={containment:.3f} (flat baseline={uniform:.3f}){warning}"
        )
        line_2 = (
            f"CAM range=[{float(cam.min()):+.3f}, {float(cam.max()):+.3f}]  |  "
            f"mean positive: object={mean_pos_in:.4f} background={mean_pos_out:.4f}  |  "
            f"mask coverage={float(inside.mean()):.3f}"
        )

        strip = Image.new("RGB", (width, _CAPTION_HEIGHT), (18, 18, 18))
        draw = ImageDraw.Draw(strip)
        draw.text((6, 5), line_1, fill=(245, 245, 245), font=self.font)
        draw.text((6, 25), line_2, fill=(190, 190, 190), font=self.font)
        return np.asarray(strip)
