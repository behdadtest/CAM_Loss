"""Render CAM overlays for a random sample of images from a checkpoint.

Run from the project root::

    python -m src.debug.debug_cam_loss.cam_loss_debugger \\
        --checkpoint runs/<timestamp>/best_resnet18_cam.pth

Everything is read from ``configs/config.yaml`` unless overridden on the command
line; ``--checkpoint`` defaults to the most recent one under ``runs/``.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.data import get_dataloaders
from src.mask_config_utils import build_mask_config
from src.model import ResNet18CAM
from src.utils import latest_checkpoint_path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# ==========================================================
# Defaults. main() overwrites these from the config + CLI.
# ==========================================================

DATA_DIR = ""
CHECKPOINT_PATH = ""
OUTPUT_DIR = PROJECT_ROOT / "debug_outputs" / "random_cams"

IMAGE_SIZE = 224
BATCH_SIZE = 8

VAL_RATIO = 0.0
TEST_RATIO = 0.0

SEED = 42
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_RANDOM_IMAGES = 12

# Tried first; falls through to another split if this one is empty.
PREFERRED_SPLIT = "test"

# Show an OpenCV window at the end (off by default: no display on a server).
SHOW_WINDOW = False

OVERLAY_ALPHA = 0.45

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def parse_args_into_globals() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--checkpoint", default=None, help="defaults to the newest under runs/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-images", type=int, default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--show-window", action="store_true")
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    checkpoint = args.checkpoint or latest_checkpoint_path(args.runs_dir)
    if checkpoint is None:
        raise SystemExit(
            f"No checkpoint given and none found under '{args.runs_dir}'. "
            f"Pass --checkpoint <path>."
        )

    data_dir = args.data_dir or config.get("data_dir")
    if not data_dir:
        raise SystemExit("No data_dir: set it in the config or pass --data-dir.")

    globals().update(
        DATA_DIR=str(data_dir),
        CHECKPOINT_PATH=str(checkpoint),
        OUTPUT_DIR=Path(args.output_dir) if args.output_dir else OUTPUT_DIR,
        IMAGE_SIZE=int(config.get("image_size", IMAGE_SIZE)),
        BATCH_SIZE=int(config.get("batch_size", BATCH_SIZE)),
        VAL_RATIO=float(config.get("val_ratio", VAL_RATIO)),
        TEST_RATIO=float(config.get("test_ratio", TEST_RATIO)),
        SEED=int(config.get("seed", SEED)),
        NUM_WORKERS=int(config.get("num_workers", NUM_WORKERS)),
        NUM_RANDOM_IMAGES=int(args.num_images or NUM_RANDOM_IMAGES),
        PREFERRED_SPLIT=args.split or PREFERRED_SPLIT,
        SHOW_WINDOW=bool(args.show_window),
        CONFIG=config,
    )
    return args


CONFIG: dict = {}


# ==========================================================
# Main entry point
# ==========================================================

def main() -> None:
    parse_args_into_globals()
    set_seed(SEED)

    print("=" * 70)
    print("Random CAM Debugger")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Data: {DATA_DIR}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    recreate_directory(OUTPUT_DIR)

    mask_root = CONFIG.get("mask_root")
    if mask_root:
        mask_dirs, mask_manifests = build_mask_config(mask_root)
    else:
        mask_dirs = CONFIG.get("mask_dirs", {})
        mask_manifests = CONFIG.get("mask_manifests", {})

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir=DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        num_workers=NUM_WORKERS,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        test_ratio=TEST_RATIO,
        augment=False,
        strict_mask_coverage=False,
    )

    loader, split_name = choose_loader(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        preferred=PREFERRED_SPLIT,
    )

    print(f"Classes: {classes}")
    print(f"Selected split: {split_name}")
    print(f"Number of batches: {safe_length(loader)}")

    model = ResNet18CAM(
        num_classes=len(classes),
        pretrained=False,
        dilate_last_block=bool(CONFIG.get("dilate_last_block", False)),
    )

    load_checkpoint(
        model=model,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
    )

    model.to(DEVICE)
    model.eval()

    print("Selecting random images...")

    random_samples = reservoir_sample(
        loader=loader,
        count=NUM_RANDOM_IMAGES,
        seed=SEED,
    )

    if not random_samples:
        raise RuntimeError("No images were found in the selected dataloader.")

    print(f"Selected {len(random_samples)} random images.")
    print("Running forward pass...")

    saved_overlays = []

    with torch.inference_mode():
        for index, sample in enumerate(random_samples, start=1):
            image = sample["image"].unsqueeze(0).to(DEVICE)

            # The model returns per-class CAMs directly, so there is no classifier
            # weight to contract with -- logits are just their spatial mean.
            logits, cams = model(image)

            validate_outputs(logits=logits, cams=cams, num_classes=len(classes))

            probabilities = torch.softmax(logits, dim=1)

            prediction = int(probabilities.argmax(dim=1).item())
            confidence = float(probabilities[0, prediction].item())

            predicted_cam = cams[0, prediction]

            prediction_name = get_class_name(
                classes=classes,
                index=prediction,
            )

            panel, overlay = create_visualization(
                image=sample["image"],
                cam=predicted_cam,
                prediction=prediction,
                prediction_name=prediction_name,
                confidence=confidence,
                image_path=sample["image_path"],
            )

            source_name = safe_filename(
                Path(sample["image_path"]).stem
                if sample["image_path"]
                else f"sample_{index:02d}"
            )

            output_name = (
                f"sample_{index:02d}_"
                f"pred_{prediction_name}_"
                f"conf_{confidence:.4f}_"
                f"{source_name}.png"
            )

            output_path = OUTPUT_DIR / output_name

            if not cv2.imwrite(str(output_path), panel):
                raise RuntimeError(f"Could not save image: {output_path}")

            saved_overlays.append(
                add_grid_caption(
                    overlay=overlay,
                    index=index,
                    prediction_name=prediction_name,
                    confidence=confidence,
                )
            )

            print(
                f"[{index:02d}/{len(random_samples):02d}] "
                f"pred={prediction_name} | "
                f"confidence={confidence:.4f} | "
                f"file={Path(sample['image_path']).name}"
            )

    grid = make_grid(
        images=saved_overlays,
        columns=4,
    )

    grid_path = OUTPUT_DIR / "random_cams_grid.png"

    if not cv2.imwrite(str(grid_path), grid):
        raise RuntimeError(f"Could not save grid: {grid_path}")

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Saved individual images to: {OUTPUT_DIR}")
    print(f"Saved grid to: {grid_path}")

    if SHOW_WINDOW:
        cv2.imshow("Random predicted CAMs", grid)
        print("Press any key on the OpenCV window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ==========================================================
# انتخاب تصادفی یکنواخت از کل دیتالودر
# ==========================================================

def reservoir_sample(
    loader,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    با Reservoir Sampling از کل دیتالودر نمونه می‌گیرد.

    مزیت:
    لازم نیست همه تصاویر هم‌زمان وارد RAM شوند و هر تصویر شانس یکسانی دارد.
    """

    if count < 1:
        raise ValueError("NUM_RANDOM_IMAGES must be at least 1.")

    rng = random.Random(seed)

    selected: list[dict[str, Any]] = []
    seen = 0

    for batch in loader:
        images, image_paths = unpack_images_and_paths(batch)

        batch_size = images.shape[0]

        for index in range(batch_size):
            seen += 1

            sample = {
                "image": images[index].detach().cpu().clone(),
                "image_path": get_path(
                    image_paths=image_paths,
                    index=index,
                    fallback=f"sample_{seen:06d}",
                ),
            }

            if len(selected) < count:
                selected.append(sample)
                continue

            replacement_index = rng.randrange(seen)

            if replacement_index < count:
                selected[replacement_index] = sample

    return selected


# ==========================================================
# CAM
# ==========================================================

# ==========================================================
# ساخت تصویر خروجی
# ==========================================================

def create_visualization(
    image: torch.Tensor,
    cam: torch.Tensor,
    prediction: int,
    prediction_name: str,
    confidence: float,
    image_path: str,
) -> tuple[np.ndarray, np.ndarray]:

    image_rgb = image_to_uint8(image)

    height, width = image_rgb.shape[:2]

    cam_uint8 = cam_to_uint8(
        cam=cam,
        height=height,
        width=width,
    )

    image_bgr = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR,
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET,
    )

    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - OVERLAY_ALPHA,
        heatmap,
        OVERLAY_ALPHA,
        0.0,
    )

    panels = [
        add_panel_title(
            image=image_bgr,
            title="Input image",
        ),
        add_panel_title(
            image=heatmap,
            title="Predicted CAM",
        ),
        add_panel_title(
            image=overlay,
            title="CAM overlay",
        ),
    ]

    canvas = np.concatenate(
        panels,
        axis=1,
    )

    footer = np.zeros(
        (74, canvas.shape[1], 3),
        dtype=np.uint8,
    )

    line_1 = (
        f"prediction={prediction_name} ({prediction}) | "
        f"confidence={confidence:.4f}"
    )

    line_2 = (
        f"file={Path(image_path).name}"
        if image_path
        else "file=unknown"
    )

    cv2.putText(
        footer,
        line_1,
        (10, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        footer,
        line_2,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    canvas = np.concatenate(
        [canvas, footer],
        axis=0,
    )

    return canvas, overlay


def add_grid_caption(
    overlay: np.ndarray,
    index: int,
    prediction_name: str,
    confidence: float,
) -> np.ndarray:

    tile = overlay.copy()

    caption_height = 54

    caption = np.zeros(
        (caption_height, tile.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        caption,
        f"#{index}  {prediction_name}",
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        caption,
        f"confidence={confidence:.3f}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    return np.concatenate(
        [tile, caption],
        axis=0,
    )


def make_grid(
    images: list[np.ndarray],
    columns: int,
) -> np.ndarray:

    if not images:
        raise ValueError("No images were provided for the grid.")

    columns = max(1, columns)

    tile_height = max(image.shape[0] for image in images)
    tile_width = max(image.shape[1] for image in images)

    rows = (len(images) + columns - 1) // columns

    grid = np.zeros(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns

        y = row * tile_height
        x = column * tile_width

        grid[
            y:y + image.shape[0],
            x:x + image.shape[1],
        ] = image

    return grid


# ==========================================================
# تبدیل Tensor به تصویر
# ==========================================================

def image_to_uint8(
    image: torch.Tensor,
) -> np.ndarray:

    image = image.detach().float().cpu()

    if image.ndim == 2:
        image = image.unsqueeze(0)

    if image.ndim != 3:
        raise ValueError(
            f"Image must have shape [C,H,W], got {tuple(image.shape)}."
        )

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    elif image.shape[0] > 3:
        image = image[:3]

    mean = torch.tensor(
        IMAGE_MEAN,
        dtype=image.dtype,
    ).view(3, 1, 1)

    std = torch.tensor(
        IMAGE_STD,
        dtype=image.dtype,
    ).view(3, 1, 1)

    restored = image * std + mean

    # اگر دیتالودر ImageNet normalization نداشته باشد،
    # از min-max صرفاً برای نمایش استفاده می‌شود.
    if (
        not torch.isfinite(restored).all()
        or float(restored.min()) < -0.25
        or float(restored.max()) > 1.25
    ):
        image_min = image.min()
        image_max = image.max()

        if float(image_max - image_min) > 1e-8:
            restored = (
                image - image_min
            ) / (
                image_max - image_min
            )
        else:
            restored = torch.zeros_like(image)

    restored = restored.clamp(0.0, 1.0)

    return (
        restored.permute(1, 2, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def cam_to_uint8(
    cam: torch.Tensor,
    height: int,
    width: int,
) -> np.ndarray:

    cam = cam.detach().float().cpu()

    if cam.ndim != 2:
        raise ValueError(
            f"CAM must have shape [H,W], got {tuple(cam.shape)}."
        )

    # ReLU و min-max فقط برای visualization هستند.
    cam = F.relu(cam)

    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    cam_min = cam.min()
    cam_max = cam.max()

    if float(cam_max - cam_min) > 1e-8:
        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )
    else:
        cam = torch.zeros_like(cam)

    return (
        cam.numpy() * 255.0
    ).round().astype(np.uint8)


def add_panel_title(
    image: np.ndarray,
    title: str,
) -> np.ndarray:

    header = np.zeros(
        (38, image.shape[1], 3),
        dtype=np.uint8,
    )

    cv2.putText(
        header,
        title,
        (9, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.concatenate(
        [header, image],
        axis=0,
    )


# ==========================================================
# مدل و checkpoint
# ==========================================================

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: str,
) -> None:

    checkpoint_file = Path(checkpoint_path)

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_file}"
        )

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get(
                "state_dict",
                checkpoint,
            ),
        )
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict)

    except RuntimeError:
        # checkpoint saved through DataParallel
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

        model.load_state_dict(state_dict)

    print(f"Loaded checkpoint: {checkpoint_file}")


def validate_outputs(
    logits: torch.Tensor,
    cams: torch.Tensor,
    num_classes: int,
) -> None:

    if logits.ndim != 2:
        raise ValueError(
            f"logits must be [B,K], got {tuple(logits.shape)}."
        )

    if cams.ndim != 4:
        raise ValueError(
            f"cams must be [B,K,H,W], got {tuple(cams.shape)}."
        )

    if cams.shape[1] != num_classes:
        raise ValueError(
            "CAM channels do not match the number of classes: "
            f"{cams.shape[1]} != {num_classes}"
        )


# ==========================================================
# دیتالودر
# ==========================================================

def unpack_images_and_paths(
    batch: Any,
) -> tuple[torch.Tensor, Any]:

    if isinstance(batch, dict):
        images = first_existing_value(
            mapping=batch,
            keys=("image", "images", "x"),
        )

        paths = first_existing_value(
            mapping=batch,
            keys=("image_path", "image_paths", "path", "paths"),
            required=False,
        )

        return images, paths

    if isinstance(batch, (tuple, list)):
        if len(batch) < 1:
            raise ValueError("The batch is empty.")

        images = batch[0]

        # معمولاً tuple به شکل image, label, path است.
        paths = batch[2] if len(batch) > 2 else None

        return images, paths

    raise TypeError(
        f"Unsupported batch type: {type(batch).__name__}"
    )


def first_existing_value(
    mapping: dict,
    keys: tuple[str, ...],
    required: bool = True,
):

    for key in keys:
        if key in mapping:
            return mapping[key]

    if required:
        raise KeyError(
            f"None of these keys were found: {keys}. "
            f"Available keys: {tuple(mapping.keys())}"
        )

    return None


def choose_loader(
    train_loader,
    val_loader,
    test_loader,
    preferred: str,
):

    loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }

    preferred = preferred.lower()

    if preferred not in loaders:
        raise ValueError(
            "PREFERRED_SPLIT must be train, val, or test."
        )

    order = [preferred] + [
        name
        for name in ("test", "val", "train")
        if name != preferred
    ]

    for name in order:
        loader = loaders[name]

        if loader is not None and safe_length(loader) > 0:
            if name != preferred:
                print(
                    f"Warning: '{preferred}' is empty; "
                    f"using '{name}' instead."
                )

            return loader, name

    raise RuntimeError("All dataloaders are empty.")


def safe_length(loader) -> int:
    try:
        return len(loader)
    except TypeError:
        return 0


def get_path(
    image_paths,
    index: int,
    fallback: str,
) -> str:

    if image_paths is None:
        return fallback

    try:
        return str(image_paths[index])
    except (IndexError, TypeError):
        return fallback


# ==========================================================
# ابزارهای عمومی
# ==========================================================

def get_class_name(
    classes,
    index: int,
) -> str:

    try:
        return str(classes[index])
    except (IndexError, TypeError):
        return str(index)


def safe_filename(
    value: str,
) -> str:

    result = []

    for character in value:
        if character.isalnum() or character in "-_.":
            result.append(character)
        else:
            result.append("_")

    cleaned = "".join(result).strip("._")

    return cleaned[:100] or "sample"


def recreate_directory(
    directory: Path,
) -> None:

    if directory.exists():
        shutil.rmtree(directory)

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()