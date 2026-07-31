import copy
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _normalize_mapping(mapping: Optional[dict]) -> Dict[str, Path]:
    if mapping is None:
        return {}
    return {str(k): Path(v) for k, v in mapping.items()}


def load_valid_mask_stems(manifest_path: Path) -> Optional[set]:

    if not manifest_path or not manifest_path.exists():
        return None

    valid = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            has_mask = str(row.get("has_mask", "0")).strip()
            status = str(row.get("status", "")).strip().lower()
            mask_path = row.get("mask_path", "")

            if has_mask == "1" and status == "ok":
                if mask_path:
                    valid.add(Path(mask_path).stem)
                else:
                    image_path = row.get("image_path", "")
                    if image_path:
                        valid.add(Path(image_path).stem)

    return valid


class JointTransform:
    """Applies the *same* geometric transform to an image and its mask.

    This has to be one object rather than two ``Compose`` pipelines: if the image
    is flipped and the mask is not, the supervision now says the object is on the
    side it isn't, and training actively teaches the model to be wrong.

    Masks are resized with a filter and re-thresholded rather than sampled with
    nearest-neighbour, so thin structures (legs, tails) survive downscaling.
    """

    def __init__(
        self,
        image_size: int,
        augment: bool = False,
        hflip_prob: float = 0.5,
        scale: Tuple[float, float] = (0.6, 1.0),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        color_jitter: float = 0.2,
    ):
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.hflip_prob = float(hflip_prob)
        self.scale = tuple(scale)
        self.ratio = tuple(ratio)

        self.jitter = (
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
            if (augment and color_jitter > 0)
            else None
        )

    def __call__(self, image: Image.Image, mask: Optional[Image.Image]):
        size = [self.image_size, self.image_size]

        if self.augment:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, list(self.scale), list(self.ratio)
            )
            image = TF.resized_crop(image, i, j, h, w, size, TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resized_crop(mask, i, j, h, w, size, TF.InterpolationMode.BILINEAR)

            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                if mask is not None:
                    mask = TF.hflip(mask)

            if self.jitter is not None:
                image = self.jitter(image)
        else:
            image = TF.resize(image, size, TF.InterpolationMode.BILINEAR)
            if mask is not None:
                mask = TF.resize(mask, size, TF.InterpolationMode.BILINEAR)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = (TF.to_tensor(mask) > 0.5).float() if mask is not None else None

        return image_tensor, mask_tensor


class ImageFolderWithClassMasks(Dataset):
    """Multi-class ImageFolder with an optional SAM pseudo-mask per image.

    Expected image layout::

        data_dir/
          class_a/*.jpg
          class_b/*.jpg

    Mask directories and manifests come from config, keyed by class name. The
    mask filename is ``{mask_prefix}{image stem}{mask_suffix}{mask_extension}``,
    all configurable -- a silent naming mismatch here used to give every sample
    ``has_mask=0``, leaving the CAM loss at exactly 0.0 for the whole run with
    nothing to indicate anything was wrong.
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        mask_dirs: Optional[dict] = None,
        mask_manifests: Optional[dict] = None,
        augment: bool = False,
        mask_extension: str = ".png",
        mask_prefix: str = "",
        mask_suffix: str = "",
        min_fg_fraction: float = 0.01,
        max_fg_fraction: float = 0.95,
        bg_randomize_prob: float = 0.0,
        augment_kwargs: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_dir}")

        self.image_size = image_size
        self.mask_dirs = _normalize_mapping(mask_dirs)
        self.mask_manifests = _normalize_mapping(mask_manifests)

        self.mask_extension = mask_extension
        self.mask_prefix = mask_prefix
        self.mask_suffix = mask_suffix
        self.min_fg_fraction = float(min_fg_fraction)
        self.max_fg_fraction = float(max_fg_fraction)
        self.bg_randomize_prob = float(bg_randomize_prob)
        self.augment = bool(augment)
        self.augment_kwargs = dict(augment_kwargs or {})

        base_dataset = datasets.ImageFolder(
            root=str(self.data_dir),
            transform=None,
            is_valid_file=is_valid_image,
        )

        self.samples: List[Tuple[str, int]] = base_dataset.samples
        self.classes = base_dataset.classes
        self.class_to_idx = base_dataset.class_to_idx

        self.valid_mask_stems_by_class = {}
        for class_name in self.classes:
            manifest_path = self.mask_manifests.get(class_name)
            self.valid_mask_stems_by_class[class_name] = (
                load_valid_mask_stems(manifest_path) if manifest_path is not None else None
            )

        self._build_transform()

        print("Classes:", self.classes)
        print("Total samples:", len(self.samples))
        for class_name in self.classes:
            print(
                f"Class '{class_name}': mask_dir={self.mask_dirs.get(class_name)}, "
                f"manifest={self.mask_manifests.get(class_name)}"
            )

    # ------------------------------------------------------------------ setup

    def _build_transform(self):
        self.transform = JointTransform(
            image_size=self.image_size,
            augment=self.augment,
            **self.augment_kwargs,
        )

    def clone(self, **overrides) -> "ImageFolderWithClassMasks":
        """Shallow copy with different settings, sharing the scanned file list.

        Used to give the training split augmentation while validation and test
        keep a deterministic resize, without scanning the directory twice.
        """
        other = copy.copy(self)
        for key, value in overrides.items():
            setattr(other, key, value)
        other._build_transform()
        return other

    # ------------------------------------------------------------------ masks

    def mask_path_for(self, image_path: Path, label: int) -> Optional[Path]:
        mask_dir = self.mask_dirs.get(self.classes[label])
        if mask_dir is None:
            return None
        name = f"{self.mask_prefix}{image_path.stem}{self.mask_suffix}{self.mask_extension}"
        return mask_dir / name

    def _mask_is_listed(self, image_path: Path, label: int) -> bool:
        valid_stems = self.valid_mask_stems_by_class.get(self.classes[label])
        return valid_stems is None or image_path.stem in valid_stems

    def has_resolvable_mask(self, index: int) -> bool:
        image_path_str, label = self.samples[index]
        image_path = Path(image_path_str)
        if not self._mask_is_listed(image_path, label):
            return False
        mask_path = self.mask_path_for(image_path, label)
        return mask_path is not None and mask_path.exists()

    def report_mask_coverage(self, indices=None, name: str = "dataset", strict: bool = True):
        """Print how many samples actually resolve to a mask file, per class.

        Raises when a split that is supposed to have masks has none: a silent 0%
        coverage means the CAM loss contributes literally nothing while the log
        happily prints ``cam=0.0000``.
        """
        indices = list(range(len(self.samples))) if indices is None else list(indices)

        totals = {c: 0 for c in self.classes}
        found = {c: 0 for c in self.classes}
        for index in indices:
            class_name = self.classes[self.samples[index][1]]
            totals[class_name] += 1
            if self.has_resolvable_mask(index):
                found[class_name] += 1

        total_found = sum(found.values())
        print(f"Mask coverage [{name}]:")
        for class_name in self.classes:
            n_total = totals[class_name]
            pct = 100.0 * found[class_name] / n_total if n_total else 0.0
            print(f"  {class_name:<20} {found[class_name]:>6}/{n_total:<6} ({pct:5.1f}%)")
        print(f"  {'TOTAL':<20} {total_found:>6}/{len(indices):<6}")

        if strict and self.mask_dirs and total_found == 0:
            example = self.mask_path_for(Path(self.samples[indices[0]][0]), self.samples[indices[0]][1])
            raise RuntimeError(
                f"Mask directories are configured but not one mask file was found for "
                f"split '{name}'. The CAM loss would be exactly 0.0 for the entire run.\n"
                f"  expected a file like: {example}\n"
                f"  check mask_dirs / mask_root, and mask_prefix / mask_suffix / "
                f"mask_extension if your masks are named differently."
            )

        return {"total": len(indices), "found": total_found, "per_class": found}

    def _load_mask_image(self, image_path: Path, label: int) -> Optional[Image.Image]:
        if not self._mask_is_listed(image_path, label):
            return None
        mask_path = self.mask_path_for(image_path, label)
        if mask_path is None or not mask_path.exists():
            return None
        return Image.open(mask_path).convert("L")

    # ------------------------------------------------------------------ items

    def __len__(self):
        return len(self.samples)

    def _random_background(self, exclude_index: int) -> torch.Tensor:
        """An unrelated image, resized and normalised, to paste behind an object."""
        for _ in range(4):
            index = random.randrange(len(self.samples))
            if index == exclude_index:
                continue
            try:
                with Image.open(self.samples[index][0]) as other:
                    other = other.convert("RGB")
                    other = TF.resize(
                        other,
                        [self.image_size, self.image_size],
                        TF.InterpolationMode.BILINEAR,
                    )
                    return TF.normalize(TF.to_tensor(other), IMAGENET_MEAN, IMAGENET_STD)
            except Exception:
                continue
        return None

    def __getitem__(self, idx):
        image_path_str, label = self.samples[idx]
        image_path = Path(image_path_str)

        image = Image.open(image_path).convert("RGB")
        mask = self._load_mask_image(image_path, label)

        image_tensor, mask_tensor = self.transform(image, mask)

        has_mask = torch.tensor(0.0, dtype=torch.float32)
        if mask_tensor is None:
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        else:
            # Reject degenerate masks. An all-foreground mask has no background at
            # all, so it carries no suppression signal -- only "light up
            # everywhere", which is the opposite of the point.
            fg_fraction = float(mask_tensor.mean().item())
            if self.min_fg_fraction <= fg_fraction <= self.max_fg_fraction:
                has_mask = torch.tensor(1.0, dtype=torch.float32)

        if (
            self.bg_randomize_prob > 0
            and float(has_mask) > 0
            and random.random() < self.bg_randomize_prob
        ):
            background = self._random_background(idx)
            if background is not None:
                # Normalisation is affine and per-pixel, so compositing after it is
                # identical to compositing before it.
                image_tensor = image_tensor * mask_tensor + background * (1.0 - mask_tensor)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask_tensor,
            "has_mask": has_mask,
            "image_path": str(image_path),
        }


def _subsample_train_indices(
    base_dataset: ImageFolderWithClassMasks,
    indices: List[int],
    shots_per_class: Optional[int] = None,
    fraction: Optional[float] = None,
    seed: int = 42,
) -> List[int]:

    if shots_per_class is None and fraction is None:
        return indices

    if shots_per_class is not None and fraction is not None:
        raise ValueError("Specify either shots_per_class or fraction, not both.")

    rng = random.Random(seed)

    by_class: Dict[int, List[int]] = {}
    for idx in indices:
        _, label = base_dataset.samples[idx]
        by_class.setdefault(label, []).append(idx)

    selected: List[int] = []
    for label, idxs in sorted(by_class.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)

        if shots_per_class is not None:
            k = min(shots_per_class, len(idxs))
        else:
            k = min(max(1, round(len(idxs) * fraction)), len(idxs))

        chosen = idxs[:k]
        selected.extend(chosen)

        class_name = base_dataset.classes[label]
        print(f"  [subsample] class='{class_name}': {len(chosen)}/{len(idxs)} samples kept for training")

    rng.shuffle(selected)
    return selected


def seed_worker(worker_id: int):
    """Make DataLoader workers deterministic given the base seed."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed % (2 ** 32))
    except ImportError:
        pass


def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    val_ratio: float,
    seed: int,
    num_workers: int,
    mask_dirs: Optional[dict] = None,
    mask_manifests: Optional[dict] = None,
    test_ratio: float = 0.0,
    train_shots_per_class: Optional[int] = None,
    train_fraction: Optional[float] = None,
    augment: bool = True,
    augment_kwargs: Optional[dict] = None,
    bg_randomize_prob: float = 0.0,
    mask_extension: str = ".png",
    mask_prefix: str = "",
    mask_suffix: str = "",
    min_fg_fraction: float = 0.01,
    max_fg_fraction: float = 0.95,
    strict_mask_coverage: bool = True,
):

    # Built once without augmentation; this instance serves val and test.
    dataset = ImageFolderWithClassMasks(
        data_dir=data_dir,
        image_size=image_size,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
        augment=False,
        mask_extension=mask_extension,
        mask_prefix=mask_prefix,
        mask_suffix=mask_suffix,
        min_fg_fraction=min_fg_fraction,
        max_fg_fraction=max_fg_fraction,
    )

    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size - test_size

    if train_size <= 0:
        raise ValueError(
            f"val_ratio + test_ratio too large for dataset size {total_size} "
            f"(train_size would be {train_size})."
        )

    # Same permutation semantics as the previous random_split call, so splits stay
    # comparable with earlier runs at the same seed.
    permutation = torch.randperm(
        total_size, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    train_indices = permutation[:train_size]
    val_indices = permutation[train_size:train_size + val_size]
    test_indices = permutation[train_size + val_size:]

    print(
        f"Split sizes | train={train_size} val={val_size} test={test_size} "
        f"(total={total_size})"
    )

    if train_shots_per_class is not None or train_fraction is not None:
        print("Subsampling training set:")
        train_indices = _subsample_train_indices(
            base_dataset=dataset,
            indices=train_indices,
            shots_per_class=train_shots_per_class,
            fraction=train_fraction,
            seed=seed,
        )
        print(f"Training set size after subsampling: {len(train_indices)}")

    # Augmentation and background randomisation apply to training only.
    train_view = dataset.clone(
        augment=bool(augment),
        augment_kwargs=augment_kwargs or {},
        bg_randomize_prob=bg_randomize_prob,
    )

    dataset.report_mask_coverage(train_indices, name="train", strict=strict_mask_coverage)
    dataset.report_mask_coverage(val_indices, name="val", strict=False)
    if test_indices:
        dataset.report_mask_coverage(test_indices, name="test", strict=False)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )

    train_loader = DataLoader(
        Subset(train_view, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
        **common,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices), batch_size=batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices), batch_size=batch_size, shuffle=False, **common
    )

    return train_loader, val_loader, test_loader, dataset.classes
