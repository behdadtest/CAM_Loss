from pathlib import Path
import csv
from typing import Dict, Optional, Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets


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
    """
    Returns a set of valid mask stems if a manifest exists.
    If the manifest does not exist, returns None, meaning: use mask file
    existence as validity.
    """
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


class ImageFolderWithClassMasks(Dataset):
    """
    Generic multi-class ImageFolder dataset with optional pseudo-mask per class.

    Expected image layout:
        data_dir/
          class_a/*.jpg
          class_b/*.jpg
          ...

    Mask paths are given through config:
        mask_dirs:
          class_a: path/to/class_a/masks
          class_b: path/to/class_b/masks

    Optional manifests:
        mask_manifests:
          class_a: path/to/class_a/manifest.csv
          class_b: path/to/class_b/manifest.csv

    Mask filename must match the image stem:
        image: dog_000001.jpg
        mask:  dog_000001.png   
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int,
        mask_dirs: Optional[dict] = None,
        mask_manifests: Optional[dict] = None,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.data_dir}")

        self.image_size = image_size
        self.mask_dirs = _normalize_mapping(mask_dirs)
        self.mask_manifests = _normalize_mapping(mask_manifests)

        # Use ImageFolder only to build samples/classes/class_to_idx reliably.
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
            if manifest_path is not None:
                self.valid_mask_stems_by_class[class_name] = load_valid_mask_stems(manifest_path)
            else:
                self.valid_mask_stems_by_class[class_name] = None

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

        print("Classes:", self.classes)
        print("Total samples:", len(self.samples))
        for class_name in self.classes:
            mask_dir = self.mask_dirs.get(class_name)
            manifest = self.mask_manifests.get(class_name)
            print(f"Class '{class_name}': mask_dir={mask_dir}, manifest={manifest}")

    def __len__(self):
        return len(self.samples)

    def _load_mask(self, image_path: Path, label: int):
        class_name = self.classes[label]
        mask_dir = self.mask_dirs.get(class_name)

        mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        has_mask = torch.tensor(0.0, dtype=torch.float32)

        if mask_dir is None:
            return mask_tensor, has_mask

        mask_path = mask_dir / f"{image_path.stem}.png"
        valid_stems = self.valid_mask_stems_by_class.get(class_name)

        # If manifest exists, require the stem to be valid in manifest.
        if valid_stems is not None and image_path.stem not in valid_stems:
            return mask_tensor, has_mask

        if not mask_path.exists():
            return mask_tensor, has_mask

        mask = Image.open(mask_path).convert("L")
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        if mask_tensor.sum() > 0:
            has_mask = torch.tensor(1.0, dtype=torch.float32)

        return mask_tensor, has_mask

    def __getitem__(self, idx):
        image_path_str, label = self.samples[idx]
        image_path = Path(image_path_str)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        mask_tensor, has_mask = self._load_mask(image_path, label)

        return {
            "image": image_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "mask": mask_tensor,
            "has_mask": has_mask,
            "image_path": str(image_path),
        }


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
):
    dataset = ImageFolderWithClassMasks(
        data_dir=data_dir,
        image_size=image_size,
        mask_dirs=mask_dirs,
        mask_manifests=mask_manifests,
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

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )

    print(
        f"Split sizes | train={train_size} val={val_size} test={test_size} "
        f"(total={total_size})"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, dataset.classes