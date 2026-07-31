from pathlib import Path


def build_mask_config(mask_root: str):
    mask_root = Path(mask_root)
    mask_dirs = {}
    mask_manifests = {}

    for class_dir in sorted(mask_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        mask_dirs[class_name] = str(class_dir / "masks")
        mask_manifests[class_name] = str(class_dir / f"manifest_{class_name}_masks.csv")

    return mask_dirs, mask_manifests