# ResNet18 + Multi-Class CAM Loss

This project trains a ResNet18 classifier with:

```text
total_loss = classification_loss + lambda_cam * cam_mask_loss
```

The model produces CAMs for **all classes**:

```text
cams: [B, num_classes, H, W]
```

For each sample, the CAM of the ground-truth class is compared with the pseudo-mask of that same class.

## Expected data layout

```text
project/
├── cats_vs_dogs_folder/
│   ├── cat/
│   └── dog/
├── sam_pseudo_masks_cat_yolo_sam/
│   ├── masks/
│   └── manifest_cat_masks.csv
├── sam_pseudo_masks_dog_yolo_sam/
│   ├── masks/
│   └── manifest_dog_masks.csv
├── configs/
│   └── config.yaml
├── src/
└── main.py
```

Mask file stems must match image file stems:

```text
cats_vs_dogs_folder/dog/dog_000123.jpg
sam_pseudo_masks_dog_yolo_sam/masks/dog_000123.png
```

For more classes, add their image folders under `data_dir` and add corresponding entries under `mask_dirs` and optionally `mask_manifests` in `configs/config.yaml`.

## Run

```powershell
cd D:\CVLab\SAM
.\sam_env\Scripts\activate
python -m pip install torch torchvision pillow tqdm pyyaml
python main.py
```
