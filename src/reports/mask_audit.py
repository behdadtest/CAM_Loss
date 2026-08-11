"""
Dataset-level audit of the SAM pseudo-masks.

`CAMMaskLoss` is silent when it has nothing to work with: a sample whose mask
is missing, excluded by the manifest, or thresholded away simply gets
`has_mask = 0` and drops out of the loss. A whole run can therefore report a
falling `cam_loss` while the mask term never touched a single image.

This module reconstructs, without touching `src/data.py`, exactly what the
dataset does per sample and answers:

  * do the class folder names under `mask_root` match the dataset classes?
  * how many samples end up with `has_mask = 1`, and why do the rest not?
  * are the mask PNGs stored 0/255 (survives `ToTensor()` + `> 0.5`) or 0/1
    (silently thresholded to an all-zero mask)?
  * is the mask the object or the background? (border-vs-center heuristic)
  * once the mask is area-downsampled to the CAM grid, how low can the
    containment loss physically go?
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.reports.common import (
    CRITICAL,
    INFO,
    OK,
    WARNING,
    downsample_mask,
    finding,
    fmt,
    loader_base_and_indices,
    print_findings,
    render_findings,
    summarize,
    write_json,
    write_lines,
)

# Mirrors `ImageFolderWithClassMasks._load_mask`: ToTensor() divides by 255 and
# the dataset keeps pixels with value > 0.5, i.e. raw uint8 >= 128.
_BINARY_THRESHOLD = 128

STATUS_OK = "ok"
STATUS_NO_MASK_DIR = "no_mask_dir_configured"
STATUS_MASK_DIR_MISSING = "mask_dir_missing_on_disk"
STATUS_EXCLUDED = "excluded_by_manifest"
STATUS_FILE_MISSING = "mask_file_missing"
STATUS_EMPTY = "empty_after_threshold"
STATUS_UNREADABLE = "mask_unreadable"


def _binary_mask_from_path(path: Path, image_size: int):
    """Reproduce the dataset's mask pipeline and also keep the raw pixels."""
    with Image.open(path) as handle:
        gray = handle.convert("L")
        raw = np.asarray(gray)
        resized = gray.resize((image_size, image_size), resample=Image.NEAREST)
        resized_arr = np.asarray(resized)

    binary = (resized_arr >= _BINARY_THRESHOLD).astype(np.float32)
    return raw, binary


def _region_coverage(binary: np.ndarray) -> Dict[str, float]:
    """
    Coverage of a border frame vs. the central box.

    A foreground mask concentrates in the middle; a background mask hugs the
    frame. When border coverage beats center coverage the mask polarity is
    almost certainly inverted, and the loss is then pushing the CAM *into*
    the background.
    """
    h, w = binary.shape
    band_h = max(1, h // 8)
    band_w = max(1, w // 8)

    border = np.ones((h, w), dtype=bool)
    border[band_h:h - band_h, band_w:w - band_w] = False

    center = np.zeros((h, w), dtype=bool)
    center[h // 4:3 * h // 4, w // 4:3 * w // 4] = True

    return {
        "border": float(binary[border].mean()) if border.any() else 0.0,
        "center": float(binary[center].mean()) if center.any() else 0.0,
    }


def _cam_grid_stats(binary: np.ndarray, cam_size) -> Dict[str, float]:
    """
    What the loss can and cannot achieve once the mask hits the CAM grid.

    `uniform_containment` is the loss value of a flat, uninformative CAM, and
    `floor_containment` is the value of a perfect CAM that dumps all of its
    energy into the single most-inside cell. Every measured `cam_loss` should
    be read between these two numbers.
    """
    tensor = torch.from_numpy(binary).unsqueeze(0).unsqueeze(0)
    down = downsample_mask(tensor, cam_size)[0, 0]

    outside = 1.0 - down
    n_cells = down.numel()

    return {
        "coverage_cam_grid": float(down.mean()),
        "uniform_containment": float(outside.mean()),
        "floor_containment": float(outside.min()),
        "pure_background_cell_frac": float((down <= 1e-6).float().mean()),
        "pure_object_cell_frac": float((down >= 1.0 - 1e-6).float().mean()),
        "n_cam_cells": int(n_cells),
    }


def _classify_sample(base_dataset, index: int) -> Dict[str, object]:
    """Cheap, decode-free replay of the dataset's mask lookup."""
    image_path_str, label = base_dataset.samples[index]
    image_path = Path(image_path_str)
    class_name = base_dataset.classes[label]

    mask_dir = base_dataset.mask_dirs.get(class_name)
    record = {
        "index": index,
        "class_name": class_name,
        "image_path": str(image_path),
        "mask_path": None,
        "status": STATUS_OK,
    }

    if mask_dir is None:
        record["status"] = STATUS_NO_MASK_DIR
        return record

    if not Path(mask_dir).exists():
        record["status"] = STATUS_MASK_DIR_MISSING
        return record

    valid_stems = base_dataset.valid_mask_stems_by_class.get(class_name)
    # The dataset checks the manifest before it checks the file, so do the same.
    if valid_stems is not None and image_path.stem not in valid_stems:
        record["status"] = STATUS_EXCLUDED
        return record

    mask_path = Path(mask_dir) / f"{image_path.stem}.png"
    record["mask_path"] = str(mask_path)

    if not mask_path.exists():
        record["status"] = STATUS_FILE_MISSING

    return record


def _audit_split(
    base_dataset,
    indices: List[int],
    split_name: str,
    image_size: int,
    cam_size,
    max_decoded: Optional[int],
    seed: int,
) -> Dict[str, object]:
    records = [_classify_sample(base_dataset, i) for i in indices]

    status_counts: Dict[str, int] = {}
    for record in records:
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    decodable = [r for r in records if r["status"] == STATUS_OK]

    if max_decoded is not None and len(decodable) > max_decoded:
        rng = random.Random(seed)
        to_decode = rng.sample(decodable, max_decoded)
        capped = True
    else:
        to_decode = decodable
        capped = False

    coverage_full: List[float] = []
    coverage_cam: List[float] = []
    uniform_containment: List[float] = []
    floor_containment: List[float] = []
    pure_bg_cells: List[float] = []
    pure_fg_cells: List[float] = []
    border_cov: List[float] = []
    center_cov: List[float] = []
    raw_max_values: List[int] = []

    n_empty = 0
    n_unreadable = 0
    n_full_frame = 0
    decoded_examples: List[Dict[str, object]] = []

    for record in to_decode:
        try:
            raw, binary = _binary_mask_from_path(Path(record["mask_path"]), image_size)
        except Exception:  # noqa: BLE001 - a corrupt PNG is a finding, not a crash
            record["status"] = STATUS_UNREADABLE
            n_unreadable += 1
            continue

        raw_max_values.append(int(raw.max()) if raw.size else 0)

        cov = float(binary.mean())
        if cov <= 0.0:
            record["status"] = STATUS_EMPTY
            n_empty += 1
            continue
        if cov >= 0.999:
            n_full_frame += 1

        grid = _cam_grid_stats(binary, cam_size)
        regions = _region_coverage(binary)

        coverage_full.append(cov)
        coverage_cam.append(grid["coverage_cam_grid"])
        uniform_containment.append(grid["uniform_containment"])
        floor_containment.append(grid["floor_containment"])
        pure_bg_cells.append(grid["pure_background_cell_frac"])
        pure_fg_cells.append(grid["pure_object_cell_frac"])
        border_cov.append(regions["border"])
        center_cov.append(regions["center"])

        decoded_examples.append({
            "class_name": record["class_name"],
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
            "coverage_full": cov,
            "coverage_cam_grid": grid["coverage_cam_grid"],
        })

    n_decoded = len(to_decode)
    n_usable = len(coverage_full)

    stats: Dict[str, object] = {
        "split": split_name,
        "n_samples": len(indices),
        "status_counts": status_counts,
        "n_decoded": n_decoded,
        "n_decode_capped": capped,
        "n_mask_empty_after_threshold": n_empty,
        "n_mask_unreadable": n_unreadable,
        "n_mask_covers_whole_image": n_full_frame,
        "n_usable_masks_decoded": n_usable,
        "usable_rate_among_decoded": (n_usable / n_decoded) if n_decoded else 0.0,
        "raw_pixel_max_over_decoded": max(raw_max_values) if raw_max_values else None,
        "raw_pixel_max_min_over_decoded": min(raw_max_values) if raw_max_values else None,
    }

    # `has_mask = 1` requires status ok AND a non-empty thresholded mask, so
    # extrapolate the decoded empty-rate back over every ok-status sample.
    # `status_counts` is the pre-decode tally, so STATUS_OK is exactly the pool
    # the decode pass sampled from.
    n_status_ok = status_counts.get(STATUS_OK, 0)
    usable_rate = stats["usable_rate_among_decoded"]
    stats["estimated_has_mask_count"] = int(round(n_status_ok * usable_rate))
    stats["estimated_has_mask_rate"] = (
        stats["estimated_has_mask_count"] / len(indices) if indices else 0.0
    )

    stats.update(summarize("coverage_full", coverage_full, full=True))
    stats.update(summarize("coverage_cam_grid", coverage_cam, full=True))
    stats.update(summarize("uniform_containment", uniform_containment, full=True))
    stats.update(summarize("floor_containment", floor_containment, full=True))
    stats.update(summarize("pure_background_cell_frac", pure_bg_cells))
    stats.update(summarize("pure_object_cell_frac", pure_fg_cells))
    stats.update(summarize("border_coverage", border_cov))
    stats.update(summarize("center_coverage", center_cov))

    stats["_examples"] = decoded_examples
    return stats


def _save_examples(
    examples: List[Dict[str, object]],
    out_dir: Path,
    image_size: int,
    cam_size,
    num_examples: int,
    seed: int,
) -> Optional[str]:
    """
    image | mask | image with the mask boundary drawn on it.

    Nothing beats looking at three of these to settle a polarity argument.
    """
    if not examples or num_examples <= 0:
        return None

    rng = random.Random(seed)
    by_class: Dict[str, List[dict]] = {}
    for example in examples:
        by_class.setdefault(str(example["class_name"]), []).append(example)

    chosen: List[dict] = []
    per_class = max(1, num_examples // max(1, len(by_class)))
    for class_examples in by_class.values():
        rng.shuffle(class_examples)
        chosen.extend(class_examples[:per_class])
    chosen = chosen[:num_examples]

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for example in chosen:
        try:
            with Image.open(example["image_path"]) as handle:
                image = handle.convert("RGB").resize(
                    (image_size, image_size), resample=Image.BILINEAR
                )
            image_np = np.asarray(image).astype(np.float32)

            _, binary = _binary_mask_from_path(Path(example["mask_path"]), image_size)
        except Exception:  # noqa: BLE001
            continue

        mask_np = np.repeat((binary * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        boundary = _mask_boundary(binary)
        outlined = image_np.copy()
        outlined[boundary] = np.array([0.0, 255.0, 0.0])

        # Dim whatever the loss treats as background, so an inverted mask is
        # immediately obvious.
        shaded = image_np * (0.25 + 0.75 * binary[:, :, None])

        panel = np.concatenate(
            [
                image_np.astype(np.uint8),
                mask_np,
                outlined.astype(np.uint8),
                shaded.astype(np.uint8),
            ],
            axis=1,
        )

        stem = Path(str(example["image_path"])).stem
        name = (
            f"{example['class_name']}_{stem}"
            f"_cov{example['coverage_full']:.2f}"
            f"_camcov{example['coverage_cam_grid']:.2f}.png"
        )
        Image.fromarray(panel).save(out_dir / name)
        saved += 1

    if saved == 0:
        return None

    write_lines(
        out_dir / "README.txt",
        [
            "Panels, left to right:",
            "  1. input image (resized like training)",
            "  2. binary mask after the dataset's resize + threshold",
            "  3. image with the mask boundary in green",
            "  4. image with everything the loss calls BACKGROUND dimmed",
            "",
            "Panel 4 is the polarity check: the dimmed area must be the",
            "background. If the object is the dimmed part, the masks are",
            "inverted and CAMMaskLoss is pushing the CAM off the object.",
            f"CAM grid used for the reported statistics: {tuple(cam_size)}",
        ],
    )
    return str(out_dir)


def _mask_boundary(binary: np.ndarray) -> np.ndarray:
    """4-neighbourhood erosion boundary, without pulling in scipy/cv2."""
    m = binary > 0.5
    eroded = m.copy()
    eroded[1:, :] &= m[:-1, :]
    eroded[:-1, :] &= m[1:, :]
    eroded[:, 1:] &= m[:, :-1]
    eroded[:, :-1] &= m[:, 1:]
    return m & ~eroded


def _build_findings(
    base_dataset,
    per_split: Dict[str, dict],
    overall: dict,
    cam_size,
) -> List[dict]:
    findings: List[dict] = []

    configured = set(base_dataset.mask_dirs.keys())
    dataset_classes = set(base_dataset.classes)

    missing = sorted(dataset_classes - configured)
    if missing:
        findings.append(finding(
            CRITICAL,
            "mask_dir_not_configured",
            f"No mask directory is configured for class(es) {missing}. "
            f"Configured mask keys are {sorted(configured)}. Every sample of "
            f"those classes gets has_mask=0 and contributes nothing to the CAM loss.",
        ))

    unused = sorted(configured - dataset_classes)
    if unused:
        findings.append(finding(
            WARNING,
            "mask_dir_unused",
            f"Mask directories configured for {unused}, which are not dataset "
            f"classes ({sorted(dataset_classes)}). Likely a folder-naming mismatch "
            f"between data_dir and mask_root.",
        ))

    for class_name, mask_dir in base_dataset.mask_dirs.items():
        if not Path(mask_dir).exists():
            findings.append(finding(
                CRITICAL,
                "mask_dir_missing_on_disk",
                f"Mask directory for class '{class_name}' does not exist: {mask_dir}",
            ))

    for class_name, manifest in base_dataset.mask_manifests.items():
        if not Path(manifest).exists():
            findings.append(finding(
                INFO,
                "manifest_missing",
                f"No manifest for class '{class_name}' at {manifest}; falling back "
                f"to plain file existence (no manifest filtering).",
            ))

    rate = overall.get("estimated_has_mask_rate") or 0.0
    if overall.get("estimated_has_mask_count", 0) == 0:
        findings.append(finding(
            CRITICAL,
            "no_usable_masks",
            "No sample in the dataset ends up with has_mask=1. CAMMaskLoss returns "
            "exactly 0 for every batch, so lambda_cam has no effect whatsoever.",
        ))
    elif rate < 0.5:
        breakdown = overall.get("status_counts", {})
        findings.append(finding(
            WARNING,
            "low_mask_coverage_rate",
            f"Only {fmt(rate * 100, 1)}% of samples reach has_mask=1 "
            f"({overall.get('estimated_has_mask_count')}/{overall.get('n_samples')}). "
            f"Reasons: {breakdown}, empty_after_threshold="
            f"{overall.get('n_mask_empty_after_threshold')}.",
        ))
    else:
        findings.append(finding(
            OK,
            "mask_coverage_rate",
            f"{fmt(rate * 100, 1)}% of samples carry a usable mask.",
        ))

    raw_max = overall.get("raw_pixel_max_over_decoded")
    if raw_max is not None and raw_max <= 1:
        findings.append(finding(
            CRITICAL,
            "mask_pixel_scale",
            f"The brightest pixel found in any decoded mask PNG is {raw_max}. The "
            f"dataset does ToTensor() (divide by 255) then keeps pixels > 0.5, so "
            f"0/1-valued masks threshold to all-zero and are dropped. Save the masks "
            f"as 0/255, or change the threshold in ImageFolderWithClassMasks._load_mask.",
        ))
    elif raw_max is not None and raw_max < _BINARY_THRESHOLD:
        findings.append(finding(
            CRITICAL,
            "mask_pixel_scale",
            f"The brightest pixel in any decoded mask is {raw_max}, below the "
            f"effective threshold of {_BINARY_THRESHOLD}. Every mask thresholds to "
            f"empty and is dropped.",
        ))

    n_empty = overall.get("n_mask_empty_after_threshold", 0)
    n_decoded = overall.get("n_decoded", 0)
    if n_decoded and n_empty / n_decoded > 0.1:
        findings.append(finding(
            WARNING,
            "masks_empty_after_threshold",
            f"{n_empty}/{n_decoded} decoded masks are empty after thresholding and "
            f"are silently skipped by the loss.",
        ))

    border = overall.get("border_coverage_mean")
    center = overall.get("center_coverage_mean")
    if border is not None and center is not None:
        if border > center + 0.05:
            findings.append(finding(
                CRITICAL,
                "mask_polarity_inverted",
                f"Mask coverage is higher on the image border ({fmt(border, 3)}) than "
                f"in the center ({fmt(center, 3)}). The masks most likely mark the "
                f"BACKGROUND, not the object. CAMMaskLoss treats mask=1 as the object, "
                f"so it is currently pushing the CAM into the background. Check the "
                f"example panels under diagnostics/mask_examples/.",
            ))
        else:
            findings.append(finding(
                OK,
                "mask_polarity",
                f"Center coverage {fmt(center, 3)} > border coverage {fmt(border, 3)}, "
                f"consistent with mask=1 meaning object.",
            ))

    coverage = overall.get("coverage_full_mean")
    if coverage is not None:
        if coverage > 0.9:
            findings.append(finding(
                CRITICAL,
                "mask_covers_everything",
                f"Mean mask coverage is {fmt(coverage, 3)}: there is almost no "
                f"background left, so the containment loss is near zero regardless of "
                f"what the CAM does and carries no training signal.",
            ))
        elif coverage < 0.02:
            findings.append(finding(
                WARNING,
                "mask_too_small",
                f"Mean mask coverage is only {fmt(coverage, 3)}; masks may be degenerate.",
            ))

    floor = overall.get("floor_containment_mean")
    uniform = overall.get("uniform_containment_mean")
    if floor is not None:
        if floor > 0.3:
            findings.append(finding(
                WARNING,
                "high_loss_floor",
                f"On the {tuple(cam_size)} CAM grid the best achievable containment "
                f"averages {fmt(floor, 3)} (a flat CAM scores {fmt(uniform, 3)}). "
                f"cam_loss can never go below the floor, so do not read a plateau at "
                f"that value as a failure to learn.",
            ))
        else:
            findings.append(finding(
                INFO,
                "loss_floor",
                f"Reference values on the {tuple(cam_size)} CAM grid: perfect CAM "
                f"scores {fmt(floor, 3)}, flat/uninformative CAM scores "
                f"{fmt(uniform, 3)}. Compare every reported cam_loss against these.",
            ))

    pure_bg = overall.get("pure_background_cell_frac_mean")
    if pure_bg is not None and pure_bg < 0.05:
        findings.append(finding(
            WARNING,
            "no_pure_background_cells",
            f"Only {fmt(pure_bg * 100, 1)}% of CAM cells are pure background after "
            f"area-downsampling the mask to {tuple(cam_size)}. Almost every cell is "
            f"part object, part background, so the mask barely discriminates at CAM "
            f"resolution. Consider a larger CAM grid (dilated layer4 / stride-1 / "
            f"a bigger input) if you need sharper supervision.",
        ))

    for split_name, stats in per_split.items():
        split_rate = stats.get("estimated_has_mask_rate") or 0.0
        if stats.get("n_samples") and split_rate == 0.0:
            findings.append(finding(
                CRITICAL,
                f"no_masks_in_{split_name}",
                f"The '{split_name}' split has {stats['n_samples']} samples and none "
                f"of them carry a usable mask.",
            ))

    return findings


_SUMMARY_SUFFIXES = ("_mean", "_std", "_p10", "_median", "_p90")


def _merge_split_stats(per_split: Dict[str, dict]) -> dict:
    """Recompute an overall view by treating all splits as one pool."""
    overall: Dict[str, object] = {"split": "all"}

    counted = ["n_samples", "n_decoded", "n_mask_empty_after_threshold",
               "n_mask_unreadable", "n_mask_covers_whole_image",
               "n_usable_masks_decoded", "estimated_has_mask_count"]
    for key in counted:
        overall[key] = sum(int(s.get(key) or 0) for s in per_split.values())

    status_counts: Dict[str, int] = {}
    for stats in per_split.values():
        for status, count in (stats.get("status_counts") or {}).items():
            status_counts[status] = status_counts.get(status, 0) + count
    overall["status_counts"] = status_counts

    overall["estimated_has_mask_rate"] = (
        overall["estimated_has_mask_count"] / overall["n_samples"]
        if overall["n_samples"] else 0.0
    )
    overall["usable_rate_among_decoded"] = (
        overall["n_usable_masks_decoded"] / overall["n_decoded"]
        if overall["n_decoded"] else 0.0
    )

    raw_maxes = [s.get("raw_pixel_max_over_decoded") for s in per_split.values()]
    raw_maxes = [v for v in raw_maxes if v is not None]
    overall["raw_pixel_max_over_decoded"] = max(raw_maxes) if raw_maxes else None

    # Weight each split's mean by how many masks it actually decoded.
    metric_keys = {
        key
        for stats in per_split.values()
        for key in stats
        if key.endswith(_SUMMARY_SUFFIXES)
    }
    for key in sorted(metric_keys):
        total_weight = 0.0
        total = 0.0
        for stats in per_split.values():
            value = stats.get(key)
            weight = float(stats.get("n_usable_masks_decoded") or 0)
            if value is None or weight <= 0:
                continue
            total += float(value) * weight
            total_weight += weight
        overall[key] = (total / total_weight) if total_weight > 0 else None

    return overall


def run_mask_audit(
    loaders: Dict[str, object],
    run_dir,
    image_size: int,
    cam_size,
    max_decoded: Optional[int] = 600,
    num_examples: int = 12,
    seed: int = 42,
) -> Optional[dict]:
    """
    Audit every split's masks and write the report under `<run_dir>/diagnostics/`.

    `loaders` maps a split name to a DataLoader (any of them may be None).
    """
    diagnostics_dir = Path(run_dir) / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = None
    per_split: Dict[str, dict] = {}
    all_examples: List[dict] = []

    for split_name, loader in loaders.items():
        base, indices = loader_base_and_indices(loader)
        if base is None or not indices:
            continue
        base_dataset = base

        stats = _audit_split(
            base_dataset=base,
            indices=indices,
            split_name=split_name,
            image_size=image_size,
            cam_size=cam_size,
            max_decoded=max_decoded,
            seed=seed,
        )
        all_examples.extend(stats.pop("_examples", []))
        per_split[split_name] = stats

    if base_dataset is None or not per_split:
        print("[reports] mask audit skipped: no usable dataloaders.")
        return None

    overall = _merge_split_stats(per_split)
    findings = _build_findings(base_dataset, per_split, overall, cam_size)

    examples_dir = _save_examples(
        examples=all_examples,
        out_dir=diagnostics_dir / "mask_examples",
        image_size=image_size,
        cam_size=cam_size,
        num_examples=num_examples,
        seed=seed,
    )

    payload = {
        "image_size": image_size,
        "cam_grid": list(cam_size),
        "classes": list(base_dataset.classes),
        "mask_dirs": {k: str(v) for k, v in base_dataset.mask_dirs.items()},
        "mask_manifests": {k: str(v) for k, v in base_dataset.mask_manifests.items()},
        "max_decoded_per_split": max_decoded,
        "per_split": per_split,
        "overall": overall,
        "findings": findings,
        "examples_dir": examples_dir,
    }

    write_json(diagnostics_dir / "mask_audit.json", payload)
    write_lines(diagnostics_dir / "mask_audit.md", _render_markdown(payload))

    print_findings(findings, "Mask audit")
    print(f"  saved: {diagnostics_dir / 'mask_audit.md'}")

    return payload


def _render_markdown(payload: dict) -> List[str]:
    overall = payload["overall"]
    lines: List[str] = [
        "# Mask audit",
        "",
        f"- image size: `{payload['image_size']}`",
        f"- CAM grid: `{tuple(payload['cam_grid'])}`",
        f"- dataset classes: `{payload['classes']}`",
        "",
        "## Findings",
        "",
    ]
    lines += render_findings(payload["findings"])
    lines += [
        "",
        "## Mask sources",
        "",
        "| class | mask dir | manifest |",
        "| --- | --- | --- |",
    ]
    for class_name in payload["classes"]:
        mask_dir = payload["mask_dirs"].get(class_name, "(none)")
        manifest = payload["mask_manifests"].get(class_name, "(none)")
        lines.append(f"| `{class_name}` | `{mask_dir}` | `{manifest}` |")

    lines += [
        "",
        "## Per-split summary",
        "",
        "| split | samples | est. has_mask | rate | decoded | usable | empty after threshold |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split_name, stats in payload["per_split"].items():
        lines.append(
            f"| {split_name} | {stats['n_samples']} | "
            f"{stats['estimated_has_mask_count']} | "
            f"{fmt(stats['estimated_has_mask_rate'], 3)} | "
            f"{stats['n_decoded']} | {stats['n_usable_masks_decoded']} | "
            f"{stats['n_mask_empty_after_threshold']} |"
        )

    lines += [
        "",
        "### Why samples are dropped (structural, counted over every sample)",
        "",
        "| status | count |",
        "| --- | --- |",
    ]
    for status, count in sorted(overall["status_counts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{status}` | {count} |")
    lines += [
        "",
        f"Counted over the {overall['n_decoded']} decoded masks only: "
        f"{overall['n_mask_empty_after_threshold']} thresholded to empty, "
        f"{overall['n_mask_unreadable']} could not be read, "
        f"{overall['n_mask_covers_whole_image']} cover the whole frame.",
    ]

    lines += [
        "",
        "## Mask geometry (over decoded, usable masks)",
        "",
        "| metric | value | meaning |",
        "| --- | --- | --- |",
        f"| coverage (full res) | {fmt(overall.get('coverage_full_mean'), 4)} | "
        "fraction of pixels the mask calls object |",
        f"| coverage (CAM grid) | {fmt(overall.get('coverage_cam_grid_mean'), 4)} | "
        "same, after area-downsampling to the CAM resolution |",
        f"| border coverage | {fmt(overall.get('border_coverage_mean'), 4)} | "
        "coverage in the outer frame; should be LOW for object masks |",
        f"| center coverage | {fmt(overall.get('center_coverage_mean'), 4)} | "
        "coverage in the central box; should be HIGH for object masks |",
        f"| uniform containment | {fmt(overall.get('uniform_containment_mean'), 4)} | "
        "cam_loss of a flat, uninformative CAM &mdash; the beat-me baseline |",
        f"| floor containment | {fmt(overall.get('floor_containment_mean'), 4)} | "
        "cam_loss of a perfect CAM &mdash; the value it can never go below |",
        f"| pure-background cells | {fmt(overall.get('pure_background_cell_frac_mean'), 4)} | "
        "fraction of CAM cells with zero object overlap |",
        f"| pure-object cells | {fmt(overall.get('pure_object_cell_frac_mean'), 4)} | "
        "fraction of CAM cells fully inside the object |",
        "",
    ]

    if payload.get("examples_dir"):
        lines += [
            f"Example panels: `{payload['examples_dir']}`",
            "",
        ]

    return lines
