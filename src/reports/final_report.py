"""
End-of-run diagnostics: plots plus a rule-based reading of what happened.

The rules encode the specific ways this setup fails quietly. Each one is a
statement that can be checked against the numbers the run already collected,
so the report says "the CAM loss never touched a sample" instead of leaving a
suspiciously flat curve for someone to notice later.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.reports.common import (
    CRITICAL,
    INFO,
    OK,
    WARNING,
    finding,
    fmt,
    get_pyplot,
    nan_series,
    print_findings,
    render_findings,
    write_json,
    write_lines,
)


def _last(values: Sequence, default=None):
    for value in reversed(list(values)):
        if value is not None:
            return value
    return default


def _first(values: Sequence, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def _delta(values: Sequence) -> Optional[float]:
    first, last = _first(values), _last(values)
    if first is None or last is None:
        return None
    return float(last) - float(first)


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

def write_cam_metrics_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _plot(run_dir: Path, history: dict, cam_rows: List[dict], lambda_cam: float) -> Optional[str]:
    plt = get_pyplot()
    if plt is None or not cam_rows:
        return None

    epochs = [row.get("epoch") for row in cam_rows]

    def col(name: str) -> np.ndarray:
        return nan_series([row.get(name) for row in cam_rows])

    fig, axes = plt.subplots(3, 3, figsize=(19, 14))

    # 1. loss terms
    ax = axes[0][0]
    ax.plot(history["epoch"], history["train_cls_loss"], label="train CE")
    ax.plot(history["epoch"], history["val_cls_loss"], label="val CE")
    ax.plot(history["epoch"], history["train_cam_loss"], label="train CAM (raw)")
    ax.plot(history["epoch"], history["val_cam_loss"], label="val CAM (raw)")
    ax.plot(
        history["epoch"],
        [lambda_cam * v for v in history["train_cam_loss"]],
        "--", label=f"train CAM x lambda ({lambda_cam})",
    )
    ax.set_title("Loss terms")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2. containment against its two baselines
    ax = axes[0][1]
    ax.plot(epochs, col("train_containment_mean"), label="train containment")
    ax.plot(epochs, col("val_containment_mean"), label="val containment")
    ax.plot(epochs, col("val_uniform_containment_mean"), "k--",
            label="flat-CAM baseline (no localisation)")
    ax.plot(epochs, col("val_floor_containment_mean"), "g--",
            label="perfect-CAM floor")
    ax.set_title("Containment vs. what is achievable")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3. relative containment: the only scale-free "is it working" curve
    ax = axes[0][2]
    ax.plot(epochs, col("train_relative_containment_mean"), label="train")
    ax.plot(epochs, col("val_relative_containment_mean"), label="val")
    ax.axhline(1.0, color="r", ls="--", label="1.0 = no better than a flat CAM")
    ax.set_title("Relative containment (containment / flat baseline)")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4. absolute activation - the "is the background near zero?" panel
    ax = axes[1][0]
    ax.plot(epochs, col("train_mean_pos_inside_mean"), label="train object")
    ax.plot(epochs, col("train_mean_pos_outside_mean"), label="train background")
    ax.plot(epochs, col("val_mean_pos_inside_mean"), "--", label="val object")
    ax.plot(epochs, col("val_mean_pos_outside_mean"), "--", label="val background")
    ax.set_title("Mean POSITIVE activation, raw CAM units")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. signed activation
    ax = axes[1][1]
    ax.plot(epochs, col("train_mean_cam_inside_mean"), label="train object")
    ax.plot(epochs, col("train_mean_cam_outside_mean"), label="train background")
    ax.plot(epochs, col("val_mean_cam_inside_mean"), "--", label="val object")
    ax.plot(epochs, col("val_mean_cam_outside_mean"), "--", label="val background")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_title("Mean SIGNED CAM value")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 6. how much of the map is still switched on, and how much has collapsed
    ax = axes[1][2]
    ax.plot(epochs, col("train_positive_background_cell_frac_mean"),
            label="train: background cells with CAM > 0")
    ax.plot(epochs, col("val_positive_background_cell_frac_mean"),
            label="val: background cells with CAM > 0")
    ax.plot(epochs, col("train_cam_all_negative_mean"), "r-", lw=2,
            label="train: samples with an ALL-NEGATIVE CAM")
    ax.plot(epochs, col("val_cam_all_negative_mean"), "r--", lw=2,
            label="val: samples with an ALL-NEGATIVE CAM")
    ax.set_title("Live background vs. collapsed CAMs")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 7. localisation vs. accuracy
    ax = axes[2][0]
    ax.plot(epochs, col("val_pointing_hit_mean"), label="val pointing game")
    ax.plot(epochs, col("val_mask_coverage_cam_grid_mean"), "k--",
            label="mask coverage (chance level)")
    ax.plot(history["epoch"], history["val_acc"], label="val accuracy")
    ax.set_title("Localisation vs. classification")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 8. gradient magnitudes
    ax = axes[2][1]
    ce = col("grad_all/ce_grad_norm")
    cam = col("grad_all/weighted_cam_grad_norm")
    if np.isfinite(ce).any():
        ax.plot(epochs, ce, label="|grad CE|")
        ax.plot(epochs, cam, label="|lambda * grad CAM|")
        ax.set_yscale("log")
    ax.set_title("Gradient magnitude reaching the weights")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 9. gradient conflict
    ax = axes[2][2]
    cosine = col("grad_all/cosine_ce_cam")
    if np.isfinite(cosine).any():
        ax.plot(epochs, cosine, label="cos(grad CE, grad CAM)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylim(-1, 1)
    ax.set_title("Do the two objectives agree?")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = Path(run_dir) / "diagnostics" / "cam_diagnostics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------
# Findings
# ----------------------------------------------------------------------

def _build_findings(
    history: dict,
    cam_rows: List[dict],
    lambda_cam: float,
    mask_audit: Optional[dict],
) -> List[dict]:
    findings: List[dict] = []

    if not cam_rows:
        return [finding(WARNING, "no_cam_stats", "No CAM statistics were collected.")]

    def col(name: str) -> List[Optional[float]]:
        return [row.get(name) for row in cam_rows]

    # Trend-based rules need at least a few points; on a 1-2 epoch run the
    # first and last epoch are the same measurement and every delta is noise.
    trends_meaningful = len(cam_rows) >= 3

    masked = _last(col("train_n_samples_with_mask"), 0) or 0
    val_masked = _last(col("val_n_samples_with_mask"), 0) or 0

    if masked == 0:
        findings.append(finding(
            CRITICAL,
            "cam_loss_inactive",
            "Not a single TRAINING sample reached the CAM loss (has_mask=0 "
            "everywhere), so cam_loss was identically 0 and lambda_cam changed "
            "nothing. Start from diagnostics/mask_audit.md.",
        ))
        return findings

    train_rate = _last(col("train_mask_rate"))
    if train_rate is not None and train_rate < 0.5:
        findings.append(finding(
            WARNING,
            "sparse_mask_supervision",
            f"Only {fmt(train_rate * 100, 1)}% of training samples carried a mask. "
            f"The logged cam_loss is averaged over batches, so batches with no "
            f"masked sample contribute a hard 0 and drag the curve down without "
            f"anything having been learned.",
        ))

    if val_masked == 0:
        findings.append(finding(
            WARNING,
            "no_val_masks",
            "No validation sample carried a mask, so val_cam_loss is structurally 0 "
            "and cannot be compared against training.",
        ))

    # --- the degenerate minimum ----------------------------------------
    # containment = sum(relu(cam)*outside) / (sum(relu(cam)) + eps). Drive the
    # entire CAM below zero and both sums vanish: the loss returns 0/eps = 0,
    # a perfect score from a map that carries no information. Nothing else in
    # the objective objects, because the logits are GAP(cam) and cross-entropy
    # only reads the difference between classes, not their absolute level.
    all_negative = _last(col("train_cam_all_negative_mean"))
    val_all_negative = _last(col("val_cam_all_negative_mean"))
    if all_negative is not None and all_negative > 0.05:
        findings.append(finding(
            CRITICAL,
            "cam_collapsed_negative",
            f"{fmt(all_negative * 100, 1)}% of masked training samples "
            f"({fmt((val_all_negative or 0) * 100, 1)}% on val) end the run with a CAM "
            f"that is negative EVERYWHERE. relu() then zeroes both the numerator and "
            f"the denominator of the containment ratio, so cam_loss reports a perfect "
            f"0 for a map that localises nothing. This is a genuine minimum of the "
            f"loss as written, not a fluke: cross-entropy over GAP(cam) only constrains "
            f"the difference between class maps, so it never penalises pushing them all "
            f"below zero. Any reported cam_loss near 0 has to be read together with "
            f"`total_positive_energy` before it means anything.",
        ))

    collapsed = bool(all_negative is not None and all_negative > 0.05)

    positive_frac = col("train_positive_cell_frac_mean")
    first_frac, last_frac = _first(positive_frac), _last(positive_frac)
    if (
        trends_meaningful and first_frac and last_frac is not None
        and first_frac > 0.05 and last_frac < 0.25 * first_frac
        and not collapsed
    ):
        findings.append(finding(
            WARNING,
            "cam_positive_region_collapsing",
            f"The share of CAM cells above zero fell from {fmt(first_frac, 3)} to "
            f"{fmt(last_frac, 3)}. The loss is being satisfied by switching cells off "
            f"rather than by concentrating energy on the object; cells that go negative "
            f"stop receiving any CAM gradient at all. Watch the sign panels in "
            f"diagnostics/snapshots/ to see how much of the map is left alive.",
        ))

    # --- is the ratio actually improving? ------------------------------
    relative = col("train_relative_containment_mean")
    last_relative = _last(relative)
    if last_relative is not None:
        if last_relative > 0.95:
            findings.append(finding(
                CRITICAL,
                "cam_not_localised",
                f"Final train relative containment is {fmt(last_relative, 3)}: the CAM "
                f"puts essentially the same share of its energy in the background as a "
                f"flat, uninformative CAM would ({fmt(_last(col('train_uniform_containment_mean')), 3)}). "
                f"The absolute cam_loss value looks small only because the masks are "
                f"large, not because the CAM is localised.",
            ))
        elif last_relative < 0.6 and not collapsed:
            findings.append(finding(
                OK,
                "cam_localised",
                f"Final train relative containment is {fmt(last_relative, 3)}, "
                f"comfortably better than a flat CAM (1.0).",
            ))

    containment_delta = _delta(col("train_containment_mean")) if trends_meaningful else None
    if containment_delta is not None and containment_delta > -0.005:
        findings.append(finding(
            WARNING,
            "containment_not_improving",
            f"Train containment moved by {fmt(containment_delta, 4)} over the whole run "
            f"(negative = improving). The CAM loss is not making progress; check the "
            f"gradient panel before raising lambda_cam.",
        ))

    # --- the scale-inflation trap --------------------------------------
    # containment is a *ratio*, so it can fall while the background CAM value
    # rises. If the goal is "CAM near zero in the background", that is a
    # failure dressed up as success.
    bg_delta = _delta(col("train_mean_pos_outside_mean")) if trends_meaningful else None
    fg_delta = _delta(col("train_mean_pos_inside_mean")) if trends_meaningful else None
    if containment_delta is not None and bg_delta is not None:
        if containment_delta < -0.01 and bg_delta > 0:
            findings.append(finding(
                CRITICAL,
                "scale_inflation",
                f"Containment improved by {fmt(containment_delta, 4)} while the mean "
                f"POSITIVE background activation still ROSE by {fmt(bg_delta, 4)} "
                f"(object activation changed by {fmt(fg_delta, 4)}). CAMMaskLoss "
                f"optimises a scale-invariant ratio, so the model satisfied it by "
                f"inflating the object activation rather than by pushing the "
                f"background toward zero. If you want background CAM ~ 0 in absolute "
                f"terms, the ratio has to be paired with an absolute penalty on "
                f"relu(cam)*outside.",
            ))
        elif bg_delta < 0 and not collapsed:
            findings.append(finding(
                OK,
                "background_suppressed",
                f"Mean positive background activation fell by {fmt(-bg_delta, 4)} in "
                f"absolute CAM units.",
            ))

    bg_positive = _last(col("train_positive_background_cell_frac_mean"))
    if bg_positive is not None and bg_positive > 0.7:
        findings.append(finding(
            WARNING,
            "background_still_active",
            f"{fmt(bg_positive * 100, 1)}% of background cells still hold a positive "
            f"CAM value at the end of training. Cells that go negative drop out of the "
            f"relu and stop receiving any CAM gradient, so this number is the share of "
            f"the background the loss is still fighting over.",
        ))

    # --- pointing game vs chance ---------------------------------------
    pointing = _last(col("val_pointing_hit_mean"))
    coverage = _last(col("val_mask_coverage_cam_grid_mean"))
    if pointing is not None and coverage is not None:
        if pointing <= coverage + 0.02:
            findings.append(finding(
                CRITICAL,
                "pointing_game_at_chance",
                f"The CAM peak lands on the object {fmt(pointing * 100, 1)}% of the time, "
                f"against a chance level of {fmt(coverage * 100, 1)}% (the mask covers "
                f"that fraction of the CAM grid). The CAM carries no usable localisation.",
            ))
        elif not collapsed:
            findings.append(finding(
                OK,
                "pointing_game",
                f"CAM peak hits the object {fmt(pointing * 100, 1)}% of the time vs. "
                f"{fmt(coverage * 100, 1)}% chance.",
            ))

    # --- gradients ------------------------------------------------------
    ratio = _last(col("grad_all/cam_over_ce_ratio"))
    cam_grad_norm = _last(col("grad_all/cam_grad_norm"))
    if ratio is not None:
        if cam_grad_norm is not None and cam_grad_norm <= 0.0:
            findings.append(finding(
                CRITICAL,
                "cam_gradient_zero",
                "The CAM loss produced EXACTLY zero gradient in the last epoch. It is "
                "saturated, not weak: raising lambda_cam multiplies zero and changes "
                "nothing. With this loss that means every CAM cell is already negative, "
                "so relu() blocks the whole backward path. The loss needs a term that "
                "still has gradient there (for example an absolute penalty on the "
                "background CAM, or dropping the relu).",
            ))
        elif ratio < 0.01:
            findings.append(finding(
                WARNING,
                "cam_gradient_negligible",
                f"|lambda * grad(cam)| is {fmt(ratio * 100, 2)}% of |grad(ce)|. The CAM "
                f"term is decorative at lambda_cam={lambda_cam}; raise it by roughly "
                f"{fmt(1.0 / max(ratio, 1e-9), 1)}x to make the two comparable.",
            ))
        elif ratio > 10:
            findings.append(finding(
                WARNING,
                "cam_gradient_dominant",
                f"|lambda * grad(cam)| is {fmt(ratio, 2)}x |grad(ce)|. The CAM term is "
                f"drowning the classifier; expect accuracy to suffer.",
            ))
        else:
            findings.append(finding(
                OK,
                "gradient_balance",
                f"|lambda * grad(cam)| / |grad(ce)| = {fmt(ratio, 3)}, a reasonable balance.",
            ))

    cosine = _last(col("grad_all/cosine_ce_cam"))
    if cosine is not None and cosine < -0.2:
        findings.append(finding(
            WARNING,
            "objectives_conflict",
            f"cos(grad CE, grad CAM) = {fmt(cosine, 3)}: the two objectives pull the "
            f"weights in opposing directions, so localisation is being bought directly "
            f"with accuracy. Worth checking mask quality before increasing lambda_cam.",
        ))

    head_ratio = _last(col("grad_cam_head/cam_over_ce_ratio"))
    if head_ratio is not None and ratio is not None and head_ratio > 5 * max(ratio, 1e-9):
        findings.append(finding(
            INFO,
            "cam_gradient_concentrated_in_head",
            f"The CAM gradient is {fmt(head_ratio, 3)}x the CE gradient at the 1x1 CAM "
            f"head but only {fmt(ratio, 3)}x overall: the loss is mostly rescaling the "
            f"classifier weights rather than reshaping the features.",
        ))

    # --- generalisation of the localisation ----------------------------
    train_c = _last(col("train_containment_mean"))
    val_c = _last(col("val_containment_mean"))
    if train_c is not None and val_c is not None and val_c > train_c + 0.1:
        findings.append(finding(
            WARNING,
            "localisation_overfit",
            f"Containment is {fmt(train_c, 3)} on train but {fmt(val_c, 3)} on val. The "
            f"CAM shaping is memorising the training images rather than generalising "
            f"(expected with train_shots_per_class set low).",
        ))

    # --- accuracy cost --------------------------------------------------
    val_acc = history.get("val_acc") or []
    if val_acc:
        findings.append(finding(
            INFO,
            "accuracy",
            f"Best val accuracy {fmt(max(val_acc), 4)}, final {fmt(val_acc[-1], 4)}, "
            f"at lambda_cam={lambda_cam}. Compare against a lambda_cam=0 run to price "
            f"the localisation.",
        ))

    if mask_audit:
        for item in mask_audit.get("findings", []):
            if item.get("level") == CRITICAL:
                findings.append(finding(
                    CRITICAL,
                    f"mask_audit::{item.get('code')}",
                    f"(from the mask audit) {item.get('message')}",
                ))

    return findings


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def write_final_report(
    run_dir,
    history: dict,
    cam_rows: List[dict],
    lambda_cam: float,
    config: dict,
    mask_audit: Optional[dict] = None,
) -> dict:
    run_dir = Path(run_dir)
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    write_cam_metrics_csv(diagnostics_dir / "cam_metrics.csv", cam_rows)
    plot_path = _plot(run_dir, history, cam_rows, lambda_cam)
    findings = _build_findings(history, cam_rows, lambda_cam, mask_audit)

    payload = {
        "lambda_cam": lambda_cam,
        "epochs_recorded": len(cam_rows),
        "findings": findings,
        "plot": plot_path,
        "last_epoch_metrics": cam_rows[-1] if cam_rows else {},
    }
    write_json(diagnostics_dir / "cam_diagnostics.json", payload)
    write_lines(
        diagnostics_dir / "report.md",
        _render_markdown(payload, history, cam_rows, lambda_cam, config, mask_audit),
    )

    print_findings(findings, "CAM training diagnostics")
    print(f"  saved: {diagnostics_dir / 'report.md'}")

    return payload


_GLOSSARY = [
    ("containment", "share of positive CAM energy sitting in the background. "
                    "This IS the value CAMMaskLoss returns."),
    ("flat baseline", "containment of a CAM that is constant everywhere. Beating it "
                      "is the minimum bar; equalling it means no localisation."),
    ("floor", "containment of a perfect CAM that puts all energy in the single "
              "most-inside cell. cam_loss cannot go below this."),
    ("relative containment", "containment / flat baseline. < 1 good, ~1 useless, > 1 "
                             "anti-localised. Scale-free, so it is comparable across "
                             "datasets and mask sizes."),
    ("mean positive background", "average of relu(CAM) over background cells, in raw "
                                 "CAM units. This is the number that has to fall if "
                                 "the goal is 'CAM near zero in the background'."),
    ("positive background cells", "fraction of background cells with CAM > 0. Cells "
                                  "below zero get no gradient from the CAM term."),
    ("pointing game", "how often the CAM's peak cell lies on the object. Chance level "
                      "is the mask's coverage of the CAM grid."),
    ("all-negative CAM", "fraction of samples whose CAM is below zero everywhere. "
                         "relu() empties both sides of the ratio, so these score a "
                         "perfect containment of 0 while localising nothing. Always "
                         "check this before believing a low cam_loss."),
    ("total positive CAM energy", "sum of relu(CAM) over the whole map. Collapsing "
                                  "toward 0 is the signature of the degenerate "
                                  "solution above."),
]


def _render_markdown(
    payload: dict,
    history: dict,
    cam_rows: List[dict],
    lambda_cam: float,
    config: dict,
    mask_audit: Optional[dict],
) -> List[str]:
    lines = [
        "# CAM training diagnostics",
        "",
        f"- `lambda_cam` = `{lambda_cam}`",
        f"- `model_type` = `{config.get('model_type')}`, "
        f"`freeze_backbone` = `{config.get('freeze_backbone')}`",
        f"- epochs recorded: {len(cam_rows)}",
        "",
        "## Findings",
        "",
    ]
    lines += render_findings(payload["findings"])

    if cam_rows:
        last = cam_rows[-1]
        first = cam_rows[0]
        lines += [
            "",
            "## Key numbers (first epoch -> last epoch)",
            "",
            "| metric | train first | train last | val last |",
            "| --- | --- | --- | --- |",
        ]
        rows = [
            ("containment (= cam_loss)", "containment_mean"),
            ("flat-CAM baseline", "uniform_containment_mean"),
            ("perfect-CAM floor", "floor_containment_mean"),
            ("relative containment", "relative_containment_mean"),
            ("mean positive activation, object", "mean_pos_inside_mean"),
            ("mean positive activation, background", "mean_pos_outside_mean"),
            ("mean signed CAM, background", "mean_cam_outside_mean"),
            ("positive background cells", "positive_background_cell_frac_mean"),
            ("peak background / peak object", "peak_outside_over_inside_mean"),
            ("total positive CAM energy", "total_positive_energy_mean"),
            ("samples with an ALL-NEGATIVE CAM", "cam_all_negative_mean"),
            ("pointing game", "pointing_hit_mean"),
            ("samples with a mask", "n_samples_with_mask"),
        ]
        for label, key in rows:
            lines.append(
                f"| {label} | {fmt(first.get('train_' + key))} | "
                f"{fmt(last.get('train_' + key))} | {fmt(last.get('val_' + key))} |"
            )

        lines += [
            "",
            "## Gradient balance (last epoch)",
            "",
            "| group | \\|grad CE\\| | \\|lambda*grad CAM\\| | ratio | cos |",
            "| --- | --- | --- | --- | --- |",
        ]
        for group in ("all", "cam_head", "adapters", "backbone"):
            ce = last.get(f"grad_{group}/ce_grad_norm")
            if ce is None:
                continue
            lines.append(
                f"| `{group}` | {fmt(ce, 6)} | "
                f"{fmt(last.get(f'grad_{group}/weighted_cam_grad_norm'), 6)} | "
                f"{fmt(last.get(f'grad_{group}/cam_over_ce_ratio'), 4)} | "
                f"{fmt(last.get(f'grad_{group}/cosine_ce_cam'), 4)} |"
            )

    if history.get("val_acc"):
        lines += [
            "",
            "## Classification",
            "",
            f"- best val accuracy: `{fmt(max(history['val_acc']), 4)}`",
            f"- final val accuracy: `{fmt(history['val_acc'][-1], 4)}`",
            f"- final train accuracy: `{fmt(history['train_acc'][-1], 4)}`",
        ]

    lines += [
        "",
        "## How to read these numbers",
        "",
        "| term | meaning |",
        "| --- | --- |",
    ]
    lines += [f"| {name} | {meaning} |" for name, meaning in _GLOSSARY]

    lines += [
        "",
        "## Artifacts",
        "",
        "- `diagnostics/cam_metrics.csv` &mdash; every metric, per epoch, per split",
        "- `diagnostics/cam_diagnostics.png` &mdash; the nine diagnostic panels",
        "- `diagnostics/snapshots/epoch_XXX.png` &mdash; the same images every epoch",
        "- `diagnostics/mask_audit.md` &mdash; are the masks reaching the loss at all",
        "- `cam_debug/` &mdash; best/worst IoU overlays on the test split",
        "",
    ]

    if mask_audit:
        overall = mask_audit.get("overall", {})
        lines += [
            "## Mask audit summary",
            "",
            f"- samples with a usable mask: `{overall.get('estimated_has_mask_count')}"
            f"/{overall.get('n_samples')}` "
            f"({fmt((overall.get('estimated_has_mask_rate') or 0) * 100, 1)}%)",
            f"- mean mask coverage: `{fmt(overall.get('coverage_full_mean'), 4)}`",
            f"- flat-CAM baseline: `{fmt(overall.get('uniform_containment_mean'), 4)}`, "
            f"floor: `{fmt(overall.get('floor_containment_mean'), 4)}`",
            "",
            "See `diagnostics/mask_audit.md` for the full breakdown.",
            "",
        ]

    return lines
