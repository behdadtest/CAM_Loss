"""Accuracy-vs-shots sweep over lambda_cam.

Every cell is run with several seeds and reported as mean +/- std. With one image
per class, re-running the same configuration can move accuracy by several points,
so a single-seed difference between two lambda values carries no information.
"""

import csv
import statistics
from pathlib import Path

import yaml

from main import build_dataloaders, load_config, run
from src.utils import set_seed

# ---------------- sweep settings ----------------
CONFIG_PATH = Path("configs/config.yaml")
OUTPUT_DIR = Path("sweep_results")
SHOTS = [1, 2, 4, 8, 16]
LAMBDAS = [0.0, 0.3, 0.5, 1.0, 3.0]
SEEDS = [0, 1, 2]
SWEEP_EPOCHS = 20
# -------------------------------------------------


def run_single(base_config: dict, shots: int, lam: float, seed: int) -> dict:
    config = dict(base_config)
    config["train_shots_per_class"] = shots
    config["train_fraction"] = None
    config["lambda_cam"] = lam
    config["epochs"] = SWEEP_EPOCHS
    config["seed"] = seed
    config["output_dir"] = str(OUTPUT_DIR / "runs" / f"shots{shots}_lambda{lam}_seed{seed}")
    # cam_debug per cell would be 75 sets of panels; the aggregate CSV is enough.
    config["cam_debug"] = bool(base_config.get("sweep_cam_debug", False))

    result = run(config)

    cam_summary = result.get("cam_summary") or {}

    return {
        "shots": shots,
        "lambda_cam": lam,
        "seed": seed,
        "best_val_acc": result["best_val_acc"],
        "last_train_acc": result["history"]["train_acc"][-1],
        "last_val_acc": result["history"]["val_acc"][-1],
        "bg_energy_mean": cam_summary.get("bg_energy_mean"),
        "pointing_game_accuracy": cam_summary.get("pointing_game_accuracy"),
        "run_dir": str(result["run_dir"]),
    }


def aggregate(rows):
    """Group per-seed rows into mean/std cells."""
    cells = {}
    for row in rows:
        cells.setdefault((row["shots"], row["lambda_cam"]), []).append(row)

    out = []
    for (shots, lam), group in sorted(cells.items()):
        accs = [r["best_val_acc"] for r in group]
        bg = [r["bg_energy_mean"] for r in group if r["bg_energy_mean"] is not None]
        out.append({
            "shots": shots,
            "lambda_cam": lam,
            "n_seeds": len(group),
            "best_val_acc_mean": statistics.mean(accs),
            "best_val_acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "bg_energy_mean": statistics.mean(bg) if bg else None,
            "bg_energy_std": statistics.stdev(bg) if len(bg) > 1 else 0.0,
        })
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_config = load_config(CONFIG_PATH)

    rows = []
    per_seed_path = OUTPUT_DIR / "results_per_seed.csv"

    fieldnames = [
        "shots", "lambda_cam", "seed", "best_val_acc", "last_train_acc", "last_val_acc",
        "bg_energy_mean", "pointing_game_accuracy", "run_dir",
    ]

    with open(per_seed_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for shots in SHOTS:
            for lam in LAMBDAS:
                for seed in SEEDS:
                    print(f"\n{'=' * 60}")
                    print(f"RUNNING: shots={shots} | lambda_cam={lam} | seed={seed}")
                    print(f"{'=' * 60}\n")

                    row = run_single(base_config, shots, lam, seed)
                    rows.append(row)

                    writer.writerow(row)
                    f.flush()

                    print(f"\n>>> DONE shots={shots} lambda={lam} seed={seed} "
                          f"-> best_val_acc={row['best_val_acc']:.4f}\n")

    summary = aggregate(rows)
    summary_path = OUTPUT_DIR / "results.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nPer-seed results : {per_seed_path}")
    print(f"Aggregated       : {summary_path}")
    plot_results(summary)


def plot_results(summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for lam in LAMBDAS:
        cells = [c for c in summary if c["lambda_cam"] == lam]
        cells.sort(key=lambda c: c["shots"])
        if not cells:
            continue

        xs = [c["shots"] for c in cells]
        axes[0].errorbar(
            xs,
            [c["best_val_acc_mean"] * 100 for c in cells],
            yerr=[c["best_val_acc_std"] * 100 for c in cells],
            marker="o", capsize=3, label=f"lambda_cam={lam}",
        )

        bg_cells = [c for c in cells if c["bg_energy_mean"] is not None]
        if bg_cells:
            axes[1].errorbar(
                [c["shots"] for c in bg_cells],
                [c["bg_energy_mean"] for c in bg_cells],
                yerr=[c["bg_energy_std"] for c in bg_cells],
                marker="o", capsize=3, label=f"lambda_cam={lam}",
            )

    axes[0].set_xlabel("Labeled training examples per class (shots)")
    axes[0].set_ylabel("Best val accuracy (%)")
    axes[0].set_title(f"Accuracy vs shots (mean +/- std over {len(SEEDS)} seeds)")

    axes[1].set_xlabel("Labeled training examples per class (shots)")
    axes[1].set_ylabel("Background energy fraction (lower is better)")
    axes[1].set_title("CAM localisation vs shots")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(SHOTS)
        ax.set_xticklabels(SHOTS)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / "sweep_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Plot saved to: {out_path}")


if __name__ == "__main__":
    main()
