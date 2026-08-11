"""
Diagnostic reports for the CAM-mask training pipeline.

Nothing in this package changes how the model is trained. Every module here
only observes the run and writes artifacts under `<run_dir>/diagnostics/`, so
that a finished run can be inspected offline to answer:

  1. Are the SAM pseudo-masks actually reaching the loss? (`mask_audit`)
  2. Is the CAM loss measurably reshaping the CAM? (`cam_stats`)
  3. Is the CAM gradient large enough to matter, and does it fight the
     classification gradient? (`grad_probe`)
  4. What does the CAM look like, epoch by epoch? (`snapshots`)
  5. Pulled together, what is going wrong? (`final_report`)
"""

from src.reports.cam_stats import CAMStatsAccumulator
from src.reports.final_report import write_final_report
from src.reports.grad_probe import run_grad_probe
from src.reports.mask_audit import run_mask_audit
from src.reports.snapshots import CAMSnapshotter

__all__ = [
    "CAMStatsAccumulator",
    "CAMSnapshotter",
    "run_grad_probe",
    "run_mask_audit",
    "write_final_report",
]
