# ResNet18 + Multi-Class CAM Loss

This project trains a ResNet18 classifier with:

```text
total_loss = classification_loss + lambda_cam * cam_mask_loss
```

`cam_mask_loss` uses SAM pseudo-masks to discourage the class activation map
from firing on the background.

## What the CAM loss actually computes

`src/losses.py` does **not** implement an MSE-to-zero penalty. It implements a
normalized *energy containment ratio* on the target-class CAM:

```text
positive     = relu(cam[label])
containment  = sum(positive * outside) / (sum(positive) + eps)
```

Two properties of that formula drive most of what you will see in a run, and
both are measured explicitly by the diagnostics below:

1. **It is scale-invariant.** Multiplying the whole CAM by 10 leaves the loss
   unchanged, so the model can lower it by inflating the object activation
   instead of pushing the background toward zero. If you want "CAM near zero in
   the background" in absolute terms, watch `mean_pos_outside`, not the loss.
2. **`relu` gates the gradient.** A cell that is already negative contributes
   nothing to either sum and receives exactly zero gradient from this term.
   Taken to its conclusion, a CAM that is negative *everywhere* scores
   `0 / eps = 0` &mdash; a perfect loss from a map that localises nothing. Since
   the logits are `GAP(cam)` and cross-entropy only reads differences between
   classes, nothing else in the objective penalises that collapse.

## Layout

```text
main.py                    single training run, driven by configs/config.yaml
sweep.py                   grid over train_shots_per_class x lambda_cam
configs/config.yaml        all settings
src/model.py               ResNet18 + 1x1 CAM head + GAP
src/model_with_adapter.py  same, with Conv-Adapters parallel to each BasicBlock
src/losses.py              CAMMaskLoss (the containment ratio above)
src/data.py                ImageFolder + per-class pseudo-masks
src/train.py               training loop
src/evaluate.py            validation loop
src/cam_debug.py           end-of-run CAM overlays on the test split
src/reports/               diagnostics (observation only, never changes training)
```

## Run

```bash
python main.py            # one run
python sweep.py           # the shots x lambda grid
```

## Diagnostics

Every run writes `<run_dir>/diagnostics/`. Nothing in there changes training;
turn it all off with `diagnostics: false`.

| file | what it answers |
| --- | --- |
| `mask_audit.md` / `.json` | Are the masks reaching the loss at all? |
| `mask_examples/*.png` | Is `mask=1` the object or the background? |
| `cam_metrics.csv` | Every metric, per epoch, per split |
| `cam_diagnostics.png` | Nine panels covering loss, localisation, gradients |
| `snapshots/epoch_XXX.png` | The same validation images, every epoch |
| `report.md` | The above, read back as a list of findings |
| `report_errors.log` | Only appears if a diagnostic itself failed |

Plus, from the existing `cam_debug`:
`cam_debug/all_test_samples.csv` (per-sample test dump) and
`cam_debug/test_localization_summary.json`.

### Read the mask audit first

`CAMMaskLoss` is silent when it has nothing to work with. A sample whose mask
is missing, excluded by the manifest, or thresholded away gets `has_mask = 0`
and drops out, so a run can report a happily falling `cam_loss` while the mask
term never touched a single image. The audit catches, before training starts:

- **class-name mismatch** &mdash; folder names under `mask_root` must match the
  class folders under `data_dir` exactly. `sam_pseudo_masks_cat` vs `cat`
  silently disables the loss for every sample of that class.
- **pixel scale** &mdash; the dataset does `ToTensor()` (divide by 255) then keeps
  pixels `> 0.5`, so masks saved as 0/1 instead of 0/255 threshold to empty and
  are dropped without a word.
- **polarity** &mdash; if mask coverage is higher on the image border than in the
  centre, the masks probably mark the background, and the loss is pushing the
  CAM *off* the object.
- **the achievable range** &mdash; after area-downsampling to the 7x7 CAM grid, it
  reports the containment of a perfect CAM (`floor`) and of a flat one
  (`uniform`). Every `cam_loss` number should be read between those two.

### The metrics that matter

| metric | meaning |
| --- | --- |
| `containment` | the value `CAMMaskLoss` returns |
| `uniform_containment` | what a flat, uninformative CAM would score |
| `floor_containment` | what a perfect CAM would score; the loss cannot go below it |
| `relative_containment` | `containment / uniform`. **< 1** good, **~1** useless, **> 1** anti-localised |
| `mean_pos_outside` | mean `relu(CAM)` over background cells, in raw units. This is the number that must fall for "background near zero" |
| `positive_background_cell_frac` | share of background cells still above zero |
| `cam_all_negative` | share of samples whose CAM is negative everywhere &mdash; the degenerate solution |
| `total_positive_energy` | `sum(relu(CAM))`. Collapsing to 0 is that same degenerate solution |
| `pointing_hit` | how often the CAM peak lands on the object; chance level is the mask's coverage |
| `grad_*/cam_over_ce_ratio` | `\|lambda*grad(cam)\| / \|grad(ce)\|`, i.e. how much the term actually moves weights |
| `grad_*/cosine_ce_cam` | whether the two objectives pull the same way |

A low `containment` on its own means nothing. Check it against
`uniform_containment`, and check `cam_all_negative` before believing it.

### Diagnostics settings

```yaml
diagnostics: true             # master switch

mask_audit: true
mask_audit_max_decoded: 600   # masks opened per split; null = all
mask_audit_examples: 12

cam_stats: true               # per-epoch CAM statistics

grad_probe: true              # CE vs CAM gradient comparison
grad_probe_batches: 3

cam_snapshots: true
cam_snapshots_count: 8
cam_snapshots_every: 1
```

## Data layout

```text
data_dir/
  class_a/*.jpg
  class_b/*.jpg

mask_root/
  class_a/masks/<image_stem>.png          # 0/255, white = object
  class_a/manifest_class_a_masks.csv      # optional: has_mask, status columns
  class_b/masks/<image_stem>.png
  class_b/manifest_class_b_masks.csv
```

The mask filename stem must match the image stem, and the class folder names
under `mask_root` must match those under `data_dir`.
