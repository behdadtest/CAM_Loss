# ResNet18 + Multi-Class CAM Loss

This project trains a ResNet18 classifier with:

```text
total_loss = classification_loss + lambda_cam * cam_mask_loss
```

`cam_mask_loss` uses SAM pseudo-masks to discourage the class activation map
from firing on the background.

## The CAM loss

Two are implemented; pick with `cam_loss_type`.

### `softmax` &mdash; `CAMSoftmaxMaskLoss` (default)

```text
p     = softmax(cam[label].flatten() / T)
loss  = sum(p * outside)
```

Reads the CAM as a distribution over *where* the evidence sits, and asks how
much of it lands on the background. Bounded in `[0, 1]`, gradient at every
cell, and invariant to the CAM's absolute level.

`d(loss)/d(z_j) = (p_j / T) * (o_j - loss)`: cells more background than the
current average are pushed down, cells more object than average are pushed up.

**What it does not do:** being level- *and* scale-invariant, it drives
background activation down *relative to* the object, never toward zero in
absolute units. If you want literal "CAM near zero in the background", add a
penalty on `relu(cam) * outside` on top &mdash; the diagnostics report
`mean_pos_outside` for exactly this reason.

### `ratio` &mdash; `CAMMaskLoss` (the original)

```text
positive     = relu(cam[label])
containment  = sum(positive * outside) / (sum(positive) + eps)
```

Kept so earlier runs reproduce. It has two failure modes that the softmax
variant removes, both measured by the diagnostics:

1. **Scale-invariance.** The model can lower it by inflating object activation
   rather than suppressing background.
2. **`relu` gates the gradient.** A cell already below zero contributes to
   neither sum and receives *exactly zero* gradient. A CAM that is negative
   *everywhere* therefore scores `0 / eps = 0` &mdash; a perfect loss from a map
   that localises nothing, and a minimum reachable just by lowering the map.
   Since the logits are `GAP(cam)` and cross-entropy only constrains
   differences between class maps, nothing else in the objective pushes back.

On a fixed synthetic run, identical apart from the loss, scored on the neutral
`softmax_containment` axis (flat-CAM baseline `0.869`):

| | `ratio` | `softmax` |
| --- | --- | --- |
| softmax containment, train | 0.829 (0.95x flat) | **0.234** (0.27x flat) |
| CAMs negative everywhere | 100% | **0%** |
| pointing game | 0.47 | **1.00** |
| signed object &minus; background | +0.33 | **+5.11** |
| CAM gradient, last epoch | **0.000** | 0.415 |

`softmax_containment` is recorded on every run whichever loss is training, so
the two are directly comparable.

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
tests/                     property checks on the CAM losses
```

## Run

```bash
python main.py                            # one run
python sweep.py                           # the shots x lambda grid
python tests/test_cam_softmax_loss.py     # property checks on the CAM losses
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
| `softmax_containment` | the value `CAMSoftmaxMaskLoss` returns. **Recorded on every run whichever loss trains, and it cannot be zeroed by lowering the map &mdash; the fair axis for comparing runs** |
| `softmax_entropy_norm` | spread of `p`, in [0,1]. `1` = uniform (says nothing), near `0` = one-hot (confident, but starves other cells of gradient since the gradient scales with `p_j`). Tune with `cam_softmax_temperature` |
| `containment` | the value `CAMMaskLoss` returns |
| `uniform_containment` | what a flat, uninformative CAM would score. **Both losses share this baseline** |
| `floor_containment` | what a perfect CAM would score; neither loss can go below it |
| `relative_containment` | `containment / uniform`. **< 1** good, **~1** useless, **> 1** anti-localised |
| `*_active` | the same, over samples that still have a positive CAM. The plain mean is dragged to 0 by collapsed samples and flatters a dead run |
| `cam_inside_minus_outside` | mean **signed** CAM on the object minus on the background. Must be positive. The only metric that catches a CAM inverted in absolute terms, because containment never looks below zero |
| `mean_pos_outside` | mean `relu(CAM)` over background cells, in raw units. This is the number that must fall for "background near zero" |
| `positive_background_cell_frac` | share of background cells still above zero |
| `cam_all_negative` | share of samples whose CAM is negative everywhere &mdash; the degenerate solution |
| `total_positive_energy` | `sum(relu(CAM))`. Collapsing to 0 is that same degenerate solution |
| `pointing_hit` | how often the CAM peak lands on the object; chance level is the mask's coverage |
| `grad_*/cam_over_ce_ratio` | `\|lambda*grad(cam)\| / \|grad(ce)\|`, i.e. how much the term actually moves weights |
| `grad_*/cosine_ce_cam` | whether the two objectives pull the same way |

A low `containment` on its own means nothing. Check it against
`uniform_containment`, and check `cam_all_negative` before believing it.

The findings in `report.md` adapt to `cam_loss_type`: an all-negative CAM is
CRITICAL under `ratio` (it disarms the loss) but only INFO under `softmax`
(the loss is level-invariant, though `cam_debug`'s min-max overlays will still
manufacture a hot region out of such a map).

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
