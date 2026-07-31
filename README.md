# ResNet18 + CAM background suppression with SAM pseudo-masks

Train an image classifier whose class activation maps stay **on the object**, using
SAM segmentation masks as supervision for where the CAM should *not* be:

```text
total_loss = classification_loss + lambda_cam * cam_loss
```

The idea: if the model says "dog" because it saw grass, its CAM lights up outside
the SAM mask. Penalise that, and the model is pushed to look at the animal.

## Quick start

```bash
pip install -r requirements.txt
# edit configs/config.yaml -> data_dir, mask_root
python main.py
```

Expected layout:

```
data_dir/                       mask_root/
  cat/*.jpg                       cat/masks/*.png
  dog/*.jpg                       cat/manifest_cat_masks.csv
                                  dog/masks/*.png
                                  dog/manifest_dog_masks.csv
```

Mask filenames are `mask_prefix + <image stem> + mask_suffix + mask_extension`
(all configurable). If nothing resolves, training **stops with an error** rather
than silently running with `cam_loss == 0.0`.

## Choosing a CAM loss

`cam_loss_type` in `configs/config.yaml`:

| value | what it does |
|---|---|
| `containment` *(default)* | Fraction of the target CAM's positive energy that landed in the background. Bounded in `[0, 1]`, scale-free, pins nothing to a constant. |
| `normalized_mse` | The original MSE-to-0-outside / MSE-to-1-inside, but on a max-normalised CAM with a **detached** peak. |
| `legacy_mse` | The original implementation. Kept only to reproduce old runs — see below. |

`containment` has three terms, all in `[0, 1]` so `lambda_cam` keeps its meaning:

- `outside_weight` — background energy fraction (the main term).
- `coverage_weight` — stops the CAM collapsing onto one discriminative pixel.
- `nontarget_weight` — at every object pixel, the true class should win.

Two settings matter more than they look:

- `cam_ignore_band` (default 8 px) — a CAM cell's receptive field is far larger
  than its stride, so a cell just outside the object genuinely sees the object.
  The band makes that ring *ignored* rather than penalised.
- `cam_warmup_epochs` (default 3) — ramps `lambda_cam` in. At step 0 the CAM head
  is random, and full background suppression immediately rewards the all-zero map.

### Picking `lambda_cam`

Set `grad_probe_every: 50` and read `grad_norm_ratio` from `training_log.csv`. It
is `||grad CE|| / ||grad CAM||` at the CAM head; multiply `lambda_cam` by it to put
the two losses on equal footing. This is a much better guide than the ratio of the
loss *values*, which live on unrelated scales.

## What the model outputs

`ResNet18CAM.forward` returns `(logits, cams)` where

```python
cams   = cam_conv(features)        # raw, signed
logits = GAP(cams)                 # no ReLU in between
```

This is the standard CAM formulation: the logits are exactly a linear classifier on
the pooled features. **ReLU belongs in the loss and in the visualisation, not here**
— with a ReLU in front of the pooling, a class whose map goes fully negative is
clamped to 0 *and* receives zero gradient, so it can never be predicted again.

`dilate_last_block: true` converts layer4's stride to dilation, giving 14x14 CAMs
instead of 7x7 at 224px input for roughly 4x the cost of that one stage.

## Evaluating whether it worked

After training, `cam_debug` reports three numbers over the masked test split:

- **background energy fraction** — the headline metric. Scale-free, threshold-free,
  and exactly what the loss minimises.
- **pointing game accuracy** — does the hottest pixel land on the object?
- **IoU @ threshold** — kept for continuity. Read it with suspicion: min-max
  normalisation stretches *any* map to the full range, so even a flat CAM scores
  respectably.

Compare two checkpoints directly:

```bash
python -m src.compare_cams --baseline runs/A/best_resnet18_cam.pth \
                           --cam_model runs/B/best_resnet18_cam.pth
```

### The baseline worth running

Set `bg_randomize_prob: 0.3` to paste segmented objects onto random other
backgrounds. It attacks the same shortcut far more directly than CAM
regularisation, costs almost nothing, and if the CAM loss doesn't beat it that is
important to know.

## Sweeps

```bash
python sweep.py
```

Runs shots x lambda x **seed** and reports mean +/- std. The seeds are not
optional: at 1 image per class, re-running the same configuration moves accuracy
by several points, so a single-seed difference between two lambdas is noise.

## Other scripts

```bash
python -m src.cam_debug        --checkpoint runs/<ts>/best_resnet18_cam.pth
python -m src.debug_features   --checkpoint runs/<ts>/best_resnet18_cam.pth
python -m src.debug.debug_cam_loss.cam_loss_debugger --checkpoint runs/<ts>/best_resnet18_cam.pth
```

All default to the newest checkpoint under `runs/` if `--checkpoint` is omitted.

## Why `legacy_mse` is kept but not recommended

Measured with the repository's own classes:

1. It targets `cam == 1` inside the mask on **un-normalised** values. The logit is
   the spatial mean of the map, so that pins the target logit to roughly the
   foreground area fraction (~0.2) and caps softmax confidence near **60%** on a
   two-class problem, with cross-entropy stuck around 0.51.
2. The outside term is not scale-invariant: scaling the CAM down by 100x reduces it
   by 10000x. That free minimum teaches no localisation.
3. It downsamples the mask with `mode="nearest"`, which agrees with a correct
   area-based downsample only about **45%** of the time.

The default `containment` loss reaches 99.98% confidence at the same `lambda_cam`,
is exactly invariant to CAM scale, and computes the loss at the mask's own
resolution by upsampling the CAM instead of destroying the mask.
