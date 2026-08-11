"""
Property checks for CAMSoftmaxMaskLoss against the claims made for it, and
against the specific failure of CAMMaskLoss it exists to remove.

Run from anywhere:  python tests/test_cam_softmax_loss.py
Exits non-zero on the first broken property.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import CAMMaskLoss, CAMSoftmaxMaskLoss, build_cam_loss  # noqa: E402

torch.manual_seed(0)
B, K, H, W = 6, 4, 7, 7

cams = torch.randn(B, K, H, W, requires_grad=True)
labels = torch.randint(0, K, (B,))

# A coherent central blob, not noise: area-downsampling random noise to 7x7
# gives a near-uniform mask, which makes the localisation checks vacuous.
mask = torch.zeros(B, 1, 224, 224)
for i in range(B):
    top, left = 40 + 10 * i, 30 + 12 * i
    mask[i, 0, top:top + 96, left:left + 96] = 1.0
has_mask = torch.ones(B)

soft = CAMSoftmaxMaskLoss(temperature=1.0)
ratio = CAMMaskLoss()

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}   {detail}")
    if not cond:
        ok = False


# 1. bounded in [0, 1]
v = soft(cams=cams, mask=mask, labels=labels, has_mask=has_mask)
check("bounded [0,1]", 0.0 <= v.item() <= 1.0, f"loss={v.item():.4f}")

# 2. LEVEL INVARIANCE -- the whole point. Shift the CAM far negative.
shifted = cams.detach() - 50.0
v_shift = soft(cams=shifted, mask=mask, labels=labels, has_mask=has_mask)
r_now = ratio(cams=cams.detach(), mask=mask, labels=labels, has_mask=has_mask)
r_shift = ratio(cams=shifted, mask=mask, labels=labels, has_mask=has_mask)
check("softmax level-invariant", abs(v_shift.item() - v.item()) < 1e-5,
      f"{v.item():.6f} -> {v_shift.item():.6f}")
check("ratio collapses to 0 when shifted (the bug)", r_shift.item() < 1e-6,
      f"{r_now.item():.6f} -> {r_shift.item():.6f}")

# 3. GRADIENT SURVIVES an all-negative CAM (ratio's does not)
neg = (cams.detach() - 50.0).requires_grad_(True)
soft(cams=neg, mask=mask, labels=labels, has_mask=has_mask).backward()
g_soft = neg.grad.abs().sum().item()

neg2 = (cams.detach() - 50.0).requires_grad_(True)
ratio(cams=neg2, mask=mask, labels=labels, has_mask=has_mask).backward()
g_ratio = neg2.grad.abs().sum().item()
check("softmax keeps gradient on all-negative CAM", g_soft > 1e-6, f"|g|={g_soft:.4e}")
check("ratio loses gradient on all-negative CAM", g_ratio < 1e-12, f"|g|={g_ratio:.4e}")

# 4. BASELINES match what the diagnostics report.
#    uniform p (T -> inf) == mean(outside); p on the most-inside cell == min(outside)
m = F.interpolate(mask, size=(H, W), mode="area")
outside = (1.0 - m).reshape(B, -1)
flat = CAMSoftmaxMaskLoss(temperature=1e6)(
    cams=cams.detach(), mask=mask, labels=labels, has_mask=has_mask)
check("T->inf equals flat-CAM baseline mean(outside)",
      abs(flat.item() - outside.mean(dim=1).mean().item()) < 1e-4,
      f"{flat.item():.6f} vs {outside.mean(dim=1).mean().item():.6f}")

best = torch.full((B, K, H, W), -10.0)
rows = torch.arange(B)
peak = outside.argmin(dim=1)
best[rows, labels] = best[rows, labels].reshape(B, -1).index_put(
    (rows[:, None], peak[:, None]), torch.full((B, 1), 10.0)).reshape(B, H, W)
floor = CAMSoftmaxMaskLoss(temperature=0.05)(
    cams=best, mask=mask, labels=labels, has_mask=has_mask)
check("T->0 on the best cell equals the floor min(outside)",
      abs(floor.item() - outside.min(dim=1).values.mean().item()) < 1e-4,
      f"{floor.item():.6f} vs {outside.min(dim=1).values.mean().item():.6f}")

# 5. it prefers object-concentrated CAMs over background-concentrated ones
inside_flat = 1.0 - outside
on_object = torch.full((B, K, H, W), -10.0)
on_object[rows, labels] = (inside_flat * 20.0 - 10.0).reshape(B, H, W)
on_bg = torch.full((B, K, H, W), -10.0)
on_bg[rows, labels] = (outside * 20.0 - 10.0).reshape(B, H, W)
lo = soft(cams=on_object, mask=mask, labels=labels, has_mask=has_mask).item()
hi = soft(cams=on_bg, mask=mask, labels=labels, has_mask=has_mask).item()
check("object-concentrated scores lower than background-concentrated",
      lo < hi, f"object={lo:.4f} background={hi:.4f}")

# 6. gradient direction. d(loss)/d(z_j) = (p_j/T) * (o_j - containment), so the
#    sign flips at the CURRENT containment, not at outside=0.5: a cell is pushed
#    up exactly when it is less background than the map's present average.
z = torch.zeros(B, K, H, W, requires_grad=True)
soft(cams=z, mask=mask, labels=labels, has_mask=has_mask).backward()
g = z.grad[rows, labels].reshape(B, -1)
per_sample = (torch.full_like(outside, 1.0 / (H * W)) * outside).sum(dim=1, keepdim=True)
more_bg = outside > per_sample + 1e-6
more_fg = outside < per_sample - 1e-6
check("cells above the containment average are pushed DOWN",
      (g[more_bg] > 0).all().item(), f"n={int(more_bg.sum())}")
check("cells below the containment average are pushed UP",
      (g[more_fg] < 0).all().item(), f"n={int(more_fg.sum())}")
# and the analytic form itself
expected = (1.0 / (H * W)) * (outside - per_sample) / B
check("gradient matches (p_j/T)*(o_j - containment)",
      torch.allclose(g, expected, atol=1e-7), f"max err={(g - expected).abs().max():.2e}")

# 7. has_mask gating and empty batch
z2 = torch.zeros(B, K, H, W, requires_grad=True)
none_ = soft(cams=z2, mask=mask, labels=labels, has_mask=torch.zeros(B))
check("no-mask batch returns 0", none_.item() == 0.0)
half = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
per = [soft(cams=cams.detach()[i:i + 1], mask=mask[i:i + 1], labels=labels[i:i + 1],
            has_mask=torch.ones(1)).item() for i in range(3)]
got = soft(cams=cams.detach(), mask=mask, labels=labels, has_mask=half).item()
check("averages only over masked samples",
      abs(got - sum(per) / 3) < 1e-4, f"{got:.6f} vs {sum(per)/3:.6f}")

# 8. factory
check("factory: softmax", isinstance(
    build_cam_loss({"cam_loss_type": "softmax", "cam_softmax_temperature": 2.0}),
    CAMSoftmaxMaskLoss))
check("factory: temperature honoured", build_cam_loss(
    {"cam_loss_type": "softmax", "cam_softmax_temperature": 2.0}).temperature == 2.0)
check("factory: default stays 'ratio'", isinstance(build_cam_loss({}), CAMMaskLoss))
try:
    build_cam_loss({"cam_loss_type": "nope"})
    check("factory: rejects unknown", False)
except ValueError:
    check("factory: rejects unknown", True)
try:
    CAMSoftmaxMaskLoss(temperature=0.0)
    check("rejects T<=0", False)
except ValueError:
    check("rejects T<=0", True)

# 9. non-square / different grid + fp16 path
c16 = torch.randn(2, 3, 5, 9).half()
v16 = CAMSoftmaxMaskLoss()(cams=c16, mask=torch.rand(2, 1, 64, 64).round(),
                           labels=torch.tensor([0, 2]), has_mask=torch.ones(2))
check("works on a non-square grid in fp16", torch.isfinite(v16).item(), f"loss={v16.item():.4f}")

print("\nALL PASS" if ok else "\nFAILURES ABOVE")
raise SystemExit(0 if ok else 1)
