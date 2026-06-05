# Pipeline outputs — what each file is and how to read it

Every run of `python demo.py --case <CASE>` (or the notebook
`SEGA_aorta_inference_mesh_tutorial.ipynb`) writes its artifacts to
`outputs/<CASE>/`. This document explains each file, the units/ranges of the
values it holds, and how to interpret it.

Two families of artifacts are produced:

1. **Segmentation** — the predicted aorta lumen mask, surface meshes, and
   2D/3D overlays comparing prediction vs. ground truth.
2. **Explainability** (on by default; skipped with `--no-explain`) — two
   complementary post-hoc views of the model:
   - **MC Dropout uncertainty** — *where the model is unsure.*
   - **Seg-Grad-CAM saliency** — *which image features drove the prediction.*

Coordinate conventions and geometry: all NIfTI outputs share the exact geometry
(size, spacing, direction, origin) of the source CT, so they overlay the
original `data/volumes_full/<CASE>.nii.gz` and each other voxel-for-voxel.

---

## Quick reference

| File (`<CASE>` = case id, e.g. `K18`) | Type | Meaning |
|---|---|---|
| `<CASE>_predicted_aorta_lumen.nii.gz` | NIfTI mask (uint8) | Predicted aorta lumen (1 = aorta, 0 = background) |
| `<CASE>_aortic_vessel_tree_smoothed.obj` | Mesh | Smoothed surface of the **prediction** |
| `<CASE>_aortic_vessel_tree_volume_mesh.obj` | Mesh | Un-smoothed (repaired) surface of the prediction |
| `<CASE>_ground_truth_aortic_vessel_tree_smoothed.obj` | Mesh | Smoothed surface of the **ground truth** |
| `<CASE>_ct_gt_pred_overlay_best_slice.png` | PNG | One axial slice: CT + GT/Pred overlay |
| `<CASE>_ct_gt_pred_overlay_video.mp4` (or `.gif`) | Video | The same overlay swept through all labelled slices |
| `<CASE>_uncertainty_mean_prob.nii.gz` | NIfTI (float, 0–1) | Mean foreground probability across MC Dropout passes |
| `<CASE>_uncertainty_entropy.nii.gz` | NIfTI (float, 0–ln2) | Predictive **entropy** — the headline uncertainty map |
| `<CASE>_uncertainty_std.nii.gz` | NIfTI (float) | Std of foreground probability across passes |
| `<CASE>_uncertainty_overlay_best_slice.png` | PNG | Entropy heatmap on the best CT slice |
| `<CASE>_uncertainty_overlay_video.mp4` (or `.gif`) | Video | Entropy heatmap swept through the labelled slices |
| `<CASE>_seggradcam.nii.gz` | NIfTI (float, 0–1) | Seg-Grad-CAM saliency volume |
| `<CASE>_seggradcam_overlay_best_slice.png` | PNG | Saliency heatmap on the best CT slice |
| `<CASE>_seggradcam_overlay_video.mp4` (or `.gif`) | Video | Saliency heatmap swept through the labelled slices |
| `<CASE>_gt_pred_3d_comparison.html` | Interactive HTML | 3D meshes: GT, Pred, and the predicted surface colored by entropy & saliency |

---

## 1. Segmentation outputs

### `<CASE>_predicted_aorta_lumen.nii.gz`
The model's prediction: a binary mask where `1` = aorta lumen, `0` = background,
in the original CT geometry. When explainability is on, this mask is the
`argmax` of the **mean** MC Dropout probability (i.e. it is slightly more stable
than a single deterministic pass); with `--no-explain` it is a single
deterministic forward pass. The console prints the **Dice** score against the
ground-truth label as a one-number accuracy summary (1.0 = perfect overlap).

### Meshes (`.obj`)
Surface meshes are extracted from the masks with marching cubes and live in the
CT's physical (millimetre) coordinate space, so they align with the CT and with
each other. Open them in any 3D viewer (MeshLab, Blender, ParaView, …).

- `*_aortic_vessel_tree_smoothed.obj` — prediction surface after volume-preserving
  Laplacian smoothing. **Use this one for visualization.**
- `*_aortic_vessel_tree_volume_mesh.obj` — prediction surface before smoothing
  (marching-cubes + mesh repair only); blockier, closer to the raw voxels.
- `*_ground_truth_aortic_vessel_tree_smoothed.obj` — the ground-truth surface,
  smoothed the same way, for side-by-side comparison.

### 2D overlay PNG + video (`*_ct_gt_pred_overlay_*`)
The CT with the GT and prediction masks overlaid. The PNG shows the single
"best" axial slice (the slice with the most labelled voxels); the video sweeps
through every slice that contains labels. Color key:

- **Green** = ground truth only (model missed it — a *false negative*)
- **Magenta** = prediction only (model over-segmented — a *false positive*)
- **Yellow** = GT ∩ Prediction (correct overlap — *true positive*)

A good segmentation is mostly yellow with thin green/magenta fringes at the
boundary.

---

## 2. Explainability outputs

These answer two different questions and should be read together.

### 2a. MC Dropout uncertainty — *"where is the model unsure?"*

The network is run **N times** (default `--mc-passes 20`) with its dropout layer
kept active, so each pass is a slightly different model. Aggregating the passes
turns a single prediction into a distribution, from which we derive:

- **`*_uncertainty_mean_prob.nii.gz`** — the per-voxel foreground probability
  averaged over the passes (0 = confident background, 1 = confident aorta). The
  binary prediction above is just this map thresholded at 0.5.

- **`*_uncertainty_entropy.nii.gz`** — the **predictive entropy**
  `H = -[p·log p + (1-p)·log(1-p)]`, the headline uncertainty map. It is bounded
  in **`[0, ln 2] ≈ [0, 0.693]`**: `0` where the model is certain (p≈0 or p≈1)
  and maximal (`ln 2`) where it is maximally torn (p≈0.5). For a well-behaved
  segmentation, entropy **concentrates in a thin shell at the mask boundary** —
  the interior of the aorta and the distant background are confident; only the
  exact location of the wall is uncertain. The demo prints a sanity line
  comparing mean entropy at the boundary vs. distant background (boundary should
  be higher).

- **`*_uncertainty_std.nii.gz`** — the per-voxel standard deviation of the
  foreground probability across passes. A second, closely-related view of the
  same uncertainty (it also peaks at the boundary). Kept as a secondary signal;
  entropy is the one to look at first.

The **`*_uncertainty_overlay_*`** PNG/video render the entropy as a *magma*
colormap on the CT (dark = certain, bright = uncertain), with a cyan contour
marking the predicted mask edge. Near-zero entropy is left transparent so the CT
shows through.

> **How to use it clinically/practically:** bright entropy that forms a clean,
> thin outline around the vessel is normal and reassuring. Bright entropy that
> spreads into *blobs* or appears *inside* the lumen or far outside it flags
> regions where the prediction is unreliable and may deserve manual review.

### 2b. Seg-Grad-CAM saliency — *"what made the model predict aorta here?"*

`*_seggradcam.nii.gz` is a **saliency volume in `[0, 1]`** (gradient-based). It
highlights the image regions whose features most increased the model's aorta
score. It is computed by back-propagating the **summed foreground logit over the
predicted mask** to the last decoder layer (`up_layers[-1]`), weighting that
layer's feature channels by their average gradient, ReLU-ing, and
**max-normalizing to `[0, 1]`** (1 = most influential, 0 = no positive
contribution). This is the segmentation-aware variant of Grad-CAM (Vinogradova
et al., 2020).

The **`*_seggradcam_overlay_*`** PNG/video render it as an *inferno* colormap on
the CT, again with the predicted-mask contour. You expect the saliency to sit
**on and around the predicted aorta** — the demo prints a sanity line confirming
mean saliency inside the mask exceeds mean saliency outside it.

> **Saliency vs. uncertainty are complementary, not the same.** Saliency says
> *"this is what the model looked at"*; uncertainty says *"this is where the
> model might be wrong."* High saliency + low uncertainty = a confident,
> well-supported prediction. High uncertainty regardless of saliency = treat
> with caution.

---

## 3. The interactive 3D HTML (`<CASE>_gt_pred_3d_comparison.html`)

Open in any web browser (needs internet — Plotly.js loads from a CDN). It is a
2×2 grid of independently-rotatable 3D scenes:

| | |
|---|---|
| **Ground truth** (solid red) | **Prediction** (solid red) |
| **Prediction · entropy** (magma) | **Prediction · Seg-Grad-CAM** (inferno) |

- **Top row** — the GT and predicted aorta surfaces, both red, for shape
  comparison.
- **Bottom row** — the *same predicted surface*, colored per-vertex by the
  entropy and saliency volumes. This projects the volumetric maps onto the
  vessel wall so you can see *where on the aorta* the model is unsure / what it
  attended to. Each panel has its own colorbar (entropy in nats `0–ln 2`,
  saliency `0–1`).

Each vertex is colored by the **maximum value in a 3×3×3 voxel neighborhood**
around it (`sample_radius=1`). This is deliberate: the entropy shell is only
~1 voxel thick and sits exactly on the surface, so a plain nearest-voxel lookup
would mostly read the confident interior and render the surface black. The
neighborhood-max lets the thin uncertainty shell register on the wall.

---

## 4. Theory behind the methods

This section explains *why* the methods work, for readers who want the model
behind the maps. It is not needed to use the outputs.

### 4.0 The segmentation backbone (context)

The predictor is **SegResNet** (Myronenko, 2018), an encoder–decoder CNN with
residual blocks and GroupNorm. The encoder downsamples the CT three times while
growing the channel count; the decoder upsamples back to full resolution with
skip connections, ending in a `1×1×1` convolution that emits **two logits per
voxel** (background, aorta). A softmax turns those into a per-voxel foreground
probability `p = softmax(logits)[aorta]`.

Because a whole CT volume does not fit in GPU memory, inference uses
**sliding-window inference**: the network is run on overlapping `160³` patches
(`overlap=0.25`) and the patch predictions are blended back together. The input
is first windowed in Hounsfield units, resampled to `1×1×1.5 mm`, oriented to
RAS, and foreground-cropped; an inverse transform (`Invertd`) maps every output
back into the original CT geometry so all NIfTIs line up.

The two explainability methods below are **post-hoc**: they wrap this fixed,
already-trained network and require no retraining.

### 4.1 MC Dropout — uncertainty as approximate Bayesian inference

A standard network gives a single answer with no notion of confidence. Bayesian
deep learning instead treats the weights `W` as random and seeks the posterior
`p(W | D)` given the training data `D`, then the **predictive distribution**

```
p(y | x, D) = ∫ p(y | x, W) · p(W | D) dW.
```

This integral is intractable for a deep net. **Gal & Ghahramani (2016)** showed
that a network trained with **dropout** is, mathematically, a *variational
approximation* to this Bayesian model: the dropout sampling distribution `q(W)`
approximates the true posterior `p(W | D)`. The practical consequence is
**Monte-Carlo (MC) Dropout** — instead of switching dropout *off* at test time
(the usual deterministic prediction), you keep it *on* and run `T` stochastic
forward passes, which Monte-Carlo–estimates the integral:

```
p(y | x, D) ≈ (1/T) Σ_t  p(y | x, Ŵ_t),     Ŵ_t ~ q(W).
```

For our binary problem each pass `t` yields a per-voxel foreground probability
`p_t`. We aggregate them into:

- **Predictive mean** `p̄ = (1/T) Σ_t p_t` → `*_uncertainty_mean_prob.nii.gz`.
  The binary mask is `p̄ > 0.5`.
- **Predictive entropy** of the mean, the *total* uncertainty:

  ```
  H[p̄] = -( p̄·ln p̄ + (1-p̄)·ln(1-p̄) ),     H ∈ [0, ln 2].
  ```

  It is `0` when the model is certain (`p̄≈0` or `p̄≈1`) and maximal (`ln 2`) at
  `p̄ = 0.5`, the decision boundary. → `*_uncertainty_entropy.nii.gz`.
- **Sample standard deviation** `σ = sqrt( (1/T) Σ_t (p_t - p̄)² )`, the spread
  of the passes → `*_uncertainty_std.nii.gz`.

**What the two numbers mean.** Total predictive uncertainty decomposes into
*aleatoric* (irreducible noise/ambiguity in the data) and *epistemic* (model
uncertainty, reducible with more data). The **entropy `H[p̄]`** captures the
total; the **disagreement between passes** (`σ`, or formally the mutual
information `I = H[p̄] − (1/T)Σ_t H[p_t]`, the "BALD" score) isolates the
*epistemic* part. We report entropy as the headline and `σ` as a complementary
spread; both peak at the boundary, which is why entropy forms a thin shell on
the vessel wall.

**Caveat specific to this model.** MC Dropout's approximation is only as good as
the dropout structure used in training. This SegResNet has a *single* dropout
layer (after the initial convolution), so the sampled randomness is limited; the
resulting uncertainty is a useful **qualitative** signal, not a calibrated
probability of error.

### 4.2 Seg-Grad-CAM — saliency from gradients

**Grad-CAM** (Selvaraju et al., 2017) explains a CNN decision by asking which
feature channels of a chosen convolutional layer most increased the target
score. Let `A^k` be the feature maps (channel `k`) of the target layer and `y`
the scalar score being explained. Grad-CAM:

1. Back-propagates `y` to get gradients `∂y / ∂A^k`.
2. **Global-average-pools** those gradients over space into a per-channel
   importance weight

   ```
   α^k = (1/Z) Σ_voxels  ∂y / ∂A^k(voxel).
   ```

3. Forms a weighted sum of the channels and keeps only positive contributions:

   ```
   L = ReLU( Σ_k  α^k · A^k ).
   ```

The `ReLU` is essential: it discards features that *suppress* the target so the
map shows only evidence *for* it. Intuitively `α^k A^k` is a first-order
(Taylor) estimate of channel `k`'s contribution, so `L` is large where
class-supporting features are active. `L` is then upsampled to the input size.

**The segmentation twist.** In classification `y` is a single class logit. In
dense segmentation there is one logit *per voxel*, so there is no single `y`.
**Seg-Grad-CAM** (Vinogradova et al., 2020) defines the score as the logit of
the class of interest **summed over a chosen set of voxels `M`**:

```
y = Σ_{voxel ∈ M}  logit_aorta(voxel).
```

Here `M` is the **predicted foreground mask** and the class is the aorta, so the
saliency answers *"what features made the network segment the aorta where it
did?"* We use the **sum** over `M` (rather than the mean): the mean divides the
gradient by the ~thousands of mask voxels, shrinking it toward floating-point
zero; the sum keeps it at a usable magnitude. Because the final map is
**max-normalized** to `[0, 1]`, sum vs. mean only rescales — it does not change
the relative pattern. (Using the mean was in fact the original cause of the
all-zero CAM that this pipeline previously produced.)

**Layer choice (`up_layers[-1]`).** Grad-CAM is a trade-off between semantics
and spatial resolution. Deep encoder layers carry abstract meaning but are
spatially coarse (here `1/8` resolution after three downsamples); the final
`1×1×1` conv has only 2 channels — too few for meaningful channel weighting.
The **last decoder block** sits at full input resolution with 8 feature channels:
semantic enough to be informative, fine enough to localize on the vessel.

**Why a single full-volume forward, not sliding-window.** Sliding-window
inference stitches independent patches and discards their intermediate
activations, so there is no single computational graph to back-propagate
through. Seg-Grad-CAM therefore runs one whole-volume forward+backward (padding
the volume to a multiple of 8 for the three up/down-sampling levels), falling
back to a centered `160³` patch only if the volume does not fit in GPU memory.

### 4.3 Why both, together

The two methods are deliberately complementary:

- **Seg-Grad-CAM** is a property of the **deterministic** network — it tells you
  *what the model looked at* to make its (single) prediction.
- **MC Dropout** is a property of the **distribution** of networks — it tells you
  *how much the prediction would wobble* under the model's own uncertainty.

Saliency without uncertainty can be confidently wrong; uncertainty without
saliency tells you *that* something is shaky but not *why*. Reading them side by
side — which the 3D HTML is built to support — gives both the "what" and the
"how reliable."

---

## Notes & caveats

- **Explainability is post-hoc and approximate.** Neither MC Dropout nor
  Grad-CAM is a calibrated probability of error; they are diagnostic lenses, not
  ground truth about the model's correctness.
- **More passes = smoother uncertainty.** `--mc-passes` defaults to 20; raising
  it reduces noise in entropy/std at the cost of runtime, lowering it (e.g. 5)
  is fine for a quick look.
- **Grad-CAM target layer** is `up_layers.-1` by default (last full-resolution
  decoder block). You can point it elsewhere with `--gradcam-layer` (e.g.
  `up_layers.0` for coarser, more abstract features).
- **`--no-explain`** skips section 2 and 3's overlays entirely and produces only
  the segmentation outputs (faster; the 3D HTML falls back to the plain GT-vs-Pred
  two-panel view).
- All maps are written with `float32` precision except the binary prediction
  mask (`uint8`).
