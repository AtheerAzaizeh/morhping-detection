# Part 1 — Face Morph Generation: Short Report

*Notebook: `notebooks/Part1_Face_Morphing.ipynb` · Deliverables: `results/morph_pairs/` (6 pairs of originals + morphs), `results/part1_showcase_pairs.png`, `results/part1_morph_sequence.png`, and the labelled dataset `data/dataset/` used by Part 2.*

## 1. Method used

We implemented a classical **landmark-based morphing pipeline** from scratch with **OpenCV**, **MediaPipe** and **SciPy** (no external morphing software):

1. **Data acquisition** — face photographs are downloaded from public GitHub-hosted sources (dlib example images, `face_recognition` examples, OpenCV sample data). Group photos are used deliberately, since each contributes several different individuals.
2. **Face detection & cropping** — MediaPipe **BlazeFace** locates faces; each is cropped as a square patch (1.9× the detection box, mirror-padded at borders) and resized to **512×512**. The large collage image is additionally scanned in overlapping 2×-upscaled tiles so small faces are found. Every crop is validated by requiring the 478-point landmarker to succeed on it, giving a pool of **50 face crops**.
3. **Landmark detection** — the MediaPipe **Face Landmarker** produces **478 landmarks** per face.
4. **Warping + blending** — Delaunay triangulation on the interpolated shape, piece-wise affine warping of both faces onto it, then pixel-wise cross-dissolve (details below).

Six showcase pairs (α = 0.5) and 170 dataset morphs (α ∈ {0.3…0.7}) were generated fully automatically.

## 2. Landmark alignment strategy

Alignment relies on the **fixed topology of the MediaPipe face mesh**: landmark *i* always refers to the same anatomical point (jawline, brows, eyes, iris, nose, lips) on every face, so the two landmark sets are in dense point-to-point correspondence *by construction* — no manual annotation and no rigid pre-registration is needed. Two refinements matter in practice:

- **Boundary points.** Eight frame points (4 corners + 4 edge midpoints) are appended to the 478 landmarks so the Delaunay triangulation tiles the *entire* image; hair and background are then warped consistently instead of leaving holes.
- **Triangulate the mid-way shape.** The triangulation is computed once on the *interpolated* landmark set, not on either source face. Transplanting one face's triangulation onto a very different geometry can produce flipped or degenerate triangles; triangulating the target shape avoids this.

## 3. Interpolation technique

Two independent linear interpolations are controlled by the same blend factor α:

- **Shape (geometry) interpolation:** `P_M = (1 − α)·P_A + α·P_B` for each of the 486 correspondence points. For every Delaunay triangle, the affine map from its vertices in A (resp. B) to its vertices in M is estimated with `cv2.getAffineTransform` and applied with `cv2.warpAffine` (bilinear, mirror border), assembling warped images `W_A`, `W_B` triangle by triangle.
- **Appearance (colour) interpolation — cross-dissolve:** `M = (1 − α)·W_A + α·W_B`, computed per pixel **after** both images are geometry-aligned, which is what prevents the double-exposure look of naive averaging.

Showcase morphs use α = 0.5 (the classical morphing-attack setting where one photo should match both contributing subjects); dataset morphs draw α from {0.3, 0.4, 0.5, 0.6, 0.7} for variety.

## 4. Observations and limitations

**Observations**

- Morphs of frontal, similarly-lit faces are highly convincing; identity shifts smoothly along the α-sequence (`results/part1_morph_sequence.png`).
- Characteristic artifacts appear exactly where the landmark mesh gives no constraint: **ghosting** around hairlines, ears, glasses, and clothing edges; slight global blur from bilinear resampling plus averaging. These high-frequency artifacts are precisely the cues the Part 2 detector learns to exploit.
- Larger pose differences between donors produce visible double contours; pairs with similar pose and skin tone give the cleanest morphs.

**Limitations**

- Single 2-D triangulated mesh: no occlusion handling, no 3-D head-pose correction.
- No colour rebalancing between donors and no Poisson/seamless cloning or GAN-based post-processing, so background seams remain visible in hard pairs.
- Landmark quality degrades on strongly non-frontal or low-resolution faces, limiting usable source pairs.
- The face pool is small (50 crops from 15 photos) and celebrity-biased; a production study would use a controlled passport-style dataset.
