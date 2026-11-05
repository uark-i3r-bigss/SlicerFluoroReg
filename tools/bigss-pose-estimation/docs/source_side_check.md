# Checking the source side for a new set-up

Read this before using predicted poses from a new C-arm, table set-up or site.

## Why it matters

The X-ray source can be on the patient's anterior side (`prior.alpha_deg` around 0, the config
default) or on the posterior side (around 180, e.g. `[135, 225]`). The pelvis is nearly left/right
symmetric, so a model trained on the **wrong** side still returns poses that look plausible, but
with the patient's left and right exchanged. Such a pose then scores on the wrong femur and flips
left/right landmark roles. A pose from a wrong-side model must never seed registration.

## The check

For every new set-up, before staging poses for xreg:

1. Train one model per side: the config as written, and a copy with the opposite
   `prior.alpha_deg` and separate `output.*` directories.
2. Run `scripts/pose.sh predict --no-refine --stage network --no-stage` with both models on the
   same projections.
3. Compare the median `gncc` in each run's `network/quality.json` and look at the
   `qc/network_overlays_*.png` contact sheets. Keep the side that clearly wins and record the
   decision and both run directories with the project.
4. Do not flip the images to "fix" the side. A left/right image flip approximates the other side
   only up to perspective, and it scored below the correct side in the check below.

A whole run with a low median `gncc` (well under 0.5 after refinement, on views that clearly show
the pelvis) is the signature of the wrong side, not of hard images.

## Known set-ups

| Set-up | Source side | Evidence |
|---|---|---|
| Siemens Cios Fusion, native 1536² projections, fused Loop-X CBCT (142 views) | **posterior** (alpha 135–225) | median network `gncc` 0.37 posterior vs 0.23 anterior (0.30 with flipped images); 0.91 vs 0.30 after refinement |
| Siemens Cios Fusion, 768² projections (13 views) | **anterior** (default alpha) | median `gncc` after refinement 0.64 anterior vs 0.21 posterior; network 0.20 vs 0.15 |

## Choosing views from `quality.json`

Every projection gets a pose. `quality.json` in the staged pose directory tiers each one
`confident`, `plausible` or `suspect`, with flags (see [outputs](outputs.md)). Select registration
views from it, and use `targets.<name>.coverage` to decide which targets a view can serve. Do not
assume that every pose file is usable, or that a `confident` tier is a measured error.

## Related limits

- The detector crop must have the same physical width across detectors: `crop_px` 200 at
  0.194 mm, 100 at 0.388 mm.
- Refinement cost is set in detector millimetres. On a 1536² Cios it takes about 3.5 s per view;
  the network pose takes about 40 ms plus a 33 ms H5 read. Each run records its timing in
  `timing.json`.
- Training needs about 20.3 GiB free on GPU 0 at batch 48 and refuses to start otherwise. Lower
  `model.batch_size` (44 fits in 19.7 GiB) instead of stopping other users' processes.
