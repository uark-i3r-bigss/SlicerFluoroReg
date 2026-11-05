# Synthetic sample case

A self-contained smoke test for the landmark pipeline: no TotalSegmentator, no GPU, no patient
data. `make_synthetic_case.py` writes a crude bone phantom (CT + per-anatomy masks) that
`process_medical_scans.py --skip-segmentation` can run end to end in seconds.

Use it to check that your environment, the STL stage, the landmark detectors and the ROI cropper
all work. It is **not** a validation of anatomical accuracy — see the warning at the bottom.

## Run it

```bash
conda run --prefix "$HOME/.local/share/fluororeg/annotation-env" python examples/make_synthetic_case.py --output-dir "$HOME/.cache/fluororeg/synthetic"
conda run --prefix "$HOME/.local/share/fluororeg/annotation-env" python process_medical_scans.py \
    $HOME/.cache/fluororeg/synthetic/synthetic_hip.nii.gz $HOME/.cache/fluororeg/synthetic \
    --skip-segmentation --no-confirm
```

The generator prints the exact second command (with absolute paths) when it finishes. Both steps
together take ~11 s on a workstation CPU (generator 3.6 s, pipeline 7.0 s) and use well under 1 GB
of RAM. Useful flags: `--output-dir`, `--case-id`, `--spacing` (default 1.5 mm), `--seed`, and
`--verify-landmarks` (runs the STL + landmark stages directly on the masks and prints the
femoral-head error without invoking the CLI).

A successful pipeline run *consumes* `segmentations/` (it is moved to
`TotalSegmentator_outputs/individual_seg/`), so re-run the generator before re-running the
pipeline, otherwise you get
`✗ --skip-segmentation needs existing masks in .../synthetic_hip/segmentations`.

## What the generator writes

```
$HOME/.cache/fluororeg/synthetic/
  synthetic_hip.nii.gz          int16 HU CT, 181 x 155 x 340 @ 1.5 mm (~11 MB)
  ground_truth.json             exact phantom geometry in world LPS mm
  synthetic_hip/
    segmentations/              11 uint8 masks: hip_left/right, femur_left/right,
                                vertebrae_L1..L5, vertebrae_S1, sacrum
```

## What the pipeline adds

```
$HOME/.cache/fluororeg/synthetic/synthetic_hip/
  meta_data/                    synthetic_hip_COR_CT.nii.gz, _CT_seg.nii.gz, anatomy_config, scan_analysis
  combined + full label maps -> TotalSegmentator_outputs/synthetic_hip_totalseg_full_labels.nii.gz
  TotalSegmentator_outputs/
    individual_seg/             the 11 masks, moved here from segmentations/
    models/                     11 per-anatomy STLs (label_25_sacrum … label_78_hip_right)
  models/                       10 STLs for the pipeline anatomy config, incl. label_1_hip_left_hip_right
  landmarks/                    synthetic_hip_landmarks.fcsv (43 fiducials), synthetic_hip_debug.json
  cropped_data/                 femur_left, femur_right, hip, lumbar, pelvis  (all 5 crop groups)
```

Everything above is ~80 MB, dominated by the STL meshes.

## Historical measured result

The earlier RANSAC-only implementation recorded femoral-head center errors of 0.32 mm on this
phantom, with 0.17–0.92 mm variation over five repeats. Those measurements are historical; they
are not measurements of the current implementation.

The current femoral-head estimator starts with an algebraic sphere fit and bounded,
iteratively trimmed least squares. RANSAC is used only to seed an outlier-rescue fit when
needed. Stochastic steps use the fixed-root, case-and-stage-derived generators in `rng.py`,
not an unseeded global generator. Repeated inputs therefore use the same random streams.
`PEL_FH_L`/`PEL_FH_R` copy the corresponding femoral-head centers. To measure the current fit,
run `--verify-landmarks` and compare with `ground_truth.json`; no new accuracy result is claimed
here.

## Landmarks produced

45 landmarks are detected (`pelvis=16, femur=8, lumbar=21`); 43 reach the FCSV, because
`PEL_IPR_L` and `PEL_IPR_R` are marked `"visible": false` in `configs/landmark_config.json`.

- Femur (8): `FEM_L_FH`, `FEM_L_FT_S`, `FEM_L_FT_T`, `FEM_L_LT`, `FEM_R_FH`, `FEM_R_FT_S`,
  `FEM_R_FT_T`, `FEM_R_LT`
- Pelvis (14 written): `PEL_ASIS_L/R`, `PEL_AIIS_L/R`, `PEL_PT_L/R`, `PEL_IT_L/R`,
  `PEL_PSIS_L/R`, `PEL_PS`, `PEL_COM`, `PEL_FH_L/R` — plus `PEL_IPR_L/R`, detected but suppressed
- Lumbar (21): `LUM_{L1..L5}_{COM,SP,TP_L,TP_R}` and `LUM_SAC_A`

If you change the phantom geometry, the pelvis detectors are the fragile ones: `PEL_PSIS_*` and
`PEL_AIIS_*` require the iliac knobs to sit in specific bands relative to the detected ASIS, and
losing them is a phantom problem, not a pipeline problem.

## Warning

The phantom is boxes, spheres, capsules and slabs. Landmarks land on it because the detectors are
geometric, **not** because the shapes are anatomically correct. A successful run here says the
pipeline executes and is numerically self-consistent; it says nothing about landmark accuracy on
real CT. Never cite these numbers as accuracy evidence — validate that on annotated clinical data.

No example scans, generated outputs or test files are included in this snapshot. Only the
synthetic generator and this guide are shipped; keep generated artifacts outside the extension.
