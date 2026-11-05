# Pose-estimation config, schema version 1

One file per CT case at `PROJECT/configs/pose_estimation/<case>.json`; every command of `scripts/pose.sh`
reads it. Relative paths resolve against the **project root**, the nearest ancestor of the config holding
`project_manifest.json`. Unknown keys are an error.

```json
{
  "schema_version": 1,
  "case_id": "case01",
  "volume":      { "ct": "meta_data/volumes/case01_ct.nii.gz",
                   "seg": "meta_data/volumes/case01_seg.nii.gz" },
  "projections": { "dir": "meta_data/XrayProjData", "pattern": "{id}.h5", "ids": ["001", "002"] },
  "detector":    { "reference_id": "001", "crop_px": 100, "intensity": "raw" },
  "prior":       { "center_label": 1, "alpha_deg": [-45, 45], "beta_deg": [-45, 45],
                   "gamma_deg": [-15, 15], "lateral_mm": 150, "axial_mm": 150,
                   "source_to_center_mm": [450, 1000] },
  "model":       { "method": "finetune", "iterations": 1200, "batch_size": 48,
                   "learning_rate": 0.001, "warmup_iterations": 10, "grad_accum": 1,
                   "seed": 20260924, "self_check_samples": 100, "self_check_min_capture": 0.9 },
  "predict":     { "refine": false, "refine_labels": [1], "stage": "network",
                   "qc_targets": { "pelvis": 1, "femur_right": 3 } },
  "output":      { "models_dir": "meta_data/pose_models/case01",
                   "poses_dir": "meta_data/initial_poses/case01",
                   "runs_dir": "meta_data/tmp/pose_estimation/case01" }
}
```

Only `schema_version`, `case_id`, `volume` and `projections.dir` are required; the values above
are the defaults otherwise, except `predict.qc_targets` (default `{"target": prior.center_label}`).

| key | meaning |
|---|---|
| `volume.ct`, `volume.seg` | The staged CT (HU) and its label map on the same grid. |
| `projections` | xreg projection H5s, same `dir`/`pattern`/`ids` meaning as xreg. Without `ids`, every file matching `pattern` is used, sorted. Each H5 must hold one projection with an `origin-at-focal-pt-det-neg-z` camera whose principal point is the detector centre (native-grid conversion). |
| `detector.reference_id` | Projection whose camera defines the training detector (default: first id). All projections should share SDD and spacing; a SDD mismatch above 1 % is warned about. |
| `detector.crop_px` | Total pixels removed per dimension before the model sees an image (half per border). Keep it the same physical width across detectors: 100 at 0.388 mm (768² detector, xreg's 50-px runtime crop), 200 at 0.194 mm (native 1536² Cios), both ≈ 39 mm and a 2.02 mm model pixel. |
| `detector.intensity` | `raw`: stored pixels are transmission (bone dark), log-linearised as `xvr register` does. `log`: already attenuation (e.g. DRRs); used as is. |
| `prior.center_label` | Label whose centroid the sampled central rays aim near; also the self-check's reference point. |
| `prior.alpha_deg`, `beta_deg`, `gamma_deg` | xvr ZXY Euler ranges in DiffDRR "AP" orientation. alpha ≈ 0 puts the source on the anterior side of the CT, alpha ≈ 180 (e.g. `[135, 225]`) on the posterior side. **Check the side for every new set-up** ([procedure](source_side_check.md)): the pelvis is nearly left/right symmetric, so the wrong side still gives plausible-looking poses with the patient's left and right exchanged. The default is anterior; the procedure lists the set-ups checked so far. |
| `prior.lateral_mm`, `axial_mm` | Half-widths of the central-ray offset range around the centroid. |
| `prior.source_to_center_mm` | Source-to-isocentre distance range. |
| `model.method` | `finetune` (from the pinned xvr foundation model, default) or `de_novo` (from scratch; defaults 15000 iterations, warmup 1000, accumulation 4; too short to pass the self-check in the one case tried). |
| `model.*` | Training hyperparameters and the self-check size/threshold. |
| `predict.refine` | Also run xvr's gradient intensity refinement on `refine_labels`. |
| `predict.stage` | Which stage becomes the active pose directory: `network`, `refined`, or `best` (refined unless refinement lowered the image similarity by more than 0.02; needs `refine`). `refine` defaults to true for `refined` and `best`. |
| `predict.qc_targets` | `{name: label}` of the structures the automatic pose checks project into each view (coverage, in-view share). Default `{"target": prior.center_label}`. See `docs/outputs.md` for the checks and tiers. |
| `output.*` | Locations of models, the active pose directory and per-run records. |
