# Outputs and conventions

## Model directory — `meta_data/pose_models/<case>/<YYYYMMDD_HHMMSS>_<method>/`

| file | content |
|---|---|
| `model.pth` | weights + the xvr training config; loadable by `xvr.model.network.load_model` |
| `model_card.json` | estimator version, base weights and hash, CT/seg/config hashes, detector geometry, prior and resolved pose ranges, training parameters, final metrics and peak reserved GPU memory (`training.peak_reserved_gib`), wall-clock `timing` per stage (setup, trainer init = CT load, training loop and iterations/s, save, self-check, total), self-check summary, `self_check_passed`, `model_sha256` |
| `self_check.json` | every self-check sample (pose index, contrast, label coverage, errors, captured) |
| `train_log.jsonl` | every 25th training iteration: loss terms, rotation (deg) and translation (mm) error on the batch |

`meta_data/pose_models/<case>/active.json` = `{"model_dir": ..., "activated": ...}` names the model
`predict` uses. Only a model that passed its self-check (or was force-activated) becomes active,
unless `activate --model-dir D` names one explicitly (D is copied into the models folder if outside it;
a models folder as D selects its active model, else its only model subfolder). `model_dir` is relative
to the folder holding `active.json`, so a moved or copied project keeps its model (absolute values from
older files still work).

## Pose directory

The active set is `meta_data/initial_poses/<case>/`; each prediction run keeps its own copy under
`meta_data/tmp/pose_estimation/<case>/<YYYYMMDD_HHMMSS>_predict/{network,refined,best}/` with
`run.json`, `timing.json` and `qc/` (below). `best` (with refinement) holds per view the refined pose
unless refinement lowered `gncc` by more than 0.02, in which case the network pose; its
`provenance.json` records `source_stage` per view.

| file | content |
|---|---|
| `view-<id>_target-ct_CT_from_CAMWORLD.h5` | one per projection id: ITK `AffineTransform_float_3_3`, zero centre — the layout xreg writes itself; the only name xreg's `init.method: "run"` reads |
| `provenance.json` | key, pose meaning, `transform` (`{"to": "CT", "from": "CAMWORLD"}`), `file_pattern`, model directory and hash, config and hash, crop/intensity, run directory, per-file name and sha256, and the exact xreg `init` block (`xreg_init`; its `from_run` is relative to the project root, which is xreg's `experiment_dir`; absent when the config has no project root) |
| `quality.json` | automatic checks of every pose of this stage: thresholds, targets, a summary (tier counts and ids, flag counts, views per target in view) and per view the metrics, `tier` and `flags` (below) |

The matrix is `CT_from_CAMWORLD` (A_from_B: `p_CT = CT_from_CAMWORLD · p_CAMWORLD`): it maps
camera-world points of X-ray `<id>` into CT LPS mm, i.e. xreg's `cam_to_vol`. Camera world is the
projection H5's frame `extrinsic⁻¹` of the camera frame (identity for native-grid conversions). It is
the pose of the **whole CT**, so every target of that CT starts from it at CT pose; key `ct` (the
`target-ct` part of the name) makes that explicit. xreg, whose `experiment_dir` is the project
root, reads it with

```json
"init": { "method": "run", "from_run": "meta_data/initial_poses/<case>", "from_target": "ct" }
```

and fails if any configured projection id has no file (it never falls back to PnP silently).

## Frame bridge

See `src/bigss_pose/bridge.py`: `P = W·X·E⁻¹·C·R⁻¹`, `X = W⁻¹·P·R·C·E` with `C = diag(1,−1,−1)`,
`W` = LPS→RAS then minus the torchio volume isocentre, `R` = DiffDRR "AP" reorientation, `E` the
projection's extrinsic. It was checked pixel by pixel against xreg's projection of a real 768² X-ray.

## Automatic pose checks (`quality.json`, `qc/`)

`predict` scores every pose it writes, per stage, without ground truth (`src/bigss_pose/quality.py`):

| metric | meaning |
|---|---|
| `outside_prior` | xvr parameters beyond the training prior by more than 5° / 25 mm: extrapolation |
| `targets.<name>.coverage`, `.in_view` | each `predict.qc_targets` label projected through the pose: share of the cropped detector it covers (32×32 raster) and share of its voxels on the detector |
| `gncc` | gradient NCC (Sobel, σ = 1) of a whole-CT DRR at the pose against the X-ray over the cropped detector at 256 px, 12 px border excluded (DiffDRR's zero-padded Sobel otherwise lets the fixed image border dominate: every pose scored 0.8–0.9) |
| `basin` | mean `gncc` lost when the pose is displaced ±3° about and ±8 mm (in plane) / ±40 mm (depth) along each camera axis at the prior label's centroid; `local_max` whether no displacement scores higher |
| `refine_change` | refined stage: rotation and in-plane/depth displacement from the network pose |

Tiers: `suspect` on any hard flag (`no_target_in_view` = no target covers ≥ 5 %, `blank_drr`,
`low_similarity` = `gncc` < 0.20, and `outside_prior` for a network pose); otherwise `confident`
without flags and `plausible` with soft flags (`moderate_similarity` = `gncc` < 0.30,
`flat_similarity` = `basin` < 0.02, `refine_moved_far` = more than 10° or 20 mm in plane from the
network pose, `outside_prior` for a refined pose). Thresholds were calibrated on 142 Cios Fusion views
of a Loop-X CBCT case (2026-09-29) by reviewing the overlays; each `quality.json` stores them.
They rank views for registration; they are not error bounds. A whole run scoring low (median
`gncc` well under 0.5 for views that clearly show the pelvis) is the signature of a prior on the
wrong source side: there the anterior prior gave a refined median of 0.30 with left and right
exchanged, the posterior prior 0.91.
`qc/<stage>_overlays_<n>.png` shows every view with the DRR's strongest gradients in red, titled
with id, tier (colour) and `gncc`.

## xreg run scores (`score`)

`score --config C --run R --out F [--targets key=label ...]` reads `R/run_manifest.json` (schema 4)
and scores every `R/poses/view-<id>_target-<key>_CT_from_CAMWORLD.h5`, single-view and multiview
runs alike, against the run's projection `<id>` (`views[].projection.path`, relative to the
experiment folder = `R` minus `run.dir`). The CT, segmentation, crop and intensity come from the pose
config; a run CT or segmentation other than the config's is a `warning`.

| key | content |
|---|---|
| `schema_version`, `run`, `config`, `seconds` | `1`, the run directory and pose config (absolute), wall-clock seconds |
| `scores[]` | per pose file: `view`, `target` (key), `label`, `gncc`, `coverage` |

`gncc` is the Sobel-gradient NCC of `gncc` above restricted to the target's footprint: its label's
voxels projected through the pose onto the 256 px grid, closed (3 px), holes filled, dilated (4 px),
12 px border excluded; `null` below 50 px or for a blank DRR. Content the CT lacks (spine above it,
soft tissue, the DRR's field-of-view edge) thus stays out of the score. `coverage` is
`targets.<name>.coverage` above. The label of a target key is `--targets`, else the manifest's
`targets[].segmentation_label`, else `predict.qc_targets`; none is exit 2. Well-aligned pelvis views
score about 0.88–0.95.

## Timing (`timing.json`, also `run.json.timing`)

Wall-clock seconds on the recorded GPU: `setup_s` (model load, CT frame, refinement subject, check
renderer), `first_view_s` (includes CUDA warm-up), `per_view_after_first_s` per stage (`read_s`,
`preprocess_s`, `network_s`, `qc_network_s`, `refine_s`, `qc_refined_s`: mean, median, p90, min,
max, total) and `per_frame_latency_after_first_s` for the network pose, network pose + checks and
refined pose + checks, plus `peak_allocated_gib`.
