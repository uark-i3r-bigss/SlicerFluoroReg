# FluoroReg pipeline

The command line `fluororeg` does every step; the Slicer module **FluoroReg** and the scripts
`scripts/prepare_case.sh` and `scripts/register.sh` only run its commands. Scripts have `--help`;
`prepare_case.sh` skips each step whose output exists (`--redo-pose` retrains after the pose config is edited).

## In Slicer

Load the CT and the X-rays into the scene (drag and drop, DICOM browser, …), open **FluoroReg** and
go down the checklist; a step's mark is ✓ done, ○ to do, ! warning, ✗ error.

| Step | What you do | Done when |
|---|---|---|
| Settings | manual xreg paths; **Check environments**, then explicitly **Set up environments** for bundled backends, once per snapshot | paths and external runtimes are ready |
| Experiment | choose the experiment folder, or a run folder inside one (see below) | a folder is open |
| CT | pick the CT from the scene (or Load file…), **Segment and annotate**; segmentation, landmarks and models appear in the scene | the case is annotated |
| X-rays | the scene's X-rays are listed and checked; **Add to experiment** copies and converts them. PNG/JPG/TIFF images need a **Template DICOM** from the same C-arm | at least one X-ray in the experiment |
| Pose model | **Train** (~8 min on the GPU) or **Use existing model…**; **Edit pose config** for e.g. the source side | the CT has a model |
| Registration | pick the registration (single view or multiview, pelvis or pelvis then femur) and check the X-rays; **Register** predicts the initial poses, registers, assesses and shows the result (**Show result**; tick **Replay videos** for the optimizer replays). **Edit config** opens its JSON | ready to run (else the reason is shown) |
| Results | the run viewer: runs with the number of results of high / medium / low certainty of correct registration; **Show run** loads one into the scene (view, per-target stages, DRR overlay, look from source; landmarks are hidden until **landmarks** is ticked); **Assess** a run without a current `assessment.json` (an older schema is assessed again) | at least one run |

Status, progress, Stop and Log (the exact command lines and their output) stay on top. Warnings are only
shown, never acted on. Collapsing Slicer's Data Probe gives the panels more room.

## Experiment folder

Files are copies, never symbolic links, and paths inside the folder are relative to it, so a copied or
downloaded folder works anywhere. Registration configs are created once from the bundled defaults in
`configs/xreg/`; the pipeline changes only `volume.ct`, `volume.seg`, `volume.landmarks_3d`,
`projections.ids` and, in a moved folder, `experiment_dir` (Register sets it to the open folder; until then
the config's inputs are read from the folder that holds it). Edit everything else by hand.

Completed experiments open without being written to, even ones not made with the pipeline. A folder with
`project_manifest.json` opens as usual; one without it but with runs (under `outputs/` or the output folder
of a `configs/xreg` config) opens for viewing only, and **Set up this folder for the pipeline** enables the
other steps; an empty or new folder is set up on the spot. A run folder opens its experiment and shows the run.

## Command line

```bash
PYTHONPATH=/path/to/SlicerFluoroReg/FluoroReg /path/to/Slicer/bin/PythonSlicer -m fluororeg <command> [--help]
```

Python >= 3.9; `python3` works except for image X-rays, which need pydicom, numpy and Pillow (Slicer's
`PythonSlicer` has them). Tool paths come from flags, then `--settings FILE`, then
`~/.config/fluororeg/settings.json` (or `$FLUOROREG_SETTINGS`; written by `scripts/configure.sh`
and the module's Settings): `xreg_dir`, `xreg_build_dir`, `annotation_3d_dir`, `conda`, `annotation_3d_env`,
`annotation_3d_env_prefix`, `pose_dir`, `pose_env_dir`, `gpu`, `xray_template`, and `slicer`
(launch scripts; `$SLICER` overrides it). Bundled pose/annotation discovery supplies defaults only
after flags/settings. Named annotation environments are explicit overrides, not the bundled default.
Runtime environments and model caches stay outside the extension under XDG data/cache directories;
see [setup.md](setup.md). Startup and registration never install dependencies or fall back to sibling
source checkouts. Setup runs an external job with the normal progress/log/stop controls.

| Command | Does |
| --- | --- |
| `tools status [--component all\|pose\|annotation]` | Inspect backend source/runtime readiness; no project or installation |
| `tools setup [--component all\|pose\|annotation]` | Explicitly install frozen backend dependencies into external environments; may download models; no project |
| `init --project P` | Create the folder layout (idempotent) |
| `status --project P` | Cases, X-rays, pose model, configs and problems, runs and assessments; `set_up` false without `project_manifest.json`. Never writes |
| `import-ct --project P --input PATH [--case C]` | Copy a NIfTI/NRRD file or DICOM folder in; case default = file stem or folder name |
| `annotate-3d --project P --case C [--device gpu:N\|cpu]` | Segmentation and 3D landmarks (CT or MRI, detected by the tool), then `stage` |
| `stage --project P --case C [--overwrite]` | Copy CT, segmentation, models and landmarks into `meta_data/` |
| `import-xrays --project P --input PATH [PATH ...] [--no-convert]` | Add DICOM or image files (or folders of them) as the next ids and convert them to `.h5` (C-arm camera frame; a DICOM without a Modality tag is converted as a C-arm image) |
| `convert-xrays --project P [--ids ...] [--overwrite]` | Convert again |
| `pose train\|use-model\|init-config --project P --case C` | Train a pose model, activate an existing one (`--model-dir D`: a trained model folder, or a case's models folder whose active model is used), or write the pose config |
| `config create --project P --case C --app APP [--femur left\|right] [--name N] [--ids ...]` | New config from the default of `single-target`, `single-target-multiview`, `multi-target` or `multi-target-multiview` |
| `config set --config F [--experiment-dir DIR] [--ct REL] [--seg REL] [--landmarks-3d REL] [--ids ...]` | Change only these fields; `register` refuses a config whose `experiment_dir` is not the folder holding it |
| `config show --config F` | App, inputs, ids, targets, problems |
| `register --config F [--predict-poses] [--replay] [--no-assess]` | Predict initial poses for the config's X-rays, run xreg (replay videos only with `--replay`), assess |
| `assess --run R [--no-gncc]` | `R/assessment.json` (schema 3): per registration and target, heuristic high / medium / low certainty from target mover-stage objective, gNCC and complete registered-view coverage; see below. Older assessments are stale and are recomputed by the UI |

Status lines are `FLUOROREG {json}` (`event`: `start`, `progress`, `warning`, `error`, `result`; progress
has `phase`, `done`, `total`). Exit status: 0 ok, 2 invalid input or missing tool path, 3 finished but needs
attention, 1 other failure.

## Assessment and optimization provenance

Schema 3 does not equate the last joint objective with every target's score. A target uses the
numeric objective from the latest completed optimization stage that **moves** that target. The
additive run-manifest `quality.target_objectives` records those values; the original
`quality.final_objective` remains the last/joint-stage result for old-run reproducibility. Stage
movement/scope metadata establishes provenance. For older schema-4 multi-target runs without
the additive field, stages supply the same evidence only if the target mover is recorded. A
later femur stage which holds the pelvis must not become a new pelvis objective. If a manifest
declares exactly one target, its latest numeric stage objective (or overall final objective
when stage values are absent) is unambiguously attributable to that target; the assessment
records this as `inferred_single_target`. No such fallback is permitted for multi-target runs.
Stage objectives may include shared-image and held-anatomy terms; they are not independent
single-bone errors. Otherwise, insufficient provenance remains explicitly unavailable.

The existing rating thresholds remain: high requires gNCC >= 0.8 and objective <= 0.85; low
requires gNCC < 0.7 and objective > 0.85. A target cannot be high unless **every admitted registered
view** has a numeric metric and finite coverage in `(0, 1]`. There is no new fixed percentage
threshold: the existing scorer's insufficient-footprint rule (null gNCC) remains authoritative.
Missing metrics, missing/invalid coverage and unavailable objective provenance produce warnings
and prevent high even when a subset scores well. Items expose `target_objective`,
`objective_provenance` (stage, movers, held targets, views and shared-image scope),
`required_view_ids`, `scored_view_ids`, `missing_view_ids`, `missing_coverage_view_ids`,
`insufficient_coverage_view_ids`, `missing_objective_view_ids`, `coverage_by_view` and
`evidence_complete`. `--no-gncc` therefore does not provide evidence for high. WholeCT footprint
coverage keeps its existing definition; this requirement does not turn it into a bone-specific
segmentation score. Ratings are research triage aids, not error estimates or clinical validation.

## Registration seeds and result display

Registration-time pose prediction is not a write through the mutable staged
`meta_data/initial_poses/<case>/` directory. Prediction's selected `best/` contents are copied
(not symlinked) out of scratch into a durable, unique
`meta_data/initial_poses/predictions/<case>/<prediction-id>/` directory. The registration
execution config in `logs/<case>/register/execution_*.json` selects only that durable copy,
without rewriting the user's editable config or reading quality from a mutable active-model
pointer. Keep the durable pose copy and execution config with the experiment; removing the
scratch prediction run must not invalidate registration inputs. A later prediction must not
retroactively change a completed registration's seed provenance. When `--predict-poses` is not
used, stability of manually configured seed sources remains the operator's responsibility.

The viewer's saved DRRs are **final-result** images. Selecting an initial or intermediate model
stage does not recompute the DRR: the final overlay is hidden and a final-only notice is shown
until the relevant targets are back at their final stages. This avoids presenting a final image
as an intermediate-stage reprojection.


## Scripting the module

Every button is a widget method and `state()` returns what the panels show; see
[developer.md](developer.md#scripting-through-the-webserver):

```python
w = slicer.modules.fluororeg.widgetRepresentation().self()
w.set_project("/path/to/experiment")
w.run_annotate()                       # returns at once; poll w.state()["busy"]
w.add_xrays_to_experiment(); w.run_pose_train()
w.select_registration("Multiview (pelvis)"); w.set_checked_ids(["005", "006", "010"]); w.run_register()
w.show_run(); w.set_view("005"); w.run_viewer.scene  # the shown run's RunScene
```
