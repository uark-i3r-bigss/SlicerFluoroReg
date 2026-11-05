# Developer notes

## Layout

The extension `SlicerFluoroReg` has one module, `FluoroReg` (`add_subdirectory(FluoroReg)` in the
top-level `CMakeLists.txt`; `scripts/package_extension.py` and `scripts/module_dirs.sh` read the module list
from there). The module directory goes on `--additional-module-paths` and on `PYTHONPATH`. It holds
`FluoroReg.py`, its `CMakeLists.txt` (`MODULE_PYTHON_SCRIPTS`, `MODULE_PYTHON_RESOURCES`; every tracked
file must be listed) and its packages. Helper code lives in packages: Slicer treats every top-level `.py` in a
module directory as a module.

```text
CMakeLists.txt                    extension description (EXTENSION_*), add_subdirectory(FluoroReg)
FluoroReg/CMakeLists.txt      module files; the package holds exactly these
FluoroReg/FluoroReg.py    scripted module: steps, QProcess runner of the CLI, Results viewer, state(), self-test
FluoroReg/fluororeg_viewer/     run viewer (Slicer): RunScene, RunViewer, run checks of the self-test
FluoroReg/fluororeg_runs/     run library (numpy only, Python >= 3.9, no Slicer/VTK/Qt/h5py)
  manifest.py    XregRun: load/validate a schema 4 manifest, experiment folder, paths, poses
  transforms.py  4x4 helpers, RAS_from_LPS, Slicer to-parent of an ITK file, CT[primary]_from_CT[t]
  camera.py      RawCamera: projection, detector-plane points, frustum
  scene.py       node matrices for a view, per-target stages, CT- or camera-centred frame
  landmarks.py   swap-map inverse, residual recomputation
  fcsv.py, mha.py, imaging.py   FCSV reader, MetaImage reader, X-ray + DRR composite
  contract.py    contract_check(run_dir); CLI in __main__.py
FluoroReg/fluororeg/     the pipeline CLI (standard library only), see docs/pipeline.md
tools/VENDORED_COMPONENTS.json    curated backend allowlist, original revisions and snapshot hashes
tools/bigss-pose-estimation/      frozen ordinary pose source; not on Slicer's Python module path
tools/3d_landmark_annotation/     frozen annotation source and pinned Python 3.12.4 environment
scripts/configure.sh              write the tool settings file shared by the CLI and the GUI
scripts/setup_tools.sh            explicit external backend setup via the shared tools CLI
scripts/prepare_case.sh           CT, X-ray and pose steps of one case, skipping finished steps
scripts/register.sh               config create/set, register, assess; prints the run and its counts
scripts/pipeline_common.sh        helpers shared by the three scripts above
scripts/package_extension.py      build the extension package (standard library only)
scripts/launch_slicer.sh          open Slicer with FluoroReg, optionally an experiment or run folder
scripts/slicer_startup.py         startup of launch_slicer.sh: open folder (FLUOROREG_OPEN), select module, WebServer, FLUOROREG_SLICER_READY
scripts/run_tests.sh              local unit tests (if present), contract check, headless Slicer self-tests
scripts/slicer_selftest.py        runs the FluoroReg self-test
scripts/find_slicer.sh            Slicer lookup shared by the shell scripts
scripts/module_dirs.sh            module discovery shared by the shell scripts
```

## Launch and package

```bash
scripts/launch_slicer.sh [--module NAME] [<folder>]                   # experiment or run folder opened
scripts/launch_slicer.sh --headless --webserver [<port>] <folder>     # xvfb, WebServer on 127.0.0.1 (default 2017)
python3 scripts/package_extension.py [--slicer-version 5.8]          # dist/SlicerFluoroReg-linux-x86_64-Slicer5.8-git<sha>[-dirty].tar.gz
```

The start module is `--module`, else FluoroReg; a folder is opened with `set_project` and startup prints
`FLUOROREG_SLICER_READY`. The repository's module takes precedence over an installed extension. `--webserver`
enables Python `exec` for any local process; use it only when needed. Only tracked module files,
required release documents and curated vendor-manifest files are packaged (tracked working-tree
changes mark the archive `-dirty`). Local archives and CMake/CPack share the same validated vendor
allowlist and source/payload manifest generator; the latter generates explicit install rules during
configure. Shell executable modes are preserved. Backend source goes under
`share/Slicer-X.Y/SlicerFluoroReg/tools`, never `qt-scripted-modules`.
The package targets Linux x86-64 and the Slicer minor version named in the archive. Install it with
Extensions Manager → "Install from file…" after uninstalling the previous extension and restarting.
xreg executables, Python environments, weights and datasets are not bundled. See [setup.md](setup.md).

On Linux, GUI jobs launch through `setsid` so Stop reaches both PythonSlicer and its Python process.
The CLI reaps its separate worker group; the GUI stays busy until the launcher group exits before
allowing another command. A 15-second hard-kill deadline leaves time for the CLI's 10-second worker
cleanup. The built-in self-test exercises cancellation through the real launcher and worker.

The package includes `LICENSE`, `THIRD_PARTY_NOTICES`, `README.md` and the setup/pipeline/developer
docs under `share/Slicer-X.Y/SlicerFluoroReg/`. Module Help resolves setup from that installed
location, not a checkout-only absolute path. `RELEASE_MANIFEST.json` is generated from the source
tree being packaged: exact source HEAD, tracked dirty status, Slicer/platform identity, module list,
setup-document path, component inventory (bundled pose/annotation source versus manual xreg,
external runtimes and weights), and size/SHA-256 for every declared payload member. It does not hash
itself or claim to describe independently installed environments. CPack's Slicer-generated extension
metadata is separate from this source payload inventory. Required documents and vendor files must
be tracked. Never commit environments, caches, weights, patient examples, tests or agent state under
`tools/`; the snapshot allowlist and its hashes reject stray tracked payload.

## Release checklist

Publication is a maintainer/user action, not a side effect of packaging. Follow this checklist for
each release; do not attach old demo archives just because their historical names are known.

1. Finalize changes in the development repositories. Review license grants and
   `THIRD_PARTY_NOTICES`; keep external model/data rights separate from the MIT code grants.
   Exclude machine paths, patient data, local settings, weights, build trees and internal artifacts.
2. Record final original source revisions and curated content in `tools/VENDORED_COMPONENTS.json`.
   The snapshot digest is SHA-256 of sorted UTF-8 `relative_path + NUL + file_sha256 + newline` records.
   Review transparent curation transformations; do not fork algorithms in the frozen copy. Publish
   bundled versions only as part of an extension release. Keep original source repositories/history
   intact; public access to separate pose/annotation repos is not an installation prerequisite.
   Make the exact manual xreg SHA/tag accessible, and record runtime versions, drivers, model
   task/checksum and tested hardware. No default branch tip substitutes for a frozen revision.
3. Run the local unit tests and Slicer fixture self-test, then contract/Slicer checks on fresh
   authorized outputs. Exercise all four registration modes, PnP per view, mover-stage objectives,
   incomplete-view assessments, immutable seed reuse, and intermediate-stage overlay behavior.
   Keep real-input/memory failures explicit; neither a fixture pass nor a successful build proves
   registration accuracy. Do not include private acceptance evidence in the source package.
4. Commit the reviewed development source. Export that committed source with `release/export.sh`
   into the separate `SlicerFluoroReg` release-source checkout (when operating in the development
   repository; release exports omit `release/`). Review, commit and tag the release source.
   No export, commit, tag, push or publication is performed by the package builder.
5. Build **in that clean release-source checkout**, not in the development tree:
   `python3 scripts/package_extension.py --slicer-version 5.8`. Require no `-dirty` suffix.
   Inspect `RELEASE_MANIFEST.json`: revision must be the release-source commit, `source_dirty` false,
   docs/licenses and bundled source present, snapshot/payload hashes matching the archive, and no
   runtime/cache/model/medical-data payload. Record archive SHA-256 separately in release notes.
6. Install that exact archive in a fresh Slicer profile without development module paths or sibling
   source clones. Open Help, check notices and bundled discovery, configure manual xreg, and run
   explicit tools status/setup (including stop/failure handling and read-only extension layout).
   Confirm setup/launch use identical external XDG runtime/cache paths. Exercise normal registration
   and existing-run viewing without any checkout on the module search path.
7. Only after review and explicit publication action, attach the archive and checksum to a release
   in the [lab release repository](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases).
   The bundled release line starts at **v2.0.0**; preserve the previous repository privately as
   `SlicerFluoroReg_old`, never present historical v1 assets as the new release. Upload the verified
   archive using the stable asset name `SlicerFluoroReg-linux-x86_64-Slicer5.8.tar.gz`, plus a
   `SHA256SUMS` file containing its bare filename. Keep the Git-stamped local archive for provenance.
   xreg remains private for this release; document its permission-required revision rather than
   claiming public end-to-end installation. Attach samples or video only after separate rights
   approval and de-identification review; otherwise leave them unavailable.

## Scripting through the WebServer

```bash
scripts/launch_slicer.sh --headless --webserver 2017 <run_dir>
curl -s -X POST http://127.0.0.1:2017/slicer/exec --data 'import slicer
w = slicer.modules.fluororeg.widgetRepresentation().self()
s = w.run_viewer.scene  # RunScene of the shown run (None when nothing is shown)
s.set_view("004"); s.set_stage("pelvis", "init"); s.set_camera_centred(True); s.look_from_source()
__execResult = {"stages": s.stages}'
curl -s http://127.0.0.1:2017/slicer/threeD -o threeD.png
```

`w.state()["viewer"]` reports the shown run; the widget's `show_run`, `set_view`, `set_stage`, `set_display`
drive the Results panel. `RunScene` methods: `set_view`, `set_stage`, `set_camera_centred`, `set_stages_all_views`,
`set_show_all_views`, `set_drr_overlay(on, opacity)`, `set_models_visible`, `set_volume_rendering`,
`look_from_source`, `overview`, `show_detector_in_slice_view`.

## Frames used by the scene

Full conventions: `docs/frames_and_conventions.md` in xreg.

- Transforms are `A_from_B`, 4x4, mm. CT is LPS, Slicer world is RAS, `RAS_from_LPS = diag(-1, -1, 1, 1)`.
- C-arm node per view: `RAS_from_CAMWORLD[view] = RAS_from_LPS @ CT_from_CAMWORLD`.
- Non-primary target `t`: `CT[primary]_from_CT[t] = P_primary @ inv(P_t)` (poses of the selected view).
- Camera-centred mode expresses the scene in the selected view's CAMWORLD.
- ITK `.h5` files are from-parent: loading `M = CT_from_CAMWORLD` gives to-parent
  `RAS_from_LPS @ inv(M) @ RAS_from_LPS`. The scene is built from the manifest matrices, not the files.
- FCSV files are read in their declared `# CoordinateSystem`; a file without the header is rejected.

## Contract check and self-test

```bash
PYTHONPATH=FluoroReg python3 -m fluororeg_runs check [--json report.json] <run_dir> [...]
scripts/run_tests.sh [<run_dir> ...]   # always Slicer fixture self-test; real-run checks when supplied
```

`scripts/run_tests.sh` first runs local unit tests if `tests/` is present, then always runs the
headless Slicer fixture self-test, even with no real run directories and in a release tree without
local tests. Only real-run contract checks are skipped when no runs are supplied. Its summary lists
executed passes and explicit skips; a skipped check is never reported as passed.

`check` fails on: a non-schema-4 manifest or a `run.dir` not matching the end of the run directory;
missing or non-relative run files or inputs (absolute inputs need the `input_outside_experiment` flag); unknown frames or non-rigid matrices; missing or mismatched
view transform references; `poses/` not matching the per-view final poses; `.mha` size/spacing not
matching the raw camera; recomputed landmark residuals off by more than `--tol-px` (0.5 px); multiview
final poses not equal to `J @ F_i` within `--joint-tol-mm` / `--joint-tol-rot`.

`scripts/slicer_selftest.py` runs the FluoroReg self-test and fails when the module is not loaded. For
each run in `FLUOROREG_TEST_RUNS` the self-test checks the CT path, the `.h5`
to-parent matrices, image node size/spacing/pixels, every stage's node matrices (CT- and
camera-centred) against the manifest, and reproduces xreg's landmark residuals within 0.5 px from
Slicer nodes. Without runs it needs no external tool: settings round-trip, the argv of every
step (parsed by the CLI's parser), node file → project-relative path, the X-ray list from tagged nodes,
status-line parsing, config selection → selectors / X-ray checks / `config set` fields on a fixture
project, and the step checklist (marks from a fixture status, `show_step` pages, X-ray list in Registration).
To run the self-test against the installed extension instead of this repository's module:

```bash
FLUOROREG_TEST_RUNS=<run_dir> xvfb-run -a "$SLICER" --no-splash --no-main-window --python-script scripts/slicer_selftest.py
```
