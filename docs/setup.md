# Installation and operating environment

## Supported deployment

The integrated extension targets **Linux x86-64, 3D Slicer 5.8**. Windows and macOS are not supported
by these instructions. Viewing completed runs does not require inference environments. New runs
need the bundled pose/annotation sources, explicitly installed external runtimes, and a compatible
manual xreg build. No separate pose or annotation source clones are required.

| Component | Environment / hardware |
| --- | --- |
| Slicer UI, viewer, pipeline CLI | Slicer 5.8 `bin/PythonSlicer`, Python 3.9; never install training dependencies here |
| Bundled annotation | Conda prefix, pinned Python **3.12.4**, TotalSegmentator 2.7.0, PyTorch 2.3.1 |
| Bundled pose | Python **3.10**, uv, committed `uv.lock`; NVIDIA GPU, about 20 GB free GPU memory for the documented training configuration |
| Manual xreg | Native C++ build; ITK/VTK/NLopt and an OpenCL runtime/device matching the registration settings |

GUI commands require Linux `setsid` (`util-linux`, normally included with the operating system).

Annotation's pinned baseline also includes NumPy 1.26.4, SciPy 1.14.0, nibabel 5.2.1, nnunetv2 2.4.2
and PyVista 0.46.3. The bundled `tools/3d_landmark_annotation/environment.yml` reflects the recorded
working Python 3.12.4 environment, **not** a claim that an older Python 3.10 resolver reproduces it.
Keep the full environment specification with the release. Pose pins PyTorch 2.10.0 (Linux CUDA 12.8
wheels; NVIDIA driver 570 or newer), DiffDRR 0.6.0 and xvr source
`caa55cc8096294cf70a218126bf16008dee0dec7`.

CUDA for PyTorch is not OpenCL for xreg. Install the GPU vendor's OpenCL ICD and inspect the exposed
device (for example with `clinfo`). A CUDA GPU alone does not establish OpenCL availability. CPU
registration must be explicitly configured; it does not make GPU pose training CPU-supported.
Full-resolution CT, segmentation and DRR work can exceed host/GPU memory. Preserve the failing
command and log; never silently change resolution or objectives to make a run finish.

Annotation's bundled `docs/low_memory.md` documents explicit low-memory options; record such changes
in the operating configuration rather than applying them silently.

## Release contents and compatibility

The **v2.0.0** distribution includes frozen backend sources; historical v1 packages do not.
The previous repository is preserved as `SlicerFluoroReg_old`; use the current repository's
matching release assets rather than its historical packages.

Installed curated sources are ordinary files under
`<extension-root>/share/Slicer-5.8/SlicerFluoroReg/tools/`:

- `bigss-pose-estimation/`: frozen BIGSS pose source, lockfile, scripts, licenses and provenance.
- `3d_landmark_annotation/`: frozen BIGSS annotation source, pinned environment and notices.
- `VENDORED_COMPONENTS.json`: exact original source revisions, curated file hashes, source
  transformations and aggregate snapshot SHA-256. The bundled source snapshots are the compatibility
  authority; default branch tips of standalone development repositories are not substitutes.

Settings marks included source paths as **bundled** and existing alternative source paths as **external**.

Snapshots change only through a reviewed extension release. Canonical original repositories retain
independent development/history; the bundled copy is not an ad hoc algorithm fork. No environment,
weight, patient scan, test dataset, agent state or unapproved image is part of the source snapshot.
The release manifest distinguishes **bundled source** from **manual xreg**, external environments
and upstream model downloads, and hashes the packaged vendor files alongside the module/docs.

## Manual xreg download/build

The BIGSS fork at https://github.com/uark-i3r-bigss/xreg remains **private for this release**.
The required compatibility commit is currently maintainer-held and is not available on the lab
remote. Request the matching private source/build before running new registrations; public
end-to-end installation is not yet available. Viewing existing completed runs does not require
xreg. Upstream xreg or xvr is not an integration substitute. The exact compatibility revision is:

```text
0b2e0dc9aa7598d49cb634eed931b769fc9918dd
```

After receiving the source, verify its revision rather than substituting a branch tip:

```bash
git -C /path/to/xreg rev-parse HEAD
```

Follow that source's `INSTALL.md`. The source-build baseline uses ITK 5.3.0, VTK 9.3.1, system NLopt,
CMake/Ninja and a consistent compiler ABI (example GCC 12):

```bash
cmake -S /path/to/xreg -B /path/to/xreg/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/path/to/itk-install;/path/to/vtk-install"
cmake --build /path/to/xreg/build --parallel 2 --target \
  xreg-single-target-regi xreg-single-target-multiview-regi \
  xreg-multi-target-regi xreg-multi-target-multiview-regi \
  xreg-drr-sim xreg-convert-dicom-radiograph xreg-regi2d3d-replay xreg-draw-xray-scene
```

Choose a job count appropriate to memory. Build success is not registration validation. Set xreg
source/build paths in Settings; executables are grouped under `build/apps/`, not all at its root.
No xreg download, build or license acceptance is performed by backend setup.

## Install the extension

Use a maintainer-published archive matching Slicer 5.8 when available, or build from the matching
release-source checkout:

```bash
python3 scripts/package_extension.py --slicer-version 5.8
```

The output is `dist/SlicerFluoroReg-linux-x86_64-Slicer5.8-git<shortsha>[-dirty].tar.gz`.
A `-dirty` package is development-only. Uninstall an older extension, restart, use **Extensions
Manager → Install from file…**, then restart again. Do not add development module search paths for
installed-package acceptance. Both the local archive builder and CMake/CPack install curated tools
under `share/`, never as Slicer Python modules, and preserve executable script permissions.

Installed documentation and `RELEASE_MANIFEST.json` live beside `tools/` under
`share/Slicer-5.8/SlicerFluoroReg/`. Help links to the installed setup guide. The manifest records the
release-source commit, dirty status, component inventory and hashes of payload files (not itself).
The extension root is above its `lib/`, not the Slicer application's root.

## Explicit backend setup

Install [uv](https://docs.astral.sh/uv/), Conda and required system tools first. `dcm2niix` must be on
`PATH` for DICOM CT conversion. Configure the manual xreg paths and Conda executable in
**FluoroReg → Settings → Backend runtimes**, then choose **Check environments** or explicitly
**Set up environments** (all, pose or annotation). Setup runs an external CLI job with the
normal progress/log/stop controls, **not pip inside Slicer**. Startup and registration never create
environments or download/install dependencies. A missing runtime provides the setup instruction;
it does not fall back to a neighboring source checkout.

Managed runtimes stay unready after cancelled setup. Run setup explicitly again; pose setup reinstalls
its locked packages to repair interrupted wheel copies.

Equivalent terminal commands, with `FluoroReg` on `PYTHONPATH` (PythonSlicer or Python >= 3.9):

```bash
export PYTHONPATH=/path/to/qt-scripted-modules  # installed lib/Slicer-5.8/qt-scripted-modules
python -m fluororeg tools status
python -m fluororeg tools setup --component all
# Or set up just pose or annotation:
python -m fluororeg tools setup --component pose
python -m fluororeg tools setup --component annotation
```

No project is required. Existing `--settings`, `--conda`, source and environment tool flags also
apply. From a source checkout the wrappers set `PYTHONPATH` for you:

```bash
bash scripts/configure.sh --xreg /path/to/xreg --conda /path/to/conda
bash scripts/setup_tools.sh --component all
```

`configure.sh` only writes explicitly supplied machine paths plus a discovered Slicer executable;
it resolves bundled sources through the same settings/discovery helpers as the GUI and CLI. It
does not persist a release-specific bundled path as an override. Precedence is **flags > settings >
bundled defaults**. Existing intentional user source paths are preserved; remove an old source
setting in Settings if you want bundled discovery. xreg always remains a manual setting.

## Stable external environments and caches

The extension installation may be read-only. All default runtime/cache writes are outside it:

```text
${XDG_DATA_HOME:-~/.local/share}/fluororeg/backends/<component>/<snapshot_sha256>/env
${XDG_CACHE_HOME:-~/.cache}/fluororeg/models
${XDG_CACHE_HOME:-~/.cache}/fluororeg/totalsegmentator
```

Setup and launch use the same full snapshot hash and directories. A new snapshot gets its own
runtime environment; do not rename an old environment to make a different snapshot look ready.
Shared model caches survive extension upgrades. Pose receives `BIGSS_POSE_ENV_DIR` and
`BIGSS_POSE_WEIGHTS_DIR` consistently, and frozen source is installed non-editably. Annotation uses
an explicit Conda prefix by default, not the globally named `totalseg` environment. Advanced
operators may override `--pose-env-dir`, `--annotation-3d-env-prefix`, or explicitly select a named
annotation environment with `--annotation-3d-env`; these overrides carry operator responsibility
for dependency compatibility. They do not merge annotation/pose into Slicer's Python environment.

An external annotation source override also requires an explicit Conda environment name or
prefix; an ordinary Python virtual environment is not a Conda prefix. Explicit
`BIGSS_POSE_ENV_DIR`, `BIGSS_POSE_WEIGHTS_DIR` and `TOTALSEG_HOME_DIR` environment overrides
remain supported; a configured `--pose-env-dir` takes precedence over the ambient environment.

First setup needs network access. Pose setup downloads the xvr foundation file from
`https://huggingface.co/eigenvivek/xvr/resolve/main/wbct/model.pth` and requires SHA-256
`82a9fd05a61f22da949885e890bce9333e4d5abc091ec011ddfc4b3454d5ffba`. Preserve this check for
offline deployment and seed the external model cache with the exact verified file. Patient-specific
trained models stay in the operator's experiment, not the source/extension tree.

TotalSegmentator may download task weights on first explicitly requested inference; its cache is
selected with `TOTALSEG_HOME_DIR`. To prefetch for offline use, run upstream's
`totalseg_download_weights -t total` in the annotation environment (`total_mr` only if needed).
Explicit annotation setup disables TotalSegmentator's upstream usage reporting in the selected
cache's configuration, preserving other fields such as an operator-supplied task license.
Review the selected task/model terms: optional licensed tasks are not covered by a general source
license. See [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES).

## Data availability and publication

Download the Linux/Slicer 5.8 package from the
[lab releases page](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases), or use the stable
asset link in the README. The [release checklist](developer.md#release-checklist) describes the
build and publication procedure. No separate public pose/annotation repository access is
required. No patient sample or historical demo asset is supplied; bring authorized local
CT/X-ray data. Code licensing does not grant medical-data or upstream-model redistribution
rights. See [pipeline.md](pipeline.md) for operator workflows.
