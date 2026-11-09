# SlicerFluoroReg

[![SlicerFluoroReg Results viewer in 3D Slicer](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/download/v2.0.0/slicer-results.gif)](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/download/v2.0.0/slicer-results.mp4)

Results viewer: inspect initial/final poses and switch X-ray views. [Video](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/download/v2.0.0/slicer-results.mp4) · [Screenshot](https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/download/v2.0.0/slicer-results.png)

A Linux 3D Slicer extension that registers C-arm X-rays to CT and displays the results.
The FluoroReg module prepares inputs, trains or selects an initial-pose model, and runs single-target or pelvis/femur registration in single-view or multiview mode.
It can also open completed experiments read-only.

- **Input:** authorized local CT and X-rays, with their imaging geometry.
- **Output:** an experiment folder containing prepared inputs, models, registration poses and viewable results.

## Setup

```bash
curl -fLO https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/latest/download/SlicerFluoroReg-linux-x86_64-Slicer5.8.tar.gz
curl -fLO https://github.com/uark-i3r-bigss/SlicerFluoroReg/releases/latest/download/SHA256SUMS
sha256sum -c SHA256SUMS
```

Requires **Linux x86-64 and 3D Slicer 5.8**. Install the archive with **Extensions Manager → Install from file…**, then restart Slicer.
In **FluoroReg → Settings**, configure the separate [BIGSS xreg](https://github.com/uark-i3r-bigss/xreg) build and Conda executable, then select **Set up environments**.
Install uv, Conda and the required GPU/system tools first; [docs/setup.md](docs/setup.md) gives the exact versions and commands.
New registrations require a maintainer-provided, matching private xreg source/build; viewing completed runs does not.

Pose and 3D annotation source snapshots are included; no separate backend source clones are needed.
Their environments and model caches remain outside the extension. Setup is explicit—startup and registration never install dependencies.
Slicer's Python, annotation's Python 3.12.4/Conda environment and pose's Python 3.10/uv environment stay separate.

## Usage

```bash
bash scripts/configure.sh --xreg /path/to/xreg --conda /path/to/conda
bash scripts/setup_tools.sh
scripts/launch_slicer.sh /path/to/experiment
```

From a matching source checkout, these commands configure tools, set up the external environments and open the GUI.
For terminal operation, prepare a case and choose a registration mode:

```bash
scripts/prepare_case.sh --project /path/to/experiment --ct /path/to/ct --xrays /path/to/xrays
scripts/register.sh --project /path/to/experiment --app multi-target-multiview --ids 005 006 010
```

The other modes are `single-target`, `single-target-multiview` and `multi-target`.
Use each command's `--help`; [docs/pipeline.md](docs/pipeline.md) covers configuration and results.

## Output structure

```text
<experiment>/
├── configs/xreg/          # registration configurations
├── meta_data/             # prepared inputs, pose model and initial poses
├── logs/
└── outputs/<case>/<run>/  # poses, views, run_manifest.json, assessment.json
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/setup.md](docs/setup.md) | Installation, exact dependencies, external caches and model/data terms |
| [docs/pipeline.md](docs/pipeline.md) | Commands, settings, GUI and result interpretation |
| [docs/developer.md](docs/developer.md) | Scripting, packaging and release procedure |
| [LICENSE](LICENSE) | MIT license for the extension |
| [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) | Bundled-source, upstream software, model and data terms |

## Notes

Research software, not a medical device; completion and assessment ratings do not establish clinical accuracy.
No patient samples or model weights are included, and the code license grants no redistribution rights to medical data or upstream models.

## Contact

Ping-Cheng Ku (pku1@jh.edu)
