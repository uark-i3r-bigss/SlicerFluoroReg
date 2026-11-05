# BIGSS Pose Estimation

BIGSS Pose Estimation gives xreg a starting pose for every X-ray, so 2D/3D registration needs no 2D
landmarks. For each CT, it trains a network on simulated X-rays of that CT, starting from the
[xvr](https://github.com/eigenvivek/xvr) foundation model. The trained network then predicts where the
X-ray source and detector were relative to the CT for each real X-ray.

- **Input**: a project folder made by the BIGSS Pipeline: the CT and its segmentation in
  `meta_data/volumes/`, the X-rays as xreg projection files in `meta_data/XrayProjData/`
- **Output**: one pose file per X-ray in `meta_data/initial_poses/<case>/`, which xreg reads as its
  starting pose, and `quality.json`, which rates each pose

## Setup

Supported BIGSS integration: **Linux x86-64**, [uv](https://docs.astral.sh/uv/), Python **3.10**
(`pyproject.toml` requires >=3.10,<3.11), and an NVIDIA GPU with about 20 GB of free memory for the
documented training configuration. Do not install this environment into Slicer's Python 3.9.
`uv.lock` pins PyTorch 2.10.0 (Linux CUDA 12.8 wheels; NVIDIA driver >=570), DiffDRR 0.6.0 and the
xvr source revision `caa55cc8096294cf70a218126bf16008dee0dec7`. CUDA support here does not install
the separate OpenCL runtime used by xreg.

```bash
scripts/setup_env.sh    # creates .venv and downloads the xvr weights into models/
scripts/pose.sh check   # confirms the GPU and the weights
```

`setup_env.sh` uses `uv sync --frozen --reinstall` (editable for the standalone `.venv`) and downloads
`https://huggingface.co/eigenvivek/xvr/resolve/main/wbct/model.pth` into
`models/xvr_wbct/model.pth` (models root override: `BIGSS_POSE_WEIGHTS_DIR`).
It requires SHA-256 `82a9fd05a61f22da949885e890bce9333e4d5abc091ec011ddfc4b3454d5ffba`;
a mismatch is an error, not permission to bypass the check. First setup requires network access.
For offline use, retain the exact verified weights and the locked dependency artifacts.
Rerunning setup reinstalls the locked packages to repair a partially interrupted installation.
GPU memory failures should remain explicit; do not silently change the training/render resolution.

For a frozen extension snapshot, keep the environment and weights outside the installed source:

```bash
export BIGSS_POSE_ENV_DIR="$HOME/.local/share/fluororeg/backends/pose/<snapshot_sha256>/env"
export BIGSS_POSE_WEIGHTS_DIR="$HOME/.cache/fluororeg/models"
scripts/setup_env.sh
scripts/pose.sh check
```

Both scripts set `UV_PROJECT_ENVIRONMENT` from `BIGSS_POSE_ENV_DIR` (default: the standalone
checkout's `.venv`). For an external environment directory, setup adds `--no-editable` to install
a wheel, not retain an editable import link to a different checkout. The original standalone
`.venv` remains editable for source development. Use the same overrides for setup and every
launch. Relative override paths are resolved from the caller's current directory before setup
changes directory, and both launchers export the selected weights root for wheel installations.
The extension selects these paths using `XDG_DATA_HOME` and `XDG_CACHE_HOME` when set; the
standalone defaults remain `.venv` and `models/`. Setup is explicit, never an automatic inference
or training prerequisite.

The standalone development source and the extension's frozen snapshots are versioned
independently. The authoritative source revision for a released extension is the `pose`
component's `source_revision` in that extension's `tools/VENDORED_COMPONENTS.json`. Operators use
the bundled snapshot; access to the private standalone development repository is not required.

## License, model and data provenance

[LICENSE](LICENSE) grants MIT terms for the BIGSS-owned wrapper implementation and documentation,
Copyright (c) 2026 Ping-Cheng Ku and contributors. It does **not** relicense xvr, third-party
dependencies, foundation weights, training data or patient-specific models.

The pinned [xvr source LICENSE](https://github.com/eigenvivek/xvr/blob/caa55cc8096294cf70a218126bf16008dee0dec7/LICENSE)
is MIT, Copyright (c) 2025 Vivek Gopalakrishnan. The
[upstream model card](https://huggingface.co/eigenvivek/xvr/blob/main/README.md) declares `license: mit`
and identifies `wbct/model.pth` as the foundation model. Its linked standalone model LICENSE was
unavailable (HTTP 404) when reviewed on 2026-11-04: this is model-card evidence, not a new BIGSS
grant or a blanket clearance for upstream datasets. Keep upstream terms and attribution with any
authorized redistribution; the extension does not redistribute the weights.

No patient scans, upstream benchmark data or historical FluoroReg demo assets are licensed for
redistribution by this code license. Authorized inputs and trained outputs require their own data
governance/rights review. Sample publication remains pending separate rights approval.

## Usage

```bash
scripts/pose.sh init-config --project <project> --case <case>              # once per CT
scripts/pose.sh train   --config <project>/configs/pose_estimation/<case>.json   # once per CT, ~7 min
scripts/pose.sh predict --config <project>/configs/pose_estimation/<case>.json   # after adding X-rays
```

`init-config` writes the case's settings file; edit it to change them. Run `scripts/pose.sh --help` or
`scripts/pose.sh <command> --help` to see every command and option.

To start an xreg registration from the predicted poses, set the target's `init` in the xreg config to
`{"method": "run", "from_run": "meta_data/initial_poses/<case>", "from_target": "ct"}`.

Before you use a new C-arm or table set-up, check which side of the patient the X-ray source is on
([source-side check](docs/source_side_check.md)). A model trained for the wrong side swaps the
patient's left and right, and its poses still look plausible.

## Output structure

```text
<project>/
├── configs/pose_estimation/<case>.json   # settings of the case
└── meta_data/
    ├── pose_models/<case>/               # trained models; active.json names the one predict uses
    ├── initial_poses/<case>/             # predicted poses, quality.json
    └── tmp/pose_estimation/<case>/       # full record of each predict run, with overlay images
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/config_schema.md](docs/config_schema.md) | Settings file keys and defaults |
| [docs/outputs.md](docs/outputs.md) | Model and pose files, quality ratings, conventions |
| [docs/source_side_check.md](docs/source_side_check.md) | Source-side check for a new set-up |

## Notes

The ratings in `quality.json` help choose which X-rays to register; they are not error estimates.

## Contact

Ping-Cheng Ku (pku1@jh.edu)
