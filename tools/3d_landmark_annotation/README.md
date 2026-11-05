# 3D Landmark Annotation

This is a frozen, curated SlicerFluoroReg backend snapshot. Configure it with the extension's
`python -m fluororeg tools setup --component annotation` command; the manual environment
instructions below describe the recorded standalone baseline. No scans or model weights
are included. See `../VENDORED_COMPONENTS.json` for the source commit and curation record.

Automatic detection of pelvis, femur, lumbar, and sacral landmarks from CT (DICOM or NIfTI) using
TotalSegmentator masks plus 3D mesh analysis. Outputs 3D Slicer `.fcsv` landmarks, STL meshes, and
per-anatomy crops for downstream 2D/3D registration.

- **Pelvis**: ASIS, AIIS, pubic tubercle, ischial tuberosity, PSIS, pubic symphysis, acetabular
  (femoral head) centers
- **Femur**: femoral head center, greater/lesser trochanter, femur-local frame
- **Lumbar / sacrum**: L1–L5 centroid, spinous and transverse processes, sacral apex
- **Interface**: CLI for one case or a directory of cases
- **Coordinates**: LPS world coordinates in all meshes and FCSV output

## Setup

Supported BIGSS integration: **Linux x86-64**, with a separate **Python 3.12.4** Conda environment
reproducing the recorded `totalseg` baseline. Do not install TotalSegmentator/PyTorch into Slicer's
Python 3.9. CPU annotation is supported;
GPU annotation needs a PyTorch CUDA build compatible with the installed NVIDIA driver. CUDA for
annotation is separate from xreg's OpenCL backend. Windows/macOS support of upstream components
does not establish support of this integrated shell pipeline.

```bash
conda env create --prefix "$HOME/.local/share/fluororeg/annotation-env" --file environment.yml
conda activate "$HOME/.local/share/fluororeg/annotation-env"
```

For DICOM input, `dcm2niix` must be available in the environment or on `PATH`.

`environment.yml` records the actual Linux baseline exported with `conda env export --no-builds`:
Python 3.12.4, TotalSegmentator 2.7.0, PyTorch 2.3.1, NumPy 1.26.4, SciPy 1.14.0, nibabel 5.2.1,
nnUNet v2 2.4.2 and PyVista 0.46.3. It pins Conda/PyPI versions and the `defaults` channel, omitting
the source machine's environment name and prefix. The export's `python-graphviz==0.20.3` Conda
alias is normalized to the actual PyPI distribution `graphviz==0.20.3`; its version is unchanged.
This is a platform-specific reproduction baseline, not a promise that a new install has been
validated on every machine. The extension
uses its snapshot-versioned external environment prefix; an explicit named environment override
remains available for separately managed setups.

`requirements.txt` is an unpinned development dependency list, not a reproduction lockfile.
Record PyTorch/CUDA, device, task and model identity with reproduction evidence. The default
CT task is `total`; MR uses `total_mr`. TotalSegmentator downloads selected weights on first
inference; its `totalseg_download_weights -t total` can prefetch the CT task, and
`TOTALSEG_HOME_DIR` selects the model cache. Keep upstream model terms and the exact cache when
preparing an offline environment. Do not replace normal-resolution processing silently after an
out-of-memory failure; explicitly select/document any low-memory configuration.

The standalone development source and the extension's frozen snapshots are versioned
independently. The authoritative source revision for a released extension is the `annotation`
component's `source_revision` in that extension's `tools/VENDORED_COMPONENTS.json`. Operators use
the bundled snapshot; access to the private standalone development repository is not required.

## License and data boundaries

[LICENSE](LICENSE) grants MIT terms for the BIGSS-owned implementation and documentation,
Copyright (c) 2026 Ping-Cheng Ku and contributors. It does not relicense TotalSegmentator,
optional task weights, third-party code, reference implementations, scans or sample data.
See [PROVENANCE.md](PROVENANCE.md) for source/model evidence and restrictions.
The code license grants no redistribution rights for `examples/data`, its archive, or historical
FluoroReg patient samples. Use the synthetic phantom below when no authorized real input is
available; sample publication requires separate rights review.

## Quick start

Try the pipeline without any CT data. The generator writes a synthetic phantom (CT + masks) and
prints the exact follow-up command:

```bash
python examples/make_synthetic_case.py --output-dir "$HOME/.cache/fluororeg/synthetic"
```

See [`examples/README.md`](examples/README.md). The phantom is a smoke test, not anatomical
validation.

### Real CT

```bash
conda activate "$HOME/.local/share/fluororeg/annotation-env"
python process_medical_scans.py /path/to/input /path/to/output
```

`/path/to/input` is a NIfTI file, a DICOM series folder, or a directory of case folders.
`/path/to/output` is a **raw output destination**, not a registration project. Output is written to
`/path/to/output/<detected_case_id>/` and an existing case directory is reused/rewritten, so use a
fresh destination to preserve a previous run.

Useful flags: `--no-confirm`, `--ap-flip`, `--device cpu`, `--roi-subset ...`, `--skip-combination`,
and `--skip-segmentation`. The last one reuses masks already present in
`/path/to/output/<case_id>/segmentations/` (one NIfTI per TotalSegmentator anatomy name, e.g.
`hip_left.nii.gz`) instead of running inference, and restricts processing to the masks it finds.
Run `--help` for the full list; see [`docs/low_memory.md`](docs/low_memory.md) for large volumes and
CPU-only machines.

## Output structure

```text
<case_id>/
├── cropped_data/              # per-anatomy crops: femur_left|femur_right|hip|lumbar|pelvis
├── landmarks/
│   ├── <case_id>_landmarks.fcsv   # 3D Slicer format, LPS
│   └── <case_id>_debug.json
├── meta_data/
│   ├── <case_id>_CT.nii.gz
│   └── <case_id>_CT_seg.nii.gz
├── models/                    # STL meshes for target anatomy
└── TotalSegmentator_outputs/  # individual_seg/, models/, full-label volume
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/landmark_method.md](docs/landmark_method.md) | Detection method, anatomy-local frames, landmark keys, configs, known limitations |
| [docs/low_memory.md](docs/low_memory.md) | CPU / low-memory runs, proxy volumes, validation checklist |
| [PROVENANCE.md](PROVENANCE.md) | Licence position and the source of every parameter |
| [examples/README.md](examples/README.md) | Synthetic phantom sample |

## Notes

Tested on full-body CT. Partial-field-of-view scans may need adjustment, and job completion is not
evidence of anatomical accuracy — review landmarks before using them for registration.

## Contact

Ping-Cheng Ku (pku1@jh.edu)
