# Low-Memory TotalSegmentator Runs

This guide covers CPU and low-memory execution for large CT volumes. The
pipeline supports TotalSegmentator's lower-memory switches through the CLI.

## When to use this

Use low-memory mode when:

- the machine has limited RAM or swap
- `TotalSegmentator` is killed with `SIGKILL` or exit code 137
- the input has very fine spacing or an unusually large slice count
- you only need the pelvis, femur, lumbar, and sacral landmarks used by this
  repository

`SIGKILL` usually means the operating system killed the process for memory
pressure. Retrying the exact same command is unlikely to help.

## CLI usage

For a CPU run that limits TotalSegmentator to this repository's target anatomy:

```bash
conda activate "$HOME/.local/share/fluororeg/annotation-env"
python process_medical_scans.py /path/to/input.nii.gz /path/to/output \
  --device cpu \
  --fastest \
  --body-seg \
  --force-split \
  --roi-subset \
    hip_left hip_right femur_left femur_right \
    vertebrae_L1 vertebrae_L2 vertebrae_L3 vertebrae_L4 vertebrae_L5 \
    vertebrae_S1 sacrum
```

The low-memory flags are:

- `--device cpu`: force CPU inference instead of GPU auto-detection
- `--fastest`: use TotalSegmentator's 6 mm model for CT input
- `--body-seg`: run a rough body crop before the target segmentation
- `--force-split`: process the image in chunks
- `--roi-subset`: restrict work to the masks needed downstream

`--fastest` is lower resolution than `--fast`; inspect the resulting masks and
landmarks before using them for measurement-sensitive work.

## Proxy volume fallback

Some volumes are still too large for a laptop even with all low-memory
TotalSegmentator flags. For example, a `512 x 512 x 7837` CT with `0.1 mm`
through-plane spacing has more than 2 billion voxels. In that case, create a
downsampled proxy NIfTI for segmentation while preserving physical coordinates
through the affine.

Example proxy creation with `2 x 2 x 20` strides:

```bash
conda activate "$HOME/.local/share/fluororeg/annotation-env"
python - <<'PY'
from pathlib import Path
import nibabel as nib
import numpy as np

src = Path("/path/to/input.nii.gz")
out = Path("/path/to/input_totalseg_proxy_stride2x2x20.nii.gz")
strides = np.array([2, 2, 20], dtype=int)

img = nib.load(str(src), mmap=True)
data = np.asanyarray(img.dataobj[::strides[0], ::strides[1], ::strides[2]])
if data.dtype != np.int16:
    data = np.rint(data)
    data = np.clip(data, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
else:
    data = np.ascontiguousarray(data)

affine = img.affine.copy()
affine[:3, 0] *= strides[0]
affine[:3, 1] *= strides[1]
affine[:3, 2] *= strides[2]

header = img.header.copy()
header.set_data_shape(data.shape)
zooms = tuple(float(z) * int(s) for z, s in zip(img.header.get_zooms()[:3], strides))
header.set_zooms(zooms)
header.set_data_dtype(np.int16)

nib.save(nib.Nifti1Image(data, affine, header), str(out))
print(out)
print(data.shape, zooms)
PY
```

Then run the pipeline on the proxy path with the same low-memory settings. The
STL and landmark stages apply the NIfTI affine, so coordinates remain in the
proxy volume's physical coordinate system.

## Validation checklist

After a low-memory run, check:

- the CLI completed successfully
- combined segmentation exists under `meta_data/*_CT_seg.nii.gz`
- target STLs exist under `models/`
- landmark FCSV exists under `landmarks/`
- CLI output and generated artifacts contain the expected anatomies
- warnings do not hide missing anatomy needed for your task

If lumbar vertebrae are missing, the pipeline can still produce pelvis, femur,
and sacral landmarks, but L1-L5 landmarks will not be available.
