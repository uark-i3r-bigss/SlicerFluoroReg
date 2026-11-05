# Provenance and licence position

## Grant and scope

The BIGSS-owned implementation and documentation are distributed under [LICENSE](LICENSE)
(MIT), Copyright (c) 2026 Ping-Cheng Ku and contributors. This grant does not cover external
software, pretrained models, third-party reference implementations or medical data. It does not
create redistribution permission for example scans/archives or derived patient-specific outputs.
Sample data rights remain subject to separate review; synthetic examples are not clinical data.

This repository implements published geometric methods. Every algorithm here was re-derived
from the **papers**; no source code was copied from any of the reference implementations
below, and none of them is a dependency.

## Reference implementations: read, never copied

| Project | Licence as published | Why no code was taken |
|---|---|---|
| RWTHmediTEC/PelvicLandmarkIdentification (Fischer et al., *Sci Rep* 9:13322, 2019) | EUPL-1.2, declared by a README badge; the repository carries no `LICENSE` file | EUPL-1.2 is copyleft and reaches derivative works. The paper is CC-BY, so the *method* is free to re-implement. |
| RWTHmediTEC/FemoralCoordinateSystem (Fischer et al., *Sci Rep* 10:20859, 2020) | EUPL-1.2, badge only | As above. |
| modenaxe/msk-STAPLE, including GIBOC-core (Modenese & Renault, *J Biomech* 116:110186, 2021) | **CC BY-NC 4.0** (`LICENSE.txt`) | Non-commercial: unusable in a product, in any amount. |
| naamiinepal/xrayto3D-morphometry | **No licence grant**: the only licence-like statement is a trove classifier in `setup.py`; the repository has no `LICENSE` file | Absent a grant, default copyright applies. It is also itself a port of the EUPL-1.2 MATLAB above. |

Those four projects were read as *documentation of published methods* during design. The
checked-in implementation is independent: it is written against the papers' descriptions and
against the numerical behaviour we measured on our own data.

## Methods cited

* Fischer, M. C. M. et al. *A robust method for automatic identification of landmarks on
  surface models of the pelvis.* Sci Rep 9, 13322 (2019). CC-BY. Source of the mesh
  conditioning target (decimation bounded to a 0.05 mm Hausdorff distance) and of the
  pose-invariance acceptance protocol (100 random rigid transforms per case).
* Fischer, M. C. M. et al. *A robust method for automatic identification of femoral
  landmarks, axes, planes and bone coordinate systems using surface models.* Sci Rep 10,
  20859 (2020). Source of the published manual dispersion attached to the trochanter
  landmarks (6.7 mm superior greater trochanter, 6.1 mm lesser trochanter, against 0.4 mm
  for the femoral head centre) and of the fact that neither is used by any coordinate system.
* Cerveri, P. et al. *Automated method for computing the morphological and clinical
  parameters of the proximal femur using heuristic modeling techniques.* Ann Biomed Eng 38,
  1752 (2010). Source of the proximal-femur-only approach: least-squares sphere fitting for
  the head centre on an isolated head region.
* Wu, G. et al. *ISB recommendation on definitions of joint coordinate systems ... Part I.*
  J Biomech 35, 543 (2002). The standard the pelvis and femur frames are compared against.
* Wasserthal, J. et al. *TotalSegmentator.* Radiology: AI 5, e230024 (2023). The
  segmentation front end; structure names follow its vocabulary.

## Third-party runtime dependencies

Dependencies keep their own distribution licenses and notices. This includes `numpy`, `scipy`,
`nibabel`, `pyvista`/VTK, `numpy-stl`, `connected-components-3d`, `pyzmq`, PyYAML,
TotalSegmentator, PyTorch and nnU-Net and their transitive dependencies. The implementation's
MIT grant must not be read as a blanket assertion that an installed environment or every model
has permissive terms. Preserve each installed distribution's original notices.

### TotalSegmentator source and weights

Upstream source [LICENSE](https://github.com/wasserth/TotalSegmentator/blob/master/LICENSE)
is Apache-2.0. The upstream
[task list](https://github.com/wasserth/TotalSegmentator#subtasks), reviewed 2026-11-04, explicitly
lists the default `total` (CT) and `total_mr` (MR) tasks under Apache-2.0, but also lists separately
licensed/restricted tasks. Check the exact installed version, selected task and downloaded weights;
do not generalize the default-task statement to all optional models or the training datasets.
Weights are downloaded by TotalSegmentator, not licensed anew or bundled by this implementation.

Reference datasets, user scans, segmentations and generated landmarks require their own consent,
governance and redistribution review. Upstream dataset links and code licenses do not provide
blanket permission to republish medical data.

## Parameters and their sources

Constants that affect a coordinate are annotated in the code with their origin. The three
classes are:

1. **From a paper** - e.g. the 0.05 mm decimation bound (Fischer 2019) and the 20-iteration,
   0.1 pass-band Taubin smoothing (VTK's documented defaults for the windowed-sinc filter).
2. **Measured on this corpus** - e.g. the voxel spacings that motivate isotropic resampling
   (1.39-1.56 mm in plane by 3.0 mm slices over the four cases available on 2026-09-22) and
   the pose-invariance spreads recorded for the lumbar landmarks.
3. **Ours, stated as such** - e.g. the femoral-head acceptance gates (RMS, radius interval,
   sphere support, angular coverage, bootstrap spread). These are engineering thresholds,
   not published values, and are named in the code as such.

Anything still carrying an unexplained number is a defect; see
`docs/landmark_method.md` for the definitions and the known limitations.
