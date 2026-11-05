"""Training geometry and pose prior derived from the case's own data.

The model's detector must reproduce the (cropped) real detector: xvr's ``predict_pose`` resamples
an X-ray to the model's source-to-detector distance and field of view, centre-crops it square and
resizes it to ``height`` pixels. With ``side = min(rows, cols) - crop``, the matching model pixel is
``delx = side * spacing / height``.

The prior samples xvr's pose parameters (ZXY Euler angles, camera-frame translation, DiffDRR "AP"
orientation). With zero rotation the central ray runs from the source at ``(tx, ty, tz)`` along -y,
so ``tx``/``tz`` place the central ray and ``ty`` is the source-to-isocentre distance. They are
centred on the centroid of the prior's segmentation label, in DiffDRR world coordinates.
alpha near 0 puts the source on the anterior side of the CT, alpha near 180 on the posterior side.
Which side is right depends on the set-up and must be checked per system: the pelvis is nearly
left/right symmetric, so a prior on the wrong side still yields plausible-looking poses with the
patient's left and right exchanged. The default (anterior, from a 768² Cios set-up's registered poses,
alpha 9-13 deg) scored clearly worse on 142 views of a Cios Fusion / Loop-X set-up than a posterior prior
(alpha 135-225, median network gncc 0.23 vs 0.37), which also matches a mobile C-arm tube under the table.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import nibabel as nib
import numpy as np

from .bridge import Bridge
from .config import PoseConfig
from .xreg_io import XregCamera

MODEL_HEIGHT = 128


@dataclass(frozen=True)
class TrainGeometry:
    sdd: float
    height: int
    delx: float
    rows: int
    cols: int
    spacing: float
    crop_px: int

    def as_dict(self) -> dict:
        return asdict(self)


def training_geometry(cam: XregCamera, crop_px: int, height: int = MODEL_HEIGHT) -> TrainGeometry:
    cam.require_centered()
    if not np.isclose(cam.row_spacing, cam.col_spacing):
        raise ValueError("non-square detector pixels are not supported")
    side = min(cam.rows, cam.cols) - crop_px
    if side <= height:
        raise ValueError(f"cropped detector ({side} px) is not larger than the model input ({height} px)")
    return TrainGeometry(sdd=cam.sdd, height=height, delx=side * cam.col_spacing / height,
                         rows=cam.rows, cols=cam.cols, spacing=cam.col_spacing, crop_px=crop_px)


def label_centroid_lps(seg_path, label: int) -> np.ndarray:
    seg = nib.load(str(seg_path))
    data = np.asarray(seg.dataobj)
    ijk = np.argwhere(data == label)
    if len(ijk) == 0:
        raise ValueError(f"segmentation {seg_path} has no voxels with label {label}")
    ras = seg.affine @ np.r_[ijk.mean(0), 1.0]
    return ras[:3] * np.array([-1.0, -1.0, 1.0])


def pose_ranges(cfg: PoseConfig, bridge: Bridge) -> dict:
    """xvr Trainer keyword arguments for the pose distribution."""
    c_lps = label_centroid_lps(cfg.seg, cfg.prior.center_label)
    cx, _, cz = bridge.lps_to_world(c_lps[None])[0]
    p = cfg.prior
    return {
        "alphamin": p.alpha_deg[0], "alphamax": p.alpha_deg[1],
        "betamin": p.beta_deg[0], "betamax": p.beta_deg[1],
        "gammamin": p.gamma_deg[0], "gammamax": p.gamma_deg[1],
        "txmin": float(cx - p.lateral_mm), "txmax": float(cx + p.lateral_mm),
        "tymin": p.source_to_center_mm[0], "tymax": p.source_to_center_mm[1],
        "tzmin": float(cz - p.axial_mm), "tzmax": float(cz + p.axial_mm),
    }
