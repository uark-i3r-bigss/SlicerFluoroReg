"""Frame bridge between xreg camera-to-volume poses and DiffDRR/xvr camera poses.

xreg (projection H5, ``origin-at-focal-pt-det-neg-z``):
  camera frame   source at the origin, detector plane at z = -sdd; pixel (row i, col j) at
                 (s*(j - cx), s*(i - cy), -sdd) with (cx, cy) the principal point.
  camera world   ``extrinsic`` maps camera world -> camera frame.
  pose X         ``cam_to_vol``: camera world -> CT physical LPS (mm).

DiffDRR / xvr (``diffdrr.data.read(..., center_volume=True)``):
  world          CT physical RAS with the volume isocentre (torchio ``get_center``) at the origin.
  camera frame   source at the origin, detector centre at (0, 0, +sdd); pixel (i, j) at
                 (s*(j - cx), -s*(i - cy), sdd) when ``reverse_x_axis`` is False.
  pose P         world = P @ R @ q for a camera point q, R = the orientation reorientation.

With C = diag(1, -1, -1) (DiffDRR camera -> xreg camera, a proper rotation, its own inverse)
and W = T(-iso_ras) @ diag(-1, -1, 1) (LPS -> centred RAS):

    P = W @ X @ E^-1 @ C @ R^-1          X = W^-1 @ P @ R @ C @ E

Verified on a real 768² projection: all 589,824 detector pixel positions agree within 1e-4 mm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

C_DIFFDRR_TO_XREG = np.diag([1.0, -1.0, -1.0, 1.0])
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

REORIENT = {
    "AP": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
    "PA": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float64),
}


@dataclass(frozen=True)
class Bridge:
    iso_ras: np.ndarray        # volume isocentre, RAS mm, as torchio computes it
    orientation: str = "AP"

    @classmethod
    def from_ct(cls, ct_path: str | Path, orientation: str = "AP") -> "Bridge":
        from torchio import ScalarImage

        iso = np.asarray(ScalarImage(str(ct_path)).get_center(), dtype=np.float64)
        return cls(iso_ras=iso, orientation=orientation)

    @property
    def W(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = -self.iso_ras
        return T @ LPS_TO_RAS

    @property
    def R(self) -> np.ndarray:
        if self.orientation not in REORIENT:
            raise ValueError(f"unsupported orientation {self.orientation!r}")
        return REORIENT[self.orientation]

    def xreg_to_diffdrr(self, cam_to_vol: np.ndarray, extrinsic: np.ndarray | None = None) -> np.ndarray:
        E = np.eye(4) if extrinsic is None else extrinsic
        return self.W @ cam_to_vol @ np.linalg.inv(E) @ C_DIFFDRR_TO_XREG @ np.linalg.inv(self.R)

    def diffdrr_to_xreg(self, pose: np.ndarray, extrinsic: np.ndarray | None = None) -> np.ndarray:
        E = np.eye(4) if extrinsic is None else extrinsic
        return np.linalg.inv(self.W) @ pose @ self.R @ C_DIFFDRR_TO_XREG @ E

    def lps_to_world(self, pts_lps: np.ndarray) -> np.ndarray:
        p = np.c_[pts_lps, np.ones(len(pts_lps))]
        return (self.W @ p.T).T[:, :3]


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def xvr_parameters(pose: np.ndarray) -> dict:
    """xvr's pose parameterisation (ZXY Euler degrees + camera-frame translation) of a DiffDRR pose."""
    import torch
    from diffdrr.pose import RigidTransform

    rot, xyz = RigidTransform(torch.as_tensor(pose, dtype=torch.float64)[None]).convert("euler_angles", "ZXY")
    rot = torch.rad2deg(rot)[0].tolist()
    return {"alpha": rot[0], "beta": rot[1], "gamma": rot[2],
            "tx": float(xyz[0, 0]), "ty": float(xyz[0, 1]), "tz": float(xyz[0, 2])}
