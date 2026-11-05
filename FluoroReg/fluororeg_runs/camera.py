"""The xreg pinhole camera of one X-ray on its raw detector grid (a view's camera in the manifest).

CAM: origin at the X-ray source, +x along increasing detector column, +y along increasing row.
For frame_type origin-at-focal-pt-det-neg-z the detector is the plane z = -source_to_detector_mm
and the focal entries of the intrinsic are negative; origin-at-focal-pt-det-pos-z puts it at
z = +source_to_detector_mm. A point p_CAMWORLD is seen at
[u, v, 1] ~ K @ (CAM_from_CAMWORLD @ p_CAMWORLD)[:3], with (u, v) = (column, row), integer values
at pixel centres and row 0 the first stored image row.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .transforms import as_matrix, apply, inv

FRAME_TYPE_NEG_Z = "origin-at-focal-pt-det-neg-z"
FRAME_TYPE_POS_Z = "origin-at-focal-pt-det-pos-z"


class RawCamera:
    """Projection and detector geometry of a manifest camera (always the raw grid)."""

    def __init__(self, intrinsic, cam_from_camworld, rows: int, cols: int, row_spacing_mm: float,
                 col_spacing_mm: float, source_to_detector_mm: float, frame_type: str):
        self.K = as_matrix(intrinsic, (3, 3))
        self.cam_from_camworld = as_matrix(cam_from_camworld)
        self.rows = int(rows)
        self.cols = int(cols)
        self.row_spacing_mm = float(row_spacing_mm)
        self.col_spacing_mm = float(col_spacing_mm)
        self.source_to_detector_mm = float(source_to_detector_mm)
        if frame_type not in (FRAME_TYPE_NEG_Z, FRAME_TYPE_POS_Z):
            raise ValueError(f"unsupported camera frame_type {frame_type!r}")
        self.frame_type = frame_type

    @classmethod
    def from_manifest(cls, camera: dict) -> "RawCamera":
        return cls(camera["intrinsic"], camera["CAM_from_CAMWORLD"], camera["rows"], camera["cols"],
                   camera["row_spacing_mm"], camera["col_spacing_mm"], camera["source_to_detector_mm"],
                   camera["frame_type"])

    @property
    def camworld_from_cam(self) -> np.ndarray:
        return inv(self.cam_from_camworld)

    @property
    def detector_z(self) -> float:
        """z of the detector plane in CAM."""
        s = self.source_to_detector_mm
        return -s if self.frame_type == FRAME_TYPE_NEG_Z else s

    # ------------------------------------------------------------------ projection
    def project_cam(self, points_cam: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
        """(u, v) of N x 3 CAM points (or one point)."""
        p = np.asarray(points_cam, dtype=np.float64)
        single = p.ndim == 1
        uvw = np.atleast_2d(p) @ self.K.T
        uv = uvw[:, :2] / uvw[:, 2:3]
        return uv[0] if single else uv

    def project(self, points_camworld) -> np.ndarray:
        """(u, v) of N x 3 CAMWORLD points (or one point)."""
        return self.project_cam(apply(self.cam_from_camworld, points_camworld))

    def project_ct(self, points_ct, ct_from_camworld: np.ndarray) -> np.ndarray:
        """(u, v) of CT (LPS) points seen with the registration pose CT_from_CAMWORLD."""
        return self.project(apply(inv(ct_from_camworld), points_ct))

    # ------------------------------------------------------------------ detector plane
    def cam_from_pixel(self) -> np.ndarray:
        """4x4 map from homogeneous pixel index (u, v, k, 1) to the CAM point on the detector plane.

        Pixel (u, v) goes to z * inv(K) @ [u, v, 1] with z the detector plane; k moves 1 mm along
        +z (only used to give a one-slice image a valid third axis).
        """
        z = self.detector_z
        k_inv = inv(self.K)
        m = np.eye(4)
        m[:3, 0] = z * k_inv[:, 0]
        m[:3, 1] = z * k_inv[:, 1]
        m[:3, 2] = [0.0, 0.0, 1.0]
        m[:3, 3] = z * k_inv[:, 2]
        return m

    def camworld_from_pixel(self) -> np.ndarray:
        return self.camworld_from_cam @ self.cam_from_pixel()

    def detector_point_cam(self, uv) -> np.ndarray:
        """CAM point(s) on the detector plane for pixel coordinate(s) (u, v)."""
        uv = np.asarray(uv, dtype=np.float64)
        single = uv.ndim == 1
        uv2 = np.atleast_2d(uv)
        pix = np.column_stack([uv2, np.zeros(len(uv2))])
        out = apply(self.cam_from_pixel(), pix)
        return out[0] if single else out

    def detector_point_camworld(self, uv) -> np.ndarray:
        return apply(self.camworld_from_cam, self.detector_point_cam(uv))

    def pixel_of_detector_point_cam(self, points_cam) -> np.ndarray:
        """(u, v) of CAM point(s) lying on the detector plane (inverse of detector_point_cam)."""
        p = np.asarray(points_cam, dtype=np.float64)
        single = p.ndim == 1
        pix = apply(inv(self.cam_from_pixel()), np.atleast_2d(p))[:, :2]
        return pix[0] if single else pix

    def ray_to_detector_cam(self, points_cam) -> np.ndarray:
        """Central projection of CAM point(s) from the source onto the detector plane, in CAM."""
        p = np.atleast_2d(np.asarray(points_cam, dtype=np.float64))
        out = p * (self.detector_z / p[:, 2:3])
        return out[0] if np.asarray(points_cam).ndim == 1 else out

    def source_camworld(self) -> np.ndarray:
        return apply(self.camworld_from_cam, [0.0, 0.0, 0.0])

    def detector_corners_px(self) -> np.ndarray:
        """Outer corners of the detector in pixel coordinates (pixel centres are integers)."""
        u0, v0 = -0.5, -0.5
        u1, v1 = self.cols - 0.5, self.rows - 0.5
        return np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])

    def frustum_camworld(self) -> np.ndarray:
        """5 x 3 points in CAMWORLD: the source, then the four detector corners (first row first)."""
        corners = self.detector_point_camworld(self.detector_corners_px())
        return np.vstack([self.source_camworld(), corners])
