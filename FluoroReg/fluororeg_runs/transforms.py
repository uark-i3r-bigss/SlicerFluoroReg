"""4x4 homogeneous transform helpers under the xreg run convention.

Every transform is named A_from_B and maps a point expressed in frame B to frame A:
p_A = A_from_B @ p_B and A_from_B @ B_from_C = A_from_C. Matrices are 4x4, row-major,
millimetres. CT space is LPS; Slicer's world is RAS.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# RAS_from_LPS; it is its own inverse (LPS_from_RAS).
RAS_FROM_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
LPS_FROM_RAS = RAS_FROM_LPS


def as_matrix(value: Sequence[Sequence[float]] | np.ndarray, shape: tuple = (4, 4)) -> np.ndarray:
    """A float64 copy of a nested list or array; raises ValueError on a wrong shape or non-finite entries."""
    m = np.array(value, dtype=np.float64)
    if m.shape != shape:
        raise ValueError(f"expected a {shape[0]}x{shape[1]} matrix, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix has non-finite entries")
    return m


def inv(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def compose(*mats: np.ndarray) -> np.ndarray:
    """compose(A_from_B, B_from_C, C_from_D) = A_from_D."""
    out = np.eye(4)
    for m in mats:
        out = out @ np.asarray(m, dtype=np.float64)
    return out


def apply(m: np.ndarray, points: Iterable[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Transforms N x 3 points (or one 3-vector) by a 4x4 matrix; returns the same shape."""
    p = np.asarray(points, dtype=np.float64)
    single = p.ndim == 1
    p2 = np.atleast_2d(p)
    out = p2 @ np.asarray(m, dtype=np.float64)[:3, :3].T + np.asarray(m, dtype=np.float64)[:3, 3]
    return out[0] if single else out


def lps_to_ras(m_lps: np.ndarray) -> np.ndarray:
    """The same geometric map written between RAS frames: RAS_from_LPS @ M @ LPS_from_RAS."""
    return RAS_FROM_LPS @ np.asarray(m_lps, dtype=np.float64) @ LPS_FROM_RAS


def slicer_to_parent_of_itk_file(file_matrix: np.ndarray) -> np.ndarray:
    """To-parent (RAS) matrix of the Slicer node created by loading an ITK transform file holding M.

    ITK files are resampling (from-parent) transforms in LPS, so Slicer shows the node's to-parent
    matrix as RAS_from_LPS @ inv(M) @ RAS_from_LPS. For M = CT_from_CAMWORLD the node moves CT
    content into the camera world.
    """
    return lps_to_ras(inv(file_matrix))


def ras_from_camworld(ct_from_camworld: np.ndarray) -> np.ndarray:
    """C-arm placement in Slicer's RAS world: RAS_from_LPS @ CT_from_CAMWORLD."""
    return RAS_FROM_LPS @ np.asarray(ct_from_camworld, dtype=np.float64)


def primary_from_target(p_primary: np.ndarray, p_target: np.ndarray) -> np.ndarray:
    """CT[primary]_from_CT[t] = P_primary @ inv(P_t) for two CT_from_CAMWORLD poses of one view.

    A point of target t in CT, seen by the camera at pose P_t, lands at this CT[primary] point
    when the camera is placed with the primary target's pose P_primary.
    """
    return np.asarray(p_primary, dtype=np.float64) @ inv(p_target)


def is_rigid(m: np.ndarray, tol: float = 1e-4) -> bool:
    """True for a homogeneous rigid transform (orthonormal rotation, det +1, last row 0 0 0 1)."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (4, 4) or not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0], atol=tol):
        return False
    r = m[:3, :3]
    return bool(np.allclose(r @ r.T, np.eye(3), atol=tol) and abs(np.linalg.det(r) - 1.0) < tol)


def rotation_angle_deg(r: np.ndarray) -> float:
    """Rotation angle of a 3x3 rotation (or the rotation block of a 4x4) in degrees."""
    r = np.asarray(r, dtype=np.float64)[:3, :3]
    c = (np.trace(r) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def pose_difference(a: np.ndarray, b: np.ndarray) -> dict:
    """Entry-wise and geometric differences between two 4x4 transforms.

    translation_mm: |t_a - t_b|; rotation_max_abs: max entry difference of the rotation blocks;
    rotation_deg: angle of R_a^T R_b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return {
        "translation_mm": float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        "rotation_max_abs": float(np.max(np.abs(a[:3, :3] - b[:3, :3]))),
        "rotation_deg": rotation_angle_deg(a[:3, :3].T @ b[:3, :3]),
    }
