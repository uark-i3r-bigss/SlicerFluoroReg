"""Readers and writers for the xreg file formats this package consumes and produces.

Everything here is plain numpy/h5py so it can be tested without a GPU.

- Projection H5 (``xregH5ProjDataIO``): one projection per file, ``proj-000/{img,cam,landmarks}``.
- ITK affine transform H5 (``AffineTransform_float_3_3``), the layout xreg reads for
  ``targets[].init`` and writes for every registered pose.
- Pose directory: ``view-<id>_target-<lower key>_CT_from_CAMWORLD.h5`` per projection id, the only
  name xreg's ``init.method: "run"`` reads. The matrix is ``CT_from_CAMWORLD`` (A_from_B:
  ``p_CT = CT_from_CAMWORLD @ p_CAMWORLD``, CT in LPS mm), i.e. xreg's ``cam_to_vol``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

DET_NEG_Z = "origin-at-focal-pt-det-neg-z"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


POSE_FILE_PATTERN = "view-<id>_target-<key>_CT_from_CAMWORLD.h5"


def pose_file_name(proj_id: str, key: str) -> str:
    """Pose-directory file name: ``view-<id>_target-<lower key>_CT_from_CAMWORLD.h5``."""
    return f"view-{proj_id}_target-{key.lower()}_CT_from_CAMWORLD.h5"


# ----------------------------------------------------------------------------- ITK transforms


def read_itk_xform(path: str | Path) -> np.ndarray:
    """Read an ITK affine transform H5 as a 4x4 matrix (honours a non-zero ITK centre)."""
    with h5py.File(path, "r") as f:
        g = f["TransformGroup/0"]
        p = np.asarray(g["TransformParameters"][()], dtype=np.float64)
        c = np.asarray(g["TransformFixedParameters"][()], dtype=np.float64)
    if p.size != 12:
        raise ValueError(f"{path}: expected a 3D affine (12 parameters), found {p.size}")
    A = p[:9].reshape(3, 3)
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = p[9:12] + c - A @ c
    return M


def write_itk_xform(path: str | Path, M: np.ndarray, os_version: str = "bigss-pose-estimation") -> None:
    """Write a 4x4 rigid matrix in the layout xreg writes (float affine, zero centre).

    HDF5 structures are limited to v1.12 so xreg's ITK 5.3 / HDF5 1.12 reader opens the file.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4) or not np.allclose(M[3], [0, 0, 0, 1]):
        raise ValueError("expected a 4x4 homogeneous matrix")
    params = np.r_[M[:3, :3].reshape(-1), M[:3, 3]].astype(np.float32)
    str_dt = h5py.special_dtype(vlen=bytes)

    def text(v: str):
        return np.array([v.encode()], dtype=object)

    with h5py.File(path, "w", libver=("earliest", "v112")) as f:
        f.create_dataset("HDFVersion", data=text("HDF5 library version: " + h5py.version.hdf5_version), dtype=str_dt)
        f.create_dataset("ITKVersion", data=text("5.3.0"), dtype=str_dt)
        f.create_dataset("OSName", data=text("Linux"), dtype=str_dt)
        f.create_dataset("OSVersion", data=text(os_version), dtype=str_dt)
        g = f.create_group("TransformGroup/0")
        g.create_dataset("TransformFixedParameters", data=np.zeros(3, dtype=np.float64))
        g.create_dataset("TransformParameters", data=params)
        g.create_dataset("TransformType", data=text("AffineTransform_float_3_3"), dtype=str_dt)


# ----------------------------------------------------------------------------- projections


@dataclass(frozen=True)
class XregCamera:
    """The pinhole camera stored in a projection H5 (``origin-at-focal-pt-det-neg-z``)."""

    intrinsic: np.ndarray  # 3x3; negative focal entries for det-neg-z
    extrinsic: np.ndarray  # 4x4 camera-world -> camera
    rows: int
    cols: int
    row_spacing: float
    col_spacing: float

    @property
    def sdd(self) -> float:
        return float(abs(self.intrinsic[0, 0]) * self.col_spacing)

    @property
    def principal_point(self) -> tuple[float, float]:
        """(col, row) of the principal point in continuous pixel indices."""
        return float(self.intrinsic[0, 2]), float(self.intrinsic[1, 2])

    def require_centered(self, tol_px: float = 0.51) -> None:
        cx, cy = self.principal_point
        if abs(cx - (self.cols - 1) / 2) > tol_px or abs(cy - (self.rows - 1) / 2) > tol_px:
            raise ValueError(
                f"principal point ({cx:.2f}, {cy:.2f}) is not the detector centre "
                f"({(self.cols - 1) / 2:.1f}, {(self.rows - 1) / 2:.1f}); only centred "
                "native-grid cameras are supported")

    def project(self, cam_to_vol: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """(col, row) continuous pixel indices of volume points, exactly as xreg computes them."""
        vol_to_cam = self.extrinsic @ np.linalg.inv(cam_to_vol)
        p = np.c_[pts, np.ones(len(pts))]
        idx = self.intrinsic @ (vol_to_cam @ p.T)[:3]
        return (idx[:2] / idx[2]).T


@dataclass(frozen=True)
class Projection:
    path: Path
    image: np.ndarray  # (rows, cols) float32, pixels as stored (native grid)
    camera: XregCamera
    landmarks: dict[str, np.ndarray]  # name -> (col, row)


def read_projection(path: str | Path, with_image: bool = True) -> Projection:
    path = Path(path)
    with h5py.File(path, "r") as f:
        n = int(f["num-projs"][()])
        if n != 1:
            raise ValueError(f"{path}: expected exactly one projection, found {n}")
        g = f["proj-000/cam"]
        frame = g["cam-coord-frame-type"][()]
        frame = frame.decode() if isinstance(frame, bytes) else str(frame)
        if frame != DET_NEG_Z:
            raise ValueError(f"{path}: camera frame {frame!r} is not {DET_NEG_Z!r}")
        cam = XregCamera(
            intrinsic=np.asarray(g["intrinsic"][()], dtype=np.float64),
            extrinsic=np.asarray(g["extrinsic"][()], dtype=np.float64),
            rows=int(g["num-rows"][()]),
            cols=int(g["num-cols"][()]),
            row_spacing=float(g["row-spacing"][()]),
            col_spacing=float(g["col-spacing"][()]),
        )
        image = np.asarray(f["proj-000/img/pixels"][()], dtype=np.float32) if with_image else np.empty(0)
        lands = {}
        if "proj-000/landmarks" in f:
            for k, d in f["proj-000/landmarks"].items():
                lands[k] = np.asarray(d[()], dtype=np.float64).reshape(-1)
    if with_image and image.shape != (cam.rows, cam.cols):
        raise ValueError(f"{path}: image {image.shape} does not match camera {cam.rows}x{cam.cols}")
    return Projection(path, image, cam, lands)


def read_fcsv(path: str | Path) -> dict[str, np.ndarray]:
    """Slicer FCSV → {label: LPS point}. The file must declare its coordinate system."""
    lines = Path(path).read_text().splitlines()
    header = [ln for ln in lines if ln.startswith("#")]
    ras = any("CoordinateSystem = RAS" in ln or "CoordinateSystem = 0" in ln for ln in header)
    lps = any("CoordinateSystem = LPS" in ln or "CoordinateSystem = 1" in ln for ln in header)
    if ras == lps:
        raise ValueError(f"{path}: cannot determine the coordinate system")
    out = {}
    for ln in lines:
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split(",")
        p = np.array([float(f[1]), float(f[2]), float(f[3])])
        out[f[11]] = p * np.array([-1.0, -1.0, 1.0]) if ras else p
    return out
