"""2D landmark name mapping and landmark reprojection residuals.

views[].landmarks_2d.annotated holds points under the names the annotator gave them, on the raw
grid. registration_names maps each annotated name to the name the registration used (the side
swap when swap_2d_sides is true) or null when the swap dropped it. xreg's residual for a target
projects every 3D landmark whose name starts with the target's landmark_prefix and has a 2D point
under that registration name.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Mapping, Optional

import numpy as np

from .camera import RawCamera


class LandmarkMapError(ValueError):
    pass


def invert_registration_names(registration_names: Mapping[str, Optional[str]]) -> "Dict[str, str]":
    """registration name -> annotated name. Dropped (null) entries are skipped; two annotated
    names mapping to one registration name are an error (the swap should have dropped them)."""
    out: "Dict[str, str]" = OrderedDict()
    for annotated, registered in registration_names.items():
        if registered is None:
            continue
        if registered in out:
            raise LandmarkMapError(
                f"annotated names {out[registered]!r} and {annotated!r} both map to {registered!r}")
        out[registered] = annotated
    return out


def registration_points_2d(landmarks_2d: Mapping) -> "Dict[str, np.ndarray]":
    """The 2D points (raw grid) under the names the registration used."""
    annotated = landmarks_2d["annotated"]
    names = landmarks_2d.get("registration_names") or {n: n for n in annotated}
    missing = [n for n in names if n not in annotated]
    if missing:
        raise LandmarkMapError(f"registration_names lists names without an annotated point: {missing}")
    out: "Dict[str, np.ndarray]" = OrderedDict()
    for registered, annotated_name in invert_registration_names(names).items():
        out[registered] = np.asarray(annotated[annotated_name], dtype=np.float64)
    return out


def landmark_residuals_px(camera: RawCamera, ct_from_camworld: np.ndarray,
                          landmarks_3d_lps: Mapping[str, np.ndarray],
                          points_2d: Mapping[str, np.ndarray], prefix: str) -> "Dict[str, float]":
    """Pixel distance between each projected 3D landmark and its 2D point, xreg's selection rule:
    2D names starting with `prefix` that also have a 3D landmark."""
    names = [n for n in points_2d if n.startswith(prefix) and n in landmarks_3d_lps]
    if not names:
        return OrderedDict()
    uv = camera.project_ct(np.array([landmarks_3d_lps[n] for n in names]), ct_from_camworld)
    return OrderedDict((n, float(np.hypot(*(uv[i] - points_2d[n])))) for i, n in enumerate(names))
