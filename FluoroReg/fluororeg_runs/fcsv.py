"""Slicer markups fiducial (.fcsv) reader that honours the file's coordinate system.

Slicer 4.11+ writes "# CoordinateSystem = LPS" or "= RAS"; files of 4.10 and earlier write
"= 0" (RAS; 1 is LPS). A file without the header is rejected: guessing the frame of landmarks
silently mirrors them.
"""
from __future__ import annotations

import csv
import io
from collections import OrderedDict
from typing import Dict

import numpy as np

_FRAMES = {"RAS": "RAS", "0": "RAS", "LPS": "LPS", "1": "LPS"}
_DEFAULT_COLUMNS = ["id", "x", "y", "z", "ow", "ox", "oy", "oz", "vis", "sel", "lock", "label", "desc",
                    "associatedNodeID"]


class FcsvError(ValueError):
    pass


def read_fcsv_text(text: str, frame: str = "LPS") -> "Dict[str, np.ndarray]":
    """Named points of an FCSV document expressed in `frame` ("LPS" or "RAS"), in file order."""
    if frame not in ("LPS", "RAS"):
        raise ValueError(f"frame must be LPS or RAS, got {frame!r}")

    file_frame = None
    columns = list(_DEFAULT_COLUMNS)
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            key, sep, value = stripped[1:].partition("=")
            if not sep:
                continue
            key = key.strip().lower()
            value = value.strip()
            if key == "coordinatesystem":
                if value not in _FRAMES:
                    raise FcsvError(f"unsupported FCSV CoordinateSystem {value!r}")
                file_frame = _FRAMES[value]
            elif key == "columns":
                columns = [c.strip() for c in value.split(",")]
            continue
        rows.append(stripped)

    if file_frame is None:
        raise FcsvError("FCSV has no '# CoordinateSystem' header")
    for needed in ("x", "y", "z", "label"):
        if needed not in columns:
            raise FcsvError(f"FCSV columns lack {needed!r}: {columns}")
    ix, iy, iz, il = (columns.index(c) for c in ("x", "y", "z", "label"))

    flip = np.array([-1.0, -1.0, 1.0]) if file_frame != frame else np.ones(3)
    points: "Dict[str, np.ndarray]" = OrderedDict()
    for rec in csv.reader(io.StringIO("\n".join(rows))):
        name = rec[il]
        if name in points:
            raise FcsvError(f"duplicate FCSV point name {name!r}")
        points[name] = np.array([float(rec[ix]), float(rec[iy]), float(rec[iz])]) * flip
    return points


def read_fcsv(path, frame: str = "LPS") -> "Dict[str, np.ndarray]":
    with open(path, "r", encoding="utf-8") as f:
        return read_fcsv_text(f.read(), frame)
