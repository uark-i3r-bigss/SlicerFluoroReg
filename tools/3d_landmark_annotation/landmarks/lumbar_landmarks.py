"""Lumbar and sacral landmarks.

The geometric constructions here are the original ones and are known to be pose dependent:
the process tips are extrema along the **scanner** axes, not along a vertebra-local frame, and
``LUM_*_COM`` is the mean of the mesh vertices rather than the ISB vertebral-body centroid.
The pose-invariance harness measures the consequence on the phantom - the tips move by 99 to
131 mm under random rigid re-posing (tools/run_invariance_baseline.py, 100 transforms) - and
replacing the constructions is Phase P2.

What this module does now is stop hiding the situation: every landmark carries a status
record with its provenance and a ``pose_dependent`` flag, and a level that cannot be measured
abstains instead of returning an arbitrary vertex.
"""

import os
from pathlib import Path

import numpy as np
from stl import mesh

from . import status
from .model_lookup import find_model_by_name

LUMBAR_LEVELS = ("L1", "L2", "L3", "L4", "L5")

#: Measured on the analytic phantom with 100 random rigid transforms, 2026-09-22.
POSE_SPREAD_MM = {"SP": 99.0, "TP": 121.0, "SAC_A": 127.0, "COM": 0.0002}


def detect_vertebrae_landmarks(vertebra_vertices, vertebra_name):
    """Centroid plus scanner-axis process extrema for one vertebra."""
    if vertebra_vertices is None or len(vertebra_vertices) == 0:
        return {}

    return {
        f"LUM_{vertebra_name}_COM": vertebra_vertices.mean(axis=0),
        f"LUM_{vertebra_name}_SP": vertebra_vertices[np.argmax(vertebra_vertices[:, 1])],
        f"LUM_{vertebra_name}_TP_L": vertebra_vertices[np.argmax(vertebra_vertices[:, 0])],
        f"LUM_{vertebra_name}_TP_R": vertebra_vertices[np.argmin(vertebra_vertices[:, 0])],
    }


def detect_sacral_apex(sacrum_vertices):
    if sacrum_vertices is None or len(sacrum_vertices) == 0:
        return None
    return sacrum_vertices[np.argmin(sacrum_vertices[:, 2])]


def _record(landmark_id, position, kind):
    """Status record carrying the construction's known pose dependence."""
    provenance = "mesh_vertex_mean" if kind == "COM" else "scanner_axis_extremum"
    metrics = {
        "pose_dependent": kind != "COM",
        "invariance_spread_mm": POSE_SPREAD_MM[kind],
        "vertex_density_weighted": kind == "COM",
    }
    confidence = 0.5 if kind == "COM" else 0.2
    return status.ok(landmark_id, position, confidence=confidence, provenance=provenance,
                     metrics=metrics)


def _expected_ids():
    ids = []
    for level in LUMBAR_LEVELS:
        ids.extend([f"LUM_{level}_COM", f"LUM_{level}_SP",
                    f"LUM_{level}_TP_L", f"LUM_{level}_TP_R"])
    ids.append("LUM_SAC_A")
    return ids


def _kind_of(landmark_id):
    if landmark_id.endswith("_COM"):
        return "COM"
    if landmark_id.endswith("_SP"):
        return "SP"
    if landmark_id == "LUM_SAC_A":
        return "SAC_A"
    return "TP"


def _load_vertices(path):
    body = mesh.Mesh.from_file(path)
    return np.unique(body.vectors.reshape(-1, 3), axis=0)


def generate_lumbar_landmarks(stl_dir, with_records=False):
    """Detect the lumbar/sacral landmarks available in ``stl_dir``.

    Returns the landmark map; with ``with_records=True`` also returns a status record for
    every landmark this stage owns, including the levels that were absent.
    """
    landmarks = {}
    records = {landmark_id: status.not_attempted(landmark_id, status.Reason.MISSING_INPUT,
                                                 provenance="model_lookup")
               for landmark_id in _expected_ids()}

    stl_dir = Path(stl_dir)
    if stl_dir.name != "models":
        for candidate in (stl_dir / "TotalSegmentator_outputs" / "models",
                          stl_dir.parent / "TotalSegmentator_outputs" / "models"):
            if candidate.exists():
                stl_dir = candidate
                break

    for level in LUMBAR_LEVELS:
        path = find_model_by_name(str(stl_dir), f"*vertebrae_{level}.stl")
        if not path:
            continue
        print(f"  Processing {level} vertebra...")
        vertices = _load_vertices(path)
        level_landmarks = detect_vertebrae_landmarks(vertices, level)
        if not level_landmarks:
            for landmark_id in [k for k in records if k.startswith(f"LUM_{level}_")]:
                records[landmark_id] = status.abstained(
                    landmark_id, status.Reason.EMPTY_REGION, provenance="mesh_load"
                )
            continue
        landmarks.update(level_landmarks)
        for landmark_id, position in level_landmarks.items():
            records[landmark_id] = _record(landmark_id, position, _kind_of(landmark_id))

    sacrum_path = find_model_by_name(str(stl_dir), "*sacrum.stl")
    if sacrum_path:
        print("  Processing sacrum...")
        apex = detect_sacral_apex(_load_vertices(sacrum_path))
        if apex is None:
            records["LUM_SAC_A"] = status.abstained(
                "LUM_SAC_A", status.Reason.EMPTY_REGION, provenance="mesh_load"
            )
        else:
            landmarks["LUM_SAC_A"] = apex
            records["LUM_SAC_A"] = _record("LUM_SAC_A", apex, "SAC_A")

    print(f"  Detected {len(landmarks)} lumbar/sacral landmarks")
    return (landmarks, records) if with_records else landmarks
