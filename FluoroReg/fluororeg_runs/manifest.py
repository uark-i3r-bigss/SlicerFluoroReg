"""Loader for an xreg run directory described by run_manifest.json (schema 4).

Paths (xreg docs/frames_and_conventions.md, "Paths"): files the run wrote (`file`, `file_8bit`,
`images.*`, `registrations[].files[].file`) are relative to the run directory. `run.dir`, input
`path`s, transform `source`s and `targets[].init.path`/`from_run` are relative to the experiment
folder, which is the run directory with run.dir's components removed from its end. An input outside
the experiment folder (a config file or a DICOM) keeps its absolute path. Schema 2 and 3 runs are not
read.

Transform entries carry name, stage, view (null for multiview joint poses), target, to, from, matrix,
file (null when the run wrote none) and, for initial and ground-truth poses, source.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import numpy as np

from .camera import RawCamera
from .fcsv import read_fcsv
from .landmarks import registration_points_2d
from .transforms import as_matrix

MANIFEST_NAME = "run_manifest.json"
MANIFEST_KIND = "xreg-run-manifest"
SCHEMA_VERSION = 4

# Stages that are CT_from_CAMWORLD poses of one view and target, in pipeline order.
POSE_STAGES = ("init", "perturbed_init", "fiducial", "final", "ground_truth")


class ManifestError(ValueError):
    pass


def camworld_frame(view_id: str) -> str:
    return f"CAMWORLD[{view_id}]"


def experiment_dir_of(run_dir, run_dir_rel) -> Path:
    """The experiment folder of `run_dir` (absolute) given the manifest's run.dir (relative to it)."""
    run_dir = Path(run_dir)
    if not isinstance(run_dir_rel, str) or not run_dir_rel:
        raise ManifestError(f"run.dir is {run_dir_rel!r}, expected the run directory relative to the "
                            f"experiment folder")
    rel = PurePosixPath(run_dir_rel)
    if rel.is_absolute():
        raise ManifestError(f"run.dir {run_dir_rel!r} is absolute; schema {SCHEMA_VERSION} writes it relative to "
                            f"the experiment folder (run directory: {run_dir})")
    parts = rel.parts
    if not parts or ".." in parts or run_dir.parts[-len(parts):] != parts:
        raise ManifestError(f"run.dir {run_dir_rel!r} does not match the end of the run directory {run_dir}: "
                            f"cannot derive the experiment folder")
    return Path(*run_dir.parts[:-len(parts)])


class XregRun:
    """A parsed schema 4 run manifest plus path resolution against its run directory and experiment folder."""

    def __init__(self, manifest_path, data: dict):
        self.manifest_path = Path(manifest_path).resolve()
        self.run_dir = self.manifest_path.parent
        self.data = data
        if data.get("kind") != MANIFEST_KIND:
            raise ManifestError(f"{manifest_path}: kind is {data.get('kind')!r}, expected {MANIFEST_KIND!r}")
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ManifestError(
                f"{manifest_path}: schema_version is {data.get('schema_version')!r}; only schema {SCHEMA_VERSION} "
                f"runs are read (earlier runs are not migrated)")
        for key in ("run", "inputs", "targets", "views", "transforms"):
            if key not in data:
                raise ManifestError(f"{manifest_path}: missing top-level key {key!r}")
        self.experiment_dir = experiment_dir_of(self.run_dir, data["run"].get("dir"))
        self._by_name: Dict[str, dict] = {}
        for t in data["transforms"]:
            if t["name"] in self._by_name:
                raise ManifestError(f"{manifest_path}: duplicate transform name {t['name']!r}")
            self._by_name[t["name"]] = t
        self._landmarks_3d: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def load(cls, path) -> "XregRun":
        """Load from a run directory or a run_manifest.json path."""
        p = Path(path)
        if p.is_dir():
            p = p / MANIFEST_NAME
        if not p.is_file():
            raise ManifestError(f"no {MANIFEST_NAME} at {p}")
        with open(p, "r", encoding="utf-8") as f:
            return cls(p, json.load(f))

    # ------------------------------------------------------------------ basics
    @property
    def app(self) -> str:
        return self.data["run"].get("app", "")

    @property
    def name(self) -> str:
        return self.run_dir.name

    def run_file(self, path: str) -> Path:
        """Absolute path of a file the run wrote (relative to the run directory)."""
        if PurePosixPath(path).is_absolute():
            raise ManifestError(f"{path!r} is a file the run wrote; it must be relative to the run directory")
        return self.run_dir / path

    def input_file(self, path: str) -> Path:
        """Absolute path of an input: relative to the experiment folder, or absolute when it lies outside it."""
        p = PurePosixPath(path)
        return Path(path) if p.is_absolute() else self.experiment_dir / path

    def input_path(self, role: str) -> Optional[Path]:
        """inputs.<role>.path resolved, or None (no such input, or a shipped configuration given by id)."""
        rec = self.data["inputs"].get(role)
        return self.input_file(rec["path"]) if rec and rec.get("path") else None

    # ------------------------------------------------------------------ targets
    @property
    def targets(self) -> List[dict]:
        return self.data["targets"]

    @property
    def target_keys(self) -> List[str]:
        return [t["key"] for t in self.targets]

    def target(self, key: str) -> dict:
        for t in self.targets:
            if t["key"] == key:
                return t
        raise KeyError(f"no target {key!r}")

    def _role(self, role: str) -> Optional[str]:
        keys = [t["key"] for t in self.targets if role in (t.get("roles") or [])]
        if len(keys) > 1:
            raise ManifestError(f"{len(keys)} targets have role {role!r}: {keys}")
        return keys[0] if keys else None

    @property
    def primary_key(self) -> str:
        key = self._role("primary")
        if key is None:
            raise ManifestError("no target has role 'primary'")
        return key

    @property
    def fiducial_key(self) -> Optional[str]:
        return self._role("fiducial")

    # ------------------------------------------------------------------ views
    @property
    def views(self) -> List[dict]:
        return self.data["views"]

    @property
    def view_ids(self) -> List[str]:
        return [v["id"] for v in self.views]

    def view(self, view_id: str) -> dict:
        for v in self.views:
            if v["id"] == view_id:
                return v
        raise KeyError(f"no view {view_id!r}")

    def camera_record(self, view_id: str) -> dict:
        """The raw-grid camera of a view: its own when it differs from view_defaults, else the default."""
        cam = self.view(view_id).get("camera") or (self.data.get("view_defaults") or {}).get("camera")
        if not cam:
            raise ManifestError(f"view {view_id}: no camera and no view_defaults.camera")
        return cam

    def camera(self, view_id: str) -> RawCamera:
        return RawCamera.from_manifest(self.camera_record(view_id))

    def points_2d(self, view_id: str) -> Dict[str, np.ndarray]:
        """Raw-grid 2D landmarks of a view under their registration names."""
        lands = self.view(view_id).get("landmarks_2d")
        return registration_points_2d(lands) if lands else {}

    def image_file(self, view_id: str, key: str = "xray_raw") -> Optional[Path]:
        rel = (self.view(view_id).get("images") or {}).get(key)
        return self.run_file(rel) if rel else None

    def drr_file(self, view_id: str, target_key: str) -> Optional[Path]:
        rec = ((self.view(view_id).get("targets") or {}).get(target_key) or {}).get("drr")
        return self.run_file(rec["file"]) if rec and rec.get("file") else None

    @property
    def is_multiview(self) -> bool:
        return any(t.get("from") == "W" for t in self.transforms)

    # ------------------------------------------------------------------ transforms
    @property
    def transforms(self) -> List[dict]:
        return self.data["transforms"]

    def transform(self, name: str) -> dict:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"no transform named {name!r}") from None

    @staticmethod
    def matrix(entry: dict) -> np.ndarray:
        return as_matrix(entry["matrix"])

    def pose(self, view_id: str, target_key: str, stage: str) -> Optional[dict]:
        """The CT_from_CAMWORLD[view] transform entry of one stage, or None."""
        found = [t for t in self.transforms
                 if t.get("stage") == stage and t.get("view") == view_id and t.get("target") == target_key
                 and t.get("to") == "CT" and t.get("from") == camworld_frame(view_id)]
        if len(found) > 1:
            raise ManifestError(f"{len(found)} {stage} poses for view {view_id} target {target_key}")
        return found[0] if found else None

    def find_transform(self, name: str) -> Optional[dict]:
        return self._by_name.get(name)

    def pose_matrix(self, view_id: str, target_key: str, stage: str) -> Optional[np.ndarray]:
        entry = self.pose(view_id, target_key, stage)
        return None if entry is None else self.matrix(entry)

    def pose_stages(self, view_id: str, target_key: str) -> List[str]:
        """Pose stages present for a view and target, in pipeline order."""
        return [s for s in POSE_STAGES if self.pose(view_id, target_key, s) is not None]

    def joint_pose(self, target_key: str) -> Optional[dict]:
        """The multiview joint pose CT_from_W of a target, or None."""
        found = [t for t in self.transforms
                 if t.get("view") is None and t.get("target") == target_key and t.get("to") == "CT"
                 and t.get("from") == "W" and t.get("stage") == "final"]
        return found[0] if found else None

    # ------------------------------------------------------------------ landmarks
    def landmarks_3d(self) -> Dict[str, np.ndarray]:
        """3D landmarks of inputs.landmarks_3d in CT (LPS), cached."""
        if self._landmarks_3d is None:
            path = self.input_path("landmarks_3d")
            self._landmarks_3d = read_fcsv(path, "LPS") if path else {}
        return self._landmarks_3d

    def target_of_landmark(self, name: str) -> Optional[str]:
        for t in self.targets:
            if name.startswith(t["landmark_prefix"]):
                return t["key"]
        return None
