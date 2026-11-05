"""Contract check of an xreg run directory against its run_manifest.json (schema 4).

Checks, each recorded as a named result:
  manifest    kind and schema 4; run.dir is relative and matches the end of the run directory, which
              gives the experiment folder
  files       every file the run wrote exists and is relative to the run directory; every input exists
              and is relative to the experiment folder, unless it is absolute, outside the experiment
              folder and flagged input_outside_experiment (such an input may be absent: it is not part of
              the experiment folder, and is only noted)
  frames      every to/from/in of a matrix-bearing entry is a frame of the conventions (CT, W,
              CAMWORLD/CAM/DET[<view>], REGIREF[<target>]); matrices are rigid
  links       transform names referenced by views[].targets exist and match view/target/stage
  pose_dir    per-view final poses live in poses/ under the pose-directory name, and nothing else does
  images      raw X-ray and DRR .mha dimensions and spacing equal the view's raw camera
  residuals   landmark residuals recomputed from the raw camera, final pose, 3D landmarks and
              raw-grid 2D landmarks (through registration_names) match xreg's per_landmark values
  joint       multiview: every per-view final pose T_i equals J @ F_i (J = joint CT_from_W of the
              target, F_i = the fiducial target's fiducial pose of view i)
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .landmarks import landmark_residuals_px
from .manifest import ManifestError, XregRun, camworld_frame
from .mha import mha_size, mha_spacing, read_mha_header
from .transforms import as_matrix, is_rigid, pose_difference

OUTSIDE_FLAG = "input_outside_experiment"
# Subtrees that hold prose, flags or the config as the user wrote it, not the run's matrices.
_SKIP_TOP = ("summary", "flags", "config")


class Check:
    def __init__(self, name: str):
        self.name = name
        self.failures: List[str] = []
        self.count = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "count": self.count, "failures": list(self.failures)}


class ContractReport:
    def __init__(self, run_dir):
        self.run_dir = str(run_dir)
        self.app: Optional[str] = None
        self.experiment_dir: Optional[str] = None
        self.checks: List[Check] = []
        self.metrics: dict = {}
        self.notes: List[str] = []

    def check(self, name: str) -> Check:
        c = Check(name)
        self.checks.append(c)
        return c

    def get(self, name: str) -> Optional[Check]:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    def to_dict(self) -> dict:
        return {"run_dir": self.run_dir, "experiment_dir": self.experiment_dir, "app": self.app, "ok": self.ok,
                "metrics": self.metrics, "notes": list(self.notes), "checks": [c.to_dict() for c in self.checks]}

    def format(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'} {self.run_dir} ({self.app or 'unreadable'})"]
        if self.experiment_dir:
            lines.append(f"  experiment folder {self.experiment_dir}")
        for c in self.checks:
            lines.append(f"  {'ok  ' if c.ok else 'FAIL'} {c.name:10s} {c.count} checked")
            lines.extend(f"         - {f}" for f in c.failures)
        lines.extend(f"  note: {n}" for n in self.notes)
        for k, v in self.metrics.items():
            lines.append(f"  {k} = {v}")
        return "\n".join(lines)


def _dicts(node, path: str) -> Iterator[Tuple[str, dict]]:
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _dicts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _dicts(v, f"{path}[{i}]")


def run_written_files(run: XregRun) -> List[Tuple[str, str]]:
    """(json path, path as written) of every file the run wrote, relative to the run directory."""
    out = []
    for i, t in enumerate(run.transforms):
        if t.get("file") is not None:
            out.append((f"transforms[{i}].file", t["file"]))
    for v in run.views:
        where = f"views[{v['id']}]"
        for key, rel in (v.get("images") or {}).items():
            out.append((f"{where}.images.{key}", rel))
        for key, entry in (v.get("targets") or {}).items():
            for field, rel in (entry.get("drr") or {}).items():
                out.append((f"{where}.targets[{key}].drr.{field}", rel))
    for r in run.data.get("registrations") or []:
        for i, f in enumerate(r.get("files") or []):
            out.append((f"registrations[{r.get('id')}].files[{i}].file", f.get("file")))
    return out


class InputRef:
    """One input path of the manifest: where it is written, the record holding it, whether it is a directory
    and which flag view (None: run-wide) an input_outside_experiment flag for it carries."""

    def __init__(self, where: str, path, record: Optional[dict], is_dir: bool = False, view: Optional[str] = None):
        self.where, self.path, self.record, self.is_dir, self.view = where, path, record, is_dir, view


def input_refs(run: XregRun) -> List[InputRef]:
    """Every input path of the manifest (relative to the experiment folder, or absolute outside it)."""
    out = []
    for role, rec in run.data["inputs"].items():
        if isinstance(rec, dict) and "path" in rec:
            out.append(InputRef(f"inputs.{role}.path", rec["path"], rec))
    for t in run.targets:
        init = t.get("init") or {}
        if "path" in init:
            out.append(InputRef(f"targets[{t['key']}].init.path", init["path"], None))
        if "from_run" in init:
            out.append(InputRef(f"targets[{t['key']}].init.from_run", init["from_run"], None, is_dir=True))
    for v in run.views:
        for key in ("projection", "dicom"):
            rec = v.get(key)
            if isinstance(rec, dict) and "path" in rec:
                out.append(InputRef(f"views[{v['id']}].{key}.path", rec["path"], rec, view=v["id"]))
    for i, t in enumerate(run.transforms):
        if "source" in t:
            out.append(InputRef(f"transforms[{i}].source", t["source"], None, view=t.get("view")))
    return out


def _relative_problem(value) -> Optional[str]:
    """Why a manifest path is not a valid relative path, or None."""
    if not isinstance(value, str) or not value:
        return f"{value!r} is not a path"
    p = PurePosixPath(value)
    if p.is_absolute():
        return f"{value!r} is absolute"
    if ".." in p.parts:
        return f"{value!r} has a '..' component"
    return None


def _inside(path: PurePosixPath, folder: PurePosixPath) -> bool:
    return path.parts[:len(folder.parts)] == folder.parts


def _check_files(run: XregRun, rep: ContractReport) -> None:
    c = rep.check("files")
    for where, rel in run_written_files(run):
        c.count += 1
        problem = _relative_problem(rel)
        if problem:
            c.fail(f"{where}: {problem}; a file the run wrote is relative to the run directory")
        elif not run.run_file(rel).is_file():
            c.fail(f"{where}: {rel} does not exist in the run directory")

    flagged_views = {f.get("view") for f in run.data.get("flags") or [] if f.get("code") == OUTSIDE_FLAG}
    experiment = PurePosixPath(run.experiment_dir)
    outside = 0
    for ref in input_refs(run):
        c.count += 1
        if isinstance(ref.path, str) and PurePosixPath(ref.path).is_absolute():
            p = PurePosixPath(ref.path)
            if _inside(p, experiment):
                c.fail(f"{ref.where}: {ref.path} is inside the experiment folder but written absolute")
                continue
            if ref.view not in flagged_views:
                c.fail(f"{ref.where}: absolute input {ref.path} without an {OUTSIDE_FLAG} flag"
                       f"{f' for view {ref.view}' if ref.view else ''}")
                continue
            outside += 1
            if not (Path(ref.path).is_dir() if ref.is_dir else Path(ref.path).is_file()):
                rep.notes.append(f"{ref.where}: {ref.path} is outside the experiment folder and absent here")
            continue
        problem = _relative_problem(ref.path)
        if problem:
            c.fail(f"{ref.where}: {problem}; an input is relative to the experiment folder")
            continue
        if ref.record is not None and ref.record.get("missing"):
            rep.notes.append(f"{ref.where}: {ref.path} was missing when the run ran")
            continue
        target = run.input_file(ref.path)
        if not (target.is_dir() if ref.is_dir else target.is_file()):
            c.fail(f"{ref.where}: {ref.path} does not exist in the experiment folder {run.experiment_dir}")
    rep.metrics["inputs_outside_experiment"] = outside


def _frame_names(run: XregRun) -> set:
    names = {"CT", "W"}
    for vid in run.view_ids:
        names.update({camworld_frame(vid), f"CAM[{vid}]", f"DET[{vid}]"})
    names.update(f"REGIREF[{key}]" for key in run.target_keys)
    return names


def _check_frames(run: XregRun, rep: ContractReport) -> None:
    c = rep.check("frames")
    frames = _frame_names(run)
    for top, node in run.data.items():
        if top in _SKIP_TOP:
            continue
        for path, d in _dicts(node, top):
            sides = [s for s in ("to", "from", "in") if s in d]
            if "matrix" not in d or not sides:
                continue
            c.count += 1
            for side in sides:
                if d[side] not in frames:
                    c.fail(f"{path}: {side} frame {d[side]!r} is not a frame of this run")
            try:
                m = as_matrix(d["matrix"])
            except ValueError as e:
                c.fail(f"{path}: {e}")
                continue
            if not is_rigid(m):
                c.fail(f"{path}: matrix is not a rigid transform")


def _check_links(run: XregRun, rep: ContractReport) -> None:
    c = rep.check("links")
    target_keys = set(run.target_keys)
    for t in run.transforms:
        c.count += 1
        if t.get("target") is not None and t["target"] not in target_keys:
            c.fail(f"transform {t['name']}: unknown target {t['target']!r}")
        if t.get("view") is not None and t["view"] not in run.view_ids:
            c.fail(f"transform {t['name']}: unknown view {t['view']!r}")
    for v in run.views:
        for key, entry in (v.get("targets") or {}).items():
            where = f"views[{v['id']}].targets[{key}]"
            if key not in target_keys:
                c.fail(f"{where}: unknown target")
            refs = [("final", entry.get("final"), v["id"], "final"),
                    ("joint", entry.get("joint"), None, "final"),
                    ("relative_to_primary", entry.get("relative_to_primary"), v["id"], "relative"),
                    ("ground_truth", entry.get("ground_truth"), v["id"], "ground_truth")]
            for field, name, want_view, want_stage in refs:
                if name is None:
                    continue
                c.count += 1
                t = run.find_transform(name)
                if t is None:
                    c.fail(f"{where}.{field}: no transform named {name!r}")
                    continue
                if t.get("view") != want_view or t.get("target") != key or t.get("stage") != want_stage:
                    c.fail(f"{where}.{field}: {name} has view {t.get('view')!r} target {t.get('target')!r} "
                           f"stage {t.get('stage')!r}")
            if entry.get("status") == "registered" and entry.get("final") is None:
                c.fail(f"{where}: registered without a final transform")
            final = run.find_transform(entry["final"]) if entry.get("final") else None
            if final is not None and (final.get("to") != "CT" or final.get("from") != camworld_frame(v["id"])):
                c.fail(f"{where}.final: {final['name']} is {final.get('to')}_from_{final.get('from')}, "
                       f"expected CT_from_{camworld_frame(v['id'])}")


def _check_pose_dir(run: XregRun, rep: ContractReport) -> None:
    c = rep.check("pose_dir")
    expected = set()
    for t in run.transforms:
        if t.get("stage") != "final" or t.get("view") is None:
            continue
        c.count += 1
        want = f"poses/view-{t['view']}_target-{t['target']}_CT_from_CAMWORLD.h5"
        expected.add(want)
        if t.get("file") != want:
            c.fail(f"transform {t['name']}: file {t.get('file')!r}, expected {want!r}")
    poses = run.run_dir / "poses"
    if poses.is_dir():
        for f in sorted(poses.iterdir()):
            rel = f"poses/{f.name}"
            if rel not in expected:
                c.fail(f"{rel} is not a per-view final pose of this run")


def _check_images(run: XregRun, rep: ContractReport) -> None:
    c = rep.check("images")
    for v in run.views:
        try:
            cam = run.camera_record(v["id"])
        except ManifestError as e:
            c.fail(str(e))
            continue
        want_size = (cam.get("cols"), cam.get("rows"))
        want_spacing = (cam.get("col_spacing_mm"), cam.get("row_spacing_mm"))
        files = [(f"views[{v['id']}].images.{k}", rel) for k, rel in (v.get("images") or {}).items()]
        files += [(f"views[{v['id']}].targets[{k}].drr.file", e["drr"].get("file"))
                  for k, e in (v.get("targets") or {}).items() if e.get("drr")]
        for where, rel in files:
            if not str(rel or "").endswith(".mha") or _relative_problem(rel):
                continue
            c.count += 1
            try:
                header = read_mha_header(run.run_file(rel))
            except (OSError, ValueError) as e:
                c.fail(f"{where}: cannot read {rel}: {e}")
                continue
            size = mha_size(header)
            if size != want_size:
                c.fail(f"{where}: {rel} is {size[0]} x {size[1]} (cols x rows), camera is "
                       f"{want_size[0]} x {want_size[1]}")
            spacing = mha_spacing(header)
            if len(spacing) != 2 or not np.allclose(spacing, want_spacing, rtol=0, atol=1e-6):
                c.fail(f"{where}: {rel} spacing {spacing} differs from the camera's {want_spacing}")


def _check_residuals(run: XregRun, rep: ContractReport, tol_px: float) -> None:
    c = rep.check("residuals")
    try:
        lands_3d = run.landmarks_3d()
    except (OSError, ValueError) as e:
        c.fail(f"cannot read the 3D landmarks: {e}")
        return
    max_diff = 0.0
    for v in run.views:
        try:
            cam = run.camera(v["id"])
            pts_2d = run.points_2d(v["id"])
        except (KeyError, ValueError) as e:
            c.fail(f"views[{v['id']}]: {e}")
            continue
        for key, entry in (v.get("targets") or {}).items():
            reported = (entry.get("landmark_residual_px") or {}).get("per_landmark")
            if reported is None or entry.get("final") is None:
                continue
            where = f"views[{v['id']}].targets[{key}]"
            try:
                pose = run.matrix(run.transform(entry["final"]))
                prefix = run.target(key)["landmark_prefix"]
            except KeyError as e:
                c.fail(f"{where}: {e}")
                continue
            ours = landmark_residuals_px(cam, pose, lands_3d, pts_2d, prefix)
            n = entry["landmark_residual_px"].get("n")
            if n is not None and n != len(reported):
                c.fail(f"{where}: n = {n} but per_landmark has {len(reported)} entries")
            for name in sorted(set(reported) - set(ours)):
                c.fail(f"{where}: reported landmark {name} is missing from the recomputation")
            for name in sorted(set(ours) - set(reported)):
                c.fail(f"{where}: recomputed landmark {name} ({ours[name]:.2f} px) is not reported")
            for name in sorted(set(ours) & set(reported)):
                c.count += 1
                d = abs(ours[name] - float(reported[name]))
                max_diff = max(max_diff, d)
                if d > tol_px:
                    c.fail(f"{where}: {name} recomputed {ours[name]:.3f} px, reported "
                           f"{float(reported[name]):.3f} px")
    rep.metrics["residuals_compared"] = c.count
    rep.metrics["max_residual_diff_px"] = max_diff


def _check_joint(run: XregRun, rep: ContractReport, tol_mm: float, tol_rot: float) -> None:
    joint_refs = [(v["id"], k, e) for v in run.views for k, e in (v.get("targets") or {}).items() if e.get("joint")]
    if not joint_refs and not run.is_multiview:
        return
    c = rep.check("joint")
    fid = run.fiducial_key
    if fid is None:
        c.fail("multiview run without a target of role 'fiducial'")
        return
    worst_t, worst_r = 0.0, 0.0
    for view_id, key, entry in joint_refs:
        where = f"views[{view_id}].targets[{key}]"
        f_entry = run.pose(view_id, fid, "fiducial")
        if f_entry is None:
            c.fail(f"{where}: no fiducial pose of {fid} for view {view_id}")
            continue
        try:
            j = run.matrix(run.transform(entry["joint"]))
            t = run.matrix(run.transform(entry["final"]))
        except KeyError as e:
            c.fail(f"{where}: {e}")
            continue
        c.count += 1
        d = pose_difference(t, j @ run.matrix(f_entry))
        worst_t = max(worst_t, d["translation_mm"])
        worst_r = max(worst_r, d["rotation_max_abs"])
        if d["translation_mm"] > tol_mm or d["rotation_max_abs"] > tol_rot:
            c.fail(f"{where}: T_i differs from J @ F_i by {d['translation_mm']:.2e} mm, "
                   f"{d['rotation_max_abs']:.2e} rotation entry")
    for v in run.views:
        for key in run.target_keys:
            if run.pose(v["id"], key, "final") is not None and run.joint_pose(key) is not None \
                    and not (v.get("targets") or {}).get(key, {}).get("joint"):
                c.fail(f"views[{v['id']}].targets[{key}]: has a final pose but no joint reference")
    rep.metrics["joint_compared"] = c.count
    rep.metrics["max_joint_translation_mm"] = worst_t
    rep.metrics["max_joint_rotation_entry"] = worst_r


def contract_check(run_dir, residual_tol_px: float = 0.5, joint_tol_mm: float = 1e-3,
                   joint_tol_rot: float = 1e-5) -> ContractReport:
    """Checks one run directory; the report's ok is False on any failed check."""
    rep = ContractReport(run_dir)
    try:
        run = XregRun.load(run_dir)
    except (OSError, ValueError, KeyError) as e:
        rep.check("manifest").fail(str(e))
        return rep
    rep.app = run.app
    rep.experiment_dir = str(run.experiment_dir)
    rep.check("manifest").count = 1
    _check_files(run, rep)
    _check_frames(run, rep)
    _check_links(run, rep)
    _check_pose_dir(run, rep)
    _check_images(run, rep)
    _check_residuals(run, rep, residual_tol_px)
    _check_joint(run, rep, joint_tol_mm, joint_tol_rot)
    return rep
