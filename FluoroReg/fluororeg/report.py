"""``status``: one summary of a folder, read from its files (never writes).

Works on any existing folder: without ``project_manifest.json`` (``set_up`` false) there are no cases and the
rest (X-rays, pose configs, xreg configs, runs) is read as found.
"""

import glob
import json
import os

from .assess import ASSESSMENT_NAME, SCHEMA_VERSION as ASSESSMENT_SCHEMA
from .configs import show_config
from .pose import model_info, pose_config_rel, pose_dirs, quality_summary
from .project import abs_path, load_manifest, manifest_path, rel, staged_paths
from .status import InputError
from .xrays import DICOM_DIR, PROJ_DIR, converted_ids, dicom_ids, load_index, natural_key


def _present(project, relative):
    return {"path": relative, "present": bool(relative) and os.path.exists(abs_path(project, relative))}


def _cases(project, manifest):
    out = []
    for entry in manifest.get("cases", []):
        case_id = entry.get("case_id")
        staged = dict(staged_paths(case_id))
        staged.update({k: v for k, v in (entry.get("staged") or {}).items() if k in staged and v})
        models = (entry.get("staged") or {}).get("models") or []
        out.append({
            "case_id": case_id,
            "status": entry.get("status"),
            "input": _present(project, entry.get("input")),
            "annotation_output": _present(project, entry.get("annotation_output")),
            "staged": {k: _present(project, v) for k, v in staged.items()},
            "models": [m for m in models if os.path.isfile(abs_path(project, m))],
            "landmark_sets": (entry.get("staged") or {}).get("landmark_sets") or [],
        })
    return out


def _xrays(project):
    index = load_index(project)
    h5 = set(converted_ids(project))
    ids = sorted(set(dicom_ids(project)) | set(index), key=natural_key)
    entries = [{"id": i, "source_name": (index.get(i) or {}).get("source_name"), "h5": i in h5} for i in ids]
    return {"dicom_dir": DICOM_DIR, "proj_dir": PROJ_DIR, "ids": ids, "entries": entries,
            "converted": sorted(h5, key=natural_key)}


def _pose(project, manifest):
    cases = [c.get("case_id") for c in manifest.get("cases", [])]
    for path in glob.glob(os.path.join(project, "configs", "pose_estimation", "*.json")):
        case_id = os.path.basename(path)[:-5]
        if case_id not in cases:
            cases.append(case_id)
    out = {}
    for case_id in cases:
        poses_dir, models_dir = pose_dirs(project, case_id)
        active = None
        active_path = os.path.join(models_dir, "active.json")
        if os.path.isfile(active_path):
            try:
                with open(active_path, "r", encoding="utf-8") as handle:
                    active = json.load(handle).get("model_dir")
            except (OSError, ValueError, AttributeError):
                active = "unreadable active.json"
        pose_files = sorted(glob.glob(os.path.join(poses_dir, "view-*_target-ct_CT_from_CAMWORLD.h5")))
        out[case_id] = {
            "config": _present(project, pose_config_rel(case_id)),
            "active_model": active,
            "model": model_info(project, case_id),
            "initial_poses": rel(project, poses_dir),
            "pose_ids": [os.path.basename(p)[len("view-"):].split("_target-", 1)[0] for p in pose_files],
            "quality": quality_summary(poses_dir),
        }
    return out


def _xreg_configs(project):
    out = []
    for path in sorted(glob.glob(os.path.join(project, "configs", "xreg", "*.json"))):
        if path.endswith(".gui.json"):
            continue
        try:
            info = show_config(path)
        except (InputError, AttributeError, TypeError) as exc:  # one broken file must not hide the others
            out.append({"config": rel(project, path), "problems": [str(exc)]})
            continue
        info["config"] = rel(project, path)
        out.append(info)
    return out


def _run_manifests(project, xreg_configs):
    """Every run_manifest.json under ``outputs/`` and under the parent folder of each xreg config's
    ``output.dir`` (relative to the project), deduplicated. Folders outside the project are not searched:
    a copied config may still name the original experiment."""
    project_real = os.path.realpath(project)
    roots = [os.path.join(project, "outputs")]
    for info in xreg_configs:
        out_dir = info.get("output_dir")
        if not isinstance(out_dir, str) or not out_dir:
            continue
        root = os.path.dirname(os.path.normpath(os.path.join(project, out_dir)))
        real = os.path.realpath(root)
        if real == project_real or real.startswith(project_real + os.sep):
            roots.append(root)
    seen = set()
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for path in glob.glob(os.path.join(glob.escape(root), "**", "run_manifest.json"), recursive=True):
            key = os.path.realpath(path)
            if key not in seen:
                seen.add(key)
                found.append(os.path.abspath(path))
    return sorted(found, key=lambda p: rel(project, p))


def _runs(project, xreg_configs):
    out = []
    for path in _run_manifests(project, xreg_configs):
        run_dir = os.path.dirname(path)
        entry = {"run_dir": rel(project, run_dir)}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                m = json.load(handle)
        except (OSError, ValueError) as exc:
            entry["error"] = f"unreadable run_manifest.json: {exc}"
            out.append(entry)
            continue
        entry["schema_version"] = m.get("schema_version")
        if m.get("schema_version") != 4:
            entry["error"] = "not schema 4"
            out.append(entry)
            continue
        run = m.get("run") or {}
        summary = m.get("summary") or {}
        entry.update({"app": run.get("app"), "status": run.get("status"), "started_utc": run.get("started_utc"),
                      "summary": summary.get("text"), "views": summary.get("views"),
                      "views_registered": summary.get("views_registered"), "flags": summary.get("flags"),
                      "registrations": [{"id": r.get("id"), "status": r.get("status"),
                                         "final_objective": (r.get("quality") or {}).get("final_objective"),
                                         "target_objectives": (r.get("quality") or {}).get("target_objectives"),
                                         "target_objective_provenance":
                                             (r.get("quality") or {}).get("target_objective_provenance")}
                                        for r in m.get("registrations") or []]})
        assessment = os.path.join(run_dir, ASSESSMENT_NAME)
        if os.path.isfile(assessment):
            try:
                with open(assessment, "r", encoding="utf-8") as handle:
                    doc = json.load(handle)
                if doc.get("schema_version") != ASSESSMENT_SCHEMA:
                    entry["assessment"] = {"path": rel(project, assessment), "stale": True,
                                           "error": f"assessment.json schema {doc.get('schema_version')}; "
                                                    f"assess the run again"}
                else:
                    counts = {}
                    for item in doc.get("items") or []:
                        counts[item.get("certainty")] = counts.get(item.get("certainty"), 0) + 1
                    entry["assessment"] = {"path": rel(project, assessment), "counts": counts,
                                           "items": doc.get("items")}
            except (OSError, ValueError) as exc:
                entry["assessment"] = {"path": rel(project, assessment), "error": str(exc)}
        out.append(entry)
    return out


def project_status(project):
    project = os.path.abspath(project)
    if not os.path.isdir(project):
        raise InputError(f"{project} is not a folder")
    set_up = os.path.isfile(manifest_path(project))
    manifest = load_manifest(project) if set_up else {"cases": []}
    xreg_configs = _xreg_configs(project)
    return {"project": project, "set_up": set_up, "project_name": manifest.get("project_name"),
            "cases": _cases(project, manifest), "xrays": _xrays(project), "pose": _pose(project, manifest),
            "xreg_configs": xreg_configs, "runs": _runs(project, xreg_configs)}
