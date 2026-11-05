"""Project folder layout and ``project_manifest.json``.

Every path written into a project file is relative to the project, with ``/`` separators. Files
are real copies, never symbolic links.
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone

from .status import InputError

SCHEMA_VERSION = 1
MANIFEST_NAME = "project_manifest.json"
LAYOUT_DIRS = (
    "meta_data/volumes",
    "meta_data/models",
    "meta_data/landmarks/landmarks_3d",
    "meta_data/landmarks/landmarks_2d",
    "meta_data/XrayDICOM",
    "meta_data/XrayProjData",
    "meta_data/tmp",
    "configs/xreg",
    "configs/pose_estimation",
    "logs",
    "outputs",
)


def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_case_id(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip())
    return normalized.strip("._-") or "case"


def manifest_path(project):
    return os.path.join(project, MANIFEST_NAME)


def rel(project, path):
    """``path`` relative to the project, ``/``-separated."""
    return os.path.relpath(os.path.abspath(path), os.path.abspath(project)).replace(os.sep, "/")


def abs_path(project, relative):
    return os.path.join(os.path.abspath(project), *relative.split("/"))


def is_inside(path, root):
    path = os.path.realpath(path)
    root = os.path.realpath(root)
    return path.startswith(root.rstrip(os.sep) + os.sep)


def write_json(path, data, indent=2):
    """Atomic JSON write with a trailing newline."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)
        handle.write("\n")
    os.replace(tmp, path)


def read_json(path, what):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise InputError(f"{what} not found: {path}")
    except ValueError as exc:
        raise InputError(f"{what} {path} is not valid JSON: {exc}")


def copy_file(src, dst):
    """Copy the bytes of ``src`` (following symbolic links) to ``dst`` through a temp file."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def remove_project_path(project, path):
    """Delete a file or folder strictly inside the project."""
    if not is_inside(path, project):
        raise InputError(f"refusing to delete {path}: not inside the project {project}")
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def init_project(project):
    project = os.path.abspath(project)
    created = []
    for d in LAYOUT_DIRS:
        path = os.path.join(project, *d.split("/"))
        if not os.path.isdir(path):
            os.makedirs(path)
            created.append(d)
    if not os.path.isfile(manifest_path(project)):
        write_json(manifest_path(project), {
            "schema_version": SCHEMA_VERSION,
            "project_name": os.path.basename(project),
            "created_utc": utc_now_iso(),
            "cases": [],
        })
        created.append(MANIFEST_NAME)
    else:
        load_manifest(project)
    return created


def load_manifest(project):
    path = manifest_path(project)
    if not os.path.isfile(path):
        raise InputError(f"{os.path.abspath(project)} is not a project (no {MANIFEST_NAME}); run init first")
    data = read_json(path, "project manifest")
    if not isinstance(data, dict) or not isinstance(data.get("cases", []), list):
        raise InputError(f"{path}: expected an object with a 'cases' list")
    data.setdefault("cases", [])
    return data


def save_manifest(project, manifest):
    write_json(manifest_path(project), manifest)


def find_case(manifest, case_id):
    for entry in manifest.get("cases", []):
        if entry.get("case_id") == case_id:
            return entry
    return None


def require_case(project, case_id):
    manifest = load_manifest(project)
    entry = find_case(manifest, case_id)
    if entry is None:
        known = ", ".join(c.get("case_id", "?") for c in manifest["cases"]) or "none"
        raise InputError(f"case '{case_id}' is not in the project (cases: {known}); run import-ct first")
    return manifest, entry


def update_case(project, case_id, **fields):
    """Create or update a case entry, keeping the cases sorted by id. Returns the entry."""
    manifest = load_manifest(project)
    entry = find_case(manifest, case_id)
    if entry is None:
        entry = {"case_id": case_id}
        manifest["cases"].append(entry)
        manifest["cases"].sort(key=lambda c: c.get("case_id", ""))
    entry.update(fields)
    entry["updated_utc"] = utc_now_iso()
    save_manifest(project, manifest)
    return entry


def staged_paths(case_id):
    """Project-relative staged files of a case."""
    return {
        "ct": f"meta_data/volumes/{case_id}_ct.nii.gz",
        "seg": f"meta_data/volumes/{case_id}_seg.nii.gz",
        "landmarks_3d": f"meta_data/landmarks/landmarks_3d/{case_id}_landmarks_3d.fcsv",
    }


def annotation_output_rel(case_id):
    return f"meta_data/tmp/{case_id}/annotation_output"


def original_input_dir_rel(case_id):
    return f"meta_data/tmp/original_inputs/{case_id}"


def nifti_suffix(name):
    lower = name.lower()
    for suffix in (".nii.gz", ".nii"):
        if lower.endswith(suffix):
            return suffix
    return None
