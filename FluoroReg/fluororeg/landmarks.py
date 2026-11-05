"""Project landmark configuration ``configs/landmark_sets.json`` (xreg landmark config schema 2).

Ported from xreg_gui's ``landmark_set_registry`` (only what staging needs). Shipped sets come from
xreg's ``configs/landmark_configs/bigss_canonical.json``; a set already in the project is never
modified.
"""

import copy
import json
import os
import re

from .project import write_json
from .status import InputError

SCHEMA_VERSION = 2
DEFAULT_SET_ID = "default"
CANONICAL_SEP = "::"
PNP_EXCLUDE_KEY = "pnp_exclude"
CANONICAL_CONFIG_ID = "bigss_canonical"
REGISTRY_REL_PATH = "configs/landmark_sets.json"


def sanitize_set_id(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", (value or "").strip()).strip("_")
    return text or DEFAULT_SET_ID


def canonical_landmark_name(set_id, landmark_name):
    name = (landmark_name or "").strip()
    if not name:
        return ""
    if CANONICAL_SEP in name:
        return name
    return f"{sanitize_set_id(set_id)}{CANONICAL_SEP}{name}"


def set_landmark_names(entry):
    """Member names of a set, schema v2 (``landmarks``) or v1 (``landmark_names``)."""
    entry = entry or {}
    if isinstance(entry.get("landmarks"), list):
        names = [item.get("name", "") for item in entry["landmarks"] if isinstance(item, dict)]
    else:
        names = list(entry.get("landmark_names") or [])
    return [name.strip() for name in names if (name or "").strip()]


def _normalize_pnp_exclude(values):
    cleaned, seen = [], set()
    for value in values or []:
        name = (value or "").strip()
        if name and name not in seen:
            cleaned.append(name)
            seen.add(name)
    return cleaned


def canonical_config_path(xreg_dir):
    return os.path.join(xreg_dir, "configs", "landmark_configs", f"{CANONICAL_CONFIG_ID}.json")


def load_canonical_config(xreg_dir):
    path = canonical_config_path(xreg_dir)
    if not os.path.isfile(path):
        raise InputError(f"xreg's shipped landmark configuration was not found: {path} (check --xreg-dir)")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def registry_path(project):
    return os.path.join(os.path.abspath(project), *REGISTRY_REL_PATH.split("/"))


def load_registry(project):
    path = registry_path(project)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            registry = json.load(handle)
    else:
        registry = {"schema_version": SCHEMA_VERSION, "sets": []}
    registry.setdefault("sets", [])
    return registry


def _v2_entry(entry):
    entry = dict(entry or {})
    if not isinstance(entry.get("landmarks"), list):
        entry["landmarks"] = [{"name": name, "visible": True} for name in set_landmark_names(entry)]
    entry.pop("landmark_names", None)
    if PNP_EXCLUDE_KEY in entry:
        excluded = _normalize_pnp_exclude(entry.get(PNP_EXCLUDE_KEY))
        if excluded:
            entry[PNP_EXCLUDE_KEY] = excluded
        else:
            entry.pop(PNP_EXCLUDE_KEY, None)
    return entry


def save_registry(project, registry):
    registry = dict(registry or {})
    registry.pop("version", None)
    saved = {"schema_version": SCHEMA_VERSION}
    saved.update({k: v for k, v in registry.items() if k != "schema_version"})
    saved["sets"] = [_v2_entry(entry) for entry in registry.get("sets", [])]
    path = registry_path(project)
    write_json(path, saved)
    return path


def find_set(registry, set_id):
    wanted = sanitize_set_id(set_id)
    for entry in registry.get("sets", []):
        if entry.get("set_id") == wanted:
            return entry
    return None


def ensure_canonical_sets(project, landmark_names, canonical):
    """Project sets the given 3D landmark names belong to.

    A shipped set (``canonical``, may be None) with a member among ``landmark_names`` is added
    to the project landmark configuration when the project has no set with its id. Returns
    the project's entries of every set holding one of the names, shipped sets first.
    """
    detected = set(landmark_names or [])
    registry = load_registry(project)
    existed = os.path.isfile(registry_path(project))
    sets = {entry.get("set_id"): entry for entry in registry.get("sets", [])}
    detected_ids = []
    added = False
    for entry in (canonical or {}).get("sets", []):
        if not detected.intersection(set_landmark_names(entry)):
            continue
        set_id = entry.get("set_id")
        detected_ids.append(set_id)
        if set_id not in sets:
            new_entry = copy.deepcopy(entry)
            new_entry["source"] = "3d_auto"
            sets[set_id] = new_entry
            added = True
    for set_id, entry in sorted(sets.items()):
        if set_id not in detected_ids and detected.intersection(set_landmark_names(entry)):
            detected_ids.append(set_id)
    if added or not existed:
        registry["sets"] = sorted(sets.values(), key=lambda item: item.get("set_id", ""))
        if "cross_set_mirror_pairs" not in registry:
            registry["cross_set_mirror_pairs"] = [
                pair for pair in (canonical or {}).get("cross_set_mirror_pairs", [])
                if all(name.split(CANONICAL_SEP, 1)[0] in sets for name in pair)
            ]
        save_registry(project, registry)
        registry = load_registry(project)
    return [entry for entry in (find_set(registry, set_id) for set_id in detected_ids) if entry]


def load_fcsv_points(path):
    """{label: [x, y, z]} of a Slicer FCSV (label in column 12)."""
    points = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split(",")
            if len(parts) < 12:
                continue
            try:
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
            except ValueError:
                continue
            points[parts[11]] = coords
    return points


def canonicalize_fcsv_labels_with_map(src_path, dst_path, label_map):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = dst_path + ".tmp"
    with open(src_path, "r", encoding="utf-8") as src, open(tmp, "w", encoding="utf-8") as dst:
        for line in src:
            if line.startswith("#") or not line.strip():
                dst.write(line)
                continue
            newline = "\n" if line.endswith("\n") else ""
            parts = line.rstrip("\n").split(",")
            if len(parts) >= 12 and parts[11].strip():
                label = parts[11].strip()
                parts[11] = label if CANONICAL_SEP in label else label_map.get(label, label)
                dst.write(",".join(parts) + newline)
            else:
                dst.write(line)
    os.replace(tmp, dst_path)
    return dst_path
