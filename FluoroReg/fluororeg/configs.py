"""xreg registration configurations: ``config create``, ``config set``, ``config show``.

Configurations follow xreg's registration config schema: the absolute ``experiment_dir`` (the
project) and every other path relative to it. The pipeline only ever changes ``experiment_dir``,
``volume.ct``, ``volume.seg``, ``volume.landmarks_3d`` and ``projections.ids``; everything else is the
user's.

A config at ``<P>/configs/xreg/<name>.json`` belongs to the experiment ``<P>``. When its
``experiment_dir`` names another folder (the experiment was copied or moved), it is *relocated*: its
paths are read against ``<P>`` and ``register`` refuses it until ``config set --experiment-dir <P>``.
"""

import gzip
import json
import os
import posixpath
import re
import struct
import zlib

from .project import abs_path, load_manifest, find_case, read_json, staged_paths
from .status import InputError
from .xrays import PROJ_DIR, converted_ids

APPS = ("single-target", "single-target-multiview", "multi-target", "multi-target-multiview")
MULTIVIEW_APPS = ("single-target-multiview", "multi-target-multiview")
MULTI_TARGET_APPS = ("multi-target", "multi-target-multiview")
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
_PLACEHOLDER = re.compile(r"<[a-z_0-9]+>")
SETTABLE = (("ct", "volume", "ct"), ("seg", "volume", "seg"), ("landmarks_3d", "volume", "landmarks_3d"))


def load_template(app):
    if app not in APPS:
        raise InputError(f"unknown app '{app}'; expected one of {', '.join(APPS)}")
    with open(os.path.join(TEMPLATE_DIR, app + ".json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def default_name(app, case_id, femur=None):
    name = f"{app.replace('-', '_')}_{case_id}"
    return name + (f"_femur-{femur}" if femur else "")


def _fill(value, mapping):
    if isinstance(value, dict):
        return {k: _fill(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, mapping) for v in value]
    if isinstance(value, str):
        return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), value)
    return value


def _remaining_placeholders(value):
    if isinstance(value, dict):
        return [p for v in value.values() for p in _remaining_placeholders(v)]
    if isinstance(value, list):
        return [p for v in value for p in _remaining_placeholders(v)]
    if isinstance(value, str):
        return _PLACEHOLDER.findall(value)
    return []


def create_config(project, case_id, app, femur=None, name=None, ids=None):
    """Write ``configs/xreg/<name>.json`` from the bundled template; returns (path, config, warnings)."""
    project = os.path.abspath(project)
    if app not in APPS:
        raise InputError(f"unknown --app '{app}'; expected one of {', '.join(APPS)}")
    if app in MULTI_TARGET_APPS and femur not in ("left", "right"):
        raise InputError(f"--app {app} needs --femur left|right")
    if app not in MULTI_TARGET_APPS and femur:
        raise InputError(f"--femur applies only to the multi-target apps, not --app {app}")
    manifest = load_manifest(project)
    entry = find_case(manifest, case_id)
    if entry is None:
        raise InputError(f"case '{case_id}' is not in the project; run import-ct and stage first")
    staged = dict(staged_paths(case_id))
    staged.update({k: v for k, v in (entry.get("staged") or {}).items() if k in staged and v})
    missing = [v for v in staged.values() if not os.path.isfile(abs_path(project, v))]
    if missing:
        raise InputError(f"case '{case_id}' is not staged (missing {', '.join(missing)}); run stage first")
    name = name or default_name(app, case_id, femur)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise InputError(f"--name '{name}' must be a file-name-safe string")
    out = os.path.join(project, "configs", "xreg", name + ".json")
    if os.path.exists(out):
        raise InputError(f"configs/xreg/{name}.json exists; choose another --name or edit it by hand")

    warnings = []
    if ids:
        ids = list(ids)
    elif app in MULTIVIEW_APPS:
        ids = []
        warnings.append("projections.ids is empty: select the views for this multiview configuration "
                        "(config set --ids ...)")
    else:
        ids = converted_ids(project)
        if not ids:
            raise InputError(f"no converted X-rays in {PROJ_DIR}; run convert-xrays or pass --ids")

    config = _fill(load_template(app), {
        "<case>": case_id, "<name>": name, "<side>": femur or "",
        "<ct>": staged["ct"], "<seg>": staged["seg"], "<landmarks_3d>": staged["landmarks_3d"],
    })
    config["experiment_dir"] = project
    config["projections"]["ids"] = ids
    leftover = _remaining_placeholders(config)
    if leftover:
        raise RuntimeError(f"template {app}.json has unfilled placeholders: {', '.join(sorted(set(leftover)))}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    warnings += [p for p in config_problems(config) if not p.startswith("projections.ids is empty")]
    return out, config, warnings


def infer_app(config):
    """``seed`` present -> multiview; more than one target -> multi-target."""
    multi_target = len(config.get("targets") or []) > 1
    if "seed" in config:
        return "multi-target-multiview" if multi_target else "single-target-multiview"
    return "multi-target" if multi_target else "single-target"


def rel_path_error(value):
    """Why ``value`` is not a valid experiment-relative path, or None."""
    if not isinstance(value, str) or not value:
        return "is empty"
    if value.startswith("/"):
        return "is absolute"
    if ".." in value.split("/"):
        return "contains '..'"
    return None


def _experiment_dir(config):
    exp = config.get("experiment_dir")
    if not isinstance(exp, str) or not exp.startswith("/"):
        return None
    return exp


def holding_project(path):
    """``<P>`` of a config at ``<P>/configs/xreg/<name>.json``, else None."""
    xreg_dir = os.path.dirname(os.path.abspath(path))
    configs_dir = os.path.dirname(xreg_dir)
    if os.path.basename(xreg_dir) != "xreg" or os.path.basename(configs_dir) != "configs":
        return None
    return os.path.dirname(configs_dir)


def is_relocated(config, path):
    """True when the config's folder belongs to an experiment other than its ``experiment_dir``."""
    holder = holding_project(path)
    if holder is None:
        return False
    exp = _experiment_dir(config)
    return exp is None or os.path.realpath(exp) != os.path.realpath(holder)


def resolved_dir(config, path=None):
    """The folder the config's relative paths are read against: the holding experiment of a relocated
    config, else ``experiment_dir`` (None when that is missing or not absolute)."""
    if path is not None and is_relocated(config, path):
        return holding_project(path)
    return _experiment_dir(config)


def require_current(config, path):
    """Raise when the config is relocated: xreg would read the inputs from the old ``experiment_dir``."""
    if is_relocated(config, path):
        holder = holding_project(path)
        raise InputError(f"{path}: experiment_dir {config.get('experiment_dir')} is not the experiment folder "
                         f"holding the config; run: config set --config {os.path.abspath(path)} "
                         f"--experiment-dir {holder}", {"relocated": True, "experiment_dir": holder})


def _nifti_dims(path):
    """``dim[0..7]`` of a NIfTI-1/2 header (``.nii`` or ``.nii.gz``), or None when unreadable."""
    opener = gzip.open if path.lower().endswith(".gz") else open
    try:
        with opener(path, "rb") as handle:
            header = handle.read(540)
    except (OSError, EOFError, zlib.error):
        return None
    for endian in ("<", ">"):
        if len(header) >= 348 and struct.unpack(endian + "i", header[:4])[0] == 348:
            return struct.unpack(endian + "8h", header[40:56])
        if len(header) >= 540 and struct.unpack(endian + "i", header[:4])[0] == 540:
            return struct.unpack(endian + "8q", header[16:80])
    return None


def is_2d_image(path):
    """True for a single DICOM file or a NIfTI image with one slice (or fewer than 3 dimensions)."""
    lower = path.lower()
    if lower.endswith(".dcm"):
        return True
    if lower.endswith((".nii", ".nii.gz")):
        dims = _nifti_dims(path)
        return dims is not None and (dims[0] < 3 or dims[3] == 1)
    return False


def config_problems(config, path=None):
    """Problems that would stop the registration, as messages. With the config's ``path``, a relocated
    config is checked against the experiment holding it (and its stale ``experiment_dir`` is no problem)."""
    problems = []
    exp = resolved_dir(config, path)
    if exp is None:
        return ["experiment_dir is missing or not absolute"]
    if not os.path.isdir(exp):
        problems.append(f"experiment_dir {exp} does not exist")

    def check_file(key, value, kind="file"):
        err = rel_path_error(value)
        if err:
            problems.append(f"{key} {err}")
            return False
        path = os.path.join(exp, *value.split("/"))
        if kind == "file" and not os.path.isfile(path):
            problems.append(f"{key} {value} does not exist")
            return False
        if kind == "dir" and not os.path.isdir(path):
            problems.append(f"{key} {value} does not exist")
            return False
        return True

    volume = config.get("volume") or {}
    for key in ("ct", "seg", "landmarks_3d"):
        if check_file(f"volume.{key}", volume.get(key)) and key == "ct" and \
                is_2d_image(os.path.join(exp, *volume["ct"].split("/"))):
            problems.append(f"volume.ct {volume['ct']} is a 2D image, not a CT volume")
    registry = (config.get("landmarks") or {}).get("registry")
    if isinstance(registry, str) and ("/" in registry or registry.endswith(".json")):
        check_file("landmarks.registry", registry)

    proj = config.get("projections") or {}
    ids = proj.get("ids") or []
    pattern = proj.get("pattern") or "{id}.h5"
    proj_ok = check_file("projections.dir", proj.get("dir"), kind="dir")
    if not ids:
        problems.append("projections.ids is empty" + (": select the views" if "seed" in config else ""))
    elif proj_ok:
        missing = [i for i in ids
                   if not os.path.isfile(os.path.join(exp, *proj["dir"].split("/"), pattern.replace("{id}", i)))]
        if missing:
            problems.append(f"no projection file for id(s) {', '.join(missing)} in {proj['dir']}")

    for target in config.get("targets") or []:
        init = target.get("init") or {}
        if init.get("method") != "run":
            continue
        from_run = init.get("from_run")
        key = target.get("key", "?")
        if not check_file(f"targets[{key}].init.from_run", from_run, kind="dir"):
            continue
        from_target = (init.get("from_target") or key).lower()
        missing = [i for i in ids if not os.path.isfile(os.path.join(
            exp, *from_run.split("/"), f"view-{i}_target-{from_target}_CT_from_CAMWORLD.h5"))]
        if missing:
            problems.append(f"target {key}: no initial pose in {from_run} for id(s) {', '.join(missing)}")

    err = rel_path_error((config.get("output") or {}).get("dir"))
    if err:
        problems.append(f"output.dir {err}")
    return problems


def load_config(path):
    config = read_json(path, "configuration")
    if not isinstance(config, dict):
        raise InputError(f"{path}: expected a JSON object")
    return config


def show_config(path):
    config = load_config(path)
    volume = config.get("volume") or {}
    proj = config.get("projections") or {}
    return {
        "config": os.path.abspath(path),
        "app": infer_app(config),
        "experiment_dir": config.get("experiment_dir"),
        "relocated": is_relocated(config, path),
        "resolved_dir": resolved_dir(config, path),
        "ct": volume.get("ct"),
        "seg": volume.get("seg"),
        "landmarks_3d": volume.get("landmarks_3d"),
        "ids": list(proj.get("ids") or []),
        "targets": [{"key": t.get("key"), "landmark_set": t.get("landmark_set"),
                     "init": (t.get("init") or {}).get("method", "pnp")} for t in config.get("targets") or []],
        "output_dir": (config.get("output") or {}).get("dir"),
        "problems": config_problems(config, path),
    }


def detect_indent(text):
    """The indent unit of a JSON file (``None`` for a one-line file)."""
    for line in text.splitlines()[1:]:
        stripped = line.lstrip(" \t")
        if stripped and len(stripped) < len(line):
            return line[:len(line) - len(stripped)]
    return None


def _as_relative(exp, value, flag):
    """``value`` relative to ``exp``; an absolute path must lie inside it."""
    if os.path.isabs(value):
        exp_real = os.path.realpath(exp)
        real = os.path.realpath(value)
        if not real.startswith(exp_real.rstrip(os.sep) + os.sep):
            raise InputError(f"{flag} {value} is not inside experiment_dir {exp}")
        value = os.path.relpath(real, exp_real).replace(os.sep, "/")
    value = posixpath.normpath(value.replace("\\", "/"))
    err = rel_path_error(value)
    if err or value == ".":
        raise InputError(f"{flag} {value} {err or 'is not a file'}")
    if not os.path.isfile(os.path.join(exp, *value.split("/"))):
        raise InputError(f"{flag} {value} does not exist under experiment_dir {exp}")
    return value


def _new_experiment_dir(path, value):
    """``--experiment-dir`` checked: an existing absolute folder, and the holding experiment when the config
    lies in ``<P>/configs/xreg/``."""
    if not os.path.isabs(value):
        raise InputError(f"--experiment-dir {value} is not an absolute path")
    if not os.path.isdir(value):
        raise InputError(f"--experiment-dir {value} is not a folder")
    holder = holding_project(path)
    if holder is not None and os.path.realpath(value) != os.path.realpath(holder):
        raise InputError(f"--experiment-dir {value}: a config in {os.path.join(holder, 'configs', 'xreg')} "
                         f"belongs to the experiment {holder}")
    return os.path.normpath(value)


def set_config(path, ct=None, seg=None, landmarks_3d=None, ids=None, experiment_dir=None):
    """Change only the given fields in place, keeping key order and the file's indent."""
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        config = json.loads(text)
    except ValueError as exc:
        raise InputError(f"{path} is not valid JSON: {exc}")
    if not isinstance(config, dict):
        raise InputError(f"{path}: expected a JSON object")
    changes = []
    if experiment_dir is not None:
        new = _new_experiment_dir(path, experiment_dir)
        old = config.get("experiment_dir")
        if old != new:
            config["experiment_dir"] = new
            changes.append({"field": "experiment_dir", "old": old, "new": new})
    exp = resolved_dir(config, path)
    if exp is None or not os.path.isdir(exp):
        raise InputError(f"{path}: experiment_dir is missing, not absolute or not a folder")
    requested = {"ct": ct, "seg": seg, "landmarks_3d": landmarks_3d}
    for key, section, field in SETTABLE:
        value = requested[key]
        if value is None:
            continue
        new = _as_relative(exp, value, "--" + key.replace("_", "-"))
        block = config.get(section)
        if not isinstance(block, dict):
            raise InputError(f"{path}: no '{section}' object")
        old = block.get(field)
        if old != new:
            block[field] = new
            changes.append({"field": f"{section}.{field}", "old": old, "new": new})
    if ids is not None:
        proj = config.get("projections")
        if not isinstance(proj, dict) or not isinstance(proj.get("dir"), str):
            raise InputError(f"{path}: no 'projections.dir'")
        pattern = proj.get("pattern") or "{id}.h5"
        missing = [i for i in ids
                   if not os.path.isfile(os.path.join(exp, *proj["dir"].split("/"), pattern.replace("{id}", i)))]
        if missing:
            raise InputError(f"--ids: no projection file for {', '.join(missing)} in {proj['dir']}")
        old = proj.get("ids")
        new = list(ids)
        if old != new:
            proj["ids"] = new
            changes.append({"field": "projections.ids", "old": old, "new": new})
    if changes:
        indent = detect_indent(text)
        out = json.dumps(config, indent=indent, ensure_ascii=False)
        if text.endswith("\n"):
            out += "\n"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(out)
        os.replace(tmp, path)
    return changes


def config_case(config):
    """The case a configuration belongs to: ``output.dir`` ``outputs/<case>/...``, else None."""
    parts = str((config.get("output") or {}).get("dir") or "").split("/")
    if len(parts) >= 3 and parts[0] == "outputs" and parts[1]:
        return parts[1]
    return None
