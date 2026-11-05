"""FluoroReg (SlicerFluoroReg): runs the fluororeg command line (next to this file) from 3D Slicer and shows its runs.

The CT and X-rays come from the Slicer scene. Every step is a `PythonSlicer -m fluororeg ...` process
started through a non-blocking QProcess; the module builds command lines from the scene (exporting nodes
that are not files), shows one progress bar, and loads the results back into the scene. The Results step shows
finished runs in the scene (fluororeg_viewer). Any folder with runs opens; one without project_manifest.json
opens for viewing only and nothing is written into it. Every button is a public widget method and
`widget.state()` reports progress, for scripting through the WebServer.
"""
import glob
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile

import slicer
from slicer.ScriptedLoadableModule import (ScriptedLoadableModule, ScriptedLoadableModuleLogic,
                                           ScriptedLoadableModuleTest, ScriptedLoadableModuleWidget)
from slicer.util import VTKObservationMixin

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from fluororeg.backends import bundled_defaults, tools_status  # noqa: E402
from fluororeg.configs import MULTIVIEW_APPS, default_name  # noqa: E402
from fluororeg.project import nifti_suffix, sanitize_case_id, staged_paths  # noqa: E402
from fluororeg.register import IMAGE_RE  # noqa: E402
from fluororeg.settings import (KEYS, Settings, flag_of, read_user_settings, user_settings_path,
                               write_user_settings)  # noqa: E402
from fluororeg.status import InputError, PREFIX, parse_status_line  # noqa: E402
from fluororeg.xrays import is_dicom, is_image, natural_key  # noqa: E402
from fluororeg_viewer import RunChecks, RunViewer, TEST_RUNS_ENV, test_run_dirs  # noqa: E402
from fluororeg_runs import XregRun  # noqa: E402

PROJECTION_ID_ATTR = "FluoroReg.ProjectionId"
EXPERIMENT_ATTR = "FluoroReg.Experiment"
DICOM_UIDS_ATTR = "DICOM.instanceUIDs"
POSE_PREFIX = "BIGSS_POSE"
SETTINGS_GROUP = "FluoroReg"
EXPORT_DIR = "meta_data/tmp/slicer_export"
XRAY_DIR = "meta_data/XrayDICOM"
LOG_MAX_LINES = 5000
SYNC_TIMEOUT_S = 60
STOP_KILL_MS = 15000  # allow runner.py's 10 s child-group cleanup before killing the launcher group
SECTION_MIN_WIDTH = 440
# Tool paths (key, label, default, kind); kind: "dir", "exe" (path or a name on PATH) or "text". Stored in the
# user settings file shared with the terminal (~/.config/fluororeg/settings.json) and passed as flags.
SETTINGS_FIELDS = (
    ("xreg_dir", "xreg repo", "", "dir"),
    ("xreg_build_dir", "xreg build dir", "", "dir"),
    ("annotation_3d_dir", "3D annotation source override", "", "dir"),
    ("conda", "conda executable", "", "exe"),
    ("annotation_3d_env", "Annotation named env override", "", "text"),
    ("annotation_3d_env_prefix", "Annotation env prefix override", "", "text"),
    ("pose_dir", "Pose source override", "", "dir"),
    ("pose_env_dir", "Pose env directory override", "", "text"),
    ("gpu", "GPU index", "0", "text"),
)
TEMPLATE_KEY = "xray_template"  # a pipeline setting edited in the X-rays panel
OPTIONAL_SETTINGS = ("xreg_build_dir", "conda", "annotation_3d_env", "annotation_3d_env_prefix", "pose_env_dir")
SELECTOR_KINDS = (("ct", "CT"), ("seg", "segmentation"), ("landmarks_3d", "3D landmarks"))
CERTAINTIES = ("high", "medium", "low")  # assessment.json schema 3: complete per-target registration evidence
# Display checkboxes of the shown run (set_display flag, text, default); the DRR overlay goes with its opacity.
# Landmarks start hidden: their labels cover the anatomy and the DRR overlays.
DISPLAY_CHECKS = (("show_all_views", "every view", True), ("camera_centred", "camera-centred", False),
                  ("stages_all_views", "primary stage on every camera", False), ("models", "models", True),
                  ("landmarks", "landmarks", False), ("volume_rendering", "CT volume rendering", False),
                  ("drr_overlay", "Final-result DRR overlay", True))
STEPS = (("settings", "Settings"), ("experiment", "Experiment"), ("ct", "CT"), ("xrays", "X-rays"),
         ("pose", "Pose model"), ("registration", "Registration"), ("results", "Results"))
STEP_MARKS = {"done": "✓", "pending": "○", "warning": "!", "error": "✗"}
# Default registrations: (app, femur, label); the config is configs/xreg/<default_name(app, case, femur)>.json.
DEFAULT_REGISTRATIONS = (
    ("single-target", None, "Single view (pelvis)"),
    ("single-target-multiview", None, "Multiview (pelvis)"),
    ("multi-target", "left", "Single view (pelvis, then left femur)"),
    ("multi-target", "right", "Single view (pelvis, then right femur)"),
    ("multi-target-multiview", "left", "Multiview (pelvis, then left femur)"),
    ("multi-target-multiview", "right", "Multiview (pelvis, then right femur)"),
)
MODEL_COLORS = ((0.90, 0.80, 0.60), (0.40, 0.70, 1.00), (0.50, 0.90, 0.50), (1.00, 0.60, 0.40),
                (0.80, 0.50, 0.90), (1.00, 0.85, 0.30), (0.40, 0.90, 0.90), (0.95, 0.45, 0.65),
                (0.70, 0.70, 0.70), (0.60, 0.55, 1.00))
RUN_DIR_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})\d{2}_(.+)$")
SET_UP_REASON = "set up this folder for the pipeline (Experiment step)"
RELOCATED_NOTE = "experiment_dir points elsewhere; Register sets it to this folder."


def is_set_up(status):
    """Whether a `status` result is of a folder set up for the pipeline (project_manifest.json present); a
    result without the `set_up` field is judged by the file."""
    if not status:
        return False
    if "set_up" in status:
        return bool(status["set_up"])
    project = status.get("project")
    return bool(project) and os.path.isfile(os.path.join(project, "project_manifest.json"))


def run_label(run_dir):
    """'10-05 20:36 single_target_c' for '.../20261005_203645_single_target_c'; the folder name otherwise."""
    name = run_dir.rstrip("/").rsplit("/", 1)[-1]
    m = RUN_DIR_RE.match(name)
    return f"{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)} {m.group(6)}" if m else name


#
# Module
#
class FluoroReg(ScriptedLoadableModule):
    def __init__(self, parent):
        super().__init__(parent)
        parent.title = "FluoroReg"
        parent.categories = ["Registration"]
        parent.dependencies = []
        parent.contributors = ["BIGSS lab"]
        parent.helpText = (
            "Registers C-arm radiographs to CT (fluororeg command line) on an experiment folder with the "
            "CT and X-rays of the Slicer scene: segmentation and 3D landmarks, X-ray import, pose model, "
            "registration with predicted initial poses, assessment. The Results step shows runs in the scene "
            "(views, pose stages, DRR overlay). Opening a run folder or a folder of finished runs without "
            "project_manifest.json views them without writing into the folder.")
        version = f"Slicer-{slicer.app.majorVersion}.{slicer.app.minorVersion}"
        prefix = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
        setup_doc = os.path.join(prefix, "share", version, "SlicerFluoroReg", "docs", "setup.md")
        if not os.path.isfile(setup_doc):
            setup_doc = os.path.join(os.path.dirname(_HERE), "docs", "setup.md")
        if os.path.isfile(setup_doc):
            from pathlib import Path
            parent.helpText += f' <a href="{Path(setup_doc).as_uri()}">Setup and dependency guide</a>.'
        parent.acknowledgementText = ""


#
# Slicer-independent helpers (tested by the self-test)
#
def load_settings(path=None):
    """{key: value} of the tool settings and the X-ray template (defaults when unset) from the user settings
    file ``path`` (default: the CLI's user file)."""
    try:
        resolved = Settings.resolve({}, user_file=path)
        stored = {key: resolved.get(key) for key in KEYS}
    except InputError as exc:
        logging.warning(f"FluoroReg: {exc}")
        stored = {}
    values = {key: stored.get(key) or default for key, _label, default, _kind in SETTINGS_FIELDS}
    values[TEMPLATE_KEY] = stored.get(TEMPLATE_KEY) or ""
    return values


def save_settings(values, path=None):
    """Stores the given pipeline settings in the user settings file when they changed."""
    defaults = bundled_defaults()
    wanted = {key: ("" if values[key] == defaults.get(key) else values[key]) for key in KEYS if key in values}
    stored = read_user_settings(path)
    if wanted and any((stored.get(k) or "") != (v or "") for k, v in wanted.items()):
        write_user_settings(wanted, path)


def setting_check(key, kind, value):
    """(ok, text) of one setting for the inline status."""
    import shutil

    value = (value or "").strip()
    if not value:
        if key in OPTIONAL_SETTINGS:
            return True, "default"
        return False, "missing"
    if kind == "dir":
        return (True, "OK") if os.path.isdir(value) else (False, "missing")
    if kind == "exe":
        return (True, "OK") if shutil.which(value) else (False, "missing")
    if key == "gpu" and not value.isdigit():
        return False, "not a number"
    return True, "OK"


def tool_flags(values):
    out = []
    for key in KEYS:
        value = (values.get(key) or "").strip()
        if value:
            out += [flag_of(key), value]
    return out


def cli_python():
    """Slicer's PythonSlicer (`<Slicer home>/bin/PythonSlicer`), the interpreter of the CLI."""
    name = "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer"
    return os.path.join(os.path.realpath(slicer.app.slicerHome), "bin", name)


def step_args(step, project=None, case=None, input_path=None, inputs=None, model_dir=None, ids=None, config=None,
              app=None, femur=None, name=None, ct=None, seg=None, landmarks_3d=None, experiment_dir=None,
              replay=False, run=None, component="all"):
    """(command arguments, tool flags accepted) of one pipeline step."""
    if step in ("tools-status", "tools-setup"):
        return ["tools", step[len("tools-"):], "--component", component], True
    if step in ("init", "status"):
        return [step, "--project", project], step != "status"
    if step == "import-ct":
        return ["import-ct", "--project", project, "--input", input_path] + (["--case", case] if case else []), True
    if step == "annotate-3d":
        return ["annotate-3d", "--project", project, "--case", case], True
    if step == "import-xrays":
        return ["import-xrays", "--project", project, "--input", *inputs], True
    if step in ("pose-init-config", "pose-train"):
        return ["pose", step[len("pose-"):], "--project", project, "--case", case], True
    if step == "pose-use-model":
        return ["pose", "use-model", "--project", project, "--case", case, "--model-dir", model_dir], True
    if step == "config-create":
        args = ["config", "create", "--project", project, "--case", case, "--app", app]
        if femur:
            args += ["--femur", femur]
        if name:
            args += ["--name", name]
        if ids:
            args += ["--ids", *ids]
        return args, False
    if step == "config-set":
        args = ["config", "set", "--config", config] + (["--experiment-dir", experiment_dir] if experiment_dir else [])
        for flag, value in (("--ct", ct), ("--seg", seg), ("--landmarks-3d", landmarks_3d)):
            if value:
                args += [flag, value]
        if ids:
            args += ["--ids", *ids]
        return args, False
    if step == "config-show":
        return ["config", "show", "--config", config], False
    if step == "register":
        return ["register", "--config", config, "--predict-poses"] + (["--replay"] if replay else []), True
    if step == "assess":
        return ["assess", "--run", run], True
    raise ValueError(f"unknown step {step}")


def cli_argv(values, step, python=None, **kwargs):
    """The full argv of one step: <python> -m fluororeg <args> [tool flags]."""
    args, tools = step_args(step, **kwargs)
    return [python or cli_python(), "-m", "fluororeg", *args] + (tool_flags(values) if tools else [])


def flag_counts_text(flags):
    """'3 warn, 1 error' from the manifest summary's {severity: count}; empty when there are none."""
    if not isinstance(flags, dict):
        return ""
    return ", ".join(f"{n} {severity}" for severity, n in flags.items() if n)


def command_line(argv):
    """A shell line reproducing argv in a terminal."""
    return f"PYTHONPATH={shlex.quote(_HERE)} " + " ".join(shlex.quote(a) for a in argv)


def parse_line(line):
    """("pipeline"|"pose"|"xreg", record) of a status line, else None."""
    record = parse_status_line(line, PREFIX)
    if record is not None:
        return "pipeline", record
    record = parse_status_line(line, POSE_PREFIX)
    if record is not None:
        return "pose", record
    m = IMAGE_RE.search(line)
    if m:
        return "xreg", {"event": "image", "id": m.group(1), "status": m.group(2), "reason": m.group(3)}
    return None


def progress_update(label, source, record):
    """(text "<label>: <phase> k/n", done, total) of a progress record (done/total None when unknown), or None.
    The phase is the record's `phase`, the annotation stage (`message` of annotate-3d) or the pose stage."""
    if record.get("event") != "progress":
        return None
    if source == "pipeline":
        phase = record.get("phase") or (record.get("message") if record.get("step") == "annotate-3d" else None) \
            or record.get("step") or ""
    elif source == "pose":
        phase = record.get("stage") or "pose"
    else:
        return None
    phase = str(phase).replace("_", " ")
    done, total = record.get("done", record.get("iteration")), record.get("total")
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        return f"{label}: {phase} {done}/{total}", min(done, total), total
    return f"{label}: {phase}", None, None


def default_case_name(name):
    """The case name of a CT node name: without a trailing _ct/_CT, file-name safe."""
    name = (name or "").strip()
    if name.lower().endswith("_ct"):
        name = name[:-3]
    return sanitize_case_id(name) if name else ""


def project_relative(path, project):
    """`path` relative to `project` (posix separators) when it lies inside it, else None."""
    if not path or not project:
        return None
    root = os.path.realpath(project)
    real = os.path.realpath(path)
    if not real.startswith(root.rstrip(os.sep) + os.sep):
        return None
    return os.path.relpath(real, root).replace(os.sep, "/")


def free_path(directory, stem, suffix):
    """`directory/stem+suffix`, or with _1, _2, ... appended when that exists."""
    path, k = os.path.join(directory, stem + suffix), 1
    while os.path.exists(path):
        path, k = os.path.join(directory, f"{stem}_{k}{suffix}"), k + 1
    return path


def find_case(status, case):
    return next((c for c in (status or {}).get("cases") or [] if c.get("case_id") == case), None) if case else None


def case_staged(entry):
    staged = (entry or {}).get("staged") or {}
    return bool(staged) and all(v.get("present") for v in staged.values())


def pose_model(status, case):
    """(present, "model <name>, self-check 98 %" | "no model") of a case's active pose model."""
    entry = ((status or {}).get("pose") or {}).get(case) or {} if case else {}
    model = entry.get("model") if isinstance(entry.get("model"), dict) else {}
    active = model.get("active") or entry.get("active_model")
    if not active:
        return False, "no model"
    text = f"model {model.get('name') or os.path.basename(str(active).rstrip('/'))}"
    capture = model.get("self_check_capture")
    if isinstance(capture, (int, float)):
        text += f", self-check {capture * 100 if capture <= 1 else capture:.0f} %"
    return True, text


def registration_choices(status, case):
    """The registrations offered for a case: the defaults (whether or not their config exists), then every
    other config of configs/xreg/. Each: {key (config stem), label, app, femur, config (relative), exists,
    multiview}."""
    configs = {e.get("config"): e for e in (status or {}).get("xreg_configs") or [] if e.get("config")}
    out = []
    if case:
        for app, femur, label in DEFAULT_REGISTRATIONS:
            name = default_name(app, case, femur)
            rel = f"configs/xreg/{name}.json"
            out.append({"key": name, "label": label, "app": app, "femur": femur, "config": rel,
                        "exists": rel in configs, "multiview": app in MULTIVIEW_APPS})
    taken = {c["config"] for c in out}
    for rel in sorted(configs):
        if rel in taken:
            continue
        name = os.path.basename(rel)[:-len(".json")] if rel.endswith(".json") else os.path.basename(rel)
        app = configs[rel].get("app")
        out.append({"key": name, "label": name, "app": app, "femur": None, "config": rel, "exists": True,
                    "multiview": app in MULTIVIEW_APPS})
    return out


def registration_reason(status, case, choice, checked_ids, nodes):
    """The first reason Register cannot run, or None: experiment (set up), annotated CT, pose model, registration,
    X-ray selection (two for multiview, converted), scene selectors (the CT a 3D volume)."""
    if not status:
        return "open an experiment folder"
    if not is_set_up(status):
        return SET_UP_REASON
    entry = find_case(status, case)
    if entry is None:
        return "no case: segment and annotate a CT"
    if not case_staged(entry):
        return f"{case}: CT not annotated"
    if not pose_model(status, case)[0]:
        return "no pose model: train one or use an existing one"
    if choice is None:
        return "select a registration"
    if not checked_ids:
        return "check at least one X-ray"
    if choice.get("multiview") and len(checked_ids) < 2:
        return "multiview needs at least two X-rays"
    converted = set(((status or {}).get("xrays") or {}).get("converted") or [])
    missing = [i for i in checked_ids if i not in converted]
    if missing:
        return f"X-ray {', '.join(missing)} not converted to H5"
    for key, what in SELECTOR_KINDS:
        node = (nodes or {}).get(key)
        if node is None:
            return f"select the {what}"
        if key == "ct" and is_xray_like(node):
            return f"the CT must be a 3D volume ({node.GetName()} is a 2D image)"
    return None


def step_checklist(status, case=None, missing_settings=(), registration=None):
    """{step key: {"mark": done|pending|warning|error, "summary": one line}} from a `status` result (None when no
    experiment is open), the case, the labels of the failing required settings and the registration state
    {"label", "reason"} (reason None = ready). A folder not set up for the pipeline shows results only."""
    st = status or {}

    def entry(mark, summary):
        return {"mark": mark, "summary": summary}

    out = {"settings": entry("error", "missing: " + ", ".join(missing_settings)) if missing_settings
           else entry("done", "tool paths OK")}
    out["experiment"] = (entry("done", f"{st.get('project_name')}") if status
                         else entry("pending", "choose an experiment folder"))
    c = find_case(st, case)
    if c is None:
        out["ct"] = entry("pending", "not annotated")
    elif case_staged(c):
        out["ct"] = entry("done", f"{case}: annotated")
    elif (c.get("annotation_output") or {}).get("present"):
        out["ct"] = entry("warning", f"{case}: annotated, not staged")
    else:
        out["ct"] = entry("pending", f"{case}: not annotated")
    xr = st.get("xrays") or {}
    ids, converted = xr.get("ids") or [], set(xr.get("converted") or [])
    unconverted = [i for i in ids if i not in converted]
    if not ids:
        out["xrays"] = entry("pending", "none in the experiment")
    elif unconverted:
        out["xrays"] = entry("warning", f"{len(ids)} in the experiment, {len(unconverted)} not converted")
    else:
        out["xrays"] = entry("done", f"{len(ids)} in the experiment")
    present, text = pose_model(st, case)
    out["pose"] = entry("done" if present else "pending", text)
    reg = registration or {}
    if not status:
        out["registration"] = entry("pending", "open an experiment folder")
    elif reg.get("reason") is None and reg.get("label"):
        out["registration"] = entry("done", f"{reg['label']}: ready")
    else:
        out["registration"] = entry("pending", reg.get("reason") or "select a registration")
    runs = st.get("runs") or []
    if not runs:
        out["results"] = entry("pending", "no runs")
    else:
        latest = max(runs, key=lambda r: (r.get("run_dir") or "").rsplit("/", 1)[-1])
        assessment = latest.get("assessment") or {}
        counts = assessment.get("counts") or {}
        bad = [f"{counts[k]} {k} certainty" for k in ("medium", "low") if counts.get(k)]
        if assessment.get("error"):
            bad.append(assessment["error"])
        if latest.get("error"):
            bad.append(latest["error"])
        out["results"] = entry("warning" if bad else "done",
                               f"{len(runs)} run(s); latest " + (", ".join(bad) if bad else "ok"))
    if status and not is_set_up(status):
        out["experiment"] = entry("warning", "results only")
        for key in ("ct", "xrays", "pose", "registration"):
            out[key] = entry("pending", SET_UP_REASON)
    return out


def config_dir(info):
    """The folder a `config show` result's relative inputs resolve against: `resolved_dir` (the folder holding a
    relocated config), else its experiment_dir."""
    return info.get("resolved_dir") or info.get("experiment_dir") or ""


def config_targets(info):
    """{"ct"|"seg"|"landmarks_3d": absolute path, "ids": [...]} of a `config show` result."""
    exp = config_dir(info)
    out = {}
    for key, _what in SELECTOR_KINDS:
        value = info.get(key)
        out[key] = os.path.join(exp, *value.split("/")) if value else None
    out["ids"] = list(info.get("ids") or [])
    return out


def config_changes(info, nodes, checked_ids, project=None):
    """`config set` fields that differ from the file: {"experiment_dir": the open experiment folder (only for a
    relocated config), "ct"|"seg"|"landmarks_3d": path relative to the experiment folder, "ids": [...]};
    `nodes` {key: node}; ids only when some X-rays are checked."""
    exp = config_dir(info)
    changes = {}
    if info.get("relocated"):
        changes["experiment_dir"] = project or exp
    for key, _what in SELECTOR_KINDS:
        rel = project_relative(node_file(nodes.get(key)), exp)
        if rel and rel != info.get(key):
            changes[key] = rel
    ids = sorted(checked_ids, key=natural_key)
    if ids and ids != list(info.get("ids") or []):
        changes["ids"] = ids
    return changes


def xreg_config_entry(status, rel):
    """The status `xreg_configs` entry of a config (relative path), or {}."""
    return next((e for e in (status or {}).get("xreg_configs") or [] if e.get("config") == rel), {})


def run_sync(argv, env, timeout=SYNC_TIMEOUT_S):
    """(exit status, last result or error record, stderr) of a short CLI call."""
    proc = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
                          universal_newlines=True)
    record = None
    for line in proc.stdout.splitlines():
        parsed = parse_status_line(line)
        if parsed is not None and parsed.get("event") in ("result", "error"):
            record = parsed
    return proc.returncode, record, proc.stderr


#
# Scene helpers
#
def node_file(node):
    storage = node.GetStorageNode() if node is not None else None
    return storage.GetFileName() if storage is not None else None


def unmodified_file(node):
    """The node's storage file when it exists and the node was not modified since it was read, else None."""
    path = node_file(node)
    if path and os.path.isfile(path) and not node.GetModifiedSinceRead():
        return path
    return None


def is_xray_like(node):
    """A scalar volume (not a label map) with a single slice."""
    if node is None or not node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode"):
        return False
    image = node.GetImageData()
    return image is not None and 1 in image.GetDimensions()


def is_ct_like(node):
    """A 3D scalar volume: not a label map, vector volume or X-ray-like."""
    return (node is not None and node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode")
            and not node.IsA("vtkMRMLVectorVolumeNode") and node.GetImageData() is not None
            and not is_xray_like(node))


def ct_export_plan(node, export_dir, case):
    """(path for import-ct, save first): the node's unmodified NIfTI file, else `<export_dir>/<case>.nii.gz`
    (import-ct takes NIfTI files or DICOM folders, so other formats are saved as NIfTI)."""
    path = unmodified_file(node)
    if path and nifti_suffix(os.path.basename(path)):
        return path, False
    return os.path.join(export_dir, f"{case}.nii.gz"), True


def dicom_file_for_instance(uid):
    db = getattr(slicer, "dicomDatabase", None)
    if db is None or not uid:
        return None
    try:
        return db.fileForInstance(uid) or None
    except Exception:  # an unopened database
        return None


def xray_source(node, file_for_instance=dicom_file_for_instance):
    """("dicom" | "image", file) an X-ray node came from: its unmodified DICOM or image storage file, else the
    DICOM database file of its first DICOM instance UID; (None, None) when the image must be exported."""
    path = unmodified_file(node)
    if path:
        if is_dicom(path):
            return "dicom", path
        if is_image(path):
            return "image", path
    uids = (node.GetAttribute(DICOM_UIDS_ATTR) or "").split()
    found = file_for_instance(uids[0]) if uids and file_for_instance else None
    if found and os.path.isfile(found) and is_dicom(found):
        return "dicom", found
    return None, None


def save_xray_png(node, path):
    """Writes the node's single-slice image as a 16-bit grayscale PNG (row 0 = first image row)."""
    import numpy as np
    from PIL import Image

    arr = np.asarray(slicer.util.arrayFromVolume(node), dtype=np.float64)
    if node.IsA("vtkMRMLVectorVolumeNode"):
        arr = arr.mean(axis=-1)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{node.GetName()} is not a single 2D image (shape {arr.shape})")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.clip(np.rint(arr), 0, 65535).astype(np.uint16)).save(path)
    return path


def xray_experiment_id(node, project, file_for_instance=dicom_file_for_instance):
    """The experiment's projection id of a scene node: tagged for this experiment, or loaded from
    meta_data/XrayDICOM/<id>.dcm of it (its storage file or DICOM database file); else None."""
    if node is None or not project:
        return None
    pid, exp = node.GetAttribute(PROJECTION_ID_ATTR), node.GetAttribute(EXPERIMENT_ATTR)
    if pid and exp and os.path.realpath(exp) == os.path.realpath(project):
        return pid
    path = node_file(node)
    uids = (node.GetAttribute(DICOM_UIDS_ATTR) or "").split()
    if not path and uids and file_for_instance:
        path = file_for_instance(uids[0])
    rel = project_relative(path, project)
    prefix = XRAY_DIR + "/"
    if rel and rel.startswith(prefix) and rel.endswith(".dcm") and "/" not in rel[len(prefix):]:
        return rel[len(prefix):-len(".dcm")]
    return None


def tag_xray(node, project, pid):
    node.SetAttribute(PROJECTION_ID_ATTR, pid)
    node.SetAttribute(EXPERIMENT_ATTR, os.path.realpath(project))


def scene_volumes():
    return list(slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"))


def experiment_xray_nodes(project):
    """{projection id: scene node} of the experiment's X-rays in the scene, natural order."""
    found = {}
    for node in scene_volumes():
        pid = xray_experiment_id(node, project)
        if pid and pid not in found:
            found[pid] = node
    return {pid: found[pid] for pid in sorted(found, key=natural_key)}


def same_image(a, b):
    """Same image grid (shape, spacing, origin, direction) and voxels."""
    import numpy as np
    import vtk

    ia, ib = a.GetImageData(), b.GetImageData()
    if ia is None or ib is None or ia.GetDimensions() != ib.GetDimensions():
        return False
    if not (np.allclose(a.GetSpacing(), b.GetSpacing(), atol=1e-5)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=1e-4)):
        return False
    ma, mb = vtk.vtkMatrix4x4(), vtk.vtkMatrix4x4()
    a.GetIJKToRASDirectionMatrix(ma)
    b.GetIJKToRASDirectionMatrix(mb)
    if not np.allclose(slicer.util.arrayFromVTKMatrix(ma), slicer.util.arrayFromVTKMatrix(mb), atol=1e-6):
        return False
    return bool(np.array_equal(slicer.util.arrayFromVolume(a), slicer.util.arrayFromVolume(b)))


def remove_node(node):
    """Removes a node with its display and storage nodes."""
    if node is None or node.GetScene() is None:
        return
    if node.IsA("vtkMRMLDisplayableNode"):
        for i in range(node.GetNumberOfDisplayNodes() - 1, -1, -1):
            d = node.GetNthDisplayNode(i)
            if d is not None:
                slicer.mrmlScene.RemoveNode(d)
    if node.IsA("vtkMRMLStorableNode") and node.GetStorageNode() is not None:
        slicer.mrmlScene.RemoveNode(node.GetStorageNode())
    slicer.mrmlScene.RemoveNode(node)


def case_for_ct(status, node, project):
    """The case entry of a CT node: its file is the case's staged CT or imported input, or (else) its name
    gives the case name; None otherwise."""
    if node is None:
        return None
    cases = (status or {}).get("cases") or []
    path = node_file(node)
    if path and project:
        real = os.path.realpath(path)
        for c in cases:
            for rel in (((c.get("staged") or {}).get("ct") or {}).get("path"), (c.get("input") or {}).get("path")):
                if rel and os.path.realpath(os.path.join(project, *rel.split("/"))) == real:
                    return c
    return find_case(status, default_case_name(node.GetName()))


#
# Logic
#
class FluoroRegLogic(ScriptedLoadableModuleLogic):
    """Scene loading of experiment files."""

    @staticmethod
    def find_node_by_file(class_name, path, exclude_class=None):
        target = os.path.realpath(path)
        for node in slicer.util.getNodesByClass(class_name):
            if exclude_class and node.IsA(exclude_class):
                continue
            f = node_file(node)
            if f and os.path.realpath(f) == target:
                return node
        return None

    def load_file(self, kind, path, name=None):
        """The scene node of an experiment file (loaded unless present); kind: ct, seg, landmarks_3d, model."""
        if kind == "ct":
            node = self.find_node_by_file("vtkMRMLScalarVolumeNode", path, "vtkMRMLLabelMapVolumeNode")
            if node is None:
                node = slicer.util.loadVolume(path, {"name": name} if name else {})
        elif kind == "seg":
            node = self.find_node_by_file("vtkMRMLLabelMapVolumeNode", path)
            if node is None:
                node = slicer.util.loadLabelVolume(path, {"name": name} if name else {})
        elif kind == "model":
            node = self.find_node_by_file("vtkMRMLModelNode", path)
            if node is None:
                node = slicer.util.loadModel(path)
        else:
            node = self.find_node_by_file("vtkMRMLMarkupsFiducialNode", path)
            if node is None:
                node = slicer.util.loadMarkups(path)
                if name:
                    node.SetName(name)
        return node

    def adopt_staged_ct(self, node, staged):
        """The scene node of the staged CT file: `node` itself when it is that file or holds the same image
        (its storage then points at the staged file), else the staged file loaded."""
        existing = self.find_node_by_file("vtkMRMLScalarVolumeNode", staged, "vtkMRMLLabelMapVolumeNode")
        if existing is not None:
            return existing
        if node is None:
            return slicer.util.loadVolume(staged)
        loaded = slicer.util.loadVolume(staged, {"show": False})
        if not same_image(node, loaded):
            return loaded
        remove_node(loaded)
        if node.GetStorageNode() is None:
            node.AddDefaultStorageNode()
        node.GetStorageNode().SetFileName(staged)
        return node

    def load_case_outputs(self, project, case, ct_node=None):
        """Staged CT (adopting `ct_node`), coloured label map, 3D landmarks and every meta_data/models/<case>_*.stl
        (distinct colours, visible in 3D). Returns {"ct", "seg", "landmarks_3d", "models"}."""
        paths = {k: os.path.join(project, *v.split("/")) for k, v in staged_paths(case).items()}
        ct = self.adopt_staged_ct(ct_node, paths["ct"])
        seg = self.load_file("seg", paths["seg"], name=f"{case}_seg")
        landmarks = self.load_file("landmarks_3d", paths["landmarks_3d"], name=f"{case}_landmarks_3d")
        models = []
        stls = sorted(glob.glob(os.path.join(glob.escape(os.path.join(project, "meta_data", "models")),
                                             glob.escape(case) + "_*.stl")), key=natural_key)
        for i, path in enumerate(stls):
            model = self.load_file("model", path)
            display = model.GetDisplayNode()
            if display is not None:
                display.SetColor(*MODEL_COLORS[i % len(MODEL_COLORS)])
                display.SetVisibility(True)
            models.append(model)
        slicer.util.setSliceViewerLayers(background=ct, label=seg, fit=True)
        return {"ct": ct, "seg": seg, "landmarks_3d": landmarks, "models": models}

    @staticmethod
    def load_xray_file(path):
        """Loads one X-ray file (DICOM or image) into the scene, not shown."""
        name = os.path.splitext(os.path.basename(path))[0]
        return slicer.util.loadVolume(path, {"singleFile": True, "show": False, "name": name})

    def load_xray(self, project, pid):
        """The scene node of experiment X-ray <pid> (meta_data/XrayDICOM/<pid>.dcm loaded and tagged if needed)."""
        node = experiment_xray_nodes(project).get(pid)
        if node is None:
            node = self.load_xray_file(os.path.join(project, *XRAY_DIR.split("/"), f"{pid}.dcm"))
            node.SetName(f"xray-{pid}")
            tag_xray(node, project, pid)
        return node


class Job:
    def __init__(self, label, argv, on_done=None):
        self.label = label
        self.argv = argv
        self.on_done = on_done
        self.records = []
        self.stop_requested = False

    def result(self, step=None):
        found = [r for r in self.records if r.get("event") == "result" and (step is None or r.get("step") == step)]
        return found[-1] if found else None


#
# Widget
#
class FluoroRegWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)

    # ---------------------------------------------------------------- setup
    def setup(self):
        super().setup()
        import ctk
        import qt

        self.logic = FluoroRegLogic()
        self.status = None
        self.job = None
        self.queue = []
        self.process = None
        self.last_event = None
        self.last_error = None
        self.last_exit_code = None
        self.last_argv = None
        self.messages = {}
        self._buffers = {"out": b"", "err": b""}
        self._xray_checked = set()   # scene node ids checked in the X-rays panel
        self._xray_seen = set()      # scene node ids listed before (new ones not in the experiment start checked)
        self._checked = set()        # projection ids checked for the registration
        self._loaded_outputs = set()  # (project, case, CT node id) whose annotation was loaded
        self._updating = False
        self._rebuild_pending = False
        self._ct_filter_pending = False
        self.run_viewer = RunViewer()  # .scene: the RunScene of the shown run (None when none is shown)
        self.shown_run = None         # its directory relative to the experiment folder
        self._loading_run = False

        def named(widget, name):
            widget.objectName = f"pipeline.{name}"
            return widget

        stretched = []

        def page(key, title, stretch=True):
            """A step panel in the stack: (panel, vertical box, form on top). With `stretch` the form stays at
            the top; without, widgets added to the box with a stretch factor take the remaining height."""
            panel = named(qt.QWidget(), f"panel.{key}")
            box = qt.QVBoxLayout(panel)
            box.setContentsMargins(0, 4, 0, 0)
            panel.accessibleName = title
            form = qt.QFormLayout()
            box.addLayout(form)
            self.pages[key] = panel
            self.stack.addWidget(panel)
            if stretch:
                stretched.append(box)
            return panel, box, form

        def button(name, text, slot):
            b = named(qt.QPushButton(text), name)
            b.connect("clicked(bool)", lambda _=False: slot())
            return b

        def label(name, text="", wrap=True):
            w = named(qt.QLabel(text), name)
            w.wordWrap = wrap
            return w

        def message(name):
            w = label(name)
            w.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
            w.visible = False
            return w

        def row(*widgets, stretch=None):
            box = qt.QHBoxLayout()
            for w in widgets:
                box.addWidget(w, 1 if w is stretch else 0)
            return box

        # Slicer's module panel sizes its content to the (height-for-width) size hint of every panel in the
        # stack, so the scrolling views ask for little; they still take the free height (stretch factor).
        class CompactTable(qt.QTableWidget):
            def sizeHint(self):
                return qt.QSize(SECTION_MIN_WIDTH, 90)

        class CompactList(qt.QListWidget):
            def sizeHint(self):
                return qt.QSize(SECTION_MIN_WIDTH, 60)

        class CompactText(qt.QPlainTextEdit):
            def sizeHint(self):
                return qt.QSize(SECTION_MIN_WIDTH, 120)

        def table(name, headers):
            t = named(CompactTable(), name)
            t.setColumnCount(len(headers))
            t.setHorizontalHeaderLabels(headers)
            t.verticalHeader().visible = False
            t.verticalHeader().setDefaultSectionSize(22)
            t.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
            t.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
            t.setSelectionMode(qt.QAbstractItemView.SingleSelection)
            t.horizontalHeader().setSectionResizeMode(qt.QHeaderView.ResizeToContents)
            t.horizontalHeader().setStretchLastSection(True)
            t.setMinimumHeight(90)
            return t

        def checklist(name):
            w = named(CompactList(), name)
            w.setMinimumHeight(66)
            return w

        # Status bar on top, so a running step stays in view whichever panel is shown
        self.pages = {}
        self.current_step = None
        self.checklist_state = {}
        status_box = named(qt.QFrame(), "statusBar")
        status_box.setMinimumWidth(SECTION_MIN_WIDTH)
        status_rows = qt.QVBoxLayout(status_box)
        status_rows.setContentsMargins(0, 0, 0, 0)
        status_row = qt.QHBoxLayout()
        self.status_label = label("statusLabel", "idle")
        self.status_label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        self.progress = named(qt.QProgressBar(), "progressBar")
        self.progress.setRange(0, 1)
        self.progress.value = 0
        self.progress.textVisible = False
        self.progress.format = "%v / %m"
        self.progress.visible = False  # shown while a command runs, full width under the status line
        self.stop_button = button("stopButton", "Stop", self.stop)
        self.log_button = named(qt.QPushButton("Log"), "logButton")
        self.log_button.checkable = True
        status_row.addWidget(label("statusTitle", "Status:", wrap=False))
        status_row.addWidget(self.status_label, 1)
        for w in (self.stop_button, self.log_button):
            status_row.addWidget(w)
        status_rows.addLayout(status_row)
        status_rows.addWidget(self.progress)
        self.layout.addWidget(status_box)

        # Checklist: one row per step (mark, step button, summary); the buttons pick the panel shown below
        check_box = named(qt.QFrame(), "checklist")
        check_box.setMinimumWidth(SECTION_MIN_WIDTH)
        check_box.frameShape = qt.QFrame.StyledPanel
        grid = qt.QGridLayout(check_box)
        grid.setContentsMargins(4, 2, 4, 2)
        grid.setVerticalSpacing(0)
        grid.setColumnStretch(2, 1)
        self.step_group = qt.QButtonGroup(check_box)
        self.step_group.exclusive = True
        self.step_buttons = {"log": self.log_button}
        self.step_marks = {}
        self.step_summaries = {}
        for r, (key, title) in enumerate(STEPS):
            mark = label(f"stepMark.{key}", wrap=False)
            mark.setMinimumWidth(16)
            mark.alignment = qt.Qt.AlignCenter
            b = named(qt.QPushButton(title), f"step.{key}")
            b.checkable = True
            b.flat = True
            b.setStyleSheet("text-align: left; padding: 1px 6px")
            summary = label(f"stepSummary.{key}", wrap=False)
            summary.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)  # long text never widens
            grid.addWidget(mark, r, 0)
            grid.addWidget(b, r, 1)
            grid.addWidget(summary, r, 2)
            self.step_group.addButton(b)
            self.step_buttons[key] = b
            self.step_marks[key] = mark
            self.step_summaries[key] = summary
        self.step_group.addButton(self.log_button)
        for key, b in self.step_buttons.items():
            b.connect("clicked(bool)", lambda _=False, k=key: self.show_step(k))
        self.layout.addWidget(check_box)

        self.stack = named(qt.QStackedWidget(), "stepPanels")
        self.stack.setMinimumWidth(SECTION_MIN_WIDTH)  # keeps tables readable when Slicer narrows the panel
        self.layout.addWidget(self.stack, 1)

        # 1. Settings: tool paths
        stored = load_settings()
        _panel, _box, form = page("settings", "Settings: tool paths")
        self.setting_edits = {}
        self.setting_checks = {}
        for key, text, _default, kind in SETTINGS_FIELDS:
            if kind in ("dir", "exe"):
                edit = ctk.ctkPathLineEdit()
                edit.filters = (ctk.ctkPathLineEdit.Dirs if kind == "dir"
                                else ctk.ctkPathLineEdit.Files | ctk.ctkPathLineEdit.Executable)
                edit.currentPath = stored[key]
                edit.connect("currentPathChanged(QString)", lambda _t: self._on_setting_changed())
            else:
                edit = qt.QLineEdit(stored[key])
                edit.connect("textChanged(QString)", lambda _t: self._on_setting_changed())
            named(edit, f"setting.{key}")
            check = label(f"setting.{key}.check", wrap=False)
            check.setMinimumWidth(50)
            self.setting_edits[key] = edit
            self.setting_checks[key] = check
            form.addRow(f"{text}:", row(edit, check, stretch=edit))
        form.addRow(label("settingsFileLabel", f"Stored in {user_settings_path()} (shared with the terminal "
                                               f"scripts). The command line runs with Slicer's PythonSlicer."))
        self.backend_status_label = label("backendStatusLabel")
        self.backend_component = named(qt.QComboBox(), "backendComponent")
        self.backend_component.addItems(["all", "pose", "annotation"])
        self.backend_status_button = button("backendStatusButton", "Check environments", self.check_backend_tools)
        self.backend_setup_button = button("backendSetupButton", "Set up environments", self.setup_backend_tools)
        form.addRow("Backend runtimes:", self.backend_status_label)
        form.addRow(row(self.backend_component, self.backend_status_button, self.backend_setup_button))
        form.addRow(label("backendSetupNote", "Setup is explicit and may download dependencies and model weights. "
                         "Environments and caches are stored outside the extension. Stop cancels the setup job. "
                         "xreg remains a separate manual installation."))
        self._refresh_setting_checks()

        # 2. Experiment
        _panel, _box, form = page("experiment", "Experiment")
        self.project_edit = named(qt.QLineEdit(), "projectPath")
        self.project_edit.placeholderText = "experiment folder"
        self.project_edit.connect("editingFinished()", lambda: self.set_project(self.project_edit.text))
        self.project_browse_button = button("projectBrowseButton", "Choose…", self._browse_project)
        form.addRow("Experiment folder:", row(self.project_edit, self.project_browse_button, stretch=self.project_edit))
        self.case_row = named(qt.QWidget(), "caseRow")
        case_box = qt.QHBoxLayout(self.case_row)
        case_box.setContentsMargins(0, 0, 0, 0)
        self.case_combo = named(qt.QComboBox(), "caseCombo")
        self.case_combo.sizeAdjustPolicy = qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.case_combo.minimumContentsLength = 10
        self.case_combo.connect("currentIndexChanged(int)", lambda _i: self._on_case_changed())
        case_box.addWidget(label("caseTitle", "Case:", wrap=False))
        case_box.addWidget(self.case_combo, 1)
        self.case_row.visible = False
        form.addRow(self.case_row)
        self.set_up_button = button("experimentSetUpButton", "Set up for the pipeline", self.set_up_experiment)
        self.set_up_button.toolTip = "Runs init: writes project_manifest.json and the folder layout"
        self.set_up_button.visible = False
        form.addRow(row(self.set_up_button))
        self.project_label = message("experimentMessage")
        form.addRow(self.project_label)
        self.set_up_notes = {}

        def set_up_note(key, form):
            """The panel's note shown when the folder is open for viewing only."""
            note = label(f"setUpNote.{key}", f"{SET_UP_REASON[0].upper()}{SET_UP_REASON[1:]}.")
            note.setStyleSheet("color: #d08000")
            note.visible = False
            form.addRow(note)
            self.set_up_notes[key] = note

        # 3. CT
        _panel, _box, form = page("ct", "CT")
        set_up_note("ct", form)
        self.ct_combo = named(qt.QComboBox(), "ctVolume")
        self.ct_combo.sizeAdjustPolicy = qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.ct_combo.minimumContentsLength = 12
        self.ct_combo.connect("currentIndexChanged(int)", lambda _i: self._on_ct_changed())
        self.ct_load_button = button("ctLoadFileButton", "Load file…", self._browse_ct_file)
        form.addRow("CT volume:", row(self.ct_combo, self.ct_load_button, stretch=self.ct_combo))
        self.ct_case_edit = named(qt.QLineEdit(), "ctCaseName")
        self.ct_case_edit.connect("textChanged(QString)", lambda _t: self._update_buttons())
        form.addRow("Case name:", self.ct_case_edit)
        self.ct_info_label = label("ctInfo")
        form.addRow(self.ct_info_label)
        self.annotate_button = button("annotateButton", "Segment and annotate", self.run_annotate)
        form.addRow(row(self.annotate_button))
        self.ct_label = message("ctMessage")
        form.addRow(self.ct_label)

        # 4. X-rays (the scene's single-slice volumes)
        _panel, box, form = page("xrays", "X-rays", stretch=False)
        set_up_note("xrays", form)
        form.addRow(label("xraySceneTitle", "X-rays in the scene (checked ones are added):"))
        self.xray_scene_list = checklist("xraySceneList")
        self.xray_scene_list.connect("itemChanged(QListWidgetItem*)", self._on_xray_scene_item_changed)
        box.addWidget(self.xray_scene_list, 1)
        self.add_files_button = button("xrayAddFilesButton", "Add files…", self._browse_xray_files)
        self.add_to_experiment_button = button("xrayAddToExperimentButton", "Add to experiment",
                                               self.add_xrays_to_experiment)
        box.addLayout(row(self.add_files_button, self.add_to_experiment_button))
        template_form = qt.QFormLayout()
        self.template_edit = named(ctk.ctkPathLineEdit(), "xrayTemplate")
        self.template_edit.filters = ctk.ctkPathLineEdit.Files
        self.template_edit.currentPath = stored[TEMPLATE_KEY]
        self.template_edit.connect("currentPathChanged(QString)", lambda _t: self._on_setting_changed())
        template_form.addRow("Template DICOM (for images):", self.template_edit)
        box.addLayout(template_form)
        self.xray_summary_label = label("xraySummary")
        box.addWidget(self.xray_summary_label)
        self.xray_label = message("xrayMessage")
        box.addWidget(self.xray_label)

        # 5. Pose model
        _panel, _box, form = page("pose", "Pose model")
        set_up_note("pose", form)
        form.addRow(label("poseHelp", "Predicts each X-ray's initial pose for registration; one model per CT."))
        self.pose_model_label = label("poseModel")
        form.addRow(self.pose_model_label)
        self.pose_train_button = button("poseTrainButton", "Train (~8 min)", self.run_pose_train)
        self.pose_use_button = button("poseUseModelButton", "Use existing model…", self._browse_pose_model)
        self.pose_use_button.toolTip = ("Activate a trained pose model: pick its folder (holding model.pth) or a "
                                        "case's models folder (e.g. meta_data/pose_models/<case>), whose active or "
                                        "only model is used")
        self.pose_config_button = button("poseEditSettingsButton", "Edit pose config", self.open_pose_config)
        form.addRow(row(self.pose_train_button, self.pose_use_button, self.pose_config_button))
        self.pose_reason_label = label("poseReason")
        form.addRow(self.pose_reason_label)
        self.pose_label = message("poseMessage")
        form.addRow(self.pose_label)

        # 6. Registration
        _panel, box, form = page("registration", "Registration", stretch=False)
        set_up_note("registration", form)
        self.reg_combo = named(qt.QComboBox(), "registrationCombo")
        self.reg_combo.sizeAdjustPolicy = qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.reg_combo.minimumContentsLength = 12
        self.reg_combo.connect("currentIndexChanged(int)", lambda _i: self._on_reg_combo())
        self.config_open_button = button("editConfigButton", "Edit config", self.open_config)
        form.addRow("Registration:", row(self.reg_combo, self.config_open_button, stretch=self.reg_combo))
        self.reg_info_label = label("registrationInfo")
        form.addRow(self.reg_info_label)
        self.reg_relocated_label = label("registrationRelocated")
        self.reg_relocated_label.visible = False
        form.addRow(self.reg_relocated_label)
        form.addRow(label("registrationXraysTitle", "X-rays:"))
        self.xray_list = checklist("xrayList")
        self.xray_list.connect("itemChanged(QListWidgetItem*)", self._on_xray_item_changed)
        box.addWidget(self.xray_list, 1)
        sel_form = qt.QFormLayout()
        self.selectors = {}
        for key, what in SELECTOR_KINDS:
            combo = named(slicer.qMRMLNodeComboBox(), f"selector.{key}")
            if key == "ct":
                combo.nodeTypes = ["vtkMRMLScalarVolumeNode"]
                combo.hideChildNodeTypes = ["vtkMRMLLabelMapVolumeNode", "vtkMRMLVectorVolumeNode"]
            elif key == "seg":
                combo.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
            else:
                combo.nodeTypes = ["vtkMRMLMarkupsFiducialNode"]
            combo.noneEnabled = True
            combo.addEnabled = False
            combo.removeEnabled = False
            combo.renameEnabled = False
            combo.setMRMLScene(slicer.mrmlScene)
            combo.setCurrentNode(None)
            # The selection changes only by choice: wheeling over the list while scrolling the panel does not
            # change it, and removing the selected node empties it instead of moving it to a neighbouring node.
            inner = combo.findChild(ctk.ctkComboBox)
            if inner is not None:
                inner.scrollWheelEffect = ctk.ctkComboBox.ScrollWithFocus
                inner.focusPolicy = qt.Qt.StrongFocus
            combo.focusPolicy = qt.Qt.StrongFocus
            combo.connect("nodeAboutToBeRemoved(vtkMRMLNode*)",
                          lambda node, c=combo: c.setCurrentNode(None) if c.currentNodeID == node.GetID() else None)
            combo.connect("currentNodeChanged(vtkMRMLNode*)", lambda _n: self._refresh_registration())
            self.selectors[key] = combo
            sel_form.addRow(f"{what[0].upper() + what[1:]}:", combo)
        box.addLayout(sel_form)
        self.reg_reason_label = label("registrationReason")
        self.reg_reason_label.setStyleSheet("color: #d08000")
        box.addWidget(self.reg_reason_label)
        self.show_result_check = named(qt.QCheckBox("Show result"), "showResultCheck")
        self.show_result_check.checked = True
        self.show_result_check.toolTip = "Show the new run in the Results step"
        self.replay_check = named(qt.QCheckBox("Replay videos"), "replayCheck")
        self.replay_check.checked = False
        self.replay_check.toolTip = "Also render the optimizer replay videos of every view (slower)"
        box.addLayout(row(self.show_result_check, self.replay_check))
        self.register_button = button("registerButton", "Register", self.run_register)
        self.register_button.setMinimumHeight(36)
        font = self.register_button.font
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        self.register_button.font = font
        box.addWidget(self.register_button)
        self.reg_label = message("registrationMessage")
        box.addWidget(self.reg_label)

        # 7. Results: the runs, their assessment, and the display of the run shown in the scene
        _panel, box, form = page("results", "Results", stretch=False)
        self.runs_table = table("runsTable", ["run", "app", "views", "high", "med", "low", "flags"])
        for col, tip in enumerate(["run directory", "xreg app", "views registered / views",
                                   *[f"results with {c} certainty of correct registration (assessment.json)"
                                     for c in CERTAINTIES], "manifest flags"]):
            self.runs_table.horizontalHeaderItem(col).setToolTip(tip)
        header = self.runs_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        self.runs_table.setTextElideMode(qt.Qt.ElideRight)
        self.runs_table.connect("itemSelectionChanged()", self._on_run_selected)
        box.addWidget(self.runs_table, 1)
        self.show_run_button = button("showRunButton", "Show run", self.show_run)
        self.clear_run_button = button("clearRunButton", "Clear", self.clear_run)
        self.assess_button = button("assessRunButton", "Assess", self.run_assess)
        box.addLayout(row(self.show_run_button, self.clear_run_button, self.assess_button))
        self.items_table = table("itemsTable", ["view", "target", "mover-stage objective", "gNCC", "certainty", "reasons"])
        self.items_table.horizontalHeaderItem(2).setToolTip(
            "Latest completed stage moving this target; includes shared-image and held-anatomy terms, "
            "not an independent single-target score. High requires evidence on every registered view.")
        box.addWidget(self.items_table, 1)
        self.results_label = message("resultsMessage")
        box.addWidget(self.results_label)
        view_form = qt.QFormLayout()
        view_form.setContentsMargins(0, 0, 0, 0)
        self.view_combo = named(qt.QComboBox(), "viewerView")
        self.view_combo.sizeAdjustPolicy = qt.QComboBox.AdjustToMinimumContentsLengthWithIcon
        self.view_combo.minimumContentsLength = 12
        self.view_combo.connect("currentIndexChanged(int)", lambda _i: self._on_view_combo())
        self.look_button = button("viewerLookFromSourceButton", "Look from source", self.look_from_source)
        self.overview_button = button("viewerOverviewButton", "Overview", self.overview)
        view_form.addRow("View:", row(self.view_combo, self.look_button, self.overview_button, stretch=self.view_combo))
        self.stage_widget = named(qt.QWidget(), "viewerStages")
        self.stage_grid = qt.QGridLayout(self.stage_widget)
        self.stage_grid.setContentsMargins(0, 0, 0, 0)
        self.stage_combos, self._stage_labels = {}, []
        view_form.addRow("Pose stage:", self.stage_widget)
        box.addLayout(view_form)
        checks = qt.QGridLayout()
        checks.setVerticalSpacing(0)
        self.display_checks = {}
        for i, (key, text, on) in enumerate(DISPLAY_CHECKS):
            c = named(qt.QCheckBox(text), f"viewer.{key}")
            c.checked = on
            c.connect("toggled(bool)", lambda value, k=key: None if self._updating
                      else self.set_display(**{k: value}))
            self.display_checks[key] = c
            if key != "drr_overlay":
                checks.addWidget(c, i // 2, i % 2)
        self.drr_opacity_slider = named(qt.QSlider(qt.Qt.Horizontal), "viewer.drr_opacity")
        self.drr_opacity_slider.setRange(0, 100)
        self.drr_opacity_slider.singleStep = 5
        self.drr_opacity_slider.value = 50
        self.drr_opacity_slider.toolTip = "DRR overlay opacity"
        self.drr_opacity_slider.connect("valueChanged(int)", lambda v: None if self._updating
                                        else self.set_display(drr_opacity=v / 100.0))
        drr_index = [k for k, _t, _o in DISPLAY_CHECKS].index("drr_overlay")
        checks.addLayout(row(self.display_checks["drr_overlay"], self.drr_opacity_slider,
                             stretch=self.drr_opacity_slider), drr_index // 2, drr_index % 2)
        box.addLayout(checks)
        self.viewer_info_label = label("viewerInfo")
        box.addWidget(self.viewer_info_label)
        self.viewer_controls = [self.view_combo, self.stage_widget, self.drr_opacity_slider, self.look_button,
                                self.overview_button, *self.display_checks.values()]

        # Log (status bar's Log button; not a checklist step)
        _panel, box, form = page("log", "Log", stretch=False)
        self.command_edit = named(qt.QLineEdit(), "commandLine")
        self.command_edit.readOnly = True
        form.addRow("Command:", self.command_edit)
        self.log_edit = named(CompactText(), "log")
        self.log_edit.readOnly = True
        self.log_edit.setMaximumBlockCount(LOG_MAX_LINES)
        self.log_edit.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
        self.log_edit.setMinimumHeight(120)
        self.log_edit.setFont(qt.QFontDatabase.systemFont(qt.QFontDatabase.FixedFont))
        box.addWidget(self.log_edit, 1)
        for b in stretched:
            b.addStretch(1)

        self._message_labels = {"experiment": self.project_label, "ct": self.ct_label, "xray": self.xray_label,
                                "pose": self.pose_label, "registration": self.reg_label,
                                "results": self.results_label}
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent, self._on_scene_changed)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self._on_scene_changed)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self._on_scene_changed)
        self._rebuild_scene_lists()
        self._fill_viewer()
        if self.project:
            self.set_project(self.project)
        else:
            self._fill_from_status()
        self.show_step(next((k for k, _t in STEPS if self.checklist_state[k]["mark"] != "done"), STEPS[0][0]))

    def cleanup(self):
        self.removeObservers()
        self._closed = True  # a queued scene-list rebuild must not touch the deleted widgets
        if self.process is not None:
            self.stop()
        self.run_viewer.clear()

    # ---------------------------------------------------------------- helpers
    def _setting(self, key):
        edit = self.template_edit if key == TEMPLATE_KEY else self.setting_edits[key]
        return (edit.currentPath if hasattr(edit, "currentPath") else edit.text).strip()

    def settings_values(self):
        values = {key: self._setting(key) for key, _l, _d, _k in SETTINGS_FIELDS}
        values[TEMPLATE_KEY] = self._setting(TEMPLATE_KEY)
        return values

    def set_settings(self, values):
        """Fills (and stores) settings from a {key: value} dict (keys of SETTINGS_FIELDS and xray_template)."""
        for key, value in values.items():
            edit = self.template_edit if key == TEMPLATE_KEY else self.setting_edits[key]
            if hasattr(edit, "currentPath"):
                edit.currentPath = str(value)
            else:
                edit.text = str(value)
        self._on_setting_changed()

    def set_xray_template(self, path):
        self.set_settings({TEMPLATE_KEY: path})

    def _on_setting_changed(self):
        try:
            save_settings(self.settings_values())
        except (OSError, InputError) as exc:
            logging.error(f"FluoroReg: settings not saved: {exc}")
        self._refresh_setting_checks()
        self._refresh_checklist()

    def _refresh_setting_checks(self):
        defaults = bundled_defaults()
        for key, _label, _default, kind in SETTINGS_FIELDS:
            ok, text = setting_check(key, kind, self._setting(key))
            if ok and key in ("annotation_3d_dir", "pose_dir"):
                text = "bundled" if self._setting(key) == defaults.get(key) else "external"
            w = self.setting_checks[key]
            w.text = text
            w.setStyleSheet("color: #3a9a3a" if ok else "color: #d04040")
        if hasattr(self, "backend_status_label"):
            try:
                state = tools_status(Settings.resolve({key: self._setting(key) for key, _l, _d, _k in SETTINGS_FIELDS}),
                                     probe=False)
                self.backend_status_label.text = "\n".join(
                    f"{name}: {'runtime configured; Check environments to verify' if item['ready'] else item['problem']}"
                    for name, item in state.items())
            except InputError as exc:
                self.backend_status_label.text = str(exc)

    def check_backend_tools(self):
        if self._start([self._job("Check backend environments", "tools-status",
                                 on_done=self._backend_tools_done, component=self.backend_component.currentText)]):
            self.backend_status_label.text = "Checking environments; see Log for progress."

    def setup_backend_tools(self):
        if self._start([self._job("Set up backend environments", "tools-setup",
                                 on_done=self._backend_tools_done, component=self.backend_component.currentText)]):
            self.backend_status_label.text = "Setup running; see Log for progress."

    def _backend_tools_done(self, code, job):
        self._refresh_setting_checks()
        result = job.result("tools") or {}
        if result.get("components"):
            self.backend_status_label.text = "\n".join(
                f"{name}: {'ready' if item['ready'] else item['problem']}"
                for name, item in result["components"].items())
        elif code and job.stop_requested:
            self.backend_status_label.text = "Stopped by user; run the command again when ready."
        elif code:
            self.backend_status_label.text = (self.last_error.get("message", "Setup failed; see Log")
                                              if self.last_error else "Setup failed; see Log")

    @property
    def project(self):
        return self.project_edit.text.strip().rstrip(os.sep)

    @property
    def case(self):
        return self.case_combo.currentText.strip()

    def _has_experiment(self):
        """An open folder set up for the pipeline."""
        return bool(self.project) and is_set_up(self.status)

    def _view_only(self):
        """An open folder without project_manifest.json: results only, nothing is written into it."""
        return self.status is not None and not is_set_up(self.status)

    def _msg(self, where, text, error=False):
        w = self._message_labels[where]
        self.messages[where] = {"text": text, "error": error} if text else None
        w.text = text or ""
        w.setStyleSheet("color: #d04040" if error else "")
        w.visible = bool(text)

    def _env_dict(self):
        env = dict(slicer.util.startupEnvironment())
        env["PYTHONPATH"] = _HERE
        return env

    def _log(self, text):
        self.log_edit.appendPlainText(text)

    def _node(self, node_id):
        return slicer.mrmlScene.GetNodeByID(node_id) if node_id else None

    def ct_node(self):
        return self._node(self.ct_combo.itemData(self.ct_combo.currentIndex))

    def ct_case_name(self):
        return sanitize_case_id(self.ct_case_edit.text) if self.ct_case_edit.text.strip() \
            else self.ct_case_edit.placeholderText

    def _pipeline_volumes(self):
        """Scene volumes except the nodes of the shown run (its X-rays, DRRs and CT are not pipeline inputs)."""
        scene = self.run_viewer.scene
        owned = {n.GetID() for n in scene.nodes} if scene is not None else set()
        return [n for n in scene_volumes() if n.GetID() not in owned]

    def xray_scene_nodes(self):
        return [n for n in self._pipeline_volumes() if is_xray_like(n)]

    def checked_xray_nodes(self):
        return [n for n in self.xray_scene_nodes() if n.GetID() in self._xray_checked]

    def experiment_ids(self):
        return list(((self.status or {}).get("xrays") or {}).get("ids") or [])

    def checked_ids(self):
        present = set(self.experiment_ids())
        return sorted(self._checked & present, key=natural_key)

    def selector_nodes(self):
        return {key: combo.currentNode() for key, combo in self.selectors.items()}

    def registration_choice(self):
        key = self.reg_combo.itemData(self.reg_combo.currentIndex)
        return next((c for c in registration_choices(self.status, self.case) if c["key"] == key), None)

    def _config_path(self, choice=None):
        choice = choice or self.registration_choice()
        return os.path.join(self.project, *choice["config"].split("/")) if choice else ""

    def state(self):
        """JSON-serialisable state for scripting (poll it while a command runs)."""
        choice = self.registration_choice()
        ct = self.ct_node()
        return {
            "busy": self.process is not None,
            "step": self.current_step,
            "checklist": self.checklist_state,
            "job": self.job.label if self.job else None,
            "argv": self.last_argv,
            "command_line": self.command_edit.text,
            "queue": [j.label for j in self.queue],
            "status_text": self.status_label.text,
            "progress": [self.progress.value, self.progress.maximum],
            "last_event": self.last_event,
            "last_exit_code": self.last_exit_code,
            "last_error": self.last_error,
            "project": self.project,
            "set_up": is_set_up(self.status),
            "case": self.case,
            "ct": {"node": ct.GetName() if ct else None, "case_name": self.ct_case_name(),
                   "info": self.ct_info_label.text},
            "xrays_scene": [{"node": n.GetName(), "id": xray_experiment_id(n, self.project),
                             "checked": n.GetID() in self._xray_checked} for n in self.xray_scene_nodes()],
            "pose_model": self.pose_model_label.text,
            "registration": {"key": choice["key"] if choice else None, "label": choice["label"] if choice else None,
                             "exists": choice["exists"] if choice else None,
                             "relocated": bool(self.reg_relocated_label.text),
                             "reason": self.reg_reason_label.text or None},
            "registrations": [c["key"] for c in registration_choices(self.status, self.case)],
            "checked_ids": self.checked_ids(),
            "xray_list": [self.xray_list.item(i).text() for i in range(self.xray_list.count)],
            "selectors": {key: {"node": node.GetName() if node else None,
                                "file": project_relative(node_file(node), self.project) if node else None}
                          for key, node in self.selector_nodes().items()},
            "messages": {k: v for k, v in self.messages.items() if v},
            "settings": self.settings_values(),
            "selected_run": self._selected_run(),
            "viewer": self.viewer_state(),
        }

    # ---------------------------------------------------------------- checklist
    def show_step(self, key):
        """Shows the panel of a step (a key of STEPS, or "log")."""
        if key not in self.pages:
            raise ValueError(f"unknown step {key!r}; one of {', '.join(self.pages)}")
        self.stack.setCurrentWidget(self.pages[key])
        self.step_buttons[key].checked = True
        self.current_step = key
        if key == "registration" and self.status and all(n is None for n in self.selector_nodes().values()):
            self._default_selectors()
        return True

    def _refresh_checklist(self):
        import qt

        missing = [text for key, text, _d, kind in SETTINGS_FIELDS
                   if not setting_check(key, kind, self._setting(key))[0]]
        choice = self.registration_choice()
        registration = {"label": choice["label"] if choice else None, "reason": self.reg_reason_label.text or None}
        self.checklist_state = step_checklist(self.status, self.case, missing, registration)
        palette = self.stack.palette
        colors = {"done": palette.color(qt.QPalette.Highlight).name(),
                  "pending": palette.color(qt.QPalette.WindowText).name(),
                  "warning": "#d08000", "error": "#d04040"}
        for key, item in self.checklist_state.items():
            mark = self.step_marks[key]
            mark.text = STEP_MARKS[item["mark"]]
            mark.toolTip = item["mark"]
            mark.setStyleSheet(f"color: {colors[item['mark']]}; font-weight: bold")
            summary = self.step_summaries[key]
            summary.text = item["summary"]
            summary.toolTip = item["summary"]
        self.xray_summary_label.text = "Experiment: " + self.checklist_state["xrays"]["summary"]

    # ---------------------------------------------------------------- process runner
    def _start(self, jobs):
        """Queues jobs (run one after another while they succeed); False with an inline error when busy."""
        if self.process is not None or self.queue:
            self.status_label.text = "a command is running; wait or press Stop"
            return False
        self.queue = list(jobs)
        self._start_next()
        return True

    def _job(self, label, step, on_done=None, **kwargs):
        return Job(label, cli_argv(self.settings_values(), step, **kwargs), on_done)

    def _start_next(self):
        import qt

        if not self.queue:
            return
        job = self.queue.pop(0)
        self.job = job
        self.last_argv = list(job.argv)
        self.last_event = None
        self.last_error = None
        self.last_exit_code = None
        self._process_group = None
        self._buffers = {"out": b"", "err": b""}
        self.command_edit.text = command_line(job.argv)
        self.command_edit.setCursorPosition(0)
        self._log(f"$ {command_line(job.argv)}")
        self.status_label.text = f"{job.label}: running"
        self.progress.setRange(0, 0)
        self.progress.visible = True
        env = qt.QProcessEnvironment()
        for k, v in self._env_dict().items():
            env.insert(k, v)
        proc = qt.QProcess()
        proc.setProcessEnvironment(env)
        proc.setWorkingDirectory(self.project if os.path.isdir(self.project) else _HERE)
        proc.connect("started()", lambda: self._on_started(proc))
        proc.connect("readyReadStandardOutput()", lambda: self._read(proc, "out"))
        proc.connect("readyReadStandardError()", lambda: self._read(proc, "err"))
        proc.connect("finished(int,QProcess::ExitStatus)", lambda code, status: self._on_finished(proc, code, status))
        proc.connect("errorOccurred(QProcess::ProcessError)", lambda err: self._on_process_error(proc, err))
        self.process = proc
        self._update_buttons()
        if os.name == "posix":
            proc.start("setsid", job.argv)
        else:
            proc.start(job.argv[0], job.argv[1:])

    def _on_started(self, proc):
        if proc is self.process and os.name == "posix":
            self._process_group = int(proc.processId())
            if self.job.stop_requested:
                self._signal_process(proc)

    def _process_group_active(self):
        if not self._process_group:
            return False
        try:
            os.killpg(self._process_group, 0)
        except ProcessLookupError:
            return False
        return True

    def _signal_process(self, proc, force=False):
        if proc is not self.process:
            return
        if self._process_group:
            try:
                os.killpg(self._process_group, signal.SIGKILL if force else signal.SIGTERM)
                return
            except ProcessLookupError:
                pass  # a Stop received before setsid established the group
        proc.kill() if force else proc.terminate()

    def _read(self, proc, channel):
        data = proc.readAllStandardOutput() if channel == "out" else proc.readAllStandardError()
        self._buffers[channel] += bytes(data.data())
        *lines, rest = self._buffers[channel].split(b"\n")
        self._buffers[channel] = rest
        for raw in lines:
            self._on_line(raw.decode("utf-8", errors="replace").rstrip("\r"))

    def _flush(self):
        for channel in ("out", "err"):
            rest = self._buffers[channel]
            self._buffers[channel] = b""
            if rest:
                self._on_line(rest.decode("utf-8", errors="replace"))

    def _on_line(self, line):
        if getattr(self, "_closed", False):
            return
        self._log(line)
        parsed = parse_line(line)
        if parsed is None:
            return
        source, record = parsed
        job = self.job
        label = job.label if job else "command"
        if source == "pipeline":
            self.last_event = record
            if job is not None:
                job.records.append(record)
            event, step = record.get("event"), record.get("step")
            if event == "error":
                self.last_error = record
                self.status_label.text = f"{label}: {step}: error: {record.get('message')}"
            elif event == "warning":
                self.status_label.text = f"{label}: warning: {record.get('message')}"
        update = progress_update(label, source, record)
        if update is not None:
            text, done, total = update
            self.status_label.text = text
            if total:
                self.progress.setRange(0, total)
                self.progress.value = done
                self.progress.textVisible = True
            else:
                self.progress.setRange(0, 0)
                self.progress.textVisible = False

    def _on_process_error(self, proc, err):
        import qt

        if err != qt.QProcess.FailedToStart or proc is not self.process:
            return
        if getattr(self, "_closed", False):
            self._finish(-1)
            return
        self._log(f"failed to start {proc.program()}: {proc.errorString()}")
        self.last_error = {"event": "error", "message": f"failed to start {proc.program()}: {proc.errorString()}"}
        self._finish(-1)

    def _on_finished(self, proc, code, exit_status, waiting=False):
        import qt

        if proc is not self.process:
            return
        if self.job.stop_requested and self._process_group_active():
            if not waiting:
                self._signal_process(proc)
            qt.QTimer.singleShot(50, lambda: self._on_finished(proc, code, exit_status, True))
            return
        self._flush()
        self._finish(code if exit_status == qt.QProcess.NormalExit else -1)

    def _finish(self, code):
        job = self.job
        self.process = None
        self._process_group = None
        self.job = None
        self.last_exit_code = code
        if getattr(self, "_closed", False):
            self.queue = []
            return
        self.progress.setRange(0, 1)
        self.progress.value = 1 if code == 0 else 0
        self.progress.textVisible = False
        self.progress.visible = False
        err = f"; {self.last_error.get('message')}" if self.last_error and code != 0 else ""
        outcome = "done" if code == 0 else f"exit {code}{err}"
        if code and job is not None and job.stop_requested:
            outcome = "stopped"
        self.status_label.text = f"{job.label if job else 'command'}: {outcome}"
        self._log(f"[exit {code}]")
        self.refresh_status()
        if job is not None and job.on_done is not None:
            try:
                job.on_done(code, job)
            except Exception as exc:  # keep the runner usable; the error is shown inline
                self._log(f"after {job.label}: {type(exc).__name__}: {exc}")
                self.status_label.text = f"after {job.label}: {exc}"
                self.queue = []
        if code == 0 and self.queue:
            self._start_next()
        else:
            self.queue = []
        self._update_buttons()

    def stop(self):
        """Stops the launcher group and lets the CLI reap its workers before another command can start."""
        import qt

        self.queue = []
        proc = self.process
        if proc is None:
            return
        if self.job is not None:
            self.job.stop_requested = True
        self._log("[stop requested]")
        self.status_label.text = f"{self.job.label}: stopping"
        self._signal_process(proc)
        qt.QTimer.singleShot(STOP_KILL_MS, lambda: self._signal_process(proc, force=True))

    def experiment_reason(self):
        """Why the pipeline steps cannot use the folder, or None."""
        if self._view_only():
            return SET_UP_REASON
        if not self._has_experiment():
            return "choose an experiment folder first"
        return None

    def annotate_reason(self):
        reason = self.experiment_reason()
        if reason:
            return reason
        if self.ct_node() is None:
            return "select or load the CT volume"
        if not self.ct_case_name():
            return "enter a case name"
        return None

    def train_reason(self):
        reason = self.experiment_reason()
        if reason:
            return reason
        if not case_staged(find_case(self.status, self.case)):
            return "needs the annotated CT (CT step)"
        if not self.experiment_ids():
            return "needs at least one X-ray in the experiment (X-rays step)"
        return None

    def assess_reason(self, rel=None):
        """Why Assess cannot run on a run (default: the selected one), or None."""
        rel = rel or self._selected_run()
        if not rel:
            return "select a run"
        entry = self._run_entry(rel)
        if entry is None:
            return f"{rel} is not a run of this folder"
        if entry.get("error"):
            return entry["error"]
        if entry.get("assessment") and not entry["assessment"].get("stale"):
            return "the run has an assessment.json"
        return None

    def _update_buttons(self):
        busy = self.process is not None or bool(self.queue)
        view_only = self._view_only()
        annotate = self.annotate_reason()
        self.annotate_button.enabled = not busy and annotate is None
        self.annotate_button.toolTip = annotate or ""
        self.ct_load_button.enabled = not busy
        self.add_files_button.enabled = not busy
        to_add = [n for n in self.checked_xray_nodes() if xray_experiment_id(n, self.project) is None]
        self.add_to_experiment_button.enabled = not busy and self._has_experiment() and bool(to_add)
        train = self.train_reason()
        self.pose_train_button.enabled = not busy and train is None
        self.pose_reason_label.text = f"Train: {train}" if train else ""
        self.pose_reason_label.visible = bool(train) and not view_only
        has_case = self._has_experiment() and bool(self.case)
        self.pose_use_button.enabled = not busy and has_case
        self.pose_config_button.enabled = not busy and has_case
        self.config_open_button.enabled = not busy and self._has_experiment() and self.registration_choice() is not None
        self.register_button.enabled = not busy and not self.reg_reason_label.text
        self.stop_button.enabled = self.process is not None
        self.backend_status_button.enabled = not busy
        self.backend_setup_button.enabled = not busy
        # A folder open for viewing only: the steps that write into it wait for Set up.
        self.set_up_button.visible = view_only
        self.set_up_button.enabled = not busy
        for note in self.set_up_notes.values():
            note.visible = view_only
        self.reg_reason_label.visible = bool(self.reg_reason_label.text) and not view_only  # the note says it
        for b in (self.ct_load_button, self.add_files_button, self.add_to_experiment_button, self.pose_train_button,
                  self.pose_use_button, self.pose_config_button, self.config_open_button, self.register_button):
            if view_only:
                b.enabled = False
                b.toolTip = SET_UP_REASON
            else:
                b.toolTip = ""
        assess = self.assess_reason()
        self.assess_button.enabled = not busy and assess is None
        self.assess_button.toolTip = assess or "Writes assessment.json into the selected run"
        self.show_run_button.enabled = self._selected_run() is not None
        self.clear_run_button.enabled = self.run_viewer.scene is not None

    # ---------------------------------------------------------------- experiment / status
    def _browse_project(self):
        import qt

        path = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Experiment folder", self.project)
        if path:
            self.set_project(path)

    def set_project(self, path):
        """Opens a folder (the Experiment folder field, Choose…, the launcher):
        1. a run folder (run_manifest.json): its experiment folder (by the rules below), then the run is shown;
        2. a folder with project_manifest.json: `status`;
        3. a folder without one whose `status` finds runs: opened for viewing only, nothing is written into it
           (Set up runs `init`);
        4. any other folder: `init`, then `status`.
        Returns the status (None when the folder does not exist or init/status failed)."""
        path = str(path or "").strip().rstrip(os.sep)
        if path and os.path.isfile(os.path.join(path, "run_manifest.json")):
            return self._open_run_folder(path)
        changed = path != self.project or self.status is None or self.status.get("project") != path
        if path != self.project_edit.text:
            self.project_edit.text = path
        if changed:
            self._checked = set()
            for key in self.selectors:
                self.selectors[key].setCurrentNode(None)
            for where in self._message_labels:
                self._msg(where, "")
            self.clear_run()
        if not path:
            self.status = None
            self._fill_from_status()
            return None
        if not os.path.isdir(path):
            self.status = None
            self._fill_from_status()
            self._msg("experiment", f"{path} is not a folder", error=True)
            return None
        if not changed and self.status is not None:
            return self.status
        if os.path.isfile(os.path.join(path, "project_manifest.json")):
            return self.refresh_status()
        status = self.refresh_status()
        if status is None or status.get("runs"):
            return status
        return self.set_up_experiment()

    def _open_run_folder(self, path):
        """Rule 1 of set_project: the run's experiment folder (the run folder minus the manifest's run.dir) is
        opened and the run shown."""
        try:
            run = XregRun.load(path)
        except Exception as exc:
            self._msg("experiment", f"{path}: {exc}", error=True)
            return None
        status = self.set_project(str(run.experiment_dir))
        if status is None:
            return None
        rel = os.path.relpath(str(run.run_dir), self.project).replace(os.sep, "/")
        self.select_run(rel)
        self.show_step("results")
        self.show_run(rel)
        return status

    def set_up_experiment(self):
        """Sets the open folder up for the pipeline: `init` (idempotent; writes project_manifest.json and the
        folder layout), then `status`. Returns the status (None on failure)."""
        path = self.project
        if not path or not os.path.isdir(path):
            self._msg("experiment", "choose an experiment folder first", error=True)
            return None
        _rc, _record, failure = self._sync("init", project=path)
        if failure:
            self.status = None
            self._fill_from_status()
            self._msg("experiment", f"setting up the experiment failed: {failure}", error=True)
            return None
        return self.refresh_status()

    def _sync(self, step, **kwargs):
        """(exit status, result record, failure text or None) of a short CLI call."""
        try:
            rc, record, stderr = run_sync(cli_argv(self.settings_values(), step, **kwargs), self._env_dict())
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, None, str(exc)
        if rc != 0 or record is None or record.get("event") != "result":
            msg = record.get("message") if record else (stderr.strip().splitlines() or ["no output"])[-1]
            return rc, record, f"exit {rc}: {msg}"
        return rc, record, None

    def refresh_status(self):
        """Runs `status --project` synchronously (it never writes) and refreshes every panel."""
        if not self.project or not os.path.isdir(self.project):
            self.status = None
            self._fill_from_status()
            return None
        _rc, record, failure = self._sync("status", project=self.project)
        if failure:
            self.status = None
            self._fill_from_status()
            self._msg("experiment", f"status failed: {failure}", error=True)
            return None
        self.status = record
        self._fill_from_status()
        return record

    def _fill_from_status(self):
        st = self.status or {}
        self._updating = True
        try:
            current = self.case
            self.case_combo.clear()
            for c in st.get("cases") or []:
                self.case_combo.addItem(c.get("case_id"))
            idx = self.case_combo.findText(current)
            self.case_combo.currentIndex = idx if idx >= 0 else (0 if self.case_combo.count else -1)
            self.case_row.visible = self.case_combo.count > 1
        finally:
            self._updating = False
        if st and is_set_up(st):
            xr = st.get("xrays") or {}
            self._msg("experiment", f"{st.get('project_name')}: {len(st.get('cases') or [])} case(s), "
                                    f"{len(xr.get('ids') or [])} X-ray(s), {len(st.get('runs') or [])} run(s)")
        elif st:
            self._msg("experiment", f"Results only: {len(st.get('runs') or [])} run(s); nothing is written into "
                                    "this folder until Set up.")
        self._fill_registrations()  # before the CT step: loading an annotated case defaults the selectors
        self._refresh_ct_step()
        self._rebuild_xray_scene_list()
        self.pose_model_label.text = pose_model(self.status, self.case)[1] if self.status else ""
        self._fill_runs()
        self._update_buttons()

    def _on_case_changed(self):
        if self._updating:
            return
        self.pose_model_label.text = pose_model(self.status, self.case)[1] if self.status else ""
        self._fill_registrations()
        self._update_buttons()

    def _select_case(self, case):
        idx = self.case_combo.findText(case or "")
        if idx >= 0 and idx != self.case_combo.currentIndex:
            self.case_combo.currentIndex = idx

    # ---------------------------------------------------------------- scene lists
    def _on_scene_changed(self, _caller=None, _event=None):
        import qt

        if self._rebuild_pending:
            return
        self._rebuild_pending = True
        qt.QTimer.singleShot(0, self._rebuild_scene_lists)

    def _rebuild_scene_lists(self):
        self._rebuild_pending = False
        if getattr(self, "_closed", False) or self._loading_run:
            return  # show_run rebuilds once the run is in the scene
        scene = self.run_viewer.scene
        if scene is not None and scene.nodes and scene.nodes[0].GetScene() is None:  # the scene was closed
            self.run_viewer.scene = None
            self.shown_run = None
            self._fill_viewer()
        self._rebuild_ct_combo()
        self._rebuild_xray_scene_list()
        self._rebuild_xray_list()
        self._refresh_registration()

    def _queue_ct_selector_filter(self):
        """Updates the CT selector's hidden nodes once control returns to the event loop: never while its model
        is changing (a node being added or removed)."""
        import qt

        if not self._ct_filter_pending:
            self._ct_filter_pending = True
            qt.QTimer.singleShot(0, self._hide_2d_from_ct_selector)

    def _hide_2d_from_ct_selector(self):
        """The Registration CT selector lists 3D volumes only: single-slice (X-ray-like) volumes are hidden, except
        the selected node (hiding it would move the selection)."""
        self._ct_filter_pending = False
        combo = self.selectors.get("ct")
        if combo is None or getattr(self, "_closed", False):
            return
        current = combo.currentNodeID
        hidden = [n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
                  if is_xray_like(n) and n.GetID() != current]
        proxy = combo.sortFilterProxyModel()
        if list(proxy.hiddenNodeIDs) != hidden:
            proxy.hiddenNodeIDs = hidden

    def _rebuild_ct_combo(self):
        current = self.ct_combo.itemData(self.ct_combo.currentIndex)
        nodes = [n for n in self._pipeline_volumes() if is_ct_like(n)]
        self._updating = True
        try:
            self.ct_combo.clear()
            for n in nodes:
                self.ct_combo.addItem(n.GetName(), n.GetID())
            idx = self.ct_combo.findData(current) if current else -1
            # A new CT in the scene is selected when none was.
            self.ct_combo.currentIndex = idx if idx >= 0 else (self.ct_combo.count - 1 if nodes else -1)
        finally:
            self._updating = False
        if self.ct_combo.itemData(self.ct_combo.currentIndex) != current:
            self._on_ct_changed()

    # ---------------------------------------------------------------- CT
    def _browse_ct_file(self):
        import qt

        path = qt.QFileDialog.getOpenFileName(slicer.util.mainWindow(), "CT volume", self.project,
                                              "Volumes (*.nii *.nii.gz *.nrrd *.nhdr *.mha *.mhd *.dcm);;All files (*)")
        if path:
            self.load_ct_file(path)

    def load_ct_file(self, path):
        """Loads a volume file (NIfTI, NRRD, DICOM) into the scene and selects it as the CT."""
        try:
            node = slicer.util.loadVolume(str(path))
        except Exception as exc:
            self._msg("ct", f"loading {path} failed: {exc}", error=True)
            return None
        self.select_ct(node)
        return node

    def select_ct(self, node):
        """Selects a scene volume as the CT (by node or node name)."""
        if isinstance(node, str):
            node = slicer.util.getFirstNodeByName(node)
        self._rebuild_ct_combo()
        idx = self.ct_combo.findData(node.GetID()) if node is not None else -1
        if idx < 0:
            self._msg("ct", f"{node.GetName() if node else node} is not a 3D volume", error=True)
            return False
        self.ct_combo.currentIndex = idx
        return True

    def _on_ct_changed(self):
        if self._updating:
            return
        self._msg("ct", "")
        self._refresh_ct_step()
        self._update_buttons()

    def _refresh_ct_step(self):
        """Case name placeholder and annotation state of the selected CT; an annotated case CT gets its
        segmentation, 3D landmarks and models loaded (once per node)."""
        node = self.ct_node()
        entry = case_for_ct(self.status, node, self.project) if self._has_experiment() else None
        self.ct_case_edit.placeholderText = entry["case_id"] if entry else default_case_name(
            node.GetName() if node else "")
        if node is None:
            self.ct_info_label.text = "Load the CT (drag it into Slicer or Load file…)."
        elif entry is None:
            self.ct_info_label.text = "Not annotated yet."
        elif case_staged(entry):
            self._select_case(entry["case_id"])
            self.ct_info_label.text = f"Annotated: case {entry['case_id']}."
            key = (self.project, entry["case_id"], node.GetID())
            if key not in self._loaded_outputs:
                self._loaded_outputs.add(key)
                self._load_outputs(entry["case_id"], node)
        elif (entry.get("annotation_output") or {}).get("present"):
            self.ct_info_label.text = f"Case {entry['case_id']}: annotated, not staged (see the log)."
        else:
            self.ct_info_label.text = f"Case {entry['case_id']}: not annotated yet."

    def _load_outputs(self, case, ct_node):
        try:
            with slicer.util.WaitCursor():
                loaded = self.logic.load_case_outputs(self.project, case, ct_node)
        except Exception as exc:
            self._msg("ct", f"loading the annotation of {case} failed: {exc}", error=True)
            return None
        self._loaded_outputs.add((self.project, case, loaded["ct"].GetID()))
        if loaded["ct"].GetID() != (ct_node.GetID() if ct_node else None):
            self.select_ct(loaded["ct"])
        lm = slicer.app.layoutManager()
        if lm is not None and lm.threeDWidget(0) is not None:
            lm.threeDWidget(0).threeDView().resetFocalPoint()
        self._msg("ct", f"loaded {loaded['seg'].GetName()}, {loaded['landmarks_3d'].GetName()} and "
                        f"{len(loaded['models'])} model(s)")
        if self.case == case:
            self._default_selectors()
        return loaded

    def run_annotate(self, node=None, case=None):
        """Segment and annotate: `import-ct` of the CT node (its unmodified NIfTI file, else saved to
        meta_data/tmp/slicer_export/<case>.nii.gz), `annotate-3d`, then the results are loaded into the scene."""
        if node is not None:
            self.select_ct(node)
        if case is not None:
            self.ct_case_edit.text = case
        reason = self.annotate_reason()
        if reason:
            self._msg("ct", reason, error=True)
            return False
        node, case = self.ct_node(), self.ct_case_name()
        path, save = ct_export_plan(node, os.path.join(self.project, *EXPORT_DIR.split("/")), case)
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not slicer.util.saveNode(node, path):
                self._msg("ct", f"saving {node.GetName()} to {path} failed", error=True)
                return False
        node_id = node.GetID()

        def done(code, _job):
            if code != 0:
                return
            self._select_case(case)
            self._loaded_outputs.add((self.project, case, node_id))
            self._load_outputs(case, self._node(node_id))
            self._refresh_ct_step()

        self._msg("ct", f"input: {path}")
        return self._start([self._job("Import CT", "import-ct", project=self.project, input_path=path, case=case),
                            self._job("Segment and annotate", "annotate-3d", project=self.project, case=case,
                                      on_done=done)])

    # ---------------------------------------------------------------- X-rays
    def _browse_xray_files(self):
        import qt

        paths = qt.QFileDialog.getOpenFileNames(slicer.util.mainWindow(), "X-ray files", self.project,
                                                "X-rays (*.dcm *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)")
        if paths:
            self.add_xray_files(list(paths))

    def add_xray_files(self, paths):
        """Loads DICOM or image files into the scene and checks them; returns the nodes."""
        nodes, errors = [], []
        with slicer.util.WaitCursor():
            for path in paths:
                try:
                    nodes.append(self.logic.load_xray_file(str(path)))
                except Exception as exc:
                    errors.append(f"{os.path.basename(str(path))}: {exc}")
        self._xray_checked |= {n.GetID() for n in nodes}
        self._rebuild_xray_scene_list()
        if errors:
            self._msg("xray", "loading failed for " + "; ".join(errors), error=True)
        else:
            self._msg("xray", f"loaded {len(nodes)} X-ray(s)")
        return nodes

    def set_xray_checked(self, nodes):
        """Checks exactly these scene X-rays (nodes or names) in the X-rays panel."""
        found = [slicer.util.getFirstNodeByName(n) if isinstance(n, str) else n for n in nodes]
        self._xray_checked = {n.GetID() for n in found if n is not None}
        self._xray_seen |= {n.GetID() for n in self.xray_scene_nodes()}
        self._rebuild_xray_scene_list()

    def _rebuild_xray_scene_list(self):
        import qt

        self._updating = True
        try:
            self.xray_scene_list.clear()
            for node in self.xray_scene_nodes():
                pid = xray_experiment_id(node, self.project) if self._has_experiment() else None
                if node.GetID() not in self._xray_seen:
                    self._xray_seen.add(node.GetID())
                    if pid is None:
                        self._xray_checked.add(node.GetID())
                state = f"in experiment (id {pid})" if pid else "not added"
                item = qt.QListWidgetItem(f"{node.GetName()} — {state}")
                item.setData(qt.Qt.UserRole, node.GetID())
                item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable)
                item.setCheckState(qt.Qt.Checked if node.GetID() in self._xray_checked else qt.Qt.Unchecked)
                item.setToolTip(node_file(node) or node.GetName())
                self.xray_scene_list.addItem(item)
        finally:
            self._updating = False
        self._update_buttons()

    def _on_xray_scene_item_changed(self, item):
        import qt

        if self._updating:
            return
        node_id = item.data(qt.Qt.UserRole)
        if item.checkState() == qt.Qt.Checked:
            self._xray_checked.add(node_id)
        else:
            self._xray_checked.discard(node_id)
        self._update_buttons()

    def add_xrays_to_experiment(self, nodes=None):
        """`import-xrays` of the checked scene X-rays not yet in the experiment (or `nodes`): their DICOM or
        image file, else the image saved as a 16-bit PNG; the CLI converts them to H5. Afterwards the nodes are
        tagged with their projection id."""
        if not self._has_experiment():
            self._msg("xray", self.experiment_reason(), error=True)
            return False
        nodes = self.checked_xray_nodes() if nodes is None else list(nodes)
        nodes = [n for n in nodes if xray_experiment_id(n, self.project) is None]
        if not nodes:
            self._msg("xray", "check X-rays that are not in the experiment yet", error=True)
            return False
        export_dir = os.path.join(self.project, *EXPORT_DIR.split("/"))
        inputs, images = [], 0
        try:
            for node in nodes:
                kind, path = xray_source(node)
                if kind is None:
                    path = save_xray_png(node, free_path(export_dir, sanitize_case_id(node.GetName()), ".png"))
                    kind = "image"
                images += kind == "image"
                inputs.append(path)
        except Exception as exc:
            self._msg("xray", f"exporting X-rays failed: {exc}", error=True)
            return False
        if images and not self._setting(TEMPLATE_KEY):
            self._msg("xray", f"{images} X-ray(s) are images without DICOM geometry: set the Template DICOM "
                              "(for images)", error=True)
            return False
        node_ids = [n.GetID() for n in nodes]
        project = self.project

        def done(_code, job):
            result = job.result("import-xrays") or {}
            ids = list(result.get("ids") or [])
            if len(ids) == len(node_ids):
                pairs = zip(ids, node_ids)
            else:  # some inputs were ignored: match by source file name
                by_name = {os.path.basename(p): i for p, i in zip(inputs, node_ids)}
                sources = result.get("sources") or {}
                pairs = [(pid, by_name.get(sources.get(pid))) for pid in ids]
            for pid, node_id in pairs:
                node = self._node(node_id)
                if node is not None:
                    tag_xray(node, project, pid)
            self._rebuild_xray_scene_list()
            self._rebuild_xray_list()
            failed = result.get("failed") or []
            self._msg("xray", f"added {', '.join(ids) or 'nothing'}"
                              + (f"; conversion failed for {', '.join(failed)}" if failed else ""), error=bool(failed))

        return self._start([self._job("Add X-rays", "import-xrays", project=project, inputs=inputs, on_done=done)])

    # ---------------------------------------------------------------- pose model
    def run_pose_train(self):
        reason = self.train_reason()
        if reason:
            self._msg("pose", reason, error=True)
            return False
        return self._start([self._job("Train pose model", "pose-train", project=self.project, case=self.case)])

    def _browse_pose_model(self):
        import qt

        path = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(), "Pose model folder (a trained model, or a case's models folder)", self.project)
        if path:
            self.use_pose_model(path)

    def use_pose_model(self, model_dir):
        """`pose use-model`: activates a pose model for the case; `model_dir` is a trained model folder or a case's
        models folder (its active or only model)."""
        if not (self._has_experiment() and self.case):
            self._msg("pose", self.experiment_reason() or "needs an experiment with a case", error=True)
            return False
        return self._start([self._job("Use pose model", "pose-use-model", project=self.project, case=self.case,
                                      model_dir=str(model_dir))])

    def _open_file(self, where, path):
        import qt

        if not os.path.isfile(path):
            self._msg(where, f"{path} does not exist", error=True)
            return False
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path))
        return True

    def open_pose_config(self):
        """Opens configs/pose_estimation/<case>.json (`pose init-config` first when it is missing)."""
        if not (self._has_experiment() and self.case):
            self._msg("pose", self.experiment_reason() or "needs an experiment with a case", error=True)
            return False
        path = os.path.join(self.project, "configs", "pose_estimation", f"{self.case}.json")
        if os.path.isfile(path):
            return self._open_file("pose", path)
        return self._start([self._job("Pose config", "pose-init-config", project=self.project, case=self.case,
                                      on_done=lambda code, _j: self._open_file("pose", path) if code == 0 else None)])

    # ---------------------------------------------------------------- registration
    def _fill_registrations(self):
        current = self.reg_combo.itemData(self.reg_combo.currentIndex)
        choices = registration_choices(self.status, self.case)
        self._updating = True
        try:
            self.reg_combo.clear()
            for c in choices:
                self.reg_combo.addItem(c["label"], c["key"])
            idx = self.reg_combo.findData(current) if current else -1
            self.reg_combo.currentIndex = idx if idx >= 0 else (0 if choices else -1)
            selected = self.registration_choice()
            if selected is not None and selected["key"] != current and selected["exists"]:
                # A newly selected existing config: its X-rays are checked (selectors follow on demand).
                entry = next((e for e in self.status.get("xreg_configs") or []
                              if e.get("config") == selected["config"]), {})
                self._checked = set(entry.get("ids") or [])
        finally:
            self._updating = False
        self._rebuild_xray_list()
        self._refresh_registration()

    def _on_reg_combo(self):
        if self._updating:
            return
        self.select_registration(self.reg_combo.itemData(self.reg_combo.currentIndex))

    def select_registration(self, key):
        """Selects a registration by key (config name) or label; an existing config sets the selectors and
        X-ray checks, a default one not created yet uses the case's annotated files."""
        idx = self.reg_combo.findData(key)
        if idx < 0:
            idx = self.reg_combo.findText(key)
        if idx < 0:
            self._msg("registration", f"{key} is not a registration of this experiment", error=True)
            return False
        if idx != self.reg_combo.currentIndex:
            self._updating = True
            self.reg_combo.currentIndex = idx
            self._updating = False
        return self._default_selectors()

    def _default_selectors(self):
        """Selectors (and X-ray checks) from the selected registration's config, else the case's files."""
        choice = self.registration_choice()
        if choice is None or not self._has_experiment():
            self._refresh_registration()
            return False
        if choice["exists"]:
            _rc, info, failure = self._sync("config-show", config=self._config_path(choice))
            if failure:
                self._msg("registration", f"config show failed: {failure}", error=True)
                return False
            return self.apply_config_info(info)
        entry = find_case(self.status, self.case)
        if not case_staged(entry):
            self._refresh_registration()
            return True
        info = {"experiment_dir": self.project, **staged_paths(self.case), "ids": self.checked_ids()}
        return self.apply_config_info(info, set_ids=False)

    def apply_config_info(self, info, set_ids=True):
        """Sets the selectors (loading files not in the scene) and, with `set_ids`, the X-ray checks from a
        `config show` result."""
        targets = config_targets(info)
        errors = []
        with slicer.util.WaitCursor():
            for key, _what in SELECTOR_KINDS:
                path = targets[key]
                if not path:
                    continue
                if not os.path.isfile(path):
                    errors.append(f"{info.get(key)} does not exist")
                    continue
                try:
                    node = self.logic.load_file(key, path)
                except Exception as exc:
                    errors.append(f"loading {info.get(key)} failed: {exc}")
                    continue
                self.selectors[key].setCurrentNode(node)
        if set_ids:
            self._checked = set(targets["ids"])
        self._rebuild_xray_list()
        self._refresh_registration()
        if errors:
            self._msg("registration", "; ".join(errors), error=True)
        else:
            self._msg("registration", "")
        return not errors

    def _rebuild_xray_list(self):
        import qt

        nodes = experiment_xray_nodes(self.project) if self._has_experiment() else {}
        converted = set(((self.status or {}).get("xrays") or {}).get("converted") or [])
        self._updating = True
        try:
            self.xray_list.clear()
            for pid in self.experiment_ids():
                node = nodes.get(pid)
                if node is None:
                    text = f"{pid} (not in the scene)"
                else:
                    text = pid if node.GetName() == pid else f"{pid} — {node.GetName()}"
                if pid not in converted:
                    text += ", no H5"
                item = qt.QListWidgetItem(text)
                item.setData(qt.Qt.UserRole, pid)
                item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable | qt.Qt.ItemIsSelectable)
                item.setCheckState(qt.Qt.Checked if pid in self._checked else qt.Qt.Unchecked)
                self.xray_list.addItem(item)
        finally:
            self._updating = False

    def _on_xray_item_changed(self, item):
        import qt

        if self._updating:
            return
        pid = item.data(qt.Qt.UserRole)
        ids = [i for i in self.checked_ids() if i != pid] + ([pid] if item.checkState() == qt.Qt.Checked else [])
        qt.QTimer.singleShot(0, lambda: self.set_checked_ids(ids))  # the list is rebuilt: not inside its signal

    def set_checked_ids(self, ids):
        """Checks exactly these experiment X-rays for the registration; checked ones not in the scene are loaded."""
        wanted = [i for i in ids if i in set(self.experiment_ids())]
        self._checked = set(wanted)
        present = experiment_xray_nodes(self.project) if self._has_experiment() else {}
        errors = []
        for pid in wanted:
            if pid not in present:
                try:
                    self.logic.load_xray(self.project, pid)
                except Exception as exc:
                    errors.append(f"{pid}: {exc}")
        if errors:
            self._msg("registration", "loading failed for " + "; ".join(errors), error=True)
        self._rebuild_xray_list()
        self._refresh_registration()
        return self.checked_ids()

    def _refresh_registration(self):
        self._queue_ct_selector_filter()
        choice = self.registration_choice()
        if choice is None:
            self.reg_info_label.text = ""
        elif choice["exists"]:
            self.reg_info_label.text = f"Config {choice['config']}."
        else:
            self.reg_info_label.text = f"Config {choice['config']} is created from the default by Register."
        relocated = bool(choice and choice["exists"]
                         and xreg_config_entry(self.status, choice["config"]).get("relocated"))
        self.reg_relocated_label.text = RELOCATED_NOTE if relocated else ""
        self.reg_relocated_label.visible = relocated
        reason = registration_reason(self.status, self.case, choice, self.checked_ids(), self.selector_nodes())
        self.reg_reason_label.text = reason or ""
        self._refresh_checklist()
        self._update_buttons()

    def export_selector_nodes(self):
        """Saves selector nodes that are not files of the experiment into it (CT -> meta_data/volumes/<name>.nii.gz,
        label map -> meta_data/volumes/<name>_seg.nii.gz, markups -> meta_data/landmarks/landmarks_3d/<name>.fcsv);
        returns {key: saved path}."""
        targets = {"ct": ("meta_data/volumes", "", ".nii.gz"), "seg": ("meta_data/volumes", "_seg", ".nii.gz"),
                   "landmarks_3d": ("meta_data/landmarks/landmarks_3d", "", ".fcsv")}
        saved = {}
        for key, node in self.selector_nodes().items():
            if node is None or project_relative(node_file(node), self.project) is not None:
                continue
            folder, tail, suffix = targets[key]
            stem = sanitize_case_id(node.GetName())
            if tail and stem.endswith(tail):
                stem = stem[:-len(tail)]
            directory = os.path.join(self.project, *folder.split("/"))
            os.makedirs(directory, exist_ok=True)
            path = free_path(directory, stem + tail, suffix)
            if not slicer.util.saveNode(node, path):
                raise RuntimeError(f"saving {node.GetName()} to {path} failed")
            saved[key] = path
        return saved

    def run_register(self):
        """Register: off-experiment selector nodes saved into it, the config created from the default when
        missing, `config set` with the scene selection, then `register --predict-poses`; with Show result the
        new run is shown in the Results step."""
        self._refresh_registration()
        reason = self.reg_reason_label.text
        if reason:
            self._msg("registration", reason, error=True)
            return False
        choice = self.registration_choice()
        try:
            saved = self.export_selector_nodes()
        except Exception as exc:
            self._msg("registration", str(exc), error=True)
            return False
        ids = self.checked_ids()
        config = self._config_path(choice)
        jobs = []
        if choice["exists"]:
            _rc, info, failure = self._sync("config-show", config=config)
            if failure:
                self._msg("registration", f"config show failed: {failure}", error=True)
                return False
        else:
            info = {"experiment_dir": self.project, **staged_paths(self.case), "ids": ids}
            jobs.append(self._job("Create config", "config-create", project=self.project, case=self.case,
                                  app=choice["app"], femur=choice["femur"], name=choice["key"], ids=ids))
        changes = config_changes(info, self.selector_nodes(), ids, self.project)
        if changes:
            jobs.append(self._job("Set config", "config-set", config=config, **changes))
        show_result = self.show_result_check.checked
        replay = self.replay_check.checked

        def done(_code, job):
            result = job.result("register")
            if result and result.get("run_dir"):
                self.show_step("results")
                self.select_run(result["run_dir"])
                if show_result:
                    self.show_run(result["run_dir"])

        jobs.append(self._job("Register", "register", config=config, replay=replay, on_done=done))
        notes = [f"saved {os.path.relpath(p, self.project)}" for p in saved.values()]
        notes += [f"{k}={' '.join(v) if isinstance(v, list) else v}" for k, v in changes.items()]
        self._msg("registration", "; ".join(notes))
        return self._start(jobs)

    def open_config(self):
        """Opens the selected registration's config (created from the default first when missing)."""
        choice = self.registration_choice()
        if choice is None:
            self._msg("registration", "select a registration", error=True)
            return False
        if not self._has_experiment():
            self._msg("registration", self.experiment_reason(), error=True)
            return False
        path = self._config_path(choice)
        if choice["exists"]:
            return self._open_file("registration", path)
        if not case_staged(find_case(self.status, self.case)):
            self._msg("registration", "needs the annotated CT", error=True)
            return False
        return self._start([self._job("Create config", "config-create", project=self.project, case=self.case,
                                      app=choice["app"], femur=choice["femur"], name=choice["key"],
                                      ids=self.checked_ids() or None,
                                      on_done=lambda code, _j: self._open_file("registration", path)
                                      if code == 0 else None)])

    # ---------------------------------------------------------------- results
    def _runs(self):
        return (self.status or {}).get("runs") or []

    def _fill_runs(self):
        import qt

        current = self._selected_run()
        runs = self._runs()
        self.runs_table.setRowCount(len(runs))
        for r, run in enumerate(runs):
            counts = (run.get("assessment") or {}).get("counts") or {}
            name = run_label(run.get("run_dir", ""))
            app = run.get("app") or run.get("error") or ""
            if app.startswith("xreg-") and app.endswith("-regi"):
                app = app[len("xreg-"):-len("-regi")]
            views = f"{run.get('views_registered', '')}/{run.get('views', '')}" if run.get("views") else ""
            values = [name, app, views,
                      *[str(counts.get(c, "")) for c in CERTAINTIES], flag_counts_text(run.get("flags"))]
            for col, text in enumerate(values):
                item = qt.QTableWidgetItem(str(text))
                item.setData(qt.Qt.UserRole, run.get("run_dir"))
                item.setToolTip("\n".join(x for x in (run.get("run_dir"), run.get("summary")) if x))
                self.runs_table.setItem(r, col, item)
        if current:
            self.select_run(current)
        else:
            self._fill_items()

    def _selected_run(self):
        import qt

        rows = self.runs_table.selectionModel().selectedRows() if self.runs_table.selectionModel() else []
        if not rows:
            return None
        item = self.runs_table.item(rows[0].row(), 0)
        return item.data(qt.Qt.UserRole) if item else None

    def select_run(self, run_dir):
        """Selects a run of the table by its directory relative to the experiment folder."""
        import qt

        for r in range(self.runs_table.rowCount):
            item = self.runs_table.item(r, 0)
            if item and item.data(qt.Qt.UserRole) == run_dir:
                self.runs_table.selectRow(r)
                self.runs_table.scrollToItem(item)
                self._fill_items()
                self._update_buttons()
                return True
        return False

    def _run_entry(self, rel):
        return next((r for r in self._runs() if r.get("run_dir") == rel), None) if rel else None

    def _on_run_selected(self):
        self._fill_items()
        self._update_buttons()

    def _fill_items(self):
        import qt

        rel = self._selected_run()
        run = next((r for r in self._runs() if r.get("run_dir") == rel), None) if rel else None
        items = ((run or {}).get("assessment") or {}).get("items") or []
        self.items_table.setRowCount(len(items))
        for r, item in enumerate(items):
            obj, gncc = item.get("target_objective"), item.get("gncc")
            values = [item.get("view") or "all", item.get("target") or "",
                      f"{obj:.3f}" if isinstance(obj, (int, float)) else "",
                      f"{gncc:.2f}" if isinstance(gncc, (int, float)) else "", item.get("certainty") or "",
                      "; ".join(item.get("reasons") or [])]
            for col, text in enumerate(values):
                self.items_table.setItem(r, col, qt.QTableWidgetItem(str(text)))
        if run is not None and not items:
            self._msg("results", ((run.get("assessment") or {}).get("error")) or run.get("summary")
                      or run.get("error") or "no assessment.json")
        else:
            self._msg("results", "")

    def show_run(self, rel=None):
        """Shows a run (relative to the experiment folder; default: the selected run) in the scene: CT, models,
        3D landmarks and per view the C-arm, frustum, X-ray with DRR overlay and 2D landmarks. Replaces the run
        shown before; nothing is written."""
        rel = rel or self._selected_run()
        if not rel:
            self._msg("results", "select a run", error=True)
            return False
        path = os.path.normpath(os.path.join(self.project, *rel.split("/")))
        flags = self.display_flags()
        self._loading_run = True
        try:
            with slicer.util.WaitCursor():
                scene = self.run_viewer.load_run(path, volume_rendering=flags["volume_rendering"])
                scene.set_drr_overlay(on=flags["drr_overlay"], opacity=flags["drr_opacity"])
                scene.set_show_all_views(flags["show_all_views"])
                scene.set_camera_centred(flags["camera_centred"])
                scene.set_stages_all_views(flags["stages_all_views"])
                scene.set_models_visible(flags["models"])
                scene.set_landmarks_visible(flags["landmarks"])
                scene.overview()
            self.shown_run = rel
            error = None
        except Exception as exc:
            self.run_viewer.clear()
            self.shown_run = None
            error = f"showing {rel} failed: {exc}"
        finally:
            self._loading_run = False
        self._fill_viewer()
        self._rebuild_scene_lists()
        if error:
            self._msg("results", error, error=True)
            return False
        return True

    def clear_run(self):
        """Removes the shown run's nodes from the scene (a CT that was in the scene before stays)."""
        self.run_viewer.clear()
        self.shown_run = None
        self._fill_viewer()
        return True

    def run_assess(self, rel=None):
        """`assess --run` of a run without assessment.json (default: the selected run); writes into that run only."""
        rel = rel or self._selected_run()
        reason = self.assess_reason(rel)
        if reason:
            self._msg("results", reason, error=True)
            return False
        path = os.path.normpath(os.path.join(self.project, *rel.split("/")))
        return self._start([self._job("Assess", "assess", run=path, on_done=lambda _c, _j: self.select_run(rel))])

    # ---------------------------------------------------------------- display of the shown run
    def display_flags(self):
        """{flag: value} of the display controls (the keys of set_display)."""
        flags = {key: self.display_checks[key].checked for key, _t, _o in DISPLAY_CHECKS}
        flags["drr_opacity"] = self.drr_opacity_slider.value / 100.0
        return flags

    def _fill_viewer(self):
        """View combo, stage combos, enabled state and info of the shown run."""
        scene = self.run_viewer.scene
        self._updating = True
        try:
            self.view_combo.clear()
            if scene is not None:
                entry = self._run_entry(self.shown_run) or {}
                items = (entry.get("assessment") or {}).get("items") or []
                for v in scene.run.views:
                    levels = []
                    for item in items:  # multiview items (view None) hold for every view
                        level = item.get("certainty")
                        if item.get("view") in (None, v["id"]) and level and f"{level} certainty" not in levels:
                            levels.append(f"{level} certainty")
                    text = ", ".join([v.get("status") or "?"] + levels)
                    self.view_combo.addItem(f"{v['id']} ({text})", v["id"])
                self.view_combo.currentIndex = scene.run.view_ids.index(scene.view_id)
        finally:
            self._updating = False
        for w in self.viewer_controls:
            w.enabled = scene is not None
        self.view_combo.toolTip = "" if scene is not None else "show a run first"
        self._fill_stage_combos()
        self._update_buttons()

    def _fill_stage_combos(self):
        import qt

        for w in list(self.stage_combos.values()) + self._stage_labels:
            self.stage_grid.removeWidget(w)
            w.hide()
            w.deleteLater()
        self.stage_combos, self._stage_labels = {}, []
        scene = self.run_viewer.scene
        if scene is not None:
            self._updating = True
            try:
                for i, key in enumerate(scene.run.target_keys):
                    text = key + (" (primary)" if key == scene.primary else "")
                    title = qt.QLabel(text)
                    title.objectName = f"pipeline.viewerStageTitle.{key}"
                    combo = qt.QComboBox()
                    combo.objectName = f"pipeline.viewerStage.{key}"
                    stages = scene.stages_for(scene.view_id, key)
                    for s in stages:
                        combo.addItem(s)
                    if scene.stages.get(key) in stages:
                        combo.currentIndex = stages.index(scene.stages[key])
                    combo.enabled = bool(stages)
                    combo.connect("currentIndexChanged(int)", lambda _i, k=key: self._on_stage_combo(k))
                    self.stage_grid.addWidget(title, i // 2, 2 * (i % 2))
                    self.stage_grid.addWidget(combo, i // 2, 2 * (i % 2) + 1)
                    self.stage_combos[key] = combo
                    self._stage_labels.append(title)
            finally:
                self._updating = False
        self._refresh_viewer_info()

    def _refresh_viewer_info(self):
        """Info line of the shown run; a display change that worked replaces an earlier error message."""
        scene = self.run_viewer.scene
        if scene is None:
            self.viewer_info_label.text = "Show a run to display it in the scene."
            return
        if (self.messages.get("results") or {}).get("error"):
            self._fill_items()
        view = scene.run.view(scene.view_id)
        lines = []
        for key, entry in (view.get("targets") or {}).items():
            res = entry.get("landmark_residual_px") or {}
            if res.get("n"):
                lines.append(f"{key}: {entry.get('status')}, landmark residual median {res['median']:.1f} px, "
                             f"max {res['max']:.1f} px (n={res['n']})")
            else:
                lines.append(f"{key}: {entry.get('status')}")
        frame = scene.scene_state.frame if scene.scene_state is not None else "CT"
        lines.append(f"scene frame {frame}")
        final_views = scene.scene_state.final_drr_views if scene.scene_state is not None else set()
        if scene.view_id not in final_views:
            lines.append("Final-result DRR hidden: overlays are final-only; select final for every target")
        else:
            lines.append("DRR overlay: final result only (not an init/fiducial rendering)")
        self.viewer_info_label.text = "; ".join(lines)

    def _scene_or_message(self):
        scene = self.run_viewer.scene
        if scene is None:
            self._msg("results", "show a run first", error=True)
        return scene

    def _on_view_combo(self):
        if not self._updating:
            self.set_view(self.view_combo.itemData(self.view_combo.currentIndex))

    def _on_stage_combo(self, key):
        if not self._updating:
            self.set_stage(key, self.stage_combos[key].currentText)

    def set_view(self, view_id):
        """Selects the shown run's X-ray view (its id); the pose stages return to the view's defaults."""
        scene = self._scene_or_message()
        if scene is None:
            return False
        try:
            with slicer.util.WaitCursor():
                scene.set_view(str(view_id))
        except (KeyError, ValueError) as exc:
            self._msg("results", str(exc), error=True)
            return False
        self._updating = True
        self.view_combo.currentIndex = self.view_combo.findData(scene.view_id)
        self._updating = False
        self._fill_stage_combos()
        return True

    def set_stage(self, target, stage):
        """Shows a target's pose of a stage (e.g. "init", "final") in the selected view."""
        scene = self._scene_or_message()
        if scene is None:
            return False
        try:
            with slicer.util.WaitCursor():
                scene.set_stage(target, stage)
        except (KeyError, ValueError) as exc:
            self._msg("results", str(exc), error=True)
            return False
        combo = self.stage_combos.get(target)
        if combo is not None and combo.currentText != stage:
            self._updating = True
            combo.currentIndex = combo.findText(stage)
            self._updating = False
        self._refresh_viewer_info()
        return True

    def set_display(self, **flags):
        """Sets display flags of the shown run (and their controls): show_all_views, camera_centred,
        stages_all_views, models, landmarks, volume_rendering, drr_overlay (bool), drr_opacity (0-1)."""
        known = {key for key, _t, _o in DISPLAY_CHECKS} | {"drr_opacity"}
        unknown = sorted(set(flags) - known)
        if unknown:
            raise ValueError(f"unknown display flag(s) {', '.join(unknown)}; one of {', '.join(sorted(known))}")
        self._updating = True
        try:
            for key, value in flags.items():
                if key == "drr_opacity":
                    self.drr_opacity_slider.value = int(round(float(value) * 100))
                else:
                    self.display_checks[key].checked = bool(value)
        finally:
            self._updating = False
        scene = self.run_viewer.scene
        if scene is None:
            return True
        setters = {"show_all_views": scene.set_show_all_views, "camera_centred": scene.set_camera_centred,
                   "stages_all_views": scene.set_stages_all_views, "models": scene.set_models_visible,
                   "landmarks": scene.set_landmarks_visible, "volume_rendering": scene.set_volume_rendering}
        with slicer.util.WaitCursor():
            for key, value in flags.items():
                if key == "drr_overlay":
                    scene.set_drr_overlay(on=bool(value))
                elif key == "drr_opacity":
                    scene.set_drr_overlay(opacity=float(value))
                else:
                    setters[key](bool(value))
        self._refresh_viewer_info()
        return True

    def look_from_source(self):
        """Puts the 3D camera at the selected view's X-ray source, looking at the detector."""
        scene = self._scene_or_message()
        if scene is not None:
            scene.look_from_source()
        return scene is not None

    def overview(self):
        """3D camera beside the selected view's beam, framing the frustums and the anatomy."""
        scene = self._scene_or_message()
        if scene is not None:
            scene.overview()
        return scene is not None

    def viewer_state(self):
        """The shown run's display (None when no run is shown)."""
        scene = self.run_viewer.scene
        if scene is None:
            return None
        return {"run": self.shown_run, "run_dir": str(scene.run.run_dir), "app": scene.run.app,
                "primary": scene.primary, "views": list(scene.run.view_ids), "view": scene.view_id,
                "view_text": self.view_combo.currentText, "stages": dict(scene.stages),
                "frame": scene.scene_state.frame if scene.scene_state is not None else "CT",
                **self.display_flags(), "info": self.viewer_info_label.text}


#
# Test
#
class FluoroRegTest(RunChecks, ScriptedLoadableModuleTest):
    """Headless checks without external tools: settings, argv of every button, progress text, checklist,
    registration choices and readiness, CT export and adoption, X-ray sources and lists, case loading. With run
    directories in $FLUOROREG_TEST_RUNS also: each run's scene against its manifest, the Results step's display
    controls, and opening folders for viewing only (scratch copies; nothing written)."""

    def setUp(self):
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        self.setUp()
        self.test_settings_round_trip()
        self.test_argv_of_every_step()
        self.test_events_and_progress()
        self.test_checklist_rules()
        self.test_registration_choices_and_reasons()
        self.test_relocated_config()
        runs = test_run_dirs()
        widget = slicer.modules.fluororeg.widgetRepresentation().self()
        previous = (widget.project, widget.current_step, widget._setting(TEMPLATE_KEY))
        env_before = os.environ.get("FLUOROREG_SETTINGS")
        try:
            with tempfile.TemporaryDirectory(prefix="fluororeg_selftest_") as tmp:
                # The widget writes settings (X-ray template): into a scratch file, never the user's.
                os.environ["FLUOROREG_SETTINGS"] = os.path.join(tmp, "settings.json")
                project = self.make_fixture_project(tmp)
                self.test_ct_export_and_adoption(project, tmp)
                self.test_xray_sources(project, tmp)
                self.test_xray_lists(project)
                self.test_case_outputs_and_selectors(project)
                self.test_button_argv(project, tmp)
                self.test_panels(project)
                slicer.mrmlScene.Clear(0)
                widget.__dict__.pop("_start", None)
                self.test_stop_reaps_launcher_workers()
                if runs:
                    for path in runs:
                        self.check_run(path)
                        slicer.mrmlScene.Clear(0)
                    self.check_ct_reuse(runs[0])
                    slicer.mrmlScene.Clear(0)
                    for path in runs:
                        self.test_results_viewer(path)
                    self.test_view_only(runs[0], tmp)
                else:
                    print(f"FluoroReg self-test: ${TEST_RUNS_ENV} is not set; run checks skipped", flush=True)
                widget.template_edit.currentPath = previous[2]
        finally:
            if env_before is None:
                os.environ.pop("FLUOROREG_SETTINGS", None)
            else:
                os.environ["FLUOROREG_SETTINGS"] = env_before
            widget.__dict__.pop("_start", None)
            widget.clear_run()
            widget._loaded_outputs = set()
            widget.project_edit.text = previous[0]
            widget._checked, widget._xray_checked, widget._xray_seen = set(), set(), set()
            widget.refresh_status()
            widget.show_step(previous[1])
        self.delayDisplay(f"FluoroReg self-test passed ({len(runs)} run(s))")

    def test_stop_reaps_launcher_workers(self):
        """Stop must reach the Python behind PythonSlicer and its separate worker session."""
        if not sys.platform.startswith("linux"):
            return
        import time

        widget = slicer.modules.fluororeg.widgetRepresentation().self()
        previous = widget.project_edit.text
        widget.project_edit.text = ""
        pid = None
        identity = None

        def worker_state():
            try:
                with open(f"/proc/{pid}/stat") as stream:
                    fields = stream.read().rsplit(") ", 1)[1].split()
                return fields[19], fields[0]
            except (OSError, IndexError):
                return None

        try:
            with tempfile.TemporaryDirectory(prefix="fluororeg_cancel_") as tmp:
                pid_file = os.path.join(tmp, "worker.pid")
                late_file = os.path.join(tmp, "finished")
                worker = (f"import os,time\nopen({pid_file!r},'w').write(str(os.getpid()))\n"
                          f"time.sleep(60)\nopen({late_file!r},'w').write('completed')\n")
                log_path = os.path.join(tmp, "worker.log")
                code = ("import os,sys\nfrom fluororeg.runner import run_tool\n"
                        f"raise SystemExit(run_tool([sys.executable, '-c', {worker!r}], cwd={tmp!r}, "
                        f"log_path={log_path!r}, env=dict(os.environ)))\n")
                self.assertTrue(widget._start([Job("Cancellation regression", [cli_python(), "-c", code])]))
                deadline = time.monotonic() + 20
                while not os.path.isfile(pid_file) and widget.process is not None and time.monotonic() < deadline:
                    slicer.app.processEvents()
                    time.sleep(0.01)
                details = widget.status_label.text
                if os.path.isfile(log_path):
                    with open(log_path) as stream:
                        details += "\n" + stream.read()
                self.assertTrue(os.path.isfile(pid_file), details)
                with open(pid_file) as stream:
                    pid = int(stream.read())
                identity = worker_state()
                self.assertIsNotNone(identity)
                widget.stop()
                deadline = time.monotonic() + 20
                while widget.process is not None and time.monotonic() < deadline:
                    slicer.app.processEvents()
                    time.sleep(0.01)
                self.assertIsNone(widget.process)
                self.assertNotEqual(widget.last_exit_code, 0)
                self.assertIsNone(worker_state(), "Stop returned before its worker was reaped")
                self.assertFalse(os.path.exists(late_file), "worker finished after Stop")
        finally:
            current = worker_state()
            if identity and current and current[0] == identity[0] and current[1] != "Z":
                os.kill(pid, signal.SIGKILL)
            if widget.process is not None:
                widget.stop()
            widget.project_edit.text = previous

    # ------------------------------------------------------------ pure helpers
    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "settings.json")
            defaults = load_settings(path)
            self.assertEqual(defaults["annotation_3d_env"], "")
            self.assertEqual(defaults["xreg_dir"], "")
            self.assertEqual(defaults[TEMPLATE_KEY], "")
            self.assertNotIn("python3", defaults)
            values = {key: f"/x/{key}" for key, _l, _d, _k in SETTINGS_FIELDS}
            values["gpu"] = "1"
            values[TEMPLATE_KEY] = "/x/template.dcm"
            save_settings(values, path)
            self.assertEqual(load_settings(path), values)
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle), values)
        self.assertEqual(setting_check("xreg_build_dir", "dir", ""), (True, "default"))
        self.assertEqual(setting_check("xreg_dir", "dir", ""), (False, "missing"))
        self.assertEqual(setting_check("gpu", "text", "x"), (False, "not a number"))
        python = cli_python()
        self.assertEqual(os.path.basename(python), "PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
        self.assertTrue(os.path.isfile(python), python)

    def test_argv_of_every_step(self):
        values = {"xreg_dir": "/x", "gpu": "1", "pose_dir": "", "annotation_3d_env": "totalseg",
                  TEMPLATE_KEY: "/t.dcm"}
        tools = ["--xreg-dir", "/x", "--annotation-3d-env", "totalseg", "--gpu", "1", "--xray-template", "/t.dcm"]
        head = ["/py", "-m", "fluororeg"]
        P = "/p"
        cases = [
            (("init", {"project": P}), ["init", "--project", P] + tools),
            (("status", {"project": P}), ["status", "--project", P]),
            (("import-ct", {"project": P, "input_path": "/ct.nii.gz", "case": "c"}),
             ["import-ct", "--project", P, "--input", "/ct.nii.gz", "--case", "c"] + tools),
            (("annotate-3d", {"project": P, "case": "c"}), ["annotate-3d", "--project", P, "--case", "c"] + tools),
            (("import-xrays", {"project": P, "inputs": ["/a.dcm", "/b.png"]}),
             ["import-xrays", "--project", P, "--input", "/a.dcm", "/b.png"] + tools),
            (("pose-init-config", {"project": P, "case": "c"}),
             ["pose", "init-config", "--project", P, "--case", "c"] + tools),
            (("pose-train", {"project": P, "case": "c"}), ["pose", "train", "--project", P, "--case", "c"] + tools),
            (("pose-use-model", {"project": P, "case": "c", "model_dir": "/m"}),
             ["pose", "use-model", "--project", P, "--case", "c", "--model-dir", "/m"] + tools),
            (("config-create", {"project": P, "case": "c", "app": "multi-target", "femur": "left", "name": "n",
                                "ids": ["001", "002"]}),
             ["config", "create", "--project", P, "--case", "c", "--app", "multi-target", "--femur", "left",
              "--name", "n", "--ids", "001", "002"]),
            (("config-set", {"config": "/p/c.json", "ct": "a", "landmarks_3d": "b", "ids": ["003"]}),
             ["config", "set", "--config", "/p/c.json", "--ct", "a", "--landmarks-3d", "b", "--ids", "003"]),
            (("config-set", {"config": "/p/c.json", "experiment_dir": "/p", "ct": "a"}),
             ["config", "set", "--config", "/p/c.json", "--experiment-dir", "/p", "--ct", "a"]),
            (("config-show", {"config": "/p/c.json"}), ["config", "show", "--config", "/p/c.json"]),
            (("register", {"config": "/p/c.json"}), ["register", "--config", "/p/c.json", "--predict-poses"] + tools),
            (("register", {"config": "/p/c.json", "replay": True}),
             ["register", "--config", "/p/c.json", "--predict-poses", "--replay"] + tools),
            (("assess", {"run": "/p/outputs/c/r"}), ["assess", "--run", "/p/outputs/c/r"] + tools),
        ]
        for (step, kwargs), expected in cases:
            self.assertEqual(cli_argv(values, step, python="/py", **kwargs), head + expected, step)
        # Every argv parses with the real command-line parser.
        from fluororeg.cli import build_parser

        for (step, kwargs), _expected in cases:
            build_parser().parse_args(cli_argv(values, step, python="/py", **kwargs)[3:])
        self.assertEqual(cli_argv({}, "init", project=P)[0], cli_python())

    def test_events_and_progress(self):
        rec = parse_line('FLUOROREG {"event": "progress", "step": "register", "phase": "predict", "done": 2, '
                         '"total": 5}')
        self.assertEqual(rec[0], "pipeline")
        self.assertEqual(progress_update("Register", *rec), ("Register: predict 2/5", 2, 5))
        annotate = {"event": "progress", "step": "annotate-3d", "done": 2, "total": 7, "message": "totalseg"}
        self.assertEqual(progress_update("Segment and annotate", "pipeline", annotate),
                         ("Segment and annotate: totalseg 2/7", 2, 7))
        kind, pose = parse_line('BIGSS_POSE {"event": "progress", "stage": "train", "iteration": 30, "total": 100}')
        self.assertEqual(progress_update("Train pose model", kind, pose), ("Train pose model: train 30/100", 30, 100))
        self.assertEqual(progress_update("Train pose model", "pose", {"event": "progress", "stage": "self_check",
                                                                      "done": 4, "total": 100})[0],
                         "Train pose model: self check 4/100")
        self.assertEqual(progress_update("Register", "pipeline", {"event": "progress", "step": "register",
                                                                  "phase": "replay"}),
                         ("Register: replay", None, None))
        self.assertIsNone(progress_update("x", "pipeline", {"event": "result", "step": "register"}))
        kind, img = parse_line('[XREG:IMAGE] id=003 status=failed reason="no landmarks"')
        self.assertEqual((kind, img["id"], img["status"], img["reason"]), ("xreg", "003", "failed", "no landmarks"))
        self.assertIsNone(progress_update("x", kind, img))
        self.assertIsNone(parse_line("FLUOROREG not json"))
        self.assertIsNone(parse_line("plain tool output"))
        self.assertEqual(default_case_name("case01_ct"), "case01")
        self.assertEqual(default_case_name("Pelvis CT_CT"), sanitize_case_id("Pelvis CT"))
        self.assertEqual(default_case_name("scan"), "scan")

    @staticmethod
    def fixture_status(case="c", staged=True, model=True):
        present = {"present": staged}
        return {"project": "/p", "set_up": True, "project_name": "p",
                "cases": [{"case_id": case, "status": "staged" if staged else "imported",
                           "input": {"path": f"meta_data/tmp/original_inputs/{case}/{case}.nii.gz", "present": True},
                           "annotation_output": {"present": staged},
                           "staged": {"ct": {"path": f"meta_data/volumes/{case}_ct.nii.gz", **present},
                                      "seg": {"path": f"meta_data/volumes/{case}_seg.nii.gz", **present},
                                      "landmarks_3d": {"path": f"meta_data/landmarks/landmarks_3d/"
                                                               f"{case}_landmarks_3d.fcsv", **present}}}],
                "xrays": {"ids": ["001", "002", "003"], "converted": ["001", "002"],
                          "entries": [{"id": i, "source_name": f"{i}.dcm", "h5": i != "003"}
                                      for i in ("001", "002", "003")]},
                "pose": {case: {"active_model": "/abs/m" if model else None,
                                "model": {"active": "meta_data/pose_models/c/20261005_164312",
                                          "name": "20261005_164312",
                                          "self_check_capture": 0.977, "trained": "2026-10-05"} if model
                                else {"active": None, "name": None, "self_check_capture": None, "trained": None}}},
                "xreg_configs": [{"config": f"configs/xreg/single_target_{case}.json", "app": "single-target",
                                  "problems": []},
                                 {"config": "configs/xreg/mine.json", "app": "single-target-multiview",
                                  "problems": []}],
                "runs": [{"run_dir": "outputs/c/20260101_000000_a", "assessment": {"counts": {"high": 2}}},
                         {"run_dir": "outputs/c/20260102_000000_a",
                          "assessment": {"counts": {"high": 1, "medium": 1}}}]}

    def test_checklist_rules(self):
        status = self.fixture_status()
        got = step_checklist(status, "c", ["xreg repo"], {"label": "Single view (pelvis)", "reason": None})
        self.assertEqual({k: v["mark"] for k, v in got.items()},
                         {"settings": "error", "experiment": "done", "ct": "done", "xrays": "warning",
                          "pose": "done", "registration": "done", "results": "warning"})
        self.assertEqual(got["xrays"]["summary"], "3 in the experiment, 1 not converted")
        self.assertEqual(got["pose"]["summary"], "model 20261005_164312, self-check 98 %")
        self.assertEqual(got["registration"]["summary"], "Single view (pelvis): ready")
        self.assertEqual(got["results"]["summary"], "2 run(s); latest 1 medium certainty")
        status["xrays"]["converted"] = ["001", "002", "003"]
        status["runs"] = status["runs"][:1]
        got = step_checklist(status, "c", (), {"label": "x", "reason": "check at least one X-ray"})
        self.assertEqual((got["settings"]["mark"], got["xrays"]["mark"], got["results"]["mark"]),
                         ("done", "done", "done"))
        self.assertEqual((got["registration"]["mark"], got["registration"]["summary"]),
                         ("pending", "check at least one X-ray"))
        status["cases"][0]["staged"]["seg"]["present"] = False
        self.assertEqual(step_checklist(status, "c")["ct"]["mark"], "warning")  # annotated, not staged
        status["cases"][0]["annotation_output"]["present"] = False
        self.assertEqual(step_checklist(status, "c")["ct"]["mark"], "pending")
        unmodelled = self.fixture_status(model=False)
        self.assertEqual(step_checklist(unmodelled, "c")["pose"], {"mark": "pending", "summary": "no model"})
        old = self.fixture_status()
        del old["pose"]["c"]["model"]  # a status without the model section: the active model folder name
        self.assertEqual(pose_model(old, "c"), (True, "model m"))
        empty = step_checklist(None)
        self.assertEqual((empty["experiment"]["mark"], empty["xrays"]["mark"], empty["results"]["mark"]),
                         ("pending", "pending", "pending"))
        # A folder without project_manifest.json: results only, the pipeline steps wait for Set up.
        viewing = {**self.fixture_status(), "set_up": False, "project_name": None, "cases": []}
        got = step_checklist(viewing, None, (), {"label": None, "reason": SET_UP_REASON})
        self.assertEqual(got["experiment"], {"mark": "warning", "summary": "results only"})
        for key in ("ct", "xrays", "pose", "registration"):
            self.assertEqual(got[key], {"mark": "pending", "summary": SET_UP_REASON})
        self.assertEqual(got["results"]["summary"], "2 run(s); latest 1 medium certainty")
        unknown = {k: v for k, v in viewing.items() if k != "set_up"}  # a status without the field: the file
        self.assertFalse(is_set_up(unknown))
        with tempfile.TemporaryDirectory() as tmp:
            unknown["project"] = tmp
            open(os.path.join(tmp, "project_manifest.json"), "w").close()
            self.assertTrue(is_set_up(unknown))

    def test_registration_choices_and_reasons(self):
        import numpy as np

        status = self.fixture_status()
        choices = registration_choices(status, "c")
        self.assertEqual([c["label"] for c in choices],
                         [label for _a, _f, label in DEFAULT_REGISTRATIONS] + ["mine"])
        self.assertEqual([c["key"] for c in choices[:3]],
                         ["single_target_c", "single_target_multiview_c", "multi_target_c_femur-left"])
        self.assertEqual([c["exists"] for c in choices], [True, False, False, False, False, False, True])
        self.assertEqual([c["multiview"] for c in choices], [False, True, False, False, True, True, True])
        self.assertEqual(choices[4]["app"], "multi-target-multiview")
        self.assertEqual(choices[4]["femur"], "left")
        self.assertEqual([c["key"] for c in registration_choices(status, None)], ["mine", "single_target_c"])
        ct = slicer.util.addVolumeFromArray(np.zeros((3, 4, 5), dtype=np.int16), name="reason ct")
        flat = slicer.util.addVolumeFromArray(np.zeros((1, 4, 5), dtype=np.int16), name="reason xray")
        nodes = {"ct": ct, "seg": object(), "landmarks_3d": object()}
        single, multi = choices[0], choices[1]
        self.assertIsNone(registration_reason(status, "c", single, ["001"], nodes))
        self.assertEqual(registration_reason(None, "c", single, ["001"], nodes), "open an experiment folder")
        self.assertEqual(registration_reason(status, "x", single, ["001"], nodes),
                         "no case: segment and annotate a CT")
        self.assertEqual(registration_reason(self.fixture_status(staged=False), "c", single, ["001"], nodes),
                         "c: CT not annotated")
        self.assertEqual(registration_reason(self.fixture_status(model=False), "c", single, ["001"], nodes),
                         "no pose model: train one or use an existing one")
        self.assertEqual(registration_reason(status, "c", None, ["001"], nodes), "select a registration")
        self.assertEqual(registration_reason(status, "c", single, [], nodes), "check at least one X-ray")
        self.assertEqual(registration_reason(status, "c", multi, ["001"], nodes), "multiview needs at least two X-rays")
        self.assertIsNone(registration_reason(status, "c", multi, ["001", "002"], nodes))
        self.assertEqual(registration_reason(status, "c", single, ["003"], nodes), "X-ray 003 not converted to H5")
        self.assertEqual(registration_reason(status, "c", single, ["001"], {**nodes, "seg": None}),
                         "select the segmentation")
        self.assertEqual(registration_reason({**status, "set_up": False}, "c", single, ["001"], nodes),
                         SET_UP_REASON)
        # A 2D image selected as the CT (e.g. by scripting) is not a CT volume.
        self.assertEqual(registration_reason(status, "c", single, ["001"], {**nodes, "ct": flat}),
                         "the CT must be a 3D volume (reason xray is a 2D image)")
        remove_node(ct)
        remove_node(flat)

    def test_relocated_config(self):
        """A config whose experiment_dir names another folder: its inputs resolve against the folder holding
        it and Register sets experiment_dir to the open experiment."""
        moved = {"experiment_dir": "/old/exp", "resolved_dir": "/new/exp", "relocated": True,
                 "ct": "meta_data/volumes/c_ct.nii.gz", "seg": None, "landmarks_3d": None, "ids": ["001"]}
        self.assertEqual(config_targets(moved)["ct"], os.path.join("/new/exp", "meta_data", "volumes", "c_ct.nii.gz"))
        self.assertEqual(config_changes(moved, {}, ["001"], "/new/exp"), {"experiment_dir": "/new/exp"})
        self.assertEqual(config_changes(moved, {}, ["001", "002"], "/new/exp"),
                         {"experiment_dir": "/new/exp", "ids": ["001", "002"]})
        here = {**moved, "experiment_dir": "/new/exp", "relocated": False}
        self.assertEqual(config_changes(here, {}, ["001"], "/new/exp"), {})
        older = {k: v for k, v in here.items() if k not in ("resolved_dir", "relocated")}  # without the new fields
        self.assertEqual(config_targets(older)["ct"], os.path.join("/new/exp", "meta_data", "volumes", "c_ct.nii.gz"))

    # ------------------------------------------------------------ scene fixtures
    def make_fixture_project(self, tmp):
        """An experiment with case "case" staged (CT, label map, 3D landmarks, two STL models), DICOM X-rays
        001 and 002 and one xreg config."""
        import numpy as np
        import vtk

        from fluororeg.project import init_project, update_case

        project = os.path.join(tmp, "project")
        init_project(project)
        vol_dir = os.path.join(project, "meta_data", "volumes")
        ct = slicer.util.addVolumeFromArray(np.arange(120, dtype=np.int16).reshape(4, 5, 6), name="case_ct")
        ct.SetSpacing(0.8, 0.8, 1.5)
        self.assertTrue(slicer.util.saveNode(ct, os.path.join(vol_dir, "case_ct.nii.gz")))
        other = slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), dtype=np.int16), name="other_ct")
        self.assertTrue(slicer.util.saveNode(other, os.path.join(vol_dir, "other_ct.nii.gz")))
        seg = slicer.util.addVolumeFromArray(np.ones((4, 5, 6), dtype=np.uint8), name="case_seg",
                                             nodeClassName="vtkMRMLLabelMapVolumeNode")
        self.assertTrue(slicer.util.saveNode(seg, os.path.join(vol_dir, "case_seg.nii.gz")))
        lm = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "case_lm")
        lm.AddControlPoint(1.0, 2.0, 3.0, "ASIS-l")
        self.assertTrue(slicer.util.saveNode(lm, os.path.join(project, "meta_data", "landmarks", "landmarks_3d",
                                                              "case_landmarks_3d.fcsv")))
        for name in ("case_label_1_hip.stl", "case_label_2_femur_left.stl"):
            sphere = vtk.vtkSphereSource()
            sphere.Update()
            writer = vtk.vtkSTLWriter()
            writer.SetInputData(sphere.GetOutput())
            writer.SetFileName(os.path.join(project, "meta_data", "models", name))
            writer.Write()
        update_case(project, "case", status="staged", staged=dict(staged_paths("case")))
        for pid in ("001", "002"):
            self.write_dicom(os.path.join(project, *XRAY_DIR.split("/"), f"{pid}.dcm"))
        config = {"experiment_dir": project,
                  "volume": {"ct": "meta_data/volumes/case_ct.nii.gz", "seg": "meta_data/volumes/case_seg.nii.gz",
                             "landmarks_3d": "meta_data/landmarks/landmarks_3d/case_landmarks_3d.fcsv"},
                  "projections": {"dir": "meta_data/XrayProjData", "pattern": "{id}.h5", "ids": ["002"]},
                  "targets": [{"key": "pelvis", "landmark_set": "pelvis"}],
                  "output": {"dir": "outputs/case"}}
        with open(os.path.join(project, "configs", "xreg", "single_target_case.json"), "w", encoding="utf-8") as h:
            json.dump(config, h, indent=2)
        slicer.mrmlScene.Clear(0)
        return project

    @staticmethod
    def write_dicom(path, rows=4, cols=5):
        """A minimal single-frame 16-bit DX DICOM file."""
        import numpy as np
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = generate_uid(), generate_uid()
        ds.Modality, ds.PatientName, ds.PatientID = "DX", "test", "test"
        ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = rows, cols, 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.PixelSpacing = [0.2, 0.2]
        ds.PixelData = np.arange(rows * cols, dtype="<u2").tobytes()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ds.save_as(path, write_like_original=False)
        assert pydicom.dcmread(path).Rows == rows
        return path

    def _widget(self, project, status=None):
        widget = slicer.modules.fluororeg.widgetRepresentation().self()
        widget._updating = True
        widget.project_edit.text = project
        widget._updating = False
        widget.status = status
        widget._fill_from_status()
        return widget

    def test_ct_export_and_adoption(self, project, tmp):
        import numpy as np

        logic = FluoroRegLogic()
        export = os.path.join(tmp, "export")
        outside = os.path.join(tmp, "raw_ct.nii.gz")
        node = slicer.util.addVolumeFromArray(np.arange(120, dtype=np.int16).reshape(4, 5, 6), name="raw_ct")
        node.SetSpacing(0.8, 0.8, 1.5)
        self.assertEqual(ct_export_plan(node, export, "raw"), (os.path.join(export, "raw.nii.gz"), True))  # no file
        self.assertTrue(slicer.util.saveNode(node, outside))
        node = slicer.util.loadVolume(outside)
        self.assertEqual(ct_export_plan(node, export, "raw"), (outside, False))  # unmodified NIfTI passed through
        slicer.util.arrayFromVolume(node)[0, 0, 0] = 7
        slicer.util.arrayFromVolumeModified(node)
        self.assertTrue(node.GetModifiedSinceRead())
        self.assertEqual(ct_export_plan(node, export, "raw"), (os.path.join(export, "raw.nii.gz"), True))
        nrrd = os.path.join(tmp, "raw_ct.nrrd")
        self.assertTrue(slicer.util.saveNode(node, nrrd))
        self.assertEqual(ct_export_plan(slicer.util.loadVolume(nrrd), export, "raw")[1], True)  # NIfTI only
        # Adoption: same image -> the node now points at the staged file, no second copy.
        staged = os.path.join(project, "meta_data", "volumes", "case_ct.nii.gz")
        same = slicer.util.loadVolume(outside)
        n_volumes = len(scene_volumes())
        self.assertEqual(logic.adopt_staged_ct(same, staged).GetID(), same.GetID())
        self.assertEqual(os.path.realpath(node_file(same)), os.path.realpath(staged))
        self.assertEqual(len(scene_volumes()), n_volumes)
        self.assertEqual(logic.adopt_staged_ct(None, staged).GetID(), same.GetID())  # found by file
        slicer.mrmlScene.Clear(0)
        # Different image -> the staged CT is loaded as its own node.
        modified = slicer.util.loadVolume(outside)
        slicer.util.arrayFromVolume(modified)[0, 0, 0] = 9
        slicer.util.arrayFromVolumeModified(modified)
        adopted = logic.adopt_staged_ct(modified, staged)
        self.assertNotEqual(adopted.GetID(), modified.GetID())
        self.assertEqual(os.path.realpath(node_file(adopted)), os.path.realpath(staged))
        self.assertEqual(os.path.realpath(node_file(modified)), os.path.realpath(outside))
        slicer.mrmlScene.Clear(0)

    def test_xray_sources(self, project, tmp):
        import numpy as np
        from PIL import Image

        dcm = self.write_dicom(os.path.join(tmp, "raw", "5_1.dcm"))
        node = FluoroRegLogic.load_xray_file(dcm)
        self.assertTrue(is_xray_like(node))
        self.assertFalse(is_ct_like(node))
        self.assertEqual(xray_source(node, None), ("dicom", dcm))
        png = os.path.join(tmp, "raw", "shot.png")
        Image.fromarray(np.arange(20, dtype=np.uint16).reshape(4, 5) * 1000).save(png)
        image_node = FluoroRegLogic.load_xray_file(png)
        self.assertTrue(is_xray_like(image_node))
        self.assertEqual(xray_source(image_node, None), ("image", png))
        loose = slicer.util.addVolumeFromArray(np.arange(20, dtype=np.float32).reshape(1, 4, 5) * 3000.0,
                                               name="from DICOM db")
        loose.SetAttribute(DICOM_UIDS_ATTR, "1.2.3 1.2.4")
        lookup = {"1.2.3": dcm}.get
        self.assertEqual(xray_source(loose, lookup), ("dicom", dcm))
        self.assertEqual(xray_source(loose, {}.get), (None, None))
        out = save_xray_png(loose, os.path.join(tmp, "export", "loose.png"))
        with Image.open(out) as img:
            self.assertEqual(img.mode, "I;16")
            self.assertEqual(img.size, (5, 4))
            self.assertEqual(np.asarray(img).astype(np.int64).tolist(),
                             np.clip(np.arange(20).reshape(4, 5) * 3000, 0, 65535).tolist())
        ct = slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), dtype=np.int16), name="ct")
        self.assertTrue(is_ct_like(ct))
        self.assertFalse(is_xray_like(ct))
        slicer.mrmlScene.Clear(0)

    def test_xray_lists(self, project):
        import numpy as np
        import qt

        status = self.fixture_status("case")
        status["project"] = project
        status["xrays"] = {"ids": ["001", "002"], "converted": ["001"],
                           "entries": [{"id": "001", "h5": True}, {"id": "002", "h5": False}]}
        logic = FluoroRegLogic()
        from_file = logic.load_xray_file(os.path.join(project, *XRAY_DIR.split("/"), "001.dcm"))  # by its file
        tagged = slicer.util.addVolumeFromArray(np.zeros((1, 4, 5), dtype=np.float32), name="tagged")
        tag_xray(tagged, project, "002")
        other = slicer.util.addVolumeFromArray(np.zeros((1, 4, 5), dtype=np.float32), name="other experiment")
        tag_xray(other, project + "_other", "002")
        slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), dtype=np.float32), name="a CT")
        self.assertEqual(xray_experiment_id(from_file, project), "001")
        self.assertEqual(xray_experiment_id(tagged, project), "002")
        self.assertIsNone(xray_experiment_id(other, project))
        from_db = slicer.util.addVolumeFromArray(np.zeros((1, 4, 5), dtype=np.float32), name="from db")
        from_db.SetAttribute(DICOM_UIDS_ATTR, "1.2.9")
        lookup = {"1.2.9": os.path.join(project, *XRAY_DIR.split("/"), "002.dcm")}.get
        self.assertEqual(xray_experiment_id(from_db, project, lookup), "002")  # its DICOM database file
        self.assertIsNone(xray_experiment_id(from_db, project, {}.get))
        slicer.mrmlScene.RemoveNode(from_db)
        self.assertEqual({k: v.GetName() for k, v in experiment_xray_nodes(project).items()},
                         {"001": from_file.GetName(), "002": "tagged"})
        widget = self._widget(project, status)
        texts = [widget.xray_scene_list.item(i).text() for i in range(widget.xray_scene_list.count)]
        self.assertEqual(texts, [f"{from_file.GetName()} — in experiment (id 001)", "tagged — in experiment (id 002)",
                                 "other experiment — not added"])
        widget.set_xray_checked(["other experiment", "tagged"])
        self.assertEqual([n.GetName() for n in widget.checked_xray_nodes()], ["tagged", "other experiment"])
        self.assertTrue(widget.add_to_experiment_button.enabled)
        # Registration list: the experiment's ids (status) with the scene node names (just the id when equal).
        texts = [widget.xray_list.item(i).text() for i in range(widget.xray_list.count)]
        self.assertEqual(texts, ["001", "002 — tagged, no H5"])
        slicer.mrmlScene.RemoveNode(tagged)
        widget._rebuild_xray_list()
        self.assertEqual(widget.xray_list.item(1).text(), "002 (not in the scene), no H5")
        n_volumes = len(scene_volumes())
        widget.xray_list.item(1).setCheckState(qt.Qt.Checked)  # checking loads it
        slicer.app.processEvents()
        self.assertEqual(widget.checked_ids(), ["002"])
        self.assertEqual(len(scene_volumes()), n_volumes + 1)
        self.assertEqual(widget.xray_list.item(1).text(), "002 — xray-002, no H5")
        self.assertEqual(widget.set_checked_ids(["001", "002", "999"]), ["001", "002"])
        json.dumps(widget.state())
        widget._checked, widget._xray_checked, widget._xray_seen = set(), set(), set()
        slicer.mrmlScene.Clear(0)

    def test_case_outputs_and_selectors(self, project):
        import numpy as np

        status = self.fixture_status("case")
        status["project"] = project
        status["xreg_configs"] = [{"config": "configs/xreg/single_target_case.json", "app": "single-target",
                                   "problems": []}]
        status["xrays"]["converted"] = ["001"]
        ct = slicer.util.loadVolume(os.path.join(project, "meta_data", "volumes", "case_ct.nii.gz"))
        ct.SetName("case_ct")
        widget = self._widget(project, status)
        widget.select_ct(ct)  # the annotated case CT: its segmentation, landmarks and models are loaded
        self.assertEqual(widget.ct_case_name(), "case")
        self.assertIn("Annotated: case case", widget.ct_info_label.text)
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")), 1)
        self.assertEqual(len(slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")), 1)
        models = [m for m in slicer.util.getNodesByClass("vtkMRMLModelNode") if node_file(m)]
        self.assertEqual(len(models), 2)
        colors = {tuple(round(c, 3) for c in m.GetDisplayNode().GetColor()) for m in models}
        self.assertEqual(len(colors), 2)
        self.assertEqual(len([n for n in scene_volumes() if is_ct_like(n)]), 1)  # no second CT
        # Registration selectors default to the selected registration's config (the case's files).
        widget._fill_registrations()
        self.assertTrue(widget.select_registration("single_target_case"))
        nodes = widget.selector_nodes()
        self.assertEqual(nodes["ct"].GetID(), ct.GetID())
        self.assertEqual({k: project_relative(node_file(n), project) for k, n in nodes.items()},
                         staged_paths("case"))
        self.assertEqual(widget.checked_ids(), ["002"])
        self.assertEqual(widget.reg_reason_label.text, "X-ray 002 not converted to H5")
        self.assertFalse(widget.register_button.enabled)
        widget.set_checked_ids(["001"])
        self.assertEqual(widget.reg_reason_label.text, "")
        self.assertTrue(widget.register_button.enabled)
        self.assertEqual(widget.checklist_state["registration"]["summary"], "Single view (pelvis): ready")
        # The CT selector lists 3D volumes only and keeps its node when X-rays are checked (loaded) or loaded.
        ct_combo = widget.selectors["ct"]

        def listed():
            slicer.app.processEvents()  # the hidden nodes are updated from the event loop
            return [ct_combo.nodeFromIndex(i).GetID() for i in range(ct_combo.nodeCount())]

        self.assertEqual(len([n for n in scene_volumes() if is_xray_like(n)]), 1)  # xray-001, loaded by its check
        self.assertEqual(listed(), [ct.GetID()])
        widget.set_checked_ids(["001", "002"])  # loads 002
        loaded = FluoroRegLogic.load_xray_file(os.path.join(project, *XRAY_DIR.split("/"), "001.dcm"))
        slicer.app.processEvents()
        self.assertEqual(ct_combo.currentNodeID, ct.GetID())
        self.assertEqual(len([n for n in scene_volumes() if is_xray_like(n)]), 3)
        self.assertEqual(listed(), [ct.GetID()])
        widget.set_checked_ids(["001"])  # 002 has no H5: its reason would come first
        # Removing the selected node empties the selector (it does not move to a neighbouring node).
        spare = slicer.util.addVolumeFromArray(np.zeros((3, 4, 5), dtype=np.int16), name="spare ct")
        ct_combo.setCurrentNode(spare)
        remove_node(spare)
        self.assertIsNone(ct_combo.currentNode())
        self.assertEqual(widget.reg_reason_label.text, "select the CT")
        # A 2D image selected anyway (scripting, before the list hides it) is no CT for Register.
        flat = slicer.util.addVolumeFromArray(np.zeros((1, 4, 5), dtype=np.int16), name="flat shot")
        ct_combo.setCurrentNode(flat)
        self.assertEqual(widget.reg_reason_label.text, "the CT must be a 3D volume (flat shot is a 2D image)")
        self.assertFalse(widget.register_button.enabled)
        ct_combo.setCurrentNode(ct)
        slicer.app.processEvents()
        self.assertEqual(listed(), [ct.GetID()])  # once deselected it is hidden
        remove_node(flat)
        remove_node(loaded)
        self.assertEqual(widget.reg_reason_label.text, "")
        # A default registration whose config does not exist yet uses the case's files too.
        self.assertTrue(widget.select_registration("Multiview (pelvis)"))
        self.assertEqual(widget.reg_reason_label.text, "multiview needs at least two X-rays")
        self.assertIn("created from the default", widget.reg_info_label.text)
        widget.status = None
        widget._loaded_outputs = set()
        slicer.mrmlScene.Clear(0)

    def test_button_argv(self, project, tmp):
        """Every button's command lines (the process start replaced by a recorder)."""
        import numpy as np
        from PIL import Image

        status = self.fixture_status("case")
        status["project"] = project
        status["xreg_configs"] = [{"config": "configs/xreg/single_target_case.json", "app": "single-target",
                                   "problems": []}]
        status["xrays"]["converted"] = ["001", "002", "003"]
        widget = self._widget(project, status)
        jobs = []
        widget._start = lambda js: jobs.extend(js) or True
        widget.set_xray_template("")
        tools = tool_flags(widget.settings_values())
        py = [cli_python(), "-m", "fluororeg"]

        def argvs():
            out = [j.argv for j in jobs]
            jobs.clear()
            return out

        # Segment and annotate: an unmodified NIfTI file is passed as is; a node without a file is saved first.
        raw = os.path.join(tmp, "raw", "pelvis_ct.nii.gz")
        node = slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), dtype=np.int16), name="pelvis_ct")
        self.assertTrue(slicer.util.saveNode(node, raw))
        node = slicer.util.loadVolume(raw)
        node.SetName("pelvis_ct")
        self.assertTrue(widget.run_annotate(node))
        self.assertEqual(argvs(), [py + ["import-ct", "--project", project, "--input", raw, "--case", "pelvis"] + tools,
                                   py + ["annotate-3d", "--project", project, "--case", "pelvis"] + tools])
        unsaved = slicer.util.addVolumeFromArray(np.zeros((4, 5, 6), dtype=np.int16), name="scan")
        self.assertTrue(widget.run_annotate(unsaved, case="hip"))
        exported = os.path.join(project, *EXPORT_DIR.split("/"), "hip.nii.gz")
        self.assertTrue(os.path.isfile(exported))
        self.assertEqual(argvs()[0], py + ["import-ct", "--project", project, "--input", exported, "--case", "hip"]
                         + tools)
        widget.ct_case_edit.text = ""
        # Add to experiment: DICOM file, image file (template set), array-only image exported as PNG.
        dcm = self.write_dicom(os.path.join(tmp, "raw", "9_1.dcm"))
        widget.set_xray_template(dcm)
        png = os.path.join(tmp, "raw", "shot2.png")
        Image.fromarray(np.full((4, 5), 1000, dtype=np.uint16)).save(png)
        nodes = widget.add_xray_files([dcm, png])
        loose = slicer.util.addVolumeFromArray(np.ones((1, 4, 5), dtype=np.float32), name="loose shot")
        self.assertTrue(widget.add_xrays_to_experiment(nodes + [loose]))
        argv = argvs()[0]
        exported = os.path.join(project, *EXPORT_DIR.split("/"), "loose_shot.png")
        self.assertEqual(argv, py + ["import-xrays", "--project", project, "--input",
                                     *[node_file(n) for n in nodes], exported] + tools + ["--xray-template", dcm])
        self.assertTrue(os.path.isfile(exported))
        widget.set_xray_template("")
        self.assertFalse(widget.add_xrays_to_experiment([loose]))  # image without a template
        self.assertIn("Template DICOM", widget.messages["xray"]["text"])
        self.assertEqual(argvs(), [])
        # Pose model.
        self.assertTrue(widget.run_pose_train())
        self.assertTrue(widget.use_pose_model("/models/m1"))
        self.assertTrue(widget.open_pose_config())  # missing: init-config first
        self.assertEqual(argvs(), [py + ["pose", "train", "--project", project, "--case", "case"] + tools,
                                   py + ["pose", "use-model", "--project", project, "--case", "case", "--model-dir",
                                         "/models/m1"] + tools,
                                   py + ["pose", "init-config", "--project", project, "--case", "case"] + tools])
        # Register, existing config: config set with the changed fields, then register --predict-poses.
        widget._fill_registrations()
        self.assertTrue(widget.select_registration("single_target_case"))
        widget.set_checked_ids(["001", "002"])
        config = os.path.join(project, "configs", "xreg", "single_target_case.json")
        self.assertTrue(widget.run_register())
        self.assertEqual(argvs(), [py + ["config", "set", "--config", config, "--ids", "001", "002"],
                                   py + ["register", "--config", config, "--predict-poses"] + tools])
        # Register on a moved experiment's config (experiment_dir names another folder): its files resolve against
        # this experiment, an info line says so, and config set sets experiment_dir to this folder.
        with open(config, encoding="utf-8") as handle:
            original = handle.read()
        moved = json.loads(original)
        moved["experiment_dir"] = os.path.join(tmp, "old location")
        with open(config, "w", encoding="utf-8") as handle:
            json.dump(moved, handle)
        widget.status["xreg_configs"][0]["relocated"] = True
        self.assertTrue(widget.select_registration("single_target_case"))
        self.assertEqual(widget.reg_relocated_label.text, RELOCATED_NOTE)
        self.assertEqual({k: project_relative(node_file(n), project) for k, n in widget.selector_nodes().items()},
                         staged_paths("case"))
        self.assertEqual(widget.reg_reason_label.text, "")
        self.assertTrue(widget.run_register())
        self.assertEqual(argvs(), [py + ["config", "set", "--config", config, "--experiment-dir", project],
                                   py + ["register", "--config", config, "--predict-poses"] + tools])
        with open(config, "w", encoding="utf-8") as handle:
            handle.write(original)
        widget.status["xreg_configs"][0]["relocated"] = False
        widget._refresh_registration()
        self.assertEqual(widget.reg_relocated_label.text, "")
        widget.set_checked_ids(["001", "002"])
        # Register, default config not created yet, CT selector not a file of the experiment: saved into it.
        self.assertTrue(widget.select_registration("multi_target_case_femur-right"))
        widget.selectors["ct"].setCurrentNode(unsaved)
        unsaved.SetName("edited ct")
        storage = unsaved.GetStorageNode()
        if storage is not None:  # saved by Segment and annotate above: make it a scene-only node again
            slicer.mrmlScene.RemoveNode(storage)
        self.assertTrue(widget.run_register())
        config = os.path.join(project, "configs", "xreg", "multi_target_case_femur-right.json")
        self.assertEqual(argvs(), [
            py + ["config", "create", "--project", project, "--case", "case", "--app", "multi-target", "--femur",
                  "right", "--name", "multi_target_case_femur-right", "--ids", "001", "002"],
            py + ["config", "set", "--config", config, "--ct", "meta_data/volumes/edited_ct.nii.gz"],
            py + ["register", "--config", config, "--predict-poses"] + tools])
        self.assertTrue(os.path.isfile(os.path.join(project, "meta_data", "volumes", "edited_ct.nii.gz")))
        # Edit config of a default not created yet: config create.
        self.assertTrue(widget.select_registration("single_target_multiview_case"))
        self.assertTrue(widget.open_config())
        self.assertEqual(argvs()[0][3:7], ["config", "create", "--project", project])
        # Assess: only a run without a current assessment.json (it writes into that run); an assessment of an
        # older schema is stale and can be assessed again.
        done_run, new_run = "outputs/case/20260102_000000_a", "outputs/case/20260103_000000_b"
        stale_run = "outputs/case/20260101_000000_c"
        widget.status["runs"] = [{"run_dir": new_run, "app": "xreg-single-target-regi"},
                                 {"run_dir": done_run, "assessment": {"counts": {"high": 1}, "items": []}},
                                 {"run_dir": stale_run, "assessment": {"stale": True, "error": "schema 1"}}]
        widget._fill_runs()
        self.assertTrue(widget.select_run(done_run))
        self.assertFalse(widget.assess_button.enabled)
        self.assertFalse(widget.run_assess())
        self.assertEqual(widget.messages["results"]["text"], "the run has an assessment.json")
        self.assertTrue(widget.select_run(stale_run))
        self.assertTrue(widget.assess_button.enabled)
        self.assertTrue(widget.select_run(new_run))
        self.assertTrue(widget.assess_button.enabled)
        self.assertTrue(widget.run_assess())
        self.assertEqual(argvs(), [py + ["assess", "--run", os.path.join(project, *new_run.split("/"))] + tools])
        self.assertTrue(widget.show_run_button.enabled)
        self.assertFalse(widget.show_run())  # no such run folder: inline error, nothing shown
        self.assertIn(f"showing {new_run} failed", widget.messages["results"]["text"])
        self.assertIsNone(widget.state()["viewer"])
        widget.status = None
        widget._checked, widget._xray_checked, widget._xray_seen = set(), set(), set()
        widget._loaded_outputs = set()
        slicer.mrmlScene.Clear(0)

    def test_panels(self, project):
        widget = self._widget(project, self.fixture_status())
        self.assertEqual(widget.stack.count, len(STEPS) + 1)
        for key in [k for k, _t in STEPS] + ["log"]:
            self.assertTrue(widget.show_step(key))
            self.assertIs(widget.stack.currentWidget(), widget.pages[key])
            self.assertTrue(widget.step_buttons[key].checked)
            self.assertEqual(widget.state()["step"], key)
        self.assertTrue(widget.pages["registration"].isAncestorOf(widget.xray_list))
        self.assertTrue(widget.pages["xrays"].isAncestorOf(widget.xray_scene_list))
        self.assertTrue(widget.pages["log"].isAncestorOf(widget.log_edit))
        self.assertFalse(widget.case_row.visible)  # one case: no case combo
        widget.step_buttons["pose"].click()
        self.assertEqual(widget.current_step, "pose")
        self.assertEqual(widget.pose_model_label.text, "model 20261005_164312, self-check 98 %")
        self.assertEqual(slicer.util.findChildren(widget.parent, className="ctkCollapsibleButton"), [])
        with self.assertRaises(ValueError):
            widget.show_step("nope")
        json.dumps(widget.state())
        # Results: the display controls of the shown run sit in the panel, disabled until a run is shown.
        for w in widget.viewer_controls + [widget.show_run_button, widget.clear_run_button, widget.assess_button]:
            self.assertTrue(widget.pages["results"].isAncestorOf(w), w.objectName)
            self.assertTrue(w.objectName.startswith("pipeline."), w.objectName)
        self.assertFalse(any(w.enabled for w in widget.viewer_controls))
        self.assertFalse(widget.clear_run_button.enabled)
        self.assertFalse(widget.set_view("1"))
        self.assertEqual(widget.messages["results"]["text"], "show a run first")
        self.assertEqual(widget.show_result_check.objectName, "pipeline.showResultCheck")
        self.assertTrue(widget.show_result_check.checked)
        with self.assertRaises(ValueError):
            widget.set_display(nope=True)
        widget.status = None
        widget._fill_from_status()

    # ------------------------------------------------------------ runs ($FLUOROREG_TEST_RUNS)
    def test_results_viewer(self, path):
        """Rule 1 on a run folder (its experiment opens, only read, and the run is shown), then the Results
        step's display controls driven through the public methods."""
        import numpy as np

        run = XregRun.load(path)
        widget = slicer.modules.fluororeg.widgetRepresentation().self()
        manifest = os.path.join(str(run.experiment_dir), "project_manifest.json")
        stamp = os.path.getmtime(manifest) if os.path.isfile(manifest) else None
        self.assertIsNotNone(widget.set_project(path), widget.messages)
        self.assertEqual(widget.project, str(run.experiment_dir))
        self.assertEqual(os.path.getmtime(manifest) if stamp is not None else None, stamp)  # status only
        rel = os.path.relpath(str(run.run_dir), widget.project).replace(os.sep, "/")
        state = widget.state()
        self.assertEqual((state["step"], state["selected_run"], state["viewer"]["run"]), ("results", rel, rel))
        self.assertEqual(state["set_up"], stamp is not None)
        scene = widget.run_viewer.scene
        self.assertTrue(all(w.enabled for w in widget.viewer_controls))
        self.assertEqual(widget.view_combo.count, len(run.view_ids))
        for view_id in [v for v in run.view_ids if run.pose_stages(v, scene.primary)]:
            self.assertTrue(widget.set_view(view_id))
            self.assertEqual(scene.view_id, view_id)
            self.assertEqual(widget.view_combo.itemData(widget.view_combo.currentIndex), view_id)
            self.assertTrue(widget.view_combo.currentText.startswith(f"{view_id} ("))
            self.assertEqual(set(widget.stage_combos), set(run.target_keys))
            for key in run.target_keys:
                for stage in run.pose_stages(view_id, key):
                    self.assertTrue(widget.set_stage(key, stage))
                    self.assertEqual(scene.stages[key], stage)
                    self.assertEqual(widget.stage_combos[key].currentText, stage)
                    all_final = all(scene.stages.get(t) == "final" for t in run.target_keys)
                    self.assertEqual(view_id in scene.scene_state.final_drr_views, all_final)
                    if not all_final:
                        self.assertIn("Final-result DRR hidden", widget.viewer_info_label.text)
                        vn = scene.views[view_id]
                        from fluororeg_viewer.scene import composite_rgb
                        import numpy as np
                        np.testing.assert_array_equal(
                            slicer.util.arrayFromVolume(vn.composite)[0],
                            composite_rgb(vn.xray_pixels, [], scene.drr_opacity))
        self.assertFalse(widget.set_stage(scene.primary, "no such stage"))
        defaults = widget.display_flags()
        flags = {"show_all_views": False, "camera_centred": True, "stages_all_views": True, "models": False,
                 "landmarks": True, "volume_rendering": True, "drr_overlay": False, "drr_opacity": 0.25}
        self.assertFalse(defaults["landmarks"])  # landmarks start hidden
        shown = [n for n in scene.landmarks_3d.values() if n.GetDisplayNode().GetVisibility()]
        self.assertEqual(shown, [])
        self.assertTrue(widget.set_display(**flags))
        viewer = widget.state()["viewer"]
        self.assertEqual({k: viewer[k] for k in flags}, flags)
        self.assertEqual((scene.show_all_views, scene.camera_centred, scene.stages_all_views, scene.models_visible,
                          scene.landmarks_visible, scene.drr_overlay), (False, True, True, False, True, False))
        self.assertAlmostEqual(scene.drr_opacity, 0.25)
        if scene.ct is not None:
            display = slicer.modules.volumerendering.logic().GetFirstVolumeRenderingDisplayNode(scene.ct)
            self.assertTrue(display is not None and display.GetVisibility())
        carm = slicer.util.arrayFromTransformMatrix(scene.views[scene.view_id].carm, toWorld=True)
        self.assertTrue(np.allclose(carm, np.eye(4), atol=1e-6))  # camera-centred: the selected camera stays
        widget.display_checks["camera_centred"].click()  # the checkbox drives the scene as well
        self.assertFalse(scene.camera_centred)
        self.assertTrue(widget.look_from_source())
        self.assertTrue(widget.overview())
        self.assertTrue(widget.set_display(**defaults))
        json.dumps(widget.state())
        self.assertTrue(widget.clear_run())
        self.assertIsNone(widget.state()["viewer"])
        self.assertFalse(any(w.enabled for w in widget.viewer_controls))
        self.assertTrue(widget.show_run())  # the selected run
        self.assertEqual(widget.state()["viewer"]["run"], rel)
        widget.clear_run()
        slicer.mrmlScene.Clear(0)

    def test_view_only(self, path, tmp):
        """Rule 3: a folder with runs but no project_manifest.json (a scratch copy of the run and its inputs,
        files linked) opens for viewing only and nothing is written into it; rule 1 on its run folder; Set up
        runs init; rule 4: an empty folder is set up."""
        import shutil

        run = XregRun.load(path)
        exp = os.path.join(os.path.realpath(tmp), "results_only")
        rel = os.path.relpath(str(run.run_dir), str(run.experiment_dir)).replace(os.sep, "/")
        for root, _dirs, files in os.walk(str(run.run_dir)):
            out = os.path.normpath(os.path.join(exp, *rel.split("/"), os.path.relpath(root, str(run.run_dir))))
            os.makedirs(out, exist_ok=True)
            for name in files:
                src = os.path.join(root, name)
                if name == "run_manifest.json":  # XregRun resolves the manifest's path: a real copy
                    shutil.copyfile(src, os.path.join(out, name))
                else:
                    os.symlink(src, os.path.join(out, name))
        for role, rec in run.data["inputs"].items():
            relative = rec.get("path") if isinstance(rec, dict) else None
            if relative and not os.path.isabs(relative):
                dst = os.path.join(exp, *relative.split("/"))
                if not os.path.lexists(dst):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    os.symlink(str(run.input_path(role)), dst)

        def listing():
            return sorted(os.path.relpath(os.path.join(r, n), exp) for r, ds, fs in os.walk(exp) for n in ds + fs)

        before = listing()
        widget = slicer.modules.fluororeg.widgetRepresentation().self()
        status = widget.set_project(exp)
        self.assertIsNotNone(status, widget.messages)
        state = widget.state()
        self.assertFalse(state["set_up"])
        self.assertEqual(state["checklist"]["experiment"]["summary"], "results only")
        self.assertIn(rel, [r["run_dir"] for r in status["runs"]])
        self.assertFalse(widget.set_up_button.isHidden())
        self.assertFalse(any(n.isHidden() for n in widget.set_up_notes.values()))
        for b in (widget.ct_load_button, widget.annotate_button, widget.add_files_button,
                  widget.add_to_experiment_button, widget.pose_train_button, widget.pose_use_button,
                  widget.pose_config_button, widget.config_open_button, widget.register_button):
            self.assertFalse(b.enabled, b.objectName)
            self.assertIn(SET_UP_REASON, b.toolTip, b.objectName)  # Slicer wraps tooltips in <p>
        self.assertFalse(widget.run_annotate())
        self.assertEqual(widget.messages["ct"]["text"], SET_UP_REASON)
        self.assertFalse(widget.use_pose_model("/models/m"))
        self.assertTrue(widget.select_run(rel))
        self.assertTrue(widget.show_run())
        self.assertEqual(widget.state()["viewer"]["run"], rel)
        self.assertTrue(widget.set_view(widget.run_viewer.scene.view_id))
        widget.clear_run()
        slicer.mrmlScene.Clear(0)
        self.assertIsNotNone(widget.set_project(os.path.join(exp, *rel.split("/"))))
        state = widget.state()
        self.assertEqual((widget.project, state["set_up"], state["viewer"]["run"]), (exp, False, rel))
        widget.clear_run()
        slicer.mrmlScene.Clear(0)
        self.assertEqual(listing(), before)
        self.assertIsNotNone(widget.set_up_experiment())
        self.assertTrue(widget.state()["set_up"])
        self.assertTrue(os.path.isfile(os.path.join(exp, "project_manifest.json")))
        self.assertTrue(all(n.isHidden() for n in widget.set_up_notes.values()))
        empty = os.path.join(tmp, "new_experiment")
        os.makedirs(empty)
        self.assertIsNotNone(widget.set_project(empty))
        self.assertTrue(widget.state()["set_up"])
        self.assertTrue(os.path.isfile(os.path.join(empty, "project_manifest.json")))
