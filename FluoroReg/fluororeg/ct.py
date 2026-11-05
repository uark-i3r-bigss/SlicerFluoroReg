"""CT steps: ``import-ct``, ``annotate-3d`` and ``stage``.

``stage`` is a Qt-free port of xreg_gui's ``ct_workspace_builder.stage_case_into_project`` that
writes no xreg configuration, ``.gui.json`` or create_drr configuration.
"""

import glob
import os
import re
import shutil

from . import landmarks
from .backends import annotation_conda_args, backend_env, require_environment
from .project import (abs_path, annotation_output_rel, copy_file, nifti_suffix, original_input_dir_rel, rel,
                      remove_project_path, require_case, sanitize_case_id, staged_paths, update_case)
from .runner import new_log_path, run_tool
from .status import InputError, ToolError, emit

# ``[STAGE] <id>`` lines of process_medical_scans.py (after its timestamp prefix), in the order they run.
ANNOTATION_STAGES = ("analyze_input", "totalseg", "combine_segmentations", "generate_stl", "generate_landmarks",
                     "crop_roi", "organize_output")
STAGE_RE = re.compile(r"\[STAGE\]\s+(\S+)")


def _on_annotation_line(line):
    match = STAGE_RE.search(line)
    if match and match.group(1) in ANNOTATION_STAGES:
        stage_id = match.group(1)
        emit("progress", "annotate-3d", done=ANNOTATION_STAGES.index(stage_id) + 1,
             total=len(ANNOTATION_STAGES), message=stage_id)


def ct_input(input_path, case_id=None):
    """(absolute input path, kind "nifti"/"dicom", NIfTI suffix or None, case id); the case id defaults to
    the file stem or folder name."""
    input_path = os.path.abspath(os.path.expanduser(input_path))
    if os.path.isfile(input_path):
        suffix = nifti_suffix(os.path.basename(input_path))
        if suffix is None:
            raise InputError(f"--input {input_path}: expected a .nii/.nii.gz file or a DICOM folder")
        kind = "nifti"
        default_case = os.path.basename(input_path)[:-len(suffix)]
    elif os.path.isdir(input_path):
        suffix = None
        kind = "dicom"
        default_case = os.path.basename(input_path.rstrip(os.sep))
    else:
        raise InputError(f"--input {input_path} does not exist")
    return input_path, kind, suffix, sanitize_case_id(case_id or default_case)


def import_ct(project, input_path, case_id=None):
    input_path, kind, suffix, case_id = ct_input(input_path, case_id)
    require_layout = os.path.join(project, "meta_data", "tmp")
    if not os.path.isdir(require_layout):
        raise InputError(f"{project}: missing meta_data/tmp; run init first")

    target_dir = abs_path(project, original_input_dir_rel(case_id))
    if os.path.lexists(target_dir):
        remove_project_path(project, target_dir)
    os.makedirs(target_dir)
    emit("progress", "import-ct", message=f"copying {kind} input into {rel(project, target_dir)}")
    if kind == "nifti":
        # Named after the case so the annotation CLI derives the same case id from it.
        dst = os.path.join(target_dir, case_id + suffix)
        copy_file(input_path, dst)
        n_files = 1
    else:
        os.rmdir(target_dir)
        shutil.copytree(input_path, target_dir, symlinks=False, copy_function=shutil.copyfile)
        dst = target_dir
        n_files = sum(len(files) for _, _, files in os.walk(target_dir))
    entry = update_case(project, case_id, input=rel(project, dst), input_kind=kind,
                        source_name=os.path.basename(input_path.rstrip(os.sep)), status="imported")
    return {"case": case_id, "input": entry["input"], "input_kind": kind, "files": n_files}


def annotate_3d(project, case_id, settings, device="gpu", roi_subset=None, ap_flip=False):
    """Run the 3D annotation CLI on the case's imported input; returns its result fields."""
    _, entry = require_case(project, case_id)
    input_rel = entry.get("input")
    if not input_rel or not os.path.exists(abs_path(project, input_rel)):
        raise InputError(f"case '{case_id}' has no imported input; run import-ct first")
    repo = settings.require("annotation_3d_dir")
    script = os.path.join(repo, "process_medical_scans.py")
    if not os.path.isfile(script):
        raise InputError(f"--annotation-3d-dir {repo} has no process_medical_scans.py")
    spec = require_environment(settings, "annotation")
    env = backend_env(settings, "annotation", spec)
    conda = settings.require_executable("conda", env.get("PATH"))
    out_rel = annotation_output_rel(case_id)
    out_dir = abs_path(project, out_rel)
    os.makedirs(out_dir, exist_ok=True)
    argv = [conda, "run", *annotation_conda_args(settings, spec), "--no-capture-output", "python",
            "process_medical_scans.py", abs_path(project, input_rel), out_dir, "--no-confirm", "--device", device]
    if roi_subset:
        argv += ["--roi-subset"] + list(roi_subset)
    if ap_flip:
        argv.append("--ap-flip")
    log_path = new_log_path(project, case_id, "annotate-3d")
    rc = run_tool(argv, cwd=repo, log_path=log_path, env=env, on_stdout_line=_on_annotation_line)
    if rc != 0:
        update_case(project, case_id, status="annotation_failed", annotation_output=out_rel, ap_flip=bool(ap_flip))
        raise ToolError(f"3D annotation exited with status {rc}; see {rel(project, log_path)}")
    update_case(project, case_id, status="annotated", annotation_output=out_rel, ap_flip=bool(ap_flip))
    return {"case": case_id, "annotation_output": out_rel, "log": rel(project, log_path)}


def _resolve_annotation_case_output(annotation_output_dir, case_id):
    preferred = os.path.join(annotation_output_dir, case_id)
    if os.path.isdir(os.path.join(preferred, "meta_data")) and os.path.isdir(os.path.join(preferred, "landmarks")):
        return preferred
    return annotation_output_dir


def stage(project, case_id, settings, overwrite=False):
    """Copy the case's 3D annotation output into the flat project layout."""
    _, entry = require_case(project, case_id)
    out_rel = entry.get("annotation_output") or annotation_output_rel(case_id)
    annotation_dir = abs_path(project, out_rel)
    if not os.path.isdir(annotation_dir):
        raise InputError(f"no 3D annotation output at {out_rel}; run annotate-3d first")
    source = _resolve_annotation_case_output(annotation_dir, case_id)
    source_meta = os.path.join(source, "meta_data")
    source_landmarks = os.path.join(source, "landmarks")
    source_models = os.path.join(source, "models")

    ct_candidates = sorted(glob.glob(os.path.join(source_meta, "*_CT.nii.gz")))
    seg_candidates = sorted(glob.glob(os.path.join(source_meta, "*_CT_seg.nii.gz")))
    if not ct_candidates or not seg_candidates:
        raise InputError(f"3D annotation output {rel(project, source)} is missing CT or combined segmentation files")
    native_landmarks = os.path.join(source_landmarks, f"{case_id}_landmarks.fcsv")
    if not os.path.isfile(native_landmarks):
        raise InputError(f"3D annotation output is missing the comprehensive landmarks file "
                         f"{rel(project, native_landmarks)}")
    native_points = landmarks.load_fcsv_points(native_landmarks)
    if not native_points:
        raise InputError(f"{rel(project, native_landmarks)} contains no 3D landmarks")

    paths = staged_paths(case_id)
    models = sorted(glob.glob(os.path.join(source_models, "*.stl"))) if os.path.isdir(source_models) else []
    model_rels = [f"meta_data/models/{case_id}_{os.path.basename(m)}" for m in models]
    existing = [p for p in list(paths.values()) + model_rels if os.path.exists(abs_path(project, p))]
    if existing and not overwrite:
        raise InputError(f"staged files exist ({', '.join(existing)}); use --overwrite to replace them")

    xreg_dir = settings.get("xreg_dir")
    if xreg_dir:
        canonical = landmarks.load_canonical_config(settings.require("xreg_dir"))
    elif os.path.isfile(landmarks.registry_path(project)):
        canonical = None
    else:
        raise InputError("missing --xreg-dir (or 'xreg_dir' in the settings file): the project has no "
                         f"{landmarks.REGISTRY_REL_PATH} and it is created from xreg's bigss_canonical sets")

    copy_file(ct_candidates[0], abs_path(project, paths["ct"]))
    copy_file(seg_candidates[0], abs_path(project, paths["seg"]))
    detected_sets = landmarks.ensure_canonical_sets(project, native_points.keys(), canonical)
    canonical_map = {}
    for set_entry in detected_sets:
        for name in landmarks.set_landmark_names(set_entry):
            canonical_map[name] = landmarks.canonical_landmark_name(set_entry["set_id"], name)
    landmarks.canonicalize_fcsv_labels_with_map(native_landmarks, abs_path(project, paths["landmarks_3d"]),
                                                canonical_map)
    for model, model_rel in zip(models, model_rels):
        copy_file(model, abs_path(project, model_rel))
    for d in ("meta_data/landmarks/landmarks_2d", "meta_data/tmp", "configs/xreg"):
        os.makedirs(abs_path(project, d), exist_ok=True)

    set_ids = [s.get("set_id") for s in detected_sets]
    staged = dict(paths, models=model_rels, landmark_sets=set_ids)
    update_case(project, case_id, status="staged", annotation_output=rel(project, annotation_dir), staged=staged)
    return {"case": case_id, "staged": staged, "landmark_registry": landmarks.REGISTRY_REL_PATH,
            "source": rel(project, source)}
