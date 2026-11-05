"""Learned initial poses (bigss-pose-estimation ``scripts/pose.sh``) and their warnings.

Warnings follow ``bigss-pose-estimation/docs/source_side_check.md``. A source-side mismatch
can exchange anatomical left and right while retaining a plausible-looking pelvis projection.
"""

import json
import os
import statistics

from .backends import backend_env, require_environment
from .project import abs_path, is_inside, read_json, rel
from .runner import new_log_path, run_tool
from .status import EXIT_ATTENTION, InputError, ToolError, emit, parse_status_line

REFINED_FLIP_GNCC = 0.5
NETWORK_FLIP_GNCC = 0.3
NO_TARGET_FLAG = "no_target_in_view"


def pose_config_rel(case_id):
    return f"configs/pose_estimation/{case_id}.json"


def pose_script(settings):
    script = os.path.join(settings.require("pose_dir"), "scripts", "pose.sh")
    if not os.path.isfile(script):
        raise InputError(f"--pose-dir {settings.get('pose_dir')} has no scripts/pose.sh")
    require_environment(settings, "pose")
    return script


def pose_dirs(project, case_id):
    """(poses_dir, models_dir) of a case, from its pose config's ``output`` when set."""
    poses = f"meta_data/initial_poses/{case_id}"
    models = f"meta_data/pose_models/{case_id}"
    cfg_path = abs_path(project, pose_config_rel(case_id))
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as handle:
                out = (json.load(handle).get("output") or {})
        except (OSError, ValueError, AttributeError):
            out = {}
        poses = out.get("poses_dir", poses)
        models = out.get("models_dir", models)

    def resolve(p):
        return p if os.path.isabs(p) else abs_path(project, p)

    return resolve(poses), resolve(models)


def run_pose_script(project, case_id, settings, args, log_step, step="pose", phase=None):
    """``pose.sh ARGS``; returns (exit status, last ``BIGSS_POSE`` result record, log path relative to the
    project). Pose progress lines become progress events of ``step`` (phase = ``phase`` or the pose stage)."""
    script = pose_script(settings)
    records = []

    def on_line(line):
        record = parse_status_line(line, "BIGSS_POSE")
        if record is None:
            return
        if record.get("event") == "result":
            records.append(record)
        elif record.get("event") == "progress":
            fields = {"phase": phase or record.get("stage"), "done": record.get("done", record.get("iteration")),
                      "total": record.get("total")}
            if record.get("id") is not None:
                fields["id"] = str(record["id"])
            emit("progress", step, **fields)

    env = backend_env(settings, "pose")
    log_path = new_log_path(project, case_id, log_step)
    rc = run_tool([script] + list(args), cwd=settings.get("pose_dir"), log_path=log_path, env=env,
                  on_stdout_line=on_line)
    return rc, (records[-1] if records else {}), rel(project, log_path)


def pose_config_path(project, case_id):
    return abs_path(project, pose_config_rel(case_id))


def ensure_config(project, case_id, settings):
    """Create the case's pose config with ``init-config`` when it is missing; returns True when created."""
    if os.path.isfile(pose_config_path(project, case_id)):
        return False
    emit("progress", "pose", phase="init-config", message=f"creating {pose_config_rel(case_id)}")
    rc, _, log = run_pose_script(project, case_id, settings,
                                 ["init-config", "--project", project, "--case", case_id], "pose-init-config")
    if rc != 0:
        raise ToolError(f"pose.sh init-config exited with status {rc}; see {log}", {"log": log})
    return True


def warn_quality(project, case_id):
    """Quality summary of the case's staged poses with its warnings emitted; returns the summary."""
    poses_dir, _ = pose_dirs(project, case_id)
    summary = quality_summary(poses_dir)
    for message in summary.get("warnings", []):
        emit("warning", "pose", message=message)
    return summary


def run_pose(project, case_id, command, settings, extra=None):
    """``pose.sh init-config|train|predict``; returns (exit status, result fields)."""
    pose_script(settings)
    created = False
    if command == "init-config":
        args = ["init-config", "--project", project, "--case", case_id]
    else:
        if command == "train":
            created = ensure_config(project, case_id, settings)
        elif not os.path.isfile(pose_config_path(project, case_id)):
            raise InputError(f"{pose_config_rel(case_id)} not found; run pose init-config first")
        args = [command, "--config", pose_config_path(project, case_id)]
    args += list(extra or [])
    rc, pose_result, log = run_pose_script(project, case_id, settings, args, f"pose-{command}")
    result = {"case": case_id, "command": command, "log": log}
    if rc not in (0, EXIT_ATTENTION):
        raise ToolError(f"pose.sh {command} exited with status {rc}; see {log}", result)
    if command == "init-config":
        result["config"] = pose_config_rel(case_id)
    elif command == "train":
        result["config"] = pose_config_rel(case_id)
        result["config_created"] = created
        result["activated"] = pose_result.get("activated")
        if rc == EXIT_ATTENTION or pose_result.get("activated") is False:
            emit("warning", "pose", message="the trained model was not activated (self-check failed); "
                                            "predict keeps using the previous active model, if any")
            rc = EXIT_ATTENTION
    elif command == "predict" and rc == 0:
        summary = warn_quality(project, case_id)
        result["initial_poses"] = rel(project, pose_dirs(project, case_id)[0])
        result["quality"] = summary
        if summary.get("flipped_side"):
            rc = EXIT_ATTENTION
    return rc, result


def use_model(project, case_id, model_dir, settings):
    """Make an existing model folder the case's active model (``pose.sh activate``; copied into the
    config's models_dir when it is elsewhere); returns the result fields."""
    pose_script(settings)
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if not os.path.isdir(model_dir):
        raise InputError(f"--model-dir {model_dir} is not a folder")
    created = ensure_config(project, case_id, settings)
    rc, pose_result, log = run_pose_script(
        project, case_id, settings,
        ["activate", "--config", pose_config_path(project, case_id), "--model-dir", model_dir], "pose-use-model")
    result = {"case": case_id, "command": "use-model", "config": pose_config_rel(case_id),
              "config_created": created, "log": log}
    if rc != 0:
        raise ToolError(f"pose.sh activate exited with status {rc}; see {log}", result)
    result["model_dir"] = display_path(project, pose_result.get("model_dir") or model_dir)
    return rc, result


def display_path(project, path):
    """``path`` relative to the project when inside it, else absolute."""
    real, root = os.path.realpath(path), os.path.realpath(project)
    return rel(root, real) if is_inside(real, root) else real


def model_info(project, case_id):
    """The case's active model: ``{active, name, self_check_capture, trained}`` from ``active.json`` and the
    model's ``model_card.json`` (all None without an active model)."""
    _, models_dir = pose_dirs(project, case_id)
    info = {"active": None, "name": None, "self_check_capture": None, "trained": None}
    try:
        with open(os.path.join(models_dir, "active.json"), "r", encoding="utf-8") as handle:
            model_dir = json.load(handle).get("model_dir")
    except (OSError, ValueError, AttributeError):
        return info
    if not isinstance(model_dir, str) or not model_dir:
        return info
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(models_dir, model_dir)
    info["active"] = display_path(project, model_dir)
    info["name"] = os.path.basename(os.path.normpath(model_dir))
    try:
        with open(os.path.join(model_dir, "model_card.json"), "r", encoding="utf-8") as handle:
            card = json.load(handle)
        info["self_check_capture"] = (card.get("self_check") or {}).get("capture_rate")
        info["trained"] = card.get("created")
    except (OSError, ValueError, AttributeError):
        pass
    return info


def _stage_of(poses_dir, views):
    prov_path = os.path.join(poses_dir, "provenance.json")
    if os.path.isfile(prov_path):
        try:
            stage = read_json(prov_path, "pose provenance").get("stage")
        except (InputError, AttributeError):
            stage = None
        if stage in ("network", "refined", "best"):
            return stage
    refined = any("refine_change" in v or v.get("source_stage") == "refined" for v in views.values())
    return "refined" if refined else "network"


def quality_summary(poses_dir):
    """Tiers, median gNCC and warnings of a staged pose directory's ``quality.json``."""
    path = os.path.join(poses_dir, "quality.json")
    if not os.path.isfile(path):
        return {"present": False, "warnings": []}
    quality = read_json(path, "pose quality")
    views = quality.get("views") or {}
    stage = _stage_of(poses_dir, views)
    tiers = {}
    for vid, view in sorted(views.items()):
        tiers.setdefault(view.get("tier", "unknown"), []).append(vid)
    scored = [v.get("gncc") for v in views.values()
              if NO_TARGET_FLAG not in (v.get("flags") or []) and isinstance(v.get("gncc"), (int, float))]
    median = statistics.median(scored) if scored else None
    warnings = []
    for tier in ("suspect", "plausible"):
        if tiers.get(tier):
            warnings.append(f"{len(tiers[tier])} {tier} view(s): {', '.join(tiers[tier])}")
    flipped = False
    if median is not None:
        if stage == "network":
            if median < NETWORK_FLIP_GNCC:
                flipped = True
                warnings.append(f"possible flipped source side: median network gNCC {median:.2f} < "
                                f"{NETWORK_FLIP_GNCC} over {len(scored)} view(s); run predict --refine for a "
                                "reliable check (bigss-pose-estimation docs/source_side_check.md)")
        elif median < REFINED_FLIP_GNCC:
            flipped = True
            warnings.append(f"possible flipped source side: median {stage} gNCC {median:.2f} < {REFINED_FLIP_GNCC} "
                            f"over {len(scored)} view(s) (bigss-pose-estimation docs/source_side_check.md)")
    return {"present": True, "stage": stage, "n_views": len(views), "tiers": tiers,
            "median_gncc": median, "n_scored": len(scored), "flipped_side": flipped, "warnings": warnings}
