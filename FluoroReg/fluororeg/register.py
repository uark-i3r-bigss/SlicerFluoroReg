"""``register``: optional pose prediction, then xreg's ``tools/run_registration.sh`` on one configuration.

Progress events carry ``phase`` (``predict``, ``register``, ``replay``) with ``done``/``total`` within it.
"""

import glob
import json
import os
import re
import shutil
import tempfile
import uuid

from . import pose
from .configs import config_case, config_problems, infer_app, load_config, require_current
from .project import rel
from .runner import new_log_path, run_tool
from .status import EXIT_ATTENTION, InputError, ToolError, emit

IMAGE_RE = re.compile(r'\[XREG:IMAGE\]\s+id=(\S+)\s+status=(\S+)(?:\s+reason="([^"]*)")?')
RUN_DIR_PREFIX = "RUN_DIR="
# run_registration.sh log lines (stderr) around the replays.
RUN_DIRECTORY_RE = re.compile(r"\[run_registration\] run directory: (.+)$")
REPLAY_DONE_RE = re.compile(r"\[run_registration\]\s+(wrote |replay of \S+ failed)")
POSES_PREFIX = "meta_data/initial_poses/"


def seed_targets(config):
    """{target key: case} of the targets initialised from ``meta_data/initial_poses/<case>``."""
    out = {}
    for target in config.get("targets") or []:
        init = target.get("init") or {}
        from_run = str(init.get("from_run") or "").rstrip("/")
        if init.get("method") == "run" and from_run.startswith(POSES_PREFIX) and "/" not in from_run[len(POSES_PREFIX):]:
            out[target.get("key", "?")] = from_run[len(POSES_PREFIX):]
    return out


def _pose_problem(problem, keys):
    """A problem that predicting the initial poses resolves (missing pose folder or pose files)."""
    return any(problem.startswith(f"target {k}: no initial pose") or problem.startswith(f"targets[{k}].init.from_run")
               for k in keys)


def predict_poses(project, case_id, ids, settings):
    """Predict without staging; copy ``best`` to a durable unique seed directory outside scratch."""
    cfg = pose.pose_config_path(project, case_id)
    if not os.path.isfile(cfg):
        raise InputError(f"{pose.pose_config_rel(case_id)} not found; train a pose model or use an existing one first")
    emit("progress", "register", phase="predict", done=0, total=len(ids), message="predicting initial poses")
    rc, metadata, log = pose.run_pose_script(
        project, case_id, settings,
        ["predict", "--config", cfg, "--ids", *ids, "--refine", "--stage", "best", "--no-stage"],
        "pose-predict", step="register", phase="predict")
    if rc not in (0, EXIT_ATTENTION):
        raise ToolError(f"pose.sh predict exited with status {rc}; see {log}", {"log": log})
    run_dir = metadata.get("run_dir")
    if not isinstance(run_dir, str) or not os.path.isabs(run_dir):
        raise ToolError(f"pose.sh predict returned no absolute immutable run_dir; see {log}", {"log": log})
    poses_dir = os.path.join(run_dir, "best")
    if not os.path.isdir(poses_dir):
        raise ToolError(f"pose.sh predict returned no best pose directory; see {log}", {"log": log})
    # Prediction runs are scratch. Registration inputs must survive removal of meta_data/tmp.
    durable = os.path.join(project, "meta_data", "initial_poses", "predictions", case_id,
                           f"{os.path.basename(os.path.normpath(run_dir))}_{uuid.uuid4().hex}")
    shutil.copytree(poses_dir, durable, symlinks=False)
    poses_dir = durable
    summary = pose.quality_summary(poses_dir)
    for message in summary.get("warnings", []):
        emit("warning", "pose", message=message)
    result = {"case": case_id, "ids": list(ids), "initial_poses": pose.display_path(project, poses_dir),
              "prediction_run": pose.display_path(project, run_dir), "quality": summary, "log": log}
    return result, bool(summary.get("flipped_side"))


def register(config_path, settings, replay=False, predict=False):
    """Run the registration; returns (result fields with ``run_dir`` relative to the project, attention)."""
    config_path = os.path.abspath(config_path)
    config = load_config(config_path)
    require_current(config, config_path)
    seeds = seed_targets(config) if predict else {}
    problems = [p for p in config_problems(config, config_path) if not _pose_problem(p, seeds)]
    if problems:
        raise InputError("the configuration cannot run: " + "; ".join(problems), {"problems": problems})
    if predict and not seeds:
        emit("warning", "register", message="no target is initialised from meta_data/initial_poses/<case>; "
                                            "nothing to predict")
    app = infer_app(config)
    xreg_dir = settings.require("xreg_dir")
    script = os.path.join(xreg_dir, "tools", "run_registration.sh")
    if not os.path.isfile(script):
        raise InputError(f"--xreg-dir {xreg_dir} has no tools/run_registration.sh")
    build_dir = settings.require("xreg_build_dir")
    project = os.path.normpath(config["experiment_dir"])
    ids = config["projections"]["ids"]
    cases = sorted(set(seeds.values()))
    if cases:
        pose.pose_script(settings)
    emit("start", "register", app=app, config=os.path.relpath(config_path, project), ids=ids,
         phases=(["predict"] if cases else []) + ["register"] + (["replay"] if replay else []))

    attention = False
    predicted = []
    for case_id in cases:
        result, flipped = predict_poses(project, case_id, ids, settings)
        predicted.append(result)
        attention = attention or flipped
        for target in config.get("targets") or []:
            if seeds.get(target.get("key")) == case_id:
                target["init"]["from_run"] = result["initial_poses"]
    if cases:
        problems = config_problems(config, config_path)
        if problems:
            raise InputError("the configuration cannot run after predicting: " + "; ".join(problems),
                             {"problems": problems})

    execution_config = config_path
    if predicted:
        # A run must never point at the operator config or a replaceable active pose directory.
        folder = os.path.join(project, "logs", config_case(config) or "project", "register")
        os.makedirs(folder, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="execution_", suffix=".json",
                                         dir=folder, delete=False) as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
            execution_config = handle.name

    argv = [script, "--app", app, "--config", execution_config, "--build-dir", build_dir]
    if not replay:
        argv.append("--no-replay")
    total = 1 if app in ("single-target-multiview", "multi-target-multiview") else len(ids)
    state = {"done": 0, "run_dir": None, "replays": 0, "replay_total": 0}
    emit("progress", "register", phase="register", done=0, total=total, message="registering")

    def on_stdout(line):
        m = IMAGE_RE.search(line)
        if m:
            state["done"] += 1
            message = m.group(2) + (f": {m.group(3)}" if m.group(3) else "")
            emit("progress", "register", phase="register", id=m.group(1), done=state["done"], total=total,
                 message=message)
        elif line.startswith(RUN_DIR_PREFIX):
            state["run_dir"] = line[len(RUN_DIR_PREFIX):].strip()

    def on_stderr(line):
        m = RUN_DIRECTORY_RE.search(line)
        if m and replay:
            run_dir = m.group(1).strip()
            records = glob.glob(os.path.join(run_dir, "views", "*", "view-*_optimizer_record.h5"))
            records += glob.glob(os.path.join(run_dir, "joint", "joint_optimizer_record.h5"))
            state["replay_total"] = len(records)
            if records:
                emit("progress", "register", phase="replay", done=0, total=len(records), message="replaying")
        elif REPLAY_DONE_RE.search(line) and state["replay_total"]:
            state["replays"] += 1
            emit("progress", "register", phase="replay", done=state["replays"], total=state["replay_total"])

    log_path = new_log_path(project, config_case(config) or "project", "register")
    rc = run_tool(argv, cwd=project, log_path=log_path, on_stdout_line=on_stdout, on_stderr_line=on_stderr)
    result = {"app": app, "config": os.path.relpath(config_path, project), "log": os.path.relpath(log_path, project)}
    if predicted:
        result["predicted"] = predicted
        result["execution_config"] = os.path.relpath(execution_config, project)
    if state["run_dir"]:
        result["run_dir"] = os.path.relpath(os.path.realpath(state["run_dir"]), os.path.realpath(project))
    if rc != 0:
        raise ToolError(f"run_registration.sh exited with status {rc}; see {result['log']}", result)
    if not state["run_dir"]:
        raise ToolError("run_registration.sh printed no RUN_DIR= line", result)
    return result, attention
