"""Command line: ``python3 -m fluororeg <command> ...``.

Exit status: 0 ok; 2 invalid input, configuration or missing tool path; 3 finished but needs
attention; 1 any other failure. Status lines are ``FLUOROREG {json}`` on stdout.
"""

import argparse
import os
import sys
import traceback

from . import assess as assess_mod
from . import backends, configs, ct, pose, project, register, report, xrays
from .settings import KEYS, Settings, flag_of
from .status import EXIT_ATTENTION, EXIT_FAILURE, EXIT_OK, InputError, PipelineError, emit

POSE_COMMANDS = ("init-config", "train", "predict", "use-model")
TOOL_HELP = {
    "xreg_dir": "xreg source checkout (tools/, configs/)",
    "xreg_build_dir": "xreg build folder (default: <xreg-dir>/build)",
    "annotation_3d_dir": "3D annotation source override (default: bundled snapshot)",
    "conda": "conda executable (default: conda on PATH)",
    "annotation_3d_env": "explicit named conda environment (default: versioned external prefix)",
    "pose_dir": "pose source override (default: bundled snapshot)",
    "pose_env_dir": "pose virtual environment override (default: versioned external runtime)",
    "annotation_3d_env_prefix": "annotation conda prefix override (mutually exclusive with named env)",
    "gpu": "GPU index exported as CUDA_VISIBLE_DEVICES for 3D annotation and pose (default: 0)",
    "xray_template": "X-ray DICOM from the same C-arm; image files (PNG/JPG/TIFF/BMP) are wrapped into a copy "
                     "of it (import-xrays)",
}


def _tool_parent():
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("tool paths (flag > --settings FILE > ~/.config/fluororeg/settings.json)")
    group.add_argument("--settings", metavar="FILE", help="JSON settings file with the keys " + ", ".join(KEYS))
    for key in KEYS:
        group.add_argument(flag_of(key), dest=key, metavar=key.upper(), help=TOOL_HELP[key])
    return parent


def build_parser():
    tools = _tool_parent()
    ap = argparse.ArgumentParser(prog="python3 -m fluororeg", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, help_text):
        return sub.add_parser(name, help=help_text, parents=[tools])

    p = add("init", "create the project layout and project_manifest.json (idempotent)")
    p.add_argument("--project", required=True)

    p = add("status", "summary of cases, X-rays, poses, xreg configs and runs")
    p.add_argument("--project", required=True)


    p = sub.add_parser("tools", help="inspect or explicitly set up backend environments; no project required")
    tsub = p.add_subparsers(dest="tools_command", required=True, metavar="action")
    for action in ("status", "setup"):
        t = tsub.add_parser(action, parents=[tools])
        t.add_argument("--component", choices=("all", "pose", "annotation"), default="all")
    p = add("import-ct", "copy a NIfTI file or DICOM folder into the project")
    p.add_argument("--project", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--case", help="default: the file stem or folder name")

    p = add("annotate-3d", "run the 3D landmark annotation on a case, then stage it")
    p.add_argument("--project", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--device", default="gpu", help="gpu, gpu:N or cpu (default: gpu, i.e. the --gpu device)")
    p.add_argument("--roi-subset", nargs="+")
    p.add_argument("--ap-flip", action="store_true")
    p.add_argument("--no-stage", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="let the staging replace staged files")

    p = add("stage", "copy a case's 3D annotation output into the project layout")
    p.add_argument("--project", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--overwrite", action="store_true")

    p = add("import-xrays", "add X-ray DICOMs or images as the next ids and convert them to projection H5")
    p.add_argument("--project", required=True)
    p.add_argument("--input", required=True, nargs="+", metavar="PATH",
                   help="DICOM or image files, or folders of them (natural file-name order)")
    p.add_argument("--no-convert", action="store_true", help="only copy; convert-xrays later")

    p = add("convert-xrays", "convert X-ray DICOMs to xreg projection H5 files")
    p.add_argument("--project", required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--overwrite", action="store_true")

    p = add("pose", "bigss-pose-estimation: init-config, train (creates the config when missing), predict, "
                    "use-model (extra pose.sh args after --)")
    p.add_argument("pose_command", choices=POSE_COMMANDS)
    p.add_argument("--project", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--model-dir", help="use-model: trained model folder (model.pth, model_card.json), or a models "
                                       "folder holding its active model or a single model")

    p = sub.add_parser("config", help="xreg configurations: create, set, show")
    csub = p.add_subparsers(dest="config_command", required=True, metavar="action")
    c = csub.add_parser("create", help="new configs/xreg/<name>.json from the bundled template")
    c.add_argument("--project", required=True)
    c.add_argument("--case", required=True)
    c.add_argument("--app", required=True, choices=configs.APPS)
    c.add_argument("--femur", choices=("left", "right"))
    c.add_argument("--name")
    c.add_argument("--ids", nargs="+")
    c = csub.add_parser("set", help="change experiment_dir, volume.ct/seg/landmarks_3d or projections.ids in place")
    c.add_argument("--config", required=True)
    c.add_argument("--experiment-dir", dest="experiment_dir",
                   help="existing absolute folder; for configs/xreg/<name>.json the experiment folder holding it")
    c.add_argument("--ct")
    c.add_argument("--seg")
    c.add_argument("--landmarks-3d", dest="landmarks_3d")
    c.add_argument("--ids", nargs="+")
    c = csub.add_parser("show", help="inferred app, inputs, ids, targets and problems")
    c.add_argument("--config", required=True)

    p = add("register", "run xreg's tools/run_registration.sh, then assess")
    p.add_argument("--config", required=True)
    p.add_argument("--predict-poses", action="store_true",
                   help="first predict the initial poses of the config's views with the case's pose model")
    p.add_argument("--replay", action="store_true", help="also render the optimizer replay videos (off by default)")
    p.add_argument("--no-assess", action="store_true")
    p.add_argument("--no-gncc", action="store_true", help="assess from the final objective only")

    p = add("assess", "write R/assessment.json for a run")
    p.add_argument("--run", required=True)
    p.add_argument("--no-gncc", action="store_true")
    return ap


def _settings(args):
    return Settings.resolve({k: getattr(args, k, None) for k in KEYS}, getattr(args, "settings", None))


def _project(args):
    return os.path.abspath(args.project)


def _cmd_init(args):
    created = project.init_project(_project(args))
    emit("result", "init", project=_project(args), created=created)
    return EXIT_OK


def _cmd_status(args):
    emit("result", "status", **report.project_status(_project(args)))
    return EXIT_OK


def _cmd_tools(args):
    settings = _settings(args)
    action = backends.setup_tools if args.tools_command == "setup" else backends.tools_status
    emit("result", "tools", action=args.tools_command, components=action(settings, args.component))
    return EXIT_OK


def _cmd_import_ct(args):
    emit("result", "import-ct", **ct.import_ct(_project(args), args.input, args.case))
    return EXIT_OK


def _cmd_annotate_3d(args):
    settings = _settings(args)
    result = ct.annotate_3d(_project(args), args.case, settings, device=args.device, roi_subset=args.roi_subset,
                            ap_flip=args.ap_flip)
    emit("result", "annotate-3d", **result)
    if not args.no_stage:
        emit("start", "stage", case=args.case)
        emit("result", "stage", **ct.stage(_project(args), args.case, settings, overwrite=args.overwrite))
    return EXIT_OK


def _cmd_stage(args):
    emit("result", "stage", **ct.stage(_project(args), args.case, _settings(args), overwrite=args.overwrite))
    return EXIT_OK


def _cmd_import_xrays(args):
    project.load_manifest(_project(args))
    result = xrays.import_xrays(_project(args), args.input, _settings(args), convert=not args.no_convert)
    emit("result", "import-xrays", **result)
    return EXIT_FAILURE if result["failed"] else EXIT_OK


def _cmd_convert_xrays(args):
    project.load_manifest(_project(args))
    result = xrays.convert_xrays(_project(args), _settings(args), ids=args.ids, overwrite=args.overwrite)
    emit("result", "convert-xrays", **result)
    return EXIT_OK


def _cmd_pose(args):
    project.load_manifest(_project(args))
    if args.pose_command == "use-model":
        if not args.model_dir:
            raise InputError("pose use-model needs --model-dir DIR")
        rc, result = pose.use_model(_project(args), args.case, args.model_dir, _settings(args))
    else:
        rc, result = pose.run_pose(_project(args), args.case, args.pose_command, _settings(args), extra=args.extra)
    emit("result", "pose", **result)
    return rc


def _cmd_config(args):
    if args.config_command == "create":
        path, config, warnings = configs.create_config(_project(args), args.case, args.app, femur=args.femur,
                                                       name=args.name, ids=args.ids)
        for message in warnings:
            emit("warning", "config", message=message)
        emit("result", "config", action="create", config=project.rel(_project(args), path),
             app=args.app, ids=config["projections"]["ids"], output_dir=config["output"]["dir"])
    elif args.config_command == "set":
        changes = configs.set_config(args.config, ct=args.ct, seg=args.seg, landmarks_3d=args.landmarks_3d,
                                     ids=args.ids, experiment_dir=args.experiment_dir)
        emit("result", "config", action="set", config=os.path.abspath(args.config), changes=changes)
    else:
        emit("result", "config", action="show", **configs.show_config(args.config))
    return EXIT_OK


def _assess(run_dir, settings, no_gncc):
    emit("start", "assess", run=run_dir)
    result, attention = assess_mod.assess(run_dir, settings, no_gncc=no_gncc)
    emit("result", "assess", **result)
    return EXIT_ATTENTION if attention else EXIT_OK


def _cmd_register(args):
    settings = _settings(args)
    config = configs.load_config(args.config)
    configs.require_current(config, args.config)
    if not args.no_assess:
        # Fail before a long run, not after it, when the assessment will need the pose repo.
        exp = config.get("experiment_dir") or ""
        if assess_mod.gncc_needed(exp, configs.config_case(config), args.no_gncc):
            pose.pose_script(settings)
    result, attention = register.register(args.config, settings, replay=args.replay,
                                          predict=args.predict_poses)
    emit("result", "register", **result)
    if args.no_assess:
        return EXIT_ATTENTION if attention else EXIT_OK
    rc = _assess(os.path.join(config["experiment_dir"], result["run_dir"]), settings, args.no_gncc)
    return EXIT_ATTENTION if attention else rc


def _cmd_assess(args):
    return _assess(os.path.abspath(args.run), _settings(args), args.no_gncc)


COMMANDS = {
    "init": _cmd_init, "status": _cmd_status, "import-ct": _cmd_import_ct, "annotate-3d": _cmd_annotate_3d,
    "stage": _cmd_stage, "import-xrays": _cmd_import_xrays, "convert-xrays": _cmd_convert_xrays,
    "pose": _cmd_pose, "config": _cmd_config, "register": _cmd_register, "assess": _cmd_assess,
    "tools": _cmd_tools,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if argv and argv[0] == "pose" and "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    args.extra = extra
    step = args.command
    # status and config print only a result; register and assess emit their own start with details.
    if args.command not in ("status", "config", "register", "assess"):
        emit("start", step, command=f"pose {args.pose_command}" if step == "pose" else step)
    try:
        return COMMANDS[args.command](args)
    except PipelineError as exc:
        fields = {"kind": exc.kind, "message": str(exc)}
        if exc.details:
            fields["details"] = exc.details
        emit("error", step, **fields)
        return exc.exit_code
    except KeyboardInterrupt:
        emit("error", step, kind="interrupted", message="stopped")
        return EXIT_FAILURE
    except Exception as exc:
        emit("error", step, kind=type(exc).__name__, message=str(exc))
        traceback.print_exc()
        return EXIT_FAILURE
