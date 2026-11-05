"""Learned starting poses for xreg 2D/3D registration.

Usual order: init-config and train once per CT, then predict whenever X-rays are added. Every
command except check reads the case's settings file, PROJECT/configs/pose_estimation/<case>.json.
Run '<command> --help' for its options.

Exit status: 0 success, 2 invalid settings or inputs, 3 trained but not activated (train
--require-activation), 1 any other failure. Progress lines on stdout: BIGSS_POSE {json}.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from . import status
from .config import ConfigError, load_config


def _init_config(args) -> int:
    project = Path(args.project).resolve()
    if not (project / "project_manifest.json").is_file():
        raise ConfigError(f"{project} is not a project (no project_manifest.json)")
    out = project / "configs" / "pose_estimation" / f"{args.case}.json"
    if out.exists() and not args.force:
        raise ConfigError(f"{out} exists (use --force to overwrite)")
    cfg = {
        "schema_version": 1,
        "case_id": args.case,
        "volume": {"ct": f"meta_data/volumes/{args.case}_ct.nii.gz",
                   "seg": f"meta_data/volumes/{args.case}_seg.nii.gz"},
        "projections": {"dir": "meta_data/XrayProjData", "pattern": "{id}.h5"},
        "detector": {"crop_px": 100, "intensity": "raw"},
        "prior": {"center_label": args.center_label},
        "model": {"method": "finetune"},
        "predict": {"stage": "network"},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    load_config(out)  # validate what we wrote
    status.emit("result", stage="init-config", config=str(out))
    return 0


def _check(args) -> int:
    import torch

    from .weights import FOUNDATION, foundation_path

    report = {"cuda": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "torch": torch.__version__}
    try:
        report["foundation_weights"] = str(foundation_path(verify=True))
    except Exception as e:  # reported, not fatal for predict-only use
        report["foundation_weights_error"] = str(e)
    report["foundation_sha256"] = FOUNDATION["sha256"]
    if args.config:
        from .xreg_io import read_projection

        cfg = load_config(args.config)
        for p in (cfg.ct, cfg.seg):
            if not p.exists():
                raise ConfigError(f"missing input {p}")
        ids = cfg.projection_ids()
        missing = [i for i in ids if not cfg.projection_path(i).is_file()]
        if missing:
            raise ConfigError(f"missing projections {missing}")
        cam = read_projection(cfg.projection_path(cfg.reference_id or ids[0]), with_image=False).camera
        cam.require_centered()
        report.update({"config": str(cfg.path), "case_id": cfg.case_id, "n_projections": len(ids),
                       "sdd_mm": cam.sdd, "detector": [cam.rows, cam.cols, cam.col_spacing],
                       "models_dir": str(cfg.models_dir), "poses_dir": str(cfg.poses_dir),
                       "active_model": _has_active_model(cfg.models_dir)})
    status.emit("result", stage="check", **report)
    return 0 if report["cuda"] else 1


def _train(args) -> int:
    from .train import train

    res = train(load_config(args.config), force_activate=args.force_activate)
    return 0 if (res["activated"] or not args.require_activation) else 3


def _has_active_model(models_dir: Path) -> bool:
    from .train import active_model_dir

    try:
        active_model_dir(models_dir)
    except FileNotFoundError:
        return False
    return True


def _model_folder(d: Path) -> Path:
    """``d`` itself when it holds ``model.pth``; for a models folder its active model, else its only
    model subfolder. An active.json copied with its project may still name the model by its old
    absolute path; the same-named model inside ``d`` then wins."""
    from .train import active_model_dir

    if (d / "model.pth").is_file():
        return d
    if (d / "active.json").is_file():
        named = Path(json.loads((d / "active.json").read_text())["model_dir"])
        if named.is_absolute() and (d / named.name / "model.pth").is_file():
            return d / named.name
        try:
            return active_model_dir(d).resolve()
        except FileNotFoundError as e:
            raise ConfigError(str(e)) from e
    found = sorted(p for p in d.iterdir() if (p / "model.pth").is_file()) if d.is_dir() else []
    if len(found) == 1:
        return found[0]
    if not found:
        raise ConfigError(f"{d} has no model.pth and no model subfolder")
    raise ConfigError(f"{d} holds {len(found)} models; pass one of: " + ", ".join(str(p) for p in found))


def _activate(args) -> int:
    import shutil

    from .train import activate

    cfg = load_config(args.config)
    src = _model_folder(Path(args.model_dir).resolve())
    status.emit("start", stage="activate", model_dir=str(src), models_dir=str(cfg.models_dir))
    for name in ("model.pth", "model_card.json"):
        if not (src / name).is_file():
            raise ConfigError(f"{src} has no {name}")
    models_dir = cfg.models_dir.resolve()
    dst = src
    if not src.is_relative_to(models_dir):
        dst = models_dir / src.name
        if dst.exists():
            raise ConfigError(f"{dst} exists and is a different folder than {src}")
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    activate(models_dir, dst)
    status.emit("result", stage="activate", model_dir=str(dst), activated=True)
    return 0


def _predict(args) -> int:
    from .predict import predict

    predict(load_config(args.config), model_dir=args.model_dir, ids=args.ids, refine=args.refine,
            stage=args.stage, no_stage=args.no_stage)
    return 0


def _inspect(args) -> int:
    from .bridge import Bridge, xvr_parameters
    from .xreg_io import read_itk_xform, read_projection

    cfg = load_config(args.config)
    cam = read_projection(cfg.projection_path(args.id), with_image=False).camera
    P = Bridge.from_ct(cfg.ct, "AP").xreg_to_diffdrr(read_itk_xform(args.xform), cam.extrinsic)
    status.emit("result", stage="inspect-pose", id=args.id, xform=args.xform, **xvr_parameters(P))
    return 0


def _score(args) -> int:
    from .score import parse_targets, score

    score(load_config(args.config), args.run, args.out, targets=parse_targets(args.targets))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="scripts/pose.sh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    config_help = "settings file of the case (PROJECT/configs/pose_estimation/<case>.json)"

    p = sub.add_parser("init-config", help="write the default settings file of a case",
                       description="Write PROJECT/configs/pose_estimation/<case>.json with default settings.")
    p.add_argument("--project", required=True, help="project folder (holds project_manifest.json)")
    p.add_argument("--case", required=True, help="case id; its CT is meta_data/volumes/<case>_ct.nii.gz")
    p.add_argument("--center-label", type=int, default=1,
                   help="segmentation label the poses aim at (default: 1, the pelvis)")
    p.add_argument("--force", action="store_true", help="overwrite an existing settings file")
    p.set_defaults(func=_init_config)

    p = sub.add_parser("check", help="check the GPU, the weights and optionally a settings file",
                       description="Check the GPU and the foundation weights; with --config, also the "
                                   "settings file and its inputs.")
    p.add_argument("--config", help=config_help)
    p.set_defaults(func=_check)

    p = sub.add_parser("train", help="train a model for the case's CT",
                       description="Train a model on simulated X-rays of the case's CT and make it the "
                                   "active model if it passes its self-check.")
    p.add_argument("--config", required=True, help=config_help)
    p.add_argument("--force-activate", action="store_true",
                   help="make the model active even if the self-check fails")
    p.add_argument("--require-activation", action="store_true",
                   help="exit 3 when the model was not made active")
    p.set_defaults(func=_train)

    p = sub.add_parser("activate", help="use an existing trained model",
                       description="Make an existing trained model the case's active model (it is copied "
                                   "into the case's models folder when it is elsewhere).")
    p.add_argument("--config", required=True, help=config_help)
    p.add_argument("--model-dir", required=True,
                   help="a trained model folder, or a models folder (its active model, else its only model)")
    p.set_defaults(func=_activate)

    p = sub.add_parser("predict", help="predict the pose of every X-ray",
                       description="Predict the pose of every X-ray with the active model and save the poses "
                                   "as meta_data/initial_poses/<case>/, where xreg reads them.")
    p.add_argument("--config", required=True, help=config_help)
    p.add_argument("--model-dir", help="use this trained model instead of the active one")
    p.add_argument("--ids", nargs="+", help="X-ray ids to predict (default: those listed in the settings file, else all)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--refine", dest="refine", action="store_true", default=None,
                   help="refine each pose by matching simulated and real X-rays (slower)")
    g.add_argument("--no-refine", dest="refine", action="store_false", help="do not refine")
    p.add_argument("--stage", choices=["network", "refined", "best"],
                   help="which poses to save: the network's, the refined ones, or best (refined unless "
                        "refinement made the match worse)")
    p.add_argument("--no-stage", action="store_true",
                   help="only write the run record; leave initial_poses/ unchanged")
    p.set_defaults(func=_predict)

    p = sub.add_parser("inspect-pose", help="print an xreg pose as xvr parameters",
                       description="Print the xvr pose parameters of an existing xreg pose file, e.g. to "
                                   "choose the prior ranges in the settings file.")
    p.add_argument("--config", required=True, help=config_help)
    p.add_argument("--id", required=True, help="X-ray id whose camera the pose refers to")
    p.add_argument("--xform", required=True, help="xreg pose file (.h5)")
    p.set_defaults(func=_inspect)

    p = sub.add_parser("score", help="rate the final poses of an xreg run",
                       description="Rate every final pose of an xreg run by comparing a simulated X-ray "
                                   "with the real one around each target.")
    p.add_argument("--config", required=True, help=config_help)
    p.add_argument("--run", required=True, help="xreg run folder (holds run_manifest.json)")
    p.add_argument("--out", required=True, help="JSON file to write")
    p.add_argument("--targets", nargs="+", metavar="KEY=LABEL",
                   help="segmentation label of each target (default: from the run, then the settings file)")
    p.set_defaults(func=_score)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as e:
        status.emit("error", stage=args.command, kind="config", message=str(e))
        return 2
    except Exception as e:
        status.emit("error", stage=args.command, kind=type(e).__name__, message=str(e))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
