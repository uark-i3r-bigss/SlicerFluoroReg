"""Predict the CT pose for each projection and stage it as an xreg pose directory.

Input images are the projection H5s xreg registers (native grid, camera included), so the pose is
expressed in exactly the camera xreg uses. Pixels are preprocessed as ``xvr register`` does for a
radiograph: centre crop, min-max, and (for raw transmission images) log-linearisation.

Each run writes a complete record under ``<runs_dir>/<YYYYMMDD_HHMMSS>_predict/``:
``network/`` (and ``refined/`` when refinement runs) pose directories, ``run.json`` and a
per-stage ``provenance.json``. The chosen stage is then staged, atomically, as the ACTIVE pose
directory ``<poses_dir>`` (xreg: ``init = {"method": "run", "from_run": <poses_dir>,
"from_target": "ct"}``). The previous active set stays in its own run directory.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from . import status
from .bridge import Bridge, xvr_parameters
from .config import PoseConfig
from .train import active_model_dir
from .xreg_io import POSE_FILE_PATTERN, pose_file_name, read_projection, sha256_file, write_itk_xform

POSE_KEY = "ct"
POSE_RECORD = {"key": POSE_KEY,
               "pose": "CT_from_CAMWORLD of the whole CT for each projection H5's camera world "
                       "(A_from_B: p_CT = CT_from_CAMWORLD @ p_CAMWORLD, CT in LPS mm; xreg's cam_to_vol)",
               "transform": {"to": "CT", "from": "CAMWORLD"},
               "file_pattern": POSE_FILE_PATTERN}
# Refinement pyramid as detector pixel sizes (mm): the pilot's scales 12/6/3 on its 0.388 mm
# detector. xvr's scale is a downsampling factor of the uncropped detector, so a finer native
# detector gets proportionally larger factors and the same cost and resolution (the native
# 0.194 mm Cios at 12/6/3 took 8.3 s per view against ~3 s).
REFINE_PIXEL_MM = (4.656, 2.328, 1.164)
REFINE_ITERS = (500, 500, 500)
REFINE_PATIENCE = (10, 10, 10)


def _preprocess(image: np.ndarray, crop_px: int, intensity: str) -> torch.Tensor:
    from xvr.io.xray import _preprocess_xray

    img = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
    return _preprocess_xray(img, crop_px, subtract_background=False, linearize=(intensity == "raw"))


class _GivenPose:
    """Minimal xvr initializer: the network pose we already predicted."""

    orientation = "AP"
    reverse_x_axis = False

    def __init__(self):
        self.pose = None

    def __call__(self, ctx):  # pragma: no cover - xvr interface
        return self.pose


def _refine(reg, pose, img: torch.Tensor, cam, crop_px: int) -> tuple[np.ndarray, dict]:
    """xvr's multiscale gradient registration from ``pose`` against ``img`` (already cropped).

    Scales are relative to the uncropped detector, as in ``xvr register --crop``.
    """
    from diffdrr.drr import DRR
    from xvr.register.pose import Pose

    _, _, h, w = img.shape
    drr = DRR(subject=reg.subject, height=h, width=w, reverse_x_axis=False, renderer="trilinear",
              voxel_shift=0.0, sdd=cam.sdd, delx=cam.col_spacing, dely=cam.row_spacing,
              x0=0.0, y0=0.0).cuda()
    p = Pose(pose, "euler_angles", "ZXY").cuda()
    t0 = time.perf_counter()
    log = reg._run_multiscale(drr, p, img, False, crop_px)
    torch.cuda.synchronize()
    final = p().matrix[0].detach().double().cpu().numpy()
    info = {"seconds": time.perf_counter() - t0,
            "final_similarity": float(log.losses[-1]) if log and log.losses else None,
            "steps": len(log.losses) if log else 0}
    del drr
    return final, info


def _write_stage(stage_dir: Path, poses: dict[str, np.ndarray], prov: dict, quality: dict | None = None) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for pid, X in poses.items():
        f = stage_dir / pose_file_name(pid, POSE_KEY)
        write_itk_xform(f, X)
        files[pid] = {"file": f.name, "sha256": sha256_file(f)}
    (stage_dir / "provenance.json").write_text(
        json.dumps({**POSE_RECORD, **prov, "files": files}, indent=1, default=str))
    if quality is not None:
        (stage_dir / "quality.json").write_text(json.dumps(quality, indent=1, default=str))
    return files


def _stats(values: list[float]) -> dict:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max()),
            "total": float(a.sum())}


def _timing_summary(setup: dict, views: dict, wall_s: float) -> dict:
    """Setup seconds, per-view stage statistics (first view separately: CUDA/cudnn warm-up) and
    the latency a live system would see per frame once warm."""
    ids = list(views)
    stages = sorted({k for v in views.values() for k in v["timing"]})
    per_stage = {s: _stats([views[i]["timing"].get(s) for i in ids[1:]]) for s in stages}
    first = views[ids[0]]["timing"] if ids else {}
    net_path = ("read_s", "preprocess_s", "network_s")
    latency = {
        "network_pose": _stats([sum(views[i]["timing"].get(k, 0.0) for k in net_path) for i in ids[1:]]),
        "network_pose_plus_qc": _stats([sum(views[i]["timing"].get(k, 0.0) for k in (*net_path, "qc_network_s"))
                                        for i in ids[1:]]),
        "refined_pose_plus_qc": _stats([sum(views[i]["timing"].get(k, 0.0)
                                            for k in (*net_path, "refine_s", "qc_network_s", "qc_refined_s"))
                                        for i in ids[1:]]) if "refine_s" in stages else None,
    }
    return {"setup_s": setup, "first_view_s": first, "per_view_after_first_s": per_stage,
            "per_frame_latency_after_first_s": latency, "n_views": len(ids), "wall_s": wall_s,
            "gpu": torch.cuda.get_device_name(0), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}


def stage_active(src: Path, poses_dir: Path) -> None:
    """Replace ``poses_dir`` with a copy of ``src`` (write beside, then swap)."""
    poses_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = poses_dir.with_name(poses_dir.name + ".staging")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    old = poses_dir.with_name(poses_dir.name + ".previous")
    if old.exists():
        shutil.rmtree(old)
    if poses_dir.exists():
        poses_dir.rename(old)
    tmp.rename(poses_dir)
    if old.exists():
        shutil.rmtree(old)  # its content remains in the run directory it was staged from


def predict(cfg: PoseConfig, model_dir: Path | None = None, ids: list[str] | None = None,
            refine: bool | None = None, stage: str | None = None, no_stage: bool = False) -> dict:
    from xvr.model.inference import predict_pose
    from xvr.model.network import load_model

    from .quality import QualityChecker, choose_best, overlay_sheets, summarize, thresholds

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prediction")
    t_wall = time.perf_counter()
    model_dir = Path(model_dir) if model_dir else active_model_dir(cfg.models_dir)
    card_path = model_dir / "model_card.json"
    card = json.loads(card_path.read_text()) if card_path.is_file() else {}
    ids = ids or cfg.projection_ids()
    refine = cfg.refine if refine is None else refine
    stage = stage or cfg.stage
    if stage in ("refined", "best") and not refine:
        raise ValueError(f"staging '{stage}' poses requires refinement")

    run_dir = cfg.runs_dir / f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_predict"
    run_dir.mkdir(parents=True, exist_ok=False)
    status.emit("start", stage="predict", run_dir=str(run_dir), model_dir=str(model_dir), n=len(ids))

    setup = {}
    t0 = time.perf_counter()
    model, mconfig = load_model(str(model_dir / "model.pth"))
    torch.cuda.synchronize()
    setup["model_load_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    bridge = Bridge.from_ct(cfg.ct, mconfig["orientation"])
    setup["frame_bridge_s"] = time.perf_counter() - t0
    if mconfig.get("reverse_x_axis"):
        raise ValueError("models trained with reverse_x_axis are not supported by the frame bridge")

    reg = None
    if refine:
        from xvr.register import Register

        ref_cam = read_projection(cfg.projection_path(cfg.reference_id or ids[0]), with_image=False).camera
        refine_scales = [round(p / ref_cam.col_spacing, 4) for p in REFINE_PIXEL_MM]
        t0 = time.perf_counter()
        reg = Register(initializer=_GivenPose(), imagepath=str(cfg.ct), labelpath=str(cfg.seg),
                       labels=list(cfg.refine_labels), scales=refine_scales,
                       n_itrs=list(REFINE_ITERS), patience=list(REFINE_PATIENCE))
        setup["refine_init_s"] = time.perf_counter() - t0
    checker = QualityChecker(cfg.ct, cfg.seg, cfg.qc_targets, cfg.prior.center_label, bridge,
                             card.get("prior", {}).get("resolved_ranges"), cfg.crop_px)
    setup["qc_init_s"] = checker.init_seconds
    setup["total_s"] = sum(setup.values())

    net, ref, views = {}, {}, {}
    quality = {"network": {}, "refined": {}}
    thumbs, drrs = {}, {"network": {}, "refined": {}}
    for i, pid in enumerate(ids):
        timing = {}
        path = cfg.projection_path(pid)
        t0 = time.perf_counter()
        proj = read_projection(path)
        timing["read_s"] = time.perf_counter() - t0
        cam = proj.camera
        cam.require_centered()
        if abs(cam.sdd - mconfig["sdd"]) > 0.01 * mconfig["sdd"]:
            status.emit("warning", stage="predict", id=pid,
                        message=f"projection SDD {cam.sdd:.1f} mm differs from the model's {mconfig['sdd']:.1f} mm")
        t0 = time.perf_counter()
        img = _preprocess(proj.image, cfg.crop_px, cfg.intensity)
        timing["preprocess_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        pose = predict_pose(model, mconfig, img, sdd=cam.sdd, delx=cam.col_spacing, dely=cam.row_spacing,
                            x0=0.0, y0=0.0)
        torch.cuda.synchronize()
        timing["network_s"] = time.perf_counter() - t0
        P = pose.matrix[0].double().cpu().numpy()
        X = bridge.diffdrr_to_xreg(P, cam.extrinsic)
        net[pid] = X
        views[pid] = {"projection": str(path), "projection_sha256": sha256_file(path),
                      "xvr_parameters": xvr_parameters(P)}
        xray = checker.xray_at_sim_size(img)
        thumbs[pid] = xray[0, 0].cpu().numpy()
        quality["network"][pid], drrs["network"][pid] = checker.evaluate(X, xray, cam)
        timing["qc_network_s"] = quality["network"][pid]["seconds"]
        if reg is not None:
            Pf, info = _refine(reg, pose.cuda(), img, cam, cfg.crop_px)
            timing["refine_s"] = info["seconds"]
            ref[pid] = bridge.diffdrr_to_xreg(Pf, cam.extrinsic)
            views[pid]["refine"] = info
            quality["refined"][pid], drrs["refined"][pid] = checker.evaluate(ref[pid], xray, cam, network_pose=X)
            timing["qc_refined_s"] = quality["refined"][pid]["seconds"]
        views[pid]["timing"] = timing
        best = quality["network"][pid]
        if reg is not None:
            best = quality[choose_best(quality["network"][pid], quality["refined"][pid])][pid]
        status.emit("progress", stage="predict", id=pid, done=i + 1, total=len(ids), tier=best["tier"],
                    gncc=best["gncc"], seconds=sum(timing.values()))

    qc_dir = run_dir / "qc"
    qc_dir.mkdir()
    stages = ["network"] + (["refined", "best"] if reg is not None else [])
    best_poses, source = {}, {}
    if reg is not None:
        quality["best"], drrs["best"] = {}, {}
        for pid in ids:
            source[pid] = choose_best(quality["network"][pid], quality["refined"][pid])
            best_poses[pid] = ref[pid] if source[pid] == "refined" else net[pid]
            quality["best"][pid] = {**quality[source[pid]][pid], "source_stage": source[pid]}
            drrs["best"][pid] = drrs[source[pid]][pid]
    quality_files = {}
    for s in stages:
        quality_files[s] = {"thresholds": thresholds(), "targets": cfg.qc_targets,
                            "summary": summarize(quality[s]), "views": quality[s]}
        quality_files[s]["figures"] = [Path(p).name for p in
                                       overlay_sheets(str(qc_dir / f"{s}_overlays"), ids, thumbs, drrs[s], quality[s])]
    if source:
        quality_files["best"]["summary"]["source_counts"] = {
            s: sum(1 for v in source.values() if v == s) for s in ("network", "refined")}

    timing_summary = _timing_summary(setup, views, time.perf_counter() - t_wall)
    # xreg reads pose directories relative to its experiment_dir, which is the project root.
    in_project = cfg.project_root is not None and cfg.poses_dir.is_relative_to(cfg.project_root)
    xreg_init = ({"xreg_init": {"method": "run", "from_target": POSE_KEY,
                                "from_run": cfg.poses_dir.relative_to(cfg.project_root).as_posix()}}
                 if in_project else {})
    common = {"model_dir": str(model_dir), "model_sha256": card.get("model_sha256"),
              "model_self_check_passed": card.get("self_check_passed"),
              "config": str(cfg.path), "config_sha256": sha256_file(cfg.path),
              "crop_px": cfg.crop_px, "intensity": cfg.intensity, "run_dir": str(run_dir),
              **xreg_init}
    _write_stage(run_dir / "network", net, {**common, "stage": "network"}, quality_files["network"])
    if ref:
        refine_record = {"labels": list(cfg.refine_labels), "pixel_mm": REFINE_PIXEL_MM,
                         "scales": refine_scales, "n_itrs": REFINE_ITERS, "patience": REFINE_PATIENCE}
        _write_stage(run_dir / "refined", ref, {**common, "stage": "refined", "refine": refine_record},
                     quality_files["refined"])
        _write_stage(run_dir / "best", best_poses,
                     {**common, "stage": "best", "refine": refine_record, "source_stage": source,
                      "rule": "refined pose unless refinement lowered gncc by more than best_tolerance"},
                     quality_files["best"])
    (run_dir / "timing.json").write_text(json.dumps(timing_summary, indent=1))
    run = {"ids": ids, "stage_staged": None if no_stage else stage, "views": views, "timing": timing_summary,
           "quality_summary": {s: quality_files[s]["summary"] for s in stages},
           "poses": {"network": {k: v.tolist() for k, v in net.items()},
                     "refined": {k: v.tolist() for k, v in ref.items()}},
           "best_source_stage": source, **POSE_RECORD, **common}
    (run_dir / "run.json").write_text(json.dumps(run, indent=1, default=str))

    staged = None
    if not no_stage:
        stage_active(run_dir / stage, cfg.poses_dir)
        staged = str(cfg.poses_dir)
    counts = quality_files[stage]["summary"]["counts"]
    result = {"run_dir": str(run_dir), "staged_poses_dir": staged, "pose_stage": stage, "n": len(ids),
              "tiers": counts, "wall_s": timing_summary["wall_s"]}
    status.emit("result", stage="predict", **result)
    return result
