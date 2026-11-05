"""Train (finetune or de novo) a patient-specific pose regressor with xvr's Trainer.

Output: ``<models_dir>/<YYYYMMDD_HHMMSS>_<method>/`` holding ``model.pth`` (weights + xvr config,
loadable by ``xvr.model.network.load_model``), ``model_card.json`` and ``train_log.jsonl``.
The model becomes active (``<models_dir>/active.json``) only when its synthetic self-check passes,
unless activation is forced.

xvr's own training loop swallows every exception, including CUDA out-of-memory, and keeps going;
this loop fails loudly instead, so a model is never produced from silently skipped steps.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from . import status
from .bridge import Bridge
from .config import PoseConfig
from .geometry import label_centroid_lps, pose_ranges, training_geometry
from .selfcheck import run_self_check
from .weights import FOUNDATION, foundation_path
from .xreg_io import read_projection, sha256_file

LOG_EVERY = 25

# Peak GPU memory of one training process on an RTX 4090 (torch 2.10, 128 px, crop 100):
# 15.9 GiB held + a 4.4 GiB backward allocation at batch 48 (measured from an out-of-memory
# failure). The per-batch part scales with the batch; ~1.5 GiB is context + model.
PEAK_GIB_BASE = 1.5
PEAK_GIB_PER_48 = 18.8


def required_gpu_gib(batch_size: int) -> float:
    return PEAK_GIB_BASE + PEAK_GIB_PER_48 * batch_size / 48


def check_gpu_memory(batch_size: int) -> None:
    """Fail before training when the GPU cannot hold it, instead of an OOM minutes in."""
    free, total = torch.cuda.mem_get_info()
    need = required_gpu_gib(batch_size)
    if free / 2**30 < need:
        raise RuntimeError(
            f"GPU {torch.cuda.current_device()} ({torch.cuda.get_device_name()}) has {free / 2**30:.1f} of "
            f"{total / 2**30:.1f} GiB free; training at batch {batch_size} needs about {need:.1f} GiB. "
            "Wait for other GPU jobs (registration, annotation) to finish, or lower model.batch_size.")


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def activate(models_dir: Path, model_dir: Path) -> Path:
    """Point ``<models_dir>/active.json`` at ``model_dir``; relative to ``models_dir`` when inside it,
    so the project folder can be moved or copied."""
    active = models_dir / "active.json"
    root, target = Path(models_dir).resolve(), Path(model_dir).resolve()
    value = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
    tmp = active.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"model_dir": value, "activated": dt.datetime.now().isoformat()}, indent=1))
    tmp.replace(active)
    return active


def active_model_dir(models_dir: Path) -> Path:
    """The model named by ``<models_dir>/active.json`` (a relative ``model_dir`` is under ``models_dir``)."""
    active = models_dir / "active.json"
    if not active.is_file():
        raise FileNotFoundError(f"no active pose model: {active} missing (train one first)")
    d = Path(models_dir) / json.loads(active.read_text())["model_dir"]
    if not (d / "model.pth").is_file():
        raise FileNotFoundError(f"active model {d} has no model.pth")
    return d


def train(cfg: PoseConfig, force_activate: bool = False) -> dict:
    from xvr.model import Trainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    t_start = time.perf_counter()
    ids = cfg.projection_ids()
    ref_id = cfg.reference_id or ids[0]
    cam = read_projection(cfg.projection_path(ref_id), with_image=False).camera
    geom = training_geometry(cam, cfg.crop_px)
    bridge = Bridge.from_ct(cfg.ct, "AP")
    ranges = pose_ranges(cfg, bridge)
    m = cfg.model
    check_gpu_memory(m.batch_size)

    out = cfg.models_dir / f"{_timestamp()}_{m.method}"
    out.mkdir(parents=True, exist_ok=False)
    status.emit("start", stage="train", model_dir=str(out), method=m.method, iterations=m.iterations)

    base = None
    if m.method == "finetune":
        base = foundation_path(verify=True)

    torch.manual_seed(m.seed)
    np.random.seed(m.seed)
    t_init = time.perf_counter()
    trainer = Trainer(
        volpath=str(cfg.ct), maskpath=str(cfg.seg), outpath=str(out / "xvr_checkpoints"),
        **ranges, sdd=geom.sdd, height=geom.height, delx=geom.delx, orientation="AP",
        reverse_x_axis=False, parameterization="quaternion_adjugate", convention="ZXY",
        model_name="resnet34", pretrained=False, norm_layer="groupnorm", unit_conversion_factor=1000.0,
        p_augmentation=0.333, lr=m.learning_rate, weight_ncc=1.0, weight_geo=1e-2, weight_dice=1.0,
        weight_haus=0.0, batch_size=m.batch_size, n_total_itrs=m.iterations,
        n_warmup_itrs=m.warmup_iterations, n_grad_accum_itrs=m.grad_accum,
        n_save_every_itrs=m.iterations + 1, ckptpath=str(base) if base else None, reuse_optimizer=False,
        num_workers=0,
    )

    init_s = time.perf_counter() - t_init
    log_path = out / "train_log.jsonl"
    t0 = time.perf_counter()
    last = {}
    with open(log_path, "w") as logf:
        for itr in range(m.iterations):
            log, _, _ = trainer.step(itr, None)  # raises on any failure, including OOM
            if not np.isfinite(log["loss"]):
                raise FloatingPointError(f"non-finite training loss at iteration {itr}")
            last = log
            if itr % LOG_EVERY == 0 or itr == m.iterations - 1:
                rec = {"itr": itr, **log, "rot_deg": float(np.degrees(log["rgeo"] / geom.sdd)),
                       "trans_mm": log["tgeo"], "elapsed_s": time.perf_counter() - t0}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                status.emit("progress", stage="train", iteration=itr + 1, total=m.iterations,
                            loss=log["loss"], rot_deg=rec["rot_deg"], trans_mm=rec["trans_mm"])
    train_s = time.perf_counter() - t0
    peak_gib = torch.cuda.max_memory_reserved() / 2**30
    t_save = time.perf_counter()
    ckpt = {"model_state_dict": trainer.model.state_dict(), "config": trainer.config,
            "itr": m.iterations, "date": dt.datetime.now().isoformat()}
    torch.save(ckpt, out / "model.pth")
    shutil_dir = out / "xvr_checkpoints"
    if shutil_dir.is_dir() and not any(shutil_dir.iterdir()):
        shutil_dir.rmdir()
    del trainer
    torch.cuda.empty_cache()
    save_s = time.perf_counter() - t_save

    from xvr.model.network import load_model

    model, config = load_model(str(out / "model.pth"))
    center = label_centroid_lps(cfg.seg, cfg.prior.center_label)
    status.emit("progress", stage="self_check", done=0, total=m.self_check_samples)
    t_check = time.perf_counter()
    check = run_self_check(model, config, cfg.ct, cfg.seg, cfg.prior.center_label, cam, cfg.crop_px,
                           center, m.self_check_samples, m.seed + 1, emit=status.emit)
    check_s = time.perf_counter() - t_check
    passed = check["capture_rate"] >= m.self_check_min_capture and check["n_kept"] > 0
    (out / "self_check.json").write_text(json.dumps(check, indent=1))

    card = {
        "schema_version": 1,
        "case_id": cfg.case_id,
        "created": dt.datetime.now().isoformat(),
        "estimator": {"package": "xvreg", "commit": "caa55cc8096294cf70a218126bf16008dee0dec7",
                      "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                      "host": platform.node()},
        "method": m.method,
        "base_weights": {**FOUNDATION, "path": str(base)} if base else None,
        "inputs": {"config": str(cfg.path), "config_sha256": sha256_file(cfg.path),
                   "ct": str(cfg.ct), "ct_sha256": sha256_file(cfg.ct.resolve()),
                   "seg": str(cfg.seg), "seg_sha256": sha256_file(cfg.seg.resolve()),
                   "reference_projection": str(cfg.projection_path(ref_id))},
        "geometry": geom.as_dict(),
        "prior": {**cfg.raw.get("prior", {}), "resolved_ranges": ranges},
        "training": {"iterations": m.iterations, "batch_size": m.batch_size,
                     "learning_rate": m.learning_rate, "warmup_iterations": m.warmup_iterations,
                     "grad_accum": m.grad_accum, "seed": m.seed, "seconds": train_s, "final": last,
                     "peak_reserved_gib": peak_gib},
        # Wall-clock seconds per stage; setup = inputs, prior, weight hash; init = CT load and renderer.
        "timing": {"setup_s": t_init - t_start, "trainer_init_s": init_s, "train_loop_s": train_s,
                   "iterations_per_s": m.iterations / train_s, "save_s": save_s,
                   "self_check_s": check_s, "self_check_samples": m.self_check_samples,
                   "total_s": time.perf_counter() - t_start},
        "self_check": {k: v for k, v in check.items() if k != "samples"},
        "self_check_min_capture": m.self_check_min_capture,
        "self_check_passed": bool(passed),
        "model_sha256": sha256_file(out / "model.pth"),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=1, default=str))

    activated = None
    if passed or force_activate:
        activated = str(activate(cfg.models_dir, out))
    else:
        status.emit("warning", stage="self_check",
                    message=f"self-check capture {check['capture_rate']:.1%} < {m.self_check_min_capture:.0%}; "
                            "model kept but NOT activated")
    result = {"model_dir": str(out), "self_check_passed": bool(passed),
              "capture_rate": check["capture_rate"], "activated": activated, "train_seconds": train_s,
              "timing": card["timing"]}
    status.emit("result", stage="train", **result)
    return result
