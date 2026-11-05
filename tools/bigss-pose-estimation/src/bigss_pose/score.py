"""Score the final poses of an xreg run with the image similarity of the pose checks.

For every ``<run>/poses/view-<id>_target-<key>_CT_from_CAMWORLD.h5`` (single-view and multiview
runs alike) a whole-CT DRR is rendered at the pose and compared with the run's projection ``<id>``
(``views[].projection.path`` of ``run_manifest.json`` schema 4, relative to the experiment folder =
the run directory minus ``run.dir``): the Sobel-gradient NCC of ``quality.py`` restricted to the
target's footprint (its segmentation label projected through the pose, closed, holes filled,
dilated), plus the target's ``coverage`` of the cropped detector. The footprint keeps content the
CT lacks (spine above it, soft tissue, the DRR's field-of-view edge) out of the score, so it rates
the target's alignment alone. The label of target ``<key>`` is ``--targets <key>=<label>``, else
the manifest's ``targets[].segmentation_label``, else ``predict.qc_targets[<key>]``.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from . import status
from .bridge import Bridge
from .config import ConfigError, PoseConfig
from .predict import _preprocess
from .quality import QualityChecker, footprint, footprint_mask
from .xreg_io import read_itk_xform, read_projection

SCHEMA_VERSION = 1
MANIFEST_SCHEMA = 4
POSE_FILE = re.compile(r"view-(?P<view>.+?)_target-(?P<target>.+)_CT_from_CAMWORLD\.h5")


def parse_targets(items: list[str] | None) -> dict[str, int]:
    """``["pelvis=1", ...]`` -> ``{"pelvis": 1}`` (keys lower case, as in pose file names)."""
    out = {}
    for item in items or []:
        key, sep, label = item.partition("=")
        if not sep or not key or not re.fullmatch(r"-?\d+", label.strip()):
            raise ConfigError(f"--targets expects key=label with an integer label, got {item!r}")
        out[key.strip().lower()] = int(label)
    return out


def _experiment_dir(run: Path, manifest: dict) -> Path:
    rel = Path(manifest["run"]["dir"]).parts
    if not rel or Path(manifest["run"]["dir"]).is_absolute() or run.parts[-len(rel):] != rel:
        raise ConfigError(f"{run} does not end with the manifest's run.dir {manifest['run']['dir']!r}")
    return Path(*run.parts[:-len(rel)])


def _labels(keys: set[str], manifest: dict, cfg: PoseConfig, override: dict[str, int]) -> dict[str, int]:
    recorded = {t["key"].lower(): t.get("segmentation_label") for t in manifest.get("targets", [])}
    qc = {k.lower(): v for k, v in cfg.qc_targets.items()}
    labels = {}
    for key in sorted(keys):
        label = override.get(key)
        if label is None and isinstance(recorded.get(key), int):
            label = recorded[key]
        if label is None:
            label = qc.get(key)
        if label is None:
            raise ConfigError(f"no segmentation label for target {key!r}: pass --targets {key}=<label>, "
                              "or record it in the run manifest or predict.qc_targets")
        labels[key] = int(label)
    return labels


def score(cfg: PoseConfig, run_dir: str | Path, out: str | Path, targets: dict[str, int] | None = None,
          device: str | None = None) -> dict:
    """Score every pose file of ``run_dir``; write and return the ``score`` JSON."""
    t_wall = time.perf_counter()
    run = Path(run_dir).resolve()
    manifest_path = run / "run_manifest.json"
    if not manifest_path.is_file():
        raise ConfigError(f"{run} has no run_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ConfigError(f"{manifest_path}: schema_version {manifest.get('schema_version')!r}, "
                          f"expected {MANIFEST_SCHEMA}")
    experiment = _experiment_dir(run, manifest)

    poses = []
    for path in sorted((run / "poses").glob("view-*_target-*_CT_from_CAMWORLD.h5")):
        m = POSE_FILE.fullmatch(path.name)
        if m:
            poses.append((m["view"], m["target"], path))
    if not poses:
        raise ConfigError(f"no view-<id>_target-<key>_CT_from_CAMWORLD.h5 under {run / 'poses'}")
    views = {v["id"]: v for v in manifest.get("views", [])}
    projections = {}
    for view, _, _ in poses:
        rel = (views.get(view) or {}).get("projection", {}).get("path")
        if rel is None:
            raise ConfigError(f"{manifest_path}: no projection recorded for view {view!r}")
        projections[view] = experiment / rel
        if not projections[view].is_file():
            raise ConfigError(f"missing projection {projections[view]}")
    labels = _labels({t for _, t, _ in poses}, manifest, cfg, targets or {})

    inputs = manifest.get("inputs", {})
    for key, mine in (("ct", cfg.ct), ("segmentation", cfg.seg)):
        theirs = (inputs.get(key) or {}).get("path")
        if theirs is not None and (experiment / theirs).resolve() != mine.resolve():
            status.emit("warning", stage="score",
                        message=f"run {key} {experiment / theirs} differs from the pose config's {mine}")

    status.emit("start", stage="score", run=str(run), config=str(cfg.path), n=len(poses), targets=labels)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checker = QualityChecker(cfg.ct, cfg.seg, labels, cfg.prior.center_label, Bridge.from_ct(cfg.ct, "AP"),
                             None, cfg.crop_px, device=device)
    scores, views_read = [], {}
    for i, (view, key, path) in enumerate(poses):
        if view not in views_read:
            proj = read_projection(projections[view])
            proj.camera.require_centered()
            views_read[view] = (proj.camera,
                                checker.xray_at_sim_size(_preprocess(proj.image, cfg.crop_px, cfg.intensity)))
        cam, xray = views_read[view]
        X = np.asarray(read_itk_xform(path), dtype=float)
        pts = checker.targets[key]
        drr = checker.render(cam, [X])
        gncc = (None if float(drr.std()) < 1e-8
                else checker.gncc.masked(xray, drr, footprint_mask(X, cam, pts, cfg.crop_px)))
        rec = {"view": view, "target": key, "label": labels[key], "gncc": gncc,
               "coverage": footprint(X, cam, pts, cfg.crop_px)["coverage"]}
        scores.append(rec)
        status.emit("progress", stage="score", id=view, target=key, done=i + 1, total=len(poses),
                    gncc=gncc, coverage=rec["coverage"])

    result = {"schema_version": SCHEMA_VERSION, "run": str(run), "config": str(cfg.path), "scores": scores,
              "seconds": round(time.perf_counter() - t_wall, 2)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")
    status.emit("result", stage="score", out=str(out), n=len(scores), seconds=result["seconds"])
    return result
