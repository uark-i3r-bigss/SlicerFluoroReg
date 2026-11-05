"""Pose-estimation config (schema v1): one file per CT case, shared by terminal and GUI.

Location convention: ``PROJECT/configs/pose_estimation/<case>.json``. Relative paths resolve
against the project root, the nearest ancestor of the config file that holds
``project_manifest.json``. Unknown keys are an error, as in xreg's registration configs.
See docs/config_schema.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 1

METHOD_DEFAULTS = {
    # The paper's finetune recipe (500 x 116 DRRs) at the batch that fits a shared 24 GB GPU.
    "finetune": {"iterations": 1200, "batch_size": 48, "learning_rate": 1e-3,
                 "warmup_iterations": 10, "grad_accum": 1},
    # The DeepFluoro de-novo recipe at batch 48; 15k iterations proved too short on the one CT tried.
    "de_novo": {"iterations": 15000, "batch_size": 48, "learning_rate": 1e-3,
                "warmup_iterations": 1000, "grad_accum": 4},
}


class ConfigError(ValueError):
    pass


def _reject_unknown(d: dict, allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}; allowed {sorted(allowed)}")


def _pair(v, where: str) -> tuple[float, float]:
    if not (isinstance(v, (list, tuple)) and len(v) == 2 and float(v[0]) < float(v[1])):
        raise ConfigError(f"{where}: expected [low, high] with low < high, got {v!r}")
    return float(v[0]), float(v[1])


def find_project_root(config_path: Path) -> Path | None:
    for d in [config_path.parent, *config_path.parents]:
        if (d / "project_manifest.json").is_file():
            return d
    return None


@dataclass(frozen=True)
class Prior:
    center_label: int = 1
    alpha_deg: tuple[float, float] = (-45.0, 45.0)
    beta_deg: tuple[float, float] = (-45.0, 45.0)
    gamma_deg: tuple[float, float] = (-15.0, 15.0)
    lateral_mm: float = 150.0
    axial_mm: float = 150.0
    source_to_center_mm: tuple[float, float] = (450.0, 1000.0)


@dataclass(frozen=True)
class ModelSpec:
    method: str = "finetune"
    iterations: int = 1200
    batch_size: int = 48
    learning_rate: float = 1e-3
    warmup_iterations: int = 10
    grad_accum: int = 1
    seed: int = 20260924
    self_check_samples: int = 100
    self_check_min_capture: float = 0.9


@dataclass(frozen=True)
class PoseConfig:
    path: Path
    project_root: Path | None
    case_id: str
    ct: Path
    seg: Path
    proj_dir: Path
    proj_pattern: str
    proj_ids: tuple[str, ...] | None
    reference_id: str | None
    crop_px: int
    intensity: str
    prior: Prior
    model: ModelSpec
    refine: bool
    refine_labels: tuple[int, ...]
    stage: str
    qc_targets: dict[str, int]
    models_dir: Path
    poses_dir: Path
    runs_dir: Path
    raw: dict = field(repr=False, default_factory=dict)

    def projection_path(self, proj_id: str) -> Path:
        return self.proj_dir / self.proj_pattern.replace("{id}", proj_id)

    def projection_ids(self) -> list[str]:
        if self.proj_ids is not None:
            return list(self.proj_ids)
        prefix, _, suffix = self.proj_pattern.partition("{id}")
        if not self.proj_dir.is_dir():
            raise ConfigError(f"projection directory {self.proj_dir} does not exist; training and "
                              "prediction need at least one projection H5")
        ids = []
        for p in sorted(self.proj_dir.iterdir()):
            n = p.name
            if n.startswith(prefix) and n.endswith(suffix) and len(n) > len(prefix) + len(suffix):
                ids.append(n[len(prefix): len(n) - len(suffix)])
        if not ids:
            raise ConfigError(f"no projections match {self.proj_dir}/{self.proj_pattern}")
        return ids


def load_config(path: str | Path) -> PoseConfig:
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    _reject_unknown(raw, {"schema_version", "case_id", "volume", "projections", "detector",
                          "prior", "model", "predict", "output"}, str(path))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"{path}: schema_version must be {SCHEMA_VERSION}")
    case_id = raw.get("case_id")
    if not case_id or not re.fullmatch(r"[A-Za-z0-9._-]+", case_id):
        raise ConfigError(f"{path}: case_id must be a non-empty file-name-safe string")
    root = find_project_root(path)

    def resolve(p: str, where: str) -> Path:
        q = Path(p)
        if q.is_absolute():
            return q
        if root is None:
            raise ConfigError(f"{where}: relative path {p!r} but no project_manifest.json above {path}")
        return root / q

    vol = raw.get("volume") or {}
    _reject_unknown(vol, {"ct", "seg"}, "volume")
    if "ct" not in vol or "seg" not in vol:
        raise ConfigError("volume: 'ct' and 'seg' are required")

    proj = raw.get("projections") or {}
    _reject_unknown(proj, {"dir", "pattern", "ids"}, "projections")
    if "dir" not in proj:
        raise ConfigError("projections: 'dir' is required")
    pattern = proj.get("pattern", "{id}.h5")
    if "{id}" not in pattern:
        raise ConfigError("projections.pattern must contain '{id}'")
    ids = proj.get("ids")
    if ids is not None and (not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids)):
        raise ConfigError("projections.ids must be a non-empty list of strings when given")

    det = raw.get("detector") or {}
    _reject_unknown(det, {"reference_id", "crop_px", "intensity"}, "detector")
    crop = int(det.get("crop_px", 100))
    if crop < 0 or crop % 2:
        raise ConfigError("detector.crop_px must be a non-negative even number (total pixels)")
    intensity = det.get("intensity", "raw")
    if intensity not in ("raw", "log"):
        raise ConfigError("detector.intensity must be 'raw' (transmission, log applied) or 'log'")

    pr = raw.get("prior") or {}
    _reject_unknown(pr, {"center_label", "alpha_deg", "beta_deg", "gamma_deg",
                         "lateral_mm", "axial_mm", "source_to_center_mm"}, "prior")
    d = Prior()
    prior = Prior(
        center_label=int(pr.get("center_label", d.center_label)),
        alpha_deg=_pair(pr.get("alpha_deg", d.alpha_deg), "prior.alpha_deg"),
        beta_deg=_pair(pr.get("beta_deg", d.beta_deg), "prior.beta_deg"),
        gamma_deg=_pair(pr.get("gamma_deg", d.gamma_deg), "prior.gamma_deg"),
        lateral_mm=float(pr.get("lateral_mm", d.lateral_mm)),
        axial_mm=float(pr.get("axial_mm", d.axial_mm)),
        source_to_center_mm=_pair(pr.get("source_to_center_mm", d.source_to_center_mm),
                                  "prior.source_to_center_mm"),
    )

    md = raw.get("model") or {}
    _reject_unknown(md, {"method", "iterations", "batch_size", "learning_rate", "warmup_iterations",
                         "grad_accum", "seed", "self_check_samples", "self_check_min_capture"}, "model")
    method = md.get("method", "finetune")
    if method not in METHOD_DEFAULTS:
        raise ConfigError(f"model.method must be one of {sorted(METHOD_DEFAULTS)}")
    base = {**METHOD_DEFAULTS[method], **{k: v for k, v in md.items() if k != "method"}}
    model = ModelSpec(method=method, **{k: type(getattr(ModelSpec(), k))(v) for k, v in base.items()})

    pd = raw.get("predict") or {}
    _reject_unknown(pd, {"refine", "refine_labels", "stage", "qc_targets"}, "predict")
    stage = pd.get("stage", "network")
    if stage not in ("network", "refined", "best"):
        raise ConfigError("predict.stage must be 'network', 'refined' or 'best'")
    refine = bool(pd.get("refine", stage != "network"))
    if stage != "network" and not refine:
        raise ConfigError(f"predict.stage '{stage}' requires predict.refine true")
    qc_targets = pd.get("qc_targets", {"target": prior.center_label})
    if (not isinstance(qc_targets, dict) or not qc_targets
            or not all(isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
                       for k, v in qc_targets.items())):
        raise ConfigError("predict.qc_targets must be a non-empty {name: label} mapping of integers")

    out = raw.get("output") or {}
    _reject_unknown(out, {"models_dir", "poses_dir", "runs_dir"}, "output")

    return PoseConfig(
        path=path, project_root=root, case_id=case_id,
        ct=resolve(vol["ct"], "volume.ct"), seg=resolve(vol["seg"], "volume.seg"),
        proj_dir=resolve(proj["dir"], "projections.dir"), proj_pattern=pattern,
        proj_ids=tuple(ids) if ids else None,
        reference_id=det.get("reference_id"), crop_px=crop, intensity=intensity,
        prior=prior, model=model, refine=refine,
        refine_labels=tuple(int(x) for x in pd.get("refine_labels", [prior.center_label])),
        stage=stage, qc_targets=dict(qc_targets),
        models_dir=resolve(out.get("models_dir", f"meta_data/pose_models/{case_id}"), "output.models_dir"),
        poses_dir=resolve(out.get("poses_dir", f"meta_data/initial_poses/{case_id}"), "output.poses_dir"),
        runs_dir=resolve(out.get("runs_dir", f"meta_data/tmp/pose_estimation/{case_id}"), "output.runs_dir"),
        raw=raw,
    )
