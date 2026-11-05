"""Registration assessment: ``R/assessment.json`` (schema 3) from a schema 4 run.

Thresholds are unchanged. High certainty requires a mover-stage objective with provenance covering
every registered view, and footprint gNCC plus valid positive coverage (<= 1) on every required view.
The scorer's null gNCC marks insufficient footprint; assessment adds no target-size threshold.
Stage objectives include shared-image and held-anatomy terms: they are not independent target scores.
Multiview registrations use the lowest available gNCC, but partial evidence cannot yield high certainty.
Historic sole-target runs can infer attribution from numeric stages or the registration's final
objective, explicitly marked inferred_single_target. Multiple-target runs never use that fallback.
"""

import math
import os

from .pose import pose_config_rel, run_pose_script
from .project import read_json, write_json
from .status import InputError, emit

SCHEMA_VERSION = 3
CERTAINTIES = ("high", "medium", "low")
THRESHOLDS = {"gncc_high": 0.8, "gncc_low": 0.7, "objective_max": 0.85,
              "calibration": "single-view pelvis research calibration, 2026-09-29"}
ASSESSMENT_NAME = "assessment.json"
GNCC_NAME = "assessment_gncc.json"


def load_run(run_dir):
    """(run_dir, manifest, project dir, run.dir) of a schema 4 run."""
    run_dir = os.path.abspath(run_dir)
    manifest = read_json(os.path.join(run_dir, "run_manifest.json"), "run manifest")
    if not isinstance(manifest, dict) or manifest.get("kind") != "xreg-run-manifest":
        raise InputError(f"{run_dir}/run_manifest.json is not an xreg run manifest")
    if manifest.get("schema_version") != 4:
        raise InputError(f"{run_dir}/run_manifest.json has schema_version {manifest.get('schema_version')}; "
                         "only schema 4 is read")
    run_rel = (manifest.get("run") or {}).get("dir") or ""
    parts = [p for p in run_rel.split("/") if p]
    real = os.path.realpath(run_dir)
    if not parts or os.path.isabs(run_rel) or real.split(os.sep)[-len(parts):] != parts:
        raise InputError(f"{run_dir}: run.dir '{run_rel}' does not name this run directory")
    project = os.sep.join(real.split(os.sep)[:-len(parts)]) or os.sep
    return real, manifest, project, run_rel


def run_case(run_rel, manifest):
    parts = run_rel.split("/")
    if len(parts) >= 3 and parts[0] == "outputs":
        return parts[1]
    ct = ((manifest.get("inputs") or {}).get("ct") or {}).get("path") or ""
    base = os.path.basename(ct)
    return base[:-len("_ct.nii.gz")] if base.endswith("_ct.nii.gz") else None


def classify(gncc, objective):
    """(certainty, reasons) of one (registration, target)."""
    t = THRESHOLDS
    if objective is None:
        return "medium", ["target mover-stage objective unavailable"]
    obj_ok = objective <= t["objective_max"]
    obj_text = f"objective {objective:.3f} {'<=' if obj_ok else '>'} {t['objective_max']}"
    if gncc is None:
        return "medium", ["objective_only", obj_text]
    gncc_text = f"gNCC {gncc:.3f}"
    if gncc >= t["gncc_high"] and obj_ok:
        return "high", [f"{gncc_text} >= {t['gncc_high']}", obj_text]
    if gncc < t["gncc_low"] and not obj_ok:
        return "low", [f"{gncc_text} < {t['gncc_low']}", obj_text]
    return "medium", [gncc_text, obj_text]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _target_objective(reg, key):
    """Latest completed stage moving this target; held targets never inherit the last-stage score."""
    quality = reg.get("quality") or {}
    objective = (quality.get("target_objectives") or {}).get(key)
    provenance = (quality.get("target_objective_provenance") or {}).get(key) or {}
    if (_number(objective) and provenance.get("stage_id") is not None
            and key in (provenance.get("moving") or [])):
        provenance = dict(provenance)
        provenance["scope"] = "shared_image_with_held_anatomy" if provenance.get("static") else "shared_image"
        return objective, provenance
    latest = None
    for stage in reg.get("stages") or []:
        if key in (stage.get("moving") or []) and _number(stage.get("final_sim_value")):
            latest = stage
    if latest is None:
        return None, None
    provenance = {k: latest[k] for k in ("stage_id", "phase", "moving", "static", "views", "pose_from")
                  if k in latest}
    provenance["scope"] = "shared_image_with_held_anatomy" if latest.get("static") else "shared_image"
    return latest["final_sim_value"], provenance


def build_assessment(manifest, run_rel, scores=None):
    """Build schema 3; ``scores`` maps (view, target) to score rows containing gncc and coverage."""
    scores = scores or {}
    target_keys = [t.get("key") for t in manifest.get("targets") or []]
    items = []
    for reg in manifest.get("registrations") or []:
        views = list(dict.fromkeys(str(v) for v in
                     (reg.get("views") or ([] if reg.get("id") == "joint" else [reg.get("id")]))))
        multiview = reg.get("kind") == "multiview" or len(views) > 1
        for key in target_keys:
            objective, provenance = _target_objective(reg, key)
            if objective is None and len(target_keys) == 1:
                # Historic single-target writers omitted movers. Only a sole declared target
                # unambiguously owns that registration's objective; never infer this for multiple targets.
                numeric_stages = [s for s in reg.get("stages") or [] if _number(s.get("final_sim_value"))]
                latest = numeric_stages[-1] if numeric_stages else None
                candidate = latest["final_sim_value"] if latest else (reg.get("quality") or {}).get("final_objective")
                if _number(candidate):
                    objective = candidate
                    provenance = {"scope": "inferred_single_target", "moving": [key], "static": [],
                                  "views": list(latest.get("views", views)) if latest else list(views),
                                  "source": "stages.final_sim_value" if latest else "quality.final_objective"}
                    if latest and "stage_id" in latest:
                        provenance["stage_id"] = latest["stage_id"]
            rows = {v: scores.get((v, key)) or {} for v in views}
            known = [(v, row["gncc"]) for v, row in rows.items() if _number(row.get("gncc"))]
            scored = [v for v, _ in known]
            missing = [v for v in views if v not in scored]
            missing_coverage = [v for v, row in rows.items() if not _number(row.get("coverage"))]
            insufficient = [v for v, row in rows.items()
                            if _number(row.get("coverage")) and not 0 < row["coverage"] <= 1]
            objective_views = set((provenance or {}).get("views") or [])
            missing_objective_views = [v for v in views if v not in objective_views]
            worst = min(known, key=lambda vg: vg[1]) if known else None
            gncc = worst[1] if worst else None
            complete = bool(views) and not (missing or missing_coverage or insufficient or missing_objective_views)
            if reg.get("status") not in (None, "success"):
                certainty = "low"
                reasons = [f"registration {reg.get('status')}" + (f": {reg['reason']}" if reg.get("reason") else "")]
            else:
                certainty, reasons = classify(gncc, objective)
                if certainty == "high" and not complete:
                    certainty = "medium"
                if missing:
                    reasons.append("missing gNCC for required views: " + ", ".join(missing))
                if missing_coverage:
                    reasons.append("coverage unavailable for required views: " + ", ".join(missing_coverage))
                if insufficient:
                    reasons.append("target coverage must be positive and <= 1 on views: " + ", ".join(insufficient))
                if missing_objective_views:
                    reasons.append("mover-stage objective provenance missing required views: "
                                   + ", ".join(missing_objective_views))
                if not views:
                    reasons.append("no required registered views recorded")
                if multiview and worst is not None:
                    reasons.append(f"lowest gNCC on view {worst[0]}")
            items.append({"registration": reg.get("id"), "view": None if multiview else (views[0] if views else None),
                          "target": key, "target_objective": objective, "objective_provenance": provenance,
                          "gncc": gncc, "certainty": certainty, "evidence_complete": complete,
                          "required_view_ids": views, "scored_view_ids": scored, "missing_view_ids": missing,
                          "missing_coverage_view_ids": missing_coverage,
                          "insufficient_coverage_view_ids": insufficient,
                          "missing_objective_view_ids": missing_objective_views,
                          "coverage_by_view": {v: row["coverage"] if _number(row.get("coverage")) else None
                                               for v, row in rows.items()},
                          "reasons": reasons})
    return {"schema_version": SCHEMA_VERSION, "run_dir": run_rel, "thresholds": dict(THRESHOLDS),
            "items": items, "flags": list(manifest.get("flags") or [])}


def _read_scores(path):
    data = read_json(path, "gNCC scores")
    return {(str(s.get("view")), s.get("target")): s for s in data.get("scores") or []}


def gncc_needed(project, case_id, no_gncc):
    return (not no_gncc and case_id is not None
            and os.path.isfile(os.path.join(project, *pose_config_rel(case_id).split("/"))))


def assess(run_dir, settings, no_gncc=False):
    """Write ``R/assessment.json``; returns (result fields, needs_attention)."""
    run_dir, manifest, project, run_rel = load_run(run_dir)
    case_id = run_case(run_rel, manifest)
    scores = None
    attention = False
    gncc_file = None
    if gncc_needed(project, case_id, no_gncc):
        out = os.path.join(run_dir, GNCC_NAME)
        args = ["score", "--config", os.path.join(project, *pose_config_rel(case_id).split("/")),
                "--run", run_dir, "--out", out]
        rc, _, log = run_pose_script(project, case_id, settings, args, "assess", step="assess", phase="assess")
        if rc == 0 and os.path.isfile(out):
            scores = _read_scores(out)
            gncc_file = f"{run_rel}/{GNCC_NAME}"
        else:
            attention = True
            emit("warning", "assess", message=f"gNCC scoring failed (status {rc}); certainty uses the objective only; "
                                              f"see {log}")
    elif not no_gncc:
        emit("warning", "assess", message=f"no {pose_config_rel(case_id or '<case>')}: certainty uses the objective only")
    doc = build_assessment(manifest, run_rel, scores)
    write_json(os.path.join(run_dir, ASSESSMENT_NAME), doc)
    counts = {}
    for item in doc["items"]:
        counts[item["certainty"]] = counts.get(item["certainty"], 0) + 1
        if item["certainty"] != "high":
            where = item["registration"] if item["view"] is None else f"view {item['view']}"
            emit("warning", "assess", id=item["registration"],
                 message=f"{where} target {item['target']}: {item['certainty']} certainty "
                         f"({'; '.join(item['reasons'])})")
    result = {"run_dir": run_rel, "assessment": f"{run_rel}/{ASSESSMENT_NAME}", "counts": counts,
              "items": doc["items"]}
    if gncc_file:
        result["gncc"] = gncc_file
    return result, attention
