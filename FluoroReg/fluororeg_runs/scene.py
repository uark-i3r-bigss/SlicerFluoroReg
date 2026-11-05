"""Placement of anatomy and C-arms for one choice of view, stages and scene frame (pure numpy).

Nodes of the Slicer scene and what their to-parent matrices map:
  anatomy[t]  CT content of target t (RAS, as Slicer loads CT, STL and FCSV) -> scene
  carm[v]     CAMWORLD[v] content (frustum, detector image, 2D landmarks) -> scene

CT-centred scene (Slicer's RAS world = CT in RAS):
  carm[V]          = RAS_from_LPS @ P_primary[V]                    (the selected view and stage)
  anatomy[primary] = identity
  anatomy[t]       = RAS_from_LPS @ P_primary[V] @ inv(P_t[V]) @ LPS_from_RAS   (CT[primary]_from_CT[t])
  carm[j != V]     = RAS_from_LPS @ P_primary[j] of its final pose (or the selected stage when applied
                     to all views and present)
Camera-centred scene (world = CAMWORLD[V]): every matrix above left-multiplied by inv(carm[V]), so
the selected camera stays fixed and the anatomy moves when stages change.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from .manifest import POSE_STAGES, XregRun, camworld_frame
from .transforms import RAS_FROM_LPS, inv, lps_to_ras, primary_from_target

# Pose used for a camera that is not the selected view, best first.
CAMERA_STAGE_PREFERENCE = ("final", "fiducial", "perturbed_init", "init", "ground_truth")


class SceneMatrices:
    def __init__(self, frame: str, view_id: str):
        self.frame = frame  # "CT" (RAS) or "CAMWORLD[<view>]"
        self.view_id = view_id
        self.anatomy: Dict[str, np.ndarray] = {}
        self.anatomy_stage: Dict[str, Optional[str]] = {}
        self.carm: Dict[str, np.ndarray] = {}
        self.carm_stage: Dict[str, str] = {}
        self.final_drr_views = set()  # stored DRRs are final-only, never overlays of an earlier model stage


def default_stages(run: XregRun, view_id: str) -> Dict[str, Optional[str]]:
    """Per target: "final" when present for the view, else the latest stage present, else None."""
    out: Dict[str, Optional[str]] = {}
    for key in run.target_keys:
        present = run.pose_stages(view_id, key)
        out[key] = "final" if "final" in present else (present[-1] if present else None)
    return out


def camera_stage(run: XregRun, view_id: str, preferred: Optional[str] = None) -> Optional[str]:
    primary = run.primary_key
    order = ((preferred,) if preferred else ()) + CAMERA_STAGE_PREFERENCE
    for stage in order:
        if stage in POSE_STAGES and run.pose(view_id, primary, stage) is not None:
            return stage
    return None


def scene_matrices(run: XregRun, view_id: str, stages: Optional[Mapping[str, Optional[str]]] = None,
                   camera_centred: bool = False, stages_all_views: bool = False) -> SceneMatrices:
    """Matrices of every anatomy and C-arm node for the selected view and per-target stages."""
    chosen = default_stages(run, view_id)
    chosen.update(stages or {})
    primary = run.primary_key
    p_stage = chosen.get(primary)
    p = run.pose_matrix(view_id, primary, p_stage) if p_stage else None
    if p is None:
        raise ValueError(f"view {view_id}: no {p_stage} pose of the primary target {primary}")

    scene = SceneMatrices(camworld_frame(view_id) if camera_centred else "CT", view_id)
    for key in run.target_keys:
        if key == primary:
            scene.anatomy[key], scene.anatomy_stage[key] = np.eye(4), p_stage
            continue
        stage = chosen.get(key)
        pt = run.pose_matrix(view_id, key, stage) if stage else None
        if pt is None:
            scene.anatomy[key], scene.anatomy_stage[key] = np.eye(4), None
        else:
            scene.anatomy[key], scene.anatomy_stage[key] = lps_to_ras(primary_from_target(p, pt)), stage

    for vid in run.view_ids:
        if vid == view_id:
            scene.carm[vid], scene.carm_stage[vid] = RAS_FROM_LPS @ p, p_stage
            continue
        stage = camera_stage(run, vid, p_stage if stages_all_views else None)
        if stage is not None:
            scene.carm[vid] = RAS_FROM_LPS @ run.pose_matrix(vid, primary, stage)
            scene.carm_stage[vid] = stage
    if all(stage == "final" for stage in scene.anatomy_stage.values()):
        scene.final_drr_views = {vid for vid, stage in scene.carm_stage.items() if stage == "final"}


    if camera_centred:
        to_cam = inv(scene.carm[view_id])
        scene.anatomy = {k: to_cam @ m for k, m in scene.anatomy.items()}
        scene.carm = {k: to_cam @ m for k, m in scene.carm.items()}
    return scene
