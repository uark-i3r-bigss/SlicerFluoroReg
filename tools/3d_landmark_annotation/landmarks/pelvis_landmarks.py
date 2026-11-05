import os

import numpy as np
from stl import mesh

from . import status

SUPERIOR_END_PERCENTILE = 10.0
INFERIOR_REGION_PERCENTILE = 35.0
PT_INFERIOR_PERCENTILE = 45.0
PT_MEDIAL_PERCENTILE = 45.0
IT_INFERIOR_PERCENTILE = 60.0
IT_POSTERIOR_PERCENTILE = 60.0
IT_MEDIAL_PERCENTILE = 45.0
IPR_INFERIOR_PERCENTILE = 45.0
IPR_MEDIAL_PERCENTILE = 45.0
PSIS_SUPERIOR_PERCENTILE = 65.0
PSIS_POSTERIOR_PERCENTILE = 60.0


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def to_local(points, origin, axes):
    rel = np.asarray(points) - origin
    return np.column_stack([rel @ axes["x_axis"], rel @ axes["y_axis"], rel @ axes["z_axis"]])


def load_hip_model(hip_path):
    if hip_path and os.path.exists(hip_path):
        hip_mesh = mesh.Mesh.from_file(hip_path)
        vertices = np.unique(hip_mesh.vectors.reshape(-1, 3), axis=0)
        print(
            f"  Loaded {os.path.basename(hip_path)}: {len(vertices)} vertices, "
            f"X range: [{vertices[:, 0].min():.1f}, {vertices[:, 0].max():.1f}]"
        )
        return vertices
    return None


def build_pelvis_frame(vertices, left_vertices, right_vertices):
    left_centroid = left_vertices.mean(axis=0)
    right_centroid = right_vertices.mean(axis=0)
    x_axis = normalize(left_centroid - right_centroid)
    if x_axis is None:
        raise ValueError("Unable to build pelvis frame: invalid left-right axis")

    origin = 0.5 * (left_centroid + right_centroid)
    centered = vertices - origin
    projected = centered - np.outer(centered @ x_axis, x_axis)
    _, _, vh = np.linalg.svd(projected, full_matrices=False)
    z_seed = normalize(vh[0] - np.dot(vh[0], x_axis) * x_axis)
    if z_seed is None:
        raise ValueError("Unable to build pelvis frame: invalid superior-inferior seed")

    best_axes = None
    best_score = -np.inf

    for sign in (1.0, -1.0):
        z_axis = normalize(sign * z_seed)
        y_axis = normalize(np.cross(z_axis, x_axis))
        z_axis = normalize(np.cross(x_axis, y_axis))
        axes = {"x_axis": x_axis, "y_axis": y_axis, "z_axis": z_axis}
        local = to_local(vertices, origin, axes)

        high_mask = local[:, 2] >= np.percentile(local[:, 2], 100.0 - SUPERIOR_END_PERCENTILE)
        low_mask = local[:, 2] <= np.percentile(local[:, 2], SUPERIOR_END_PERCENTILE)
        high_spread = np.mean(np.linalg.norm(local[high_mask][:, :2], axis=1))
        low_spread = np.mean(np.linalg.norm(local[low_mask][:, :2], axis=1))

        abs_x = np.abs(local[:, 0])
        inferomedial_mask = abs_x <= np.percentile(abs_x, 40.0)
        inferomedial_mask &= local[:, 2] <= np.percentile(local[:, 2], INFERIOR_REGION_PERCENTILE)
        anterior_score = 0.0
        if np.any(inferomedial_mask):
            anterior_score = -local[inferomedial_mask][:, 1].min()

        score = (high_spread - low_spread) + 0.1 * anterior_score
        if score > best_score:
            best_score = score
            best_axes = axes

    best_axes["origin"] = origin
    return best_axes


def score_extremum(local_vertices, score_terms):
    score = np.zeros(len(local_vertices))
    for weight, values in score_terms:
        values = np.asarray(values, dtype=float)
        span = values.max() - values.min()
        if span > 1e-8:
            values = (values - values.min()) / span
        else:
            values = np.zeros_like(values)
        score += weight * values
    return int(np.argmax(score))


def find_asis(hip_vertices, frame, side_sign):
    local = to_local(hip_vertices, frame["origin"], frame)
    lateral = side_sign * local[:, 0]
    medial = -side_sign * local[:, 0]
    anterior = -local[:, 1]

    # Restrict the search below the broad superior crest and favor the
    # anterior-lateral iliac tip.
    mask = local[:, 2] >= np.percentile(local[:, 2], 30.0)
    mask &= local[:, 2] <= np.percentile(local[:, 2], 62.0)
    mask &= anterior >= np.percentile(anterior, 62.0)
    mask &= lateral >= np.percentile(lateral, 40.0)
    mask &= medial <= np.percentile(medial, 85.0)
    if not np.any(mask):
        mask = local[:, 2] >= np.percentile(local[:, 2], 25.0)
        mask &= local[:, 2] <= np.percentile(local[:, 2], 68.0)
        mask &= anterior >= np.percentile(anterior, 58.0)
        mask &= lateral >= np.percentile(lateral, 35.0)
    if not np.any(mask):
        mask = np.ones(len(local), dtype=bool)

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_lateral = lateral[mask]
    candidate_anterior = -candidate_local[:, 1]
    candidate_medial = -side_sign * candidate_local[:, 0]

    target_height = np.percentile(local[:, 2], 42.0)
    height_match = -np.abs(candidate_local[:, 2] - target_height)
    anterior_tip = np.sqrt(
        np.maximum(candidate_anterior - candidate_anterior.min(), 0.0)
        * np.maximum(candidate_lateral - candidate_lateral.min(), 0.0)
    )
    idx = score_extremum(
        candidate_local,
        [
            (0.42, candidate_anterior),
            (0.23, height_match),
            (0.20, candidate_lateral),
            (0.10, anterior_tip),
            (0.05, -candidate_medial),
        ],
    )
    return candidate_vertices[idx]


def find_pt(hip_vertices, frame, side_sign):
    local = to_local(hip_vertices, frame["origin"], frame)
    medial = -side_sign * local[:, 0]
    mask = (-local[:, 2]) >= np.percentile(-local[:, 2], PT_INFERIOR_PERCENTILE)
    mask &= medial >= np.percentile(medial, PT_MEDIAL_PERCENTILE)
    if not np.any(mask):
        mask = np.ones(len(local), dtype=bool)

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_medial = medial[mask]
    idx = score_extremum(
        candidate_local,
        [
            (0.50, -candidate_local[:, 1]),  # anterior
            (0.30, -candidate_local[:, 2]),  # inferior
            (0.20, candidate_medial),        # medial
        ],
    )
    return candidate_vertices[idx]


def find_it(hip_vertices, frame, side_sign):
    local = to_local(hip_vertices, frame["origin"], frame)
    medial = -side_sign * local[:, 0]
    inferior = -local[:, 2]
    posterior = local[:, 1]

    mask = inferior >= np.percentile(inferior, IT_INFERIOR_PERCENTILE)
    mask &= posterior >= np.percentile(posterior, IT_POSTERIOR_PERCENTILE)
    mask &= medial >= np.percentile(medial, IT_MEDIAL_PERCENTILE)
    if not np.any(mask):
        mask = np.ones(len(local), dtype=bool)

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_medial = medial[mask]
    idx = score_extremum(
        candidate_local,
        [
            (0.45, -candidate_local[:, 2]),  # inferior
            (0.35, candidate_local[:, 1]),   # posterior
            (0.20, candidate_medial),        # medial
        ],
    )
    return candidate_vertices[idx]


def find_ipr(hip_vertices, frame, side_sign, pt, it):
    local = to_local(hip_vertices, frame["origin"], frame)
    pt_local = to_local(np.asarray([pt]), frame["origin"], frame)[0]
    it_local = to_local(np.asarray([it]), frame["origin"], frame)[0]

    corridor_center = 0.5 * (pt_local + it_local)
    corridor_vec = it_local - pt_local
    corridor_len_sq = np.dot(corridor_vec, corridor_vec)

    if corridor_len_sq < 1e-8:
        return None

    t = np.clip(((local - pt_local) @ corridor_vec) / corridor_len_sq, 0.0, 1.0)
    closest = pt_local + np.outer(t, corridor_vec)
    dist_to_corridor = np.linalg.norm(local - closest, axis=1)

    medial = -side_sign * local[:, 0]
    inferior = -local[:, 2]
    yz_vec = it_local[1:3] - pt_local[1:3]
    yz_len_sq = np.dot(yz_vec, yz_vec)
    if yz_len_sq < 1e-8:
        return None
    yz_t = np.clip(((local[:, 1:3] - pt_local[1:3]) @ yz_vec) / yz_len_sq, 0.0, 1.0)
    yz_on_line = pt_local[1:3] + np.outer(yz_t, yz_vec)
    z_above_ramus_line = local[:, 2] - yz_on_line[:, 1]

    base_mask = inferior >= np.percentile(inferior, IPR_INFERIOR_PERCENTILE)
    base_mask &= medial >= np.percentile(medial, IPR_MEDIAL_PERCENTILE)
    base_mask &= t >= 0.15
    base_mask &= t <= 0.8
    base_mask &= dist_to_corridor <= np.percentile(dist_to_corridor, 35.0)
    base_mask &= local[:, 2] <= (pt_local[2] + 2.0)

    mask = base_mask & (z_above_ramus_line >= np.percentile(z_above_ramus_line, 55.0))
    if not np.any(mask):
        mask = base_mask & (z_above_ramus_line >= np.percentile(z_above_ramus_line, 45.0))
    if not np.any(mask):
        mask = base_mask

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_t = t[mask]
    candidate_dist = dist_to_corridor[mask]
    candidate_z_above_line = z_above_ramus_line[mask]
    pt_to_it_local_z = max(pt_local[2] - it_local[2], 1e-6)
    superior_from_it = (candidate_local[:, 2] - it_local[2]) / pt_to_it_local_z
    superior_from_it = np.clip(superior_from_it, 0.0, 1.0)
    idx = score_extremum(
        candidate_local,
        [
            (0.35, candidate_z_above_line),              # favor the foramen-facing rim above the ramus line
            (0.25, superior_from_it),                    # keep the point above the IT floor
            (0.20, -candidate_dist),                     # remain near the PT-IT corridor
            (0.10, 1.0 - np.abs(candidate_t - 0.5)),     # avoid drifting too close to PT or IT
            (0.10, -side_sign * candidate_local[:, 0]),  # mild medial bias
        ],
    )
    return candidate_vertices[idx]


def find_psis(hip_vertices, frame, side_sign, asis):
    local = to_local(hip_vertices, frame["origin"], frame)
    asis_local = to_local(np.asarray([asis]), frame["origin"], frame)[0]
    lateral = side_sign * local[:, 0]
    medial = -side_sign * local[:, 0]
    posterior = local[:, 1]
    superior = local[:, 2]

    mask = superior >= np.percentile(superior, PSIS_SUPERIOR_PERCENTILE)
    mask &= posterior >= np.percentile(posterior, PSIS_POSTERIOR_PERCENTILE)
    mask &= posterior >= (asis_local[1] + 5.0)
    mask &= superior >= (asis_local[2] + 5.0)
    mask &= superior <= (asis_local[2] + 55.0)
    mask &= medial >= np.percentile(medial, 20.0)
    mask &= lateral <= np.percentile(lateral, 80.0)
    if not np.any(mask):
        mask = superior >= np.percentile(superior, 55.0)
        mask &= posterior >= np.percentile(posterior, 55.0)
        mask &= superior <= (asis_local[2] + 60.0)
        mask &= lateral <= np.percentile(lateral, 85.0)
    if not np.any(mask):
        return None, None

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_lateral = lateral[mask]
    candidate_medial = medial[mask]
    candidate_posterior = posterior[mask]
    candidate_superior = superior[mask]

    # Across reviewed cases, PSIS lies in a fairly consistent posterior/superior
    # band relative to ASIS in the pelvis-local frame.
    target_height = asis_local[2] + 34.0
    target_posterior = -0.92 * asis_local[1]
    target_x = 0.35 * asis_local[0]
    crest_height_match = -np.abs(candidate_superior - target_height)
    posterior_match = -np.abs(candidate_posterior - target_posterior)
    x_match = -np.abs(candidate_local[:, 0] - target_x)
    idx = score_extremum(
        candidate_local,
        [
            (0.40, posterior_match),
            (0.35, crest_height_match),
            (0.20, x_match),
            (0.05, candidate_medial),
        ],
    )
    psis = candidate_vertices[idx]
    debug_anchor = candidate_vertices.mean(axis=0)
    return psis, debug_anchor


def find_aiis(hip_vertices, frame, side_sign, asis, pt):
    local = to_local(hip_vertices, frame["origin"], frame)
    asis_local = to_local(np.asarray([asis]), frame["origin"], frame)[0]
    pt_local = to_local(np.asarray([pt]), frame["origin"], frame)[0]
    lateral = side_sign * local[:, 0]
    medial = -side_sign * local[:, 0]
    anterior = -local[:, 1]

    # AIIS should lie inferior to ASIS on the anterior ilium, slightly more
    # medial/posterior than the ASIS tip and above the pubic tubercle.
    mask = local[:, 2] <= (asis_local[2] - 18.0)
    mask &= local[:, 2] >= (asis_local[2] - 80.0)
    mask &= anterior >= np.percentile(anterior, 55.0)
    mask &= local[:, 1] >= (asis_local[1] - 25.0)
    mask &= local[:, 1] <= (pt_local[1] + 20.0)
    mask &= medial >= np.percentile(medial, 20.0)
    mask &= lateral >= np.percentile(lateral, 25.0)
    if not np.any(mask):
        mask = local[:, 2] <= (asis_local[2] - 15.0)
        mask &= local[:, 2] >= (asis_local[2] - 80.0)
        mask &= anterior >= np.percentile(anterior, 50.0)
        mask &= lateral >= np.percentile(lateral, 20.0)
    if not np.any(mask):
        return None, None

    candidate_vertices = hip_vertices[mask]
    candidate_local = local[mask]
    candidate_lateral = lateral[mask]
    candidate_medial = medial[mask]
    candidate_anterior = anterior[mask]

    target_height = asis_local[2] - 45.0
    height_match = -np.abs(candidate_local[:, 2] - target_height)
    target_y = asis_local[1] + 8.0
    anterior_offset_match = -np.abs(candidate_local[:, 1] - target_y)
    medial_preference = candidate_medial
    idx = score_extremum(
        candidate_local,
        [
            (0.30, candidate_anterior),
            (0.30, height_match),
            (0.20, medial_preference),
            (0.15, anterior_offset_match),
            (0.05, candidate_lateral),
        ],
    )
    aiis = candidate_vertices[idx]
    debug_anchor = candidate_vertices.mean(axis=0)
    return aiis, debug_anchor


def build_app_frame(asis_left, asis_right, ps, pelvis_frame):
    v1 = asis_left - asis_right
    v2 = ps - asis_right
    normal = normalize(np.cross(v1, v2))
    if normal is None:
        return None

    if np.dot(normal, pelvis_frame["y_axis"]) < 0:
        normal = -normal

    x_axis = normalize(asis_left - asis_right)
    y_axis = normal
    z_axis = normalize(np.cross(x_axis, y_axis))

    return {
        "origin": ps,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "normal": normal,
    }


#: Landmarks this detector is responsible for, with the construction each one actually uses.
#: The provenance strings are deliberately literal: a consumer can see that PEL_PS is a
#: constructed midpoint and that PEL_PSIS_* is a fixed-offset regression on ASIS rather than a
#: detection, which is what the Phase P1 port replaces.
PELVIS_PROVENANCE = {
    "PEL_ASIS_L": ("percentile_band_extremum", 0.5, {}),
    "PEL_ASIS_R": ("percentile_band_extremum", 0.5, {}),
    "PEL_PT_L": ("percentile_band_extremum", 0.5, {}),
    "PEL_PT_R": ("percentile_band_extremum", 0.5, {}),
    "PEL_IT_L": ("percentile_band_extremum", 0.5, {}),
    "PEL_IT_R": ("percentile_band_extremum", 0.5, {}),
    "PEL_IPR_L": ("percentile_band_extremum", 0.2, {"validated": False}),
    "PEL_IPR_R": ("percentile_band_extremum", 0.2, {"validated": False}),
    "PEL_AIIS_L": ("asis_offset_prior", 0.3, {"population_prior": True}),
    "PEL_AIIS_R": ("asis_offset_prior", 0.3, {"population_prior": True}),
    "PEL_PSIS_L": ("asis_offset_prior", 0.3, {"population_prior": True}),
    "PEL_PSIS_R": ("asis_offset_prior", 0.3, {"population_prior": True}),
    "PEL_PS": ("pubic_tubercle_midpoint", 0.4, {"constructed": True}),
    "PEL_COM": ("hip_centroid_midpoint", 0.6, {"mesh_density_weighted": True}),
}


def _pelvis_records(landmarks):
    """A status record for every landmark this detector owns, detected or not."""
    records = {}
    for landmark_id, (provenance, confidence, metrics) in PELVIS_PROVENANCE.items():
        position = landmarks.get(landmark_id)
        if position is None:
            records[landmark_id] = status.abstained(
                landmark_id, status.Reason.EMPTY_REGION, provenance=provenance, metrics=metrics
            )
        else:
            records[landmark_id] = status.ok(
                landmark_id, position, confidence=confidence, provenance=provenance,
                metrics=metrics,
            )
    return records


def _all_pelvis_not_attempted(reason):
    return {
        landmark_id: status.not_attempted(landmark_id, reason, provenance=provenance)
        for landmark_id, (provenance, _, _) in PELVIS_PROVENANCE.items()
    }


def detect_pelvis_landmarks(stl_path, left_hip_path, right_hip_path):

    left_hip_vertices = load_hip_model(left_hip_path)
    right_hip_vertices = load_hip_model(right_hip_path)

    if left_hip_vertices is None or right_hip_vertices is None:
        return {}, {"records": _all_pelvis_not_attempted(status.Reason.MISSING_INPUT)}

    if stl_path is not None:
        stl_mesh = mesh.Mesh.from_file(stl_path)
        vertices = np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0)
    else:
        vertices = np.unique(np.vstack([left_hip_vertices, right_hip_vertices]), axis=0)
        print("  Using merged individual hip meshes as combined pelvis point cloud")

    pelvis_frame = build_pelvis_frame(vertices, left_hip_vertices, right_hip_vertices)

    asis_left = find_asis(left_hip_vertices, pelvis_frame, side_sign=1.0)
    asis_right = find_asis(right_hip_vertices, pelvis_frame, side_sign=-1.0)
    pt_left = find_pt(left_hip_vertices, pelvis_frame, side_sign=1.0)
    pt_right = find_pt(right_hip_vertices, pelvis_frame, side_sign=-1.0)
    it_left = find_it(left_hip_vertices, pelvis_frame, side_sign=1.0)
    it_right = find_it(right_hip_vertices, pelvis_frame, side_sign=-1.0)
    if pt_left is None or it_left is None:
        print("  [WARN] PT_L or IT_L missing — PEL_IPR_L skipped")
        ipr_left = None
    else:
        ipr_left = find_ipr(left_hip_vertices, pelvis_frame, side_sign=1.0, pt=pt_left, it=it_left)

    if pt_right is None or it_right is None:
        print("  [WARN] PT_R or IT_R missing — PEL_IPR_R skipped")
        ipr_right = None
    else:
        ipr_right = find_ipr(right_hip_vertices, pelvis_frame, side_sign=-1.0, pt=pt_right, it=it_right)

    if asis_left is None:
        print("  [WARN] ASIS_L missing — PEL_PSIS_L and PEL_AIIS_L skipped")
        psis_left, psis_left_dbg = None, None
        aiis_left, aiis_left_dbg = None, None
    else:
        psis_left, psis_left_dbg = find_psis(left_hip_vertices, pelvis_frame, side_sign=1.0, asis=asis_left)
        if pt_left is None:
            print("  [WARN] PT_L missing — PEL_AIIS_L skipped")
            aiis_left, aiis_left_dbg = None, None
        else:
            aiis_left, aiis_left_dbg = find_aiis(left_hip_vertices, pelvis_frame, side_sign=1.0, asis=asis_left, pt=pt_left)

    if asis_right is None:
        print("  [WARN] ASIS_R missing — PEL_PSIS_R and PEL_AIIS_R skipped")
        psis_right, psis_right_dbg = None, None
        aiis_right, aiis_right_dbg = None, None
    else:
        psis_right, psis_right_dbg = find_psis(right_hip_vertices, pelvis_frame, side_sign=-1.0, asis=asis_right)
        if pt_right is None:
            print("  [WARN] PT_R missing — PEL_AIIS_R skipped")
            aiis_right, aiis_right_dbg = None, None
        else:
            aiis_right, aiis_right_dbg = find_aiis(right_hip_vertices, pelvis_frame, side_sign=-1.0, asis=asis_right, pt=pt_right)

    if pt_left is None or pt_right is None:
        print("  [WARN] PT landmark missing for one or both sides — PEL_PS skipped")
        ps = None
    else:
        ps = 0.5 * (pt_left + pt_right)

    pel_com = 0.5 * (left_hip_vertices.mean(axis=0) + right_hip_vertices.mean(axis=0))

    landmarks = {
        "PEL_ASIS_L": asis_left,
        "PEL_ASIS_R": asis_right,
        "PEL_PT_L": pt_left,
        "PEL_PT_R": pt_right,
        "PEL_IT_L": it_left,
        "PEL_IT_R": it_right,
        "PEL_IPR_L": ipr_left,
        "PEL_IPR_R": ipr_right,
        "PEL_AIIS_L": aiis_left,
        "PEL_AIIS_R": aiis_right,
        "PEL_PSIS_L": psis_left,
        "PEL_PSIS_R": psis_right,
        "PEL_PS": ps,
        "PEL_COM": pel_com,
        "DBG_PELV_ORIGIN": pelvis_frame["origin"],
        "DBG_PELV_LHIP_C": left_hip_vertices.mean(axis=0),
        "DBG_PELV_RHIP_C": right_hip_vertices.mean(axis=0),
    }
    if psis_left_dbg is not None:
        landmarks["DBG_PSIS_PATCH_L"] = psis_left_dbg
    if psis_right_dbg is not None:
        landmarks["DBG_PSIS_PATCH_R"] = psis_right_dbg
    if aiis_left_dbg is not None:
        landmarks["DBG_AIIS_PATCH_L"] = aiis_left_dbg
    if aiis_right_dbg is not None:
        landmarks["DBG_AIIS_PATCH_R"] = aiis_right_dbg

    if asis_left is None or asis_right is None or ps is None:
        print("  [WARN] ASIS or PS missing — APP frame skipped")
        app_frame = None
    else:
        app_frame = build_app_frame(asis_left, asis_right, ps, pelvis_frame)
    if app_frame is not None:
        landmarks["DBG_APP_ORIGIN"] = app_frame["origin"]

    debug_data = {
        "pelvis_frame": {
            "origin": pelvis_frame["origin"].tolist(),
            "x_axis": pelvis_frame["x_axis"].tolist(),
            "y_axis": pelvis_frame["y_axis"].tolist(),
            "z_axis": pelvis_frame["z_axis"].tolist(),
        }
    }

    if app_frame is not None:
        debug_data["APP_plane"] = {
            "origin": app_frame["origin"].tolist(),
            "normal": app_frame["normal"].tolist(),
            "x_axis": app_frame["x_axis"].tolist(),
            "y_axis": app_frame["y_axis"].tolist(),
            "z_axis": app_frame["z_axis"].tolist(),
        }

    debug_data["records"] = _pelvis_records(landmarks)

    return landmarks, debug_data
