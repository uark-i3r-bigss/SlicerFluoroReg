import os

import numpy as np
from stl import mesh

from rng import rng_for

from . import status

# Core algorithm parameters
CROP_DIST_MM = 120.0
TARGET_RAD = 25.0
INLIER_THRESH = 2.0
MAX_ITER = 2000
MIN_INLIERS = 4
MIN_SCORE_THRESH = 0.1
RAD_MIN = 12.0
RAD_MAX = 40.0

# Quality control parameters
DUP_CENTER_THRESH = 5.0
MAX_CANDIDATES = 10
MIN_SAMPLE_DIST = 5.0

# Scoring parameters
RAD_WEIGHT = 0.2
POS_WEIGHT = 0.2
CONF_WEIGHT = 0.6
RAD_TOLERANCE = 8.0
PENALTY_DIV = 20.0
MIN_SCORE_FLOOR = 0.2

# Femur-frame parameters
END_PERCENTILE = 10.0
HEAD_EXCLUSION_MULT = 1.15
GT_SUPERIOR_LATERAL_PERCENTILE = 65.0
PROXIMAL_SHAFT_BAND_MM = (35.0, 85.0)
LT_Z_BAND_MM = (25.0, 135.0)
LT_GT_EXCLUSION_MM = 22.0
LT_MEDIAL_PERCENTILE = 60.0

# Trochanter-frame axis: the line through cross-section circle centres of the diaphysis just distal to
# the lesser trochanter, a window every femur longer than about 160 mm contains in full. The
# whole-mesh PCA axis leans toward the neck by an amount that depends on how much shaft the scan
# holds (a 441 mm femur cut to 140-180 mm tilts it 11-18 degrees and moves the greater-trochanter
# extrema 11-28 mm), so it remains only the head-search axis it was validated for.
DIAPHYSIS_WINDOW_MM = (100.0, 150.0)  # below the most proximal vertex along the axis
DIAPHYSIS_DISTAL_MARGIN_MM = 10.0  # keep clear of a field-of-view cut face
DIAPHYSIS_MIN_LENGTH_MM = 30.0
DIAPHYSIS_SLICE_MM = 3.0
DIAPHYSIS_MIN_SLICE_VERTICES = 20
DIAPHYSIS_MIN_SLICES = 5
DIAPHYSIS_ITERATIONS = 3
SHORT_SHAFT_CONFIDENCE = 0.25


def normalize(vec):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return None
    return vec / norm


def project_onto_plane(vec, normal):
    return vec - np.dot(vec, normal) * normal


def to_local(points, origin, axes):
    points = np.asarray(points)
    rel = points - origin
    return np.column_stack([rel @ axes["x_axis"], rel @ axes["y_axis"], rel @ axes["z_axis"]])


def fit_sphere(points):
    if len(points) < 4:
        return None, None

    a = np.zeros((len(points), 4))
    b = np.zeros(len(points))

    for i, point in enumerate(points):
        x, y, z = point
        a[i] = [2 * x, 2 * y, 2 * z, -1]
        b[i] = x * x + y * y + z * z

    try:
        params = np.linalg.lstsq(a, b, rcond=None)[0]
        center = params[:3]
        d = params[3]

        radius_squared = np.dot(center, center) - d
        if radius_squared <= 0:
            return None, None

        radius = np.sqrt(radius_squared)
        return center, radius
    except (np.linalg.LinAlgError, ValueError):
        return None, None


def count_inliers(points, center, radius, threshold):
    distances_to_center = np.linalg.norm(points - center, axis=1)
    distances_to_surface = np.abs(distances_to_center - radius)
    inlier_mask = distances_to_surface <= threshold
    inlier_idx = np.where(inlier_mask)[0]
    return inlier_idx, len(inlier_idx)


# --- Femoral head: region isolation and guarded fit -------------------------------
#
# Re-derived from the published descriptions, not from the reference sources (see
# PROVENANCE.md): every reference fits the head sphere by deterministic least squares to a
# *curated head region* (Cerveri, Ann Biomed Eng 38:1752, 2010, "least squares sphere
# fitting"; Fischer, Sci Rep 10:20859, 2020, fits the atlas-mapped head area once; the GIBOC
# and Kai 2014 variants fit the isolated proximal head patch and warn above an RMSE
# threshold).  The previous implementation instead drew 2000 random 4-point samples from the
# whole 120 mm proximal crop, which keeps the greater trochanter in the candidate set and
# makes "confidence = inliers / len(points)" meaningless.

#: Keep vertices at least this fraction of the maximal off-axis offset away from the shaft
#: axis; the head is the strongly off-axis proximal structure.
HEAD_OFFSET_FRACTION = 0.35
#: Trimming band around the current sphere, same tolerance as the previous INLIER_THRESH.
HEAD_TRIM_ITERATIONS = 5
#: Fischer 2020 reports a femoral-head-centre agreement of 0.4 mm; an RMS far above the voxel
#: scale means the region is not a sphere.  1.5 mm degrades, 3.0 mm rejects.
HEAD_RMS_LOW_CONFIDENCE_MM = 1.5
HEAD_RMS_REJECT_MM = 3.0
#: Adult femoral-head radius prior.  [12, 40] mm stays the hard reject interval (unchanged
#: from RAD_MIN/RAD_MAX); [18, 32] mm is the confident band.
HEAD_RADIUS_CONFIDENT_MM = (18.0, 32.0)
#: Below this inlier fraction of the isolated head region, seed the fit from the RANSAC rescue
#: before deciding.  The fraction is reported but does not gate acceptance: the region also
#: holds the neck and the medial proximal shaft, whose share depends on the femur and its
#: segmentation and proximal morphology.
HEAD_RESCUE_INLIER_FRACTION = 0.50
#: Sphere support: share of HEAD_SUPPORT_DIRECTIONS near-uniform directions from the fitted
#: centre (Fibonacci lattice, cells of about 9 degrees, 4 mm at a 25 mm radius) that hold at
#: least one inlier.  A femoral head is about two thirds of a sphere; a sphere fitted to a neck
#: ring or a trochanter patch is backed by far less. Support complements the residual, coverage
#: and bootstrap-spread checks, which can all look acceptable for a sphere fitted off the head.
HEAD_SUPPORT_DIRECTIONS = 500
HEAD_SUPPORT_MIN = 0.50
#: Angular coverage.  For points spread uniformly over a spherical cap of half-angle theta,
#: ||mean unit vector|| = (1 + cos theta) / 2; 0.80 corresponds to theta = 53 degrees.  Less
#: coverage than that leaves the centre ill-conditioned along the cap axis.
HEAD_COVERAGE_MAX = 0.80
#: Bootstrap over the inlier set, reported as the per-landmark uncertainty.
HEAD_BOOTSTRAP_SAMPLES = 20
HEAD_BOOTSTRAP_FRACTION = 0.8
HEAD_CENTRE_SPREAD_MAX_MM = 1.0


def isolate_head_region(vertices, origin, z_axis):
    """Vertices of the proximal crop that belong to the femoral head.

    The medial direction is taken from the most proximal patch of the bone: its centroid is
    offset from the shaft axis towards the head, so the component of that offset orthogonal
    to the shaft axis points medially and separates the head from the greater trochanter.
    """
    relative = vertices - origin
    along = relative @ z_axis
    perpendicular = relative - np.outer(along, z_axis)
    offsets = np.linalg.norm(perpendicular, axis=1)
    if len(vertices) < 4 or offsets.max() <= 0:
        return np.empty((0, 3)), {}

    proximal_patch = vertices[along >= np.percentile(along, 95.0)]
    patch_offset = proximal_patch.mean(axis=0) - origin
    medial = patch_offset - np.dot(patch_offset, z_axis) * z_axis
    medial = normalize(medial)
    if medial is None:
        return np.empty((0, 3)), {}

    mask = offsets >= HEAD_OFFSET_FRACTION * offsets.max()
    mask &= (perpendicular @ medial) > 0.0
    return vertices[mask], {
        "region_vertices": int(mask.sum()),
        "max_offset_mm": float(offsets.max()),
        "medial_direction": [float(v) for v in medial],
    }


def sphere_residual(points, center, radius):
    return np.linalg.norm(points - center, axis=1) - radius


def angular_coverage(points, center):
    """``||mean unit vector||``: 0 for a full sphere, 0.5 for a hemisphere, 1 for a point."""
    directions = points - center
    norms = np.linalg.norm(directions, axis=1)
    usable = norms > 1e-9
    if not np.any(usable):
        return 1.0
    unit = directions[usable] / norms[usable, None]
    return float(np.linalg.norm(unit.mean(axis=0)))


def _fibonacci_directions(count):
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + np.sqrt(5.0)) * index
    return np.column_stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar),
                            np.cos(polar)])


SUPPORT_LATTICE = _fibonacci_directions(HEAD_SUPPORT_DIRECTIONS)


def sphere_support(points, center):
    """Share of the sphere's directions backed by a point: 1 for a full sphere, 0.5 a hemisphere."""
    directions = points - center
    norms = np.linalg.norm(directions, axis=1)
    usable = norms > 1e-9
    if not np.any(usable):
        return 0.0
    unit = directions[usable] / norms[usable, None]
    occupied = np.zeros(len(SUPPORT_LATTICE), dtype=bool)
    for start in range(0, len(unit), 8192):
        occupied[np.argmax(unit[start:start + 8192] @ SUPPORT_LATTICE.T, axis=1)] = True
    return float(occupied.mean())


def trimmed_sphere_fit(points, seed_center=None, seed_radius=None):
    """Algebraic seed, then bounded iteratively-trimmed least squares.

    Deterministic: the inlier set is decided by the current residuals, not by sampling.
    """
    if len(points) < 4:
        return None

    center, radius = (seed_center, seed_radius)
    if center is None or radius is None:
        center, radius = fit_sphere(points)
        if center is None:
            return None

    inliers = np.arange(len(points))
    for _ in range(HEAD_TRIM_ITERATIONS):
        keep = np.where(np.abs(sphere_residual(points, center, radius)) <= INLIER_THRESH)[0]
        if len(keep) < 4:
            break
        refit_center, refit_radius = fit_sphere(points[keep])
        if refit_center is None:
            break
        if len(keep) == len(inliers) and np.array_equal(keep, inliers):
            center, radius = refit_center, refit_radius
            break
        center, radius, inliers = refit_center, refit_radius, keep

    residual = sphere_residual(points[inliers], center, radius)
    return {
        "center": np.asarray(center, dtype=float),
        "radius": float(radius),
        "inliers": inliers,
        "rms_mm": float(np.sqrt((residual ** 2).mean())),
        "inlier_fraction": float(len(inliers) / len(points)),
        "coverage": angular_coverage(points[inliers], center),
    }


def bootstrap_centre_spread(points, rng):
    """Maximal centre displacement over resampled fits; the per-landmark uncertainty."""
    if len(points) < 8:
        return None
    size = max(4, int(round(HEAD_BOOTSTRAP_FRACTION * len(points))))
    centres = []
    for _ in range(HEAD_BOOTSTRAP_SAMPLES):
        sample = points[rng.choice(len(points), size, replace=False)]
        center, radius = fit_sphere(sample)
        if center is not None:
            centres.append(center)
    if len(centres) < 2:
        return None
    centres = np.asarray(centres)
    return float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max())

def fh_score(radius, inlier_ratio, center_local_z, max_local_z):
    rad_err = abs(radius - TARGET_RAD)
    if rad_err <= RAD_TOLERANCE:
        rad_score = 1.0 - rad_err / PENALTY_DIV
    else:
        rad_score = max(MIN_SCORE_FLOOR, 1.0 - rad_err / PENALTY_DIV)

    z_dist_top = max_local_z - center_local_z
    ideal_dist = TARGET_RAD
    pos_err = abs(z_dist_top - ideal_dist)
    pos_score = max(MIN_SCORE_FLOOR, 1.0 - pos_err / PENALTY_DIV)

    return RAD_WEIGHT * rad_score + POS_WEIGHT * pos_score + CONF_WEIGHT * inlier_ratio


def ransac_fh(points, origin, z_axis, rng):
    if len(points) < 4:
        return []

    candidates = []
    max_local_z = np.max((points - origin) @ z_axis)

    for iteration in range(MAX_ITER):
        sample_indices = rng.choice(len(points), 4, replace=False)
        sample_points = points[sample_indices]

        min_distance = float("inf")
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(sample_points[i] - sample_points[j])
                min_distance = min(min_distance, dist)

        if min_distance < MIN_SAMPLE_DIST:
            continue

        center, radius = fit_sphere(sample_points)
        if center is None or radius is None:
            continue

        if radius < RAD_MIN or radius > RAD_MAX:
            continue

        inlier_indices, inlier_count = count_inliers(points, center, radius, INLIER_THRESH)
        if inlier_count < MIN_INLIERS:
            continue

        confidence = inlier_count / len(points)
        center_local_z = np.dot(center - origin, z_axis)
        score = fh_score(radius, confidence, center_local_z, max_local_z)

        candidates.append(
            {
                "center": center,
                "radius": radius,
                "inliers": inlier_indices,
                "confidence": confidence,
                "score": score,
                "center_local_z": center_local_z,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    filtered_candidates = []
    for candidate in candidates:
        is_duplicate = False
        for existing in filtered_candidates:
            center_distance = np.linalg.norm(candidate["center"] - existing["center"])
            if center_distance < DUP_CENTER_THRESH:
                is_duplicate = True
                break

        if not is_duplicate:
            filtered_candidates.append(candidate)

        if len(filtered_candidates) >= MAX_CANDIDATES:
            break

    return filtered_candidates


def detect_fh(verts, origin, z_axis, rng, landmark_id="FEM_FH"):
    """Fit the femoral head centre and decide whether the fit may be used.

    Returns ``(fit, record)``.  ``fit`` is ``None`` whenever the record abstains, so a
    rejected fit cannot reach the FCSV - and, because the pelvis copies its acetabular
    centres verbatim from this landmark, it removes PEL_FH_L/R with it.
    """
    region, region_info = isolate_head_region(verts, origin, z_axis)
    if len(region) < 4:
        return None, status.abstained(landmark_id, status.Reason.EMPTY_REGION,
                                      provenance="head_region", metrics=region_info)

    fit = trimmed_sphere_fit(region)
    if fit is None:
        return None, status.abstained(landmark_id, status.Reason.EMPTY_REGION,
                                      provenance="trimmed_sphere_fit", metrics=region_info)

    rescued = False
    if fit["inlier_fraction"] < HEAD_RESCUE_INLIER_FRACTION:
        # Outlier rescue only: seed the same deterministic fit from the best RANSAC sphere.
        candidates = ransac_fh(region, origin, z_axis, rng)
        if candidates:
            seeded = trimmed_sphere_fit(region, candidates[0]["center"], candidates[0]["radius"])
            if seeded is not None and seeded["inlier_fraction"] > fit["inlier_fraction"]:
                fit, rescued = seeded, True

    spread = bootstrap_centre_spread(region[fit["inliers"]], rng)
    fit["sphere_support"] = sphere_support(region[fit["inliers"]], fit["center"])
    metrics = dict(region_info)
    metrics.update({
        "radius_mm": fit["radius"],
        "rms_mm": fit["rms_mm"],
        "inlier_fraction": fit["inlier_fraction"],
        "sphere_support": fit["sphere_support"],
        "coverage": fit["coverage"],
        "centre_spread_mm": spread,
        "ransac_rescue": rescued,
    })

    reasons = []
    if not (RAD_MIN <= fit["radius"] <= RAD_MAX):
        reasons.append(status.Reason.OUT_OF_PRIOR)
    if fit["rms_mm"] > HEAD_RMS_REJECT_MM:
        reasons.append(status.Reason.RESIDUAL_TOO_HIGH)
    if fit["sphere_support"] < HEAD_SUPPORT_MIN:
        reasons.append(status.Reason.INSUFFICIENT_SUPPORT)
    if fit["coverage"] > HEAD_COVERAGE_MAX:
        reasons.append(status.Reason.POOR_COVERAGE)
    if spread is not None and spread > HEAD_CENTRE_SPREAD_MAX_MM:
        reasons.append(status.Reason.UNSTABLE)
    if reasons:
        return None, status.abstained(landmark_id, reasons[0], provenance="sphere_fit",
                                      support=len(fit["inliers"]), metrics=metrics)

    confidence = float(np.clip(1.0 - fit["rms_mm"] / HEAD_RMS_REJECT_MM, 0.0, 1.0)
                       * np.clip(fit["sphere_support"] / HEAD_SUPPORT_MIN, 0.0, 1.0))
    degraded = (fit["rms_mm"] > HEAD_RMS_LOW_CONFIDENCE_MM
                or not (HEAD_RADIUS_CONFIDENT_MM[0] <= fit["radius"] <= HEAD_RADIUS_CONFIDENT_MM[1]))
    record_factory = status.low_confidence if degraded else status.ok
    kwargs = {"provenance": "sphere_fit", "residual_mm": fit["rms_mm"],
              "support": len(fit["inliers"]), "metrics": metrics}
    if degraded:
        record = record_factory(landmark_id, fit["center"], confidence=confidence,
                                reason=status.Reason.OUT_OF_PRIOR
                                if not (HEAD_RADIUS_CONFIDENT_MM[0] <= fit["radius"]
                                        <= HEAD_RADIUS_CONFIDENT_MM[1])
                                else status.Reason.RESIDUAL_TOO_HIGH, **kwargs)
    else:
        record = record_factory(landmark_id, fit["center"], confidence=confidence, **kwargs)
    return fit, record


def estimate_femur_shaft_axis(vertices):
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = normalize(vh[0])

    projections = centered @ axis
    distal_mask = projections <= np.percentile(projections, END_PERCENTILE)
    proximal_mask = projections >= np.percentile(projections, 100.0 - END_PERCENTILE)

    distal_spread = np.mean(
        np.linalg.norm(centered[distal_mask] - np.outer(projections[distal_mask], axis), axis=1)
    )
    proximal_spread = np.mean(
        np.linalg.norm(centered[proximal_mask] - np.outer(projections[proximal_mask], axis), axis=1)
    )

    if proximal_spread < distal_spread:
        axis = -axis
        projections = -projections
        distal_spread, proximal_spread = proximal_spread, distal_spread

    return axis, projections, {
        "distal_spread": distal_spread,
        "proximal_spread": proximal_spread,
    }


def slice_circle_centre(points, axis, level):
    """Centre of the algebraic (Kasa) circle through one slice of a tube, as a 3D point at
    ``level`` along ``axis``. Unlike the vertex centroid it does not follow the vertex density,
    which a voxel staircase makes uneven around the ring."""
    e1 = normalize(project_onto_plane(np.eye(3)[np.argmin(np.abs(axis))], axis))
    e2 = np.cross(axis, e1)
    uv = np.column_stack([points @ e1, points @ e2])
    design = np.column_stack([2.0 * uv, np.ones(len(uv))])
    solution, *_ = np.linalg.lstsq(design, (uv ** 2).sum(axis=1), rcond=None)
    return solution[0] * e1 + solution[1] * e2 + level * axis


def estimate_diaphysis_axis(vertices, axis):
    """Refine a proximally pointing shaft axis to the diaphysis centre line.

    Slices perpendicular to the current axis inside ``DIAPHYSIS_WINDOW_MM`` below the most
    proximal vertex; the axis becomes the principal direction of the slice circle centres, and
    the slicing repeats along it. Returns ``(axis, window_mm)``; ``axis`` is ``None`` when the
    mesh ends before the window holds ``DIAPHYSIS_MIN_LENGTH_MM`` of shaft.
    """
    window_mm = 0.0
    for _ in range(DIAPHYSIS_ITERATIONS):
        t = vertices @ axis
        high = t.max() - DIAPHYSIS_WINDOW_MM[0]
        low = max(t.max() - DIAPHYSIS_WINDOW_MM[1], t.min() + DIAPHYSIS_DISTAL_MARGIN_MM)
        window_mm = max(float(high - low), 0.0)
        if window_mm < DIAPHYSIS_MIN_LENGTH_MM:
            return None, window_mm
        centres = []
        for start in np.arange(low, high, DIAPHYSIS_SLICE_MM):
            in_slice = (t >= start) & (t < start + DIAPHYSIS_SLICE_MM)
            if np.count_nonzero(in_slice) >= DIAPHYSIS_MIN_SLICE_VERTICES:
                centres.append(slice_circle_centre(vertices[in_slice], axis,
                                                   start + DIAPHYSIS_SLICE_MM / 2))
        if len(centres) < DIAPHYSIS_MIN_SLICES:
            return None, window_mm
        centres = np.asarray(centres)
        _, _, vh = np.linalg.svd(centres - centres.mean(axis=0), full_matrices=False)
        axis = vh[0] if vh[0] @ axis > 0 else -vh[0]
    return normalize(axis), window_mm


def build_initial_femur_frame(vertices):
    centered = vertices - vertices.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    z_axis, projections, spread = estimate_femur_shaft_axis(vertices)

    x_candidate = project_onto_plane(vh[1], z_axis)
    x_axis = normalize(x_candidate)
    if x_axis is None:
        x_candidate = project_onto_plane(vh[2], z_axis)
        x_axis = normalize(x_candidate)
    if x_axis is None:
        raise ValueError("Unable to build initial femur frame")

    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    return {
        "origin": vertices.mean(axis=0),
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "projections": projections,
        "spread": spread,
    }


def estimate_proximal_shaft_center(vertices, z_axis):
    centered = vertices - vertices.mean(axis=0)
    projections = centered @ z_axis
    z_max = projections.max()
    band_low = z_max - PROXIMAL_SHAFT_BAND_MM[1]
    band_high = z_max - PROXIMAL_SHAFT_BAND_MM[0]
    mask = (projections >= band_low) & (projections <= band_high)
    if not np.any(mask):
        mask = projections >= np.percentile(projections, 60.0)
    return vertices[mask].mean(axis=0)


def crop_femur_proximal(vertices, frame):
    local = to_local(vertices, frame["origin"], frame)
    z_max_full = local[:, 2].max()
    z_min_full = local[:, 2].min()
    mask = local[:, 2] > z_max_full - CROP_DIST_MM
    filtered_vertices = vertices[mask]
    filtered_local = local[mask]

    print(
        f"  Femur local Z-range: [{z_min_full:.1f}, {z_max_full:.1f}], "
        f"range={z_max_full - z_min_full:.1f}mm"
    )
    print(
        f"  Proximal {CROP_DIST_MM}mm threshold: {z_max_full - CROP_DIST_MM:.1f}, "
        f"filtered local Z-range: [{filtered_local[:, 2].min():.1f}, {filtered_local[:, 2].max():.1f}]"
    )
    print(f"  Proximal crop: {len(vertices)} -> {len(filtered_vertices)} vertices")

    return filtered_vertices, filtered_local


def build_femur_frame(fh_center, shaft_center, shaft_axis):
    z_axis = normalize(shaft_axis)
    if z_axis is None:
        return None

    head_direction = fh_center - shaft_center
    medial_axis = normalize(project_onto_plane(head_direction, z_axis))
    if medial_axis is None:
        return None

    x_axis = medial_axis
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    if x_axis is None or y_axis is None:
        return None

    return {
        "origin": fh_center,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "shaft_center": shaft_center,
    }


def detect_trochanter_landmarks(vertices, frame, fh_center, fh_radius, is_left_femur=True):
    if len(vertices) == 0:
        return {}

    distances_to_head = np.linalg.norm(vertices - fh_center, axis=1)
    local = to_local(vertices, frame["origin"], frame)

    candidate_mask = distances_to_head > fh_radius * HEAD_EXCLUSION_MULT
    candidate_mask &= local[:, 2] > local[:, 2].max() - CROP_DIST_MM
    candidate_vertices = vertices[candidate_mask]
    candidate_local = local[candidate_mask]

    if len(candidate_vertices) == 0:
        return {}

    gt_t_idx = np.argmin(candidate_local[:, 0])
    gt_t = candidate_vertices[gt_t_idx]

    lateral_threshold = np.percentile(candidate_local[:, 0], 100.0 - GT_SUPERIOR_LATERAL_PERCENTILE)
    superior_mask = candidate_local[:, 0] <= lateral_threshold
    superior_vertices = candidate_vertices[superior_mask]
    superior_local = candidate_local[superior_mask]

    gt_s_idx = np.argmax(superior_local[:, 2])
    gt_s = superior_vertices[gt_s_idx]

    suffix = "L" if is_left_femur else "R"
    return {
        f"FEM_{suffix}_FT_S": gt_s,
        f"FEM_{suffix}_FT_T": gt_t,
    }


def detect_lesser_trochanter_landmarks(vertices, frame, fh_center, fh_radius, gt_s, gt_t, is_left_femur=True):
    if len(vertices) == 0:
        return {}

    local = to_local(vertices, frame["origin"], frame)
    distances_to_head = np.linalg.norm(vertices - fh_center, axis=1)
    distances_to_gt_s = np.linalg.norm(vertices - gt_s, axis=1)
    distances_to_gt_t = np.linalg.norm(vertices - gt_t, axis=1)

    z_max = local[:, 2].max()
    z_low = z_max - LT_Z_BAND_MM[1]
    z_high = z_max - LT_Z_BAND_MM[0]

    candidate_mask = distances_to_head > fh_radius * HEAD_EXCLUSION_MULT
    candidate_mask &= distances_to_gt_s > LT_GT_EXCLUSION_MM
    candidate_mask &= distances_to_gt_t > LT_GT_EXCLUSION_MM
    candidate_mask &= local[:, 2] >= z_low
    candidate_mask &= local[:, 2] <= z_high

    candidate_vertices = vertices[candidate_mask]
    candidate_local = local[candidate_mask]
    if len(candidate_vertices) == 0:
        return {}

    medial_threshold = np.percentile(candidate_local[:, 0], LT_MEDIAL_PERCENTILE)
    medial_mask = candidate_local[:, 0] >= medial_threshold
    medial_vertices = candidate_vertices[medial_mask]
    medial_local = candidate_local[medial_mask]
    if len(medial_vertices) == 0:
        return {}

    press_idx = np.argmax(np.abs(medial_local[:, 1]))
    lt_press = medial_vertices[press_idx]

    suffix = "L" if is_left_femur else "R"
    return {
        f"FEM_{suffix}_LT": lt_press,
    }


def load_femur_vertices(femur_path):
    if femur_path and os.path.exists(femur_path):
        print(f"  Loading femur from: {femur_path}")
        femur_mesh = mesh.Mesh.from_file(femur_path)
        vertices = np.unique(femur_mesh.vectors.reshape(-1, 3), axis=0)
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        z_range = z_max - z_min
        print(
            f"  Loaded {os.path.basename(femur_path)}: {len(vertices)} vertices, "
            f"Z-range: [{z_min:.1f}, {z_max:.1f}] = {z_range:.1f}mm"
        )
        return vertices

    print(f"  Femur file not found: {femur_path}")
    return None


def detect_single_femur_landmarks(vertices_full, is_left_femur=True, rng=None):
    """Detect one femur's landmarks.

    Returns ``(landmarks, debug_info)``; ``debug_info["records"]`` carries a status record for
    every landmark that was attempted, including the ones that abstained.
    """
    # An explicit generator keeps the fit reproducible; see rng.rng_for.
    rng = rng_for("unspecified-case", "femur") if rng is None else rng
    suffix = "L" if is_left_femur else "R"
    side = "Left" if is_left_femur else "Right"
    landmarks = {}
    records = {}
    debug_info = {"records": records}

    initial_frame = build_initial_femur_frame(vertices_full)

    proximal_vertices, _ = crop_femur_proximal(vertices_full, initial_frame)
    if len(proximal_vertices) < 4:
        records[f"FEM_{suffix}_FH"] = status.abstained(
            f"FEM_{suffix}_FH", status.Reason.EMPTY_REGION, provenance="proximal_crop"
        )
        return landmarks, debug_info

    z_axis = initial_frame["z_axis"]
    origin = initial_frame["origin"]

    fh, fh_record = detect_fh(proximal_vertices, origin, z_axis, rng,
                              landmark_id=f"FEM_{suffix}_FH")
    records[fh_record.landmark_id] = fh_record
    if fh is None:
        print(f"  {side} femur head: abstained ({fh_record.reason})")
        return landmarks, debug_info

    landmarks[f"FEM_{suffix}_FH"] = fh["center"]
    debug_info[f"FEM_{suffix}_FH"] = {
        "radius": fh["radius"],
        "rms_mm": fh["rms_mm"],
        "inlier_fraction": fh["inlier_fraction"],
        "sphere_support": fh["sphere_support"],
        "coverage": fh["coverage"],
        "centre_spread_mm": fh_record.metrics.get("centre_spread_mm"),
        "confidence": fh_record.confidence,
        "status": fh_record.status,
    }
    print(
        f"  {side} femur head: radius={fh['radius']:.1f}mm, rms={fh['rms_mm']:.2f}mm, "
        f"inliers={fh['inlier_fraction']:.0%}, support={fh['sphere_support']:.2f}, "
        f"coverage={fh['coverage']:.2f}, status={fh_record.status}"
    )

    # The head search above keeps the whole-mesh PCA axis; the trochanter frame uses the diaphysis.
    frame_axis, diaphysis_mm = estimate_diaphysis_axis(vertices_full, z_axis)
    short_shaft = frame_axis is None
    if short_shaft:
        frame_axis = z_axis
    shaft_center = estimate_proximal_shaft_center(vertices_full, frame_axis)
    femur_frame = build_femur_frame(fh["center"], shaft_center, frame_axis)
    trochanter_ids = [f"FEM_{suffix}_FT_S", f"FEM_{suffix}_FT_T", f"FEM_{suffix}_LT"]
    if not femur_frame:
        for landmark_id in trochanter_ids:
            records[landmark_id] = status.abstained(
                landmark_id, status.Reason.FRAME_UNRESOLVED, provenance="femur_frame"
            )
        return landmarks, debug_info

    femur_frame["axis_source"] = "whole_mesh_pca" if short_shaft else "diaphysis"
    femur_frame["diaphysis_window_mm"] = round(diaphysis_mm, 1)
    femur_frame["pca_axis_angle_deg"] = round(
        float(np.degrees(np.arccos(np.clip(abs(frame_axis @ z_axis), 0.0, 1.0)))), 2)
    debug_info[f"FP_{suffix}"] = femur_frame
    print(f"  {side} femur frame: axis={femur_frame['axis_source']}, "
          f"diaphysis window {diaphysis_mm:.0f} mm, {femur_frame['pca_axis_angle_deg']:.1f} deg from PCA")

    trochanter_landmarks = detect_trochanter_landmarks(
        proximal_vertices, femur_frame, fh["center"], fh["radius"], is_left_femur=is_left_femur
    )
    landmarks.update(trochanter_landmarks)

    gt_s = trochanter_landmarks.get(f"FEM_{suffix}_FT_S")
    gt_t = trochanter_landmarks.get(f"FEM_{suffix}_FT_T")
    if gt_s is not None and gt_t is not None:
        landmarks.update(
            detect_lesser_trochanter_landmarks(
                proximal_vertices,
                femur_frame,
                fh["center"],
                fh["radius"],
                gt_s,
                gt_t,
                is_left_femur=is_left_femur,
            )
        )

    for landmark_id in trochanter_ids:
        if landmark_id in landmarks:
            # Fischer, Sci Rep 10:20859 (2020), Table 4 reports the manual dispersion of these
            # points as 6.7 mm (superior greater trochanter) and 6.1 mm (lesser trochanter)
            # against 0.4 mm for the head centre, and uses neither in any coordinate system.
            metrics = {"expected_error_mm": 6.7 if "FT_S" in landmark_id else 6.1,
                       "used_by_frame": False, "frame_axis": femur_frame["axis_source"],
                       "diaphysis_window_mm": femur_frame["diaphysis_window_mm"]}
            if short_shaft:
                # Too little shaft for the diaphysis axis: the PCA frame's tilt, and so these
                # extrema, depend on where the scan cut the femur.
                records[landmark_id] = status.low_confidence(
                    landmark_id, landmarks[landmark_id], confidence=SHORT_SHAFT_CONFIDENCE,
                    reason=status.Reason.FRAME_UNRESOLVED, provenance="frame_extremum",
                    metrics=metrics,
                )
            else:
                records[landmark_id] = status.ok(
                    landmark_id, landmarks[landmark_id], confidence=0.5,
                    provenance="frame_extremum", metrics=metrics,
                )
        else:
            records[landmark_id] = status.abstained(
                landmark_id, status.Reason.EMPTY_REGION, provenance="frame_extremum"
            )

    return landmarks, debug_info


def detect_femur_landmarks(femur_left_path=None, femur_right_path=None, case_id=""):
    # Per-side generators so a single-sided scan gives the same answer as a bilateral one.
    landmarks = {}
    debug_info = {"records": {}}

    for path, is_left, stage in (
        (femur_left_path, True, "femur_left"),
        (femur_right_path, False, "femur_right"),
    ):
        if not (path and os.path.exists(path)):
            continue
        vertices = load_femur_vertices(path)
        if vertices is None:
            continue
        side_landmarks, side_debug = detect_single_femur_landmarks(
            vertices, is_left_femur=is_left, rng=rng_for(case_id, stage)
        )
        landmarks.update(side_landmarks)
        debug_info["records"].update(side_debug.pop("records", {}))
        debug_info.update(side_debug)

    return landmarks, debug_info
