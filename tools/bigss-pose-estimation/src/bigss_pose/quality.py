"""Automatic plausibility checks for predicted poses; no ground truth is needed.

Many radiographs of a case never show the registration targets, or show only a sliver of them,
and the network still returns a pose for each. These checks decide, per view and per pose stage
(network, refined), whether a pose is worth handing to registration:

prior    xvr parameters of the pose inside the training prior (plus a tolerance). The network
         was trained only there; outside it the output is an extrapolation.
targets  each ``predict.qc_targets`` label projected through the pose: the share of its voxels
         that land on the cropped detector and the share of the detector it covers. Training and
         the self-check keep a DRR only when the prior label covers >= 5 % of it.
gncc     gradient NCC between a DRR of the whole CT at the pose and the X-ray, over the cropped
         detector at 256 px with a 12 px border dropped: does the pose explain the image?
         (DiffDRR's own GradientNormalizedCrossCorrelation2d zero-pads its Sobel filter; the
         image border then dominates both gradient images and every pose scores ~0.8-0.9.)
basin    mean gncc lost when the pose is displaced by +/-3 deg about, and +/-8 mm (in plane) or
         +/-40 mm (depth) along, each camera axis at the target centroid. Near the true pose the
         similarity falls off in every direction; for an unrelated pose it is flat or rises.
refine   refined stage only: rotation and in-plane/depth displacement from the network pose.

The tier is 'confident', 'plausible' or 'suspect'; ``flags`` name every reason a pose missed
'confident'. Thresholds are engineering values calibrated on 142 views of a Cios Fusion / Loop-X case
(2026-09-29, see docs/outputs.md), not validated error bounds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .bridge import Bridge, xvr_parameters
from .xreg_io import XregCamera

SIM_SIZE = 256
SIM_BORDER_PX = 12
COVERAGE_GRID = 32
MAX_TARGET_POINTS = 60000
MIN_TARGET_COVERAGE = 0.05
PRIOR_TOL_DEG = 5.0
PRIOR_TOL_MM = 25.0
PERTURB_DEG = 3.0
PERTURB_INPLANE_MM = 8.0
PERTURB_DEPTH_MM = 40.0
# Calibrated on those 142 views (overlays reviewed). With the source on the wrong
# side (patient left/right exchanged) aligned-looking views scored only 0.30-0.54; with the correct
# posterior source they score 0.6-0.95, and lumbar-spine views beyond the CT, edge-on views and
# tool shots stay below 0.35. Revisit after registration gives real errors.
GNCC_CONFIDENT = 0.30
GNCC_PLAUSIBLE = 0.20
BASIN_MIN = 0.02
#: Stage "best" keeps the refined pose unless refinement lowered gncc by more than this.
BEST_TOLERANCE = 0.02
REFINE_MAX_ROT_DEG = 10.0
REFINE_MAX_INPLANE_MM = 20.0
TIERS = ("confident", "plausible", "suspect")
# Footprint gNCC (``score``): the target's voxels projected onto the SIM_SIZE grid, closed, holes
# filled and dilated, border excluded; fewer pixels than FOOTPRINT_MIN_PX give no score.
FOOTPRINT_CLOSE_PX = 3
FOOTPRINT_DILATE_PX = 4
FOOTPRINT_MIN_PX = 50


class InteriorGradientNCC(torch.nn.Module):
    """NCC of Sobel gradients (both components) over the image interior, per batch item."""

    def __init__(self, sigma: float = 1.0, border: int = SIM_BORDER_PX):
        super().__init__()
        from diffdrr.metrics import Sobel

        self.sobel = Sobel(sigma)
        self.border = border

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b = self.border
        g1, g2 = self.sobel(x1)[..., b:-b, b:-b], self.sobel(x2)[..., b:-b, b:-b]
        g1 = (g1 - g1.mean(dim=(-1, -2), keepdim=True)).flatten(1)
        g2 = (g2 - g2.mean(dim=(-1, -2), keepdim=True)).flatten(1)
        return (g1 * g2).sum(1) / (g1.norm(dim=1) * g2.norm(dim=1) + 1e-9)

    def masked(self, x1: torch.Tensor, x2: torch.Tensor, mask: np.ndarray) -> float | None:
        """The same NCC of one image pair restricted to ``mask`` (SIM_SIZE^2 bool); None when the
        mask has fewer than FOOTPRINT_MIN_PX pixels."""
        if mask.sum() < FOOTPRINT_MIN_PX:
            return None
        m = torch.as_tensor(mask, device=x1.device)
        g1, g2 = self.sobel(x1)[0][:, m], self.sobel(x2)[0][:, m]
        g1 = (g1 - g1.mean(dim=1, keepdim=True)).flatten()
        g2 = (g2 - g2.mean(dim=1, keepdim=True)).flatten()
        return float((g1 * g2).sum() / (g1.norm() * g2.norm() + 1e-9))


def thresholds() -> dict:
    return {"min_target_coverage": MIN_TARGET_COVERAGE, "prior_tol_deg": PRIOR_TOL_DEG,
            "prior_tol_mm": PRIOR_TOL_MM, "gncc_confident": GNCC_CONFIDENT,
            "gncc_plausible": GNCC_PLAUSIBLE, "basin_min": BASIN_MIN, "best_tolerance": BEST_TOLERANCE,
            "refine_max_rot_deg": REFINE_MAX_ROT_DEG, "refine_max_inplane_mm": REFINE_MAX_INPLANE_MM,
            "perturb": {"deg": PERTURB_DEG, "inplane_mm": PERTURB_INPLANE_MM, "depth_mm": PERTURB_DEPTH_MM},
            "sim_size_px": SIM_SIZE, "sim_border_px": SIM_BORDER_PX, "coverage_grid": COVERAGE_GRID}


def choose_best(network: dict, refined: dict) -> str:
    """'refined' unless refinement lowered the similarity by more than BEST_TOLERANCE (diverged)."""
    g_net, g_ref = network.get("gncc"), refined.get("gncc")
    if g_ref is None:
        return "network"
    if g_net is not None and g_ref < g_net - BEST_TOLERANCE:
        return "network"
    return "refined"


def camera_points(cam_to_vol: np.ndarray, extrinsic: np.ndarray, pts_lps: np.ndarray) -> np.ndarray:
    """Volume points in the xreg camera frame (source at the origin, detector at z = -sdd)."""
    vol_to_cam = extrinsic @ np.linalg.inv(cam_to_vol)
    return pts_lps @ vol_to_cam[:3, :3].T + vol_to_cam[:3, 3]


def footprint(cam_to_vol: np.ndarray, cam: XregCamera, pts_lps: np.ndarray, crop_px: int) -> dict:
    """Share of the points on the cropped detector (in front of the source) and share of the
    cropped detector they cover on a COVERAGE_GRID x COVERAGE_GRID raster."""
    if len(pts_lps) == 0:
        return {"in_view": 0.0, "coverage": 0.0}
    q = camera_points(cam_to_vol, cam.extrinsic, pts_lps)
    front = q[:, 2] < 0.0
    uv = cam.project(cam_to_vol, pts_lps)
    half = crop_px / 2.0
    lo_c, hi_c = half - 0.5, cam.cols - half - 0.5
    lo_r, hi_r = half - 0.5, cam.rows - half - 0.5
    inside = front & (uv[:, 0] >= lo_c) & (uv[:, 0] < hi_c) & (uv[:, 1] >= lo_r) & (uv[:, 1] < hi_r)
    cells = np.zeros((COVERAGE_GRID, COVERAGE_GRID), dtype=bool)
    if inside.any():
        c = ((uv[inside, 0] - lo_c) / (hi_c - lo_c) * COVERAGE_GRID).astype(int).clip(0, COVERAGE_GRID - 1)
        r = ((uv[inside, 1] - lo_r) / (hi_r - lo_r) * COVERAGE_GRID).astype(int).clip(0, COVERAGE_GRID - 1)
        cells[r, c] = True
    return {"in_view": float(inside.mean()), "coverage": float(cells.mean())}


def _grow(m: np.ndarray, outside: bool) -> np.ndarray:
    """``m`` or any of its 4-neighbours (pixels beyond the image count as ``outside``)."""
    p = np.pad(m, 1, constant_values=outside)
    return m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]


def _dilate(m: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        m = _grow(m, False)
    return m


def _erode(m: np.ndarray, n: int) -> np.ndarray:
    for _ in range(n):
        m = ~_grow(~m, True)
    return m


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """Background not 4-connected to the image border becomes foreground."""
    bg = ~m
    reached = np.zeros_like(m)
    while True:
        grown = _grow(reached, True) & bg
        if (grown == reached).all():
            return ~reached
        reached = grown


def footprint_mask(cam_to_vol: np.ndarray, cam: XregCamera, pts_lps: np.ndarray, crop_px: int) -> np.ndarray:
    """The target's footprint on the SIM_SIZE^2 similarity grid (the cropped detector's central
    square, as rendered by QualityChecker): projected points in front of the source, closed by
    FOOTPRINT_CLOSE_PX, holes filled, dilated by FOOTPRINT_DILATE_PX, SIM_BORDER_PX excluded."""
    mask = np.zeros((SIM_SIZE, SIM_SIZE), dtype=bool)
    if len(pts_lps):
        side = min(cam.rows, cam.cols) - crop_px
        uv = cam.project(cam_to_vol, pts_lps)
        u = np.floor((uv[:, 0] + 0.5 - (cam.cols - side) / 2) * SIM_SIZE / side)
        v = np.floor((uv[:, 1] + 0.5 - (cam.rows - side) / 2) * SIM_SIZE / side)
        ok = ((camera_points(cam_to_vol, cam.extrinsic, pts_lps)[:, 2] < 0.0)
              & (u >= 0) & (u < SIM_SIZE) & (v >= 0) & (v < SIM_SIZE))
        mask[v[ok].astype(int), u[ok].astype(int)] = True
    mask = _erode(_dilate(mask, FOOTPRINT_CLOSE_PX), FOOTPRINT_CLOSE_PX)
    mask = _dilate(_fill_holes(mask), FOOTPRINT_DILATE_PX)
    b = SIM_BORDER_PX
    mask[:b], mask[-b:], mask[:, :b], mask[:, -b:] = False, False, False, False
    return mask


def displaced(cam_to_vol: np.ndarray, extrinsic: np.ndarray, center_lps: np.ndarray, axis: int,
              rot_deg: float = 0.0, shift_mm: float = 0.0) -> np.ndarray:
    """The pose with the CT moved in the camera frame: rotated about camera axis ``axis`` through
    the target centre, or shifted along it."""
    c = camera_points(cam_to_vol, extrinsic, center_lps[None])[0]
    D = np.eye(4)
    if rot_deg:
        a = math.radians(rot_deg)
        i, j = [k for k in range(3) if k != axis]
        R = np.eye(3)
        R[i, i], R[i, j], R[j, i], R[j, j] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
        D[:3, :3] = R
        D[:3, 3] = c - R @ c
    D[axis, 3] += shift_mm
    # p_cam' = D p_cam with p_cam = E X^-1 p  =>  X' = X E^-1 D^-1 E
    return cam_to_vol @ np.linalg.inv(extrinsic) @ np.linalg.inv(D) @ extrinsic


def pose_change(X: np.ndarray, X_ref: np.ndarray, center_lps: np.ndarray, extrinsic: np.ndarray) -> dict:
    """Rotation (deg) and target-centre displacement of X relative to X_ref in X_ref's camera frame."""
    a = camera_points(X, extrinsic, center_lps[None])[0]
    b = camera_points(X_ref, extrinsic, center_lps[None])[0]
    d = a - b
    cosang = (np.trace(X[:3, :3].T @ X_ref[:3, :3]) - 1.0) / 2.0
    return {"rot_deg": float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))),
            "inplane_mm": float(np.linalg.norm(d[:2])), "depth_mm": float(d[2])}


def outside_prior(params: dict, ranges: dict) -> list[str]:
    """Pose parameters beyond the training prior by more than the tolerance. Angles are compared
    modulo 360 deg: a posterior-source prior spans alpha 135..225 while xvr reports -180..180."""
    out = []
    for key, tol in (("alpha", PRIOR_TOL_DEG), ("beta", PRIOR_TOL_DEG), ("gamma", PRIOR_TOL_DEG),
                     ("tx", PRIOR_TOL_MM), ("ty", PRIOR_TOL_MM), ("tz", PRIOR_TOL_MM)):
        lo, hi = ranges.get(f"{key}min"), ranges.get(f"{key}max")
        if lo is None or hi is None:
            continue
        values = [params[key] + k * 360.0 for k in (-1, 0, 1)] if key in ("alpha", "beta", "gamma") else [params[key]]
        if not any(lo - tol <= v <= hi + tol for v in values):
            out.append(key)
    return out


def assign_tier(record: dict) -> tuple[str, list[str]]:
    """Tier and flags from one pose's metrics (see the module docstring).

    Leaving the prior is hard for a network pose (extrapolation) but soft for a refined one, which
    the image moved there."""
    flags = []
    if record.get("outside_prior"):
        flags.append("outside_prior")
    targets = record.get("targets", {})
    if targets and not any(t["coverage"] >= MIN_TARGET_COVERAGE for t in targets.values()):
        flags.append("no_target_in_view")
    gncc = record.get("gncc")
    if gncc is None or not np.isfinite(gncc):
        flags.append("blank_drr")
    elif gncc < GNCC_PLAUSIBLE:
        flags.append("low_similarity")
    elif gncc < GNCC_CONFIDENT:
        flags.append("moderate_similarity")
    basin = record.get("basin")
    if basin is not None and np.isfinite(basin) and basin < BASIN_MIN:
        flags.append("flat_similarity")
    change = record.get("refine_change")
    if change and (change["rot_deg"] > REFINE_MAX_ROT_DEG or change["inplane_mm"] > REFINE_MAX_INPLANE_MM):
        flags.append("refine_moved_far")
    hard = {"no_target_in_view", "blank_drr", "low_similarity"}
    if change is None:
        hard.add("outside_prior")
    if hard & set(flags):
        return "suspect", flags
    return ("confident" if not flags else "plausible"), flags


def _target_points(seg_path, labels: dict[str, int]) -> dict[str, np.ndarray]:
    import nibabel as nib

    seg = nib.load(str(seg_path))
    data = np.asarray(seg.dataobj)
    out = {}
    for name, label in labels.items():
        ijk = np.argwhere(data == label)
        if len(ijk) > MAX_TARGET_POINTS:
            ijk = ijk[np.linspace(0, len(ijk) - 1, MAX_TARGET_POINTS).astype(int)]
        ras = np.c_[ijk, np.ones(len(ijk))] @ seg.affine.T
        out[name] = ras[:, :3] * np.array([-1.0, -1.0, 1.0])
    return out


@dataclass
class _Renderer:
    drr: object
    key: tuple


class QualityChecker:
    """Holds the whole-CT DRR renderer and the target point sets for one prediction run."""

    def __init__(self, ct_path, seg_path, targets: dict[str, int], center_label: int, bridge: Bridge,
                 prior_ranges: dict | None, crop_px: int, device: str = "cuda"):
        from diffdrr.data import read

        t0 = time.perf_counter()
        self.bridge = bridge
        self.device = torch.device(device)
        self.crop_px = crop_px
        self.ranges = prior_ranges or {}
        self.targets = _target_points(seg_path, {**targets, "__center__": center_label})
        self.center_lps = self.targets.pop("__center__").mean(0)
        self.subject = read(str(ct_path), orientation=bridge.orientation)
        self.gncc = InteriorGradientNCC().to(self.device)
        self._renderer: _Renderer | None = None
        self.init_seconds = time.perf_counter() - t0

    def _drr(self, cam: XregCamera):
        from diffdrr.drr import DRR

        side = min(cam.rows, cam.cols) - self.crop_px
        key = (round(cam.sdd, 3), side, round(cam.col_spacing, 6))
        if self._renderer is None or self._renderer.key != key:
            delx = side * cam.col_spacing / SIM_SIZE
            drr = DRR(self.subject, sdd=cam.sdd, height=SIM_SIZE, width=SIM_SIZE, delx=delx, dely=delx,
                      x0=0.0, y0=0.0, reverse_x_axis=False, renderer="trilinear", voxel_shift=0.0).to(self.device)
            self._renderer = _Renderer(drr, key)
        return self._renderer.drr

    def render(self, cam: XregCamera, poses: list[np.ndarray]) -> torch.Tensor:
        """Whole-CT DRRs (N, 1, SIM_SIZE, SIM_SIZE) of the cropped detector at xreg poses."""
        from diffdrr.pose import RigidTransform

        P = np.stack([self.bridge.xreg_to_diffdrr(X, cam.extrinsic) for X in poses])
        with torch.no_grad():
            return self._drr(cam)(RigidTransform(torch.as_tensor(P, dtype=torch.float32, device=self.device)))

    def xray_at_sim_size(self, img: torch.Tensor) -> torch.Tensor:
        """The preprocessed (cropped, normalised) X-ray resampled to SIM_SIZE."""
        import torch.nn.functional as F

        return F.interpolate(img.float(), size=(SIM_SIZE, SIM_SIZE), mode="area").to(self.device)

    def evaluate(self, X: np.ndarray, xray: torch.Tensor, cam: XregCamera,
                 network_pose: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
        """Metrics, tier and flags of pose X for one view; also the DRR at X (SIM_SIZE^2)."""
        t0 = time.perf_counter()
        params = xvr_parameters(self.bridge.xreg_to_diffdrr(X, cam.extrinsic))
        rec = {"xvr_parameters": params,
               "outside_prior": outside_prior(params, self.ranges),
               "targets": {name: footprint(X, cam, pts, self.crop_px) for name, pts in self.targets.items()}}
        poses = [X]
        for axis in range(3):
            for sign in (-1.0, 1.0):
                poses.append(displaced(X, cam.extrinsic, self.center_lps, axis, rot_deg=sign * PERTURB_DEG))
                shift = PERTURB_DEPTH_MM if axis == 2 else PERTURB_INPLANE_MM
                poses.append(displaced(X, cam.extrinsic, self.center_lps, axis, shift_mm=sign * shift))
        drrs = self.render(cam, poses)
        blank = drrs.flatten(1).std(1) < 1e-8
        scores = self.gncc(xray.expand(len(poses), -1, -1, -1), drrs).double().cpu().numpy()
        scores[blank.cpu().numpy()] = np.nan
        rec["gncc"] = None if not np.isfinite(scores[0]) else float(scores[0])
        displaced_scores = scores[1:][np.isfinite(scores[1:])]
        rec["basin"] = (float(scores[0] - displaced_scores.mean())
                        if rec["gncc"] is not None and len(displaced_scores) else None)
        rec["local_max"] = bool(rec["gncc"] is not None and len(displaced_scores)
                                and scores[0] >= displaced_scores.max())
        if network_pose is not None:
            rec["refine_change"] = pose_change(X, network_pose, self.center_lps, cam.extrinsic)
        rec["tier"], rec["flags"] = assign_tier(rec)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        rec["seconds"] = time.perf_counter() - t0
        return rec, drrs[0, 0].float().cpu().numpy()


def summarize(records: dict[str, dict]) -> dict:
    tiers = {t: sorted(i for i, r in records.items() if r["tier"] == t) for t in TIERS}
    flags: dict[str, int] = {}
    for r in records.values():
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    in_view = {}
    for r in records.values():
        for name, t in r["targets"].items():
            in_view.setdefault(name, 0)
            in_view[name] += int(t["coverage"] >= MIN_TARGET_COVERAGE)
    return {"n": len(records), "counts": {t: len(v) for t, v in tiers.items()}, "ids": tiers,
            "flag_counts": flags, "target_in_view_counts": in_view}


def overlay_sheets(path_stem, ids: list[str], xrays: dict[str, np.ndarray], drrs: dict[str, np.ndarray],
                   records: dict[str, dict], per_sheet: int = 72) -> list[str]:
    """Contact sheets: X-ray (equalised) with the DRR's strongest gradients in red, titled with
    id, tier and gncc. Title colours: green confident, orange plausible, red suspect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"confident": "green", "plausible": "darkorange", "suspect": "red"}

    def equalise(a):
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
        h, e = np.histogram(a, 256)
        return np.interp(a, e[:-1], np.cumsum(h) / h.sum())

    written = []
    for start in range(0, len(ids), per_sheet):
        chunk = ids[start:start + per_sheet]
        ncol = 9
        nrow = math.ceil(len(chunk) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, pid in zip(axes.flat, chunk):
            ax.imshow(equalise(xrays[pid]), cmap="gray")
            d = drrs.get(pid)
            if d is not None and np.ptp(d) > 0:
                gy, gx = np.gradient(d)
                g = np.hypot(gx, gy)
                edge = g > np.percentile(g, 92)
                ax.imshow(np.ma.masked_where(~edge, edge), cmap="autumn", alpha=0.55, vmin=0, vmax=1)
            r = records[pid]
            g = r.get("gncc")
            ax.set_title(f"{pid} {r['tier']} g={g:.2f}" if g is not None else f"{pid} {r['tier']}",
                         fontsize=10, color=colours[r["tier"]])
        fig.tight_layout()
        out = f"{path_stem}_{start // per_sheet + 1}.png"
        fig.savefig(out, dpi=50)
        plt.close(fig)
        written.append(out)
    return written
