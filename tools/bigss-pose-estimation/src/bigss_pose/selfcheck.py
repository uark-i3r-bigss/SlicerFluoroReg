"""Synthetic self-check of a trained pose model on this CT.

Poses are drawn from the model's own training prior with a fixed seed; full-CT DRRs are rendered
at the NATIVE reference detector and fed through the same path as a real radiograph (centre crop,
``predict_pose``). A sample counts when the prior's segmentation label covers >= 5 % of the
cropped detector. "Captured" = rotation <= 10 deg, label-centroid in-plane <= 20 mm and
|depth| <= 100 mm against the rendering pose (xreg's CMA-ES search scale, plus a loose depth
gate).
"""

from __future__ import annotations

import numpy as np
import torch

from .bridge import Bridge, rotation_angle_deg
from .xreg_io import XregCamera

CONTRAST_LEVELS = (1.5, 4.0, 6.5, 9.0)
ROT_CAP_DEG, INPLANE_CAP_MM, DEPTH_CAP_MM = 10.0, 20.0, 100.0
MIN_LABEL_FRACTION = 0.05


def compare(X: np.ndarray, X_ref: np.ndarray, center_lps: np.ndarray, extrinsic: np.ndarray) -> dict:
    """Rotation and centre displacement of pose X relative to X_ref, in X_ref's camera frame."""
    c = np.r_[center_lps, 1.0]
    a = (extrinsic @ np.linalg.inv(X) @ c)[:3]
    b = (extrinsic @ np.linalg.inv(X_ref) @ c)[:3]
    d = a - b
    return {"rot_deg": rotation_angle_deg(X[:3, :3], X_ref[:3, :3]),
            "depth_mm": float(d[2]), "inplane_mm": float(np.linalg.norm(d[:2]))}


def captured(m: dict) -> bool:
    return m["rot_deg"] <= ROT_CAP_DEG and m["inplane_mm"] <= INPLANE_CAP_MM and abs(m["depth_mm"]) <= DEPTH_CAP_MM


def run_self_check(model, config: dict, ct_path, seg_path, label: int, cam: XregCamera, crop_px: int,
                   center_lps: np.ndarray, n: int, seed: int, emit=None) -> dict:
    from diffdrr.data import read, transform_hu_to_density
    from diffdrr.drr import DRR
    from diffdrr.pose import RigidTransform
    from torchvision.transforms.functional import center_crop
    from xvr.model.inference import predict_pose
    from xvr.model.sampler import get_random_pose

    bridge = Bridge.from_ct(ct_path, "AP")
    geom = dict(sdd=cam.sdd, height=cam.rows, width=cam.cols, delx=cam.col_spacing,
                reverse_x_axis=False, renderer="trilinear", voxel_shift=0.0, patch_size=128)
    subj_lab = read(str(ct_path), str(seg_path), labels=[label], orientation="AP")
    drr_lab = DRR(subj_lab, **geom)
    drr_lab.mask = None
    drr_lab = drr_lab.cuda()
    subj = read(str(ct_path), orientation="AP")
    hu = subj.volume.data.squeeze()
    drr = DRR(subj, **geom).cuda()
    intr = dict(sdd=cam.sdd, delx=cam.col_spacing, dely=cam.row_spacing, x0=0.0, y0=0.0)

    keys = ("alphamin", "alphamax", "betamin", "betamax", "gammamin", "gammamax",
            "txmin", "txmax", "tymin", "tymax", "tzmin", "tzmax")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    poses = get_random_pose(**{k: config[k] for k in keys}, batch_size=n).matrix.double().numpy()
    contrasts = rng.choice(CONTRAST_LEVELS, size=n)
    rows = []
    for c in CONTRAST_LEVELS:
        drr.density = transform_hu_to_density(hu, float(c)).to(drr.density)
        for k in np.flatnonzero(contrasts == c):
            P = torch.as_tensor(poses[k], dtype=torch.float32, device="cuda")[None]
            with torch.no_grad():
                img = drr(RigidTransform(P))
                lab = drr_lab(RigidTransform(P))
            h, w = img.shape[-2:]
            img = center_crop(img, (h - crop_px, w - crop_px)).cpu()
            frac = float((center_crop(lab, (h - crop_px, w - crop_px)) > 0).float().mean())
            if frac < MIN_LABEL_FRACTION:
                rows.append({"index": int(k), "contrast": float(c), "label_fraction": frac, "kept": False})
                continue
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            pred = predict_pose(model, config, img, **intr).matrix[0].double().cpu().numpy()
            X_true = bridge.diffdrr_to_xreg(poses[k], cam.extrinsic)
            X_pred = bridge.diffdrr_to_xreg(pred, cam.extrinsic)
            m = compare(X_pred, X_true, center_lps, cam.extrinsic)
            rows.append({"index": int(k), "contrast": float(c), "label_fraction": frac, "kept": True,
                         **m, "captured": captured(m)})
        if emit:
            emit("progress", stage="self_check", done=sum(1 for r in rows), total=n)
    kept = [r for r in rows if r["kept"]]
    cap = sum(r["captured"] for r in kept)
    med = lambda k: float(np.median([abs(r[k]) for r in kept])) if kept else None  # noqa: E731
    del drr, drr_lab
    torch.cuda.empty_cache()
    return {"n_drawn": n, "n_kept": len(kept), "n_captured": int(cap),
            "capture_rate": (cap / len(kept)) if kept else 0.0,
            "median_rot_deg": med("rot_deg"), "median_inplane_mm": med("inplane_mm"),
            "median_abs_depth_mm": med("depth_mm"), "seed": seed, "samples": rows}
