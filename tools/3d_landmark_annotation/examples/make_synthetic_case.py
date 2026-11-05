#!/usr/bin/env python3
"""Generate a synthetic, GPU-free hip/pelvis/lumbar case for the landmark pipeline.

The output is a run-ready case layout:

    <output-dir>/<case-id>.nii.gz              synthetic CT (int16 HU), pipeline input
    <output-dir>/<case-id>/segmentations/      per-anatomy binary masks the pipeline
                                               consumes with --skip-segmentation
    <output-dir>/ground_truth.json             exact phantom geometry in world LPS

Nothing here is anatomically accurate; it is a crude bone phantom whose only job is
to exercise every stage of process_medical_scans.py without TotalSegmentator, a GPU,
or patient data.  Only numpy + nibabel are required for generation.

World convention: LPS millimetres (+x left, +y posterior, +z superior).  The NIfTI
affine is the RAS+ form of that mapping, which is what nibabel and the pipeline's
STL writer expect.
"""

import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Grid definition (world LPS millimetres)
# ---------------------------------------------------------------------------
SPACING = 1.5
ORIGIN_LPS = (-135.0, -115.0, -305.0)
DIMS = (181, 155, 340)

# Femoral head centres: the numeric ground truth the sphere fit is checked against.
FH_CENTER = {"left": (85.0, 0.0, 0.0), "right": (-85.0, 0.0, 0.0)}
FH_RADIUS = 25.0

ANATOMIES = [
    "hip_left",
    "hip_right",
    "femur_left",
    "femur_right",
    "vertebrae_L1",
    "vertebrae_L2",
    "vertebrae_L3",
    "vertebrae_L4",
    "vertebrae_L5",
    "vertebrae_S1",
    "sacrum",
]

VERTEBRA_Z = {"L5": 72.0, "L4": 100.0, "L3": 128.0, "L2": 156.0, "L1": 184.0}

# Hounsfield units
HU_AIR = -1000.0
HU_SOFT = 40.0
HU_TRABECULAR = 380.0
HU_CORTICAL = 950.0


class Grid:
    """Broadcastable LPS coordinate axes for the voxel grid."""

    def __init__(self, spacing=SPACING, origin=ORIGIN_LPS, dims=DIMS):
        self.spacing = float(spacing)
        self.origin = tuple(float(v) for v in origin)
        self.shape = tuple(int(v) for v in dims)
        nx, ny, nz = self.shape
        self.x = (self.origin[0] + spacing * np.arange(nx, dtype=np.float32)).reshape(-1, 1, 1)
        self.y = (self.origin[1] + spacing * np.arange(ny, dtype=np.float32)).reshape(1, -1, 1)
        self.z = (self.origin[2] + spacing * np.arange(nz, dtype=np.float32)).reshape(1, 1, -1)

    @property
    def affine(self):
        """RAS+ affine encoding the LPS index->world mapping (x_ras = -x_lps)."""
        sp = self.spacing
        ox, oy, oz = self.origin
        return np.array(
            [
                [-sp, 0.0, 0.0, -ox],
                [0.0, -sp, 0.0, -oy],
                [0.0, 0.0, sp, oz],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def empty(self):
        return np.zeros(self.shape, dtype=bool)


# ---------------------------------------------------------------------------
# Solid primitives -> boolean volumes
# ---------------------------------------------------------------------------
def ellipsoid(g, center, radii):
    cx, cy, cz = center
    rx, ry, rz = radii
    return (
        ((g.x - cx) / rx) ** 2 + ((g.y - cy) / ry) ** 2 + ((g.z - cz) / rz) ** 2
    ) <= 1.0


def sphere(g, center, radius):
    return ellipsoid(g, center, (radius, radius, radius))


def sphere_shell(g, center, r_inner, r_outer):
    cx, cy, cz = center
    d2 = (g.x - cx) ** 2 + (g.y - cy) ** 2 + (g.z - cz) ** 2
    return (d2 >= r_inner ** 2) & (d2 <= r_outer ** 2)


def capsule(g, p0, p1, radius):
    """Solid cylinder with spherical caps between two world points."""
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    seg = p1 - p0
    seg_len2 = float(seg @ seg)
    dx = g.x - p0[0]
    dy = g.y - p0[1]
    dz = g.z - p0[2]
    t = (dx * seg[0] + dy * seg[1] + dz * seg[2]) / seg_len2
    np.clip(t, 0.0, 1.0, out=t)
    dist2 = (dx - t * seg[0]) ** 2 + (dy - t * seg[1]) ** 2 + (dz - t * seg[2]) ** 2
    return dist2 <= radius ** 2


def box(g, lo, hi):
    return (
        (g.x >= lo[0]) & (g.x <= hi[0])
        & (g.y >= lo[1]) & (g.y <= hi[1])
        & (g.z >= lo[2]) & (g.z <= hi[2])
    )


def lerp_profile(z, z0, z1, v0, v1):
    """Linear ramp in z, clamped outside [z0, z1]; returns a z-broadcast array."""
    t = np.clip((z - z0) / (z1 - z0), 0.0, 1.0)
    return v0 + (v1 - v0) * t


# ---------------------------------------------------------------------------
# Anatomy
# ---------------------------------------------------------------------------
def make_hip(g, side):
    """Iliac wing + acetabular cup + pubis + ischium for one hemipelvis."""
    s = 1.0 if side == "left" else -1.0
    fh = np.array(FH_CENTER[side], dtype=np.float32)

    # Iliac wing: thin slab flaring laterally and tilting posteriorly with height.
    wing_x = s * lerp_profile(g.z, 38.0, 128.0, 50.0, 95.0)
    wing_ant = lerp_profile(g.z, 38.0, 128.0, -60.0, -40.0)
    wing_post = lerp_profile(g.z, 38.0, 128.0, 45.0, 65.0)
    wing = (np.abs(g.x - wing_x) <= 7.0) & (g.y >= wing_ant) & (g.y <= wing_post)
    wing &= (g.z >= 38.0) & (g.z <= 128.0)

    mask = wing
    # ASIS spike: the anterior-most point of the ilium, at mid-crest height.
    mask |= capsule(g, (s * 57, -52, 60), (s * 62, -80, 57), 9.0)
    # AIIS knob: anterior, one notch inferior to ASIS.
    mask |= capsule(g, (s * 55, -45, 36), (s * 52, -60, 28), 8.0)
    # PSIS knob: posterior-superior and more medial than ASIS.
    mask |= capsule(g, (s * 70, 58, 95), (s * 38, 70, 93), 8.0)
    # Acetabular cup: open laterally so the femoral head is not enclosed.
    cup = sphere_shell(g, fh, 27.0, 40.0)
    cup &= (s * (g.x - fh[0])) <= 10.0
    mask |= cup
    # Supra-acetabular column tying the cup to the wing.
    mask |= capsule(g, (s * 85, 0, 18), (s * 52, -4, 46), 14.0)
    # Superior pubic ramus ending in the pubic tubercle near the midline.
    mask |= capsule(g, (s * 70, -20, -15), (s * 10, -45, -25), 10.0)
    # Ischial body + tuberosity.
    mask |= capsule(g, (s * 78, 18, -18), (s * 40, 30, -80), 13.0)
    mask |= ellipsoid(g, (s * 38, 32, -88), (17, 16, 14))
    # Inferior pubic ramus closing the obturator foramen.
    mask |= capsule(g, (s * 38, 28, -82), (s * 14, -40, -30), 7.0)
    return mask


def make_femur(g, side):
    """Head sphere + neck + shaft cylinder + trochanter bumps."""
    s = 1.0 if side == "left" else -1.0
    fh = FH_CENTER[side]

    mask = sphere(g, fh, FH_RADIUS)
    mask |= capsule(g, (s * 88, 0, -2), (s * 112, 2, -26), 10.0)          # neck
    mask |= capsule(g, (s * 116, 3, -30), (s * 120, 5, -290), 14.0)       # shaft
    mask |= ellipsoid(g, (s * 128, 2, 2), (14, 13, 26))                   # greater troch.
    mask |= ellipsoid(g, (s * 104, 16, -42), (9, 10, 9))                  # lesser troch.
    return mask


def make_vertebra(g, name):
    """Body block + posterior spinous process + lateral transverse processes."""
    zc = VERTEBRA_Z[name]
    mask = box(g, (-22, 8, zc - 13), (22, 44, zc + 13))
    mask |= box(g, (-6, 44, zc - 8), (6, 82, zc + 8))
    mask |= box(g, (-52, 44, zc - 5), (-22, 56, zc + 5))
    mask |= box(g, (22, 44, zc - 5), (52, 56, zc + 5))
    return mask


def make_sacrum(g):
    """Wedge tapering to an inferior apex."""
    half_width = lerp_profile(g.z, -30.0, 58.0, 8.0, 48.0)
    mask = (np.abs(g.x) <= half_width) & (g.y >= 6.0) & (g.y <= 50.0)
    mask &= (g.z >= -30.0) & (g.z <= 58.0)
    return mask


def make_s1(g):
    return box(g, (-24, 8, 42), (24, 46, 58))


def build_masks(g):
    masks = {}
    for side in ("left", "right"):
        masks[f"hip_{side}"] = make_hip(g, side)
        masks[f"femur_{side}"] = make_femur(g, side)
    for name in VERTEBRA_Z:
        masks[f"vertebrae_{name}"] = make_vertebra(g, name)
    masks["sacrum"] = make_sacrum(g)
    masks["vertebrae_S1"] = make_s1(g)
    return {name: masks[name] for name in ANATOMIES}


# ---------------------------------------------------------------------------
# CT synthesis
# ---------------------------------------------------------------------------
def erode6(mask):
    """6-neighbour binary erosion (numpy only) used to carve a cortical shell."""
    out = mask.copy()
    out[1:, :, :] &= mask[:-1, :, :]
    out[:-1, :, :] &= mask[1:, :, :]
    out[:, 1:, :] &= mask[:, :-1, :]
    out[:, :-1, :] &= mask[:, 1:, :]
    out[:, :, 1:] &= mask[:, :, :-1]
    out[:, :, :-1] &= mask[:, :, 1:]
    out[0, :, :] = False
    out[-1, :, :] = False
    out[:, 0, :] = False
    out[:, -1, :] = False
    out[:, :, 0] = False
    out[:, :, -1] = False
    return out


def build_ct(g, masks, seed=0):
    """Air background, soft-tissue body, cortical shell and trabecular core."""
    rng = np.random.default_rng(seed)
    ct = rng.normal(HU_AIR, 15.0, size=g.shape).astype(np.float32)

    body = ((g.x - 0.0) / 120.0) ** 2 + ((g.y + 5.0) / 95.0) ** 2 <= 1.0
    body = np.broadcast_to(body, g.shape)
    ct[body] = rng.normal(HU_SOFT, 25.0, size=int(body.sum())).astype(np.float32)

    bone = g.empty()
    for mask in masks.values():
        bone |= mask

    inner = erode6(erode6(bone))
    cortical = bone & ~inner
    ct[inner] = rng.normal(HU_TRABECULAR, 60.0, size=int(inner.sum())).astype(np.float32)
    ct[cortical] = rng.normal(HU_CORTICAL, 45.0, size=int(cortical.sum())).astype(np.float32)

    return np.rint(ct).astype(np.int16), bone


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def ground_truth(g, case_id, ct_path, case_dir):
    return {
        "note": "Synthetic phantom. Coordinates are world LPS millimetres, matching the FCSV output.",
        "case_id": case_id,
        "ct_path": os.path.abspath(ct_path),
        "case_dir": os.path.abspath(case_dir),
        "spacing_mm": g.spacing,
        "shape": list(g.shape),
        "origin_lps_mm": list(g.origin),
        "affine_ras": g.affine.tolist(),
        "anatomies": list(ANATOMIES),
        "femoral_head_centers_lps": {
            "FEM_L_FH": list(FH_CENTER["left"]),
            "FEM_R_FH": list(FH_CENTER["right"]),
        },
        "femoral_head_radius_mm": FH_RADIUS,
        "greater_trochanter_centers_lps": {
            "FEM_L_FT": [128.0, 2.0, 2.0],
            "FEM_R_FT": [-128.0, 2.0, 2.0],
        },
        "lesser_trochanter_centers_lps": {
            "FEM_L_LT": [104.0, 16.0, -42.0],
            "FEM_R_LT": [-104.0, 16.0, -42.0],
        },
        "vertebra_body_centers_lps": {
            f"vertebrae_{name}": [0.0, 26.0, z] for name, z in VERTEBRA_Z.items()
        },
        "sacral_apex_lps": [0.0, 28.0, -30.0],
    }


# ---------------------------------------------------------------------------
# Optional landmark self-check (uses the repo's own STL + landmark code)
# ---------------------------------------------------------------------------
def verify_landmarks(repo_root, case_dir, check_dir, case_id, truth):
    """Run the pipeline's STL + landmark stages directly on the generated masks."""
    sys.path.insert(0, repo_root)
    from utils.stl_generator import generate_stls_from_segmentations
    from landmarks.fcsv_writer import create_comprehensive_landmark_file, read_fcsv

    seg_dir = os.path.join(case_dir, "segmentations")
    models_dir = os.path.join(check_dir, "models")
    totalseg_models = os.path.join(check_dir, "TotalSegmentator_outputs", "models")
    os.makedirs(totalseg_models, exist_ok=True)

    anatomy_config = os.path.join(check_dir, "anatomy_config.json")
    with open(anatomy_config, "w") as handle:
        json.dump(
            {
                "1": ["hip_left", "hip_right"],
                "2": "femur_left",
                "3": "femur_right",
                "4": "vertebrae_L1",
                "5": "vertebrae_L2",
                "6": "vertebrae_L3",
                "7": "vertebrae_L4",
                "8": "vertebrae_L5",
                "9": "vertebrae_S1",
                "10": "sacrum",
            },
            handle,
            indent=4,
        )
    all_labels = os.path.join(repo_root, "configs", "totalseg_all_labels.json")

    generate_stls_from_segmentations(seg_dir, models_dir, anatomy_config)
    generate_stls_from_segmentations(seg_dir, totalseg_models, all_labels)
    fcsv_path, counts = create_comprehensive_landmark_file(
        models_dir, os.path.join(check_dir, "landmarks"), case_id
    )

    points = read_fcsv(fcsv_path)
    print(f"\nLandmark self-check: {fcsv_path}")
    print(f"  counts: {counts}")
    for key, truth_key in (("FEM_L_FH", "FEM_L_FH"), ("FEM_R_FH", "FEM_R_FH")):
        if key in points:
            gt = np.asarray(truth["femoral_head_centers_lps"][truth_key])
            err = float(np.linalg.norm(points[key] - gt))
            print(f"  {key}: fitted {np.round(points[key], 2).tolist()} vs truth {gt.tolist()} -> {err:.2f} mm")
        else:
            print(f"  {key}: NOT DETECTED")
    return fcsv_path


# ---------------------------------------------------------------------------
def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(repo_root, "examples", "outputs", "synthetic"),
        help="Directory for the synthetic CT, masks and ground truth (gitignored).",
    )
    parser.add_argument("--case-id", default="synthetic_hip", help="Case id; also the CT filename stem.")
    parser.add_argument("--spacing", type=float, default=SPACING, help="Isotropic voxel size in mm.")
    parser.add_argument("--seed", type=int, default=0, help="Noise seed; generation is otherwise deterministic.")
    parser.add_argument(
        "--verify-landmarks",
        action="store_true",
        help="After writing, run the repo's STL + landmark detectors on the masks and report femoral-head error.",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    case_dir = os.path.join(output_dir, args.case_id)
    seg_dir = os.path.join(case_dir, "segmentations")
    os.makedirs(seg_dir, exist_ok=True)

    scale = SPACING / args.spacing
    dims = tuple(int(round(d * scale)) for d in DIMS)
    g = Grid(spacing=args.spacing, dims=dims)
    affine = g.affine

    print(f"Grid: {g.shape[0]} x {g.shape[1]} x {g.shape[2]} voxels @ {g.spacing} mm")
    masks = build_masks(g)
    for name, mask in masks.items():
        voxels = int(mask.sum())
        if voxels == 0:
            raise SystemExit(f"Phantom error: mask '{name}' is empty")
        path = os.path.join(seg_dir, f"{name}.nii.gz")
        img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        img.set_data_dtype(np.uint8)
        nib.save(img, path)
        print(f"  mask {name:<14} {voxels:>8} voxels -> {os.path.relpath(path, output_dir)}")

    ct, bone = build_ct(g, masks, seed=args.seed)
    ct_path = os.path.join(output_dir, f"{args.case_id}.nii.gz")
    ct_img = nib.Nifti1Image(ct, affine)
    ct_img.set_data_dtype(np.int16)
    nib.save(ct_img, ct_path)

    total = ct.size
    print(
        f"  CT  HU range [{int(ct.min())}, {int(ct.max())}], "
        f"air<-200 {np.count_nonzero(ct < -200) / total:.3f}, "
        f"bone>300 {np.count_nonzero(ct > 300) / total:.4f}, "
        f"bone voxels {int(bone.sum())}"
    )
    print(f"  CT  -> {os.path.relpath(ct_path, output_dir)}")

    truth = ground_truth(g, args.case_id, ct_path, case_dir)
    truth_path = os.path.join(output_dir, "ground_truth.json")
    with open(truth_path, "w") as handle:
        json.dump(truth, handle, indent=2)
    print(f"  ground truth -> {os.path.relpath(truth_path, output_dir)}")

    if args.verify_landmarks:
        verify_landmarks(repo_root, case_dir, os.path.join(output_dir, "landmark_check"), args.case_id, truth)

    print("\nRun the pipeline on it with:")
    print(
        f"  conda run -n totalseg python {os.path.join(repo_root, 'process_medical_scans.py')} "
        f"{ct_path} {output_dir} --skip-segmentation --no-confirm"
    )


if __name__ == "__main__":
    main()
