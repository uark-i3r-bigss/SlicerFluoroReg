#!/usr/bin/env python3
"""Surface generation from TotalSegmentator label maps.

The landmark detectors pick surface extrema, so the surface itself is part of the
measurement.  Contouring a strictly binary {0,1} mask at 0.5 places every marching-cubes
vertex at the exact midpoint of a voxel edge (t = (0.5-0)/(1-0) = 0.5), which confines the
surface to a half-voxel lattice and leaves every extremum on a flat plateau one voxel face
wide.  On this corpus the clinical cases are 1.39-1.56 mm in plane by 3.0 mm through plane
(measured 2026-09-22 over four cases), so that plateau is 3 mm deep along the slice axis
and the winning vertex is decided by array ordering rather than by anatomy.

This module therefore antialiases the mask into a signed distance field before contouring,
resamples that field to an isotropic grid, and conditions the surface with parameters whose
provenance is recorded per step:

  1. crop to the foreground bounding box with padding      (performance only)
  2. fill interior cavities on the label map, 6-connected  (Jordan complement of the
     26-connected foreground rule in utils/segmentation_cleanup.py)
  3. signed distance field in mm, sampling = voxel zooms   (removes the t = 0.5 degeneracy)
  4. resample the field to isotropic spacing, linear       (uniform edge length in mm)
  5. contour at 0.0 with Flying Edges                      (threaded; the previous
     grid.contour([0.5]) used pyvista's default method='contour' -> vtkContourFilter ->
     vtkSynchronizedTemplates3D, i.e. the single-threaded path, despite the comment)
  6. index -> world affine, then RAS -> LPS
  7. winding guard when det(index->world) < 0              (3 of 4 CTs in this corpus have
     a negative-determinant affine, so this fires in production)
  8. Taubin smoothing, 20 iterations, pass band 0.1        (windowed sinc; VTK's own docs
     note the Laplacian filter needs 100-200 iterations and shrinks the geometry, which for
     extremum-based landmarks is a systematic bias on every landmark at once)
  9. decimation bounded to 0.05 mm point-to-surface        (Fischer et al., Sci Rep 9:13322,
     2019, Subject data: decimation restricted to a 0.05 mm Hausdorff distance)

Nothing here is derived from the EUPL/CC-BY-NC reference implementations; see PROVENANCE.md.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count

import cc3d
import nibabel as nib
import numpy as np
import pyvista as pv
from scipy import ndimage

#: Fischer et al., Sci Rep 9:13322 (2019), *Subject data*.
DECIMATION_HAUSDORFF_MM = 0.05
#: VTK vtkWindowedSincPolyDataFilter documentation: "Ten or twenty iterations is all that is
#: usually necessary"; "A good default value for the PassBand is 0.1".
TAUBIN_ITERATIONS = 20
TAUBIN_PASS_BAND = 0.1


@dataclass(frozen=True)
class MeshConfig:
    """Resolved surface-generation parameters; echoed into the debug JSON."""

    isotropic_spacing_mm: float = 1.0
    fill_cavities: bool = True
    smooth_iterations: int = TAUBIN_ITERATIONS
    smooth_pass_band: float = TAUBIN_PASS_BAND
    decimate: bool = True
    decimation_hausdorff_mm: float = DECIMATION_HAUSDORFF_MM
    decimation_bisection_steps: int = 6
    pad_voxels: int = 2


@dataclass
class MeshResult:
    """Outcome of one surface generation.  Replaces the previous bare True/False/None."""

    status: str  # ok | empty | degenerate | error
    path: str = ""
    n_points: int = 0
    n_faces: int = 0
    achieved_reduction: float = 0.0
    hausdorff_mm: float = 0.0
    winding_flipped: bool = False
    message: str = ""
    config: dict = field(default_factory=dict)

    @property
    def ok(self):
        return self.status == "ok"

    def __bool__(self):
        return self.ok


def _fill_interior_cavities(mask):
    """Remove voxel cavities enclosed by the mask.

    Background is labelled with 6-connectivity, which is the Jordan-consistent complement of
    the 26-connected foreground rule used by utils/segmentation_cleanup.py.  Interior shells
    otherwise survive into the mesh and pollute every vertex-mean statistic.
    """
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    labels = cc3d.connected_components(~padded, connectivity=6)
    border = set(np.unique(np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])))
    border.discard(0)
    outside = np.isin(labels, list(border))
    filled = ~outside
    return filled[1:-1, 1:-1, 1:-1]


def _signed_distance(mask, zooms):
    """Signed distance to the mask boundary in millimetres (negative inside)."""
    outside = ndimage.distance_transform_edt(~mask, sampling=zooms)
    inside = ndimage.distance_transform_edt(mask, sampling=zooms)
    return (outside - inside).astype(np.float32)


def _resample_isotropic(field, zooms, spacing):
    """Linear resampling of the distance field onto an isotropic grid.

    The field is interpolated, not the label: nearest-neighbour resampling of a 3 mm mask to
    1 mm reproduces the same staircase as three thinner steps and changes no geometry.
    """
    scale = np.asarray(zooms, dtype=float) / float(spacing)
    out_shape = tuple(int(np.floor((n - 1) * s)) + 1 for n, s in zip(field.shape, scale))
    grids = np.meshgrid(*[np.arange(n) / s for n, s in zip(out_shape, scale)], indexing="ij")
    resampled = ndimage.map_coordinates(field, np.asarray(grids), order=1, mode="nearest")
    return resampled.astype(np.float32), scale


def _point_to_surface_error(decimated, reference):
    """99.9th-percentile absolute distance from the decimated points to the reference surface."""
    probed = decimated.compute_implicit_distance(reference)
    distances = np.abs(np.asarray(probed.point_data["implicit_distance"], dtype=float))
    return float(np.percentile(distances, 99.9)) if distances.size else 0.0


def _decimate_bounded(surface, cfg):
    """Largest reduction whose point-to-surface error stays within the Hausdorff budget."""
    low, high = 0.0, 0.95
    best, best_error = surface, 0.0
    for _ in range(cfg.decimation_bisection_steps):
        mid = 0.5 * (low + high)
        candidate = surface.decimate(mid, volume_preservation=True)
        if candidate.n_points == 0:
            high = mid
            continue
        error = _point_to_surface_error(candidate, surface)
        if error <= cfg.decimation_hausdorff_mm:
            best, best_error, low = candidate, error, mid
        else:
            high = mid
    reduction = 1.0 - (best.n_cells / surface.n_cells) if surface.n_cells else 0.0
    return best, reduction, best_error


def mask_to_surface(mask, affine, zooms, cfg=MeshConfig()):
    """Turn a boolean mask plus its NIfTI affine into an LPS-millimetre surface."""
    if not mask.any():
        return None, MeshResult(status="empty", message="mask has no foreground voxels",
                                config=asdict(cfg))
    if int(mask.sum()) < 3:
        return None, MeshResult(status="degenerate", message="fewer than 3 foreground voxels",
                                config=asdict(cfg))

    # 1. crop to the foreground bounding box, padded so the field closes on every face.
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - cfg.pad_voxels, 0)
    hi = np.minimum(coords.max(axis=0) + cfg.pad_voxels + 1, mask.shape)
    cropped = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # 2 & 3. cavity fill, then the signed distance field in millimetres.
    if cfg.fill_cavities:
        cropped = _fill_interior_cavities(cropped)
    field = _signed_distance(cropped, zooms)

    # 4. isotropic resampling of the field.
    field_iso, scale = _resample_isotropic(field, zooms, cfg.isotropic_spacing_mm)

    # 5. contour at the zero level set with the threaded implementation.
    grid = pv.wrap(field_iso)
    surface = grid.contour([0.0], method="flying_edges")
    if surface.n_points == 0:
        return None, MeshResult(status="degenerate", message="contour produced no points",
                                config=asdict(cfg))

    # 6. isotropic index -> original index -> world (RAS) -> LPS.
    index_to_original = np.diag(np.append(1.0 / scale, 1.0))
    index_to_original[:3, 3] = lo
    ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
    index_to_lps = ras_to_lps @ affine @ index_to_original
    points = np.asarray(surface.points, dtype=float)
    surface.points = points @ index_to_lps[:3, :3].T + index_to_lps[:3, 3]

    # 7. winding guard: an orientation-reversing map inverts every face normal.
    winding_flipped = False
    if np.linalg.det(index_to_lps[:3, :3]) < 0:
        faces = np.array(surface.faces, copy=True).reshape(-1, 4)
        faces[:, 1:] = faces[:, [3, 2, 1]]
        surface.faces = faces.ravel()
        winding_flipped = True

    # 8. Taubin smoothing in world millimetres.
    if cfg.smooth_iterations > 0:
        surface = surface.smooth_taubin(
            n_iter=cfg.smooth_iterations,
            pass_band=cfg.smooth_pass_band,
            feature_smoothing=False,
            boundary_smoothing=False,
            non_manifold_smoothing=False,
            normalize_coordinates=True,
        )

    # 9. bounded decimation, last.
    reduction, hausdorff = 0.0, 0.0
    if cfg.decimate and surface.n_cells > 0:
        surface, reduction, hausdorff = _decimate_bounded(surface, cfg)

    result = MeshResult(
        status="ok",
        n_points=int(surface.n_points),
        n_faces=int(surface.n_cells),
        achieved_reduction=round(float(reduction), 4),
        hausdorff_mm=round(float(hausdorff), 4),
        winding_flipped=winding_flipped,
        config=asdict(cfg),
    )
    return surface, result


def generate_stl(nifti_path, output_path, cfg=MeshConfig()):
    """Generate one STL from a segmentation NIfTI.  Returns a MeshResult."""
    try:
        image = nib.load(nifti_path)
        mask = np.asanyarray(image.dataobj) > 0
        zooms = [float(z) for z in image.header.get_zooms()[:3]]
        surface, result = mask_to_surface(mask, image.affine, zooms, cfg)
        if surface is None:
            result.path = output_path
            return result
        surface.save(output_path)
        result.path = output_path
        return result
    except (OSError, MemoryError, nib.filebasedimages.ImageFileError) as exc:
        # Anything else is a programming error and must not be hidden behind a status.
        return MeshResult(status="error", path=output_path, message=f"{type(exc).__name__}: {exc}",
                          config=asdict(cfg))


def _load_union_mask(seg_dir, anatomy_list):
    """Union the given anatomy masks in memory; no temporary NIfTI round trip."""
    union, affine, zooms = None, None, None
    for anatomy in anatomy_list:
        seg_file = os.path.join(seg_dir, f"{anatomy}.nii.gz")
        if not os.path.exists(seg_file):
            continue
        image = nib.load(seg_file)
        data = np.asanyarray(image.dataobj) > 0
        if union is None:
            union = np.zeros_like(data, dtype=bool)
            affine = image.affine
            zooms = [float(z) for z in image.header.get_zooms()[:3]]
        union |= data
    return union, affine, zooms


def _proc_anatomy(args):
    """Helper for multiprocessing: one label id -> one STL."""
    label_id, anatomies, seg_dir, models_dir = args

    anatomy_list = [anatomies] if isinstance(anatomies, str) else list(anatomies)
    output_name = f"label_{label_id}_{'_'.join(anatomy_list)}.stl"
    output_path = os.path.join(models_dir, output_name)
    name = "_".join(anatomy_list)

    if len(anatomy_list) == 1:
        seg_file = os.path.join(seg_dir, f"{anatomy_list[0]}.nii.gz")
        if not os.path.exists(seg_file):
            return None, name, {}
        result = generate_stl(seg_file, output_path)
    else:
        union, affine, zooms = _load_union_mask(seg_dir, anatomy_list)
        if union is None:
            return None, name, {}
        surface, result = mask_to_surface(union, affine, zooms)
        result.path = output_path
        if surface is not None:
            surface.save(output_path)

    return result.ok, name, {
        "status": result.status,
        "n_points": result.n_points,
        "n_faces": result.n_faces,
        "achieved_reduction": result.achieved_reduction,
        "hausdorff_mm": result.hausdorff_mm,
        "winding_flipped": result.winding_flipped,
        "message": result.message,
    }


def generate_stls_from_segmentations(seg_dir, models_dir, config_file, cfg=MeshConfig()):
    os.makedirs(models_dir, exist_ok=True)

    with open(config_file, 'r') as f:
        config = json.load(f)

    results = {'successful': 0, 'failed': [], 'meshes': {}, 'config': asdict(cfg)}

    tasks = [(label_id, anatomies, seg_dir, models_dir) for label_id, anatomies in config.items()]
    if not tasks:
        print("  No STL generation tasks found in config")
        return results

    num_workers = min(len(tasks), max(1, cpu_count() - 2))
    print(f"  Generating STLs in parallel using {num_workers} workers...")

    with Pool(num_workers) as pool:
        output = pool.map(_proc_anatomy, tasks)

    for success, name, detail in output:
        if detail:
            results['meshes'][name] = detail
        if success is True:
            results['successful'] += 1
        elif success is False:
            results['failed'].append(name)
            print(f"    ✗ {name}: {detail.get('status', 'error')} — {detail.get('message', '')}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--models_dir", required=True)
    parser.add_argument("--config", required=True)

    args = parser.parse_args()

    generate_stls_from_segmentations(args.seg_dir, args.models_dir, args.config)
