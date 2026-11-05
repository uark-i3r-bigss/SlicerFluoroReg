#!/usr/bin/env python3
"""
Process DICOM or NIfTI inputs through segmentation, mesh generation, landmark
detection, and anatomy-specific crop export.
"""

import os
import subprocess
import glob
import json
import argparse
import sys
import shutil
import re
import time
import logging
from pathlib import Path

import nibabel as nib
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger("3d_annotation")


def print(*args, **kwargs):
    """Override print to use logging for automatic timestamps."""
    # Handle flush and other kwargs that logger doesn't support
    msg = " ".join(map(str, args))
    logger.info(msg)


class StageTimer:
    def __init__(self):
        self.start_times = {}
        self.current_stage = None

    def start_stage(self, stage_id):
        now = time.time()
        if self.current_stage:
            duration = now - self.start_times[self.current_stage]
            print(f"[TIME] {self.current_stage}: {duration:.2f}s")
        self.current_stage = stage_id
        self.start_times[stage_id] = now
        print(f"[STAGE] {stage_id}")

    def end_last_stage(self):
        if self.current_stage:
            duration = time.time() - self.start_times[self.current_stage]
            print(f"[TIME] {self.current_stage}: {duration:.2f}s")
            self.current_stage = None


_STAGE_TIMER = StageTimer()


def run_command(cmd, log_prefix="  "):
    """Run a command and stream its output to the timestamped print."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    if process.stdout:
        for line in process.stdout:
            print(f"{log_prefix}{line.rstrip()}")
            
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return return_code


def find_executable(name):
    """Find an executable in PATH or in the same directory as the current python."""
    path = shutil.which(name)
    if path:
        return path
    
    # Try same directory as python
    py_bin_dir = os.path.dirname(sys.executable)
    path = os.path.join(py_bin_dir, name)
    if os.path.exists(path):
        return path
        
    return name


def build_totalsegmentator_command(
    ts_exe,
    nifti_file,
    segmentations_dir,
    scan_info,
    roi_subset=None,
    device=None,
    cuda_visible=None,
    force_split=False,
    fastest=False,
    body_seg=False,
):
    """Build the TotalSegmentator command while preserving explicit device choices."""
    ts_cmd = [ts_exe, "-i", nifti_file, "-o", segmentations_dir]
    if scan_info["modality"] == "MRI":
        ts_cmd.extend(["--task", "total_mr", "--fast"])
    else:
        ts_cmd.append("--fastest" if fastest else "--fast")
    if body_seg:
        ts_cmd.append("--body_seg")
    if force_split:
        ts_cmd.append("--force_split")
    if roi_subset:
        ts_cmd.extend(["--roi_subset"] + list(roi_subset))
    if device:
        ts_cmd.extend(["--device", device])
    elif cuda_visible:
        # If CUDA_VISIBLE_DEVICES is set, TotalSegmentator should use gpu:0
        # because the environment masks other GPUs.
        ts_cmd.extend(["--device", "gpu:0"])
    return ts_cmd


def build_combine_segmentations_command(
    python_executable,
    combine_script,
    input_folder,
    output_file,
    config_file,
):
    return [
        python_executable,
        combine_script,
        "-i",
        input_folder,
        "-o",
        output_file,
        "-c",
        config_file,
    ]


def _z_chunk_depth(shape, max_chunk_voxels=32_000_000):
    plane_voxels = max(1, int(shape[0]) * int(shape[1]))
    return max(1, int(max_chunk_voxels // plane_voxels))


def _iter_nifti_z_chunks(dataobj, shape, max_chunk_voxels=32_000_000):
    chunk_depth = _z_chunk_depth(shape, max_chunk_voxels=max_chunk_voxels)
    for z_start in range(0, int(shape[2]), chunk_depth):
        z_end = min(int(shape[2]), z_start + chunk_depth)
        yield np.asanyarray(dataobj[:, :, z_start:z_end])


def _nifti_min_max(dataobj, shape):
    vmin = np.inf
    vmax = -np.inf
    for chunk in _iter_nifti_z_chunks(dataobj, shape):
        if chunk.size == 0:
            continue
        vmin = min(vmin, float(np.nanmin(chunk)))
        vmax = max(vmax, float(np.nanmax(chunk)))
    return vmin, vmax


def _sample_nifti_proxy(dataobj, shape, max_sample_voxels=2_000_000):
    total_voxels = max(1, int(np.prod(shape, dtype=np.int64)))
    stride = max(1, int(np.ceil((total_voxels / max_sample_voxels) ** (1.0 / 3.0))))
    sample = np.asanyarray(dataobj[::stride, ::stride, ::stride], dtype=np.float32)
    return sample.ravel(), stride


def canonicalize_dcm2niix_stem(filename):
    """Collapse dcm2niix duplicate suffixes such as ...12345a / ...12345b."""
    if filename.endswith(".nii.gz"):
        stem = filename[:-7]
    elif filename.endswith(".json"):
        stem = filename[:-5]
    else:
        stem = Path(filename).stem
    return re.sub(r'(?<=\d)[a-z]$', '', stem)


def cleanup_stale_case_intermediates(case_output, keep_segmentations=False):
    """Remove root-level transient artifacts from incomplete or repeated runs.

    ``keep_segmentations`` preserves user-supplied masks for ``--skip-segmentation``
    runs, where ``segmentations/`` is an input rather than an intermediate.
    """
    transient_dirs = [
        "segmentations",
        "combined_segmentation",
        "combined_segmentation_models",
        "temp_models",
        "totalseg_models",
    ]
    if keep_segmentations:
        transient_dirs.remove("segmentations")
    for dir_name in transient_dirs:
        dir_path = os.path.join(case_output, dir_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)

    for pattern in ("*.stl", "*.nii.gz", "*.json"):
        for path in glob.glob(os.path.join(case_output, pattern)):
            os.remove(path)


def is_dicom_file(path):
    """Return True for common DICOM files, including extensionless exports."""
    path = Path(path)
    if not path.is_file():
        return False

    if path.suffix.lower() in (".dcm", ".dicom", ".ima"):
        return True

    try:
        with open(path, "rb") as handle:
            header = handle.read(132)
    except OSError:
        return False

    return len(header) >= 132 and header[128:132] == b"DICM"


def find_dicom_files(directory, limit=None):
    """Find DICOM files directly under a directory without relying on suffixes."""
    files = []
    try:
        entries = sorted(Path(directory).iterdir())
    except OSError:
        return files

    for entry in entries:
        if is_dicom_file(entry):
            files.append(entry)
            if limit is not None and len(files) >= limit:
                break
    return files


def is_nifti_file(path):
    """Return True only for supported NIfTI image filenames."""
    filename = Path(path).name.lower()
    return filename.endswith((".nii", ".nii.gz"))


def _nifti_stem(path):
    filename = os.path.basename(path)
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    if filename.endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def create_ap_flipped_nifti(source_path, output_path=None):
    """Create a new 180-degree axial-plane flipped CT and return its path."""
    image = nib.load(source_path)
    data = np.asanyarray(image.dataobj)
    flipped = np.flip(np.flip(data, axis=0), axis=1)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(source_path),
            f"{_nifti_stem(source_path)}_ap_flip.nii.gz",
        )

    # Intentionally keep the original affine. This makes the flipped CT become
    # the new canonical downstream volume instead of preserving the old world pose.
    out = nib.Nifti1Image(flipped, image.affine.copy(), header=image.header.copy())
    nib.save(out, output_path)
    return output_path


def deduplicate_metadata_sidecars(metadata_dir):
    seen_base_names = set()
    for config in sorted(glob.glob(os.path.join(metadata_dir, "*.json"))):
        filename = os.path.basename(config)
        if "anatomy_config" in filename:
            continue
        base_name = canonicalize_dcm2niix_stem(filename)
        if base_name in seen_base_names:
            print(f"  Removing duplicate metadata JSON: {filename}")
            os.remove(config)
            continue
        seen_base_names.add(base_name)


def detect_input_type(input_path):
    """Detect input type: single file, single DICOM folder, or directory with multiple cases"""
    input_path = Path(input_path)
    dicom_cases = []
    nifti_cases = []

    # Case 1: Single NIfTI file
    if input_path.is_file() and is_nifti_file(input_path):
        case_id = input_path.stem.replace('.nii', '')
        # Try to extract case ID from filename
        match = re.search(r'(case-\d+)', case_id)
        if match:
            case_id = match.group(1)
        
        nifti_cases.append({
            'case_id': case_id,
            'path': str(input_path),
            'filename': input_path.name
        })
        return dicom_cases, nifti_cases

    # Case 1b: Single DICOM file
    if input_path.is_file() and is_dicom_file(input_path):
        case_id = input_path.stem
        dicom_cases.append({
            'case_id': case_id,
            'path': str(input_path),
            'filename': input_path.name
        })
        return dicom_cases, nifti_cases

    # Case 2: Single DICOM folder
    if input_path.is_dir():
        # Check for DICOM files directly in this folder
        dcm_files = find_dicom_files(input_path, limit=1)
        if dcm_files:
            case_id = input_path.name
            dicom_cases.append({
                'case_id': case_id,
                'path': str(input_path),
                'dicom_dir': '.'
            })
            return dicom_cases, nifti_cases
        
        # Check for DICOM files in subdirectories (e.g., COR_ST_TORSO)
        dicom_subdirs = []
        for subdir in input_path.iterdir():
            if subdir.is_dir():
                dcm_files = find_dicom_files(subdir, limit=1)
                if dcm_files:
                    dicom_subdirs.append(subdir.name)
        
        if dicom_subdirs and not list(input_path.glob("case-*")):
            # Single case with DICOM subdirectories
            case_id = input_path.name
            dicom_cases.append({
                'case_id': case_id,
                'path': str(input_path),
                'dicom_dirs': dicom_subdirs
            })
            return dicom_cases, nifti_cases
    
    # Case 3 & 4: Directory with multiple cases
    if input_path.is_dir():
        # Check for multiple case directories with DICOM
        case_dirs = sorted(input_path.glob("case-*"))
        for case_dir in case_dirs:
            if case_dir.is_dir():
                # Check for COR_ST_TORSO or similar DICOM directories
                dicom_subdirs = []
                for subdir in case_dir.iterdir():
                    if subdir.is_dir():
                        dcm_files = find_dicom_files(subdir, limit=1)
                        if dcm_files:
                            dicom_subdirs.append(subdir.name)
                
                if dicom_subdirs:
                    case_id = case_dir.name
                    dicom_cases.append({
                        'case_id': case_id,
                        'path': str(case_dir),
                        'dicom_dirs': dicom_subdirs
                    })
        
        # Check for direct NIfTI files in input directory
        nifti_patterns = ["*.nii", "*.nii.gz"]
        for pattern in nifti_patterns:
            nifti_files = list(input_path.glob(pattern))
            for nifti_file in nifti_files:
                filename = nifti_file.name
                # Try to extract case ID from filename (e.g., case-115817_CT.nii.gz)
                match = re.search(r'(case-\d+)', filename)
                if match:
                    case_id = match.group(1)
                else:
                    # Use filename without extension as case ID
                    case_id = nifti_file.stem.replace('.nii', '')
                
                nifti_cases.append({
                    'case_id': case_id,
                    'path': str(nifti_file),
                    'filename': filename
                })
    
    return dicom_cases, nifti_cases


def display_processing_plan(dicom_cases, nifti_cases, output_dir):
    """Display what will be processed and ask for confirmation"""
    print("\n" + "="*60)
    print("PROCESSING PLAN")
    print("="*60)
    
    total_cases = len(dicom_cases) + len(nifti_cases)
    print(f"\nTotal cases to process: {total_cases}")
    
    if dicom_cases:
        print(f"\nDICOM cases ({len(dicom_cases)}):")
        for case in dicom_cases[:5]:  # Show first 5
            dicom_desc = case.get('filename') or ', '.join(
                case.get('dicom_dirs') or [case.get('dicom_dir', '.')]
            )
            print(f"  - {case['case_id']}: {dicom_desc}")
        if len(dicom_cases) > 5:
            print(f"  ... and {len(dicom_cases) - 5} more")
    
    if nifti_cases:
        print(f"\nNIfTI files ({len(nifti_cases)}):")
        for case in nifti_cases[:5]:  # Show first 5
            print(f"  - {case['case_id']}: {case['filename']}")
        if len(nifti_cases) > 5:
            print(f"  ... and {len(nifti_cases) - 5} more")
    
    print(f"\nOutput directory: {output_dir}")
    print("\nProcessing steps for each case:")
    print("  1. Convert DICOM to NIfTI (if needed)")
    print("  2. Run TotalSegmentator")
    print("  3. Combine segmentations")
    print("  4. Generate STL models")
    print("  5. Organize outputs")
    
    print("\n" + "="*60)
    
    response = input("\nProceed with processing? (y/n): ")
    return response.lower() == 'y'


def organize_output_structure(case_output_dir, case_id, view_type="COR"):
    """Reorganize the output into clean structure"""
    _STAGE_TIMER.start_stage("organize_output")
    print(f"  Organizing output structure for {case_id}...")
    
    # Create new directory structure
    metadata_dir = os.path.join(case_output_dir, "meta_data")
    totalseg_dir = os.path.join(case_output_dir, "TotalSegmentator_outputs")
    individual_seg_dir = os.path.join(totalseg_dir, "individual_seg")
    totalseg_models_dir = os.path.join(totalseg_dir, "models")
    models_dir = os.path.join(case_output_dir, "models")
    
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(individual_seg_dir, exist_ok=True)
    os.makedirs(totalseg_models_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Move and rename CT volume, removing dcm2niix duplicates in the case root.
    nifti_files = glob.glob(os.path.join(case_output_dir, "*.nii.gz"))
    root_ct_niftis = [
        nifti for nifti in nifti_files
        if "segmentation" not in os.path.basename(nifti).lower()
    ]
    seen_ct_bases = set()
    selected_ct = None
    for nifti in sorted(root_ct_niftis, key=lambda p: (len(os.path.basename(p)), os.path.basename(p))):
        filename = os.path.basename(nifti)
        base_name = canonicalize_dcm2niix_stem(filename)
        if base_name in seen_ct_bases:
            print(f"  Removing duplicate NIfTI: {filename}")
            os.remove(nifti)
            continue
        seen_ct_bases.add(base_name)
        if selected_ct is None:
            selected_ct = nifti
        else:
            print(f"  Removing extra root NIfTI: {filename}")
            os.remove(nifti)

    if selected_ct is not None:
        new_name = f"{case_id}_{view_type}_CT.nii.gz"
        shutil.move(selected_ct, os.path.join(metadata_dir, new_name))
    
    # Move combined segmentation
    combined_seg = os.path.join(case_output_dir, "combined_segmentation", "combined_segmentation.nii.gz")
    if os.path.exists(combined_seg):
        new_name = f"{case_id}_{view_type}_CT_seg.nii.gz"
        shutil.move(combined_seg, os.path.join(metadata_dir, new_name))
    
    # Move full TotalSegmentator segmentation to TotalSegmentator_outputs
    full_seg = os.path.join(case_output_dir, "combined_segmentation", "totalseg_full_labels.nii.gz")
    if os.path.exists(full_seg):
        new_name = f"{case_id}_totalseg_full_labels.nii.gz"
        shutil.move(full_seg, os.path.join(totalseg_dir, new_name))
    
    # Move individual segmentations to TotalSegmentator_outputs/individual_seg
    seg_source = os.path.join(case_output_dir, "segmentations")
    if os.path.exists(seg_source):
        for seg_file in os.listdir(seg_source):
            if seg_file.endswith(('.nii', '.nii.gz')):
                shutil.move(
                    os.path.join(seg_source, seg_file),
                    os.path.join(individual_seg_dir, seg_file)
                )
    
    # Move STL models
    possible_model_sources = [
        os.path.join(case_output_dir, "combined_segmentation_models"),
        os.path.join(case_output_dir, "temp_models") # Some temporary locations
    ]
    
    # Also check if any STLs are directly in case_output_dir (excluding those in organized folders)
    for model_file in os.listdir(case_output_dir):
        if model_file.endswith('.stl'):
            shutil.move(os.path.join(case_output_dir, model_file), os.path.join(models_dir, model_file))

    for models_source in possible_model_sources:
        if os.path.exists(models_source):
            for model_file in os.listdir(models_source):
                if model_file.endswith('.stl'):
                    source_path = os.path.join(models_source, model_file)
                    dest_path = os.path.join(models_dir, model_file)
                    shutil.move(source_path, dest_path)
    
    # Move TotalSegmentator models to TotalSegmentator_outputs/models
    totalseg_models_source = os.path.join(case_output_dir, "totalseg_models")
    if os.path.exists(totalseg_models_source):
        for model_file in os.listdir(totalseg_models_source):
            if model_file.endswith('.stl'):
                source_path = os.path.join(totalseg_models_source, model_file)
                dest_path = os.path.join(totalseg_models_dir, model_file)
                shutil.move(source_path, dest_path)
    
    # Move configuration files - avoid dcm2niix duplicate sidecars.
    config_files = glob.glob(os.path.join(case_output_dir, "*.json"))
    seen_base_names = set()
    for config in config_files:
        filename = os.path.basename(config)
        base_name = canonicalize_dcm2niix_stem(filename)
        if base_name in seen_base_names:
            print(f"  Removing duplicate JSON: {filename}")
            os.remove(config)
            continue
        seen_base_names.add(base_name)

        if "anatomy_config" in filename:
            new_name = f"{case_id}_anatomy_config.json"
            shutil.move(config, os.path.join(metadata_dir, new_name))
        else:
            shutil.move(config, os.path.join(metadata_dir, filename))

    deduplicate_metadata_sidecars(metadata_dir)
    
    # Remove transient working directories after their contents are organized.
    dirs_to_remove = [
        "combined_segmentation",
        "combined_segmentation_models", 
        "segmentations",
        "totalseg_models"
    ]
    for dir_name in dirs_to_remove:
        dir_path = os.path.join(case_output_dir, dir_name)
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    
    print(f"  ✓ Output organized for {case_id}")


def analyze_nifti_input(nifti_path):
    """
    Analyze a NIfTI file to characterize modality, physical coverage, and
    potential phantom/device characteristics.  Returns a summary dict that is
    also written to scan_analysis.json in the case output directory so
    downstream tools can read it.
    """
    img = nib.load(nifti_path)
    input_dtype = str(img.get_data_dtype())
    dataobj = img.dataobj
    shape = list(img.shape[:3])

    # True min/max from chunked reads — subsampling can silently miss sparse extreme values,
    # but materializing multi-GB volumes as float32 can exhaust memory on laptops.
    vmin, vmax = _nifti_min_max(dataobj, shape)

    # Subsample for mean/std/percentiles only.
    sample, _ = _sample_nifti_proxy(dataobj, shape)
    vmean = float(np.mean(sample))
    vstd = float(np.std(sample))
    _, p01, _, _, p99, p999 = [float(v) for v in np.percentile(sample, [0.1, 1, 5, 50, 99, 99.9])]
    air_frac = float(np.mean(sample < -200))
    bone_frac = float(np.mean(sample > 300))
    high_dense_frac = float(np.mean(sample > 3000))
    extreme_dense_frac = float(np.mean(sample > 8000))
    finite_sample = sample[np.isfinite(sample)]
    integer_like_frac = 0.0
    unique_sample_count = 0
    if finite_sample.size:
        integer_like_frac = float(np.mean(np.abs(finite_sample - np.rint(finite_sample)) < 1e-4))
        unique_sample_count = int(len(np.unique(finite_sample[:min(finite_sample.size, 300_000)])))

    voxel_sizes = [float(v) for v in np.abs(img.header.get_zooms()[:3])]
    extent_mm = [round(float(s * v), 1) for s, v in zip(shape, voxel_sizes)]

    # Foreground bounding box — downsample to ~128 voxels on longest axis for speed,
    # then find extent of non-air voxels so we report actual object size, not full FOV
    ds = max(1, max(shape) // 128)
    data_ds = np.asanyarray(dataobj[::ds, ::ds, ::ds], dtype=np.float32)
    fg_ds = data_ds > -200
    fg_any_x = fg_ds.any(axis=(1, 2))
    fg_any_y = fg_ds.any(axis=(0, 2))
    fg_any_z = fg_ds.any(axis=(0, 1))
    if fg_any_x.any() and fg_any_y.any() and fg_any_z.any():
        x_ids = np.where(fg_any_x)[0]
        y_ids = np.where(fg_any_y)[0]
        z_ids = np.where(fg_any_z)[0]
        foreground_extent_mm = [
            round(float((x_ids[-1] - x_ids[0] + 1) * ds * voxel_sizes[0]), 1),
            round(float((y_ids[-1] - y_ids[0] + 1) * ds * voxel_sizes[1]), 1),
            round(float((z_ids[-1] - z_ids[0] + 1) * ds * voxel_sizes[2]), 1),
        ]
    else:
        foreground_extent_mm = extent_mm[:]

    is_segmentation_label = (
        vmin >= 0
        and vmax <= 255
        and integer_like_frac > 0.999
        and unique_sample_count <= 64
        and vstd < 30
    )

    if is_segmentation_label:
        modality = "segmentation_label"
    elif vmin < -500:
        modality = "CT"
    elif vmin >= -50 and vmax > 100:
        modality = "MRI"
    else:
        modality = "unknown"

    # Regional determination uses foreground object extent, not full FOV
    sorted_fg_extent = sorted(foreground_extent_mm)
    is_regional = max(foreground_extent_mm) < 350.0 or sorted_fg_extent[1] < 300.0

    phantom_flags = []
    is_likely_phantom = False
    is_likely_device = False

    if modality == "CT":
        flag_count = 0

        if vmax > 5000:
            if p999 > 8000 or extreme_dense_frac >= 0.001:
                is_likely_device = True
                phantom_flags.append(
                    f"substantial extreme hyper-dense material "
                    f"(p99.9={p999:.0f} HU, {extreme_dense_frac*100:.3f}% of sampled voxels > 8000 HU)"
                )
                flag_count += 1
            elif p999 > 3000 or high_dense_frac >= 0.01:
                phantom_flags.append(
                    f"substantial hyper-dense material "
                    f"(p99.9={p999:.0f} HU, {high_dense_frac*100:.3f}% of sampled voxels > 3000 HU)"
                )
                flag_count += 1
            else:
                phantom_flags.append(
                    f"sparse hyper-dense voxels detected (max={vmax:.0f} HU, "
                    f"{high_dense_frac*100:.4f}% of sampled voxels > 3000 HU); "
                    "treating these as markers/outliers rather than a whole-volume device signal"
                )

        if air_frac < 0.05:
            phantom_flags.append(
                f"little air detected ({air_frac*100:.1f}% of voxels < -200 HU; real CT typically > 10%)"
            )
            flag_count += 1
        if bone_frac < 0.005:
            phantom_flags.append(
                f"little bone-density material ({bone_frac*100:.2f}% of voxels > 300 HU)"
            )
            flag_count += 1
        if not is_likely_device and (vstd < 50 or (vmax - vmin) < 200):
            is_likely_device = True
            phantom_flags.append(
                "very uniform intensity - may be a non-biological test object or medical device"
            )
            flag_count += 1
        elif vstd < 100 and (vmax - vmin) < 500:
            phantom_flags.append(
                f"narrow intensity range (std={vstd:.0f} HU, range [{vmin:.0f}, {vmax:.0f}] HU)"
            )
            flag_count += 1

        is_likely_phantom = flag_count >= 2 and not is_likely_device
    elif is_segmentation_label:
        phantom_flags.append(
            f"segmentation label map detected (integer labels in [{vmin:.0f}, {vmax:.0f}], "
            f"sampled unique labels <= {unique_sample_count}); use the matching CT/MR image as input"
        )

    return {
        "modality": modality,
        "input_dtype": input_dtype,
        "shape": shape,
        "extent_mm": extent_mm,
        "foreground_extent_mm": foreground_extent_mm,
        "voxel_sizes_mm": voxel_sizes,
        "intensity_min": round(vmin, 1),
        "intensity_max": round(vmax, 1),
        "intensity_mean": round(vmean, 1),
        "intensity_std": round(vstd, 1),
        "intensity_p01": round(p01, 1),
        "intensity_p99": round(p99, 1),
        "intensity_p999": round(p999, 1),
        "air_fraction_lt_minus_200": round(air_frac, 5),
        "bone_fraction_gt_300": round(bone_frac, 5),
        "high_dense_fraction_gt_3000": round(high_dense_frac, 8),
        "extreme_dense_fraction_gt_8000": round(extreme_dense_frac, 8),
        "integer_like_fraction": round(integer_like_frac, 5),
        "unique_sample_count": unique_sample_count,
        "is_regional": is_regional,
        "is_segmentation_label": is_segmentation_label,
        "input_valid_for_processing": not is_segmentation_label,
        "is_likely_phantom": is_likely_phantom,
        "is_likely_device": is_likely_device,
        "phantom_flags": phantom_flags,
    }


def _print_scan_analysis(info):
    m = info["modality"]
    sh = info["shape"]
    ext = info["extent_mm"]
    fg_ext = info.get("foreground_extent_mm", ext)
    vs = info["voxel_sizes_mm"]
    flags = info["phantom_flags"]
    is_reg = info["is_regional"]
    is_ph = info["is_likely_phantom"]
    is_dev = info["is_likely_device"]

    print(
        f"    Modality:  {m}  "
        f"(intensity [{info['intensity_min']:.0f}, {info['intensity_max']:.0f}], "
        f"std={info['intensity_std']:.0f})"
    )
    print(
        f"    Volume:    {sh[0]} x {sh[1]} x {sh[2]} voxels  "
        f"({vs[0]:.2f} x {vs[1]:.2f} x {vs[2]:.2f} mm/vox)"
    )
    print(f"    FOV:       {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
    cov_note = "regional scan" if is_reg else "adequate coverage"
    print(f"    Object:    ~{fg_ext[0]:.0f} x {fg_ext[1]:.0f} x {fg_ext[2]:.0f} mm  [{cov_note}]")
    for flag in flags:
        print(f"    WARNING:   {flag}")
    if is_dev:
        print(
            "    ASSESSMENT: likely a medical device or test object — "
            "TotalSegmentator will almost certainly not segment it"
        )
    elif info.get("is_segmentation_label"):
        print(
            "    ASSESSMENT: segmentation label map — "
            "use the source CT/MR image instead of this label file"
        )
    elif is_ph:
        print(
            "    ASSESSMENT: likely a physical phantom — "
            "TotalSegmentator may not produce useful segmentations"
        )
    elif m == "MRI":
        print("    ASSESSMENT: MRI volume — using total_mr task for TotalSegmentator")
    elif is_reg:
        print("    ASSESSMENT: regional scan — only anatomy within the FOV will be segmented")
    else:
        print("    ASSESSMENT: standard CT — proceeding normally")


def _get_non_empty_segmentations(segmentations_dir, anatomy_config):
    """Return the set of anatomy names that have non-zero voxels after TotalSegmentator."""
    all_targets = []
    for v in anatomy_config.values():
        if isinstance(v, str):
            all_targets.append(v)
        else:
            all_targets.extend(v)

    non_empty = set()
    for anatomy in all_targets:
        seg_path = os.path.join(segmentations_dir, f"{anatomy}.nii.gz")
        if not os.path.exists(seg_path):
            continue
        img = nib.load(seg_path)
        if np.count_nonzero(np.asanyarray(img.dataobj)) > 0:
            non_empty.add(anatomy)
    return non_empty


def _filter_anatomy_config(anatomy_config, non_empty_segs):
    """Return a new anatomy config limited to anatomies present in non_empty_segs."""
    filtered = {}
    for label_id, anatomies in anatomy_config.items():
        if isinstance(anatomies, str):
            if anatomies in non_empty_segs:
                filtered[label_id] = anatomies
        else:
            if any(a in non_empty_segs for a in anatomies):
                filtered[label_id] = anatomies
    return filtered


def process_dicom_case(
    case_info,
    output_dir,
    skip_steps,
    ap_flip=False,
    roi_subset=None,
    device=None,
    force_split=False,
    fastest=False,
    body_seg=False,
):
    """Process a single DICOM case"""
    case_id = case_info['case_id']
    case_dir = case_info['path']
    
    print(f"\nProcessing DICOM case: {case_id}")
    
    # For single cases, output_dir might already include case_id
    if output_dir.endswith(case_id):
        case_output = output_dir
    else:
        case_output = os.path.join(output_dir, case_id)
    os.makedirs(case_output, exist_ok=True)
    cleanup_stale_case_intermediates(case_output, keep_segmentations=skip_steps['segmentation'])
    
    # Handle different DICOM directory structures
    if 'dicom_dirs' in case_info:
        # Multiple subdirectories case (e.g., COR_ST_TORSO)
        dicom_dir = os.path.join(case_dir, case_info['dicom_dirs'][0])
    elif 'dicom_dir' in case_info:
        dicom_dir = os.path.join(case_dir, case_info['dicom_dir'])
    else:
        # Single directory case (DICOM files directly in the folder)
        dicom_dir = case_dir
    
    # Convert DICOM to NIfTI
    if not skip_steps['conversion']:
        _STAGE_TIMER.start_stage("analyze_input")
        print(f"  Converting DICOM to NIfTI...")
        try:
            dcm_exe = find_executable('dcm2niix')
            dcm2niix_cmd = [dcm_exe, '-z', 'y', '-o', case_output, dicom_dir]
            run_command(dcm2niix_cmd, log_prefix="    [dcm2niix] ")
            print(f"  ✓ DICOM conversion completed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ DICOM conversion failed (exit code {e.returncode})")
            return False
    
    # Find the converted NIfTI file
    nifti_files = glob.glob(os.path.join(case_output, "*.nii.gz"))
    if not nifti_files:
        print("  ✗ No NIfTI file found after conversion")
        return False
    
    # Prefer the one without 'a' if multiple exist (to avoid duplicates)
    nifti_files.sort(key=len)
    nifti_file = nifti_files[0]
    
    # Continue with segmentation
    if ap_flip:
        print("  Creating AP-flipped CT before segmentation...")
        flipped_nifti = create_ap_flipped_nifti(nifti_file)
        if os.path.abspath(flipped_nifti) != os.path.abspath(nifti_file) and os.path.exists(nifti_file):
            os.remove(nifti_file)
        nifti_file = flipped_nifti
        print(f"  ✓ AP-flipped CT created: {os.path.basename(nifti_file)}")

    return process_nifti_file(
        nifti_file,
        case_output,
        case_id,
        skip_steps,
        roi_subset=roi_subset,
        device=device,
        force_split=force_split,
        fastest=fastest,
        body_seg=body_seg,
    )


def process_nifti_case(
    case_info,
    output_dir,
    skip_steps,
    ap_flip=False,
    roi_subset=None,
    device=None,
    force_split=False,
    fastest=False,
    body_seg=False,
):
    """Process a single NIfTI case"""
    case_id = case_info['case_id']
    nifti_path = case_info['path']
    
    print(f"\nProcessing NIfTI case: {case_id}")
    
    # For single cases, output_dir might already include case_id
    if output_dir.endswith(case_id):
        case_output = output_dir
    else:
        case_output = os.path.join(output_dir, case_id)
    os.makedirs(case_output, exist_ok=True)
    cleanup_stale_case_intermediates(case_output, keep_segmentations=skip_steps['segmentation'])
    
    if ap_flip:
        output_nifti = os.path.join(case_output, f"{_nifti_stem(nifti_path)}_ap_flip.nii.gz")
        print("  Creating AP-flipped CT before segmentation...")
        create_ap_flipped_nifti(nifti_path, output_nifti)
        print(f"  ✓ AP-flipped CT created: {os.path.basename(output_nifti)}")
    else:
        output_nifti = os.path.join(case_output, os.path.basename(nifti_path))
        shutil.copy2(nifti_path, output_nifti)

    return process_nifti_file(
        output_nifti,
        case_output,
        case_id,
        skip_steps,
        roi_subset=roi_subset,
        device=device,
        force_split=force_split,
        fastest=fastest,
        body_seg=body_seg,
    )


def process_nifti_file(
    nifti_file,
    case_output,
    case_id,
    skip_steps,
    roi_subset=None,
    device=None,
    force_split=False,
    fastest=False,
    body_seg=False,
):
    """Process a NIfTI file through segmentation and model generation"""

    # Analyze input before anything else so the info is always written to disk
    _STAGE_TIMER.start_stage("analyze_input")
    print("  Analyzing input volume...")
    scan_info = analyze_nifti_input(nifti_file)
    _print_scan_analysis(scan_info)
    scan_analysis_path = os.path.join(case_output, "scan_analysis.json")
    with open(scan_analysis_path, 'w') as f:
        json.dump(scan_info, f, indent=2)

    if scan_info.get("is_segmentation_label"):
        print("  ✗ Input appears to be a segmentation label map, not an image volume.")
        print("    Use the corresponding CT/MR image as the input and keep label maps for validation or manual masks.")
        return False

    # Create anatomy configuration - Labels updated for TS v2.0
    anatomy_config = {
        "1": ["hip_left", "hip_right"],
        "2": "femur_left",
        "3": "femur_right",
        "4": "vertebrae_L1",
        "5": "vertebrae_L2",
        "6": "vertebrae_L3",
        "7": "vertebrae_L4",
        "8": "vertebrae_L5",
        "9": "vertebrae_S1",
        "10": "sacrum"
    }
    
    config_file = os.path.join(case_output, "anatomy_config.json")
    with open(config_file, 'w') as f:
        json.dump(anatomy_config, f, indent=4)
    
    # Run TotalSegmentator
    segmentations_dir = os.path.join(case_output, 'segmentations')
    if not skip_steps['segmentation']:
        _STAGE_TIMER.start_stage("totalseg")
        print(f"  Running TotalSegmentator...")
        try:
            ts_exe = find_executable('TotalSegmentator')
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            ts_cmd = build_totalsegmentator_command(
                ts_exe,
                nifti_file,
                segmentations_dir,
                scan_info,
                roi_subset=roi_subset,
                device=device,
                cuda_visible=cuda_visible,
                force_split=force_split,
                fastest=fastest,
                body_seg=body_seg,
            )
            if scan_info['modality'] == 'MRI':
                print("    Using total_mr task (MRI detected)")
            if fastest and scan_info['modality'] != 'MRI':
                print("    Fastest mode enabled for lower memory usage (6mm)")
            if body_seg:
                print("    Body crop enabled for lower memory usage")
            if force_split:
                print("    Force split enabled for lower memory usage")
            if roi_subset:
                print(f"    ROI subset: {', '.join(roi_subset)}")
            
            # Determine device
            if device:
                print(f"  Explicitly targeting TotalSegmentator device: {device}")
            elif cuda_visible:
                print(f"  Explicitly targeting GPU 0 via TotalSegmentator --device (CUDA_VISIBLE_DEVICES={cuda_visible})")
            
            run_command(ts_cmd, log_prefix="    [totalseg] ")
            print("  ✓ TotalSegmentator completed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ TotalSegmentator failed: {e}")
            return False

        print("  Cleaning target segmentations with largest-component filtering...")
        from utils.segmentation_cleanup import cleanup_target_segmentations
        cleanup_stats = cleanup_target_segmentations(segmentations_dir, anatomy_config)
        removed_any = False
        for label, stats in cleanup_stats.items():
            if stats['removed_size'] > 0:
                removed_any = True
                print(
                    f"    {label}: kept largest component "
                    f"({stats['kept_size']} voxels), removed {stats['removed_size']} voxels "
                    f"across {stats['num_components'] - 1} extra components"
                )
        if not removed_any:
            print("    No extra components found in target anatomies")

        # Detect which target anatomies were actually segmented and filter config
        non_empty_segs = _get_non_empty_segmentations(segmentations_dir, anatomy_config)
        all_targets = [
            a for v in anatomy_config.values()
            for a in (v if isinstance(v, list) else [v])
        ]
        missing_segs = set(all_targets) - non_empty_segs
        if non_empty_segs:
            print(f"    Detected {len(non_empty_segs)} non-empty anatomy segmentation(s): {', '.join(sorted(non_empty_segs))}")
            if missing_segs:
                print(f"    Skipping {len(missing_segs)} empty/absent anatomy segmentation(s): {', '.join(sorted(missing_segs))}")
            effective_config = _filter_anatomy_config(anatomy_config, non_empty_segs)
            with open(config_file, 'w') as f:
                json.dump(effective_config, f, indent=4)
        else:
            print("  ✗ TotalSegmentator found no target anatomy segmentations in this scan")
            if scan_info.get('is_likely_device'):
                print("    Pre-scan analysis flagged this as a LIKELY MEDICAL DEVICE or TEST OBJECT.")
                print("    TotalSegmentator cannot segment non-biological materials.")
            elif scan_info.get('is_likely_phantom'):
                print("    Pre-scan analysis flagged this as a LIKELY PHANTOM.")
                print("    TotalSegmentator is trained on real human tissue and cannot segment")
                print("    phantom materials with non-biological HU characteristics.")
            elif scan_info.get('modality') == 'MRI':
                print("    MRI was detected — verify that the scan covers the expected anatomy.")
            elif scan_info.get('is_regional'):
                print("    Regional scan detected — the FOV may not cover the expected anatomy.")
            else:
                print("    Unexpected result for a standard CT scan.")
                print("    Check that the scan covers the hip, femur, and/or lumbar anatomy.")
            if scan_info.get('phantom_flags'):
                print("    Scan flags:")
                for flag in scan_info['phantom_flags']:
                    print(f"      - {flag}")
            if not roi_subset:
                print("    Suggestions:")
                print("      - For a hip/femur region: use --roi-subset hip_left hip_right femur_left femur_right")
            print("      - For phantoms: provide manually segmented NIfTI masks instead.")
            return False
    else:
        # --skip-segmentation: `segmentations/` holds user-supplied or previously
        # generated masks. Restrict the anatomy config to what is actually there.
        if not os.path.isdir(segmentations_dir):
            print(f"  ✗ --skip-segmentation needs existing masks in {segmentations_dir}")
            return False
        non_empty_segs = _get_non_empty_segmentations(segmentations_dir, anatomy_config)
        if not non_empty_segs:
            print(f"  ✗ No non-empty target anatomy masks found in {segmentations_dir}")
            print("    Expected files such as hip_left.nii.gz, femur_left.nii.gz, vertebrae_L1.nii.gz")
            return False
        print(
            f"  Using {len(non_empty_segs)} pre-existing segmentation(s): "
            f"{', '.join(sorted(non_empty_segs))}"
        )
        effective_config = _filter_anatomy_config(anatomy_config, non_empty_segs)
        with open(config_file, 'w') as f:
            json.dump(effective_config, f, indent=4)

    # Combine segmentations
    if not skip_steps['combination']:
        _STAGE_TIMER.start_stage("combine_segmentations")
        print(f"  Combining segmentations...")
        combined_dir = os.path.join(case_output, "combined_segmentation")
        os.makedirs(combined_dir, exist_ok=True)
        
        combined_output = os.path.join(combined_dir, 'combined_segmentation.nii.gz')
        
        # Get the combine script path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        combine_script = os.path.join(script_dir, 'utils', 'totalsegmentator_postprocess.py')
        
        try:
            result = subprocess.run(
                build_combine_segmentations_command(
                    sys.executable,
                    combine_script,
                    segmentations_dir,
                    combined_output,
                    config_file,
                ),
                check=True,
                capture_output=True,
            )
            print("  ✓ Segmentation combination completed")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Combination failed: {e}")
            return False
        
        # Also create a full segmentation with all TotalSegmentator labels preserved
        print(f"  Creating full TotalSegmentator segmentation...")
        full_seg_output = os.path.join(combined_dir, 'totalseg_full_labels.nii.gz')
        
        # Use the all labels config
        all_labels_config = os.path.join(script_dir, 'configs', 'totalseg_all_labels.json')
        
        try:
            result = subprocess.run(
                build_combine_segmentations_command(
                    sys.executable,
                    combine_script,
                    segmentations_dir,
                    full_seg_output,
                    all_labels_config,
                ),
                check=True,
                capture_output=True,
            )
            print("  ✓ Full TotalSegmentator segmentation created")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Full segmentation creation failed: {e}")
            # Non-fatal - continue processing
    
    # Generate STL models
    if not skip_steps['combination']:
        _STAGE_TIMER.start_stage("generate_stl")
        print(f"  Generating STL models for selected anatomies...")
        from utils.stl_generator import generate_stls_from_segmentations
        
        seg_dir = os.path.join(case_output, "segmentations")
        models_dir = os.path.join(case_output, "models")
        
        if os.path.exists(seg_dir):
            results = generate_stls_from_segmentations(
                seg_dir, models_dir, config_file
            )
            
            if results['successful'] > 0:
                print(f"  ✓ Generated {results['successful']} STL models")
            else:
                print(f"  ✗ No STL models generated for target anatomies")
                print(f"  ✗ Landmark pipeline cannot continue without at least one anatomy STL model")
                return False
        
        # Generate STL models for ALL TotalSegmentator anatomies
        print(f"  Generating STL models for all anatomies...")
        all_models_dir = os.path.join(case_output, "totalseg_models")
        
        # Get the all labels config path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        all_labels_config = os.path.join(script_dir, 'configs', 'totalseg_all_labels.json')
        
        if os.path.exists(seg_dir):
            results_all = generate_stls_from_segmentations(
                seg_dir, all_models_dir, all_labels_config
            )
            
            if results_all['successful'] > 0:
                print(f"  ✓ Generated {results_all['successful']} STL models for all anatomies")
            else:
                print(f"  ✗ No STL models generated for all anatomies")

        # Record the resolved meshing configuration and per-mesh outcome: a landmark
        # coordinate cannot be interpreted without knowing which surface produced it.
        mesh_report = {
            "target_anatomy": results if os.path.exists(seg_dir) else {},
            "all_labels": results_all if os.path.exists(seg_dir) else {},
        }
        with open(os.path.join(case_output, "mesh_report.json"), "w") as handle:
            json.dump(mesh_report, handle, indent=2)
    
    # Organize output structure first
    organize_output_structure(case_output, case_id)
    
    # Generate comprehensive landmarks (pelvis + lumbar + femur) after organization
    if not skip_steps['combination']:
        _STAGE_TIMER.start_stage("generate_landmarks")
        print(f"  Generating comprehensive landmarks...")
        try:
            from landmarks.fcsv_writer import create_comprehensive_landmark_file
            
            # Look for models in organized structure (no processed_data layer)
            organized_models_dir = os.path.join(case_output, "models")
            
            if os.path.exists(organized_models_dir):
                landmarks_dir = os.path.join(case_output, "landmarks")
                landmark_file, landmark_counts = create_comprehensive_landmark_file(organized_models_dir, landmarks_dir, case_id)
                total_landmarks = sum(landmark_counts.values())
                print(f"  ✓ Generated landmarks: {os.path.basename(landmark_file)} ({total_landmarks} total)")
                if total_landmarks == 0:
                    print("  ✗ Landmark file is empty — no landmarks detected for any anatomy")
                    return False
                scan_info["landmark_counts"] = landmark_counts
                scan_info["total_landmarks"] = total_landmarks
                with open(scan_analysis_path, 'w') as f:
                    json.dump(scan_info, f, indent=2)
            else:
                print(f"  ✗ Models directory not found for landmark generation")
                return False
        except Exception as e:
            print(f"  ✗ Landmark generation failed: {e}")
            return False
            
    # Run ROI identification and cropping
    if not skip_steps['combination']:
        _STAGE_TIMER.start_stage("crop_roi")
        print(f"  Running ROI identification and cropping...")
        try:
            from utils.roi_cropper import run_roi_cropping
            
            # Paths in organized structure
            ct_path = glob.glob(os.path.join(case_output, "meta_data", "*_CT.nii.gz"))[0]
            seg_dir = os.path.join(case_output, "TotalSegmentator_outputs", "individual_seg")
            landmarks_path = os.path.join(case_output, "landmarks", f"{case_id}_landmarks.fcsv")
            cropped_data_dir = os.path.join(case_output, "cropped_data")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            crop_config = os.path.join(script_dir, "configs", "lumbar_pelvis_crop_config.json")
            label_map = os.path.join(script_dir, "configs", "totalseg_all_labels.json")
            
            if os.path.exists(ct_path) and os.path.exists(seg_dir) and os.path.exists(landmarks_path):
                run_roi_cropping(ct_path, seg_dir, landmarks_path, cropped_data_dir, crop_config, label_map)
                print(f"  ✓ ROI identification and cropping completed (saved to cropped_data/)")
            else:
                print(f"  ✗ Required files for cropping not found")
        except Exception as e:
            print(f"  ✗ ROI cropping failed: {e}")
    
    _STAGE_TIMER.end_last_stage()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process medical scans (DICOM or NIfTI) with automatic structure detection"
    )
    parser.add_argument("input", help="Input: single NIfTI file, single DICOM folder, or directory containing multiple cases")
    parser.add_argument("output_dir", help="Output directory for processed data")
    parser.add_argument("--skip-conversion", action="store_true", help="Skip DICOM to NIfTI conversion")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip TotalSegmentator")
    parser.add_argument("--skip-combination", action="store_true", help="Skip segmentation combination")
    parser.add_argument("--no-confirm", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--ap-flip", action="store_true", help="Create a new _ap_flip CT and use it as the default input before segmentation")
    parser.add_argument("--device", help="Explicitly set TotalSegmentator device (e.g. cpu, gpu, mps, gpu:0). If not set, uses CUDA_VISIBLE_DEVICES when present.")
    parser.add_argument("--force-split", action="store_true", help="Pass --force_split to TotalSegmentator to reduce memory usage on large volumes.")
    parser.add_argument("--fastest", action="store_true", help="Pass --fastest to TotalSegmentator for the lowest-resolution, lowest-memory CT model.")
    parser.add_argument("--body-seg", action="store_true", help="Pass --body_seg to TotalSegmentator to crop the CT to the rough body region first.")
    parser.add_argument(
        "--roi-subset",
        nargs="+",
        metavar="ANATOMY",
        default=None,
        help=(
            "Limit TotalSegmentator to these anatomy names only "
            "(e.g. --roi-subset hip_left hip_right femur_left femur_right). "
            "Useful for regional or phantom CTs."
        ),
    )
    
    args = parser.parse_args()
    
    # Detect input type
    print("Analyzing input...")
    if args.ap_flip:
        print("AP flip enabled: creating a new _ap_flip CT before processing and using it for all downstream steps.")
    if args.fastest:
        print("Fastest mode enabled: TotalSegmentator will use the 6mm model where supported.")
    if args.body_seg:
        print("Body crop enabled: TotalSegmentator will crop to the rough body region first.")
    if args.roi_subset:
        print(f"ROI subset enabled: TotalSegmentator will target only: {', '.join(args.roi_subset)}")
    dicom_cases, nifti_cases = detect_input_type(args.input)
    
    if not dicom_cases and not nifti_cases:
        print("ERROR: No valid input found")
        print("Supported inputs:")
        print("  - Single NIfTI file (e.g., /path/to/case.nii.gz)")
        print("  - Single DICOM folder (e.g., /path/to/dicom-series)")
        print("  - Directory with multiple DICOM case folders")
        print("  - Directory with multiple NIfTI files")
        return
    
    # Display plan and get confirmation
    if not args.no_confirm:
        if not display_processing_plan(dicom_cases, nifti_cases, args.output_dir):
            print("Processing cancelled.")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all cases
    skip_steps = {
        'conversion': args.skip_conversion,
        'segmentation': args.skip_segmentation,
        'combination': args.skip_combination
    }
    
    successful = 0
    failed = 0
    
    # Process DICOM cases
    for case in dicom_cases:
        if process_dicom_case(
            case,
            args.output_dir,
            skip_steps,
            ap_flip=args.ap_flip,
            roi_subset=args.roi_subset,
            device=args.device,
            force_split=args.force_split,
            fastest=args.fastest,
            body_seg=args.body_seg,
        ):
            successful += 1
        else:
            failed += 1

    # Process NIfTI cases
    for case in nifti_cases:
        if process_nifti_case(
            case,
            args.output_dir,
            skip_steps,
            ap_flip=args.ap_flip,
            roi_subset=args.roi_subset,
            device=args.device,
            force_split=args.force_split,
            fastest=args.fastest,
            body_seg=args.body_seg,
        ):
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output saved to: {args.output_dir}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
