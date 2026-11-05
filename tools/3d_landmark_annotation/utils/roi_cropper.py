#!/usr/bin/env python3
import os
import json
import numpy as np
import nibabel as nib
from multiprocessing import Pool, cpu_count

from landmarks.fcsv_writer import read_fcsv, write_fcsv



def convert_voxels_to_mm(voxels, affine):
    """Convert voxel coordinates to RAS world coordinates"""
    if voxels.ndim == 1:
        v = np.append(voxels, 1)
        return (affine @ v)[:3]
    else:
        v = np.column_stack([voxels, np.ones(len(voxels))])
        return (affine @ v.T).T[:, :3]


def convert_mm_to_voxels(mm, affine):
    """Convert RAS world coordinates to voxel indices"""
    inv_affine = np.linalg.inv(affine)
    if mm.ndim == 1:
        m = np.append(mm, 1)
        return np.round((inv_affine @ m)[:3]).astype(int)
    else:
        m = np.column_stack([mm, np.ones(len(mm))])
        return np.round((inv_affine @ m.T).T[:, :3]).astype(int)


def write_roi_fcsv(center_lps, size_lps, output_path, label):
    """Write Slicer-compatible ROI FCSV"""
    header = [
        "# Markups fiducial file version = 5.8",
        "# CoordinateSystem = LPS",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
    ]
    
    half_size = np.array(size_lps) / 2.0
    min_lps = np.array(center_lps) - half_size
    max_lps = np.array(center_lps) + half_size
    
    rows = []
    rows.append(f"1,{center_lps[0]:.6f},{center_lps[1]:.6f},{center_lps[2]:.6f},0,0,0,1,1,1,0,{label}_center,,1")
    rows.append(f"2,{min_lps[0]:.6f},{min_lps[1]:.6f},{min_lps[2]:.6f},0,0,0,1,1,1,0,{label}_min,,1")
    rows.append(f"3,{max_lps[0]:.6f},{max_lps[1]:.6f},{max_lps[2]:.6f},0,0,0,1,1,1,0,{label}_max,,1")
    
    content = "\n".join(header + rows) + "\n"
    with open(output_path, 'w') as f:
        f.write(content)

def _proc_anatomy(args):
    """Multiprocessing worker for a single anatomy"""
    (anatomy_name, params, segmentations_dir, output_base_dir, 
     affine, shape, ct_data, all_generic_landmarks, landmark_config, label_to_id) = args
    
    labels = params['labels']
    margins = params['margin_mm'] # [L, R, P, A, S, I] in LPS
    relevant_generic_keys = params.get('landmarks', [])
    
    # Compute the tight bounding box sequentially inside each anatomy worker;
    # nesting another worker pool would add overhead. An anatomy whose masks are
    # absent must not leave an empty crop directory behind, so the output
    # directory is only created once a non-empty mask is found.
    v_min = np.array([float('inf'), float('inf'), float('inf')])
    v_max = np.array([float('-inf'), float('-inf'), float('-inf')])
    found = False
    present = False
    
    for label in labels:
        seg_path = os.path.join(segmentations_dir, f"{label}.nii.gz")
        if os.path.exists(seg_path):
            present = True
            try:
                data = nib.load(seg_path).get_fdata(dtype=np.float32)
                coords = np.array(np.where(data > 0))
                if coords.size > 0:
                    found = True
                    v_min = np.minimum(v_min, coords.min(axis=1))
                    v_max = np.maximum(v_max, coords.max(axis=1))
            except Exception:
                pass

    if not found:
        return ("empty" if present else "absent"), anatomy_name

    anatomy_dir = os.path.join(output_base_dir, anatomy_name)
    os.makedirs(anatomy_dir, exist_ok=True)
        
    # 2. Find LPS bounding box (via RAS world space)
    corners_v = []
    for i in [v_min[0], v_max[0]]:
        for j in [v_min[1], v_max[1]]:
            for k in [v_min[2], v_max[2]]:
                corners_v.append([i, j, k])
    
    corners_ras = convert_voxels_to_mm(np.array(corners_v), affine)
    ras_min = corners_ras.min(axis=0)
    ras_max = corners_ras.max(axis=0)
    
    lps_min = np.array([-ras_max[0], -ras_max[1], ras_min[2]])
    lps_max = np.array([-ras_min[0], -ras_min[1], ras_max[2]])
    
    lps_min_expanded = np.array([
        lps_min[0] - margins[1], 
        lps_min[1] - margins[3], 
        lps_min[2] - margins[5]
    ])
    lps_max_expanded = np.array([
        lps_max[0] + margins[0], 
        lps_max[1] + margins[2], 
        lps_max[2] + margins[4]
    ])
    
    ras_min_final = np.array([-lps_max_expanded[0], -lps_max_expanded[1], lps_min_expanded[2]])
    ras_max_final = np.array([-lps_min_expanded[0], -lps_min_expanded[1], lps_max_expanded[2]])
    
    corners_ras_final = []
    for x in [ras_min_final[0], ras_max_final[0]]:
        for y in [ras_min_final[1], ras_max_final[1]]:
            for z in [ras_min_final[2], ras_max_final[2]]:
                corners_ras_final.append([x, y, z])
    
    corners_v_final = convert_mm_to_voxels(np.array(corners_ras_final), affine)
    v_min_final = np.maximum(0, corners_v_final.min(axis=0))
    v_max_final = np.minimum(np.array(shape) - 1, corners_v_final.max(axis=0))
    
    slice_obj = (
        slice(v_min_final[0], v_max_final[0]+1),
        slice(v_min_final[1], v_max_final[1]+1),
        slice(v_min_final[2], v_max_final[2]+1)
    )
    
    new_affine = affine.copy()
    new_affine[:3, 3] = convert_voxels_to_mm(v_min_final, affine)
    
    # Output 1: Cropped CT
    crop_ct_data = ct_data[slice_obj]
    nib.save(nib.Nifti1Image(crop_ct_data, new_affine), os.path.join(anatomy_dir, f"{anatomy_name}_ct.nii.gz"))
    
    # Output 2: ROI FCSV
    center_lps = (lps_min_expanded + lps_max_expanded) / 2.0
    size_lps = lps_max_expanded - lps_min_expanded
    write_roi_fcsv(center_lps, size_lps, os.path.join(anatomy_dir, f"{anatomy_name}_roi.fcsv"), anatomy_name)
    
    # Output 3: Filtered Landmarks
    filtered_landmarks_to_write = {}
    for gk in relevant_generic_keys:
        if gk in all_generic_landmarks:
            user_label = landmark_config.get(gk, {}).get('label', gk)
            filtered_landmarks_to_write[user_label] = all_generic_landmarks[gk]
    
    # Crop membership has already decided what belongs here, so visibility is not re-applied.
    write_fcsv(
        filtered_landmarks_to_write,
        os.path.join(anatomy_dir, f"{anatomy_name}_landmarks.fcsv"),
        include_hidden=True,
    )
    
    # Output 4: Cropped Segmentation (Multi-label)
    combined_seg_data = np.zeros(shape, dtype=np.uint8)
    for label in labels:
        seg_path = os.path.join(segmentations_dir, f"{label}.nii.gz")
        if os.path.exists(seg_path):
            try:
                label_id = label_to_id.get(label, 1)
                data = nib.load(seg_path).get_fdata(dtype=np.float32)
                combined_seg_data[data > 0] = label_id
            except:
                pass
    
    crop_seg_data = combined_seg_data[slice_obj]
    nib.save(nib.Nifti1Image(crop_seg_data, new_affine), os.path.join(anatomy_dir, f"{anatomy_name}_seg.nii.gz"))
    return "ok", anatomy_name

def run_roi_cropping(ct_path, segmentations_dir, landmarks_path, output_base_dir, config_path, label_id_map_path):
    """Main function to perform ROI identification, cropping, and organization"""
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    landmark_config_path = os.path.join(os.path.dirname(script_dir), "configs", "landmark_config.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    with open(label_id_map_path, 'r') as f:
        label_to_id = {v: int(k) for k, v in json.load(f).items()}
        
    landmark_config = {}
    label_to_generic = {}
    if os.path.exists(landmark_config_path):
        with open(landmark_config_path, 'r') as f:
            landmark_config = json.load(f).get('landmarks', {})
            for gk, entry in landmark_config.items():
                label_to_generic[entry['label']] = gk
        
    os.makedirs(output_base_dir, exist_ok=True)
    ct_img = nib.load(ct_path)
    # Keeping CT in memory for all workers to share (Linux fork will handle it efficiently)
    ct_data = ct_img.get_fdata(dtype=np.float32)
    affine = ct_img.affine
    shape = ct_img.shape
    
    all_user_landmarks = read_fcsv(landmarks_path)
    all_generic_landmarks = {}
    for user_label, coords in all_user_landmarks.items():
        generic_key = label_to_generic.get(user_label, user_label)
        all_generic_landmarks[generic_key] = coords
    
    tasks = []
    for anatomy_name, params in config['anatomies'].items():
        tasks.append((anatomy_name, params, segmentations_dir, output_base_dir, 
                      affine, shape, ct_data, all_generic_landmarks, landmark_config, label_to_id))
    
    num_workers = min(len(tasks), max(1, cpu_count() - 2))
    print(f"  Cropping ROIs in parallel using {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = pool.map(_proc_anatomy, tasks)
        
    skipped = [name for status, name in results if status == "absent"]
    empty = [name for status, name in results if status == "empty"]
    cropped = [name for status, name in results if status == "ok"]
    if cropped:
        print(f"    Cropped {len(cropped)} anatomy group(s): {', '.join(sorted(cropped))}")
    if skipped:
        print(f"    Skipped {len(skipped)} group(s) with no segmentation in this scan: {', '.join(sorted(skipped))}")
    for name in sorted(empty):
        print(f"    Warning: segmentation for {name} is present but empty; no crop written")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Crop CT volumes and segmentations based on anatomical groups")
    parser.add_argument("--ct", required=True, help="Path to original CT NIfTI")
    parser.add_argument("--seg_dir", required=True, help="Directory containing individual segmentation masks")
    parser.add_argument("--landmarks", required=True, help="Path to comprehensive landmarks FCSV")
    parser.add_argument("--output_dir", required=True, help="Output base directory (cropped_data)")
    parser.add_argument("--config", required=True, help="Path to cropping config JSON")
    parser.add_argument("--label_map", required=True, help="Path to totalseg_all_labels.json")
    
    args = parser.parse_args()
    run_roi_cropping(args.ct, args.seg_dir, args.landmarks, args.output_dir, args.config, args.label_map)
