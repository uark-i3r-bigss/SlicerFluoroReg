#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import cc3d


def flatten_anatomy_config_labels(anatomy_config):
    """Return the unique segmentation labels referenced by an anatomy config."""
    labels = []
    for anatomies in anatomy_config.values():
        if isinstance(anatomies, str):
            labels.append(anatomies)
        else:
            labels.extend(anatomies)
    return sorted(set(labels))


def keep_largest_connected_component(mask):
    """Keep only the largest connected foreground component of a binary mask."""
    binary_mask = np.asarray(mask) > 0
    if not np.any(binary_mask):
        return binary_mask.astype(bool), {
            "num_components": 0,
            "kept_size": 0,
            "removed_size": 0,
        }

    # Use cc3d for much faster connected components
    # connectivity=26 is equivalent to binary_mask.ndim-connectivity for 3D
    labeled, num_components = cc3d.connected_components(binary_mask, connectivity=26, return_N=True)
    
    if num_components == 0:
        return binary_mask.astype(bool).copy(), {
            "num_components": 0,
            "kept_size": 0,
            "removed_size": 0,
        }

    # Extract statistics efficiently
    stats = cc3d.statistics(labeled)
    component_sizes = stats['voxel_counts'][1:] # index 0 is background
    
    if num_components <= 1:
        return binary_mask.astype(bool).copy(), {
            "num_components": num_components,
            "kept_size": int(component_sizes[0]) if num_components == 1 else 0,
            "removed_size": 0,
        }

    largest_label = 1 + np.argmax(component_sizes)
    cleaned = (labeled == largest_label)

    return cleaned, {
        "num_components": num_components,
        "kept_size": int(component_sizes.max()),
        "removed_size": int(component_sizes.sum() - component_sizes.max()),
    }


def cleanup_segmentation_file(seg_path):
    """Rewrite a segmentation file in-place after largest-component cleanup."""
    seg_path = Path(seg_path)
    img = nib.load(str(seg_path))
    # Use get_fdata(dtype=np.float32) to save memory and time
    data = img.get_fdata(dtype=np.float32)

    cleaned_mask, stats = keep_largest_connected_component(data > 0)
    if stats["num_components"] <= 1 or stats["removed_size"] == 0:
        return stats

    cleaned_data = np.zeros_like(data, dtype=img.get_data_dtype())
    foreground_value = data[data > 0]
    fill_value = foreground_value.flat[0] if foreground_value.size else 1
    cleaned_data[cleaned_mask] = fill_value

    nib.save(nib.Nifti1Image(cleaned_data, img.affine, img.header), str(seg_path))
    return stats


def cleanup_target_segmentations(segmentations_dir, anatomy_config):
    """Apply largest-component cleanup to the labels used by the landmark pipeline."""
    segmentations_dir = Path(segmentations_dir)
    cleanup_stats = {}

    for label in flatten_anatomy_config_labels(anatomy_config):
        seg_path = segmentations_dir / f"{label}.nii.gz"
        if not seg_path.exists():
            continue
        cleanup_stats[label] = cleanup_segmentation_file(seg_path)

    return cleanup_stats
