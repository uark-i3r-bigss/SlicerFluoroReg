import nibabel as nib
import numpy as np
import os
import argparse
import json
from multiprocessing import Pool, cpu_count

def _load_single_seg(args):
    """Worker to load a single segmentation file and return the indices where it is non-zero."""
    anatomy, input_folder, label = args
    file_path = os.path.join(input_folder, f"{anatomy}.nii.gz")
    
    if not os.path.exists(file_path):
        return None
    
    try:
        img = nib.load(file_path)
        # Using get_fdata(dtype=np.uint8) to save memory and time
        data = img.get_fdata(dtype=np.float32)
        indices = np.where(data > 0)
        return indices, label, img.affine, img.header
    except Exception as e:
        print(f"    ✗ Error loading {file_path}: {e}")
        return None

def combine_segmentations(input_dict, input_folder, output_file):
    tasks = []
    for label, anatomies in input_dict.items():
        if isinstance(anatomies, str):
            anatomies = [anatomies]
        for anatomy in anatomies:
            tasks.append((anatomy, input_folder, int(label)))

    if not tasks:
        print("  ✗ No segmentation tasks found in config")
        return None, None

    num_workers = min(len(tasks), max(1, cpu_count() - 2))
    print(f"  Loading {len(tasks)} segmentations in parallel using {num_workers} workers...")
    
    combined_data = None
    reference_affine = None
    reference_header = None
    
    with Pool(num_workers) as pool:
        results = pool.map(_load_single_seg, tasks)
    
    for res in results:
        if res is None:
            continue
            
        indices, label, affine, header = res
        
        if combined_data is None:
            # We need the shape. Let's get it from the first valid result.
            # Using int32 to support many labels
            shape = header.get_data_shape()
            combined_data = np.zeros(shape, dtype=np.int32)
            reference_affine = affine
            reference_header = header
            
        combined_data[indices] = label

    if combined_data is not None:
        combined_img = nib.Nifti1Image(combined_data, reference_affine, reference_header)
        combined_img.set_data_dtype(np.int32)
        nib.save(combined_img, output_file)
        return combined_data, reference_affine
    else:
        print("  ✗ No data found to combine")
        return None, None

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description="Combine specific anatomical segmentations into a single file.")
    parser.add_argument("-i", "--input_folder", required=True, help="Input folder containing the segmentation files.")
    parser.add_argument("-o", "--output_file", required=True, help="Output file for the combined segmentation.")
    parser.add_argument("-c", "--config_file", required=True, help="JSON configuration file specifying the anatomical structures and labels.")
    
    args = parser.parse_args()

    input_folder = args.input_folder
    output_file = args.output_file
    config_file = args.config_file

    input_dict = load_config(config_file)
    combined_data, _ = combine_segmentations(input_dict, input_folder, output_file)
    if combined_data is None:
        raise SystemExit(1)
    
    print(f"  ✓ Combined segmentation saved to: {output_file}")

if __name__ == "__main__":
    main()
