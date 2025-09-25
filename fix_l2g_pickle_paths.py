#!/usr/bin/env python3
"""
Script to fix data_path in all .pkl files in data/l2g directory.
Converts hardcoded paths to correct image directory paths.
"""

import pickle
import os
import glob
from pathlib import Path

def get_path_mapping():
    """Define the mapping from old paths to new paths."""
    return {
        "image/lucid_cameras_x00.gige_100_f_hdr.h265": "/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x00.gige_100_f_hdr.h265",
        "image/lucid_cameras_x00.gige_60_b_hdr.h265": "/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x00.gige_60_b_hdr.h265",
        "image/lucid_cameras_x01.gige_100_fl_hdr.h265": "/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x01.gige_100_fl_hdr.h265",
        "image/lucid_cameras_x01.gige_100_fr_hdr.h265": "/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x01.gige_100_fr_hdr.h265"
    }

def fix_data_paths_in_data(data, path_mapping):
    """Fix data_path values in the data structure."""
    replacements = 0

    if 'infos' in data:
        for info in data['infos']:
            if 'cams' in info:
                for cam_key, cam_data in info['cams'].items():
                    if 'data_path' in cam_data:
                        original_path = cam_data['data_path']

                        # Check for exact matches first
                        if original_path in path_mapping:
                            cam_data['data_path'] = path_mapping[original_path]
                            replacements += 1
                        else:
                            # Check for partial matches and replace
                            for old_path, new_path in path_mapping.items():
                                if old_path in original_path:
                                    cam_data['data_path'] = original_path.replace(old_path, new_path)
                                    replacements += 1
                                    break

    return replacements

def fix_pickle_file(pickle_path, path_mapping):
    """Fix data_path values in a pickle file."""
    print(f"Processing {pickle_path}...")

    try:
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Fix paths
        replacements = fix_data_paths_in_data(data, path_mapping)

        if replacements > 0:
            # Create backup
            backup_path = pickle_path + '.backup'
            if not os.path.exists(backup_path):
                os.rename(pickle_path, backup_path)
                print(f"  Created backup: {backup_path}")

            # Save fixed pickle file
            with open(pickle_path, 'wb') as f:
                pickle.dump(data, f)

            print(f"  Fixed {replacements} paths")
        else:
            print(f"  No paths to fix")

        return replacements

    except Exception as e:
        print(f"  ERROR processing {pickle_path}: {str(e)}")
        return 0

def main():
    # Define the directory containing pickle files
    l2g_dir = "/home/bryan/Desktop/Allen/uniad_inf/data/l2g"

    # Get path mapping
    path_mapping = get_path_mapping()

    print("Path mapping:")
    for old_path, new_path in path_mapping.items():
        print(f"  {old_path} -> {new_path}")
    print()

    # Find all .pkl files in data/l2g directory
    pickle_pattern = os.path.join(l2g_dir, "*.pkl")
    pickle_files = glob.glob(pickle_pattern)

    if not pickle_files:
        print(f"No .pkl files found in {l2g_dir}")
        return

    print(f"Found {len(pickle_files)} .pkl files to process:")
    for pf in pickle_files:
        print(f"  {os.path.basename(pf)}")
    print()

    total_replacements = 0

    # Fix each pickle file
    for pickle_file in pickle_files:
        replacements = fix_pickle_file(pickle_file, path_mapping)
        total_replacements += replacements

    print(f"\nSummary:")
    print(f"  Processed files: {len(pickle_files)}")
    print(f"  Total replacements: {total_replacements}")
    print("Done!")

if __name__ == "__main__":
    main()