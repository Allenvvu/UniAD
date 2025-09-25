#!/usr/bin/env python3
"""
Script to locate all pickle files that contain wrong image paths.
"""

import pickle
import os
import glob
from pathlib import Path

def check_pickle_file_paths(pickle_path):
    """Check if a pickle file contains wrong paths and return details."""
    wrong_paths = []

    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        if 'infos' in data:
            for idx, info in enumerate(data['infos']):
                if 'cams' in info:
                    for cam_key, cam_data in info['cams'].items():
                        if 'data_path' in cam_data:
                            path = cam_data['data_path']
                            # Check if path contains our target patterns or is relative and problematic
                            if (path.startswith('image/') or
                                '/lucid_cameras_x00/' in path or
                                '/lucid_cameras_x01/' in path or
                                path.startswith('/lucid_cameras')):
                                wrong_paths.append({
                                    'info_idx': idx,
                                    'cam': cam_key,
                                    'path': path
                                })
                                # Only show first few examples per file
                                if len(wrong_paths) >= 3:
                                    break
                    if len(wrong_paths) >= 3:
                        break

        return wrong_paths

    except Exception as e:
        print(f"ERROR reading {pickle_path}: {str(e)}")
        return []

def main():
    # Search directories for pickle files
    search_dirs = [
        "/home/bryan/Desktop/Allen/uniad_inf/data/infos",
        "/home/bryan/Desktop/Allen/uniad_inf/data/l2g",
        "/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes",
        "/home/bryan/Desktop/Allen/uniad_inf/canbus",
        "/home/bryan/Desktop/Allen/uniad_inf/data/others"
    ]

    print("Searching for pickle files with wrong image paths...\n")

    total_files_found = 0
    files_with_wrong_paths = []

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        print(f"Checking directory: {search_dir}")
        pickle_pattern = os.path.join(search_dir, "*.pkl")
        pickle_files = glob.glob(pickle_pattern)

        if pickle_files:
            print(f"  Found {len(pickle_files)} .pkl files")
            for pickle_file in pickle_files:
                total_files_found += 1
                wrong_paths = check_pickle_file_paths(pickle_file)
                if wrong_paths:
                    files_with_wrong_paths.append({
                        'file': pickle_file,
                        'wrong_paths': wrong_paths
                    })
                    print(f"    {os.path.basename(pickle_file)}: {len(wrong_paths)} wrong paths found")
                else:
                    print(f"    {os.path.basename(pickle_file)}: OK")
        else:
            print(f"  No .pkl files found")
        print()

    print(f"\nSUMMARY:")
    print(f"Total pickle files checked: {total_files_found}")
    print(f"Files with wrong paths: {len(files_with_wrong_paths)}")

    if files_with_wrong_paths:
        print(f"\nFiles that need fixing:")
        for item in files_with_wrong_paths:
            print(f"\n{item['file']}:")
            for wrong_path in item['wrong_paths'][:3]:  # Show first 3 examples
                print(f"  Info {wrong_path['info_idx']}, {wrong_path['cam']}: {wrong_path['path']}")
            if len(item['wrong_paths']) > 3:
                print(f"  ... and {len(item['wrong_paths']) - 3} more")

if __name__ == "__main__":
    main()