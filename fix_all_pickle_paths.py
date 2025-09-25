#!/usr/bin/env python3
"""
Comprehensive script to fix data_path in all pickle files.
Converts directory paths to specific JPG file paths using timestamp data.
"""

import pickle
import os
import glob
import cv2
import numpy as np
from pathlib import Path

def create_placeholder_image(output_path, width=1440, height=928):
    """Create a black placeholder image with specified dimensions."""
    # Create a black image with shape (height, width, 3)
    black_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the placeholder image
    cv2.imwrite(output_path, black_img)
    return output_path

def get_directory_mapping():
    """Define the mapping from camera types to directories."""
    return {
        'CAM_FRONT': '/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x00.gige_100_f_hdr.h265',
        'CAM_BACK': '/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x00.gige_60_b_hdr.h265',
        'CAM_FRONT_LEFT': '/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x01.gige_100_fl_hdr.h265',
        'CAM_FRONT_RIGHT': '/home/bryan/Desktop/Allen/uniad_inf/data/nuscenes/image/lucid_cameras_x01.gige_100_fr_hdr.h265'
    }

def fix_data_paths_in_data(data, directory_mapping):
    """Fix data_path values in the data structure using timestamp information."""
    replacements = 0

    if 'infos' in data:
        for info in data['infos']:
            if 'cams' in info:
                for cam_key, cam_data in info['cams'].items():
                    if 'data_path' in cam_data and 'timestamp' in cam_data:
                        original_path = cam_data['data_path']

                        # Skip if already pointing to specific JPG file (but not /dev/null)
                        if original_path.endswith('.jpg') and original_path != '/dev/null':
                            continue

                        # Handle /dev/null for any camera type by creating placeholder
                        if original_path == '/dev/null':
                            # Use CAM_FRONT directory as default for placeholder creation
                            default_dir = directory_mapping['CAM_FRONT']
                            timestamp = cam_data['timestamp']
                            timestamp_ns = int(timestamp * 1e9)
                            jpg_filename = f"{timestamp_ns}.jpg"
                            placeholder_path = os.path.join(default_dir, f"{cam_key.lower()}_{jpg_filename}")

                            print(f"  INFO: Replacing /dev/null with placeholder for {cam_key}: {os.path.basename(placeholder_path)}")
                            create_placeholder_image(placeholder_path)
                            cam_data['data_path'] = placeholder_path
                            replacements += 1
                            continue

                        # Check if this cam_key matches our mapping
                        if cam_key in directory_mapping:
                            expected_dir = directory_mapping[cam_key]

                            # Check if original path matches directory pattern or is /dev/null
                            if (original_path == expected_dir or
                                original_path == '/dev/null' or
                                original_path.endswith('lucid_cameras_x00.gige_100_f_hdr.h265') or
                                original_path.endswith('lucid_cameras_x00.gige_60_b_hdr.h265') or
                                original_path.endswith('lucid_cameras_x01.gige_100_fl_hdr.h265') or
                                original_path.endswith('lucid_cameras_x01.gige_100_fr_hdr.h265')):

                                # Convert timestamp to nanoseconds filename
                                timestamp = cam_data['timestamp']
                                timestamp_ns = int(timestamp * 1e9)
                                jpg_filename = f"{timestamp_ns}.jpg"

                                # Create full path to JPG file
                                new_path = os.path.join(expected_dir, jpg_filename)

                                # Verify the JPG file exists
                                if os.path.exists(new_path):
                                    cam_data['data_path'] = new_path
                                    replacements += 1
                                else:
                                    # Handle missing images by creating a black placeholder
                                    print(f"  INFO: JPG file not found, creating black placeholder: {jpg_filename}")
                                    placeholder_path = create_placeholder_image(new_path)
                                    cam_data['data_path'] = placeholder_path
                                    replacements += 1

    return replacements

def fix_pickle_file(pickle_path, directory_mapping):
    """Fix data_path values in a pickle file."""
    print(f"Processing {pickle_path}...")

    try:
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Fix paths
        replacements = fix_data_paths_in_data(data, directory_mapping)

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
    # Search directories for pickle files
    search_dirs = [
        "/home/bryan/Desktop/Allen/uniad_inf/data/infos",
        "/home/bryan/Desktop/Allen/uniad_inf/data/l2g",
        "/home/bryan/Desktop/Allen/uniad_inf/data/others"
    ]

    # Get directory mapping
    directory_mapping = get_directory_mapping()

    print("Directory mapping:")
    for cam_type, directory in directory_mapping.items():
        print(f"  {cam_type} -> {directory}")
    print()

    print("Searching for pickle files to fix...\\n")

    total_files_processed = 0
    total_replacements = 0
    files_with_fixes = []

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        print(f"Checking directory: {search_dir}")
        pickle_pattern = os.path.join(search_dir, "*.pkl")
        pickle_files = glob.glob(pickle_pattern)

        if pickle_files:
            print(f"  Found {len(pickle_files)} .pkl files")
            for pickle_file in pickle_files:
                total_files_processed += 1
                replacements = fix_pickle_file(pickle_file, directory_mapping)
                total_replacements += replacements
                if replacements > 0:
                    files_with_fixes.append({
                        'file': pickle_file,
                        'fixes': replacements
                    })
        else:
            print(f"  No .pkl files found")
        print()

    print(f"\\nSUMMARY:")
    print(f"  Total pickle files processed: {total_files_processed}")
    print(f"  Total path fixes applied: {total_replacements}")
    print(f"  Files with fixes: {len(files_with_fixes)}")

    if files_with_fixes:
        print(f"\\nFiles that were updated:")
        for item in files_with_fixes:
            print(f"  {os.path.basename(item['file'])}: {item['fixes']} fixes")

    print("\\nDone!")

if __name__ == "__main__":
    main()