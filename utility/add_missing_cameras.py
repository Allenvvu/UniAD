#!/usr/bin/env python3

import pickle
import copy
import numpy as np

def add_missing_cameras():
    """Add missing CAM_BACK_LEFT and CAM_BACK_RIGHT to L2G dataset with zero masks"""

    # Load current dataset
    dataset_file = "data/infos/nuscenes_infos_temporal_val.pkl"
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Processing {len(data['infos'])} samples...")

    # Process each sample
    for i, sample in enumerate(data['infos']):
        # Check current cameras
        current_cameras = list(sample['cams'].keys())

        # Add missing cameras with zero/masked data
        if 'CAM_BACK_LEFT' not in current_cameras:
            # Create dummy camera with zero/invalid data
            sample['cams']['CAM_BACK_LEFT'] = {
                'data_path': '/dev/null',  # Invalid path
                'type': 'camera',
                'sensor2ego_translation': [0.0, 0.0, 0.0],
                'sensor2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
                'ego2global_translation': [0.0, 0.0, 0.0],
                'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
                'cam_intrinsic': np.zeros((3, 3)),  # Zero intrinsics
                'timestamp': sample['timestamp']
            }

        if 'CAM_BACK_RIGHT' not in current_cameras:
            # Create dummy camera with zero/invalid data
            sample['cams']['CAM_BACK_RIGHT'] = {
                'data_path': '/dev/null',  # Invalid path
                'type': 'camera',
                'sensor2ego_translation': [0.0, 0.0, 0.0],
                'sensor2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
                'ego2global_translation': [0.0, 0.0, 0.0],
                'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
                'cam_intrinsic': np.zeros((3, 3)),  # Zero intrinsics
                'timestamp': sample['timestamp']
            }

        if i == 0:
            print(f"Sample 0 cameras after adding: {list(sample['cams'].keys())}")

    # Save updated dataset
    with open(dataset_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Added masked cameras to all {len(data['infos'])} samples")
    print("Updated dataset saved!")

if __name__ == "__main__":
    add_missing_cameras()