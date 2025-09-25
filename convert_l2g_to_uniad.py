#!/usr/bin/env python3
"""
Convert L2G camera data to UniAD-compatible nuScenes format.

This script converts camera data with L2G format to the specific format
required by UniAD for inference, including proper coordinate transformations,
temporal sequencing, and metadata structure.
"""

import os
import pickle
import numpy as np
from pyquaternion import Quaternion
from collections import OrderedDict
import mmcv
from typing import Dict, List, Tuple, Optional
import argparse
import cv2


def load_canbus_data(canbus_file: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load CAN bus data from pickle file"""
    try:
        with open(canbus_file, 'rb') as f:
            data = pickle.load(f)
        return data['can_bus'], data['timestamps']
    except Exception as e:
        print(f"Error loading CAN bus data from {canbus_file}: {e}")
        return None, None


def find_closest_timestamp(target_time: float, timestamps: np.ndarray, tolerance: float = 0.5) -> Tuple[Optional[float], Optional[int]]:
    """Find closest timestamp within tolerance"""
    if len(timestamps) == 0:
        return None, None

    # Convert to numpy array if not already
    timestamps = np.array(timestamps)

    # Find closest timestamp
    time_diffs = np.abs(timestamps - target_time)
    closest_idx = np.argmin(time_diffs)
    closest_time = timestamps[closest_idx]

    if time_diffs[closest_idx] <= tolerance:
        return closest_time, closest_idx
    else:
        return None, None


def extract_ego_pose_from_canbus(canbus_data: np.ndarray, timestamp_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ego pose from canbus data.

    Args:
        canbus_data: Array of shape (N, 18) with canbus information
        timestamp_idx: Index of the timestamp to extract

    Returns:
        ego2global_translation: [x, y, z] translation
        ego2global_rotation: [w, x, y, z] quaternion
    """
    # CAN bus format from l2g_data.md:
    # 0-2: position_xyz (x, y, z coordinates)
    # 3-6: quaternion_wxyz (orientation as quaternion w,x,y,z)

    row = canbus_data[timestamp_idx]
    ego2global_translation = row[0:3]  # x, y, z
    ego2global_rotation = row[3:7]     # w, x, y, z quaternion

    return ego2global_translation, ego2global_rotation


def convert_l2g_cameras_to_nuscenes_format(camera_configs: Dict, target_timestamp: float,
                                         canbus_dir: str, bag_name: str) -> Dict:
    """
    Convert L2G camera configuration to nuScenes format.

    Args:
        camera_configs: Camera configuration from l2g_data.md format
        target_timestamp: Target timestamp for this frame
        canbus_dir: Directory containing canbus files
        bag_name: Name of the bag file to find corresponding canbus data

    Returns:
        Dictionary with nuScenes-compatible camera information
    """

    # Map L2G camera names to nuScenes equivalents
    camera_name_mapping = {
        '100f': 'CAM_FRONT',           # Front camera
        '60b': 'CAM_BACK',            # Back camera
        '100fl': 'CAM_FRONT_LEFT',    # Front left
        '100fr': 'CAM_FRONT_RIGHT',   # Front right
        # Add more mappings as needed for back left/right cameras
    }

    # Find and load canbus data
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    canbus_file = os.path.join(canbus_dir, f"{base_name}_can_bus.pkl")

    if not os.path.exists(canbus_file):
        raise FileNotFoundError(f"CAN bus file not found: {canbus_file}")

    canbus_data, canbus_timestamps = load_canbus_data(canbus_file)
    if canbus_data is None:
        raise ValueError(f"Failed to load canbus data from {canbus_file}")

    # Find closest canbus timestamp
    closest_time, closest_idx = find_closest_timestamp(target_timestamp, canbus_timestamps, tolerance=0.5)
    if closest_time is None:
        raise ValueError(f"No canbus data found near timestamp {target_timestamp}")

    # Extract ego pose
    ego2global_translation, ego2global_rotation = extract_ego_pose_from_canbus(canbus_data, closest_idx)

    nuscenes_cameras = {}

    for l2g_cam_name, l2g_cam_info in camera_configs.items():
        if l2g_cam_name not in camera_name_mapping:
            continue

        nuscenes_cam_name = camera_name_mapping[l2g_cam_name]

        # Extract camera parameters
        sensor2ego_translation = l2g_cam_info['sensor2ego_translation']
        sensor2ego_rotation = l2g_cam_info['sensor2ego_rotation']  # [w,x,y,z] quaternion
        cam_intrinsic = np.array(l2g_cam_info['cam_intrinsic']).reshape(3, 3)

        nuscenes_cameras[nuscenes_cam_name] = {
            'data_path': l2g_cam_info['data_path'],
            'type': 'camera',
            'sensor2ego_translation': sensor2ego_translation,
            'sensor2ego_rotation': sensor2ego_rotation,
            'ego2global_translation': ego2global_translation.tolist(),
            'ego2global_rotation': ego2global_rotation.tolist(),
            'cam_intrinsic': cam_intrinsic,
            'timestamp': target_timestamp
        }

    return nuscenes_cameras, ego2global_translation, ego2global_rotation


def compute_lidar_to_global_transform(ego2global_translation: np.ndarray,
                                    ego2global_rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute lidar to global transformation matrices.

    For simplicity, we assume LiDAR is at ego vehicle center (identity transform from lidar to ego).
    In practice, you would need actual LiDAR calibration parameters.

    Args:
        ego2global_translation: Translation from ego to global
        ego2global_rotation: Rotation quaternion from ego to global

    Returns:
        l2g_r_mat: 3x3 rotation matrix from lidar to global
        l2g_t: 3D translation vector from lidar to global
    """

    # Assume LiDAR is at ego center (identity transform)
    # In practice, you need actual lidar2ego calibration
    lidar2ego_translation = np.array([0.0, 0.0, 0.0])  # Adjust based on your setup
    lidar2ego_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion [w,x,y,z]

    # Convert quaternions to rotation matrices
    l2e_r_mat = Quaternion(lidar2ego_rotation).rotation_matrix
    e2g_r_mat = Quaternion(ego2global_rotation).rotation_matrix

    # Compute lidar to global transformation
    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = lidar2ego_translation @ e2g_r_mat.T + ego2global_translation

    return l2g_r_mat.astype(np.float32), l2g_t.astype(np.float32)


def create_uniad_sample_info(cameras: Dict, timestamp: float, sample_token: str,
                           scene_token: str, frame_idx: int, prev_token: str = '',
                           next_token: str = '') -> Dict:
    """
    Create UniAD-compatible sample information structure.

    Args:
        cameras: Camera information in nuScenes format
        timestamp: Sample timestamp
        sample_token: Unique token for this sample
        scene_token: Scene token for temporal grouping
        frame_idx: Frame index within scene
        prev_token: Previous sample token (for temporal linking)
        next_token: Next sample token (for temporal linking)

    Returns:
        Dictionary with UniAD-compatible sample information
    """

    # Get ego pose from first camera (they should all have same ego pose)
    first_cam = list(cameras.values())[0]
    ego2global_translation = first_cam['ego2global_translation']
    ego2global_rotation = first_cam['ego2global_rotation']

    # Compute lidar transformations
    l2g_r_mat, l2g_t = compute_lidar_to_global_transform(
        np.array(ego2global_translation),
        np.array(ego2global_rotation)
    )

    # Create sample info structure
    sample_info = {
        'token': sample_token,
        'timestamp': timestamp,
        'prev': prev_token,
        'next': next_token,
        'scene_token': scene_token,
        'frame_idx': frame_idx,

        # Pose information
        'lidar2ego_translation': [0.0, 0.0, 0.0],  # Assuming LiDAR at ego center
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],  # Identity quaternion
        'ego2global_translation': ego2global_translation,
        'ego2global_rotation': ego2global_rotation,

        # Camera information
        'cams': cameras,

        # Placeholder for other required fields
        'sweeps': [],  # LiDAR sweeps (empty for camera-only)
        'can_bus': np.zeros(18),  # CAN bus info (can be filled from canbus data)

        # For inference, these can be empty/placeholder
        'gt_boxes': [],
        'gt_names': [],
        'gt_velocity': [],
        'num_lidar_pts': 0,
        'valid_flag': True
    }

    return sample_info, l2g_r_mat, l2g_t


def create_temporal_sequence(sample_infos: List[Dict], queue_length: int = 3) -> List[Dict]:
    """
    Create temporal sequences for UniAD processing.

    Args:
        sample_infos: List of sample information dictionaries
        queue_length: Number of frames in each sequence

    Returns:
        List of temporal sequences ready for UniAD inference
    """
    sequences = []

    for i in range(len(sample_infos) - queue_length + 1):
        sequence = sample_infos[i:i + queue_length]
        sequences.append(sequence)

    return sequences


def main():
    parser = argparse.ArgumentParser(description='Convert L2G data to UniAD format')
    parser.add_argument('--camera_config', required=True, help='Python file with camera configuration')
    parser.add_argument('--canbus_dir', required=True, help='Directory containing canbus pickle files')
    parser.add_argument('--output_file', required=True, help='Output pickle file for UniAD')
    parser.add_argument('--bag_name', required=True, help='Name of the bag file')
    parser.add_argument('--timestamps', required=True, nargs='+', type=float,
                       help='List of timestamps to process')
    parser.add_argument('--scene_token', default='scene_001', help='Scene identifier')

    args = parser.parse_args()

    # Load camera configuration (you need to provide this)
    # Example format from l2g_data.md
    exec(open(args.camera_config).read(), globals())

    # Process each timestamp
    sample_infos = []
    all_l2g_transforms = []

    for i, timestamp in enumerate(args.timestamps):
        print(f"Processing timestamp {timestamp} ({i+1}/{len(args.timestamps)})")

        # Generate unique sample token
        sample_token = f"sample_{i:06d}"
        prev_token = f"sample_{i-1:06d}" if i > 0 else ''
        next_token = f"sample_{i+1:06d}" if i < len(args.timestamps)-1 else ''

        try:
            # Convert cameras to nuScenes format
            cameras, ego_translation, ego_rotation = convert_l2g_cameras_to_nuscenes_format(
                cams, timestamp, args.canbus_dir, args.bag_name
            )

            # Create sample info
            sample_info, l2g_r_mat, l2g_t = create_uniad_sample_info(
                cameras, timestamp, sample_token, args.scene_token, i, prev_token, next_token
            )

            sample_infos.append(sample_info)
            all_l2g_transforms.append({'l2g_r_mat': l2g_r_mat, 'l2g_t': l2g_t})

        except Exception as e:
            print(f"Error processing timestamp {timestamp}: {e}")
            continue

    # Create temporal sequences
    sequences = create_temporal_sequence(sample_infos, queue_length=3)

    # Prepare output data structure
    output_data = {
        'infos': sample_infos,
        'temporal_sequences': sequences,
        'l2g_transforms': all_l2g_transforms,
        'metadata': {
            'version': 'l2g_converted',
            'num_samples': len(sample_infos),
            'queue_length': 3,
            'scene_token': args.scene_token
        }
    }

    # Save to pickle file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"Converted {len(sample_infos)} samples to UniAD format")
    print(f"Created {len(sequences)} temporal sequences")
    print(f"Output saved to: {args.output_file}")


if __name__ == "__main__":
    main()