#!/usr/bin/env python3
"""
Create annotation files for ITRI training data.

This script scans the available ITRI data and creates train/val annotation files
that the training script expects.
"""

import os
import pickle
import glob
from pathlib import Path
import numpy as np


def extract_timestamp_from_filename(filename):
    """Extract timestamp from ITRI filename."""
    basename = os.path.basename(filename)
    # Remove extension
    name_only = os.path.splitext(basename)[0]

    # Look for timestamp patterns (Unix timestamp in nanoseconds)
    if name_only.isdigit() and len(name_only) >= 13:
        return float(name_only) / 1e9  # Convert nanoseconds to seconds

    # Look for timestamp in filename parts
    parts = name_only.split('_')
    for part in parts:
        if part.isdigit() and len(part) >= 10:
            if len(part) >= 13:  # nanoseconds
                return float(part) / 1e9
            else:  # seconds
                return float(part)

    # Fallback to file creation time
    try:
        return os.path.getctime(filename)
    except:
        return 0.0


def scan_scene_data(data_root, scene_name):
    """Scan all data files for a specific scene."""
    scene_data = {}

    # Data directories to check
    data_dirs = [
        'canbus',
        'sdc_embeddings',
        'track_query',
        'map_query',
        'gt_fut_traj',
        'sdc_planning'
    ]

    for data_type in data_dirs:
        data_dir = os.path.join(data_root, data_type)

        if data_type == 'canbus':
            # CAN bus files are directly in the directory
            pattern = os.path.join(data_dir, f"{scene_name}_*_can_bus.pkl")
            files = glob.glob(pattern)
        else:
            # Other data types are in scene subdirectories
            scene_dir = os.path.join(data_dir, scene_name)
            if os.path.exists(scene_dir):
                files = []
                for ext in ['*.pt', '*.pkl', '*.npz']:
                    files.extend(glob.glob(os.path.join(scene_dir, ext)))
            else:
                files = []

        scene_data[data_type] = sorted(files, key=extract_timestamp_from_filename)

    return scene_data


def create_data_info(scene_name, timestamp, frame_idx, sample_idx, scene_data):
    """Create a single data info entry."""

    # Basic info structure matching NuScenes format
    data_info = {
        'token': f"{scene_name}_{frame_idx:06d}",
        'timestamp': int(timestamp * 1e6),  # Convert to microseconds
        'scene_token': scene_name,
        'frame_idx': frame_idx,
        'sample_idx': sample_idx,

        # Pose information (dummy values for ITRI)
        'ego2global_translation': [0.0, 0.0, 0.0],
        'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
        'lidar2ego_translation': [0.0, 0.0, 0.0],
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],

        # CAN bus data (dummy, will be loaded by pipeline)
        'can_bus': np.zeros(18, dtype=np.float32),

        # Navigation info
        'prev': None,
        'next': None,

        # File paths (empty for ITRI - no lidar/images)
        'lidar_path': '',
        'sweeps': [],
        'cams': {},

        # Dummy detection data (not used in motion/occ/planning training)
        'gt_boxes': np.zeros((0, 9), dtype=np.float32),
        'gt_names': np.array([], dtype='<U10'),
        'gt_inds': np.array([], dtype=np.int64),
        'gt_velocity': np.zeros((0, 2), dtype=np.float32),
        'num_lidar_pts': np.array([], dtype=int),
        'valid_flag': np.array([], dtype=bool),

        # ITRI-specific data paths
        'itri_data': {}
    }

    # Add available ITRI data file paths
    for data_type, files in scene_data.items():
        if files and frame_idx < len(files):
            data_info['itri_data'][data_type] = files[frame_idx]

    return data_info


def get_scene_list(data_root):
    """Get list of available scenes from ITRI data."""
    scenes = set()

    # Check canbus files for scene names
    canbus_dir = os.path.join(data_root, 'canbus')
    if os.path.exists(canbus_dir):
        for filename in os.listdir(canbus_dir):
            if filename.endswith('_can_bus.pkl'):
                # Extract scene name (format: 2025-08-06-14-23-05_0_can_bus.pkl)
                scene_name = '_'.join(filename.split('_')[:-2])
                scenes.add(scene_name)

    # Also check sdc_embeddings subdirectories
    sdc_dir = os.path.join(data_root, 'sdc_embeddings')
    if os.path.exists(sdc_dir):
        for subdir in os.listdir(sdc_dir):
            if os.path.isdir(os.path.join(sdc_dir, subdir)):
                scenes.add(subdir)

    return sorted(list(scenes))


def create_annotations_for_scene(data_root, scene_name, global_sample_idx):
    """Create annotations for a single scene."""
    scene_data = scan_scene_data(data_root, scene_name)

    # Use the data type with most files as reference for frame count
    max_frames = max(len(files) for files in scene_data.values() if files)
    if max_frames == 0:
        print(f"Warning: No data files found for scene {scene_name}")
        return [], global_sample_idx

    print(f"Scene {scene_name}: {max_frames} frames")

    data_infos = []
    for frame_idx in range(max_frames):
        # Use timestamp from the first available file
        timestamp = 0.0
        for data_type, files in scene_data.items():
            if files and frame_idx < len(files):
                timestamp = extract_timestamp_from_filename(files[frame_idx])
                break

        data_info = create_data_info(
            scene_name, timestamp, frame_idx, global_sample_idx, scene_data
        )

        data_infos.append(data_info)
        global_sample_idx += 1

    # Set prev/next links
    for i in range(len(data_infos)):
        if i > 0:
            data_infos[i]['prev'] = data_infos[i-1]['token']
        if i < len(data_infos) - 1:
            data_infos[i]['next'] = data_infos[i+1]['token']

    return data_infos, global_sample_idx


def create_train_val_split(all_data_infos, train_ratio=0.8):
    """Split data into train and validation sets by scene."""
    # Group by scene
    scenes = {}
    for info in all_data_infos:
        scene_token = info['scene_token']
        if scene_token not in scenes:
            scenes[scene_token] = []
        scenes[scene_token].append(info)

    # Split scenes
    scene_names = sorted(scenes.keys())
    num_train_scenes = int(len(scene_names) * train_ratio)

    train_scenes = scene_names[:num_train_scenes]
    val_scenes = scene_names[num_train_scenes:]

    # Create train/val data
    train_infos = []
    val_infos = []

    for scene_name in train_scenes:
        train_infos.extend(scenes[scene_name])

    for scene_name in val_scenes:
        val_infos.extend(scenes[scene_name])

    print(f"Split: {len(train_scenes)} train scenes ({len(train_infos)} samples), "
          f"{len(val_scenes)} val scenes ({len(val_infos)} samples)")

    return train_infos, val_infos


def save_annotations(data_infos, output_path):
    """Save annotations to pickle file."""
    annotations = {
        'infos': data_infos,
        'metadata': {
            'version': 'itri-v1.0',
            'num_samples': len(data_infos)
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(annotations, f)

    print(f"Saved {len(data_infos)} annotations to {output_path}")


def main():
    data_root = 'data/itri/hct_train'

    if not os.path.exists(data_root):
        print(f"Error: Data root {data_root} does not exist!")
        return

    print(f"Scanning ITRI data in {data_root}...")

    # Get list of scenes
    scenes = get_scene_list(data_root)
    if not scenes:
        print("No scenes found! Please check your data directory structure.")
        return

    print(f"Found {len(scenes)} scenes: {scenes}")

    # Create annotations for all scenes
    all_data_infos = []
    global_sample_idx = 0

    for scene_name in scenes:
        scene_infos, global_sample_idx = create_annotations_for_scene(
            data_root, scene_name, global_sample_idx
        )
        all_data_infos.extend(scene_infos)

    if not all_data_infos:
        print("No data found! Please check your data directory structure.")
        return

    print(f"Total samples: {len(all_data_infos)}")

    # Create train/val split
    train_infos, val_infos = create_train_val_split(all_data_infos)

    # Save annotation files
    train_ann_path = os.path.join(data_root, 'train_annotations.pkl')
    val_ann_path = os.path.join(data_root, 'val_annotations.pkl')

    save_annotations(train_infos, train_ann_path)
    save_annotations(val_infos, val_ann_path)

    print("\nAnnotation files created successfully!")
    print(f"Train: {train_ann_path}")
    print(f"Val: {val_ann_path}")

    # Print sample info
    if train_infos:
        print(f"\nSample train info:")
        sample = train_infos[0]
        print(f"  Token: {sample['token']}")
        print(f"  Scene: {sample['scene_token']}")
        print(f"  Timestamp: {sample['timestamp']}")
        print(f"  Available ITRI data: {list(sample['itri_data'].keys())}")


if __name__ == '__main__':
    main()