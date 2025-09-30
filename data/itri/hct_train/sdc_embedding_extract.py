#!/usr/bin/env python3

import numpy as np
import torch
import os
import glob
import argparse
import pickle
from timestamp_extract import load_master_timestamps, find_closest_timestamp
from mmdet3d.core.bbox import LiDARInstance3DBoxes

def load_canbus_data(canbus_file):
    """Load CAN bus data from pickle file"""
    try:
        with open(canbus_file, 'rb') as f:
            data = pickle.load(f)
        return data['can_bus'], data['timestamps']
    except Exception as e:
        print(f"Error loading CAN bus data from {canbus_file}: {e}")
        return None, None

def find_canbus_file_for_bag(bag_name, canbus_dir):
    """Find corresponding CAN bus file for a bag file"""
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    canbus_file = os.path.join(canbus_dir, f"{base_name}_can_bus.pkl")

    if os.path.exists(canbus_file):
        return canbus_file
    else:
        print(f"Warning: CAN bus file not found: {canbus_file}")
        return None

def create_sdc_embedding_from_canbus(can_bus_sample, embed_dim=256):
    """
    Create SDC embedding from CAN bus data

    Args:
        can_bus_sample: 18D CAN bus vector [x,y,z,qw,qx,qy,qz,ax,ay,az,wx,wy,wz,vx,vy,vz,yaw,reserved]
        embed_dim: Target embedding dimension (default 256)

    Returns:
        torch.Tensor: 256D embedding vector
    """

    # Extract key features from CAN bus
    # Position: [0:3]
    position = can_bus_sample[0:3]

    # Quaternion: [3:7]
    quaternion = can_bus_sample[3:7]

    # Linear acceleration: [7:10]
    linear_accel = can_bus_sample[7:10]

    # Angular velocity: [10:13]
    angular_vel = can_bus_sample[10:13]

    # Velocity: [13:16]
    velocity = can_bus_sample[13:16]

    # Yaw: [16]
    yaw = can_bus_sample[16:17]

    # Compute derived features
    speed = np.sqrt(np.sum(velocity[:2]**2))  # 2D speed
    accel_magnitude = np.sqrt(np.sum(linear_accel**2))
    angular_speed = np.sqrt(np.sum(angular_vel**2))

    # Create rich feature vector
    features = np.concatenate([
        position,           # 3D - global position
        quaternion,         # 4D - orientation quaternion
        velocity,           # 3D - 3D velocity
        linear_accel,       # 3D - linear acceleration
        angular_vel,        # 3D - angular velocity
        yaw,                # 1D - yaw angle
        [speed],            # 1D - 2D speed magnitude
        [accel_magnitude],  # 1D - acceleration magnitude
        [angular_speed],    # 1D - angular speed magnitude
    ])  # Total: 21 features

    # Add trigonometric features for better temporal modeling
    trig_features = np.array([
        np.sin(yaw[0]), np.cos(yaw[0]),     # 2D - yaw trigonometric
        np.sin(velocity[0]/10), np.cos(velocity[0]/10),  # 2D - velocity phase encoding
        np.sin(velocity[1]/10), np.cos(velocity[1]/10),  # 2D - velocity phase encoding
    ])  # 6 more features, total: 27

    features = np.concatenate([features, trig_features])

    # Normalize features to prevent overflow
    features = features / (np.linalg.norm(features) + 1e-8)

    # Simple approach - just pad with zeros
    if len(features) < embed_dim:
        padding = np.zeros(embed_dim - len(features))
        final_embedding = np.concatenate([features, padding])
    else:
        final_embedding = features[:embed_dim]

    return torch.tensor(final_embedding, dtype=torch.float32)

def create_sdc_track_bbox_results(can_bus_sample):
    """
    Create SDC track bbox results from CAN bus data

    Args:
        can_bus_sample: 18D CAN bus vector

    Returns:
        List: Track bbox results in UniAD format [[bboxes, scores, labels, bbox_index, mask]]
    """

    # Extract 3D position
    x, y, z = can_bus_sample[0:3]

    # Extract velocity
    vel_x, vel_y = can_bus_sample[13:15]

    # Extract yaw angle
    yaw = can_bus_sample[16]

    # Vehicle dimensions (typical sedan)
    # Length x Width x Height (meters)
    length = 4.5   # front to back
    width = 1.8    # side to side
    height = 1.5   # bottom to top

    # Create 9D bounding box: [x, y, z, width, length, height, yaw, vel_x, vel_y]
    # Note: UniAD uses (w, l, h) format where w=width, l=length
    bbox_9d = np.array([x, y, z, width, length, height, yaw, vel_x, vel_y])

    # Convert to torch tensor and add batch dimension
    bbox_tensor = torch.tensor(bbox_9d, dtype=torch.float32).unsqueeze(0)  # (1, 9)

    # Create LiDARInstance3DBoxes
    bboxes = LiDARInstance3DBoxes(bbox_tensor, box_dim=9)

    # Perfect confidence for ground truth
    scores = torch.tensor([1.0], dtype=torch.float32)

    # Car class label (0 for car in most autonomous driving datasets)
    labels = torch.tensor([0], dtype=torch.long)

    # Bbox index (single SDC has index 0)
    bbox_index = torch.tensor([0], dtype=torch.long)

    # Validity mask (always valid for SDC)
    mask = torch.tensor([True], dtype=torch.bool)

    # Format as expected by UniAD: [[bboxes, scores, labels, bbox_index, mask]]
    track_bbox_results = [[bboxes, scores, labels, bbox_index, mask]]

    return track_bbox_results

def extract_sdc_data_aligned(canbus_dir, global_timestamps, bag_name, embed_dim=256):
    """
    Extract SDC embedding and track bbox results aligned to global timestamps

    Args:
        canbus_dir: Directory containing CAN bus files
        global_timestamps: Global timestamp array
        bag_name: Name of the bag file
        embed_dim: Embedding dimension

    Returns:
        Tuple: (sdc_embeddings, sdc_track_bbox_results, valid_timestamps)
    """

    # Find corresponding CAN bus file
    canbus_file = find_canbus_file_for_bag(bag_name, canbus_dir)
    if canbus_file is None:
        return None, None, None

    # Load CAN bus data
    can_bus_data, canbus_timestamps = load_canbus_data(canbus_file)
    if can_bus_data is None:
        return None, None, None

    print(f"Loaded {len(can_bus_data)} CAN bus samples for {bag_name}")

    # Extract SDC data aligned to global timestamps
    sdc_embeddings = []
    sdc_track_bbox_results = []
    valid_timestamps = []

    tolerance = 0.5  # 500ms tolerance for timestamp matching

    for target_time in global_timestamps:
        # Find closest CAN bus sample
        closest_time, closest_idx = find_closest_timestamp(
            float(target_time), canbus_timestamps, tolerance=tolerance)

        if closest_time is not None:
            # Extract CAN bus sample
            can_bus_sample = can_bus_data[closest_idx]

            # Create SDC embedding
            sdc_embedding = create_sdc_embedding_from_canbus(can_bus_sample, embed_dim)

            # Create SDC track bbox results
            track_bbox_results = create_sdc_track_bbox_results(can_bus_sample)

            # Store results
            sdc_embeddings.append(sdc_embedding)
            sdc_track_bbox_results.append(track_bbox_results)
            valid_timestamps.append(target_time)

    if len(sdc_embeddings) == 0:
        print(f"Warning: No valid SDC data found for {bag_name}")
        return None, None, None

    print(f"Extracted {len(sdc_embeddings)} SDC embedding/bbox samples for {bag_name}")

    return sdc_embeddings, sdc_track_bbox_results, np.array(valid_timestamps)

def save_sdc_data(sdc_embeddings, sdc_track_bbox_results, valid_timestamps, output_dir, bag_name):
    """
    Save SDC embedding and track bbox results in frame-aligned format

    Args:
        sdc_embeddings: List of embedding tensors
        sdc_track_bbox_results: List of track bbox results
        valid_timestamps: Array of valid timestamps
        output_dir: Output directory
        bag_name: Bag file name for naming
    """

    os.makedirs(output_dir, exist_ok=True)

    # Save aggregated data
    output_file = os.path.join(output_dir, f"{bag_name}_sdc_embeddings.pt")

    sdc_data = {
        'sdc_embeddings': sdc_embeddings,
        'sdc_track_bbox_results': sdc_track_bbox_results,
        'timestamps': valid_timestamps,
        'num_frames': len(valid_timestamps),
        'embed_dim': sdc_embeddings[0].shape[0] if len(sdc_embeddings) > 0 else 0
    }

    torch.save(sdc_data, output_file)
    print(f"Saved SDC embeddings and bbox results to {output_file}")

    # Save frame-by-frame data for training alignment
    frame_dir = os.path.join(output_dir, f"{bag_name}")
    os.makedirs(frame_dir, exist_ok=True)

    for i, timestamp in enumerate(valid_timestamps):
        frame_data = {
            'sdc_embedding': sdc_embeddings[i],
            'sdc_track_bbox_results': sdc_track_bbox_results[i],
            'timestamp': timestamp
        }

        # Use timestamp (in microseconds) as filename
        ts_us = int(round(float(timestamp) * 1_000_000))
        frame_file = os.path.join(frame_dir, f"{ts_us}.pt")
        torch.save(frame_data, frame_file)

    print(f"Saved {len(valid_timestamps)} frame-level SDC data files to {frame_dir}")

def validate_sdc_data(sdc_embeddings, sdc_track_bbox_results):
    """Validate extracted SDC data"""

    print(f"Number of SDC samples: {len(sdc_embeddings)}")

    if len(sdc_embeddings) > 0:
        # Check embedding dimensions
        embed_dim = sdc_embeddings[0].shape[0]
        print(f"SDC Embedding dimension: {embed_dim}")

        # Check embedding statistics
        embeddings_tensor = torch.stack(sdc_embeddings)
        print(f"Embedding range: [{float(torch.min(embeddings_tensor)):.3f}, {float(torch.max(embeddings_tensor)):.3f}]")
        print(f"Embedding mean norm: {float(torch.mean(torch.norm(embeddings_tensor, dim=1))):.3f}")

        # Check track bbox results structure
        sample_bbox_results = sdc_track_bbox_results[0]
        bboxes, scores, labels, bbox_index, mask = sample_bbox_results[0]

        print(f"Bbox shape: {bboxes.tensor.shape}")
        print(f"Bbox center: [{float(bboxes.tensor[0, 0]):.2f}, {float(bboxes.tensor[0, 1]):.2f}, {float(bboxes.tensor[0, 2]):.2f}]")
        print(f"Bbox dimensions: [{float(bboxes.tensor[0, 3]):.2f}, {float(bboxes.tensor[0, 4]):.2f}, {float(bboxes.tensor[0, 5]):.2f}]")
        print(f"Bbox velocity: [{float(bboxes.tensor[0, 7]):.2f}, {float(bboxes.tensor[0, 8]):.2f}]")
        print(f"Score: {float(scores[0]):.2f}, Label: {int(labels[0])}")

def process_bag_file(bag_file, output_dir, canbus_dir, global_timestamps, embed_dim=256, test_mode=False):
    """Process a single bag file and extract SDC data"""

    print(f"\nProcessing {bag_file}...")

    # Extract bag name
    bag_name = os.path.basename(bag_file).replace('.bag', '')

    # Extract SDC data aligned to global timestamps
    sdc_embeddings, sdc_track_bbox_results, valid_timestamps = extract_sdc_data_aligned(
        canbus_dir, global_timestamps, bag_file, embed_dim)

    if sdc_embeddings is None:
        print(f"Skipping {bag_file} - no valid SDC data found")
        return 0

    # Limit processing in test mode
    if test_mode and len(sdc_embeddings) > 3:
        print(f"Test mode: limiting to first 3 frames")
        sdc_embeddings = sdc_embeddings[:3]
        sdc_track_bbox_results = sdc_track_bbox_results[:3]
        valid_timestamps = valid_timestamps[:3]

    # Validate data
    validate_sdc_data(sdc_embeddings, sdc_track_bbox_results)

    # Save SDC data
    save_sdc_data(sdc_embeddings, sdc_track_bbox_results, valid_timestamps,
                  output_dir, bag_name)

    return len(valid_timestamps)

def main():
    parser = argparse.ArgumentParser(description='Extract SDC embeddings and track bbox results from CAN bus data')
    parser.add_argument('--bag_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files/sort',
                       help='Directory containing bag files (used for naming)')
    parser.add_argument('--canbus_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus/sort',
                       help='Directory containing CAN bus pickle files')
    parser.add_argument('--timestamps_file', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl',
                       help='Global timestamps pickle file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/sdc_embeddings',
                       help='Output directory for SDC embeddings and bbox results')
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='SDC embedding dimension (default: 256)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process only first 3 frames for quick verification')

    args = parser.parse_args()

    # Load global timestamps
    print(f"Loading global timestamps from {args.timestamps_file}...")
    global_timestamps, hz = load_master_timestamps(args.timestamps_file)
    if global_timestamps is None:
        print(f"Error: Could not load timestamps from {args.timestamps_file}")
        return

    print(f"Loaded {len(global_timestamps)} global timestamps at {hz} Hz")

    # Find all bag files
    bag_pattern = os.path.join(args.bag_dir, '*.bag')
    bag_files = glob.glob(bag_pattern)

    if not bag_files:
        print(f"No bag files found in {args.bag_dir}")
        return

    print(f"Found {len(bag_files)} bag files to process")
    print(f"Extracting {args.embed_dim}D SDC embeddings and track bbox results")

    if args.test:
        print("*** TEST MODE ENABLED: Processing only first 3 frames per bag ***")

    total_frames = 0
    processed_files = 0

    for bag_file in sorted(bag_files):
        frames_processed = process_bag_file(
            bag_file, args.output_dir, args.canbus_dir,
            global_timestamps, args.embed_dim, test_mode=args.test
        )
        if frames_processed > 0:
            total_frames += frames_processed
            processed_files += 1

    print(f"\n=== SDC Embedding and Bbox Extraction Complete ===")
    print(f"Processed {processed_files}/{len(bag_files)} files successfully")
    print(f"Generated {total_frames} SDC embedding/bbox samples")
    print(f"Embedding dimension: {args.embed_dim}")
    print(f"Used global timestamps with {hz} Hz sampling")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()