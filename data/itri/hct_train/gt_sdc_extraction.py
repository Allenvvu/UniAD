#!/usr/bin/env python3

import numpy as np
import torch
import os
import glob
import argparse
import pickle
from collections import defaultdict
from timestamp_extract import load_master_timestamps, find_closest_timestamp

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
    # Remove .bag extension and add _can_bus.pkl
    base_name = os.path.splitext(os.path.basename(bag_name))[0]
    canbus_file = os.path.join(canbus_dir, f"{base_name}_can_bus.pkl")

    if os.path.exists(canbus_file):
        return canbus_file
    else:
        print(f"Warning: CAN bus file not found: {canbus_file}")
        return None

def extract_car_state_from_canbus(canbus_dir, global_timestamps, bag_name):
    """Extract car state data from preprocessed CAN bus files using global timestamps"""

    # Find corresponding CAN bus file
    canbus_file = find_canbus_file_for_bag(bag_name, canbus_dir)
    if canbus_file is None:
        return None, None

    print(f"Loading CAN bus data from {canbus_file}...")

    # Load CAN bus data
    can_bus_data, canbus_timestamps = load_canbus_data(canbus_file)
    if can_bus_data is None:
        return None, None

    print(f"Loaded {len(can_bus_data)} CAN bus samples")

    # Extract position and velocity data aligned to global timestamps
    positions = []
    velocities = []
    valid_timestamps = []

    # CAN bus format: [0-2]: position (x,y,z), [13-15]: velocity (vel_x, vel_y, vel_z)
    for target_time in global_timestamps:
        # Find closest CAN bus timestamp within tolerance
        closest_time, closest_idx = find_closest_timestamp(float(target_time), canbus_timestamps, tolerance=0.5)

        if closest_time is not None:
            # Extract position [x, y] (ignoring z for 2D trajectory)
            pos_x = can_bus_data[closest_idx, 0]  # x position
            pos_y = can_bus_data[closest_idx, 1]  # y position
            positions.append([pos_x, pos_y])

            # Extract velocity [vel_x, vel_y]
            vel_x = can_bus_data[closest_idx, 13]  # velocity x
            vel_y = can_bus_data[closest_idx, 14]  # velocity y
            velocities.append([vel_x, vel_y])

            valid_timestamps.append(target_time)

    if len(positions) == 0:
        print("Warning: No valid position data found")
        return None, None

    print(f"Extracted {len(positions)} synchronized position/velocity samples")
    return np.array(positions), np.array(valid_timestamps)


def extract_sdc_future_trajectories(positions, timestamps, predict_steps=12, sample_rate=2.0, test_mode=False):
    """
    Extract SDC future trajectories from position data
    
    Args:
        positions: numpy array of shape (N, 2) with [x, y] positions
        timestamps: numpy array of shape (N,) with timestamps  
        predict_steps: number of future steps to predict (default 12)
        sample_rate: sampling rate in Hz (default 2.0 for 2Hz, 6 seconds total)
        test_mode: if True, only process first 3 frames for testing
        
    Returns:
        gt_sdc_fut_traj: list[torch.Tensor] where each tensor has shape (1, predict_steps, 2)
        gt_sdc_fut_traj_mask: list[torch.Tensor] where each tensor has shape (1, predict_steps, 2)
        valid_timestamps: numpy array of valid current timestamps
    """
    
    num_samples = len(positions)
    time_step = 1.0 / sample_rate  # 0.5 seconds at 2Hz
    
    gt_sdc_fut_traj = []
    gt_sdc_fut_traj_mask = []
    valid_timestamps = []
    
    # Determine processing range based on test mode
    max_samples = min(3, num_samples - predict_steps) if test_mode else num_samples - predict_steps
    
    if test_mode:
        print(f"Test mode: processing only first {max_samples} frames")
    
    # For each current timestep, extract future predict_steps positions
    for i in range(max_samples):
        current_time = timestamps[i]
        future_trajs = []
        future_masks = []
        
        # Extract future positions for next predict_steps
        for step in range(1, predict_steps + 1):
            # Calculate target time for this step
            target_time = current_time + step * time_step
            
            # Find the closest timestamp to the target time
            time_diffs = np.abs(timestamps - target_time)
            closest_idx = np.argmin(time_diffs)
            closest_time_diff = time_diffs[closest_idx]
            
            # Check if closest timestamp is within reasonable tolerance
            if closest_time_diff < 0.3:  # 300ms tolerance for 2Hz (500ms steps)
                # Valid future position
                future_pos = positions[closest_idx]
                future_trajs.append(future_pos)
                future_masks.append([1.0, 1.0])  # Valid mask
            else:
                # No good match - use last known position and mark invalid
                if len(future_trajs) > 0:
                    future_trajs.append(future_trajs[-1])  # Repeat last position
                else:
                    future_trajs.append(positions[i])  # Use current position
                future_masks.append([0.0, 0.0])  # Invalid mask
        
        # Convert to numpy arrays and add batch dimension
        future_trajs = np.array(future_trajs)  # (predict_steps, 2)
        future_masks = np.array(future_masks)  # (predict_steps, 2)
        
        # Add batch dimension to make shape (1, predict_steps, 2)
        future_trajs = future_trajs[np.newaxis, :, :]  # (1, predict_steps, 2)
        future_masks = future_masks[np.newaxis, :, :]  # (1, predict_steps, 2)
        
        # Convert to torch tensors and append to list
        gt_sdc_fut_traj.append(torch.tensor(future_trajs, dtype=torch.float32))
        gt_sdc_fut_traj_mask.append(torch.tensor(future_masks, dtype=torch.float32))
        valid_timestamps.append(current_time)
    
    valid_timestamps = np.array(valid_timestamps)
    
    return gt_sdc_fut_traj, gt_sdc_fut_traj_mask, valid_timestamps

def save_sdc_ground_truth(gt_sdc_fut_traj, gt_sdc_fut_traj_mask, valid_timestamps, output_dir, bag_name):
    """
    Save SDC ground truth data in frame-aligned format
    
    Args:
        gt_sdc_fut_traj: list[torch.Tensor] where each tensor has shape (1, predict_steps, 2)
        gt_sdc_fut_traj_mask: list[torch.Tensor] where each tensor has shape (1, predict_steps, 2)
        valid_timestamps: numpy array of timestamps
        output_dir: output directory path
        bag_name: name of the bag file for naming
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save aggregated data
    output_file = os.path.join(output_dir, f"{bag_name}_sdc_gt.pt")
    
    sdc_gt_data = {
        'gt_sdc_fut_traj': gt_sdc_fut_traj,
        'gt_sdc_fut_traj_mask': gt_sdc_fut_traj_mask,
        'timestamps': valid_timestamps,
        'num_frames': len(valid_timestamps),
        'predict_steps': gt_sdc_fut_traj[0].shape[1] if len(gt_sdc_fut_traj) > 0 else 0
    }
    
    torch.save(sdc_gt_data, output_file)
    print(f"Saved SDC ground truth to {output_file}")
    
    # Also save frame-by-frame data for training alignment
    frame_dir = os.path.join(output_dir, f"{bag_name}")
    os.makedirs(frame_dir, exist_ok=True)
    
    for i, timestamp in enumerate(valid_timestamps):
        frame_data = {
            'gt_sdc_fut_traj': gt_sdc_fut_traj[i],  # Already has correct shape (1, predict_steps, 2)
            'gt_sdc_fut_traj_mask': gt_sdc_fut_traj_mask[i], 
            'timestamp': timestamp
        }
        
        # Use timestamp (in microseconds) as filename for clearer identification
        ts_us = int(round(float(timestamp) * 1_000_000))
        frame_file = os.path.join(frame_dir, f"{ts_us}.pt")
        torch.save(frame_data, frame_file)
    
    print(f"Saved {len(valid_timestamps)} frame-level SDC ground truth files to {frame_dir}")

def validate_trajectory_data(gt_sdc_fut_traj, gt_sdc_fut_traj_mask):
    """Validate extracted trajectory data"""
    
    print(f"Number of SDC trajectory samples: {len(gt_sdc_fut_traj)}")
    if len(gt_sdc_fut_traj) > 0:
        print(f"SDC Trajectory Shape per sample: {gt_sdc_fut_traj[0].shape}")
        print(f"SDC Mask Shape per sample: {gt_sdc_fut_traj_mask[0].shape}")
    
    # Check for valid data across all samples
    valid_frames = []
    for i, (traj, mask) in enumerate(zip(gt_sdc_fut_traj, gt_sdc_fut_traj_mask)):
        # Check if any step in the trajectory is valid
        is_valid = torch.sum(mask[0, :, 0]) > 0  # Check x-coordinate mask
        valid_frames.append(is_valid.item())
    
    num_valid = sum(valid_frames)
    print(f"Valid frames: {num_valid}/{len(gt_sdc_fut_traj)}")
    
    # Check trajectory statistics for valid frames
    if num_valid > 0:
        # Concatenate all valid trajectories for statistics
        valid_trajs = torch.cat([gt_sdc_fut_traj[i] for i, is_valid in enumerate(valid_frames) if is_valid], dim=0)
        
        x_range = [float(torch.min(valid_trajs[:, :, 0])), float(torch.max(valid_trajs[:, :, 0]))]
        y_range = [float(torch.min(valid_trajs[:, :, 1])), float(torch.max(valid_trajs[:, :, 1]))]
        
        print(f"X position range: [{x_range[0]:.2f}, {x_range[1]:.2f}]")
        print(f"Y position range: [{y_range[0]:.2f}, {y_range[1]:.2f}]")
        
        # Check for reasonable movement
        if valid_trajs.shape[1] > 1:  # Need at least 2 steps to calculate movement
            avg_movement = torch.mean(torch.norm(valid_trajs[:, 1:] - valid_trajs[:, :-1], dim=-1))
            print(f"Average step movement: {float(avg_movement):.3f} meters")

def process_bag_file(bag_file, output_dir, canbus_dir, global_timestamps, predict_steps=12, test_mode=False):
    """Process a single bag file and extract SDC ground truth using CAN bus data"""

    print(f"\nProcessing {bag_file}...")

    # Extract bag name for output naming
    bag_name = os.path.basename(bag_file).replace('.bag', '')

    # Extract car state data from CAN bus using global timestamps
    positions, timestamps = extract_car_state_from_canbus(canbus_dir, global_timestamps, bag_file)

    if positions is None:
        print(f"Skipping {bag_file} - no valid CAN bus data found")
        return 0

    print(f"Loaded {len(positions)} synchronized car state data points")

    # Extract SDC future trajectories
    gt_sdc_fut_traj, gt_sdc_fut_traj_mask, valid_timestamps = extract_sdc_future_trajectories(
        positions, timestamps, predict_steps, test_mode=test_mode)

    # Validate data
    validate_trajectory_data(gt_sdc_fut_traj, gt_sdc_fut_traj_mask)

    # Save ground truth data
    save_sdc_ground_truth(gt_sdc_fut_traj, gt_sdc_fut_traj_mask, valid_timestamps,
                         output_dir, bag_name)

    return len(valid_timestamps)

def main():
    parser = argparse.ArgumentParser(description='Extract SDC ground truth trajectories using CAN bus data and global timestamps')
    parser.add_argument('--bag_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files',
                       help='Directory containing bag files (used for naming only)')
    parser.add_argument('--canbus_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus',
                       help='Directory containing CAN bus pickle files')
    parser.add_argument('--timestamps_file', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl',
                       help='Global timestamps pickle file')
    parser.add_argument('--output_dir', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/gt_sdc',
                       help='Output directory for SDC ground truth')
    parser.add_argument('--predict_steps', type=int, default=12,
                       help='Number of future steps to predict (12 steps = 6 seconds at 2Hz)')
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

    # Find all bag files (used for naming and mapping to CAN bus files)
    bag_pattern = os.path.join(args.bag_dir, '*.bag')
    bag_files = glob.glob(bag_pattern)

    if not bag_files:
        print(f"No bag files found in {args.bag_dir}")
        return

    print(f"Found {len(bag_files)} bag files to process")
    print(f"Extracting {args.predict_steps} future steps (6 seconds at 2Hz)")

    if args.test:
        print("*** TEST MODE ENABLED: Processing only first 3 frames per bag ***")

    total_frames = 0
    processed_files = 0

    for bag_file in sorted(bag_files):
        frames_processed = process_bag_file(
            bag_file, args.output_dir, args.canbus_dir,
            global_timestamps, args.predict_steps, test_mode=args.test
        )
        if frames_processed > 0:
            total_frames += frames_processed
            processed_files += 1

    print(f"\n=== SDC Ground Truth Extraction Complete ===")
    print(f"Processed {processed_files}/{len(bag_files)} files successfully")
    print(f"Generated {total_frames} SDC trajectory samples")
    print(f"Used global timestamps with {hz} Hz sampling")
    print(f"Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()