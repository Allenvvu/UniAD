#!/usr/bin/env python3

import rosbag
import numpy as np
from collections import defaultdict
import pickle
import os
import argparse
import tf.transformations
from typing import Dict, List, Tuple, Any
from timestamp_extract import find_closest_timestamp, load_master_timestamps

def extract_can_bus_data(
    bag_file: str,
    timestamp_file: str,
    output_file: str = None,
    tolerance: float = 0.5
) -> Dict[str, Any]:
    """Extract 18-D CAN bus vectors sampled on provided master timestamps.

    - Position and quaternion from `/car_state`
    - Linear acceleration and angular velocity from `/imu/data`
    - Velocity from `/filter/velocity`
    - Matching uses `find_closest_timestamp` against the master timestamps with tolerance
    """

    print(f"Processing {bag_file}...")

    # Load master timestamps
    master_timestamps, _hz = load_master_timestamps(timestamp_file)
    if master_timestamps is None or len(master_timestamps) == 0:
        raise ValueError(f"No timestamps found in: {timestamp_file}")

    # Open bag and cache streams
    bag = rosbag.Bag(bag_file)

    car_pose_stream: List[Tuple[float, Dict[str, float]]] = []
    velocity_stream: List[Tuple[float, Dict[str, float]]] = []
    imu_stream: List[Tuple[float, Dict[str, float]]] = []

    topics = ['/car_state', '/imu/data', '/filter/velocity']
    for topic, msg, t in bag.read_messages(topics=topics):
        ts = t.to_sec()
        if topic == '/car_state':
            car_pose_stream.append((ts, {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z,
                'qw': msg.pose.pose.orientation.w,
                'qx': msg.pose.pose.orientation.x,
                'qy': msg.pose.pose.orientation.y,
                'qz': msg.pose.pose.orientation.z,
            }))
        elif topic == '/filter/velocity':
            velocity_stream.append((ts, {
                'vel_x': msg.vector.x,
                'vel_y': msg.vector.y,
                'vel_z': msg.vector.z,
            }))
        elif topic == '/imu/data':
            imu_stream.append((ts, {
                'ori_w': msg.orientation.w,
                'ori_x': msg.orientation.x,
                'ori_y': msg.orientation.y,
                'ori_z': msg.orientation.z,
                'accel_x': msg.linear_acceleration.x,
                'accel_y': msg.linear_acceleration.y,
                'accel_z': msg.linear_acceleration.z,
                'angular_vel_x': msg.angular_velocity.x,
                'angular_vel_y': msg.angular_velocity.y,
                'angular_vel_z': msg.angular_velocity.z,
            }))

    bag.close()

    # Prepare arrays of timestamps for matching
    car_pose_stream.sort(key=lambda x: x[0])
    velocity_stream.sort(key=lambda x: x[0])
    imu_stream.sort(key=lambda x: x[0])

    car_pose_times = np.array([ts for ts, _ in car_pose_stream]) if car_pose_stream else np.array([])
    velocity_times = np.array([ts for ts, _ in velocity_stream]) if velocity_stream else np.array([])
    imu_times = np.array([ts for ts, _ in imu_stream]) if imu_stream else np.array([])

    if len(car_pose_times) == 0:
        raise ValueError("No /car_state messages found in bag")
    if len(imu_times) == 0:
        raise ValueError("No /imu/data messages found in bag")
    if len(velocity_times) == 0:
        print("Warning: No /filter/velocity messages found; velocities will be zeros")

    can_bus_rows: List[np.ndarray] = []
    matched_master_timestamps: List[float] = []

    print(f"Matching to {len(master_timestamps)} master timestamps (tol={tolerance}s)...")

    for idx, target_ts in enumerate(master_timestamps):
        # Find closest per stream
        car_time, car_idx = find_closest_timestamp(float(target_ts), car_pose_times, tolerance)
        imu_time, imu_idx = find_closest_timestamp(float(target_ts), imu_times, tolerance)
        vel_time, vel_idx = find_closest_timestamp(float(target_ts), velocity_times, tolerance) if len(velocity_times) else (None, None)

        if car_time is None or imu_time is None:
            continue  # require at least pose and imu

        car_pose = car_pose_stream[car_idx][1]
        imu = imu_stream[imu_idx][1]
        if vel_time is not None:
            vel = velocity_stream[vel_idx][1]
        else:
            vel = {'vel_x': 0.0, 'vel_y': 0.0, 'vel_z': 0.0}

        # Compute yaw from IMU quaternion if available; else from car_state
        imu_quat = (
            imu.get('ori_x', 0.0),
            imu.get('ori_y', 0.0),
            imu.get('ori_z', 0.0),
            imu.get('ori_w', 0.0),
        )
        if any(abs(c) > 1e-9 for c in imu_quat):
            _, _, yaw = tf.transformations.euler_from_quaternion(imu_quat)
        else:
            car_quat = (
                car_pose['qx'],
                car_pose['qy'],
                car_pose['qz'],
                car_pose['qw'],
            )
            _, _, yaw = tf.transformations.euler_from_quaternion(car_quat)

        row = np.array([
            car_pose['x'], car_pose['y'], car_pose['z'],
            car_pose['qw'], car_pose['qx'], car_pose['qy'], car_pose['qz'],
            imu['accel_x'], imu['accel_y'], imu['accel_z'],
            imu['angular_vel_x'], imu['angular_vel_y'], imu['angular_vel_z'],
            vel['vel_x'], vel['vel_y'], vel['vel_z'],
            yaw,
            0.0,
        ], dtype=float)

        can_bus_rows.append(row)
        matched_master_timestamps.append(float(target_ts))

    print(f"Extracted {len(can_bus_rows)} CAN bus samples (match rate: {len(can_bus_rows) / len(master_timestamps) * 100:.1f}%)")

    # Save data
    if output_file is None:
        output_file = bag_file.replace('.bag', '_can_bus.pkl')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    data_dict: Dict[str, Any] = {
        'can_bus': np.vstack(can_bus_rows) if can_bus_rows else np.zeros((0, 18), dtype=float),
        'timestamps': np.array(matched_master_timestamps, dtype=float),
        'timestamp_file': timestamp_file,
        'tolerance': float(tolerance),
        'fields': {
            'position_xyz': [0, 1, 2],
            'quaternion_wxyz': [3, 4, 5, 6],
            'linear_accel': [7, 8, 9],
            'angular_velocity': [10, 11, 12],
            'velocity_xyz': [13, 14, 15],
            'yaw': 16,
            'reserved': 17,
        }
    }

    with open(output_file, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Saved CAN bus data to {output_file}")
    return data_dict

if __name__ == "__main__":
    import glob

    parser = argparse.ArgumentParser(description="Extract CAN bus sampled on master timestamps")
    parser.add_argument(
        "--bags",
        default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files",
        help="Path to a .bag file or directory containing .bag files"
    )
    parser.add_argument(
        "--timestamps",
        default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl",
        help="Path to pickle file with master timestamps (from timestamp_extract.py)"
    )
    parser.add_argument(        
        "--tolerance",
        type=float,
        default=0.5,
        help="Matching tolerance in seconds"
    )
    parser.add_argument(
        "--outdir",
        default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus",
        help="Output directory for extracted CAN bus files"
    )

    args = parser.parse_args()

    input_path = args.bags
    if os.path.isdir(input_path):
        bag_files = sorted(glob.glob(os.path.join(input_path, "*.bag")))
    else:
        if not input_path.endswith('.bag'):
            print(f"Error: --bags must be a directory or .bag file: {input_path}")
            exit(1)
        bag_files = [input_path]

    if not bag_files:
        print(f"No bag files found in {input_path}")
        exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    for bag_file in bag_files:
        base_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_file = os.path.join(args.outdir, f"{base_name}_can_bus.pkl")

        if not os.path.exists(bag_file):
            print(f"Error: {bag_file} does not exist")
            continue

        extract_can_bus_data(
            bag_file=bag_file,
            timestamp_file=args.timestamps,
            output_file=output_file,
            tolerance=args.tolerance,
        )
    

# Output Format Documentation
"""
The extract_can_bus_data function outputs a pickle file containing a dictionary with two keys:
1. 'can_bus': numpy array of shape (N, 18) where N is the number of synchronized data points
   The 18 dimensions represent:
   - [0-2]:   Global position (x, y, z)
   - [3-6]:   Quaternion rotation (qw, qx, qy, qz)
   - [7-9]:   Linear acceleration (accel_x, accel_y, accel_z)
   - [10-12]: Angular velocity (angular_vel_x, angular_vel_y, angular_vel_z)
   - [13-15]: Vehicle velocity (vel_x, vel_y, vel_z)
   - [16]:    Ego angle (yaw) in radians
   - [17]:    Reserved value (currently set to 0.0)

2. 'timestamps': numpy array of shape (N,) containing the corresponding timestamps
   for each CAN bus data point in seconds (Unix timestamp)

Note: Data points are synchronized between car_state and IMU messages with a 100ms tolerance.
"""



