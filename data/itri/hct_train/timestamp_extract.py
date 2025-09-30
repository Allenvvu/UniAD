#!/usr/bin/env python3

import rosbag
import numpy as np
import pickle
import os
import argparse
import glob
from typing import Tuple, Dict, Any, List

def extract_bag_duration(bag_file: str) -> Tuple[float, float]:
    """Extract start and end timestamps from bag file"""
    bag = rosbag.Bag(bag_file)

    start_time = None
    end_time = None

    # Iterate through all messages to find time range
    for topic, msg, t in bag.read_messages():
        timestamp = t.to_sec()
        if start_time is None:
            start_time = timestamp
            end_time = timestamp
        else:
            start_time = min(start_time, timestamp)
            end_time = max(end_time, timestamp)

    bag.close()

    if start_time is None:
        raise ValueError(f"No messages found in bag file: {bag_file}")

    return start_time, end_time

def generate_master_timestamps(start_time: float, end_time: float, hz: float) -> np.ndarray:
    """Generate evenly spaced timestamps at specified frequency"""
    duration = end_time - start_time
    num_samples = int(duration * hz) + 1  # +1 to include end time

    # Generate evenly spaced timestamps
    timestamps = np.linspace(start_time, end_time, num_samples)

    return timestamps

def get_bag_files_from_directory(bag_dir: str) -> List[str]:
    """Get sorted list of bag files from directory"""
    bag_pattern = os.path.join(bag_dir, "*.bag")
    bag_files = sorted(glob.glob(bag_pattern))
    return bag_files

def create_continuous_timeline(bag_files: List[str], hz: float, output_file: str = None) -> Dict[str, Any]:
    """Create continuous master timeline across multiple bag files"""

    if not bag_files:
        raise ValueError("No bag files provided")

    print(f"Processing {len(bag_files)} bag files...")
    print(f"Target frequency: {hz} Hz")

    # Extract time ranges from all bag files
    bag_info = []
    overall_start = None
    overall_end = None

    for bag_file in bag_files:
        print(f"Scanning: {os.path.basename(bag_file)}")
        start_time, end_time = extract_bag_duration(bag_file)
        duration = end_time - start_time

        bag_info.append({
            'file': bag_file,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        })

        if overall_start is None:
            overall_start = start_time
            overall_end = end_time
        else:
            overall_start = min(overall_start, start_time)
            overall_end = max(overall_end, end_time)

    total_duration = overall_end - overall_start

    print(f"\nOverall time range: {overall_start:.6f} to {overall_end:.6f}")
    print(f"Total duration: {total_duration:.2f} seconds")

    # Generate continuous timestamps across all bags
    timestamps = generate_master_timestamps(overall_start, overall_end, hz)

    print(f"Generated {len(timestamps)} timestamps at {hz} Hz")

    # Create data dictionary with bag file info
    data = {
        'timestamps': timestamps,
        'hz': hz,
        'total_duration': total_duration,
        'overall_start_time': overall_start,
        'overall_end_time': overall_end,
        'num_samples': len(timestamps),
        'bag_files': [os.path.basename(info['file']) for info in bag_info],
        'bag_info': bag_info
    }

    # Save to file
    if output_file is None:
        output_file = f"continuous_timestamps_{hz}hz.pkl"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved continuous timeline to: {output_file}")

    return data

def create_master_timeline(bag_file: str, hz: float, output_file: str = None) -> Dict[str, Any]:
    """Create master timeline for synchronized data extraction (single bag file)"""

    print(f"Processing bag file: {bag_file}")
    print(f"Target frequency: {hz} Hz")

    # Extract bag duration
    start_time, end_time = extract_bag_duration(bag_file)
    duration = end_time - start_time

    print(f"Bag duration: {duration:.2f} seconds")
    print(f"Time range: {start_time:.6f} to {end_time:.6f}")

    # Generate master timestamps
    timestamps = generate_master_timestamps(start_time, end_time, hz)

    print(f"Generated {len(timestamps)} timestamps at {hz} Hz")

    # Create data dictionary
    data = {
        'timestamps': timestamps,
        'hz': hz,
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time,
        'bag_file': os.path.basename(bag_file),
        'num_samples': len(timestamps)
    }

    # Save to file
    if output_file is None:
        base_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_file = f"{base_name}_timestamps_{hz}hz.pkl"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved master timeline to: {output_file}")

    return data

def load_master_timestamps(timestamp_file: str) -> Tuple[np.ndarray, float]:
    """Load master timestamps from file"""
    with open(timestamp_file, 'rb') as f:
        data = pickle.load(f)
    return data['timestamps'], data['hz']

def find_closest_timestamp(target_time: float, master_timestamps: np.ndarray, tolerance: float = 0.5) -> Tuple[float, int]:
    """Find closest timestamp in master timeline within tolerance"""
    if len(master_timestamps) == 0:
        return None, None

    # Find index of closest timestamp
    idx = np.argmin(np.abs(master_timestamps - target_time))
    closest_time = master_timestamps[idx]

    # Check if within tolerance
    if abs(closest_time - target_time) <= tolerance:
        return closest_time, idx
    else:
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract master timeline from bag file(s)")
    parser.add_argument("input_path", help="Path to bag file or directory containing bag files")
    parser.add_argument("--hz", type=float, default=2.0, help="Target frequency in Hz (default: 2.0)")
    parser.add_argument(
        "--output", "-o",
        default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl",
        help="Output file path (default: /home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl)"
    )
    parser.add_argument("--tolerance", type=float, default=0.5, help="Timestamp matching tolerance in seconds (default: 0.5)")

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Path not found: {args.input_path}")
        exit(1)

    # Check if input is directory or single file
    if os.path.isdir(args.input_path):
        # Directory mode - process all bag files
        bag_files = get_bag_files_from_directory(args.input_path)
        if not bag_files:
            print(f"Error: No bag files found in directory: {args.input_path}")
            exit(1)

        # Create continuous timeline
        data = create_continuous_timeline(bag_files, args.hz, args.output)

        # Print summary
        print("\n" + "="*50)
        print("CONTINUOUS TIMELINE SUMMARY")
        print("="*50)
        print(f"Bag files processed: {len(data['bag_files'])}")
        print(f"Total duration: {data['total_duration']:.2f} seconds")
        print(f"Frequency: {data['hz']} Hz")
        print(f"Total samples: {data['num_samples']}")
        print(f"Time range: {data['overall_start_time']:.6f} - {data['overall_end_time']:.6f}")
        print(f"Sample interval: {1.0/data['hz']:.3f} seconds")
        print(f"\nBag files included:")
        for i, bag_file in enumerate(data['bag_files']):
            bag_info = data['bag_info'][i]
            print(f"  {i+1:2d}. {bag_file} ({bag_info['duration']:.1f}s)")

    else:
        # Single file mode
        if not args.input_path.endswith('.bag'):
            print(f"Error: Input file must be a .bag file: {args.input_path}")
            exit(1)

        # Create master timeline for single file
        data = create_master_timeline(args.input_path, args.hz, args.output)

        # Print summary
        print("\n" + "="*50)
        print("MASTER TIMELINE SUMMARY")
        print("="*50)
        print(f"Bag file: {data['bag_file']}")
        print(f"Duration: {data['duration']:.2f} seconds")
        print(f"Frequency: {data['hz']} Hz")
        print(f"Total samples: {data['num_samples']}")
        print(f"Time range: {data['start_time']:.6f} - {data['end_time']:.6f}")
        print(f"Sample interval: {1.0/data['hz']:.3f} seconds")