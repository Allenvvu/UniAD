#!/usr/bin/env python3

import os
import pickle
import shutil
import json
import glob
from typing import Dict, List, Tuple, Any
import numpy as np
from timestamp_extract import find_closest_timestamp

def load_continuous_timestamps(timestamp_file: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load continuous timestamps from pickle file"""
    with open(timestamp_file, 'rb') as f:
        data = pickle.load(f)
    return data['timestamps'], data

def parse_image_timestamp(filename: str) -> float:
    """Convert nanosecond timestamp filename to seconds"""
    # Extract timestamp from filename (remove .jpg extension)
    timestamp_ns = int(os.path.splitext(filename)[0])
    # Convert nanoseconds to seconds
    timestamp_s = timestamp_ns / 1e9
    return timestamp_s

def scan_image_directory(image_dir: str) -> List[Tuple[str, float]]:
    """Scan image directory and return list of (filename, timestamp) tuples"""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_files = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            try:
                timestamp = parse_image_timestamp(filename)
                image_files.append((filename, timestamp))
            except ValueError:
                print(f"Warning: Could not parse timestamp from filename: {filename}")
                continue

    # Sort by timestamp
    image_files.sort(key=lambda x: x[1])
    print(f"Found {len(image_files)} valid image files")

    return image_files

def extract_synchronized_images(
    timestamp_file: str,
    image_dir: str,
    output_dir: str,
    tolerance: float = 0.5,
    copy_files: bool = True
) -> Dict[str, Any]:
    """
    Extract images synchronized to continuous timestamps

    Args:
        timestamp_file: Path to continuous timestamps pickle file
        image_dir: Directory containing source images
        output_dir: Directory to save synchronized images
        tolerance: Timestamp matching tolerance in seconds
        copy_files: If True, copy files; if False, create symlinks

    Returns:
        Dictionary with extraction statistics and metadata
    """

    print("="*60)
    print("SYNCHRONIZED IMAGE EXTRACTION")
    print("="*60)

    # Load continuous timestamps
    print(f"Loading timestamps from: {timestamp_file}")
    master_timestamps, timestamp_data = load_continuous_timestamps(timestamp_file)
    print(f"Loaded {len(master_timestamps)} master timestamps")

    # Scan image directory
    print(f"Scanning image directory: {image_dir}")
    image_files = scan_image_directory(image_dir)

    if not image_files:
        raise ValueError("No valid image files found in directory")

    print(f"Time range - Images: {image_files[0][1]:.6f} to {image_files[-1][1]:.6f}")
    print(f"Time range - Master: {master_timestamps[0]:.6f} to {master_timestamps[-1]:.6f}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create image lookup for faster access
    image_lookup = {timestamp: filename for filename, timestamp in image_files}
    image_timestamps = np.array([timestamp for _, timestamp in image_files])

    # Extract synchronized images
    matches = []
    unmatched = []

    print(f"\nMatching timestamps with tolerance: {tolerance}s")
    print("Progress: ", end="", flush=True)

    for i, target_timestamp in enumerate(master_timestamps):
        # Show progress
        if i % 100 == 0:
            print(f"{i}/{len(master_timestamps)}", end=" ", flush=True)

        # Find closest image timestamp
        closest_time, closest_idx = find_closest_timestamp(
            target_timestamp, image_timestamps, tolerance
        )

        if closest_time is not None:
            # Find corresponding filename
            closest_filename = None
            for filename, img_timestamp in image_files:
                if abs(img_timestamp - closest_time) < 1e-9:  # Exact match
                    closest_filename = filename
                    break

            if closest_filename:
                # Name output by reference (master) timestamp in nanoseconds
                output_filename = f"{int(target_timestamp * 1e9)}.jpg"

                matches.append({
                    'frame_id': i + 1,
                    'master_timestamp': target_timestamp,
                    'image_timestamp': closest_time,
                    'time_diff': abs(target_timestamp - closest_time),
                    'original_filename': closest_filename,
                    'output_filename': output_filename
                })

                # Copy or link file
                src_path = os.path.join(image_dir, closest_filename)
                dst_path = os.path.join(output_dir, output_filename)

                try:
                    if copy_files:
                        shutil.copy2(src_path, dst_path)
                    else:
                        if os.path.exists(dst_path):
                            os.unlink(dst_path)
                        os.symlink(src_path, dst_path)
                except Exception as e:
                    print(f"\nError copying {closest_filename}: {e}")
                    continue
        else:
            unmatched.append({
                'frame_id': i + 1,
                'master_timestamp': target_timestamp,
                'reason': 'no_match_within_tolerance'
            })

    print(f"\n\nExtraction complete!")

    # Calculate statistics
    stats = {
        'total_master_timestamps': len(master_timestamps),
        'total_available_images': len(image_files),
        'matched_frames': len(matches),
        'unmatched_frames': len(unmatched),
        'match_rate': len(matches) / len(master_timestamps) * 100,
        'tolerance_used': tolerance,
        'time_range': {
            'start': float(master_timestamps[0]),
            'end': float(master_timestamps[-1]),
            'duration': float(master_timestamps[-1] - master_timestamps[0])
        },
        'timestamp_data': timestamp_data
    }

    if matches:
        time_diffs = [m['time_diff'] for m in matches]
        stats['timing_accuracy'] = {
            'mean_time_diff': float(np.mean(time_diffs)),
            'max_time_diff': float(np.max(time_diffs)),
            'std_time_diff': float(np.std(time_diffs))
        }

    # Save metadata (convert numpy arrays to lists for JSON serialization)
    stats_json = stats.copy()
    if 'timestamp_data' in stats_json:
        # Remove numpy arrays from timestamp_data for JSON serialization
        stats_json['timestamp_data'] = {
            k: v for k, v in stats_json['timestamp_data'].items()
            if not isinstance(v, np.ndarray)
        }

    metadata = {
        'extraction_info': {
            'timestamp_file': timestamp_file,
            'image_dir': image_dir,
            'output_dir': output_dir,
            'extraction_method': 'copy' if copy_files else 'symlink'
        },
        'statistics': stats_json,
        'matches': matches,
        'unmatched': unmatched
    }

    metadata_file = os.path.join(output_dir, 'extraction_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print_extraction_summary(stats)

    return metadata

def print_extraction_summary(stats: Dict[str, Any]):
    """Print extraction summary statistics"""
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Total master timestamps: {stats['total_master_timestamps']}")
    print(f"Total available images: {stats['total_available_images']}")
    print(f"Successfully matched: {stats['matched_frames']}")
    print(f"Unmatched frames: {stats['unmatched_frames']}")
    print(f"Match rate: {stats['match_rate']:.1f}%")
    print(f"Tolerance used: {stats['tolerance_used']}s")

    if 'timing_accuracy' in stats:
        print(f"\nTiming Accuracy:")
        print(f"  Mean time difference: {stats['timing_accuracy']['mean_time_diff']:.3f}s")
        print(f"  Max time difference: {stats['timing_accuracy']['max_time_diff']:.3f}s")
        print(f"  Std time difference: {stats['timing_accuracy']['std_time_diff']:.3f}s")

    duration = stats['time_range']['duration']
    print(f"\nTime Coverage:")
    print(f"  Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    print(f"  Start: {stats['time_range']['start']:.6f}")
    print(f"  End: {stats['time_range']['end']:.6f}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Extract synchronized images using continuous timestamps")
    parser.add_argument(
        "--timestamps",
        default="timestamps/continuous_timestamps.pkl",
        help="Path to continuous timestamps pickle file"
    )
    parser.add_argument(
        "--images",
        default="/home/bryan/Desktop/image/camera/lucid_cameras_x01.gige_100_fr_hdr.h265",
        help="Directory containing source images"
    )
    parser.add_argument(
        "--output",
        default="extracted_images",
        help="Output directory for synchronized images"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="Timestamp matching tolerance in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space)"
    )

    args = parser.parse_args()

    try:
        metadata = extract_synchronized_images(
            timestamp_file=args.timestamps,
            image_dir=args.images,
            output_dir=args.output,
            tolerance=args.tolerance,
            copy_files=not args.symlink
        )

        print(f"\nExtraction metadata saved to: {os.path.join(args.output, 'extraction_metadata.json')}")
        print(f"Synchronized images saved to: {args.output}")

    except Exception as e:
        print(f"Error during extraction: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())