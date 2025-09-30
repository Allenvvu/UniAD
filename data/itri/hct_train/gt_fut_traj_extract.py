#!/usr/bin/env python3
"""
ITRI ROS Bag to Ground Truth Future Trajectory Converter

Extracts ground truth future trajectories from ITRI ROS bag files for UniAD training.
For each frame at time T, generates future trajectories for all tracked objects
from T+1 to T+12 (6 seconds at 2Hz sampling rate).

Input: ROS bag file with /detected_objects topic
Output: Per-frame gt_fut_traj .pt files with future trajectory data

Author: UniAD Integration Project

Data Format:
- gt_fut_traj: list[torch.Tensor] = (num_objects, predict_steps, 2)
  - num_objects: Number of tracked objects in the frame
  - predict_steps: 12 (future 6 seconds at 2Hz sampling)
  - 2: 2D coordinates (x, y) in ego coordinate system

- gt_fut_traj_mask: list[torch.Tensor] = (num_objects, predict_steps, 2)
  - Same shape as gt_fut_traj
  - Mask indicating valid future positions (1.0) vs missing (0.0)
"""

import rosbag
import torch
import numpy as np
import json
import os
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import timestamp alignment functions
try:
    from timestamp_extract import load_master_timestamps, find_closest_timestamp
    logger.info("Successfully imported timestamp functions")
except ImportError as e:
    logger.error(f"Failed to import timestamp functions: {e}")
    load_master_timestamps = None
    find_closest_timestamp = None

@dataclass
class DetectedObject:
    """Standardized detected object format from ITRI ROS messages"""
    id: int
    label: str
    score: float
    tracked_period: float
    # Position
    x: float
    y: float
    z: float
    # Velocity
    vx: float
    vy: float
    vz: float
    # Meta
    space_frame: str
    behavior_state: int
    timestamp: float
    frame_index: int

    def get_position_2d(self) -> Tuple[float, float]:
        """Get 2D position (x, y) for trajectory"""
        return (self.x, self.y)


class ITRIBagToFutTrajConverter:
    """
    Converts ITRI ROS bag files to ground truth future trajectory format for UniAD training.

    For each frame at time T, extracts future trajectories of all tracked objects
    from time T+1 to T+12 (6 seconds future at 2Hz sampling rate).
    """

    def __init__(self,
                 bag_file_path: str = None,
                 bag_directory: str = None,
                 output_dir: str = None,
                 global_timestamps_file: str = None,
                 predict_steps: int = 12,
                 sampling_hz: float = 2.0):
        """
        Initialize converter.

        Args:
            bag_file_path: Path to single ITRI ROS bag file (optional)
            bag_directory: Directory containing multiple bag files (optional)
            output_dir: Output directory for gt_fut_traj files
            global_timestamps_file: Path to global timestamps pickle file
            predict_steps: Number of future steps to predict (default: 12)
            sampling_hz: Sampling frequency in Hz (default: 2.0)
        """

        # Input validation
        if bag_file_path and bag_directory:
            raise ValueError("Specify either bag_file_path OR bag_directory, not both")
        if not bag_file_path and not bag_directory:
            bag_directory = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files"

        self.bag_file_path = Path(bag_file_path) if bag_file_path else None
        self.bag_directory = Path(bag_directory) if bag_directory else None

        # Output directory setup
        if output_dir is None:
            output_dir = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/gt_fut_traj"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Trajectory parameters
        self.predict_steps = predict_steps
        self.sampling_hz = sampling_hz
        self.time_step = 1.0 / sampling_hz  # 0.5 seconds at 2Hz

        # Load global timestamps for alignment
        if global_timestamps_file is None:
            global_timestamps_file = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl"

        if load_master_timestamps and Path(global_timestamps_file).exists():
            self.global_timestamps, self.timestamp_hz = load_master_timestamps(global_timestamps_file)
            logger.info(f"Loaded {len(self.global_timestamps)} global timestamps at {self.timestamp_hz} Hz")
        else:
            logger.warning("Global timestamps not available, using sequential processing")
            self.global_timestamps = None
            self.timestamp_hz = None

        # Processing statistics
        self.stats = {
            'total_bags_processed': 0,
            'total_frames_processed': 0,
            'total_objects_processed': 0,
            'total_trajectories_generated': 0,
            'failed_frames': [],
            'bags_summary': []
        }

        logger.info(f"Initialized GT Future Trajectory Converter")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Predict steps: {self.predict_steps} (future {self.predict_steps * self.time_step:.1f} seconds)")
        logger.info(f"Sampling: {self.sampling_hz} Hz")

    def find_bag_files(self) -> List[str]:
        """
        Find all .bag files to process.

        Returns:
            List of paths to .bag files
        """
        bag_files = []

        if self.bag_file_path:
            # Single file mode
            if self.bag_file_path.exists() and self.bag_file_path.suffix == '.bag':
                bag_files.append(str(self.bag_file_path))
            else:
                logger.error(f"Bag file not found or invalid: {self.bag_file_path}")

        elif self.bag_directory:
            # Directory mode
            if not self.bag_directory.exists():
                logger.error(f"Bag directory does not exist: {self.bag_directory}")
                return bag_files

            # Search for .bag files
            for bag_file in sorted(self.bag_directory.glob("*.bag")):
                bag_files.append(str(bag_file))

        logger.info(f"Found {len(bag_files)} bag files to process")
        return bag_files

    def extract_detected_objects_from_bag(self, bag_file_path: str) -> Dict[int, List[DetectedObject]]:
        """
        Extract detected objects from a single ROS bag file aligned to global timestamps.

        Args:
            bag_file_path: Path to ROS bag file

        Returns:
            Dictionary mapping frame_index -> List[DetectedObject]
        """
        logger.info(f"Extracting detected objects from: {Path(bag_file_path).name}")

        detected_objects_by_frame = {}

        if self.global_timestamps is not None:
            # Use global timestamp alignment
            return self._extract_objects_with_global_timestamps(bag_file_path)
        else:
            # Fallback to sequential processing
            return self._extract_objects_sequential(bag_file_path)

    def _extract_objects_with_global_timestamps(self, bag_file_path: str) -> Dict[int, List[DetectedObject]]:
        """Extract objects with proper one-to-one mapping to global timestamps"""
        logger.info("Using global timestamp alignment with one-to-one mapping")

        # First, collect all ROS messages with their timestamps
        ros_messages = []

        try:
            bag = rosbag.Bag(bag_file_path)
            topic_name = '/detected_objects'

            for topic, msg, ros_time in tqdm(bag.read_messages(topics=[topic_name]),
                                           desc="Loading ROS messages"):
                timestamp = ros_time.to_sec()
                ros_messages.append((timestamp, msg))

            bag.close()
        except Exception as e:
            logger.error(f"Failed to load ROS messages: {e}")
            raise

        logger.info(f"Loaded {len(ros_messages)} ROS messages")

        # Implement one-to-one mapping: for each global timestamp, find best ROS message
        detected_objects_by_frame = {}
        used_ros_indices = set()  # Track which ROS messages have been used
        tolerance = 0.25  # 250ms tolerance for timestamp matching

        # Get the time range of ROS messages to filter relevant global timestamps
        if ros_messages:
            ros_start_time = min(ts for ts, _ in ros_messages)
            ros_end_time = max(ts for ts, _ in ros_messages)

            # Filter global timestamps to only those in the ROS message time range (with buffer)
            time_buffer = 1.0  # 1 second buffer
            relevant_global_indices = []
            for i, global_ts in enumerate(self.global_timestamps):
                if ros_start_time - time_buffer <= global_ts <= ros_end_time + time_buffer:
                    relevant_global_indices.append(i)

            logger.info(f"Filtered to {len(relevant_global_indices)} relevant global timestamps")
            logger.info(f"from {len(self.global_timestamps)} total (time range: {ros_start_time:.3f} - {ros_end_time:.3f})")
        else:
            logger.warning("No ROS messages found")
            return detected_objects_by_frame

        # For each relevant global timestamp, find the best matching ROS message
        for global_idx in tqdm(relevant_global_indices, desc="Mapping global timestamps to ROS messages"):
            global_timestamp = self.global_timestamps[global_idx]

            best_match = None
            best_time_diff = float('inf')
            best_ros_idx = None

            # Find the closest unused ROS message to this global timestamp
            for ros_idx, (ros_timestamp, msg) in enumerate(ros_messages):
                if ros_idx in used_ros_indices:
                    continue  # Skip already used ROS messages

                time_diff = abs(ros_timestamp - global_timestamp)
                if time_diff < best_time_diff and time_diff <= tolerance:
                    best_time_diff = time_diff
                    best_match = (ros_timestamp, msg)
                    best_ros_idx = ros_idx

            if best_match is not None:
                # Mark this ROS message as used
                used_ros_indices.add(best_ros_idx)

                # Process with the global timestamp to ensure proper alignment
                ros_timestamp, msg = best_match
                frame_objects = self._process_ros_message(msg, global_timestamp, global_idx)
                detected_objects_by_frame[global_idx] = frame_objects

                logger.debug(f"Mapped global timestamp {global_timestamp:.6f} to ROS message {ros_timestamp:.6f}")

        logger.info(f"Successful mappings: {len(detected_objects_by_frame)}/{len(relevant_global_indices)}")
        return detected_objects_by_frame

    def _extract_objects_sequential(self, bag_file_path: str) -> Dict[int, List[DetectedObject]]:
        """Extract objects using sequential processing (fallback method)"""
        logger.info("Using sequential processing for object extraction")

        detected_objects_by_frame = {}
        frame_index = 0

        try:
            bag = rosbag.Bag(bag_file_path)
            topic_name = '/detected_objects'

            # Process all messages from detected_objects topic
            for topic, msg, ros_time in tqdm(bag.read_messages(topics=[topic_name]),
                                           desc="Processing ROS messages"):

                timestamp = ros_time.to_sec()
                frame_objects = self._process_ros_message(msg, timestamp, frame_index)
                detected_objects_by_frame[frame_index] = frame_objects
                frame_index += 1

            bag.close()

        except Exception as e:
            logger.error(f"Failed to process ROS bag: {e}")
            raise

        return detected_objects_by_frame

    def _process_ros_message(self, msg, timestamp: float, frame_index: int) -> List[DetectedObject]:
        """Process a single ROS message and extract detected objects"""
        frame_objects = []

        # Process each object in the message
        for obj in msg.objects:
            try:
                detected_obj = DetectedObject(
                    id=obj.id,
                    label=obj.label,
                    score=obj.score,
                    tracked_period=obj.trackedPeriod,
                    # Position
                    x=obj.pose.position.x,
                    y=obj.pose.position.y,
                    z=obj.pose.position.z,
                    # Velocity
                    vx=obj.velocity.linear.x,
                    vy=obj.velocity.linear.y,
                    vz=obj.velocity.linear.z,
                    # Meta
                    space_frame=obj.space_frame,
                    behavior_state=obj.behavior_state,
                    timestamp=timestamp,
                    frame_index=frame_index
                )

                frame_objects.append(detected_obj)
                self.stats['total_objects_processed'] += 1

            except Exception as e:
                logger.error(f"Failed to process object {obj.id} in frame {frame_index}: {e}")
                continue

        return frame_objects

    def extract_future_trajectories_multi_bag(self) -> Dict[str, Any]:
        """
        Extract future trajectories from multiple bag files with continuous timeline.

        Returns:
            Processing summary dictionary
        """
        logger.info("Starting future trajectory extraction from multiple bag files...")

        # Find all bag files to process
        bag_files = self.find_bag_files()
        if not bag_files:
            logger.error("No bag files found to process")
            return {'error': 'No bag files found'}

        # Extract objects from all bag files and merge into continuous timeline
        all_objects_by_frame = {}

        for bag_file in tqdm(bag_files, desc="Processing bag files"):
            try:
                bag_objects = self.extract_detected_objects_from_bag(bag_file)
                all_objects_by_frame.update(bag_objects)
                self.stats['total_bags_processed'] += 1

                bag_summary = {
                    'bag_file': Path(bag_file).name,
                    'frames_extracted': len(bag_objects),
                    'objects_extracted': sum(len(objects) for objects in bag_objects.values())
                }
                self.stats['bags_summary'].append(bag_summary)

            except Exception as e:
                logger.error(f"Failed to process bag file {bag_file}: {e}")
                continue

        logger.info(f"Extracted objects from {len(all_objects_by_frame)} frames across {len(bag_files)} bag files")

        # Generate future trajectories for each frame
        trajectory_data = self.generate_future_trajectories(all_objects_by_frame)

        return trajectory_data

    def generate_future_trajectories(self, all_objects_by_frame: Dict[int, List[DetectedObject]]) -> Dict[str, Any]:
        """
        Generate future trajectories for all frames using multi-frame lookahead.

        Args:
            all_objects_by_frame: Dictionary mapping frame_index -> List[DetectedObject]

        Returns:
            Dictionary containing trajectory data for each frame
        """
        logger.info("Generating future trajectories with multi-frame lookahead...")

        # Sort frame indices to ensure temporal order
        sorted_frame_indices = sorted(all_objects_by_frame.keys())
        max_frame_idx = max(sorted_frame_indices) if sorted_frame_indices else 0

        logger.info(f"Processing {len(sorted_frame_indices)} frames (indices: {min(sorted_frame_indices)} to {max_frame_idx})")

        trajectory_results = {}

        for frame_idx in tqdm(sorted_frame_indices, desc="Generating trajectories"):
            try:
                # Get objects in current frame
                current_frame_objects = all_objects_by_frame[frame_idx]

                if not current_frame_objects:
                    logger.debug(f"No objects in frame {frame_idx}, skipping")
                    continue

                # Generate future trajectories for this frame
                frame_trajectory_data = self._generate_frame_future_trajectories(
                    frame_idx, current_frame_objects, all_objects_by_frame, max_frame_idx
                )

                if frame_trajectory_data:
                    trajectory_results[frame_idx] = frame_trajectory_data
                    self.stats['total_frames_processed'] += 1
                    self.stats['total_trajectories_generated'] += len(frame_trajectory_data['object_ids'])

            except Exception as e:
                logger.error(f"Failed to generate trajectories for frame {frame_idx}: {e}")
                self.stats['failed_frames'].append(frame_idx)
                continue

        logger.info(f"Generated trajectories for {len(trajectory_results)} frames")
        return trajectory_results

    def _generate_frame_future_trajectories(self,
                                          current_frame_idx: int,
                                          current_objects: List[DetectedObject],
                                          all_objects_by_frame: Dict[int, List[DetectedObject]],
                                          max_frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Generate future trajectories for a single frame.

        Args:
            current_frame_idx: Index of current frame
            current_objects: Objects present in current frame
            all_objects_by_frame: All objects across all frames
            max_frame_idx: Maximum frame index available

        Returns:
            Dictionary containing trajectory data for this frame
        """
        # Build object ID to position mapping for current frame
        current_object_positions = {obj.id: obj for obj in current_objects}
        object_ids = list(current_object_positions.keys())

        if not object_ids:
            return None

        num_objects = len(object_ids)

        # Initialize trajectory arrays
        gt_fut_traj = np.zeros((num_objects, self.predict_steps, 2), dtype=np.float32)
        gt_fut_traj_mask = np.zeros((num_objects, self.predict_steps, 2), dtype=np.float32)

        # For each object, look ahead predict_steps frames
        for obj_idx, object_id in enumerate(object_ids):
            for step in range(self.predict_steps):
                future_frame_idx = current_frame_idx + step + 1  # +1 because we want future frames

                # Check if future frame exists
                if future_frame_idx > max_frame_idx or future_frame_idx not in all_objects_by_frame:
                    # Future frame doesn't exist, leave as zeros with mask=0
                    continue

                # Find this object in the future frame
                future_objects = all_objects_by_frame[future_frame_idx]
                future_object = self._find_object_by_id(future_objects, object_id)

                if future_object:
                    # Object found in future frame, record its position
                    x, y = future_object.get_position_2d()
                    gt_fut_traj[obj_idx, step, 0] = x
                    gt_fut_traj[obj_idx, step, 1] = y
                    gt_fut_traj_mask[obj_idx, step, 0] = 1.0
                    gt_fut_traj_mask[obj_idx, step, 1] = 1.0
                else:
                    # Object not found in future frame, keep as zeros with mask=0
                    pass

        # Convert to tensors
        gt_fut_traj_tensor = torch.from_numpy(gt_fut_traj)
        gt_fut_traj_mask_tensor = torch.from_numpy(gt_fut_traj_mask)

        # Prepare frame data
        frame_timestamp = current_objects[0].timestamp if current_objects else 0.0

        frame_data = {
            'gt_fut_traj': gt_fut_traj_tensor,
            'gt_fut_traj_mask': gt_fut_traj_mask_tensor,
            'object_ids': object_ids,
            'timestamp': frame_timestamp,
            'frame_index': current_frame_idx,
            'num_objects': num_objects,
            'predict_steps': self.predict_steps,
            'current_positions': [(current_object_positions[obj_id].x, current_object_positions[obj_id].y)
                                for obj_id in object_ids]
        }

        return frame_data

    def _find_object_by_id(self, objects: List[DetectedObject], target_id: int) -> Optional[DetectedObject]:
        """
        Find an object by ID in a list of objects.

        Args:
            objects: List of DetectedObject instances
            target_id: ID to search for

        Returns:
            DetectedObject if found, None otherwise
        """
        for obj in objects:
            if obj.id == target_id:
                return obj
        return None

    def validate_trajectory_data(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate trajectory data and add quality metrics.

        Args:
            frame_data: Frame trajectory data

        Returns:
            Updated frame data with validation metrics
        """
        gt_fut_traj = frame_data['gt_fut_traj']
        gt_fut_traj_mask = frame_data['gt_fut_traj_mask']
        object_ids = frame_data['object_ids']

        validation_metrics = {
            'total_objects': len(object_ids),
            'valid_trajectories': 0,
            'partial_trajectories': 0,
            'empty_trajectories': 0,
            'trajectory_lengths': [],
            'max_displacement': 0.0,
            'avg_displacement': 0.0
        }

        total_displacement = 0.0
        max_displacement = 0.0

        for obj_idx in range(len(object_ids)):
            # Count valid steps for this object
            valid_steps = int(gt_fut_traj_mask[obj_idx, :, 0].sum().item())
            validation_metrics['trajectory_lengths'].append(valid_steps)

            if valid_steps == 0:
                validation_metrics['empty_trajectories'] += 1
            elif valid_steps == self.predict_steps:
                validation_metrics['valid_trajectories'] += 1
            else:
                validation_metrics['partial_trajectories'] += 1

            # Calculate displacement for valid steps
            if valid_steps > 0:
                trajectory = gt_fut_traj[obj_idx, :valid_steps, :]  # Only valid positions
                if len(trajectory) > 1:
                    # Calculate displacement between consecutive points
                    displacements = torch.norm(trajectory[1:] - trajectory[:-1], dim=1)
                    obj_max_displacement = displacements.max().item()
                    obj_avg_displacement = displacements.mean().item()

                    max_displacement = max(max_displacement, obj_max_displacement)
                    total_displacement += obj_avg_displacement

        validation_metrics['max_displacement'] = max_displacement
        validation_metrics['avg_displacement'] = total_displacement / max(1, len(object_ids))

        # Add validation metrics to frame data
        frame_data['validation_metrics'] = validation_metrics

        return frame_data

    def interpolate_missing_trajectory_points(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Interpolate missing points in trajectories using linear interpolation.

        Args:
            frame_data: Frame trajectory data

        Returns:
            Updated frame data with interpolated trajectories
        """
        gt_fut_traj = frame_data['gt_fut_traj'].clone()
        gt_fut_traj_mask = frame_data['gt_fut_traj_mask'].clone()

        interpolation_stats = {
            'interpolated_objects': 0,
            'interpolated_points': 0
        }

        for obj_idx in range(gt_fut_traj.shape[0]):
            mask = gt_fut_traj_mask[obj_idx, :, 0]  # Use x-coordinate mask
            valid_indices = torch.where(mask > 0)[0]

            if len(valid_indices) < 2:
                # Need at least 2 points for interpolation
                continue

            # Find gaps in the trajectory
            gaps_filled = False
            for i in range(len(valid_indices) - 1):
                start_idx = valid_indices[i]
                end_idx = valid_indices[i + 1]

                # Check if there's a gap
                if end_idx - start_idx > 1:
                    # Linear interpolation for missing points
                    start_pos = gt_fut_traj[obj_idx, start_idx, :]
                    end_pos = gt_fut_traj[obj_idx, end_idx, :]

                    # Interpolate missing points
                    for gap_idx in range(start_idx + 1, end_idx):
                        alpha = (gap_idx - start_idx) / (end_idx - start_idx)
                        interpolated_pos = start_pos + alpha * (end_pos - start_pos)

                        gt_fut_traj[obj_idx, gap_idx, :] = interpolated_pos
                        gt_fut_traj_mask[obj_idx, gap_idx, :] = 0.5  # Mark as interpolated

                        interpolation_stats['interpolated_points'] += 1
                        gaps_filled = True

            if gaps_filled:
                interpolation_stats['interpolated_objects'] += 1

        # Update frame data
        frame_data['gt_fut_traj'] = gt_fut_traj
        frame_data['gt_fut_traj_mask'] = gt_fut_traj_mask
        frame_data['interpolation_stats'] = interpolation_stats

        return frame_data

    def filter_unrealistic_trajectories(self, frame_data: Dict[str, Any],
                                       max_velocity: float = 50.0,
                                       max_acceleration: float = 10.0) -> Dict[str, Any]:
        """
        Filter out unrealistic trajectories based on velocity and acceleration constraints.

        Args:
            frame_data: Frame trajectory data
            max_velocity: Maximum allowed velocity (m/s)
            max_acceleration: Maximum allowed acceleration (m/s²)

        Returns:
            Updated frame data with filtered trajectories
        """
        gt_fut_traj = frame_data['gt_fut_traj']
        gt_fut_traj_mask = frame_data['gt_fut_traj_mask']

        filtering_stats = {
            'velocity_violations': 0,
            'acceleration_violations': 0,
            'filtered_objects': 0
        }

        time_step = self.time_step  # Time between frames

        for obj_idx in range(gt_fut_traj.shape[0]):
            mask = gt_fut_traj_mask[obj_idx, :, 0]
            valid_indices = torch.where(mask > 0)[0]

            if len(valid_indices) < 2:
                continue

            trajectory = gt_fut_traj[obj_idx, valid_indices, :]
            object_filtered = False

            # Check velocity constraints
            velocities = torch.norm(trajectory[1:] - trajectory[:-1], dim=1) / time_step
            if (velocities > max_velocity).any():
                filtering_stats['velocity_violations'] += 1
                object_filtered = True

            # Check acceleration constraints
            if len(velocities) > 1:
                accelerations = torch.abs(velocities[1:] - velocities[:-1]) / time_step
                if (accelerations > max_acceleration).any():
                    filtering_stats['acceleration_violations'] += 1
                    object_filtered = True

            # If object violates constraints, mark as invalid
            if object_filtered:
                gt_fut_traj_mask[obj_idx, :, :] = 0.0
                filtering_stats['filtered_objects'] += 1

        frame_data['gt_fut_traj_mask'] = gt_fut_traj_mask
        frame_data['filtering_stats'] = filtering_stats

        return frame_data

    def save_trajectory_data(self, trajectory_results: Dict[str, Any], bag_name: str) -> bool:
        """
        Save trajectory data in the same format as track_query structure.

        Args:
            trajectory_results: Dictionary containing trajectory data for each frame
            bag_name: Bag file name for directory naming

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output directory structure matching track_query
            bag_output_dir = self.output_dir / bag_name
            bag_output_dir.mkdir(exist_ok=True, parents=True)

            frame_dir = bag_output_dir / bag_name
            frame_dir.mkdir(exist_ok=True, parents=True)

            # Prepare aggregated data for summary file
            aggregated_data = {
                'gt_fut_traj_list': [],
                'gt_fut_traj_mask_list': [],
                'object_ids_list': [],
                'timestamps': [],
                'frame_indices': [],
                'num_frames': len(trajectory_results),
                'predict_steps': self.predict_steps,
                'validation_metrics': [],
                'interpolation_stats': [],
                'filtering_stats': []
            }

            saved_frames = 0

            # Save frame-by-frame data
            for frame_idx, frame_data in tqdm(trajectory_results.items(), desc="Saving trajectory data"):
                try:
                    # Apply validation and processing
                    processed_frame_data = self.validate_trajectory_data(frame_data)
                    processed_frame_data = self.interpolate_missing_trajectory_points(processed_frame_data)
                    processed_frame_data = self.filter_unrealistic_trajectories(processed_frame_data)

                    timestamp = processed_frame_data['timestamp']

                    # Prepare individual frame data
                    individual_frame_data = {
                        'gt_fut_traj': processed_frame_data['gt_fut_traj'],
                        'gt_fut_traj_mask': processed_frame_data['gt_fut_traj_mask'],
                        'object_ids': processed_frame_data['object_ids'],
                        'timestamp': timestamp,
                        'frame_index': frame_idx,
                        'num_objects': processed_frame_data['num_objects'],
                        'predict_steps': self.predict_steps,
                        'current_positions': processed_frame_data['current_positions'],
                        'validation_metrics': processed_frame_data.get('validation_metrics', {}),
                        'interpolation_stats': processed_frame_data.get('interpolation_stats', {}),
                        'filtering_stats': processed_frame_data.get('filtering_stats', {})
                    }

                    # Use timestamp (in microseconds) as filename - same format as track_query
                    ts_us = int(round(float(timestamp) * 1_000_000))
                    frame_file = frame_dir / f"{ts_us}.pt"
                    torch.save(individual_frame_data, frame_file)

                    # Add to aggregated data
                    aggregated_data['gt_fut_traj_list'].append(processed_frame_data['gt_fut_traj'])
                    aggregated_data['gt_fut_traj_mask_list'].append(processed_frame_data['gt_fut_traj_mask'])
                    aggregated_data['object_ids_list'].append(processed_frame_data['object_ids'])
                    aggregated_data['timestamps'].append(timestamp)
                    aggregated_data['frame_indices'].append(frame_idx)
                    aggregated_data['validation_metrics'].append(processed_frame_data.get('validation_metrics', {}))
                    aggregated_data['interpolation_stats'].append(processed_frame_data.get('interpolation_stats', {}))
                    aggregated_data['filtering_stats'].append(processed_frame_data.get('filtering_stats', {}))

                    saved_frames += 1

                except Exception as e:
                    logger.error(f"Failed to save frame {frame_idx}: {e}")
                    continue

            # Save aggregated data
            aggregated_file = bag_output_dir / f"{bag_name}_gt_fut_traj.pt"
            torch.save(aggregated_data, aggregated_file)

            # Save processing summary
            processing_summary = {
                'bag_name': bag_name,
                'output_directory': str(bag_output_dir),
                'total_frames': len(trajectory_results),
                'saved_frames': saved_frames,
                'failed_frames': len(trajectory_results) - saved_frames,
                'predict_steps': self.predict_steps,
                'sampling_hz': self.sampling_hz,
                'processing_stats': self.stats,
                'aggregated_file': str(aggregated_file),
                'frame_directory': str(frame_dir)
            }

            summary_file = bag_output_dir / "processing_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(processing_summary, f, indent=2, default=str)

            logger.info(f"Saved {saved_frames} frame-level gt_fut_traj files to {frame_dir}")
            logger.info(f"Saved aggregated gt_fut_traj data to {aggregated_file}")
            logger.info(f"Processing summary saved to {summary_file}")

            return True

        except Exception as e:
            logger.error(f"Failed to save trajectory data: {e}")
            return False

    def process_all_bags(self) -> Dict[str, Any]:
        """
        Complete processing pipeline for all bag files.

        Returns:
            Overall processing summary
        """
        logger.info("Starting complete gt_fut_traj extraction pipeline...")

        try:
            # Extract future trajectories
            trajectory_results = self.extract_future_trajectories_multi_bag()

            if not trajectory_results or 'error' in trajectory_results:
                logger.error("Failed to extract trajectories")
                return trajectory_results

            # Group results by bag for saving
            bag_trajectory_results = defaultdict(dict)

            # Get bag files for grouping (assuming one bag processed at a time for now)
            bag_files = self.find_bag_files()

            if len(bag_files) == 1:
                # Single bag mode
                bag_name = Path(bag_files[0]).stem
                save_success = self.save_trajectory_data(trajectory_results, bag_name)

                summary = {
                    'mode': 'single_bag',
                    'bag_file': bag_files[0],
                    'bag_name': bag_name,
                    'total_frames': len(trajectory_results),
                    'save_success': save_success,
                    'processing_stats': self.stats
                }

            else:
                # Multi-bag mode - would need frame-to-bag mapping for proper grouping
                logger.warning("Multi-bag processing detected - saving all results under 'multi_bag'")
                save_success = self.save_trajectory_data(trajectory_results, "multi_bag")

                summary = {
                    'mode': 'multi_bag',
                    'bag_files': bag_files,
                    'total_bags': len(bag_files),
                    'total_frames': len(trajectory_results),
                    'save_success': save_success,
                    'processing_stats': self.stats
                }

            # Save overall summary
            overall_summary_file = self.output_dir / "overall_gt_fut_traj_summary.json"
            with open(overall_summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            logger.info(f"Overall processing complete. Summary saved to: {overall_summary_file}")
            return summary

        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            return {'error': str(e), 'processing_stats': self.stats}


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description='Extract GT future trajectories from ITRI ROS bag(s)')
    parser.add_argument('--input_path', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files",
                       help='Path to ITRI ROS bag file or directory containing bag files')
    parser.add_argument('--output-dir', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/gt_fut_traj",
                       help='Output directory')
    parser.add_argument('--timestamps-file', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl",
                       help='Global timestamps pickle file')
    parser.add_argument('--predict-steps', type=int, default=12,
                       help='Number of future steps to predict')
    parser.add_argument('--sampling-hz', type=float, default=2.0,
                       help='Sampling frequency in Hz')

    args = parser.parse_args()

    input_path = Path(args.input_path)

    try:
        if input_path.is_file() and input_path.suffix == '.bag':
            # Single bag file mode
            converter = ITRIBagToFutTrajConverter(
                bag_file_path=str(input_path),
                output_dir=args.output_dir,
                global_timestamps_file=args.timestamps_file,
                predict_steps=args.predict_steps,
                sampling_hz=args.sampling_hz
            )
        elif input_path.is_dir():
            # Directory mode
            converter = ITRIBagToFutTrajConverter(
                bag_directory=str(input_path),
                output_dir=args.output_dir,
                global_timestamps_file=args.timestamps_file,
                predict_steps=args.predict_steps,
                sampling_hz=args.sampling_hz
            )
        else:
            logger.error(f"Input path must be a .bag file or directory: {input_path}")
            return 1

        # Start processing
        logger.info("Starting GT future trajectory extraction...")

        # Run the complete processing pipeline
        summary = converter.process_all_bags()

        if 'error' in summary:
            logger.error(f"Processing failed: {summary['error']}")
            print(f"GT future trajectories extraction failed!")
            print(f"Error: {summary['error']}")
            return 1

        # Print comprehensive summary
        print(f"\n=== GT FUTURE TRAJECTORIES EXTRACTION SUMMARY ===")
        print(f"Mode: {summary.get('mode', 'unknown')}")

        if summary['mode'] == 'single_bag':
            print(f"Input bag file: {summary['bag_name']}")
            print(f"Total frames processed: {summary['total_frames']}")
            print(f"Save successful: {summary['save_success']}")
        else:
            print(f"Total bags processed: {summary['total_bags']}")
            print(f"Total frames processed: {summary['total_frames']}")
            print(f"Save successful: {summary['save_success']}")

        # Print processing statistics
        stats = summary['processing_stats']
        print(f"\n=== DETAILED STATISTICS ===")
        print(f"Bags processed: {stats['total_bags_processed']}")
        print(f"Frames processed: {stats['total_frames_processed']}")
        print(f"Objects processed: {stats['total_objects_processed']}")
        print(f"Trajectories generated: {stats['total_trajectories_generated']}")
        print(f"Failed frames: {len(stats['failed_frames'])}")

        if stats['bags_summary']:
            print(f"\n=== BAG-BY-BAG SUMMARY ===")
            for i, bag_summary in enumerate(stats['bags_summary']):
                print(f"  {i+1}. {bag_summary['bag_file']}: {bag_summary['frames_extracted']} frames, {bag_summary['objects_extracted']} objects")

        print(f"\nGT future trajectories ready for training!")
        logger.info("GT future trajectory extraction completed successfully")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())