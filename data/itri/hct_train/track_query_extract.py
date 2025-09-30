#!/usr/bin/env python3
"""
ITRI ROS Bag to Track Query Converter

Extracts detected objects from ITRI ROS bag files and converts them to 
MotionFormer-compatible track query format for UniAD training.

Input: ROS bag file with /detected_objects topic
Output: Per-frame track query .pt files ready for training

Author: UniAD Integration Project

## Notes: using global timestamps for time allignment.
# can be found: data/itri/hct_train/timestamps/continuous_timestamps.pkl
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
import tf.transformations
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'track'))
# Add explicit track path
sys.path.append('/home/bryan/Desktop/Allen/UniAD/track')

try:
    from track_query_builder import ITRITrackQueryBuilder, TrackQueryData
    logger.info("Successfully imported track_query_builder")
except ImportError as e:
    logger.error(f"Failed to import track_query_builder: {e}")
    ITRITrackQueryBuilder = None
    TrackQueryData = None

# Import timestamp alignment functions
try:
    from timestamp_extract import load_master_timestamps, find_closest_timestamp
    logger.info("Successfully imported timestamp functions")
except ImportError as e:
    logger.error(f"Failed to import timestamp functions: {e}")
    load_master_timestamps = None
    find_closest_timestamp = None

# Import SDC embedding functions
try:
    from sdc_embedding_extract import (
        load_canbus_data, find_canbus_file_for_bag,
        create_sdc_embedding_from_canbus, create_sdc_track_bbox_results
    )
    logger.info("Successfully imported SDC embedding functions")
except ImportError as e:
    logger.error(f"Failed to import SDC embedding functions: {e}")
    load_canbus_data = None
    find_canbus_file_for_bag = None
    create_sdc_embedding_from_canbus = None
    create_sdc_track_bbox_results = None


@dataclass
class DetectedObject:
    """Standardized detected object format from ITRI ROS messages"""
    id: int
    label: str
    score: float
    tracked_period: float
    # Position and orientation
    x: float
    y: float  
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    # Dimensions
    length: float
    width: float
    height: float
    # Velocity
    vx: float
    vy: float
    vz: float
    # Additional fields
    variance_x: float
    variance_y: float
    variance_z: float
    space_frame: str
    behavior_state: int
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'label': self.label,
            'score': self.score,
            'tracked_period': self.tracked_period,
            'bev_x': self.x,
            'bev_y': self.y,
            'bev_z': self.z,
            'qx': self.qx,
            'qy': self.qy,
            'qz': self.qz,
            'qw': self.qw,
            'length': self.length,
            'width': self.width,
            'height': self.height,
            'vx': self.vx,
            'vy': self.vy,
            'vz': self.vz,
            'variance_x': self.variance_x,
            'variance_y': self.variance_y,
            'variance_z': self.variance_z,
            'space_frame': self.space_frame,
            'behavior_state': self.behavior_state,
            'timestamp': self.timestamp,
            # Computed fields - use quaternion-based yaw for individual objects
            'yaw': self.get_yaw(),  # Uses quaternion conversion by default
            'class_id': self.get_class_id(),
            'confidence': self.score,
            'angular_velocity': 0.0,  # Not available in current data
            'original_tracked_period': self.tracked_period
        }
    
    def get_yaw(self, canbus_data=None, canbus_timestamps=None, tolerance=0.1) -> float:
        """Get yaw angle from canbus data or fallback to quaternion conversion"""
        # Try to get yaw from aligned canbus data first
        if canbus_data is not None and canbus_timestamps is not None and find_closest_timestamp:
            closest_time, closest_idx = find_closest_timestamp(
                self.timestamp, canbus_timestamps, tolerance=tolerance
            )
            if closest_time is not None:
                # Canbus yaw is at index 16 (see canbus_data_format.md)
                return canbus_data[closest_idx][16]

        # Fallback to quaternion conversion
        quaternion = (self.qx, self.qy, self.qz, self.qw)
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw
    
    def get_class_id(self) -> int:
        """Map ITRI labels to UniAD class IDs (0-9)"""
        # UniAD class mapping
        label_mapping = {
            'car': 0,
            'truck': 1, 
            'trailer': 2,
            'bus': 3,
            'construction_vehicle': 4,
            'bicycle': 5,
            'motorbike': 6,
            'motorcycle': 6,  # Alternative name
            'pedestrian': 7,
            'person': 7,  # Alternative name
            'traffic_cone': 8,
            'barrier': 9,
            '': 0,  # Default for empty labels
            'unknown': 0  # Default for unknown labels
        }
        return label_mapping.get(self.label.lower(), 0)


class ITRIBagToTrackQueryConverter:
    """
    Converts ITRI ROS bag files to track query format for UniAD training.
    """
    
    def __init__(self,
                 bag_file_path: str,
                 output_dir: str = None,
                 device: str = 'cpu',
                 global_timestamps_file: str = None,
                 canbus_dir: str = None):
        """
        Initialize converter.

        Args:
            bag_file_path: Path to ITRI ROS bag file
            output_dir: Output directory for track queries (default: same dir as bag file)
            device: Device for track query builder
            global_timestamps_file: Path to global timestamps pickle file
            canbus_dir: Directory containing canbus data files
        """
        self.bag_file_path = Path(bag_file_path)

        if output_dir is None:
            output_dir = self.bag_file_path.parent / 'track_query'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.device = device

        # Load global timestamps
        if global_timestamps_file is None:
            global_timestamps_file = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl"

        if load_master_timestamps and Path(global_timestamps_file).exists():
            self.global_timestamps, self.timestamp_hz = load_master_timestamps(global_timestamps_file)
            logger.info(f"Loaded {len(self.global_timestamps)} global timestamps at {self.timestamp_hz} Hz")
        else:
            logger.warning("Global timestamps not available, using ROS message timestamps")
            self.global_timestamps = None
            self.timestamp_hz = None

        # Set canbus directory
        if canbus_dir is None:
            canbus_dir = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus"
        self.canbus_dir = canbus_dir

        # Load canbus data for yaw extraction
        self.canbus_data = None
        self.canbus_timestamps = None
        if load_canbus_data and find_canbus_file_for_bag:
            canbus_file = find_canbus_file_for_bag(str(self.bag_file_path), self.canbus_dir)
            if canbus_file:
                self.canbus_data, self.canbus_timestamps = load_canbus_data(canbus_file)
                if self.canbus_data is not None:
                    logger.info(f"Loaded {len(self.canbus_data)} canbus samples")
                else:
                    logger.warning("Failed to load canbus data")

        # Initialize track query builder if available
        if ITRITrackQueryBuilder:
            self.track_builder = ITRITrackQueryBuilder(device=device)
        else:
            logger.warning("Track query builder not available, will save raw data only")
            self.track_builder = None

        # Processing statistics
        self.stats = {
            'total_messages': 0,
            'total_objects': 0,
            'frames_processed': 0,
            'failed_frames': []
        }

        logger.info(f"Initialized converter for {self.bag_file_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Canbus directory: {self.canbus_dir}")
    
    def extract_detected_objects(self) -> Dict[int, List[DetectedObject]]:
        """
        Extract detected objects from ROS bag file aligned to global timestamps.

        Returns:
            Dictionary mapping frame_index -> List[DetectedObject]
        """
        logger.info("Extracting detected objects from ROS bag...")

        detected_objects_by_frame = {}

        if self.global_timestamps is not None:
            # Use global timestamp alignment
            return self._extract_objects_with_global_timestamps()
        else:
            # Fallback to sequential processing
            return self._extract_objects_sequential()

    def _extract_objects_with_global_timestamps(self) -> Dict[int, List[DetectedObject]]:
        """Extract objects with proper one-to-one mapping to global timestamps"""
        logger.info("Using global timestamp alignment with one-to-one mapping")

        # First, collect all ROS messages with their timestamps
        ros_messages = []

        try:
            bag = rosbag.Bag(str(self.bag_file_path))
            topic_name = '/detected_objects'

            for topic, msg, ros_time in tqdm(bag.read_messages(topics=[topic_name]),
                                           desc="Loading ROS messages"):
                timestamp = ros_time.to_sec()
                ros_messages.append((timestamp, msg))
                self.stats['total_messages'] += 1

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

            logger.info(f"Filtered to {len(relevant_global_indices)} relevant global timestamps ")
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

                logger.debug(f"Mapped global timestamp {global_timestamp:.6f} to ROS message {ros_timestamp:.6f} ")
                logger.debug(f"(diff: {best_time_diff:.3f}s)")
            else:
                logger.debug(f"No unused ROS message within tolerance for global timestamp {global_timestamp:.6f}")

        # Log mapping statistics
        logger.info(f"=== MAPPING STATISTICS ===")
        logger.info(f"Total ROS messages: {len(ros_messages)}")
        logger.info(f"Used ROS messages: {len(used_ros_indices)}")
        logger.info(f"Skipped ROS messages: {len(ros_messages) - len(used_ros_indices)}")
        logger.info(f"Global timestamps processed: {len(relevant_global_indices)}")
        logger.info(f"Successful mappings: {len(detected_objects_by_frame)}")
        logger.info(f"Mapping efficiency: {len(detected_objects_by_frame)}/{len(relevant_global_indices)} ")
        logger.info(f"({100.0 * len(detected_objects_by_frame) / len(relevant_global_indices):.1f}%)")

        return detected_objects_by_frame

    def _extract_objects_sequential(self) -> Dict[int, List[DetectedObject]]:
        """Extract objects using sequential processing (fallback method)"""
        logger.info("Using sequential processing for object extraction")

        detected_objects_by_frame = {}
        frame_index = 0
        
        try:
            bag = rosbag.Bag(str(self.bag_file_path))
            topic_name = '/detected_objects'

            # Process all messages from detected_objects topic
            for topic, msg, ros_time in tqdm(bag.read_messages(topics=[topic_name]),
                                           desc="Processing ROS messages"):

                timestamp = ros_time.to_sec()
                frame_objects = self._process_ros_message(msg, timestamp, frame_index)
                detected_objects_by_frame[frame_index] = frame_objects
                frame_index += 1
                self.stats['total_messages'] += 1

            bag.close()
            
        except Exception as e:
            logger.error(f"Failed to process ROS bag: {e}")
            raise
        
        logger.info(f"Extracted {self.stats['total_objects']} objects from {self.stats['total_messages']} frames")
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
                    # Orientation
                    qx=obj.pose.orientation.x,
                    qy=obj.pose.orientation.y,
                    qz=obj.pose.orientation.z,
                    qw=obj.pose.orientation.w,
                    # Dimensions
                    length=obj.dimensions.x,  # ROS convention: x=length
                    width=obj.dimensions.y,   # y=width
                    height=obj.dimensions.z,  # z=height
                    # Velocity
                    vx=obj.velocity.linear.x,
                    vy=obj.velocity.linear.y,
                    vz=obj.velocity.linear.z,
                    # Variance
                    variance_x=obj.variance.x,
                    variance_y=obj.variance.y,
                    variance_z=obj.variance.z,
                    # Meta
                    space_frame=obj.space_frame,
                    behavior_state=obj.behavior_state,
                    timestamp=timestamp
                )

                frame_objects.append(detected_obj)
                self.stats['total_objects'] += 1

            except Exception as e:
                logger.error(f"Failed to process object {obj.id} in frame {frame_index}: {e}")
                continue

        return frame_objects
    
    def convert_frame_to_track_query(self, 
                                   frame_objects: List[DetectedObject],
                                   frame_index: int) -> Optional[Any]:
        """
        Convert frame objects to track query format.
        
        Args:
            frame_objects: List of detected objects in frame
            frame_index: Frame index number
            
        Returns:
            TrackQueryData object or None if conversion fails
        """
        if not self.track_builder:
            logger.error("Track query builder not available")
            return None
            
        try:
            # Convert to format expected by track query builder
            transformed_objects = []
            for obj in frame_objects:
                obj_dict = obj.to_dict()
                # Use individual object's quaternion-based yaw (not canbus yaw)
                # Canbus yaw is for ego vehicle only, each detected object has its own orientation
                obj_dict['yaw'] = obj.get_yaw()  # This uses quaternion conversion by default
                transformed_objects.append(obj_dict)

            # Generate SDC embedding and bbox results aligned to this frame
            sdc_embedding = None
            sdc_track_bbox_results = None

            if (create_sdc_embedding_from_canbus and create_sdc_track_bbox_results and
                self.canbus_data is not None and self.canbus_timestamps is not None and
                find_closest_timestamp):

                # Find closest canbus sample for this frame
                target_time = frame_objects[0].timestamp if frame_objects else None
                if target_time is not None:
                    closest_time, closest_idx = find_closest_timestamp(
                        target_time, self.canbus_timestamps, tolerance=0.25
                    )
                    if closest_time is not None:
                        canbus_sample = self.canbus_data[closest_idx]
                        sdc_embedding = create_sdc_embedding_from_canbus(canbus_sample, embed_dim=256)
                        sdc_track_bbox_results = create_sdc_track_bbox_results(canbus_sample)

            # Generate track queries using existing builder
            track_query_data = self.track_builder.build_track_queries(
                transformed_objects, sdc_embedding, sdc_track_bbox_results
            )
            
            return track_query_data
            
        except Exception as e:
            logger.error(f"Failed to convert frame {frame_index} to track queries: {e}")
            return None
    

    def save_track_query_data(self, track_query_data_list: List[Any],
                             valid_timestamps: List[float], bag_name: str) -> bool:
        """
        Save track query data in the same format as save_sdc_data from sdc_embedding_extract.py

        Args:
            track_query_data_list: List of track query data for each frame
            valid_timestamps: List of timestamps for each frame
            bag_name: Bag file name for naming

        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Save aggregated data
            output_file = self.output_dir / f"{bag_name}_track_queries.pt"

            # Prepare aggregated data
            track_query_embeddings_list = []
            track_query_matched_idxes_list = []
            track_bbox_results_list = []
            sdc_embeddings_list = []
            sdc_track_bbox_results_list = []

            for track_data in track_query_data_list:
                track_query_embeddings_list.append(track_data.track_query_embeddings)
                track_query_matched_idxes_list.append(track_data.track_query_matched_idxes)
                track_bbox_results_list.append(track_data.track_bbox_results)
                if track_data.sdc_embedding is not None:
                    sdc_embeddings_list.append(track_data.sdc_embedding)
                if track_data.sdc_track_bbox_results is not None:
                    sdc_track_bbox_results_list.append(track_data.sdc_track_bbox_results)

            aggregated_data = {
                'track_query_embeddings': track_query_embeddings_list,
                'track_query_matched_idxes': track_query_matched_idxes_list,
                'track_bbox_results': track_bbox_results_list,
                'sdc_embeddings': sdc_embeddings_list if sdc_embeddings_list else None,
                'sdc_track_bbox_results': sdc_track_bbox_results_list if sdc_track_bbox_results_list else None,
                'timestamps': valid_timestamps,
                'num_frames': len(valid_timestamps),
                'embedding_dim': track_query_embeddings_list[0].shape[-1] if track_query_embeddings_list else 0
            }

            torch.save(aggregated_data, output_file)
            logger.info(f"Saved aggregated track queries to {output_file}")

            # Save frame-by-frame data for training alignment (same format as save_sdc_data)
            frame_dir = self.output_dir / bag_name
            frame_dir.mkdir(exist_ok=True, parents=True)

            for i, timestamp in enumerate(valid_timestamps):
                frame_data = {
                    'track_query_embeddings': track_query_data_list[i].track_query_embeddings,
                    'track_query_matched_idxes': track_query_data_list[i].track_query_matched_idxes,
                    'track_bbox_results': track_query_data_list[i].track_bbox_results,
                    'sdc_embedding': track_query_data_list[i].sdc_embedding,
                    'sdc_track_bbox_results': track_query_data_list[i].sdc_track_bbox_results,
                    'timestamp': timestamp,
                    'num_objects': len(track_query_data_list[i].track_query_matched_idxes)
                }

                # Use timestamp (in microseconds) as filename - same format as save_sdc_data
                ts_us = int(round(float(timestamp) * 1_000_000))
                frame_file = frame_dir / f"{ts_us}.pt"
                torch.save(frame_data, frame_file)

            logger.info(f"Saved {len(valid_timestamps)} frame-level track query files to {frame_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to save track query data: {e}")
            return False
    
    def process_bag_file(self) -> Dict[str, Any]:
        """
        Complete processing pipeline for ROS bag file.

        Returns:
            Processing summary dictionary
        """
        logger.info("Starting complete bag file processing...")

        # Step 1: Extract detected objects
        detected_objects_by_frame = self.extract_detected_objects()

        # Step 2: Process each frame and collect track query data
        track_query_data_list = []
        valid_timestamps = []
        processing_log = []

        for frame_index, frame_objects in tqdm(detected_objects_by_frame.items(),
                                             desc="Converting to track queries"):
            try:
                # Convert to track query format
                track_query_data = self.convert_frame_to_track_query(frame_objects, frame_index)

                if track_query_data is None:
                    logger.error(f"Failed to convert frame {frame_index}")
                    self.stats['failed_frames'].append(frame_index)
                    continue

                # Get timestamp for this frame (should be available from processed objects)
                if frame_objects:
                    timestamp = frame_objects[0].timestamp
                else:
                    logger.warning(f"No objects in frame {frame_index}, skipping")
                    self.stats['failed_frames'].append(frame_index)
                    continue

                # Collect data for batch saving
                track_query_data_list.append(track_query_data)
                valid_timestamps.append(timestamp)
                self.stats['frames_processed'] += 1

                processing_log.append({
                    'frame_index': frame_index,
                    'timestamp': timestamp,
                    'num_objects': len(frame_objects),
                    'track_queries_shape': list(track_query_data.track_query_embeddings.shape)
                })

            except Exception as e:
                logger.error(f"Error processing frame {frame_index}: {e}")
                self.stats['failed_frames'].append(frame_index)
                continue

        # Step 3: Save all track query data in timestamp-aligned format
        # Note: Only saves data for timestamps where this bag actually has ROS messages
        # This prevents creating empty/zero data files for all global timestamps
        bag_name = self.bag_file_path.stem
        if track_query_data_list:
            save_success = self.save_track_query_data(track_query_data_list, valid_timestamps, bag_name)
            if not save_success:
                logger.error("Failed to save track query data")
        else:
            logger.warning("No track query data to save")
        # Step 4: Save processing summary
        summary = {
            'bag_file': str(self.bag_file_path),
            'output_directory': str(self.output_dir),
            'processing_stats': self.stats,
            'processing_log': processing_log,
            'total_frames': len(detected_objects_by_frame),
            'successful_frames': self.stats['frames_processed'],
            'failed_frames': len(self.stats['failed_frames'])
        }
        
        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Processing complete: {self.stats['frames_processed']}/{len(detected_objects_by_frame)} frames successful")
        logger.info(f"Summary saved to: {summary_file}")
        
        return summary


def find_bag_files(directory_path: str) -> List[str]:
    """
    Find all .bag files in a directory and its subdirectories.
    
    Args:
        directory_path: Path to directory to search
        
    Returns:
        List of paths to .bag files
    """
    bag_files = []
    directory = Path(directory_path)
    
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory_path}")
        return bag_files
    
    if not directory.is_dir():
        logger.error(f"Path is not a directory: {directory_path}")
        return bag_files
    
    # Search for .bag files recursively
    for bag_file in directory.rglob("*.bag"):
        bag_files.append(str(bag_file))
    
    logger.info(f"Found {len(bag_files)} bag files in {directory_path}")
    return sorted(bag_files)


def process_single_bag(bag_file_path: str, output_dir: str, device: str, 
                      timestamps_file: str, canbus_dir: str) -> Dict[str, Any]:
    """
    Process a single bag file.
    
    Args:
        bag_file_path: Path to bag file
        output_dir: Output directory
        device: Device for processing
        timestamps_file: Global timestamps file
        canbus_dir: Canbus directory
        
    Returns:
        Processing summary
    """
    logger.info(f"Processing bag file: {bag_file_path}")
    
    # Create bag-specific output directory
    bag_name = Path(bag_file_path).stem
    bag_output_dir = Path(output_dir) / bag_name
    
    # Initialize converter
    converter = ITRIBagToTrackQueryConverter(
        bag_file_path=bag_file_path,
        output_dir=str(bag_output_dir),
        device=device,
        global_timestamps_file=timestamps_file,
        canbus_dir=canbus_dir
    )
    
    # Process bag file
    summary = converter.process_bag_file()
    return summary


def process_directory(bag_dir: str, output_dir: str, device: str, 
                     timestamps_file: str, canbus_dir: str) -> Dict[str, Any]:
    """
    Process all bag files in a directory.
    
    Args:
        bag_dir: Directory containing bag files
        output_dir: Output directory
        device: Device for processing
        timestamps_file: Global timestamps file
        canbus_dir: Canbus directory
        
    Returns:
        Overall processing summary
    """
    logger.info(f"Processing directory: {bag_dir}")
    
    # Find all bag files
    bag_files = find_bag_files(bag_dir)
    
    if not bag_files:
        logger.error("No bag files found in directory")
        return {
            'total_bags': 0,
            'successful_bags': 0,
            'failed_bags': 0,
            'bag_summaries': []
        }
    
    # Process each bag file
    successful_bags = 0
    failed_bags = 0
    bag_summaries = []
    
    for i, bag_file in enumerate(tqdm(bag_files, desc="Processing bag files")):
        try:
            logger.info(f"Processing bag {i+1}/{len(bag_files)}: {Path(bag_file).name}")
            summary = process_single_bag(bag_file, output_dir, device, 
                                       timestamps_file, canbus_dir)
            bag_summaries.append(summary)
            successful_bags += 1
            
        except Exception as e:
            logger.error(f"Failed to process {bag_file}: {e}")
            failed_bags += 1
            bag_summaries.append({
                'bag_file': bag_file,
                'error': str(e),
                'successful_frames': 0,
                'total_frames': 0
            })
    
    # Create overall summary
    overall_summary = {
        'input_directory': bag_dir,
        'output_directory': output_dir,
        'total_bags': len(bag_files),
        'successful_bags': successful_bags,
        'failed_bags': failed_bags,
        'bag_summaries': bag_summaries
    }
    
    # Save overall summary
    summary_file = Path(output_dir) / "overall_processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    logger.info(f"Directory processing complete: {successful_bags}/{len(bag_files)} bags successful")
    logger.info(f"Overall summary saved to: {summary_file}")
    
    return overall_summary


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ITRI ROS bag(s) to track queries')
    parser.add_argument('--input_path', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files",
                       help='Path to ITRI ROS bag file or directory containing bag files')
    parser.add_argument('--output-dir', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/track_query",
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for track query builder')
    parser.add_argument('--timestamps-file', type=str, 
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl",
                       help='Global timestamps pickle file')
    parser.add_argument('--canbus-dir', type=str,
                       default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus/sort",
                       help='Directory containing canbus data files')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    try:
        if input_path.is_file() and input_path.suffix == '.bag':
            # Process single bag file
            summary = process_single_bag(
                str(input_path), args.output_dir, args.device,
                args.timestamps_file, args.canbus_dir
            )
            
            print(f"\n=== CONVERSION SUMMARY ===")
            print(f"Input bag file: {summary['bag_file']}")
            print(f"Output directory: {summary['output_directory']}")
            print(f"Total messages: {summary['processing_stats']['total_messages']}")
            print(f"Total objects: {summary['processing_stats']['total_objects']}")
            print(f"Successful frames: {summary['successful_frames']}/{summary['total_frames']}")
            
            if summary['failed_frames'] > 0:
                print(f"Failed frames: {summary['failed_frames']}")
                
        elif input_path.is_dir():
            # Process directory of bag files
            summary = process_directory(
                str(input_path), args.output_dir, args.device,
                args.timestamps_file, args.canbus_dir
            )
            
            print(f"\n=== DIRECTORY PROCESSING SUMMARY ===")
            print(f"Input directory: {summary['input_directory']}")
            print(f"Output directory: {summary['output_directory']}")
            print(f"Total bags: {summary['total_bags']}")
            print(f"Successful bags: {summary['successful_bags']}")
            print(f"Failed bags: {summary['failed_bags']}")
            
            if summary['failed_bags'] > 0:
                print(f"\nFailed bag files:")
                for bag_summary in summary['bag_summaries']:
                    if 'error' in bag_summary:
                        print(f"  - {Path(bag_summary['bag_file']).name}: {bag_summary['error']}")
            
        else:
            logger.error(f"Input path must be a .bag file or directory: {input_path}")
            return 1
            
        print(f"Track queries ready for training!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())