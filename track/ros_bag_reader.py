#!/usr/bin/env python3
"""
ROS Bag Data Reader for ITRI Tracking Data

This module extracts tracking data from ITRI ROS bags, specifically targeting 
the /detected_objects topic for UniAD MotionFormer conversion.

Based on data_extraction_plan.md - Phase 1 implementation
"""

import rosbag
import rospy
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import tf.transformations
from collections import defaultdict

@dataclass
class ITRITrackingObject:
    """Data structure for extracted ITRI tracking object"""
    id: int
    label: str
    score: float
    tracked_period: float
    timestamp: float
    
    # Position in base_link frame
    position_x: float
    position_y: float  
    position_z: float
    
    # Orientation as quaternion
    orientation_x: float
    orientation_y: float
    orientation_z: float
    orientation_w: float
    
    # Object dimensions (L, W, H)
    dimension_x: float
    dimension_y: float
    dimension_z: float
    
    # Linear velocity
    velocity_x: float
    velocity_y: float
    velocity_z: float
    
    # Angular velocity (only z-component typically used)
    angular_velocity_z: float
    
    # Position variance/uncertainty
    variance_x: float
    variance_y: float
    variance_z: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
        
    def get_yaw_angle(self) -> float:
        """Convert quaternion to yaw angle using existing find_yaw.py approach"""
        quaternion = (self.orientation_x, self.orientation_y, 
                     self.orientation_z, self.orientation_w)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw
        
    def get_position_tuple(self) -> Tuple[float, float, float]:
        """Get position as tuple"""
        return (self.position_x, self.position_y, self.position_z)
        
    def get_velocity_tuple(self) -> Tuple[float, float, float]:
        """Get velocity as tuple"""
        return (self.velocity_x, self.velocity_y, self.velocity_z)
        
    def get_dimensions_tuple(self) -> Tuple[float, float, float]:
        """Get dimensions as tuple (L, W, H)"""
        return (self.dimension_x, self.dimension_y, self.dimension_z)


class ITRIBagReader:
    """
    ROS bag reader for ITRI tracking data extraction.
    
    Targets /detected_objects topic and extracts complete object information
    for MotionFormer conversion.
    """
    
    def __init__(self, bag_directory: str):
        """
        Initialize bag reader.
        
        Args:
            bag_directory: Path to directory containing ITRI bag files
        """
        self.bag_directory = Path(bag_directory)
        self.target_topic = '/detected_objects'
        
        # Find all bag files
        self.bag_files = sorted(list(self.bag_directory.glob('*.bag')))
        print(f"Found {len(self.bag_files)} bag files:")
        for bag_file in self.bag_files:
            print(f"  - {bag_file.name}")
            
        # Storage for extracted data
        self.extracted_objects: List[ITRITrackingObject] = []
        self.temporal_sequences: Dict[int, List[ITRITrackingObject]] = defaultdict(list)
        
    def extract_object_from_message(self, msg_obj, timestamp: float) -> ITRITrackingObject:
        """
        Extract ITRITrackingObject from ROS message object.
        
        Args:
            msg_obj: Individual object from DetectedObjectArray message
            timestamp: Message timestamp as float
            
        Returns:
            ITRITrackingObject with all extracted fields
        """
        try:
            return ITRITrackingObject(
                id=msg_obj.id,
                label=msg_obj.label,
                score=msg_obj.score,
                tracked_period=msg_obj.trackedPeriod,
                timestamp=timestamp,
                
                # Position
                position_x=msg_obj.pose.position.x,
                position_y=msg_obj.pose.position.y,
                position_z=msg_obj.pose.position.z,
                
                # Orientation
                orientation_x=msg_obj.pose.orientation.x,
                orientation_y=msg_obj.pose.orientation.y,
                orientation_z=msg_obj.pose.orientation.z,
                orientation_w=msg_obj.pose.orientation.w,
                
                # Dimensions
                dimension_x=msg_obj.dimensions.x,
                dimension_y=msg_obj.dimensions.y,
                dimension_z=msg_obj.dimensions.z,
                
                # Linear velocity
                velocity_x=msg_obj.velocity.linear.x,
                velocity_y=msg_obj.velocity.linear.y,
                velocity_z=msg_obj.velocity.linear.z,
                
                # Angular velocity (z-component)
                angular_velocity_z=msg_obj.velocity.angular.z,
                
                # Position variance
                variance_x=msg_obj.variance.x,
                variance_y=msg_obj.variance.y,
                variance_z=msg_obj.variance.z
            )
        except AttributeError as e:
            print(f"Warning: Missing field in object message: {e}")
            return None
            
    def extract_from_single_bag(self, bag_file: Path) -> int:
        """
        Extract tracking data from a single bag file.
        
        Args:
            bag_file: Path to bag file
            
        Returns:
            Number of objects extracted
        """
        objects_count = 0
        
        print(f"\nProcessing: {bag_file.name}")
        
        try:
            with rosbag.Bag(str(bag_file), 'r') as bag:
                # Check if target topic exists
                topics = bag.get_type_and_topic_info()[1].keys()
                if self.target_topic not in topics:
                    print(f"  Warning: {self.target_topic} not found in {bag_file.name}")
                    return 0
                    
                # Process messages
                message_count = 0
                for topic, msg, t in bag.read_messages(topics=[self.target_topic]):
                    message_count += 1
                    timestamp = t.to_sec()
                    
                    # Extract objects from message
                    for msg_obj in msg.objects:
                        extracted_obj = self.extract_object_from_message(msg_obj, timestamp)
                        if extracted_obj:
                            self.extracted_objects.append(extracted_obj)
                            # Add to temporal sequence
                            self.temporal_sequences[extracted_obj.id].append(extracted_obj)
                            objects_count += 1
                            
                print(f"  Messages processed: {message_count}")
                print(f"  Objects extracted: {objects_count}")
                
        except Exception as e:
            print(f"Error processing {bag_file.name}: {e}")
            
        return objects_count
        
    def extract_all_bags(self) -> Dict[str, Any]:
        """
        Extract tracking data from all bag files.
        
        Returns:
            Summary statistics of extraction
        """
        print("Starting extraction from all bag files...")
        
        total_objects = 0
        for bag_file in self.bag_files:
            total_objects += self.extract_from_single_bag(bag_file)
            
        # Build temporal sequence statistics
        sequence_stats = {}
        for obj_id, sequence in self.temporal_sequences.items():
            sequence_stats[obj_id] = {
                'detections': len(sequence),
                'duration': max(obj.timestamp for obj in sequence) - min(obj.timestamp for obj in sequence),
                'first_seen': min(obj.timestamp for obj in sequence),
                'last_seen': max(obj.timestamp for obj in sequence),
                'avg_tracked_period': np.mean([obj.tracked_period for obj in sequence])
            }
            
        summary = {
            'total_bag_files': len(self.bag_files),
            'total_objects_extracted': total_objects,
            'unique_object_ids': len(self.temporal_sequences),
            'temporal_sequences': sequence_stats
        }
        
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Bag files processed: {summary['total_bag_files']}")
        print(f"Total objects extracted: {summary['total_objects_extracted']}")
        print(f"Unique object IDs: {summary['unique_object_ids']}")
        
        # Show top tracked objects
        sorted_objects = sorted(sequence_stats.items(), 
                              key=lambda x: x[1]['detections'], reverse=True)
        print(f"\nTop tracked objects:")
        for obj_id, stats in sorted_objects[:10]:
            print(f"  ID {obj_id}: {stats['detections']} detections, "
                  f"{stats['duration']:.1f}s duration")
                  
        return summary
        
    def save_extracted_data(self, output_dir: str = "extracted_data") -> Dict[str, str]:
        """
        Save extracted data to files for further processing.
        
        Args:
            output_dir: Directory to save extracted data
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = self.bag_directory / output_dir
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save raw extracted objects
        raw_objects_file = output_path / "raw_objects.json"
        with open(raw_objects_file, 'w') as f:
            json.dump([obj.to_dict() for obj in self.extracted_objects], f, indent=2)
        saved_files['raw_objects'] = str(raw_objects_file)
        
        # Save temporal sequences
        sequences_file = output_path / "temporal_sequences.json"
        sequences_data = {}
        for obj_id, sequence in self.temporal_sequences.items():
            sequences_data[str(obj_id)] = [obj.to_dict() for obj in sequence]
            
        with open(sequences_file, 'w') as f:
            json.dump(sequences_data, f, indent=2)
        saved_files['temporal_sequences'] = str(sequences_file)
        
        # Save object statistics
        stats_file = output_path / "extraction_statistics.json"
        summary = self.extract_all_bags() if not hasattr(self, '_summary') else self._summary
        with open(stats_file, 'w') as f:
            json.dump(summary, f, indent=2)
        saved_files['statistics'] = str(stats_file)
        
        print(f"\nData saved to: {output_path}")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {Path(file_path).name}")
            
        return saved_files
        
    def get_object_trajectories(self) -> Dict[int, np.ndarray]:
        """
        Get position trajectories for each tracked object.
        
        Returns:
            Dictionary mapping object ID to trajectory array [N, 3] (x, y, z)
        """
        trajectories = {}
        
        for obj_id, sequence in self.temporal_sequences.items():
            # Sort by timestamp
            sorted_sequence = sorted(sequence, key=lambda obj: obj.timestamp)
            
            # Extract positions
            positions = np.array([
                [obj.position_x, obj.position_y, obj.position_z] 
                for obj in sorted_sequence
            ])
            
            trajectories[obj_id] = positions
            
        return trajectories
        
    def filter_by_quality(self, min_detections: int = 5, 
                         min_duration: float = 1.0) -> List[int]:
        """
        Filter object IDs by tracking quality criteria.
        
        Args:
            min_detections: Minimum number of detections required
            min_duration: Minimum tracking duration in seconds
            
        Returns:
            List of high-quality object IDs
        """
        quality_objects = []
        
        for obj_id, sequence in self.temporal_sequences.items():
            if len(sequence) >= min_detections:
                duration = (max(obj.timestamp for obj in sequence) - 
                           min(obj.timestamp for obj in sequence))
                if duration >= min_duration:
                    quality_objects.append(obj_id)
                    
        return quality_objects


def main():
    """Example usage of ITRIBagReader"""
    
    # Path to ITRI bag files
    bag_directory = "/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic"
    
    # Initialize reader
    reader = ITRIBagReader(bag_directory)
    
    # Process only first bag file for testing
    print("Processing first bag file only (for testing)...")
    first_bag = reader.bag_files[0]
    objects_extracted = reader.extract_from_single_bag(first_bag)
    
    # Save extracted data
    saved_files = reader.save_extracted_data()
    
    # Get high-quality trajectories
    quality_objects = reader.filter_by_quality(min_detections=5, min_duration=1.0)
    print(f"\nHigh-quality tracked objects: {quality_objects}")
    
    # Example: Get trajectories for analysis
    trajectories = reader.get_object_trajectories()
    print(f"Generated trajectories for {len(trajectories)} objects")
    
    # Show some statistics
    print(f"\nExtraction Results:")
    print(f"Objects extracted: {len(reader.extracted_objects)}")
    print(f"Unique object IDs: {len(reader.temporal_sequences)}")
    
    return reader


if __name__ == "__main__":
    main()