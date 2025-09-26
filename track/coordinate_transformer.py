#!/usr/bin/env python3
"""
Coordinate Transformation Module for ITRI to UniAD Conversion

This module converts ITRI tracking data from base_link coordinates to 
UniAD's BEV (Bird's Eye View) coordinate system, following the plan 
outlined in data_extraction_plan.md and trackq_plan.md.

Key transformations:
1. base_link → BEV grid coordinates  
2. Quaternion → Yaw angle conversion
3. ITRI labels → UniAD class mapping
4. Confidence scoring using tracking duration proxy
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import sys
import os

# Try to import ROS tf transformations, fallback if not available
try:
    import tf.transformations
    TF_AVAILABLE = True
except ImportError:
    print("TF transformations not available, using fallback implementation")
    TF_AVAILABLE = False


@dataclass
class UniADTrackingObject:
    """Transformed tracking object in UniAD format"""
    id: int
    label: str
    class_id: int
    confidence: float
    timestamp: float
    
    # BEV coordinates
    bev_x: float
    bev_y: float
    bev_z: float
    
    # Orientation (yaw angle in radians)
    yaw: float
    
    # Velocity in BEV frame
    vx: float
    vy: float
    vz: float
    
    # Object dimensions (L, W, H)
    length: float
    width: float
    height: float
    
    # Angular velocity
    angular_velocity: float
    
    # Original tracking data for reference
    original_tracked_period: float
    original_score: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
        
    def get_bbox_9dof(self) -> Tuple[float, ...]:
        """Get 9DOF bounding box format: [x, y, z, w, l, h, yaw, vx, vy]"""
        return (self.bev_x, self.bev_y, self.bev_z, 
                self.width, self.length, self.height,
                self.yaw, self.vx, self.vy)


class ITRICoordinateTransformer:
    """
    Coordinate transformer for ITRI tracking data to UniAD format.
    
    Based on trackq_plan.md implementation plan and using existing 
    utilities from find_yaw.py and coordinate frameworks.
    """
    
    def __init__(self):
        """Initialize coordinate transformer with UniAD standards"""
        
        # UniAD standard point cloud range
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        
        # BEV grid size (standard for UniAD)
        self.bev_size = (200, 200)
        
        # ITRI to UniAD class mapping (from trackq_plan.md)
        self.class_mapping = {
            "car": 0,
            "motorbike": 1,
            "pedestrian": 2, 
            "cyclist": 3,
            "truck": 4,
            "bus": 5,
            "unknown": 9,
            "": 9,  # Handle empty labels as unknown
        }
        
        # Tracking duration confidence scaling (from trackq_plan.md)
        self.confidence_scaling_duration = 15.0  # Max duration for confidence = 1.0
        self.min_confidence = 0.1
        
        print(f"Initialized coordinate transformer:")
        print(f"  PC Range: {self.pc_range}")
        print(f"  BEV Size: {self.bev_size}")
        print(f"  Class Mapping: {len(self.class_mapping)} classes")
        
    def quaternion_to_yaw(self, qx: float, qy: float, qz: float, qw: float) -> float:
        """
        Convert quaternion to yaw angle using existing find_yaw.py approach
        
        Args:
            qx, qy, qz, qw: Quaternion components
            
        Returns:
            Yaw angle in radians
        """
        if TF_AVAILABLE:
            # Use ROS tf transformations (same as find_yaw.py)
            quaternion = (qx, qy, qz, qw)
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
            return yaw
        else:
            # Fallback implementation
            # Convert quaternion to yaw using direct formula
            t3 = +2.0 * (qw * qz + qx * qy)
            t4 = +1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = np.arctan2(t3, t4)
            return yaw
            
    def base_link_to_bev(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Convert position from base_link frame to BEV coordinates
        
        Args:
            x, y, z: Position in base_link frame (meters)
            
        Returns:
            Tuple of (bev_x, bev_y, bev_z) coordinates
        """
        # Convert to BEV grid coordinates
        # X: forward in base_link → Y in BEV grid
        # Y: left in base_link → X in BEV grid (with coordinate flip)
        bev_x = (y - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * self.bev_size[0]
        bev_y = (x - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * self.bev_size[1]
        
        # Z coordinate (height) - relative to ground plane
        bev_z = z
        
        return bev_x, bev_y, bev_z
        
    def calculate_confidence_score(self, original_score: float, tracked_period: float) -> float:
        """
        Calculate confidence score using tracking duration proxy from trackq_plan.md
        
        Args:
            original_score: Original detection score from ITRI
            tracked_period: Tracking duration in seconds
            
        Returns:
            Confidence score between 0.1 and 1.0
        """
        if original_score > 0:
            # Use original score if available and valid
            return min(1.0, original_score)
        else:
            # Apply tracking duration proxy for score 0.0 objects
            duration_confidence = tracked_period / self.confidence_scaling_duration
            return max(self.min_confidence, min(1.0, duration_confidence))
            
    def map_class_label(self, label: str) -> Tuple[str, int]:
        """
        Map ITRI class label to UniAD class ID
        
        Args:
            label: ITRI class label string
            
        Returns:
            Tuple of (normalized_label, class_id)
        """
        # Normalize label
        normalized_label = label.lower().strip()
        
        # Map to class ID
        if normalized_label in self.class_mapping:
            class_id = self.class_mapping[normalized_label]
            return normalized_label if normalized_label else "unknown", class_id
        else:
            # Unknown class
            return "unknown", self.class_mapping["unknown"]
            
    def transform_velocity(self, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
        """
        Transform velocity from base_link to BEV coordinate system
        
        Args:
            vx, vy, vz: Velocity in base_link frame (m/s)
            
        Returns:
            Velocity in BEV coordinate system
        """
        # Same coordinate transformation as position
        # X: forward in base_link → Y in BEV
        # Y: left in base_link → X in BEV  
        bev_vx = vy  # Left velocity becomes BEV X velocity
        bev_vy = vx  # Forward velocity becomes BEV Y velocity
        bev_vz = vz  # Up velocity remains the same
        
        return bev_vx, bev_vy, bev_vz
        
    def transform_single_object(self, obj_data: Dict) -> Optional[UniADTrackingObject]:
        """
        Transform a single ITRI tracking object to UniAD format
        
        Args:
            obj_data: Dictionary containing ITRI object data
            
        Returns:
            UniADTrackingObject or None if transformation fails
        """
        try:
            # Extract position and convert coordinates
            bev_x, bev_y, bev_z = self.base_link_to_bev(
                obj_data['position_x'],
                obj_data['position_y'], 
                obj_data['position_z']
            )
            
            # Convert quaternion to yaw
            yaw = self.quaternion_to_yaw(
                obj_data['orientation_x'],
                obj_data['orientation_y'],
                obj_data['orientation_z'],
                obj_data['orientation_w']
            )
            
            # Transform velocity
            bev_vx, bev_vy, bev_vz = self.transform_velocity(
                obj_data['velocity_x'],
                obj_data['velocity_y'],
                obj_data['velocity_z']
            )
            
            # Map class label
            normalized_label, class_id = self.map_class_label(obj_data['label'])
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(
                obj_data['score'],
                obj_data['tracked_period']
            )
            
            # Create transformed object
            transformed_obj = UniADTrackingObject(
                id=obj_data['id'],
                label=normalized_label,
                class_id=class_id,
                confidence=confidence,
                timestamp=obj_data['timestamp'],
                
                # BEV coordinates
                bev_x=bev_x,
                bev_y=bev_y,
                bev_z=bev_z,
                
                # Orientation
                yaw=yaw,
                
                # Velocity
                vx=bev_vx,
                vy=bev_vy,
                vz=bev_vz,
                
                # Dimensions
                length=obj_data['dimension_x'],
                width=obj_data['dimension_y'],
                height=obj_data['dimension_z'],
                
                # Angular velocity
                angular_velocity=obj_data['angular_velocity_z'],
                
                # Original data for reference
                original_tracked_period=obj_data['tracked_period'],
                original_score=obj_data['score']
            )
            
            return transformed_obj
            
        except (KeyError, ValueError, TypeError) as e:
            print(f"Error transforming object {obj_data.get('id', 'unknown')}: {e}")
            return None
            
    def transform_objects_batch(self, objects_data: List[Dict]) -> List[UniADTrackingObject]:
        """
        Transform a batch of ITRI objects to UniAD format
        
        Args:
            objects_data: List of ITRI object dictionaries
            
        Returns:
            List of successfully transformed UniADTrackingObjects
        """
        transformed_objects = []
        failed_count = 0
        
        print(f"Transforming {len(objects_data)} objects...")
        
        for i, obj_data in enumerate(objects_data):
            transformed_obj = self.transform_single_object(obj_data)
            
            if transformed_obj:
                transformed_objects.append(transformed_obj)
            else:
                failed_count += 1
                
            # Progress reporting
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(objects_data)} objects")
                
        print(f"Transformation complete:")
        print(f"  Successfully transformed: {len(transformed_objects)} objects")
        print(f"  Failed transformations: {failed_count} objects")
        
        return transformed_objects
        
    def validate_transformation(self, transformed_objects: List[UniADTrackingObject]) -> Dict[str, Any]:
        """
        Validate the transformation results and provide statistics
        
        Args:
            transformed_objects: List of transformed objects
            
        Returns:
            Dictionary containing validation statistics
        """
        if not transformed_objects:
            return {"error": "No objects to validate"}
            
        # Collect statistics
        bev_x_coords = [obj.bev_x for obj in transformed_objects]
        bev_y_coords = [obj.bev_y for obj in transformed_objects]
        yaw_angles = [obj.yaw for obj in transformed_objects]
        confidence_scores = [obj.confidence for obj in transformed_objects]
        
        # Class distribution
        class_counts = {}
        for obj in transformed_objects:
            class_counts[obj.label] = class_counts.get(obj.label, 0) + 1
            
        # Confidence score distribution
        high_conf_count = sum(1 for conf in confidence_scores if conf > 0.5)
        zero_score_objects = sum(1 for obj in transformed_objects if obj.original_score == 0.0)
        
        validation_stats = {
            "total_objects": len(transformed_objects),
            "coordinate_ranges": {
                "bev_x": {"min": min(bev_x_coords), "max": max(bev_x_coords), "mean": np.mean(bev_x_coords)},
                "bev_y": {"min": min(bev_y_coords), "max": max(bev_y_coords), "mean": np.mean(bev_y_coords)},
            },
            "yaw_range": {
                "min": min(yaw_angles), "max": max(yaw_angles), "mean": np.mean(yaw_angles)
            },
            "confidence_stats": {
                "min": min(confidence_scores), "max": max(confidence_scores), "mean": np.mean(confidence_scores),
                "high_confidence_count": high_conf_count,
                "zero_score_converted": zero_score_objects
            },
            "class_distribution": class_counts,
            "unique_object_ids": len(set(obj.id for obj in transformed_objects))
        }
        
        return validation_stats
        
    def save_transformed_data(self, transformed_objects: List[UniADTrackingObject], 
                            output_dir: str = "transformed_data") -> Dict[str, str]:
        """
        Save transformed data to files
        
        Args:
            transformed_objects: List of transformed objects
            output_dir: Directory to save transformed data
            
        Returns:
            Dictionary of saved file paths
        """
        # Create output directory in track folder
        track_dir = Path(__file__).parent
        output_path = track_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save transformed objects
        transformed_file = output_path / "uniad_format_objects.json"
        with open(transformed_file, 'w') as f:
            json.dump([obj.to_dict() for obj in transformed_objects], f, indent=2)
        saved_files['transformed_objects'] = str(transformed_file)
        
        # Save validation statistics
        validation_stats = self.validate_transformation(transformed_objects)
        stats_file = output_path / "transformation_validation.json"
        with open(stats_file, 'w') as f:
            json.dump(validation_stats, f, indent=2)
        saved_files['validation_stats'] = str(stats_file)
        
        # Group by temporal sequences for MotionFormer
        temporal_sequences = {}
        for obj in transformed_objects:
            if obj.id not in temporal_sequences:
                temporal_sequences[obj.id] = []
            temporal_sequences[obj.id].append(obj.to_dict())
            
        # Sort each sequence by timestamp
        for obj_id in temporal_sequences:
            temporal_sequences[obj_id].sort(key=lambda x: x['timestamp'])
            
        sequences_file = output_path / "uniad_temporal_sequences.json"
        with open(sequences_file, 'w') as f:
            json.dump(temporal_sequences, f, indent=2)
        saved_files['temporal_sequences'] = str(sequences_file)
        
        print(f"\nTransformed data saved to: {output_path}")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {Path(file_path).name}")
            
        return saved_files


def main():
    """Example usage of ITRICoordinateTransformer"""
    
    # Initialize transformer
    transformer = ITRICoordinateTransformer()
    
    # Load extracted ITRI data
    itri_data_path = Path("/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic/extracted_data")
    raw_objects_file = itri_data_path / "raw_objects.json"
    
    if not raw_objects_file.exists():
        print(f"Error: Raw objects file not found at {raw_objects_file}")
        return
        
    print(f"Loading ITRI data from: {raw_objects_file}")
    with open(raw_objects_file, 'r') as f:
        raw_objects = json.load(f)
        
    print(f"Loaded {len(raw_objects)} raw objects")
    
    # Transform objects to UniAD format
    transformed_objects = transformer.transform_objects_batch(raw_objects)
    
    # Validate transformation
    validation_stats = transformer.validate_transformation(transformed_objects)
    print(f"\nValidation Results:")
    print(f"  Total objects: {validation_stats['total_objects']}")
    print(f"  Class distribution: {validation_stats['class_distribution']}")
    print(f"  Confidence stats: {validation_stats['confidence_stats']}")
    print(f"  BEV coordinate ranges:")
    print(f"    X: {validation_stats['coordinate_ranges']['bev_x']}")
    print(f"    Y: {validation_stats['coordinate_ranges']['bev_y']}")
    
    # Save transformed data
    saved_files = transformer.save_transformed_data(transformed_objects)
    
    return transformer, transformed_objects


if __name__ == "__main__":
    main()