#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITRI/HCT Lane Ground Truth Extraction for UniAD

This module converts ITRI/HCT semantic map polylines to UniAD-compatible lane ground truth format:
- gt_lane_labels: Lane class labels per instance
- gt_lane_bboxes: Bounding boxes [x_min, y_min, x_max, y_max] 
- gt_lane_masks: Binary segmentation masks [H, W]

Author: Generated for UniAD Integration Project
"""

import json
import numpy as np
import torch
import cv2
import os
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

# Import existing utilities
import sys
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')
from coordinate_transform import transform_points_to_ego, filter_points_by_bev_range
from geometric_utils import sample_polyline_uniform


class LaneGroundTruthExtractor:
    """
    Extracts lane ground truth data from ITRI/HCT semantic maps for UniAD training.
    
    Converts polyline-based semantic maps to rasterized BEV masks, bounding boxes,
    and class labels compatible with UniAD's PansegformerHead.
    """
    
    def __init__(
        self,
        bev_size: Tuple[int, int] = (200, 200),
        pc_range: List[float] = [-51.2, -51.2, -10.0, 51.2, 51.2, 10.0],
        line_thickness: int = 2,
        sample_distance: float = 0.5
    ):
        """
        Initialize the ground truth extractor.
        
        Args:
            bev_size: BEV map dimensions (height, width) in pixels
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            line_thickness: Thickness of rasterized lines in pixels
            sample_distance: Distance between sampled points on polylines (meters)
        """
        self.bev_h, self.bev_w = bev_size
        self.pc_range = pc_range
        self.line_thickness = line_thickness
        self.sample_distance = sample_distance
        
        # Calculate BEV resolution
        self.bev_res_x = (pc_range[3] - pc_range[0]) / self.bev_w  # meters per pixel
        self.bev_res_y = (pc_range[4] - pc_range[1]) / self.bev_h
        
        # Lane type mapping from ITRI/HCT to UniAD classes
        self.type_mapping = {
            # ITRI format types
            7: 0,   # Divider lines -> divider class (0)
            # HCT_logistic format types
            6: 0,   # Type 6 -> divider class (0)  
            # Pedestrian crossings -> crossing class (1)
            'crossing': 1,
            # Road markers -> contour class (2)
            'marker': 2,
            # Default fallback
            'default': 0
        }
    
    def world_to_bev_coords(
        self, 
        world_points: List[Tuple[float, float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Convert world coordinates to BEV pixel coordinates.
        
        Args:
            world_points: List of world coordinate points [(x, y, z), ...]
            
        Returns:
            List of BEV pixel coordinates [(u, v), ...]
        """
        bev_coords = []
        
        for x, y, z in world_points:
            # Convert to BEV pixel coordinates
            u = int((x - self.pc_range[0]) / self.bev_res_x)
            v = int((y - self.pc_range[1]) / self.bev_res_y)
            
            # Clamp to BEV bounds
            u = max(0, min(u, self.bev_w - 1))
            v = max(0, min(v, self.bev_h - 1))
            
            bev_coords.append((u, v))
        
        return bev_coords
    
    def rasterize_polyline_to_bev(
        self, 
        ego_points: List[Tuple[float, float, float]]
    ) -> np.ndarray:
        """
        Convert 3D polyline to 2D BEV binary mask.
        
        Args:
            ego_points: List of points in ego coordinate system
            
        Returns:
            Binary mask [H, W] with rasterized polyline
        """
        if len(ego_points) < 2:
            return np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        
        # Convert to BEV pixel coordinates
        bev_coords = self.world_to_bev_coords(ego_points)
        
        # Create empty mask
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        
        # Draw line segments
        for i in range(len(bev_coords) - 1):
            pt1 = bev_coords[i]
            pt2 = bev_coords[i + 1]
            
            # Draw line segment with specified thickness
            cv2.line(mask, pt1, pt2, 1, self.line_thickness)
        
        return mask
    
    def extract_bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Extract bounding box from binary mask.
        
        Args:
            mask: Binary mask [H, W]
            
        Returns:
            Bounding box (x_min, y_min, x_max, y_max)
        """
        # Find non-zero pixels
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
        
        y_coords, x_coords = coords
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        return (x_min, y_min, x_max, y_max)
    
    def _detect_dataset_format(self, data_path: Path) -> str:
        """Detect whether this is ITRI or HCT_logistic format."""
        roadlines_file = data_path / "roadlines.json"
        if not roadlines_file.exists():
            return "unknown"
        
        try:
            with open(roadlines_file, 'r') as f:
                data = json.load(f)
            
            if data['roadlines'] and data['roadlines'][0]['points']:
                first_point = data['roadlines'][0]['points'][0]
                
                # HCT format has point_id field, ITRI doesn't
                if 'point_id' in first_point:
                    return "hct_logistic"
                else:
                    return "itri"
        except:
            pass
        
        return "unknown"
    
    def load_semantic_map_data(self, semantic_map_path: str) -> List[Dict]:
        """
        Load semantic map data from ITRI/HCT format files.
        
        Args:
            semantic_map_path: Path to semantic map data directory
            
        Returns:
            List of lane segments with points, type, and metadata
        """
        data_path = Path(semantic_map_path)
        lane_segments = []
        
        # Detect dataset format
        dataset_type = self._detect_dataset_format(data_path)
        
        # Load roadlines (primary lane boundaries)
        roadlines_file = data_path / "roadlines.json"
        if roadlines_file.exists():
            with open(roadlines_file, 'r') as f:
                roadlines_data = json.load(f)
            
            for roadline in roadlines_data['roadlines']:
                if dataset_type == "hct_logistic":
                    # HCT format: points have point_id, type, x, y, z
                    points = [(p['x'], p['y'], p['z']) for p in roadline['points']]
                    line_type = roadline['points'][0].get('type', 6)
                else:
                    # ITRI format: points have x, y, z
                    points = [(p['x'], p['y'], p.get('z', 0.0)) for p in roadline['points']]
                    line_type = roadline['points'][0].get('type', 7)
                
                lane_type = self.type_mapping.get(line_type, 0)
                
                lane_segments.append({
                    'id': f"roadline_{roadline['id']}",
                    'points': points,
                    'type': lane_type,
                    'source': 'roadline',
                    'dataset': dataset_type
                })
        
        # Load pedestrian crossings
        crossings_file = data_path / "pedestrian_crossing.json"
        if crossings_file.exists():
            with open(crossings_file, 'r') as f:
                crossings_data = json.load(f)
            
            for crossing in crossings_data.get('non_accessible', []):
                points = [(p['x'], p['y'], p['z']) for p in crossing['points']]
                
                lane_segments.append({
                    'id': f"crossing_{crossing['id']}",
                    'points': points,
                    'type': 1,  # Crossing type
                    'source': 'crossing'
                })
        
        # Load road markers
        markers_file = data_path / "roadmarkers.json"
        if markers_file.exists():
            with open(markers_file, 'r') as f:
                markers_data = json.load(f)
            
            for marker in markers_data.get('roadmarkers', []):
                points = [(p['x'], p['y'], p['z']) for p in marker['points']]
                
                lane_segments.append({
                    'id': f"marker_{marker['id']}",
                    'points': points,
                    'type': 2,  # Contour type
                    'source': 'marker'
                })
        
        return lane_segments
    
    def generate_lane_ground_truth(
        self, 
        semantic_map_path: str, 
        ego_pose: Tuple[float, float, float]
    ) -> Tuple[List[int], List[List[int]], List[torch.Tensor]]:
        """
        Generate UniAD-compatible lane ground truth from semantic map data.
        
        Args:
            semantic_map_path: Path to semantic map data directory  
            ego_pose: Current ego vehicle pose (x, y, yaw) in world coordinates
            
        Returns:
            Tuple of (gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
        """
        # Load semantic map data
        lane_segments = self.load_semantic_map_data(semantic_map_path)
        
        gt_lane_labels = []
        gt_lane_bboxes = []
        gt_lane_masks = []
        
        for segment in lane_segments:
            # Transform to ego coordinate system
            ego_points = transform_points_to_ego(segment['points'], ego_pose)
            
            # Filter by BEV range
            filtered_points = filter_points_by_bev_range(ego_points, self.pc_range)
            
            # Skip if insufficient points
            if len(filtered_points) < 2:
                continue
            
            # Sample points uniformly for better rasterization
            sampled_points = sample_polyline_uniform(filtered_points, self.sample_distance)
            
            # Rasterize to BEV mask
            mask = self.rasterize_polyline_to_bev(sampled_points)
            
            # Skip if mask is empty
            if mask.sum() == 0:
                continue
            
            # Extract bounding box
            bbox = self.extract_bbox_from_mask(mask)
            
            # Add to ground truth lists
            gt_lane_labels.append(segment['type'])
            gt_lane_bboxes.append(list(bbox))
            gt_lane_masks.append(torch.tensor(mask, dtype=torch.uint8))
        
        return gt_lane_labels, gt_lane_bboxes, gt_lane_masks
    
    def process_frame(
        self, 
        frame_path: str, 
        semantic_map_path: str
    ) -> Dict:
        """
        Process a single frame to generate ground truth data.
        
        Args:
            frame_path: Path to frame directory (e.g., frame_000000)
            semantic_map_path: Path to semantic map data
            
        Returns:
            Dictionary with ground truth data and metadata
        """
        frame_path = Path(frame_path)
        
        # Load frame metadata to get ego pose
        metadata_file = frame_path / "metadata.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Extract ego pose
        ego_pose_dict = metadata.get('ego_pose', {})
        ego_pose = (
            ego_pose_dict.get('x', 0.0),
            ego_pose_dict.get('y', 0.0), 
            ego_pose_dict.get('yaw', 0.0)
        )
        
        # Generate ground truth
        gt_lane_labels, gt_lane_bboxes, gt_lane_masks = self.generate_lane_ground_truth(
            semantic_map_path, ego_pose
        )
        
        return {
            'gt_lane_labels': gt_lane_labels,
            'gt_lane_bboxes': gt_lane_bboxes,
            'gt_lane_masks': gt_lane_masks,
            'frame_index': metadata.get('frame_index', 0),
            'num_lanes': len(gt_lane_labels),
            'ego_pose': ego_pose_dict,
            'bev_size': (self.bev_h, self.bev_w),
            'pc_range': self.pc_range
        }
    
    def process_all_frames(
        self, 
        hct_train_path: str, 
        semantic_map_path: str,
        output_path: str = None
    ) -> Dict[str, Dict]:
        """
        Process all frames in the HCT training dataset.
        
        Args:
            hct_train_path: Path to HCT training data directory
            semantic_map_path: Path to semantic map data
            output_path: Optional output path for saving results
            
        Returns:
            Dictionary mapping frame names to ground truth data
        """
        hct_train_path = Path(hct_train_path)
        results = {}
        
        # Find all frame directories
        frame_dirs = sorted(glob.glob(str(hct_train_path / "map_query" / "frame_*")))
        
        print(f"Processing {len(frame_dirs)} frames...")
        
        for i, frame_dir in enumerate(frame_dirs):
            frame_name = Path(frame_dir).name
            
            try:
                # Process frame
                result = self.process_frame(frame_dir, semantic_map_path)
                results[frame_name] = result
                
                # Save individual frame if output path specified
                if output_path:
                    output_dir = Path(output_path) / "gt_lane" / frame_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save ground truth data
                    torch.save({
                        'gt_lane_labels': result['gt_lane_labels'],
                        'gt_lane_bboxes': result['gt_lane_bboxes'], 
                        'gt_lane_masks': result['gt_lane_masks']
                    }, output_dir / "lane_gt.pt")
                    
                    # Save metadata
                    with open(output_dir / "metadata.json", 'w') as f:
                        metadata = {k: v for k, v in result.items() 
                                  if k not in ['gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks']}
                        json.dump(metadata, f, indent=2)
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(frame_dirs)} frames")
                    
            except Exception as e:
                print(f"Error processing frame {frame_name}: {e}")
                continue
        
        print(f"Successfully processed {len(results)} frames")
        return results


def validate_ground_truth(gt_data: Dict) -> Dict[str, any]:
    """
    Validate ground truth data quality and format.
    
    Args:
        gt_data: Ground truth data dictionary
        
    Returns:
        Validation report with statistics and issues
    """
    report = {
        'total_frames': len(gt_data),
        'total_lanes': 0,
        'lane_type_counts': defaultdict(int),
        'avg_lanes_per_frame': 0,
        'bbox_issues': 0,
        'empty_masks': 0,
        'large_masks': 0
    }
    
    for frame_name, frame_data in gt_data.items():
        num_lanes = frame_data.get('num_lanes', 0)
        report['total_lanes'] += num_lanes
        
        # Check lane types
        for label in frame_data.get('gt_lane_labels', []):
            report['lane_type_counts'][label] += 1
        
        # Validate bboxes
        for bbox in frame_data.get('gt_lane_bboxes', []):
            if len(bbox) != 4 or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                report['bbox_issues'] += 1
        
        # Check masks
        for mask in frame_data.get('gt_lane_masks', []):
            if mask.sum() == 0:
                report['empty_masks'] += 1
            elif mask.sum() > 10000:  # Large mask threshold
                report['large_masks'] += 1
    
    if report['total_frames'] > 0:
        report['avg_lanes_per_frame'] = report['total_lanes'] / report['total_frames']
    
    return report


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Extract lane ground truth for UniAD training')
    parser.add_argument('--hct_train_path', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train',
                       help='Path to HCT training data directory')
    parser.add_argument('--semantic_map_path', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/semantic-map/data/hct_logistic',
                       help='Path to semantic map data directory')  
    parser.add_argument('--output_path', type=str,
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/gt_lane',
                       help='Output path for saving results')
    parser.add_argument('--bev_size', type=int, nargs=2, default=[200, 200],
                       help='BEV map size [height, width]')
    parser.add_argument('--line_thickness', type=int, default=2,
                       help='Line thickness for rasterization')
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after processing')
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = LaneGroundTruthExtractor(
        bev_size=tuple(args.bev_size),
        line_thickness=args.line_thickness
    )
    
    # Process all frames
    results = extractor.process_all_frames(
        args.hct_train_path,
        args.semantic_map_path, 
        args.output_path
    )
    
    # Validation
    if args.validate:
        print("\nValidating results...")
        report = validate_ground_truth(results)
        
        print(f"\nValidation Report:")
        print(f"Total frames: {report['total_frames']}")
        print(f"Total lanes: {report['total_lanes']}")
        print(f"Average lanes per frame: {report['avg_lanes_per_frame']:.2f}")
        print(f"Lane type distribution: {dict(report['lane_type_counts'])}")
        print(f"Bbox issues: {report['bbox_issues']}")
        print(f"Empty masks: {report['empty_masks']}")
        print(f"Large masks: {report['large_masks']}")
    
    print("Lane ground truth extraction completed!")


if __name__ == "__main__":
    main()