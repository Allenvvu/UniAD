#!/usr/bin/env python3
"""
MotionFormer Integration Interface

This module provides the integration interface between ITRI semantic map data
and UniAD's MotionFormer, bypassing the MapFormer head while maintaining
compatibility with the existing motion prediction pipeline.

Author: Generated for UniAD Integration Project
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from itri_to_lane_query import ITRIToLaneQueryConverter, create_converter


class MotionFormerIntegrator:
    """
    Integration class for using ITRI semantic maps with MotionFormer.
    
    This class handles the conversion of ITRI data to the format expected by
    MotionFormer and provides a seamless interface for motion prediction.
    """
    
    def __init__(
        self,
        converter_config: Optional[Dict] = None,
        device: str = 'cuda'
    ):
        """
        Initialize the integrator.
        
        Args:
            converter_config: Configuration for the lane query converter
            device: Device to run computations on
        """
        self.device = device
        self.converter = create_converter(converter_config)
        self.converter.to(device)
        
        # Cache for processed map data
        self._map_cache = {}
        
    def prepare_motion_inputs(
        self,
        itri_data_path: str,
        ego_pose: Tuple[float, float, float],
        tracked_objects: List[Dict],
        bev_features: torch.Tensor,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare all inputs needed for MotionFormer inference.
        
        Args:
            itri_data_path: Path to ITRI semantic map data
            ego_pose: Current ego vehicle pose (x, y, yaw)
            tracked_objects: List of tracked object detections
            bev_features: BEV feature tensor from perception pipeline
            use_cache: Whether to cache and reuse map conversions
            
        Returns:
            Dictionary containing outs_track and outs_seg for MotionFormer
        """
        # Generate map queries
        outs_seg = self._prepare_map_queries(
            itri_data_path, ego_pose, use_cache)
        
        # Prepare tracking queries
        outs_track = self._prepare_tracking_queries(tracked_objects)
        
        return {
            'bev_embed': bev_features,
            'outs_track': outs_track,
            'outs_seg': outs_seg
        }
    
    def _prepare_map_queries(
        self,
        itri_data_path: str,
        ego_pose: Tuple[float, float, float],
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Convert ITRI map data to lane queries.
        
        Args:
            itri_data_path: Path to ITRI data
            ego_pose: Current ego pose
            use_cache: Whether to use cached results
            
        Returns:
            outs_seg dictionary with args_tuple for MotionFormer
        """
        cache_key = f"{itri_data_path}_{ego_pose}"
        
        if use_cache and cache_key in self._map_cache:
            lane_query, lane_query_pos = self._map_cache[cache_key]
        else:
            # Convert ITRI data to lane queries
            lane_query, lane_query_pos = self.converter.convert(
                itri_data_path, ego_pose)
            
            if use_cache:
                self._map_cache[cache_key] = (lane_query, lane_query_pos)
        
        # Create args_tuple in the format MotionFormer expects
        # Based on panseg_head.py line 241: args_tuple = [memory, memory_mask, memory_pos, query, None, query_pos, hw_lvl]
        bev_h, bev_w = 200, 200  # From UniAD config
        hw_lvl = [(bev_h, bev_w)]
        
        args_tuple = [
            None,          # memory (not used directly by MotionFormer)
            None,          # memory_mask  
            None,          # memory_pos
            lane_query,    # lane queries (what MotionFormer needs)
            None,          # unused slot
            lane_query_pos, # lane position encodings (what MotionFormer needs)
            hw_lvl         # spatial dimensions
        ]
        
        return {
            'args_tuple': args_tuple,
            'bev_embed': None,  # Not needed for motion prediction
            'outputs_classes': None,
            'outputs_coords': None,
            'enc_outputs_class': None,
            'enc_outputs_coord': None,
            'reference': None
        }
    
    def _prepare_tracking_queries(
        self, 
        tracked_objects: List[Dict]
    ) -> Dict[str, Any]:
        """
        Prepare tracking queries from detection results.
        
        Args:
            tracked_objects: List of tracked object detections
            
        Returns:
            outs_track dictionary for MotionFormer
        """
        if not tracked_objects:
            # Return empty tracking results
            return self._create_empty_tracking_results()
        
        # Extract tracking information
        track_embeddings = []
        track_bboxes = []
        track_scores = []
        track_labels = []
        track_indices = []
        
        for i, obj in enumerate(tracked_objects):
            # Extract object information
            bbox = obj.get('bbox', [0, 0, 0, 1, 1, 1, 0])  # [x, y, z, l, w, h, yaw]
            score = obj.get('score', 0.5)
            label = obj.get('label', 0)
            track_id = obj.get('track_id', i)
            
            # Create embedding (this would normally come from detection head)
            embedding = self._create_object_embedding(bbox, label)
            
            track_embeddings.append(embedding)
            track_bboxes.append(bbox)
            track_scores.append(score)
            track_labels.append(label)
            track_indices.append(track_id)
        
        # Convert to tensors
        track_embeddings = torch.stack(track_embeddings).to(self.device)
        track_bboxes_tensor = torch.tensor(track_bboxes, device=self.device)
        track_scores_tensor = torch.tensor(track_scores, device=self.device)
        track_labels_tensor = torch.tensor(track_labels, device=self.device)
        track_indices_tensor = torch.tensor(track_indices, device=self.device)
        
        # Create track bbox results in format expected by MotionFormer
        # Based on motion_head.py: track_bbox_results format
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        
        bbox_results = [
            LiDARInstance3DBoxes(track_bboxes_tensor),  # 3D boxes
            track_scores_tensor,                        # scores
            track_labels_tensor,                        # labels
            track_indices_tensor,                       # indices
            torch.ones_like(track_scores_tensor, dtype=torch.bool)  # valid mask
        ]
        
        # Add SDC (ego vehicle) information
        sdc_bbox = torch.tensor([[0, 0, 0, 4.5, 2.0, 1.8, 0]], device=self.device)  # Typical car dimensions
        sdc_embedding = self._create_sdc_embedding()
        
        return {
            'track_query_embeddings': track_embeddings,
            'track_query_matched_idxes': track_indices_tensor,
            'track_bbox_results': [bbox_results],
            'sdc_embedding': sdc_embedding,
            'sdc_track_bbox_results': [[
                LiDARInstance3DBoxes(sdc_bbox),
                torch.tensor([1.0], device=self.device),
                torch.tensor([0], device=self.device),  # Car class
                torch.tensor([999], device=self.device),  # SDC track ID
                torch.tensor([True], device=self.device)
            ]]
        }
    
    def _create_object_embedding(
        self, 
        bbox: List[float], 
        label: int
    ) -> torch.Tensor:
        """
        Create object embedding from bounding box and label.
        
        Args:
            bbox: Object bounding box [x, y, z, l, w, h, yaw]
            label: Object class label
            
        Returns:
            Object embedding tensor
        """
        # Simple geometric + semantic embedding
        # In practice, this would come from the detection head
        geometric_features = torch.tensor(bbox, device=self.device)
        semantic_features = torch.nn.functional.one_hot(
            torch.tensor(label, device=self.device), num_classes=10
        ).float()
        
        # Combine and project to embedding dimension
        combined = torch.cat([geometric_features, semantic_features])
        embedding = torch.nn.Linear(
            combined.shape[0], 256, device=self.device
        )(combined)
        
        return embedding
    
    def _create_sdc_embedding(self) -> torch.Tensor:
        """Create embedding for SDC (ego vehicle)."""
        # SDC has special embedding
        sdc_embedding = torch.zeros(256, device=self.device)
        sdc_embedding[0] = 1.0  # Special marker for SDC
        return sdc_embedding
    
    def _create_empty_tracking_results(self) -> Dict[str, Any]:
        """Create empty tracking results when no objects are detected."""
        empty_embeddings = torch.zeros(1, 256, device=self.device)
        empty_bbox = torch.zeros(1, 7, device=self.device)
        empty_scores = torch.zeros(1, device=self.device)
        empty_labels = torch.zeros(1, dtype=torch.long, device=self.device)
        empty_indices = torch.zeros(1, dtype=torch.long, device=self.device)
        
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        
        bbox_results = [
            LiDARInstance3DBoxes(empty_bbox),
            empty_scores,
            empty_labels,
            empty_indices,
            torch.zeros(1, dtype=torch.bool, device=self.device)
        ]
        
        sdc_bbox = torch.tensor([[0, 0, 0, 4.5, 2.0, 1.8, 0]], device=self.device)
        sdc_embedding = self._create_sdc_embedding()
        
        return {
            'track_query_embeddings': empty_embeddings,
            'track_query_matched_idxes': empty_indices,
            'track_bbox_results': [bbox_results],
            'sdc_embedding': sdc_embedding,
            'sdc_track_bbox_results': [[
                LiDARInstance3DBoxes(sdc_bbox),
                torch.tensor([1.0], device=self.device),
                torch.tensor([0], device=self.device),
                torch.tensor([999], device=self.device),
                torch.tensor([True], device=self.device)
            ]]
        }
    
    def run_motion_prediction(
        self,
        motion_head,
        itri_data_path: str,
        ego_pose: Tuple[float, float, float],
        tracked_objects: List[Dict],
        bev_features: torch.Tensor
    ) -> Tuple[Any, Dict]:
        """
        Run complete motion prediction pipeline.
        
        Args:
            motion_head: MotionFormer head module
            itri_data_path: Path to ITRI semantic map data
            ego_pose: Current ego vehicle pose
            tracked_objects: List of tracked object detections
            bev_features: BEV feature tensor
            
        Returns:
            Tuple of (trajectory_results, motion_outputs)
        """
        # Prepare inputs
        inputs = self.prepare_motion_inputs(
            itri_data_path, ego_pose, tracked_objects, bev_features)
        
        # Run MotionFormer inference
        with torch.no_grad():
            traj_results, outs_motion = motion_head.forward_test(
                bev_embed=inputs['bev_embed'],
                outs_track=inputs['outs_track'], 
                outs_seg=inputs['outs_seg']
            )
        
        return traj_results, outs_motion
    
    def clear_cache(self):
        """Clear the map conversion cache."""
        self._map_cache.clear()
    
    def update_converter_config(self, config: Dict):
        """Update converter configuration."""
        self.converter = create_converter(config)
        self.converter.to(self.device)
        self.clear_cache()


def create_integrator(
    converter_config: Optional[Dict] = None,
    device: str = 'cuda'
) -> MotionFormerIntegrator:
    """
    Factory function to create MotionFormer integrator.
    
    Args:
        converter_config: Configuration for the converter
        device: Device to run on
        
    Returns:
        Configured MotionFormerIntegrator instance
    """
    return MotionFormerIntegrator(converter_config, device)


def validate_inputs(
    itri_data_path: str,
    ego_pose: Tuple[float, float, float],
    tracked_objects: List[Dict]
) -> bool:
    """
    Validate inputs for motion prediction.
    
    Args:
        itri_data_path: Path to ITRI data
        ego_pose: Ego vehicle pose
        tracked_objects: Tracked objects list
        
    Returns:
        True if inputs are valid
    """
    # Check ITRI data path
    data_path = Path(itri_data_path)
    if not data_path.exists():
        print(f"ITRI data path does not exist: {itri_data_path}")
        return False
    
    # Check required files
    required_files = ['roadlines.json']
    for file in required_files:
        if not (data_path / file).exists():
            print(f"Required file missing: {file}")
            return False
    
    # Validate ego pose
    if len(ego_pose) != 3:
        print(f"Invalid ego pose format: {ego_pose}")
        return False
    
    # Validate tracked objects format
    for i, obj in enumerate(tracked_objects):
        if not isinstance(obj, dict):
            print(f"Invalid object format at index {i}")
            return False
        
        if 'bbox' in obj and len(obj['bbox']) != 7:
            print(f"Invalid bbox format at index {i}")
            return False
    
    return True