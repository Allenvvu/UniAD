#!/usr/bin/env python3
"""
ITRI Semantic Map to UniAD Lane Query Converter

This module converts ITRI semantic map polylines to the lane_query and lane_query_pos 
format required by MotionFormer in UniAD.

Author: Generated for UniAD Integration Project
"""

import json
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from geometric_utils import (
    sample_polyline_uniform,
    compute_direction_features, 
    compute_curvature_features,
    points_to_relative_coords
)
from coordinate_transform import (
    transform_points_to_ego,
    filter_points_by_bev_range,
    normalize_coordinates
)


class ITRIToLaneQueryConverter:
    """
    Converter class for transforming ITRI semantic map data to UniAD lane queries.
    
    This converter processes ITRI polyline data and generates the lane_query and 
    lane_query_pos tensors that MotionFormer expects.
    """
    
    def __init__(
        self, 
        embed_dim: int = 256,
        max_queries: int = 300,
        sample_distance: float = 1.0,
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        """
        Initialize the converter.
        
        Args:
            embed_dim: Embedding dimension for lane queries (default: 256)
            max_queries: Maximum number of lane queries (default: 300) 
            sample_distance: Distance between sampled points on polylines (meters)
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.embed_dim = embed_dim
        self.max_queries = max_queries
        self.sample_distance = sample_distance
        self.pc_range = pc_range
        
        # Initialize embedding layers
        self._init_embedding_layers()
        
        # Line type to UniAD class mapping for both datasets
        self.type_mapping = {
            # ITRI format types
            7: 0,   # Divider lines → divider class (0)
            # HTC_logistic format types  
            6: 0,   # Type 6 → divider class (0)
            # Default behavior: unknown types → divider class
        }
    
    def _init_embedding_layers(self):
        """Initialize neural network layers for feature embedding."""
        # Geometric feature embedding 
        self.geometric_embedder = nn.Sequential(
            nn.Linear(6, 128),  # [x, y, direction_x, direction_y, curvature, length]
            nn.ReLU(),
            nn.Linear(128, self.embed_dim)
        )
        
        # Semantic type embedding
        self.type_embedder = nn.Embedding(3, 64)  # 3 lane types
        
        # Position embedding
        self.position_embedder = nn.Sequential(
            nn.Linear(3, 64),  # [x, y, z]
            nn.ReLU(),
            nn.Linear(64, self.embed_dim)
        )
        
        # Final projection layer
        self.final_projector = nn.Sequential(
            nn.Linear(self.embed_dim + 64, self.embed_dim),
            nn.LayerNorm(self.embed_dim)
        )
    
    def to(self, device):
        """Move all neural network components to the specified device."""
        self.geometric_embedder = self.geometric_embedder.to(device)
        self.type_embedder = self.type_embedder.to(device) 
        self.position_embedder = self.position_embedder.to(device)
        self.final_projector = self.final_projector.to(device)
        return self
    
    def convert(
        self, 
        itri_data_path: str, 
        ego_pose: Tuple[float, float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert ITRI semantic map to lane queries.
        
        Args:
            itri_data_path: Path to ITRI data folder
            ego_pose: Current ego vehicle pose (x, y, yaw) in world coordinates
            
        Returns:
            lane_query: [1, max_queries, embed_dim] - Lane feature queries
            lane_query_pos: [1, max_queries, embed_dim] - Lane positional encodings
        """
        # Load ITRI data
        lane_segments = self._load_itri_data(itri_data_path)
        
        # Transform to ego coordinate system
        ego_segments = self._transform_to_ego(lane_segments, ego_pose)
        
        # Filter by BEV range
        valid_segments = self._filter_by_bev_range(ego_segments)
        
        # Generate lane queries
        lane_queries, lane_positions = self._generate_lane_queries(valid_segments)
        
        # Pad to fixed size
        lane_query = self._pad_queries(lane_queries, self.max_queries)
        lane_query_pos = self._pad_queries(lane_positions, self.max_queries)
        
        # Add batch dimension and move to device
        device = next(self.geometric_embedder.parameters()).device
        lane_query_batched = lane_query[None, :, :].to(device)
        lane_query_pos_batched = lane_query_pos[None, :, :].to(device)
        return lane_query_batched, lane_query_pos_batched
    
    def convert_for_frame(
        self, 
        itri_data_path: str, 
        ego_pose: Tuple[float, float, float],
        frame_index: int
    ) -> Dict[str, any]:
        """
        Convert ITRI semantic map to lane queries for a specific frame.
        
        Args:
            itri_data_path: Path to ITRI data folder
            ego_pose: Current ego vehicle pose (x, y, yaw) in world coordinates
            frame_index: Frame index for metadata
            
        Returns:
            Dictionary containing lane_query, lane_query_pos, and metadata
        """
        # Generate lane queries
        lane_query, lane_query_pos = self.convert(itri_data_path, ego_pose)
        
        # Count valid lanes (non-zero queries)
        num_lanes = torch.sum(torch.any(lane_query.squeeze(0) != 0, dim=1)).item()
        
        # Create metadata
        metadata = {
            "frame_index": frame_index,
            "num_lanes": num_lanes,
            "embedding_shape": list(lane_query.shape),
            "device": str(lane_query.device),
            "ego_pose": {
                "x": ego_pose[0],
                "y": ego_pose[1], 
                "yaw": ego_pose[2]
            }
        }
        
        return {
            "lane_query": lane_query.cpu(),
            "lane_query_pos": lane_query_pos.cpu(), 
            "metadata": metadata
        }
    
    def _detect_dataset_format(self, data_path: Path) -> str:
        """Detect whether this is ITRI or HTC_logistic format by examining data structure."""
        roadlines_file = data_path / "roadlines.json"
        if not roadlines_file.exists():
            return "unknown"
        
        try:
            with open(roadlines_file, 'r') as f:
                data = json.load(f)
            
            if data['roadlines'] and data['roadlines'][0]['points']:
                first_point = data['roadlines'][0]['points'][0]
                
                # HTC format has point_id field, ITRI doesn't
                if 'point_id' in first_point:
                    return "hct_logistic"
                else:
                    return "itri"
        except:
            pass
        
        return "unknown"
    
    def _load_itri_data(self, data_path: str) -> List[Dict]:
        """Load and parse semantic map data (supports both ITRI and HTC_logistic formats)."""
        data_path = Path(data_path)
        lane_segments = []
        
        # Detect dataset format by checking first roadline structure
        dataset_type = self._detect_dataset_format(data_path)
        
        # Load roadlines (primary lane boundaries)
        roadlines_file = data_path / "roadlines.json"
        if roadlines_file.exists():
            with open(roadlines_file, 'r') as f:
                roadlines_data = json.load(f)
            
            for roadline in roadlines_data['roadlines']:
                if dataset_type == "hct_logistic":
                    # HTC format: points have point_id, type, x, y, z structure
                    points = [(p['x'], p['y'], p['z']) for p in roadline['points']]
                    line_type = roadline['points'][0].get('type', 6)
                else:
                    # ITRI format: points have x, y, z structure
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
        
        # Load road markers (additional lane features)
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
    
    def _transform_to_ego(
        self, 
        segments: List[Dict], 
        ego_pose: Tuple[float, float, float]
    ) -> List[Dict]:
        """Transform lane segments to ego vehicle coordinate system."""
        ego_segments = []
        
        for segment in segments:
            ego_points = transform_points_to_ego(segment['points'], ego_pose)
            ego_segments.append({
                **segment,
                'points': ego_points
            })
        
        return ego_segments
    
    def _filter_by_bev_range(self, segments: List[Dict]) -> List[Dict]:
        """Filter segments to only include those within BEV range."""
        valid_segments = []
        
        for segment in segments:
            filtered_points = filter_points_by_bev_range(
                segment['points'], self.pc_range)
            
            # Keep segment if at least 2 points remain
            if len(filtered_points) >= 2:
                valid_segments.append({
                    **segment,
                    'points': filtered_points
                })
        
        return valid_segments
    
    def _generate_lane_queries(
        self, 
        segments: List[Dict]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Generate lane feature queries and positional encodings."""
        lane_queries = []
        lane_positions = []
        
        for segment in segments:
            # Sample points uniformly along polyline
            sampled_points = sample_polyline_uniform(
                segment['points'], self.sample_distance)
            
            if len(sampled_points) < 2:
                continue
                
            # Create feature embedding
            lane_feature = self._create_lane_embedding(sampled_points, segment['type'])
            position_embed = self._create_position_embedding(sampled_points)
            
            lane_queries.append(lane_feature)
            lane_positions.append(position_embed)
        
        return lane_queries, lane_positions
    
    def _create_lane_embedding(
        self, 
        points: List[Tuple[float, float, float]], 
        lane_type: int
    ) -> torch.Tensor:
        """Create lane feature embedding from sampled points."""
        # Convert to relative coordinates
        relative_coords = points_to_relative_coords(points)
        
        # Compute geometric features
        direction_features = compute_direction_features(points)
        curvature_features = compute_curvature_features(points)
        
        # Aggregate features (mean pooling over points)
        geometric_features = np.array([
            np.mean([p[0] for p in relative_coords]),  # mean x
            np.mean([p[1] for p in relative_coords]),  # mean y  
            np.mean(direction_features[0]),            # mean direction_x
            np.mean(direction_features[1]),            # mean direction_y
            np.mean(curvature_features),               # mean curvature
            len(points) * self.sample_distance         # total length
        ])
        
        # Convert to tensors
        geo_tensor = torch.FloatTensor(geometric_features)
        type_tensor = torch.LongTensor([lane_type])
        
        # Generate embeddings
        geo_embed = self.geometric_embedder(geo_tensor)
        type_embed = self.type_embedder(type_tensor).squeeze(0)
        
        # Combine embeddings
        combined = torch.cat([geo_embed, type_embed], dim=0)
        lane_feature = self.final_projector(combined)
        
        return lane_feature
    
    def _create_position_embedding(
        self, 
        points: List[Tuple[float, float, float]]
    ) -> torch.Tensor:
        """Create positional encoding for lane segment."""
        # Use centroid as representative position
        centroid = np.mean(points, axis=0)
        
        # Normalize coordinates to BEV range
        normalized_pos = normalize_coordinates(centroid, self.pc_range)
        
        # Generate position embedding
        pos_tensor = torch.FloatTensor(normalized_pos)
        position_embed = self.position_embedder(pos_tensor)
        
        return position_embed
    
    def _pad_queries(
        self, 
        queries: List[torch.Tensor], 
        max_length: int
    ) -> torch.Tensor:
        """Pad query list to fixed length."""
        if len(queries) == 0:
            return torch.zeros(max_length, self.embed_dim)
        
        # Stack existing queries
        stacked = torch.stack(queries)
        
        # Truncate if too many
        if len(queries) > max_length:
            return stacked[:max_length]
        
        # Pad if too few
        if len(queries) < max_length:
            padding = torch.zeros(max_length - len(queries), self.embed_dim)
            return torch.cat([stacked, padding], dim=0)
        
        return stacked


def create_converter(config: Optional[Dict] = None) -> ITRIToLaneQueryConverter:
    """
    Factory function to create converter with configuration.
    
    Args:
        config: Configuration dictionary with converter parameters
        
    Returns:
        Configured ITRIToLaneQueryConverter instance
    """
    if config is None:
        config = {}
    
    return ITRIToLaneQueryConverter(
        embed_dim=config.get('embed_dim', 256),
        max_queries=config.get('max_queries', 300),
        sample_distance=config.get('sample_distance', 1.0),
        pc_range=config.get('pc_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
    )