#!/usr/bin/env python3
"""
ITRI Data Format Adapter

Handles data format conversions between ITRI data structures and UniAD expected formats.
Provides utilities for:
- Coordinate system transformations
- Tensor format conversions
- Metadata preparation
- Device management

Author: Generated for UniAD Integration Project
"""

import numpy as np
import torch
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinateSystemAdapter:
    """
    Handles coordinate system transformations between different reference frames.
    """
    
    def __init__(self, pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        """
        Initialize coordinate system adapter.
        
        Args:
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.pc_range = pc_range
        self.bev_size = (200, 200)  # Default BEV grid size
        
    def world_to_bev(self, world_coords: np.ndarray) -> np.ndarray:
        """
        Convert world coordinates to BEV grid coordinates.
        
        Args:
            world_coords: World coordinates [N, 2] or [N, 3]
            
        Returns:
            BEV grid coordinates [N, 2]
        """
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        # Extract x, y coordinates
        x_coords = world_coords[:, 0] if world_coords.ndim == 2 else world_coords[0]
        y_coords = world_coords[:, 1] if world_coords.ndim == 2 else world_coords[1]
        
        # Normalize to [0, 1] range
        x_norm = (x_coords - x_min) / (x_max - x_min)
        y_norm = (y_coords - y_min) / (y_max - y_min)
        
        # Convert to BEV grid coordinates
        bev_x = x_norm * self.bev_size[1]  # Width
        bev_y = y_norm * self.bev_size[0]  # Height
        
        # Clamp to valid range
        bev_x = np.clip(bev_x, 0, self.bev_size[1] - 1)
        bev_y = np.clip(bev_y, 0, self.bev_size[0] - 1)
        
        return np.stack([bev_x, bev_y], axis=-1)
    
    def bev_to_world(self, bev_coords: np.ndarray) -> np.ndarray:
        """
        Convert BEV grid coordinates to world coordinates.
        
        Args:
            bev_coords: BEV coordinates [N, 2]
            
        Returns:
            World coordinates [N, 2]
        """
        x_min, y_min = self.pc_range[0], self.pc_range[1]
        x_max, y_max = self.pc_range[3], self.pc_range[4]
        
        # Normalize BEV coordinates to [0, 1]
        x_norm = bev_coords[:, 0] / self.bev_size[1]
        y_norm = bev_coords[:, 1] / self.bev_size[0]
        
        # Convert to world coordinates
        world_x = x_norm * (x_max - x_min) + x_min
        world_y = y_norm * (y_max - y_min) + y_min
        
        return np.stack([world_x, world_y], axis=-1)
    
    def transform_ego_pose(
        self,
        ego_pose: Tuple[float, float, float],
        target_frame: str = 'bev'
    ) -> Tuple[float, float, float]:
        """
        Transform ego pose to target coordinate frame.
        
        Args:
            ego_pose: Ego pose (x, y, yaw) in world coordinates
            target_frame: Target coordinate frame ('bev' or 'world')
            
        Returns:
            Transformed pose (x, y, yaw)
        """
        x, y, yaw = ego_pose
        
        if target_frame == 'bev':
            # Convert position to BEV coordinates
            world_pos = np.array([[x, y]])
            bev_pos = self.world_to_bev(world_pos)
            return (float(bev_pos[0, 0]), float(bev_pos[0, 1]), float(yaw))
        else:
            # Already in world coordinates
            return ego_pose


class TensorFormatAdapter:
    """
    Handles tensor format conversions and device management.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize tensor format adapter.
        
        Args:
            device: Target device for tensors
        """
        self.device = device
        
    def numpy_to_tensor(
        self,
        array: np.ndarray,
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False
    ) -> torch.Tensor:
        """
        Convert numpy array to tensor.
        
        Args:
            array: Input numpy array
            dtype: Target tensor dtype
            requires_grad: Whether tensor requires gradients
            
        Returns:
            Tensor on target device
        """
        tensor = torch.from_numpy(array).to(dtype).to(self.device)
        tensor.requires_grad_(requires_grad)
        return tensor
    
    def ensure_tensor_format(
        self,
        data: Union[np.ndarray, torch.Tensor, List],
        target_shape: Optional[Tuple[int, ...]] = None,
        dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """
        Ensure data is in tensor format with correct shape and device.
        
        Args:
            data: Input data
            target_shape: Target tensor shape
            dtype: Target dtype
            
        Returns:
            Formatted tensor
        """
        # Convert to tensor
        if isinstance(data, np.ndarray):
            tensor = self.numpy_to_tensor(data, dtype)
        elif isinstance(data, list):
            tensor = torch.tensor(data, dtype=dtype, device=self.device)
        elif isinstance(data, torch.Tensor):
            tensor = data.to(dtype).to(self.device)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Reshape if target shape specified
        if target_shape is not None:
            tensor = tensor.view(target_shape)
            
        return tensor
    
    def format_batch_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Ensure all tensors have batch dimension.
        
        Args:
            tensors: Dictionary of tensors
            batch_size: Target batch size
            
        Returns:
            Formatted tensors with batch dimensions
        """
        formatted = {}
        
        for key, tensor in tensors.items():
            if tensor.dim() == 0:
                # Scalar -> [B]
                formatted[key] = tensor.unsqueeze(0).repeat(batch_size)
            elif tensor.size(0) != batch_size:
                # Add or adjust batch dimension
                if tensor.dim() == 1 and tensor.size(0) == 1:
                    formatted[key] = tensor.repeat(batch_size)
                else:
                    formatted[key] = tensor.unsqueeze(0) if tensor.size(0) != batch_size else tensor
            else:
                formatted[key] = tensor
                
        return formatted


class TrackDataAdapter:
    """
    Adapts tracking data from ITRI format to UniAD format.
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize track data adapter.
        
        Args:
            device: Device for tensor operations
        """
        self.device = device
        self.tensor_adapter = TensorFormatAdapter(device)
        
    def adapt_track_queries(
        self,
        track_data: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Adapt track queries to UniAD format.
        
        Args:
            track_data: Raw track query data
            
        Returns:
            UniAD-compatible track query format
        """
        adapted = {}
        
        # Ensure tensors are on correct device
        for key, tensor in track_data.items():
            if isinstance(tensor, torch.Tensor):
                adapted[key] = tensor.to(self.device)
            else:
                adapted[key] = tensor
                
        # Validate required fields
        required_fields = [
            'track_query_embeddings', 
            'sdc_embedding',
            'track_bbox_results',
            'sdc_track_bbox_results'
        ]
        
        for field in required_fields:
            if field not in adapted:
                logger.warning(f"Missing track field: {field}")
                adapted[field] = self._create_empty_field(field)
                
        return adapted
    
    def _create_empty_field(self, field_name: str) -> Any:
        """Create empty field for missing track data."""
        if field_name == 'track_query_embeddings':
            return torch.zeros(1, 1, 256, device=self.device)
        elif field_name == 'sdc_embedding':
            return torch.zeros(256, device=self.device)
        else:
            return []


class SemanticMapAdapter:
    """
    Adapts semantic map data from ITRI format to UniAD format.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        """
        Initialize semantic map adapter.
        
        Args:
            device: Device for tensor operations
            pc_range: Point cloud range for coordinate conversion
        """
        self.device = device
        self.coord_adapter = CoordinateSystemAdapter(pc_range)
        self.tensor_adapter = TensorFormatAdapter(device)
        
    def adapt_roadlines(
        self,
        roadlines_data: Dict[str, Any],
        ego_pose: Tuple[float, float, float]
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt ITRI roadlines to lane query format.
        
        Args:
            roadlines_data: Raw roadlines data from JSON
            ego_pose: Current ego vehicle pose
            
        Returns:
            Adapted lane queries
        """
        adapted_lanes = {
            'lane_points': [],
            'lane_types': [],
            'lane_confidences': []
        }
        
        for road_id, road_data in roadlines_data.items():
            if 'geometry' in road_data:
                # Extract lane points
                points = self._extract_lane_points(road_data['geometry'])
                
                # Transform to ego-relative coordinates
                ego_relative_points = self._transform_to_ego_frame(points, ego_pose)
                
                # Convert to BEV coordinates
                bev_points = self.coord_adapter.world_to_bev(ego_relative_points)
                
                adapted_lanes['lane_points'].append(bev_points)
                adapted_lanes['lane_types'].append(self._get_lane_type(road_data))
                adapted_lanes['lane_confidences'].append(1.0)  # Default confidence
                
        # Convert to tensors
        lane_tensors = {}
        for key, values in adapted_lanes.items():
            if values:
                if key == 'lane_points':
                    # Pad and stack lane points
                    lane_tensors[key] = self._pad_and_stack_points(values)
                else:
                    lane_tensors[key] = self.tensor_adapter.ensure_tensor_format(values)
            else:
                # Create empty tensors
                lane_tensors[key] = torch.empty(0, device=self.device)
                
        return lane_tensors
    
    def _extract_lane_points(self, geometry: Dict) -> np.ndarray:
        """Extract points from geometry data."""
        if 'coordinates' in geometry:
            coords = geometry['coordinates']
            if isinstance(coords[0][0], (list, tuple)):
                # Multi-line geometry
                points = np.vstack([np.array(line) for line in coords])
            else:
                # Single line geometry
                points = np.array(coords)
                
            return points[:, :2]  # Keep only x, y coordinates
        else:
            return np.empty((0, 2))
    
    def _transform_to_ego_frame(
        self,
        points: np.ndarray,
        ego_pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """Transform points to ego vehicle frame."""
        ego_x, ego_y, ego_yaw = ego_pose
        
        # Translate to ego position
        ego_relative = points - np.array([ego_x, ego_y])
        
        # Rotate to ego orientation
        cos_yaw, sin_yaw = np.cos(-ego_yaw), np.sin(-ego_yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])
        
        transformed = ego_relative @ rotation_matrix.T
        return transformed
    
    def _get_lane_type(self, road_data: Dict) -> int:
        """Extract lane type from road data."""
        # Map ITRI lane types to UniAD lane type indices
        lane_type_map = {
            'solid': 0,
            'dashed': 1,
            'double': 2,
            'curb': 3
        }
        
        lane_type = road_data.get('type', 'solid')
        return lane_type_map.get(lane_type, 0)
    
    def _pad_and_stack_points(self, point_lists: List[np.ndarray]) -> torch.Tensor:
        """Pad and stack variable-length point sequences."""
        if not point_lists:
            return torch.empty(0, 0, 2, device=self.device)
        
        # Find maximum sequence length
        max_len = max(len(points) for points in point_lists)
        
        # Pad sequences
        padded_points = []
        for points in point_lists:
            if len(points) < max_len:
                # Pad with last point
                padding = np.tile(points[-1:], (max_len - len(points), 1))
                padded = np.vstack([points, padding])
            else:
                padded = points[:max_len]  # Truncate if too long
                
            padded_points.append(padded)
        
        # Stack into tensor
        stacked = np.stack(padded_points, axis=0)
        return self.tensor_adapter.numpy_to_tensor(stacked)


class ITRIDataAdapter:
    """
    Main adapter class that coordinates all data format conversions.
    """
    
    def __init__(
        self,
        device: str = 'cuda',
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    ):
        """
        Initialize ITRI data adapter.
        
        Args:
            device: Device for tensor operations
            pc_range: Point cloud range for coordinate conversions
        """
        self.device = device
        self.pc_range = pc_range
        
        # Initialize sub-adapters
        self.coord_adapter = CoordinateSystemAdapter(pc_range)
        self.tensor_adapter = TensorFormatAdapter(device)
        self.track_adapter = TrackDataAdapter(device)
        self.semantic_adapter = SemanticMapAdapter(device, pc_range)
        
    def adapt_complete_frame(
        self,
        itri_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Adapt complete frame data from ITRI to UniAD format.
        
        Args:
            itri_data: Raw ITRI data dictionary
            
        Returns:
            UniAD-compatible data dictionary
        """
        adapted_data = {}
        
        # Adapt ego pose
        if 'ego_pose' in itri_data:
            adapted_data['ego_pose'] = self.coord_adapter.transform_ego_pose(
                itri_data['ego_pose'], 'bev'
            )
        
        # Adapt CAN bus data
        if 'canbus_data' in itri_data:
            adapted_data['can_bus'] = self.tensor_adapter.ensure_tensor_format(
                itri_data['canbus_data']['can_bus'][itri_data.get('frame_index', 0)]
            )
        
        # Adapt track queries
        if 'track_queries' in itri_data:
            adapted_data['track_queries'] = self.track_adapter.adapt_track_queries(
                itri_data['track_queries']
            )
        
        # Adapt image metadata
        if 'image_paths' in itri_data:
            adapted_data['img_metas'] = self._create_img_metas(
                itri_data['image_paths'],
                itri_data.get('canbus_data', {}),
                itri_data.get('frame_index', 0)
            )
        
        # Copy other fields
        for key in ['frame_index', 'timestamp', 'image_paths']:
            if key in itri_data:
                adapted_data[key] = itri_data[key]
        
        return adapted_data
    
    def _create_img_metas(
        self,
        image_paths: Dict[str, str],
        canbus_data: Dict[str, np.ndarray],
        frame_index: int
    ) -> List[Dict[str, Any]]:
        """Create image metadata for UniAD."""
        img_metas = [{
            'filename': list(image_paths.values()),
            'img_shape': [(480, 800, 3)] * len(image_paths),  # Default image shape
            'pad_shape': [(480, 800, 3)] * len(image_paths),
            'scale_factor': [1.0] * len(image_paths),
            'flip': False,
            'flip_direction': 'horizontal',
        }]
        
        # Add CAN bus data if available
        if canbus_data and 'can_bus' in canbus_data:
            img_metas[0]['can_bus'] = canbus_data['can_bus'][frame_index]
            
        # Add timestamp if available
        if canbus_data and 'timestamps' in canbus_data:
            img_metas[0]['timestamp'] = canbus_data['timestamps'][frame_index]
            
        return img_metas
    
    def validate_adapted_data(self, adapted_data: Dict[str, Any]) -> bool:
        """
        Validate that adapted data meets UniAD requirements.
        
        Args:
            adapted_data: Adapted data dictionary
            
        Returns:
            True if data is valid
        """
        required_fields = ['ego_pose', 'can_bus', 'track_queries', 'img_metas']
        
        for field in required_fields:
            if field not in adapted_data:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate tensor shapes and devices
        tensor_fields = ['can_bus']
        for field in tensor_fields:
            if field in adapted_data:
                tensor = adapted_data[field]
                if not isinstance(tensor, torch.Tensor):
                    logger.error(f"Field {field} is not a tensor")
                    return False
                if tensor.device.type != self.device.split(':')[0]:
                    logger.error(f"Field {field} on wrong device: {tensor.device}")
                    return False
        
        # Validate track queries structure
        track_queries = adapted_data.get('track_queries', {})
        required_track_fields = ['track_query_embeddings', 'sdc_embedding']
        for field in required_track_fields:
            if field not in track_queries:
                logger.error(f"Missing track field: {field}")
                return False
        
        logger.info("Adapted data validation passed")
        return True


def create_data_adapter(
    device: str = 'cuda',
    pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
) -> ITRIDataAdapter:
    """
    Factory function to create ITRI data adapter.
    
    Args:
        device: Device for tensor operations
        pc_range: Point cloud range for coordinate conversions
        
    Returns:
        Configured ITRIDataAdapter instance
    """
    return ITRIDataAdapter(device, pc_range)


def main():
    """Test the ITRI data adapter."""
    print("ITRI Data Format Adapter Test")
    print("=" * 40)
    
    # Create adapter
    adapter = create_data_adapter()
    
    # Test coordinate transformations
    coord_adapter = CoordinateSystemAdapter()
    world_coords = np.array([[10.0, 20.0], [-5.0, 15.0]])
    bev_coords = coord_adapter.world_to_bev(world_coords)
    world_back = coord_adapter.bev_to_world(bev_coords)
    
    print(f"World coords: {world_coords}")
    print(f"BEV coords: {bev_coords}")
    print(f"World back: {world_back}")
    
    # Test tensor formatting
    tensor_adapter = TensorFormatAdapter()
    test_array = np.random.randn(3, 4)
    tensor = tensor_adapter.numpy_to_tensor(test_array)
    
    print(f"Array shape: {test_array.shape}")
    print(f"Tensor shape: {tensor.shape}, device: {tensor.device}")
    
    print("✅ ITRI Data Adapter test completed")


if __name__ == "__main__":
    main()