#!/usr/bin/env python3
"""
ITRI Semantic Map to UniAD Integration Package

This package provides tools to convert ITRI semantic map data to the format
required by UniAD's MotionFormer, enabling trajectory prediction using 
existing HD map polylines without requiring MapFormer retraining.

Modules:
    itri_to_lane_query: Main converter from ITRI format to lane queries
    geometric_utils: Geometric processing utilities for polylines
    coordinate_transform: Coordinate system transformation utilities  
    motionformer_integration: Integration interface with MotionFormer
    example_usage: Example scripts and usage demonstrations

Author: Generated for UniAD Integration Project
"""

from .itri_to_lane_query import ITRIToLaneQueryConverter, create_converter
from .motionformer_integration import MotionFormerIntegrator, create_integrator
from .geometric_utils import (
    sample_polyline_uniform,
    compute_direction_features,
    compute_curvature_features,
    points_to_relative_coords
)
from .coordinate_transform import (
    transform_points_to_ego,
    filter_points_by_bev_range,
    normalize_coordinates
)

__version__ = "1.0.0"
__author__ = "UniAD Integration Project"

__all__ = [
    # Main classes
    'ITRIToLaneQueryConverter',
    'MotionFormerIntegrator',
    
    # Factory functions
    'create_converter',
    'create_integrator',
    
    # Geometric utilities
    'sample_polyline_uniform',
    'compute_direction_features', 
    'compute_curvature_features',
    'points_to_relative_coords',
    
    # Coordinate transformations
    'transform_points_to_ego',
    'filter_points_by_bev_range',
    'normalize_coordinates'
]


def get_default_config():
    """
    Get default configuration for the converter.
    
    Returns:
        Dictionary with default configuration parameters
    """
    return {
        'embed_dim': 256,
        'max_queries': 300,
        'sample_distance': 1.0,
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    }


def quick_convert(itri_data_path, ego_pose, config=None):
    """
    Quick conversion function for simple use cases.
    
    Args:
        itri_data_path: Path to ITRI semantic map data
        ego_pose: Ego vehicle pose (x, y, yaw)
        config: Optional configuration dictionary
        
    Returns:
        Tuple of (lane_query, lane_query_pos) tensors
    """
    if config is None:
        config = get_default_config()
    
    converter = create_converter(config)
    return converter.convert(itri_data_path, ego_pose)


def setup_integration(converter_config=None, device='cuda'):
    """
    Set up complete integration pipeline.
    
    Args:
        converter_config: Configuration for converter
        device: Device to run on
        
    Returns:
        Configured MotionFormerIntegrator instance
    """
    if converter_config is None:
        converter_config = get_default_config()
    
    return create_integrator(converter_config, device)