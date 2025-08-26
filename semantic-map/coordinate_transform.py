#!/usr/bin/env python3
"""
Coordinate Transformation Utilities

This module provides coordinate transformation functions for converting between
world coordinates, ego vehicle coordinates, and BEV (Bird's Eye View) coordinates.

Author: Generated for UniAD Integration Project
"""

import numpy as np
import math
from typing import List, Tuple, Optional


def transform_points_to_ego(
    points: List[Tuple[float, float, float]],
    ego_pose: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:
    """
    Transform points from world coordinates to ego vehicle coordinates.
    
    Args:
        points: List of world coordinate points [(x, y, z), ...]
        ego_pose: Ego vehicle pose (x, y, yaw) in world coordinates
        
    Returns:
        List of points in ego coordinate system
    """
    if not points:
        return []
    
    ego_x, ego_y, ego_yaw = ego_pose
    
    # Create transformation matrix
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    
    ego_points = []
    for world_point in points:
        world_x, world_y, world_z = world_point
        
        # Translate to ego position
        translated_x = world_x - ego_x
        translated_y = world_y - ego_y
        
        # Rotate to ego orientation
        ego_local_x = translated_x * cos_yaw + translated_y * sin_yaw
        ego_local_y = -translated_x * sin_yaw + translated_y * cos_yaw
        ego_local_z = world_z  # Z typically unchanged for ground-level features
        
        ego_points.append((ego_local_x, ego_local_y, ego_local_z))
    
    return ego_points


def transform_points_to_world(
    ego_points: List[Tuple[float, float, float]],
    ego_pose: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:
    """
    Transform points from ego coordinates back to world coordinates.
    
    Args:
        ego_points: List of ego coordinate points [(x, y, z), ...]
        ego_pose: Ego vehicle pose (x, y, yaw) in world coordinates
        
    Returns:
        List of points in world coordinate system
    """
    if not ego_points:
        return []
    
    ego_x, ego_y, ego_yaw = ego_pose
    
    # Create inverse transformation matrix
    cos_yaw = math.cos(ego_yaw)
    sin_yaw = math.sin(ego_yaw)
    
    world_points = []
    for ego_point in ego_points:
        ego_local_x, ego_local_y, ego_local_z = ego_point
        
        # Rotate from ego orientation to world orientation
        translated_x = ego_local_x * cos_yaw - ego_local_y * sin_yaw
        translated_y = ego_local_x * sin_yaw + ego_local_y * cos_yaw
        
        # Translate to world position
        world_x = translated_x + ego_x
        world_y = translated_y + ego_y
        world_z = ego_local_z
        
        world_points.append((world_x, world_y, world_z))
    
    return world_points


def filter_points_by_bev_range(
    points: List[Tuple[float, float, float]],
    pc_range: List[float]
) -> List[Tuple[float, float, float]]:
    """
    Filter points to only include those within BEV range.
    
    Args:
        points: List of points in ego coordinates
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        List of filtered points
    """
    if not points:
        return []
    
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    
    filtered_points = []
    for point in points:
        x, y, z = point
        
        if (x_min <= x <= x_max and 
            y_min <= y <= y_max and 
            z_min <= z <= z_max):
            filtered_points.append(point)
    
    return filtered_points


def normalize_coordinates(
    point: Tuple[float, float, float],
    pc_range: List[float]
) -> Tuple[float, float, float]:
    """
    Normalize coordinates to [0, 1] range based on PC range.
    
    Args:
        point: Single point (x, y, z)
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        
    Returns:
        Normalized coordinates
    """
    x, y, z = point
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    
    # Normalize to [0, 1]
    norm_x = (x - x_min) / (x_max - x_min) if x_max != x_min else 0.0
    norm_y = (y - y_min) / (y_max - y_min) if y_max != y_min else 0.0
    norm_z = (z - z_min) / (z_max - z_min) if z_max != z_min else 0.0
    
    # Clamp to [0, 1]
    norm_x = max(0.0, min(1.0, norm_x))
    norm_y = max(0.0, min(1.0, norm_y))
    norm_z = max(0.0, min(1.0, norm_z))
    
    return (norm_x, norm_y, norm_z)


def points_to_bev_grid(
    points: List[Tuple[float, float, float]],
    pc_range: List[float],
    grid_size: Tuple[int, int] = (200, 200)
) -> List[Tuple[int, int]]:
    """
    Convert ego coordinate points to BEV grid indices.
    
    Args:
        points: List of points in ego coordinates
        pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        grid_size: BEV grid dimensions (height, width)
        
    Returns:
        List of (row, col) grid indices
    """
    if not points:
        return []
    
    x_min, y_min, _, x_max, y_max, _ = pc_range
    grid_h, grid_w = grid_size
    
    # Calculate grid resolution
    x_res = (x_max - x_min) / grid_w
    y_res = (y_max - y_min) / grid_h
    
    grid_indices = []
    for point in points:
        x, y, _ = point
        
        # Convert to grid coordinates
        col = int((x - x_min) / x_res)
        row = int((y - y_min) / y_res)
        
        # Clamp to valid range
        col = max(0, min(grid_w - 1, col))
        row = max(0, min(grid_h - 1, row))
        
        grid_indices.append((row, col))
    
    return grid_indices


def compute_ego_pose_from_trajectory(
    trajectory: List[Tuple[float, float, float]],
    timestamp_index: int = 0
) -> Tuple[float, float, float]:
    """
    Compute ego pose from trajectory data.
    
    Args:
        trajectory: List of ego positions [(x, y, z), ...]
        timestamp_index: Index in trajectory to use as current pose
        
    Returns:
        Ego pose (x, y, yaw)
    """
    if not trajectory or timestamp_index >= len(trajectory):
        return (0.0, 0.0, 0.0)
    
    if len(trajectory) == 1:
        x, y, z = trajectory[0]
        return (x, y, 0.0)  # No heading info available
    
    # Current position
    curr_x, curr_y, curr_z = trajectory[timestamp_index]
    
    # Calculate heading from trajectory direction
    if timestamp_index < len(trajectory) - 1:
        # Use forward direction
        next_x, next_y, _ = trajectory[timestamp_index + 1]
        dx = next_x - curr_x
        dy = next_y - curr_y
    else:
        # Use backward direction
        prev_x, prev_y, _ = trajectory[timestamp_index - 1]
        dx = curr_x - prev_x
        dy = curr_y - prev_y
    
    # Calculate yaw angle
    yaw = math.atan2(dy, dx)
    
    return (curr_x, curr_y, yaw)


def interpolate_ego_pose(
    pose1: Tuple[float, float, float],
    pose2: Tuple[float, float, float],
    alpha: float
) -> Tuple[float, float, float]:
    """
    Interpolate between two ego poses.
    
    Args:
        pose1: First pose (x, y, yaw)
        pose2: Second pose (x, y, yaw)
        alpha: Interpolation factor [0, 1]
        
    Returns:
        Interpolated pose
    """
    x1, y1, yaw1 = pose1
    x2, y2, yaw2 = pose2
    
    # Linear interpolation for position
    x = x1 + alpha * (x2 - x1)
    y = y1 + alpha * (y2 - y1)
    
    # Spherical interpolation for angle
    # Handle angle wrapping
    diff = yaw2 - yaw1
    if diff > math.pi:
        diff -= 2 * math.pi
    elif diff < -math.pi:
        diff += 2 * math.pi
    
    yaw = yaw1 + alpha * diff
    
    # Normalize angle to [-pi, pi]
    while yaw > math.pi:
        yaw -= 2 * math.pi
    while yaw < -math.pi:
        yaw += 2 * math.pi
    
    return (x, y, yaw)


def compute_relative_pose(
    target_pose: Tuple[float, float, float],
    reference_pose: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Compute relative pose between two poses.
    
    Args:
        target_pose: Target pose (x, y, yaw)
        reference_pose: Reference pose (x, y, yaw)
        
    Returns:
        Relative pose from reference to target
    """
    target_x, target_y, target_yaw = target_pose
    ref_x, ref_y, ref_yaw = reference_pose
    
    # Transform target to reference coordinate system
    ego_points = transform_points_to_ego(
        [(target_x, target_y, 0.0)], 
        reference_pose
    )
    
    rel_x, rel_y, _ = ego_points[0]
    
    # Compute relative yaw
    rel_yaw = target_yaw - ref_yaw
    
    # Normalize angle
    while rel_yaw > math.pi:
        rel_yaw -= 2 * math.pi
    while rel_yaw < -math.pi:
        rel_yaw += 2 * math.pi
    
    return (rel_x, rel_y, rel_yaw)


def transform_polygon_to_ego(
    polygon_points: List[Tuple[float, float, float]],
    ego_pose: Tuple[float, float, float]
) -> List[Tuple[float, float, float]]:
    """
    Transform polygon points to ego coordinate system.
    
    This is a specialized version of transform_points_to_ego for polygon data
    that may have additional processing requirements.
    
    Args:
        polygon_points: List of polygon vertex points
        ego_pose: Ego vehicle pose (x, y, yaw)
        
    Returns:
        Transformed polygon points in ego coordinates
    """
    return transform_points_to_ego(polygon_points, ego_pose)


def project_points_to_ground(
    points: List[Tuple[float, float, float]],
    ground_height: float = 0.0
) -> List[Tuple[float, float, float]]:
    """
    Project 3D points to ground plane.
    
    Args:
        points: List of 3D points
        ground_height: Height of ground plane
        
    Returns:
        Points projected to ground plane
    """
    projected_points = []
    for point in points:
        x, y, z = point
        projected_points.append((x, y, ground_height))
    
    return projected_points