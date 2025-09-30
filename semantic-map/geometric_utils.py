#!/usr/bin/env python3
"""
Geometric Utilities for Polyline Processing

This module provides geometric operations for processing ITRI polyline data,
including sampling, feature computation, and coordinate transformations.

Author: Generated for UniAD Integration Project
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.interpolate import interp1d
from scipy import ndimage
import math


def sample_polyline_uniform(
    points: List[Tuple[float, float, float]], 
    sample_distance: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Sample points uniformly along a polyline at specified distance intervals.
    
    Args:
        points: List of 3D points [(x, y, z), ...]
        sample_distance: Distance between sampled points in meters
        
    Returns:
        List of uniformly sampled points
    """
    if len(points) < 2:
        return points
    
    points_array = np.array(points)
    
    # Calculate cumulative distances along polyline
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        dist = np.linalg.norm(points_array[i] - points_array[i-1])
        distances[i] = distances[i-1] + dist
    
    total_length = distances[-1]
    if total_length < sample_distance:
        return points
    
    # Create interpolation functions for each dimension
    interp_x = interp1d(distances, points_array[:, 0], kind='linear')
    interp_y = interp1d(distances, points_array[:, 1], kind='linear') 
    interp_z = interp1d(distances, points_array[:, 2], kind='linear')
    
    # Sample at uniform intervals
    sample_distances = np.arange(0, total_length, sample_distance)
    
    sampled_points = []
    for d in sample_distances:
        if d <= total_length:
            x = float(interp_x(d))
            y = float(interp_y(d))
            z = float(interp_z(d))
            sampled_points.append((x, y, z))
    
    # Always include the last point
    if sample_distances[-1] < total_length:
        sampled_points.append(points[-1])
    
    return sampled_points


def compute_direction_features(
    points: List[Tuple[float, float, float]]
) -> Tuple[List[float], List[float]]:
    """
    Compute direction vectors along a polyline.
    
    Args:
        points: List of 3D points
        
    Returns:
        Tuple of (direction_x, direction_y) lists
    """
    if len(points) < 2:
        return [0.0], [0.0]
    
    points_array = np.array(points)
    
    # Compute direction vectors between consecutive points
    directions = []
    for i in range(len(points) - 1):
        diff = points_array[i+1] - points_array[i]
        # Normalize to unit vector
        length = np.linalg.norm(diff[:2])  # Only x, y for direction
        if length > 0:
            direction = diff[:2] / length
        else:
            direction = np.array([0.0, 0.0])
        directions.append(direction)
    
    # Add the last direction to match point count
    if directions:
        directions.append(directions[-1])
    
    direction_x = [d[0] for d in directions]
    direction_y = [d[1] for d in directions]
    
    return direction_x, direction_y


def compute_curvature_features(
    points: List[Tuple[float, float, float]]
) -> List[float]:
    """
    Compute curvature at each point along a polyline.
    
    Args:
        points: List of 3D points
        
    Returns:
        List of curvature values
    """
    if len(points) < 3:
        return [0.0] * len(points)
    
    points_array = np.array(points)
    curvatures = []
    
    for i in range(len(points)):
        if i == 0:
            # First point: use forward difference
            p1, p2, p3 = points_array[0], points_array[1], points_array[2]
        elif i == len(points) - 1:
            # Last point: use backward difference
            p1, p2, p3 = points_array[-3], points_array[-2], points_array[-1]
        else:
            # Middle points: use central difference
            p1, p2, p3 = points_array[i-1], points_array[i], points_array[i+1]
        
        # Calculate curvature using cross product formula
        v1 = p2 - p1
        v2 = p3 - p2
        
        # Only use x, y coordinates for curvature
        v1_2d = v1[:2]
        v2_2d = v2[:2]
        
        cross = np.cross(v1_2d, v2_2d)
        mag_v1 = np.linalg.norm(v1_2d)
        mag_v2 = np.linalg.norm(v2_2d)
        
        if mag_v1 > 0 and mag_v2 > 0:
            curvature = abs(cross) / (mag_v1 * mag_v2)
        else:
            curvature = 0.0
        
        curvatures.append(curvature)
    
    return curvatures


def points_to_relative_coords(
    points: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float]]:
    """
    Convert absolute coordinates to relative coordinates (centered on first point).
    
    Args:
        points: List of absolute 3D points
        
    Returns:
        List of relative 3D coordinates
    """
    if not points:
        return []
    
    origin = np.array(points[0])
    relative_points = []
    
    for point in points:
        relative = np.array(point) - origin
        relative_points.append((relative[0], relative[1], relative[2]))
    
    return relative_points


def compute_polyline_length(points: List[Tuple[float, float, float]]) -> float:
    """
    Compute total length of a polyline.
    
    Args:
        points: List of 3D points
        
    Returns:
        Total length in meters
    """
    if len(points) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(1, len(points)):
        p1 = np.array(points[i-1])
        p2 = np.array(points[i])
        total_length += np.linalg.norm(p2 - p1)
    
    return total_length


def smooth_polyline(
    points: List[Tuple[float, float, float]], 
    sigma: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Apply Gaussian smoothing to a polyline.
    
    Args:
        points: List of 3D points
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        List of smoothed points
    """
    if len(points) < 3:
        return points
    
    points_array = np.array(points)
    
    # Apply Gaussian filter to each dimension
    smoothed_x = ndimage.gaussian_filter1d(points_array[:, 0], sigma=sigma)
    smoothed_y = ndimage.gaussian_filter1d(points_array[:, 1], sigma=sigma)
    smoothed_z = ndimage.gaussian_filter1d(points_array[:, 2], sigma=sigma)
    
    smoothed_points = []
    for i in range(len(points)):
        smoothed_points.append((
            float(smoothed_x[i]), 
            float(smoothed_y[i]), 
            float(smoothed_z[i])
        ))
    
    return smoothed_points


def compute_lane_width(
    left_boundary: List[Tuple[float, float, float]],
    right_boundary: List[Tuple[float, float, float]]
) -> float:
    """
    Compute average width between two lane boundaries.
    
    Args:
        left_boundary: Points defining left lane boundary
        right_boundary: Points defining right lane boundary
        
    Returns:
        Average lane width in meters
    """
    if not left_boundary or not right_boundary:
        return 0.0
    
    # Sample both boundaries at same intervals
    left_sampled = sample_polyline_uniform(left_boundary, 1.0)
    right_sampled = sample_polyline_uniform(right_boundary, 1.0)
    
    # Find corresponding points and compute distances
    widths = []
    for left_point in left_sampled:
        # Find closest point on right boundary
        min_dist = float('inf')
        for right_point in right_sampled:
            dist = np.linalg.norm(np.array(left_point[:2]) - np.array(right_point[:2]))
            min_dist = min(min_dist, dist)
        widths.append(min_dist)
    
    return np.mean(widths) if widths else 0.0


def compute_heading_angle(
    points: List[Tuple[float, float, float]]
) -> List[float]:
    """
    Compute heading angle at each point along polyline.
    
    Args:
        points: List of 3D points
        
    Returns:
        List of heading angles in radians
    """
    if len(points) < 2:
        return [0.0]
    
    headings = []
    
    for i in range(len(points)):
        if i == len(points) - 1:
            # Last point: use previous segment
            p1 = np.array(points[i-1][:2])
            p2 = np.array(points[i][:2])
        else:
            # Use forward direction
            p1 = np.array(points[i][:2])
            p2 = np.array(points[i+1][:2])
        
        direction = p2 - p1
        if np.linalg.norm(direction) > 0:
            heading = math.atan2(direction[1], direction[0])
        else:
            heading = 0.0
        
        headings.append(heading)
    
    return headings


def resample_polyline_by_count(
    points: List[Tuple[float, float, float]], 
    target_count: int
) -> List[Tuple[float, float, float]]:
    """
    Resample polyline to have specific number of points.
    
    Args:
        points: List of 3D points
        target_count: Desired number of points
        
    Returns:
        List of resampled points
    """
    if len(points) <= target_count:
        return points
    
    points_array = np.array(points)
    
    # Calculate cumulative distances
    distances = np.zeros(len(points))
    for i in range(1, len(points)):
        dist = np.linalg.norm(points_array[i] - points_array[i-1])
        distances[i] = distances[i-1] + dist
    
    total_length = distances[-1]
    if total_length == 0:
        return points
    
    # Create target sample distances
    target_distances = np.linspace(0, total_length, target_count)
    
    # Interpolate at target distances
    interp_x = interp1d(distances, points_array[:, 0], kind='linear')
    interp_y = interp1d(distances, points_array[:, 1], kind='linear')
    interp_z = interp1d(distances, points_array[:, 2], kind='linear')
    
    resampled_points = []
    for d in target_distances:
        x = float(interp_x(d))
        y = float(interp_y(d))
        z = float(interp_z(d))
        resampled_points.append((x, y, z))
    
    return resampled_points