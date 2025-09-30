#!/usr/bin/env python3
"""
Debug script to test coordinate transformation with actual semantic map data
"""

import json
import sys
import os
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')
from coordinate_transform import transform_points_to_ego, filter_points_by_bev_range

def load_semantic_map_sample():
    """Load a sample of roadlines from the semantic map"""
    roadlines_file = "/home/bryan/Desktop/Allen/UniAD/semantic-map/data/hct_logistic/roadlines.json"
    
    with open(roadlines_file, 'r') as f:
        data = json.load(f)
    
    # Get first few roadlines for testing
    sample_segments = []
    for i, roadline in enumerate(data['roadlines'][:5]):  # Test with first 5 roadlines
        points = [(p['x'], p['y'], p['z']) for p in roadline['points']]
        sample_segments.append({
            'id': roadline['id'],
            'points': points,
            'num_points': len(points)
        })
    
    return sample_segments

def test_coordinate_transformation():
    """Test coordinate transformation with real data"""
    print("=== Coordinate Transformation Debug ===\n")
    
    # Load semantic map sample
    segments = load_semantic_map_sample()
    
    # Test ego pose (from frame metadata)
    ego_pose = (283.1, -131.0, 3.14)  # x, y, yaw
    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    
    print(f"Ego pose: ({ego_pose[0]:.1f}, {ego_pose[1]:.1f}, {ego_pose[2]:.2f})")
    print(f"BEV range: X=[{pc_range[0]}, {pc_range[3]}], Y=[{pc_range[1]}, {pc_range[4]}]\n")
    
    total_world_points = 0
    total_ego_points = 0
    total_filtered_points = 0
    
    for segment in segments:
        world_points = segment['points']
        total_world_points += len(world_points)
        
        # Show sample world coordinates
        print(f"Roadline {segment['id']} ({len(world_points)} points)")
        if world_points:
            first_point = world_points[0]
            last_point = world_points[-1]
            print(f"  World coords: ({first_point[0]:.1f}, {first_point[1]:.1f}) to ({last_point[0]:.1f}, {last_point[1]:.1f})")
        
        # Transform to ego coordinates
        ego_points = transform_points_to_ego(world_points, ego_pose)
        total_ego_points += len(ego_points)
        
        if ego_points:
            first_ego = ego_points[0]
            last_ego = ego_points[-1]
            print(f"  Ego coords:   ({first_ego[0]:.1f}, {first_ego[1]:.1f}) to ({last_ego[0]:.1f}, {last_ego[1]:.1f})")
        
        # Filter by BEV range
        filtered_points = filter_points_by_bev_range(ego_points, pc_range)
        total_filtered_points += len(filtered_points)
        
        print(f"  Points after BEV filtering: {len(filtered_points)}")
        
        if filtered_points:
            print(f"  Sample filtered point: ({filtered_points[0][0]:.1f}, {filtered_points[0][1]:.1f})")
        
        print()
    
    print(f"=== Summary ===")
    print(f"Total world points: {total_world_points}")
    print(f"Total ego points: {total_ego_points}")
    print(f"Total filtered points: {total_filtered_points}")
    print(f"Filtering success rate: {total_filtered_points/total_world_points*100:.1f}%")

if __name__ == "__main__":
    test_coordinate_transformation()