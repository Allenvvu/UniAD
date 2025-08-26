#!/usr/bin/env python3
"""
Example Usage Script for ITRI to MotionFormer Integration

This script demonstrates how to use the ITRI semantic map converter
with UniAD's MotionFormer for trajectory prediction.

Author: Generated for UniAD Integration Project
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
import argparse
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic_map.motionformer_integration import (
    create_integrator, 
    validate_inputs
)
from semantic_map.itri_to_lane_query import create_converter


def create_sample_tracked_objects() -> List[Dict]:
    """
    Create sample tracked objects for testing.
    
    Returns:
        List of sample tracked object dictionaries
    """
    sample_objects = [
        {
            'bbox': [10.0, 5.0, 0.0, 4.5, 2.0, 1.8, 0.1],  # [x, y, z, l, w, h, yaw]
            'score': 0.95,
            'label': 0,  # Car
            'track_id': 1,
            'velocity': [2.0, 0.1]  # [vx, vy]
        },
        {
            'bbox': [-5.0, -3.0, 0.0, 4.2, 1.8, 1.6, -0.2],
            'score': 0.87,
            'label': 0,  # Car
            'track_id': 2,
            'velocity': [-1.5, 0.5]
        },
        {
            'bbox': [20.0, 0.0, 0.0, 0.6, 0.6, 1.7, 0.0],
            'score': 0.78,
            'label': 7,  # Pedestrian
            'track_id': 3,
            'velocity': [0.8, -0.2]
        }
    ]
    
    return sample_objects


def create_sample_bev_features(device: str = 'cuda') -> torch.Tensor:
    """
    Create sample BEV features for testing.
    
    Args:
        device: Device to create tensor on
        
    Returns:
        Sample BEV feature tensor [h*w, B, D]
    """
    bev_h, bev_w = 200, 200
    embed_dim = 256
    batch_size = 1
    
    # Create sample BEV features
    bev_features = torch.randn(
        bev_h * bev_w, batch_size, embed_dim, 
        device=device
    )
    
    return bev_features


def example_basic_conversion():
    """
    Example 1: Basic ITRI to lane_query conversion
    """
    print("=" * 60)
    print("Example 1: Basic ITRI to Lane Query Conversion")
    print("=" * 60)
    
    # Configuration
    converter_config = {
        'embed_dim': 256,
        'max_queries': 300,
        'sample_distance': 1.0,
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    }
    
    # Create converter
    converter = create_converter(converter_config)
    
    # Sample ego pose (x, y, yaw in world coordinates)
    ego_pose = (100.0, 50.0, 0.5)  # 100m east, 50m north, 0.5 rad heading
    
    # Path to ITRI data
    itri_data_path = "./itri"
    
    if not Path(itri_data_path).exists():
        print(f"Warning: ITRI data path '{itri_data_path}' not found!")
        print("Make sure to place your ITRI JSON files in the semantic-map/itri/ folder")
        return
    
    try:
        # Convert ITRI data to lane queries
        lane_query, lane_query_pos = converter.convert(itri_data_path, ego_pose)
        
        print(f"✓ Conversion successful!")
        print(f"  Lane query shape: {lane_query.shape}")
        print(f"  Lane query pos shape: {lane_query_pos.shape}")
        print(f"  Device: {lane_query.device}")
        
        # Analyze results
        non_zero_queries = (lane_query.norm(dim=-1) > 0).sum().item()
        print(f"  Non-zero queries: {non_zero_queries}/{lane_query.shape[1]}")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")


def example_full_integration():
    """
    Example 2: Full integration with MotionFormer
    """
    print("\n" + "=" * 60)
    print("Example 2: Full MotionFormer Integration")
    print("=" * 60)
    
    # Check device availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Configuration
    converter_config = {
        'embed_dim': 256,
        'max_queries': 300,
        'sample_distance': 1.0,
        'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    }
    
    # Create integrator
    integrator = create_integrator(converter_config, device)
    
    # Sample data
    itri_data_path = "./itri"
    ego_pose = (100.0, 50.0, 0.5)
    tracked_objects = create_sample_tracked_objects()
    bev_features = create_sample_bev_features(device)
    
    # Validate inputs
    if not validate_inputs(itri_data_path, ego_pose, tracked_objects):
        print("✗ Input validation failed!")
        return
    
    print("✓ Input validation passed!")
    
    try:
        # Prepare inputs for MotionFormer
        motion_inputs = integrator.prepare_motion_inputs(
            itri_data_path=itri_data_path,
            ego_pose=ego_pose,
            tracked_objects=tracked_objects,
            bev_features=bev_features
        )
        
        print("✓ Motion inputs prepared successfully!")
        print(f"  BEV embed shape: {motion_inputs['bev_embed'].shape}")
        print(f"  Track queries: {motion_inputs['outs_track']['track_query_embeddings'].shape}")
        print(f"  Lane queries from args_tuple: {motion_inputs['outs_seg']['args_tuple'][3].shape}")
        print(f"  Lane positions from args_tuple: {motion_inputs['outs_seg']['args_tuple'][5].shape}")
        
        print("\n✓ Ready for MotionFormer inference!")
        print("  To run actual inference, pass these inputs to motion_head.forward_test()")
        
    except Exception as e:
        print(f"✗ Integration failed: {e}")
        import traceback
        traceback.print_exc()


def example_batch_processing():
    """
    Example 3: Batch processing multiple frames
    """
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing Multiple Frames")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create integrator
    integrator = create_integrator(device=device)
    
    # Sample trajectory (multiple ego poses)
    trajectory = [
        (100.0, 50.0, 0.5),
        (102.0, 51.0, 0.52),
        (104.0, 52.0, 0.54),
        (106.0, 53.0, 0.56),
        (108.0, 54.0, 0.58)
    ]
    
    itri_data_path = "./itri"
    
    print(f"Processing {len(trajectory)} frames...")
    
    results = []
    for i, ego_pose in enumerate(trajectory):
        try:
            # Simulate different tracked objects for each frame
            tracked_objects = create_sample_tracked_objects()
            
            # Modify object positions slightly for each frame
            for obj in tracked_objects:
                obj['bbox'][0] += i * 0.5  # Move objects forward
                obj['bbox'][1] += i * 0.1  # Slight lateral movement
            
            bev_features = create_sample_bev_features(device)
            
            # Prepare inputs
            motion_inputs = integrator.prepare_motion_inputs(
                itri_data_path=itri_data_path,
                ego_pose=ego_pose,
                tracked_objects=tracked_objects,
                bev_features=bev_features,
                use_cache=True  # Enable caching for efficiency
            )
            
            results.append({
                'frame': i,
                'ego_pose': ego_pose,
                'num_objects': len(tracked_objects),
                'lane_queries_shape': motion_inputs['outs_seg']['args_tuple'][3].shape
            })
            
            print(f"  Frame {i}: ✓")
            
        except Exception as e:
            print(f"  Frame {i}: ✗ ({e})")
    
    print(f"\n✓ Processed {len(results)}/{len(trajectory)} frames successfully!")
    
    # Display cache efficiency
    print(f"  Map cache size: {len(integrator._map_cache)} entries")


def example_performance_test():
    """
    Example 4: Performance testing and optimization
    """
    print("\n" + "=" * 60)
    print("Example 4: Performance Testing")
    print("=" * 60)
    
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    integrator = create_integrator(device=device)
    
    # Test parameters
    num_iterations = 10
    ego_pose = (100.0, 50.0, 0.5)
    tracked_objects = create_sample_tracked_objects()
    itri_data_path = "./itri"
    
    print(f"Running {num_iterations} iterations...")
    
    # Warm-up
    if Path(itri_data_path).exists():
        bev_features = create_sample_bev_features(device)
        integrator.prepare_motion_inputs(
            itri_data_path, ego_pose, tracked_objects, bev_features)
    
    # Timing test
    times = []
    for i in range(num_iterations):
        start_time = time.time()
        
        try:
            bev_features = create_sample_bev_features(device)
            motion_inputs = integrator.prepare_motion_inputs(
                itri_data_path, ego_pose, tracked_objects, bev_features)
            
            # Simulate some processing time
            torch.cuda.synchronize() if device == 'cuda' else None
            
            end_time = time.time()
            times.append(end_time - start_time)
            
        except Exception as e:
            print(f"  Iteration {i}: Failed ({e})")
            continue
    
    if times:
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"✓ Performance results:")
        print(f"  Average time: {avg_time:.4f}s ± {std_time:.4f}s")
        print(f"  Max time: {max(times):.4f}s")
        print(f"  Min time: {min(times):.4f}s")
        print(f"  Throughput: ~{1/avg_time:.1f} FPS")
    else:
        print("✗ No successful iterations")


def main():
    """Main function to run examples."""
    parser = argparse.ArgumentParser(
        description="ITRI to MotionFormer Integration Examples"
    )
    parser.add_argument(
        '--example', 
        type=str, 
        choices=['basic', 'integration', 'batch', 'performance', 'all'],
        default='all',
        help='Which example to run'
    )
    parser.add_argument(
        '--itri-path',
        type=str,
        default='./itri',
        help='Path to ITRI data folder'
    )
    
    args = parser.parse_args()
    
    print("ITRI to MotionFormer Integration Examples")
    print("========================================")
    
    # Change to semantic-map directory for relative paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    if args.example == 'basic' or args.example == 'all':
        example_basic_conversion()
    
    if args.example == 'integration' or args.example == 'all':
        example_full_integration()
    
    if args.example == 'batch' or args.example == 'all':
        example_batch_processing()
    
    if args.example == 'performance' or args.example == 'all':
        example_performance_test()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Ensure your ITRI JSON files are in semantic-map/itri/")
    print("2. Integrate with your actual UniAD MotionFormer model")
    print("3. Test with real tracking data from your pipeline")
    print("4. Optimize performance for your specific requirements")


if __name__ == "__main__":
    main()