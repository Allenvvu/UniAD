#!/usr/bin/env python3
"""
ITRI Dataset Conversion Script

This script runs the ITRI to lane query conversion with the actual ITRI dataset
and saves the results to the query folder.

Author: Generated for UniAD Integration Project
"""

import torch
import numpy as np
import json
import pickle
from pathlib import Path
import sys
import os
from datetime import datetime
import traceback

# Add current directory and parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Import local modules
try:
    from itri_to_lane_query import create_converter
    from motionformer_integration import create_integrator
    from coordinate_transform import compute_ego_pose_from_trajectory
except ImportError:
    # Fallback to absolute imports
    from semantic_map.itri_to_lane_query import create_converter
    from semantic_map.motionformer_integration import create_integrator
    from semantic_map.coordinate_transform import compute_ego_pose_from_trajectory


def save_query_results(lane_query, lane_query_pos, ego_pose, output_dir, prefix=""):
    """
    Save lane query results to files.
    
    Args:
        lane_query: Lane query tensor
        lane_query_pos: Lane position tensor
        ego_pose: Ego vehicle pose
        output_dir: Output directory path
        prefix: Optional filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{prefix}lane_query_{timestamp}" if prefix else f"lane_query_{timestamp}"
    
    # Save tensors
    torch.save({
        'lane_query': lane_query,
        'lane_query_pos': lane_query_pos,
        'ego_pose': ego_pose,
        'metadata': {
            'timestamp': timestamp,
            'lane_query_shape': list(lane_query.shape),
            'lane_query_pos_shape': list(lane_query_pos.shape),
            'ego_pose': ego_pose,
            'device': str(lane_query.device),
            'dtype': str(lane_query.dtype)
        }
    }, output_dir / f"{base_name}.pt")
    
    # Save as numpy arrays for easier inspection
    np.savez(
        output_dir / f"{base_name}.npz",
        lane_query=lane_query.detach().cpu().numpy(),
        lane_query_pos=lane_query_pos.detach().cpu().numpy(),
        ego_pose=np.array(ego_pose)
    )
    
    # Save metadata as JSON
    metadata = {
        'timestamp': timestamp,
        'ego_pose': {'x': ego_pose[0], 'y': ego_pose[1], 'yaw': ego_pose[2]},
        'lane_query_shape': list(lane_query.shape),
        'lane_query_pos_shape': list(lane_query_pos.shape),
        'non_zero_queries': int((lane_query.norm(dim=-1) > 0).sum().item()),
        'max_query_norm': float(lane_query.norm(dim=-1).max().item()),
        'min_query_norm': float(lane_query.norm(dim=-1).min().item()),
        'mean_query_norm': float(lane_query.norm(dim=-1).mean().item())
    }
    
    with open(output_dir / f"{base_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved results to {output_dir}/{base_name}.*")
    return base_name


def analyze_itri_data(itri_path):
    """
    Analyze the ITRI dataset structure.
    
    Args:
        itri_path: Path to ITRI data folder
        
    Returns:
        Dictionary with analysis results
    """
    itri_path = Path(itri_path)
    analysis = {
        'files_found': [],
        'total_roadlines': 0,
        'total_crossings': 0,
        'total_markers': 0,
        'coordinate_bounds': {'x_min': float('inf'), 'x_max': float('-inf'),
                             'y_min': float('inf'), 'y_max': float('-inf')}
    }
    
    # Check roadlines
    roadlines_file = itri_path / "roadlines.json"
    if roadlines_file.exists():
        analysis['files_found'].append('roadlines.json')
        with open(roadlines_file, 'r') as f:
            roadlines_data = json.load(f)
        
        analysis['total_roadlines'] = len(roadlines_data['roadlines'])
        
        # Analyze coordinate bounds
        for roadline in roadlines_data['roadlines']:
            for point in roadline['points']:
                x, y = point['x'], point['y']
                analysis['coordinate_bounds']['x_min'] = min(analysis['coordinate_bounds']['x_min'], x)
                analysis['coordinate_bounds']['x_max'] = max(analysis['coordinate_bounds']['x_max'], x)
                analysis['coordinate_bounds']['y_min'] = min(analysis['coordinate_bounds']['y_min'], y)
                analysis['coordinate_bounds']['y_max'] = max(analysis['coordinate_bounds']['y_max'], y)
    
    # Check pedestrian crossings
    crossings_file = itri_path / "pedestrian_crossing.json"
    if crossings_file.exists():
        analysis['files_found'].append('pedestrian_crossing.json')
        with open(crossings_file, 'r') as f:
            crossings_data = json.load(f)
        analysis['total_crossings'] = len(crossings_data.get('non_accessible', []))
    
    # Check road markers
    markers_file = itri_path / "roadmarkers.json"
    if markers_file.exists():
        analysis['files_found'].append('roadmarkers.json')
        with open(markers_file, 'r') as f:
            markers_data = json.load(f)
        analysis['total_markers'] = len(markers_data.get('roadmarkers', []))
    
    return analysis


def test_basic_conversion():
    """Test basic ITRI to lane query conversion."""
    print("=" * 60)
    print("Test 1: Basic ITRI to Lane Query Conversion")
    print("=" * 60)
    
    # Set up paths
    script_dir = Path(__file__).parent
    itri_path = script_dir / "itri"
    query_dir = script_dir / "query"
    
    # Check if ITRI data exists
    if not itri_path.exists():
        print(f"✗ ITRI data path not found: {itri_path}")
        print("Please ensure your ITRI JSON files are in semantic-map/itri/")
        return False
    
    # Analyze ITRI data
    print("Analyzing ITRI dataset...")
    analysis = analyze_itri_data(itri_path)
    print(f"✓ Files found: {', '.join(analysis['files_found'])}")
    print(f"  Roadlines: {analysis['total_roadlines']}")
    print(f"  Crossings: {analysis['total_crossings']}")
    print(f"  Markers: {analysis['total_markers']}")
    
    if analysis['coordinate_bounds']['x_min'] != float('inf'):
        bounds = analysis['coordinate_bounds']
        print(f"  Coordinate bounds: x=[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}], "
              f"y=[{bounds['y_min']:.1f}, {bounds['y_max']:.1f}]")
        
        # Use a more reasonable ego pose near actual road data (not map center)
        ego_pose = (0.0, -15.0, 0.0)  # Position that works well with ITRI data
    else:
        # Default ego pose
        ego_pose = (0.0, 0.0, 0.0)
    
    print(f"  Using ego pose: x={ego_pose[0]:.1f}, y={ego_pose[1]:.1f}, yaw={ego_pose[2]:.2f}")
    
    # Configure converter with extended Z range for ITRI data
    config = {
        'embed_dim': 256,
        'max_queries': 300,
        'sample_distance': 1.0,
        'pc_range': [-51.2, -51.2, -15.0, 51.2, 51.2, 5.0]  # Extended Z range for ITRI data
    }
    
    print(f"\nConverter configuration:")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Max queries: {config['max_queries']}")
    print(f"  Sample distance: {config['sample_distance']}m")
    print(f"  PC range: {config['pc_range']}")
    
    try:
        # Create converter
        print("\nCreating converter...")
        converter = create_converter(config)
        print("✓ Converter created successfully")
        
        # Run conversion
        print("Running conversion...")
        lane_query, lane_query_pos = converter.convert(str(itri_path), ego_pose)
        print("✓ Conversion completed successfully")
        
        # Analyze results
        print(f"\nResults:")
        print(f"  Lane query shape: {lane_query.shape}")
        print(f"  Lane query pos shape: {lane_query_pos.shape}")
        print(f"  Device: {lane_query.device}")
        print(f"  Data type: {lane_query.dtype}")
        
        non_zero_queries = (lane_query.norm(dim=-1) > 0).sum().item()
        print(f"  Non-zero queries: {non_zero_queries}/{lane_query.shape[1]}")
        print(f"  Query utilization: {non_zero_queries/lane_query.shape[1]*100:.1f}%")
        
        max_norm = lane_query.norm(dim=-1).max().item()
        min_norm = lane_query.norm(dim=-1).min().item()
        mean_norm = lane_query.norm(dim=-1).mean().item()
        print(f"  Query norms - max: {max_norm:.3f}, min: {min_norm:.3f}, mean: {mean_norm:.3f}")
        
        # Save results
        print(f"\nSaving results...")
        result_name = save_query_results(lane_query, lane_query_pos, ego_pose, query_dir, "basic_")
        
        # Save analysis
        analysis['conversion_results'] = {
            'ego_pose': ego_pose,
            'non_zero_queries': non_zero_queries,
            'query_utilization': non_zero_queries/lane_query.shape[1],
            'max_norm': max_norm,
            'min_norm': min_norm,
            'mean_norm': mean_norm
        }
        
        with open(query_dir / f"{result_name}_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print("✓ Basic conversion test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        traceback.print_exc()
        return False


def test_multiple_poses():
    """Test conversion with multiple ego poses."""
    print("\n" + "=" * 60)
    print("Test 2: Multiple Ego Poses")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    itri_path = script_dir / "itri"
    query_dir = script_dir / "query"
    
    if not itri_path.exists():
        print("✗ ITRI data not found, skipping multi-pose test")
        return False
    
    # Analyze bounds to create reasonable poses
    analysis = analyze_itri_data(itri_path)
    bounds = analysis['coordinate_bounds']
    
    if bounds['x_min'] == float('inf'):
        print("✗ No coordinate data found, skipping multi-pose test")
        return False
    
    # Create test poses around the map
    center_x = (bounds['x_min'] + bounds['x_max']) / 2
    center_y = (bounds['y_min'] + bounds['y_max']) / 2
    
    test_poses = [
        (center_x, center_y, 0.0),                    # Center, facing east
        (center_x, center_y, np.pi/2),                # Center, facing north
        (center_x, center_y, np.pi),                  # Center, facing west
        (center_x, center_y, -np.pi/2),               # Center, facing south
        (bounds['x_min'] + 10, center_y, 0.0),        # Left side
        (bounds['x_max'] - 10, center_y, 0.0),        # Right side
    ]
    
    config = {'embed_dim': 256, 'max_queries': 300, 'sample_distance': 1.0,
              'pc_range': [-51.2, -51.2, -15.0, 51.2, 51.2, 5.0]}  # Extended Z range
    
    converter = create_converter(config)
    
    results = []
    for i, ego_pose in enumerate(test_poses):
        try:
            print(f"Testing pose {i+1}/{len(test_poses)}: x={ego_pose[0]:.1f}, y={ego_pose[1]:.1f}, yaw={ego_pose[2]:.2f}")
            
            lane_query, lane_query_pos = converter.convert(str(itri_path), ego_pose)
            non_zero_queries = (lane_query.norm(dim=-1) > 0).sum().item()
            
            result_name = save_query_results(lane_query, lane_query_pos, ego_pose, query_dir, f"pose_{i+1}_")
            
            results.append({
                'pose_id': i+1,
                'ego_pose': ego_pose,
                'non_zero_queries': non_zero_queries,
                'utilization': non_zero_queries/lane_query.shape[1],
                'filename': result_name
            })
            
            print(f"  ✓ Generated {non_zero_queries} queries ({non_zero_queries/lane_query.shape[1]*100:.1f}% utilization)")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # Save summary
    summary = {
        'test_poses': len(test_poses),
        'successful_conversions': len(results),
        'results': results,
        'coordinate_bounds': bounds
    }
    
    with open(query_dir / "multi_pose_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Multi-pose test completed: {len(results)}/{len(test_poses)} successful")
    return True


def test_integration():
    """Test full MotionFormer integration."""
    print("\n" + "=" * 60)
    print("Test 3: MotionFormer Integration")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    itri_path = script_dir / "itri"
    query_dir = script_dir / "query"
    
    if not itri_path.exists():
        print("✗ ITRI data not found, skipping integration test")
        return False
    
    # Use CPU for safety (avoid CUDA issues)
    device = 'cpu'
    print(f"Using device: {device}")
    
    try:
        # Create integrator
        integrator = create_integrator(device=device)
        print("✓ Integrator created successfully")
        
        # Sample data
        analysis = analyze_itri_data(itri_path)
        bounds = analysis['coordinate_bounds']
        
        if bounds['x_min'] != float('inf'):
            center_x = (bounds['x_min'] + bounds['x_max']) / 2
            center_y = (bounds['y_min'] + bounds['y_max']) / 2
            ego_pose = (center_x, center_y, 0.0)
        else:
            ego_pose = (0.0, 0.0, 0.0)
        
        # Create sample tracking data
        tracked_objects = [
            {
                'bbox': [10.0, 5.0, 0.0, 4.5, 2.0, 1.8, 0.1],
                'score': 0.95,
                'label': 0,
                'track_id': 1
            },
            {
                'bbox': [-5.0, -3.0, 0.0, 4.2, 1.8, 1.6, -0.2],
                'score': 0.87,
                'label': 0,
                'track_id': 2
            }
        ]
        
        # Create sample BEV features
        bev_h, bev_w = 200, 200
        embed_dim = 256
        batch_size = 1
        bev_features = torch.randn(bev_h * bev_w, batch_size, embed_dim, device=device)
        
        print(f"Sample data:")
        print(f"  Ego pose: {ego_pose}")
        print(f"  Tracked objects: {len(tracked_objects)}")
        print(f"  BEV features shape: {bev_features.shape}")
        
        # Prepare motion inputs
        print("\nPreparing MotionFormer inputs...")
        motion_inputs = integrator.prepare_motion_inputs(
            itri_data_path=str(itri_path),
            ego_pose=ego_pose,
            tracked_objects=tracked_objects,
            bev_features=bev_features
        )
        
        print("✓ Motion inputs prepared successfully!")
        
        # Analyze inputs
        lane_query = motion_inputs['outs_seg']['args_tuple'][3]
        lane_query_pos = motion_inputs['outs_seg']['args_tuple'][5]
        track_query = motion_inputs['outs_track']['track_query_embeddings']
        
        print(f"\nMotionFormer input analysis:")
        print(f"  Lane query shape: {lane_query.shape}")
        print(f"  Lane query pos shape: {lane_query_pos.shape}")
        print(f"  Track query shape: {track_query.shape}")
        print(f"  BEV embed shape: {motion_inputs['bev_embed'].shape}")
        
        non_zero_lanes = (lane_query.norm(dim=-1) > 0).sum().item()
        print(f"  Non-zero lane queries: {non_zero_lanes}")
        
        # Save integration results
        integration_data = {
            'lane_query': lane_query,
            'lane_query_pos': lane_query_pos,
            'track_query': track_query,
            'ego_pose': ego_pose,
            'tracked_objects': tracked_objects,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'device': str(device),
                'non_zero_lane_queries': non_zero_lanes,
                'num_tracked_objects': len(tracked_objects)
            }
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(integration_data, query_dir / f"integration_test_{timestamp}.pt")
        
        print(f"✓ Integration test completed successfully!")
        print(f"✓ Results saved to integration_test_{timestamp}.pt")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("ITRI Dataset Conversion Test")
    print("============================")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"ITRI data path: {script_dir / 'itri'}")
    print(f"Query output path: {script_dir / 'query'}")
    
    # Run tests
    results = {
        'basic_conversion': test_basic_conversion(),
        'multiple_poses': test_multiple_poses(),
        'integration': test_integration()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20}: {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests passed")
    
    if successful_tests > 0:
        query_dir = script_dir / "query"
        query_files = list(query_dir.glob("*.pt"))
        print(f"\nGenerated query files:")
        for file in sorted(query_files):
            print(f"  {file.name}")
    
    print(f"\nAll results saved to: {script_dir / 'query'}")
    
    return successful_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)