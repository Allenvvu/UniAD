#!/usr/bin/env python3
"""
HTC Logistic Dataset Conversion Script

This script runs the HTC logistic to lane query conversion and saves the results 
to the query folder.

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
    base_name = f"{prefix}hct_lane_query_{timestamp}" if prefix else f"hct_lane_query_{timestamp}"
    
    # Save tensors
    torch.save({
        'lane_query': lane_query,
        'lane_query_pos': lane_query_pos,
        'ego_pose': ego_pose,
        'dataset': 'hct_logistic',
        'metadata': {
            'timestamp': timestamp,
            'lane_query_shape': list(lane_query.shape),
            'lane_query_pos_shape': list(lane_query_pos.shape),
            'ego_pose': ego_pose,
            'device': str(lane_query.device),
            'dtype': str(lane_query.dtype),
            'dataset_type': 'hct_logistic'
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
        'dataset': 'hct_logistic',
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
    
    print(f"✓ Saved HTC results to {output_dir}/{base_name}.*")
    return base_name


def analyze_hct_data():
    """
    Analyze the HTC logistic dataset structure.
    
    Returns:
        Dictionary with analysis results
    """
    data_path = Path("data/hct_logistic")
    analysis = {
        'files_found': [],
        'total_roadlines': 0,
        'total_crossings': 0,
        'total_markers': 0,
        'coordinate_bounds': {'x_min': float('inf'), 'x_max': float('-inf'),
                             'y_min': float('inf'), 'y_max': float('-inf'),
                             'z_min': float('inf'), 'z_max': float('-inf')}
    }
    
    # Check roadlines
    roadlines_file = data_path / "roadlines.json"
    if roadlines_file.exists():
        analysis['files_found'].append('roadlines.json')
        with open(roadlines_file, 'r') as f:
            roadlines_data = json.load(f)
        
        analysis['total_roadlines'] = len(roadlines_data['roadlines'])
        
        # Analyze coordinate bounds
        for roadline in roadlines_data['roadlines']:
            for point in roadline['points']:
                x, y, z = point['x'], point['y'], point['z']
                analysis['coordinate_bounds']['x_min'] = min(analysis['coordinate_bounds']['x_min'], x)
                analysis['coordinate_bounds']['x_max'] = max(analysis['coordinate_bounds']['x_max'], x)
                analysis['coordinate_bounds']['y_min'] = min(analysis['coordinate_bounds']['y_min'], y)
                analysis['coordinate_bounds']['y_max'] = max(analysis['coordinate_bounds']['y_max'], y)
                analysis['coordinate_bounds']['z_min'] = min(analysis['coordinate_bounds']['z_min'], z)
                analysis['coordinate_bounds']['z_max'] = max(analysis['coordinate_bounds']['z_max'], z)
    
    # Check other files
    for filename in ['pedestrian_crossing.json', 'roadmarkers.json', 'lanes_info.json']:
        file_path = data_path / filename
        if file_path.exists():
            analysis['files_found'].append(filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                if filename == 'pedestrian_crossing.json':
                    analysis['total_crossings'] = len(data.get('non_accessible', []))
                elif filename == 'roadmarkers.json':
                    analysis['total_markers'] = len(data.get('roadmarkers', []))
    
    return analysis


def test_hct_conversion():
    """Test HTC logistic to lane query conversion."""
    print("=" * 60)
    print("HTC Logistic Dataset Conversion")
    print("=" * 60)
    
    # Set up paths
    script_dir = Path(__file__).parent
    hct_path = script_dir / "data" / "hct_logistic"
    query_dir = script_dir / "query"
    
    # Check if HTC data exists
    if not hct_path.exists():
        print(f"✗ HTC data path not found: {hct_path}")
        print("Please ensure your HTC JSON files are in semantic-map/data/hct_logistic/")
        return False
    
    # Analyze HTC data
    print("Analyzing HTC dataset...")
    analysis = analyze_hct_data()
    print(f"✓ Files found: {', '.join(analysis['files_found'])}")
    print(f"  Roadlines: {analysis['total_roadlines']}")
    print(f"  Crossings: {analysis['total_crossings']}")
    print(f"  Markers: {analysis['total_markers']}")
    
    bounds = analysis['coordinate_bounds']
    if bounds['x_min'] != float('inf'):
        print(f"  Coordinate bounds: x=[{bounds['x_min']:.1f}, {bounds['x_max']:.1f}], "
              f"y=[{bounds['y_min']:.1f}, {bounds['y_max']:.1f}], z=[{bounds['z_min']:.1f}, {bounds['z_max']:.1f}]")
        
        # Calculate reasonable ego poses
        center_x = (bounds['x_min'] + bounds['x_max']) / 2
        center_y = (bounds['y_min'] + bounds['y_max']) / 2
        center_z = (bounds['z_min'] + bounds['z_max']) / 2
        
        test_poses = [
            (center_x, center_y, 0.0),                    # Center, facing east
            (center_x, center_y, np.pi/2),                # Center, facing north
            (300.0, -50.0, 0.0),                          # Good performance pose from testing
        ]
    else:
        test_poses = [(300.0, -50.0, 0.0)]  # Default pose
    
    # Configure converter for HTC data range
    config = {
        'embed_dim': 256,
        'max_queries': 300,
        'sample_distance': 1.0,
        'pc_range': [-100.0, -450.0, 35.0, 720.0, 280.0, 55.0]  # Extended range for HTC data
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
        
        # Test multiple poses
        results = []
        for i, ego_pose in enumerate(test_poses):
            print(f"\nTesting pose {i+1}/{len(test_poses)}: x={ego_pose[0]:.1f}, y={ego_pose[1]:.1f}, yaw={ego_pose[2]:.2f}")
            
            # Run conversion
            lane_query, lane_query_pos = converter.convert(str(hct_path), ego_pose)
            
            # Analyze results
            non_zero_queries = (lane_query.norm(dim=-1) > 0).sum().item()
            utilization = non_zero_queries/lane_query.shape[1]*100
            
            print(f"  Lane query shape: {lane_query.shape}")
            print(f"  Non-zero queries: {non_zero_queries}/{lane_query.shape[1]} ({utilization:.1f}%)")
            
            if non_zero_queries > 0:
                max_norm = lane_query.norm(dim=-1).max().item()
                min_norm = lane_query.norm(dim=-1).min().item()
                mean_norm = lane_query.norm(dim=-1).mean().item()
                print(f"  Query norms - max: {max_norm:.3f}, min: {min_norm:.3f}, mean: {mean_norm:.3f}")
            
            # Save results
            result_name = save_query_results(lane_query, lane_query_pos, ego_pose, query_dir, f"pose_{i+1}_")
            
            results.append({
                'pose_id': i+1,
                'ego_pose': ego_pose,
                'non_zero_queries': non_zero_queries,
                'utilization': utilization,
                'filename': result_name
            })
        
        # Save summary
        summary = {
            'dataset': 'hct_logistic',
            'test_poses': len(test_poses),
            'successful_conversions': len(results),
            'results': results,
            'coordinate_bounds': bounds,
            'config': config,
            'analysis': analysis
        }
        
        with open(query_dir / "hct_conversion_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ HTC conversion completed successfully!")
        print(f"✓ {len(results)} conversions saved")
        print(f"✓ Best utilization: {max(r['utilization'] for r in results):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ HTC conversion failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("HTC Logistic Dataset Conversion")
    print("===============================")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print(f"Working directory: {os.getcwd()}")
    print(f"HTC data path: {script_dir / 'data' / 'hct_logistic'}")
    print(f"Query output path: {script_dir / 'query'}")
    
    # Run conversion
    success = test_hct_conversion()
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    
    status = "✓ SUCCESS" if success else "✗ FAILED"
    print(f"HTC Conversion: {status}")
    
    if success:
        query_dir = script_dir / "query"
        hct_files = list(query_dir.glob("*hct_lane_query*.pt"))
        print(f"\nGenerated HTC query files:")
        for file in sorted(hct_files):
            print(f"  {file.name}")
    
    print(f"\nAll results saved to: {script_dir / 'query'}")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)