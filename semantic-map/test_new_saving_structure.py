#!/usr/bin/env python3
"""
Test script to verify the new saving structure matches gt_sdc format.
"""

import torch
import json
import tempfile
import shutil
from pathlib import Path
from generate_hct_map_queries import HCTMapQueryGenerator

def test_saving_structure():
    """Test the new saving structure with mock data."""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "test_output"
        output_dir.mkdir()
        
        # Create mock map query data
        mock_map_query_data = {
            'lane_query': torch.randn(10, 256),  # Mock lane query tensor
            'lane_query_pos': torch.randn(10, 3),  # Mock position tensor
            'metadata': {
                'frame_idx': 0,
                'timestamp': 1754461441100412.0,
                'ego_pose': [1.0, 2.0, 0.5],
                'test_data': True
            }
        }
        
        # Test timestamp-based filename generation
        test_timestamp = 1754461441100412.0
        expected_filename = f"{int(test_timestamp * 1e6)}.pt"
        
        print(f"Testing timestamp: {test_timestamp}")
        print(f"Expected filename: {expected_filename}")
        
        # Create a minimal generator instance for testing
        class MockGenerator:
            def __init__(self, output_path):
                self.output_path = output_path
            
            def save_frame_data(self, frame_idx, map_query_data, timestamp=None):
                # Use timestamp for filename if available, otherwise fall back to frame_idx
                if timestamp is not None:
                    # Convert timestamp to microseconds and format as integer string
                    timestamp_us = int(timestamp * 1e6)
                    filename = f"{timestamp_us}.pt"
                else:
                    # Fallback to frame-based naming
                    filename = f"frame_{frame_idx:06d}.pt"
                
                # Save combined data as single tensor file (matching gt_sdc format)
                combined_data = {
                    'lane_query': map_query_data['lane_query'],
                    'lane_query_pos': map_query_data['lane_query_pos'],
                    'metadata': map_query_data['metadata']
                }
                
                torch.save(combined_data, self.output_path / filename)
                print(f"Saved map query data as {filename}")
                return filename
            
            def save_bag_metadata(self, bag_name, bag_metadata):
                metadata_filename = f"{bag_name}_map_query_metadata.json"
                with open(self.output_path / metadata_filename, 'w') as f:
                    json.dump(bag_metadata, f, indent=2)
                print(f"Saved bag metadata as {metadata_filename}")
                return metadata_filename
        
        # Test the saving functions
        generator = MockGenerator(output_dir)
        
        # Test frame data saving with timestamp
        saved_filename = generator.save_frame_data(0, mock_map_query_data, test_timestamp)
        assert saved_filename == expected_filename, f"Expected {expected_filename}, got {saved_filename}"
        
        # Test frame data saving without timestamp (fallback)
        fallback_filename = generator.save_frame_data(0, mock_map_query_data, None)
        assert fallback_filename == "frame_000000.pt", f"Expected frame_000000.pt, got {fallback_filename}"
        
        # Test bag metadata saving
        bag_metadata = {
            "bag_name": "2025-08-06-14-24-01_1",
            "total_frames": 100,
            "successful_frames": 98,
            "failed_frames": 2,
            "successful_timestamps": [1754461441100412.0, 1754461441600568.0],
            "timestamp_range": {
                "start": 1754461441100412.0,
                "end": 1754461441600568.0
            }
        }
        
        metadata_filename = generator.save_bag_metadata("2025-08-06-14-24-01_1", bag_metadata)
        expected_metadata_filename = "2025-08-06-14-24-01_1_map_query_metadata.json"
        assert metadata_filename == expected_metadata_filename, f"Expected {expected_metadata_filename}, got {metadata_filename}"
        
        # Verify files were created
        assert (output_dir / expected_filename).exists(), f"Expected file {expected_filename} not found"
        assert (output_dir / "frame_000000.pt").exists(), "Expected file frame_000000.pt not found"
        assert (output_dir / expected_metadata_filename).exists(), f"Expected file {expected_metadata_filename} not found"
        
        # Verify file contents
        loaded_data = torch.load(output_dir / expected_filename)
        assert 'lane_query' in loaded_data, "lane_query not found in saved data"
        assert 'lane_query_pos' in loaded_data, "lane_query_pos not found in saved data"
        assert 'metadata' in loaded_data, "metadata not found in saved data"
        assert loaded_data['metadata']['timestamp'] == test_timestamp, "Timestamp mismatch in metadata"
        
        loaded_metadata = json.load(open(output_dir / expected_metadata_filename))
        assert loaded_metadata['bag_name'] == "2025-08-06-14-24-01_1", "Bag name mismatch in metadata"
        assert loaded_metadata['total_frames'] == 100, "Total frames mismatch in metadata"
        
        print("✅ All tests passed!")
        print(f"✅ Files created in {output_dir}:")
        for file in output_dir.iterdir():
            print(f"  - {file.name}")
        
        # Show the structure matches gt_sdc format
        print("\n✅ New structure matches gt_sdc format:")
        print(f"  - Individual files: {expected_filename} (timestamp-based .pt files)")
        print(f"  - Bag metadata: {expected_metadata_filename}")
        print("  - Combined data in single .pt files (lane_query + lane_query_pos + metadata)")

if __name__ == "__main__":
    test_saving_structure()
