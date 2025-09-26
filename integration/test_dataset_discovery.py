#!/usr/bin/env python3
"""
Test script to validate dataset discovery and alignment functionality
without requiring torch or the full inference pipeline.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/integration')

# Import only the dataset discovery classes (no torch required)
from phase5_uniad_inference import DatasetDiscovery, FrameInfo

def test_dataset_discovery():
    """Test dataset discovery and alignment functionality."""
    print("🔍 Testing Dataset Discovery for HCT 2025-08-06-14-23-05_0")
    print("=" * 60)
    
    try:
        # Initialize dataset discovery
        discovery = DatasetDiscovery()
        print(f"✅ Dataset discovery initialized")
        print(f"📁 Camera base: {discovery.camera_base}")
        print(f"📄 CAN bus file: {discovery.can_bus_file}")
        print(f"🗺️  Semantic map dir: {discovery.semantic_map_dir}")
        print(f"🎯 Track queries file: {discovery.track_queries_file}")
        
        # Check if required paths exist
        missing_paths = []
        if not discovery.camera_base.exists():
            missing_paths.append(f"Camera directory: {discovery.camera_base}")
        if not discovery.can_bus_file.exists():
            missing_paths.append(f"CAN bus file: {discovery.can_bus_file}")
        if not discovery.semantic_map_dir.exists():
            missing_paths.append(f"Semantic map directory: {discovery.semantic_map_dir}")
        if not discovery.track_queries_file.exists():
            missing_paths.append(f"Track queries file: {discovery.track_queries_file}")
            
        if missing_paths:
            print("❌ Missing required paths:")
            for path in missing_paths:
                print(f"   {path}")
            return False
            
        print("✅ All required paths exist")
        
        # Test timestamp extraction
        print("\n📅 Testing timestamp extraction...")
        test_filenames = [
            'f100_142305_1.jpg',
            'f100_142455_35.jpg', 
            'f100_143115_9.jpg'
        ]
        
        for filename in test_filenames:
            timestamp = discovery.extract_timestamp_from_filename(filename)
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                print(f"   {filename} → {dt} ({timestamp})")
            else:
                print(f"   {filename} → Failed to parse")
                
        # Discover camera frames (without loading CAN bus data)
        print("\n📸 Discovering camera frames...")
        camera_frames = discovery.discover_camera_frames()
        
        total_frames = 0
        for camera_name, frames in camera_frames.items():
            total_frames += len(frames)
            print(f"   {camera_name}: {len(frames)} frames")
            if frames:
                first_dt = datetime.fromtimestamp(frames[0][0])
                last_dt = datetime.fromtimestamp(frames[-1][0])
                print(f"      Range: {first_dt} → {last_dt}")
                
        print(f"📊 Total frames in alignment window: {total_frames}")
        
        # Test frame alignment (without CAN bus data)
        print("\n🔄 Testing frame alignment across cameras...")
        aligned_frames = discovery.align_frames_across_cameras(camera_frames)
        print(f"✅ Created {len(aligned_frames)} synchronized frames")
        
        if aligned_frames:
            print("Sample aligned frames:")
            for i, frame_info in enumerate(aligned_frames[:3]):
                print(f"   Frame {i}: {frame_info}")
                print(f"      Cameras: {list(frame_info.camera_paths.keys())}")
                
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_discovery()
    if success:
        print("\n🎉 Dataset discovery test completed successfully!")
    else:
        print("\n❌ Dataset discovery test failed!")