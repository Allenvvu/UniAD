#!/usr/bin/env python3
"""
ROS Bag to JPG Image Extraction for HCT Training Data (FIXED VERSION)
Properly convert H.265 compressed video data from ROS bags to JPG images
Starting with: 2025-08-06-14-23-05_0.bag
"""

import os
import sys
import rosbag
import cv2
import numpy as np
import tempfile
import subprocess
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import time

# Camera topics to extract (4-camera setup)
CAMERA_TOPICS = [
    "/lucid_cameras_x00/gige_100_f_hdr/h265",    # front_100
    "/lucid_cameras_x00/gige_60_b_hdr/h265",     # back_60
    "/lucid_cameras_x01/gige_100_fl_hdr/h265",   # frontleft_100
    "/lucid_cameras_x01/gige_100_fr_hdr/h265"    # frontright_100
]

# Output folder mappings
OUTPUT_FOLDERS = {
    "/lucid_cameras_x00/gige_100_f_hdr/h265": "front_100",
    "/lucid_cameras_x00/gige_60_b_hdr/h265": "back_60",
    "/lucid_cameras_x01/gige_100_fl_hdr/h265": "frontleft_100",
    "/lucid_cameras_x01/gige_100_fr_hdr/h265": "frontright_100"
}

# Camera prefixes for file naming
CAMERA_PREFIXES = {
    "front_100": "f100",
    "back_60": "b60", 
    "frontleft_100": "fl100",
    "frontright_100": "fr100"
}

def extract_time_from_bag_filename(bag_filename):
    """Extract timestamp from bag filename like 2025-08-06-14-23-05_0.bag"""
    import re
    match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(\d+)', bag_filename)
    if match:
        timestamp = match.group(1)  # 2025-08-06-14-23-05
        bag_num = match.group(2)    # 0
        # Convert to HHMMSS format
        time_parts = timestamp.split('-')
        if len(time_parts) >= 6:
            time_str = time_parts[3] + time_parts[4] + time_parts[5]  # 142305
            return time_str, bag_num
    return None, None

def convert_h265_frame_to_jpg_ffmpeg(h265_data, output_path):
    """Convert single H.265 frame to JPG using ffmpeg"""
    try:
        # Handle different data formats
        if isinstance(h265_data, list):
            frame_data = bytes(h265_data)
        else:
            frame_data = h265_data
        
        # Write H.265 data to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as temp_h265:
            temp_h265.write(frame_data)
            temp_h265_path = temp_h265.name
        
        try:
            # Use ffmpeg to decode H.265 and save as JPG
            cmd = [
                'ffmpeg',
                '-f', 'hevc',  # Input format is H.265/HEVC
                '-i', temp_h265_path,
                '-frames:v', '1',  # Extract only first frame
                '-q:v', '2',       # High quality
                '-y',              # Overwrite existing
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                return False
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_h265_path):
                os.remove(temp_h265_path)
        
    except Exception as e:
        print(f"      Error in ffmpeg conversion: {e}")
        return False

def convert_h265_data_opencv(h265_data):
    """Try to convert H.265 data using OpenCV (may not work for all H.265 streams)"""
    try:
        # Handle different data formats
        if isinstance(h265_data, list):
            frame_data = bytes(h265_data)
        else:
            frame_data = h265_data
        
        # Try direct decode (works for some compressed formats)
        nparr = np.frombuffer(frame_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            return img
        
        return None
        
    except Exception as e:
        return None

def extract_h265_stream_to_video(bag_file, topic, output_video_path, max_duration=70):
    """Extract H.265 stream from bag and save as temporary video file"""
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            frame_count = 0
            
            with open(output_video_path, 'wb') as video_file:
                for topic_name, msg, t in bag.read_messages(topics=[topic]):
                    if hasattr(msg, 'data'):
                        # Handle different data formats
                        if isinstance(msg.data, list):
                            frame_data = bytes(msg.data)
                        else:
                            frame_data = msg.data
                        
                        video_file.write(frame_data)
                        frame_count += 1
                        
                        # Limit extraction to avoid huge files (70 seconds at ~30 FPS)
                        if frame_count >= max_duration * 30:  # Assume ~30 FPS
                            break
            
            return frame_count > 0
    
    except Exception as e:
        print(f"      Error extracting H.265 stream: {e}")
        return False

def extract_frames_from_h265_video(video_path, output_dir, camera_prefix, time_str, max_frames=150):
    """Extract JPG frames from H.265 video using ffmpeg"""
    try:
        # Use ffmpeg to extract frames at 2 FPS
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', 'fps=2',  # 2 FPS as specified
            '-q:v', '2',     # High quality
            '-frames:v', str(max_frames),  # Limit frames
            '-y',            # Overwrite existing
            os.path.join(output_dir, f"{camera_prefix}_{time_str}_%03d.jpg")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            # Count generated JPG files
            jpg_files = [f for f in os.listdir(output_dir) 
                        if f.startswith(f"{camera_prefix}_{time_str}_") and f.endswith('.jpg')]
            return len(jpg_files)
        else:
            print(f"      FFmpeg error: {result.stderr}")
            return 0
            
    except Exception as e:
        print(f"      Error extracting frames: {e}")
        return 0

def extract_images_from_bag(bag_file, output_base_dir, target_fps=2, max_frames_per_topic=150):
    """Extract JPG images from bag file using proper H.265 decoding"""
    
    print(f"Processing {bag_file}...")
    
    if not os.path.exists(bag_file):
        print(f"  Error: {bag_file} not found")
        return {}
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Error: ffmpeg not found. Please install ffmpeg first.")
        return {}
    
    # Extract timing info from filename
    time_str, bag_num = extract_time_from_bag_filename(os.path.basename(bag_file))
    if not time_str:
        print(f"  Warning: Cannot extract time from filename")
        time_str = "unknown"
        bag_num = "0"
    
    print(f"  Time: {time_str}, Bag: {bag_num}")
    print(f"  Max frames per camera: {max_frames_per_topic}")
    
    extraction_stats = {}
    
    try:
        with rosbag.Bag(bag_file, 'r') as bag:
            # Get bag info
            info = bag.get_type_and_topic_info()
            available_topics = set(info[1].keys())
            
            print(f"  Available topics: {len(available_topics)}")
            
            for topic in CAMERA_TOPICS:
                if topic not in available_topics:
                    print(f"  Warning: Topic {topic} not found in bag")
                    continue
                
                # Setup output directory
                folder_name = OUTPUT_FOLDERS[topic]
                camera_prefix = CAMERA_PREFIXES[folder_name]
                output_dir = os.path.join(output_base_dir, folder_name)
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"    Extracting {topic} → {folder_name}/")
                
                # Extract H.265 stream to temporary video file
                with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as temp_video:
                    temp_video_path = temp_video.name
                
                print(f"      Creating temporary H.265 video...")
                if extract_h265_stream_to_video(bag_file, topic, temp_video_path):
                    print(f"      Extracting frames using ffmpeg...")
                    frame_count = extract_frames_from_h265_video(
                        temp_video_path, output_dir, camera_prefix, time_str, max_frames_per_topic
                    )
                    
                    if frame_count > 0:
                        print(f"    ✓ Extracted {frame_count} JPG images")
                        extraction_stats[folder_name] = frame_count
                    else:
                        print(f"    ✗ Failed to extract images")
                        extraction_stats[folder_name] = 0
                else:
                    print(f"    ✗ Failed to create H.265 video stream")
                    extraction_stats[folder_name] = 0
                
                # Clean up temporary video file
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
    
    except Exception as e:
        print(f"  Error processing {bag_file}: {e}")
        return {}
    
    return extraction_stats

def main():
    # Configuration
    bag_files_dir = "bag_files"
    output_dir = "images"  # Final output directory
    target_bag = "2025-08-06-14-23-05_0.bag"
    target_fps = 2  # Extract at 2 FPS as specified in plan
    
    print("=" * 60)
    print("ROS Bag to JPG Image Extraction (FIXED VERSION)")
    print("=" * 60)
    print(f"Bag files directory: {bag_files_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target bag file: {target_bag}")
    print(f"Extraction rate: {target_fps} FPS")
    print(f"Method: H.265 stream → ffmpeg → JPG frames")
    print()
    
    # Check dependencies
    try:
        import rosbag, cv2
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("  sudo apt-get install python3-rosbag")
        sys.exit(1)
    
    # Verify directories
    if not os.path.exists(bag_files_dir):
        print(f"Error: Directory {bag_files_dir} not found")
        sys.exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create camera subdirectories
    for folder_name in OUTPUT_FOLDERS.values():
        camera_dir = os.path.join(output_dir, folder_name)
        os.makedirs(camera_dir, exist_ok=True)
    
    # Target bag file path
    bag_file_path = os.path.join(bag_files_dir, target_bag)
    
    if not os.path.exists(bag_file_path):
        print(f"Error: Target bag file {bag_file_path} not found")
        sys.exit(1)
    
    print(f"Processing single bag file: {target_bag}")
    print(f"File size: {os.path.getsize(bag_file_path) / (1024**3):.2f} GB")
    print()
    
    # Process the bag file
    start_time = time.time()
    stats = extract_images_from_bag(bag_file_path, output_dir, target_fps, max_frames_per_topic=150)
    end_time = time.time()
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Target FPS: {target_fps}")
    print()
    
    total_images = 0
    for folder_name, count in stats.items():
        print(f"{folder_name}/: {count} images")
        total_images += count
        
        # Show example files
        folder_path = os.path.join(output_dir, folder_name)
        if os.path.exists(folder_path):
            jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
            if jpg_files:
                example_files = sorted(jpg_files)[:3]
                for ex in example_files:
                    print(f"    - {ex}")
    
    print(f"\nTotal images extracted: {total_images}")
    print(f"Output directory: {output_dir}/")
    print(f"\nFile naming format: [camera_prefix]_[HHMMSS]_[frame].jpg")
    print(f"Example: f100_142305_001.jpg (front camera, 14:23:05, frame 1)")
    
    # Verify one image is a proper JPEG
    if total_images > 0:
        print("\nVerifying image format...")
        for folder_name in stats:
            if stats[folder_name] > 0:
                folder_path = os.path.join(output_dir, folder_name)
                jpg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]
                if jpg_files:
                    sample_file = os.path.join(folder_path, jpg_files[0])
                    try:
                        with open(sample_file, 'rb') as f:
                            header = f.read(2)
                            if header == b'\xff\xd8':  # JPEG magic bytes
                                print(f"✓ {jpg_files[0]} is a valid JPEG file")
                            else:
                                print(f"✗ {jpg_files[0]} is not a valid JPEG (header: {header.hex()})")
                    except Exception as e:
                        print(f"Error checking {jpg_files[0]}: {e}")
                    break

if __name__ == "__main__":
    main()