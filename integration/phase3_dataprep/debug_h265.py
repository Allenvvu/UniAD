#!/usr/bin/env python3

import rosbag
import numpy as np
import cv2
import os

def debug_h265_extraction(bag_file):
    """Debug H265 image extraction from ROS bag"""
    
    print(f"Debugging H265 extraction from: {bag_file}")
    print("=" * 60)
    
    bag = rosbag.Bag(bag_file)
    
    # Test each camera topic
    camera_topics = [
        '/lucid_cameras_x00/gige_100_f_hdr/h265',  # Front
        '/lucid_cameras_x01/gige_100_fl_hdr/h265', # Front-Left
        '/lucid_cameras_x01/gige_100_fr_hdr/h265', # Front-Right
        '/lucid_cameras_x00/gige_60_b_hdr/h265'    # Back
    ]
    
    for topic in camera_topics:
        print(f"\nTesting topic: {topic}")
        print("-" * 40)
        
        message_count = 0
        for topic_name, msg, timestamp in bag.read_messages(topics=[topic]):
            if message_count >= 3:  # Test first 3 messages
                break
                
            print(f"Message {message_count}:")
            print(f"  Timestamp: {timestamp.to_sec()}")
            print(f"  Message type: {type(msg)}")
            print(f"  Message attributes: {dir(msg)}")
            
            if hasattr(msg, 'data'):
                data_len = len(msg.data)
                print(f"  Data length: {data_len} bytes")
                
                if data_len > 0:
                    # Check first few bytes (H265 signature)
                    first_bytes = msg.data[:20]
                    print(f"  First 20 bytes: {[hex(b) for b in first_bytes]}")
                    
                    # Try to decode with OpenCV
                    image_data = np.frombuffer(msg.data, dtype=np.uint8)
                    decoded_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                    
                    if decoded_image is not None:
                        print(f"  ✅ OpenCV decode successful: {decoded_image.shape}")
                        print(f"     Image stats: min={decoded_image.min()}, max={decoded_image.max()}")
                        
                        # Save test image
                        test_filename = f"test_{topic.replace('/', '_')}_msg{message_count}.jpg"
                        test_path = f"/home/bryan/Desktop/Allen/UniAD/integration/phase3_dataprep/{test_filename}"
                        cv2.imwrite(test_path, decoded_image)
                        print(f"     Saved test image: {test_path}")
                    else:
                        print(f"  ❌ OpenCV decode failed")
                        
                        # Try alternative method - check if it's raw image data
                        print("  Trying alternative decoding methods...")
                        
                        # Check if it could be raw RGB/BGR data
                        total_pixels = data_len // 3
                        if total_pixels > 0:
                            width_candidates = [1920, 1600, 1280, 1024, 800, 640]
                            for width in width_candidates:
                                height = total_pixels // width
                                if width * height * 3 == data_len:
                                    print(f"    Trying raw RGB {width}x{height}...")
                                    raw_image = image_data.reshape((height, width, 3))
                                    test_filename = f"test_raw_{topic.replace('/', '_')}_msg{message_count}_{width}x{height}.jpg"
                                    test_path = f"/home/bryan/Desktop/Allen/UniAD/integration/phase3_dataprep/{test_filename}"
                                    cv2.imwrite(test_path, raw_image)
                                    print(f"      Saved raw test: {test_path}")
                                    break
                else:
                    print(f"  ⚠️  Empty data")
            else:
                print(f"  ❌ No 'data' attribute found")
            
            message_count += 1
        
        if message_count == 0:
            print(f"  ❌ No messages found for topic: {topic}")
    
    bag.close()
    print("\n" + "=" * 60)
    print("H265 DEBUG COMPLETE")

if __name__ == "__main__":
    bag_file = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bag_files/2025-08-06-14-23-05_0.bag"
    debug_h265_extraction(bag_file)