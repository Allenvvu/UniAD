#!/usr/bin/env python3

import rosbag
import numpy as np
import cv2
import pickle
import json
import os
import argparse
import subprocess
import tempfile
from collections import defaultdict
import tf.transformations
from cv_bridge import CvBridge

class ITRIRosBagExtractor:
    """4-Camera ROS Bag Data Extractor for UniAD Training"""
    
    def __init__(self, bag_file, output_dir, semantic_map_dir=None):
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.semantic_map_dir = semantic_map_dir
        
        # 4-camera setup: Front 100°, Front-Left 100°, Front-Right 100°, Back 60°
        self.camera_topics = {
            'CAM_FRONT': '/lucid_cameras_x00/gige_100_f_hdr/h265',        # Front 100°
            'CAM_FRONT_LEFT': '/lucid_cameras_x01/gige_100_fl_hdr/h265',   # Front-Left 100° 
            'CAM_FRONT_RIGHT': '/lucid_cameras_x01/gige_100_fr_hdr/h265',  # Front-Right 100°
            'CAM_BACK': '/lucid_cameras_x00/gige_60_b_hdr/h265'           # Back 60°
        }
        
        # Core ROS topics for extraction
        self.topics = {
            # Sensor data
            'lidar': '/ouster/top_lidar1',
            'cameras': list(self.camera_topics.values()),
            
            # Object detection & tracking
            'objects': '/detected_objects',
            'objects_prediction': '/detected_objects_prediction', 
            'trackers': '/trackers',
            
            # Vehicle state for CAN bus
            'car_state': '/car_state',
            'imu': '/imu/data',
            'vehicle_state': '/vehicle_state',
            'localization': '/ndt_scan_matcher_node/ndt_pose',
            'velocity': '/filter/velocity',
            
            # Motion & planning
            'behavior_state': '/behavior_state',
            'path_reference': '/behavior/path_reference',
            'speed_cmd': '/behavior/speed_cmd',
            'waypoints': '/waypoints',
            
            # Environment
            'traffic_lights': '/traffic_light_status'
        }
        
        # Camera calibration (from bag_to_pkl.py)
        self.camera_calibrations = {
            'CAM_FRONT': {
                'cam_intrinsic': [657.904403, 0.0, 714.717385597, 
                                 0.0, 658.500765392, 463.1351298, 
                                 0.0, 0.0, 1.0],
                'sensor2ego_translation': [0.005, -0.125, -0.171],
                'sensor2ego_rotation': [-0.525, 0.509, -0.491, -0.495]
            },
            'CAM_FRONT_LEFT': {
                'cam_intrinsic': [660.05027892, 0.0, 720.216502034, 
                                 0.0, 660.349325206, 467.66874548, 
                                 0.0, 0.0, 1.0],
                'sensor2ego_translation': [0.213, -0.073, -0.671],
                'sensor2ego_rotation': [0.691, -0.169, 0.162, 0.684]
            },
            'CAM_FRONT_RIGHT': {
                'cam_intrinsic': [659.20444086, 0.0, 713.833685513, 
                                 0.0, 659.864183739, 455.950517493, 
                                 0.0, 0.0, 1.0],
                'sensor2ego_translation': [-0.223, -0.081, -0.654],
                'sensor2ego_rotation': [-0.177, 0.688, -0.682, -0.176]
            },
            'CAM_BACK': {
                'cam_intrinsic': [1033.063670, 0.000000, 730.505500,
                                 0.000000, 1034.542542, 468.282976, 
                                 0.000000, 0.000000, 1.000000],
                'sensor2ego_translation': [-0.011, -0.034, -2.469],
                'sensor2ego_rotation': [0.508, 0.513, -0.486, 0.493]
            }
        }
        
        # Initialize data storage
        self.synchronized_frames = []
        self.extracted_samples = []
        
        # Initialize ROS CvBridge for proper image message handling
        self.bridge = CvBridge()
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/samples", exist_ok=True)
        os.makedirs(f"{self.output_dir}/sweeps", exist_ok=True)
        os.makedirs(f"{self.output_dir}/maps", exist_ok=True)
        
    def extract_synchronized_frames(self):
        """Extract temporally synchronized data at 10Hz (LiDAR frequency)"""
        print(f"Processing {self.bag_file}...")
        
        bag = rosbag.Bag(self.bag_file)
        
        # Collect all topics
        all_topics = [self.topics['lidar']] + self.topics['cameras']
        all_topics.extend([
            self.topics['objects'], self.topics['objects_prediction'], 
            self.topics['trackers'], self.topics['car_state'], 
            self.topics['imu'], self.topics['vehicle_state'],
            self.topics['localization']
        ])
        
        # Storage for message synchronization
        message_buffer = defaultdict(dict)
        lidar_timestamps = []
        
        # First pass: collect LiDAR timestamps
        for topic, msg, t in bag.read_messages(topics=[self.topics['lidar']]):
            lidar_timestamps.append(t.to_sec())
        
        print(f"Found {len(lidar_timestamps)} LiDAR frames")
        
        # Second pass: collect all messages
        for topic, msg, t in bag.read_messages(topics=all_topics):
            timestamp = t.to_sec()
            message_buffer[timestamp][topic] = msg
        
        bag.close()
        
        # Synchronize data to LiDAR timestamps
        tolerance = 0.1  # 100ms tolerance
        
        for lidar_time in lidar_timestamps:
            sync_frame = {'timestamp': lidar_time}
            
            # Find closest messages for each topic
            for topic_key, topic_name in self.topics.items():
                if isinstance(topic_name, list):
                    # Handle camera topics
                    sync_frame[topic_key] = {}
                    for cam_topic in topic_name:
                        closest_msg = self._find_closest_message(
                            message_buffer, cam_topic, lidar_time, tolerance)
                        if closest_msg:
                            sync_frame[topic_key][cam_topic] = closest_msg
                else:
                    # Handle single topics
                    closest_msg = self._find_closest_message(
                        message_buffer, topic_name, lidar_time, tolerance)
                    if closest_msg:
                        sync_frame[topic_key] = closest_msg
            
            # Only keep frames with essential data
            if (self.topics['lidar'] in message_buffer[lidar_time] and
                len(sync_frame.get('cameras', {})) >= 3):  # At least 3 cameras
                self.synchronized_frames.append(sync_frame)
        
        print(f"Synchronized {len(self.synchronized_frames)} frames")
        return self.synchronized_frames
    
    def _find_closest_message(self, message_buffer, topic, target_time, tolerance):
        """Find closest message to target timestamp within tolerance"""
        best_msg = None
        best_diff = float('inf')
        
        for timestamp, messages in message_buffer.items():
            if topic in messages:
                diff = abs(timestamp - target_time)
                if diff < tolerance and diff < best_diff:
                    best_diff = diff
                    best_msg = messages[topic]
        
        return best_msg
    
    def convert_to_nuscenes_format(self, sync_frame, sample_idx):
        """Convert synchronized frame to nuScenes sample format"""
        
        # Generate tokens
        sample_token = f"sample_{sample_idx:06d}"
        scene_token = f"scene_{os.path.basename(self.bag_file).replace('.bag', '')}"
        
        # Extract ego pose from localization
        ego_translation = [0.0, 0.0, 0.0]
        ego_rotation = [1.0, 0.0, 0.0, 0.0]
        
        if 'localization' in sync_frame:
            loc_msg = sync_frame['localization']
            ego_translation = [
                loc_msg.pose.position.x,
                loc_msg.pose.position.y,
                loc_msg.pose.position.z
            ]
            ego_rotation = [
                loc_msg.pose.orientation.w,
                loc_msg.pose.orientation.x,
                loc_msg.pose.orientation.y,
                loc_msg.pose.orientation.z
            ]
        
        # Process camera data
        camera_data = {}
        for cam_name, cam_topic in self.camera_topics.items():
            if ('cameras' in sync_frame and 
                cam_topic in sync_frame['cameras']):
                
                # Extract H265 image
                image_msg = sync_frame['cameras'][cam_topic]
                image_data = self._decode_h265_image(image_msg)
                
                # Save image
                image_filename = f"{sample_token}_{cam_name}.jpg"
                image_path = f"{self.output_dir}/samples/{image_filename}"
                cv2.imwrite(image_path, image_data)
                
                camera_data[cam_name] = {
                    'data_path': image_filename,
                    'timestamp': sync_frame['timestamp'],
                    'sensor2ego_translation': self.camera_calibrations[cam_name]['sensor2ego_translation'],
                    'sensor2ego_rotation': self.camera_calibrations[cam_name]['sensor2ego_rotation'],
                    'cam_intrinsic': self.camera_calibrations[cam_name]['cam_intrinsic'],
                    'ego2global_translation': ego_translation,
                    'ego2global_rotation': ego_rotation
                }
        
        # Extract object annotations
        annotations = self._extract_object_annotations(sync_frame)
        
        # Generate CAN bus data
        can_bus_data = self._generate_can_bus_data(sync_frame)
        
        # Create nuScenes sample
        nuscenes_sample = {
            'token': sample_token,
            'scene_token': scene_token,
            'timestamp': sync_frame['timestamp'],
            'prev': f"sample_{sample_idx-1:06d}" if sample_idx > 0 else "",
            'next': f"sample_{sample_idx+1:06d}",
            
            # LiDAR data
            'lidar_path': f"{sample_token}_LIDAR_TOP.bin",  # Will be processed separately
            'lidar2ego_translation': [0.0, 0.041, 0.02],
            'lidar2ego_rotation': [0.0, 0.02, 0.01, 1.00],
            
            # Ego pose
            'ego2global_translation': ego_translation,
            'ego2global_rotation': ego_rotation,
            
            # Camera data
            'cams': camera_data,
            
            # Annotations
            'gt_boxes': annotations['gt_boxes'],
            'gt_names': annotations['gt_names'],
            'gt_velocity': annotations['gt_velocity'],
            'gt_tokens': annotations['gt_tokens'],
            'valid_flag': annotations['valid_flag'],
            
            # CAN bus
            'can_bus': can_bus_data,
            
            # Sweeps (for temporal context)
            'sweeps': [{
                'data_path': f"{sample_token}_LIDAR_TOP.bin",
                'type': 'lidar',
                'timestamp': sync_frame['timestamp'],
                'lidar2ego_translation': [0.0, 0.041, 0.02],
                'lidar2ego_rotation': [0.0, 0.02, 0.01, 1.00]
            }]
        }
        
        return nuscenes_sample
    
    def _decode_h265_image(self, image_msg):
        """Decode H265 ROS Image message to OpenCV format using CvBridge"""
        try:
            # Method 1: Use ROS CvBridge (proper way for ROS Image messages)
            try:
                # CvBridge can handle the H265 encoding specified in ROS message
                cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
                
                if cv_image is not None and cv_image.size > 0:
                    # Resize to standard resolution (1600x900)
                    resized_image = cv2.resize(cv_image, (1600, 900))
                    return resized_image
                    
            except Exception as bridge_error:
                print(f"CvBridge decode failed: {bridge_error}, trying raw H265 decoding...")
            
            # Method 2: Raw H265 bitstream decoding with ffmpeg
            if hasattr(image_msg, 'data') and len(image_msg.data) > 0:
                # Save raw H265 bitstream to temporary file
                with tempfile.NamedTemporaryFile(suffix='.h265', delete=False) as temp_h265:
                    temp_h265.write(image_msg.data)
                    temp_h265_path = temp_h265.name
                
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_jpg:
                    temp_jpg_path = temp_jpg.name
                
                try:
                    # Use ffmpeg to decode H265 bitstream to image
                    cmd = [
                        'ffmpeg', '-y', '-f', 'hevc', '-i', temp_h265_path, 
                        '-vframes', '1', '-f', 'image2', temp_jpg_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, check=True)
                    
                    # Read the decoded image
                    decoded_image = cv2.imread(temp_jpg_path)
                    
                    # Clean up temp files
                    os.unlink(temp_h265_path)
                    os.unlink(temp_jpg_path)
                    
                    if decoded_image is not None and decoded_image.size > 0:
                        resized_image = cv2.resize(decoded_image, (1600, 900))
                        return resized_image
                        
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"FFmpeg H265 decoding failed: {e}")
                    # Clean up temp files
                    try:
                        os.unlink(temp_h265_path)
                        os.unlink(temp_jpg_path) 
                    except:
                        pass
            
            # Method 3: Check if image message has proper dimensions for raw decoding
            if (hasattr(image_msg, 'width') and hasattr(image_msg, 'height') and 
                hasattr(image_msg, 'encoding') and hasattr(image_msg, 'data')):
                
                print(f"Image info: {image_msg.width}x{image_msg.height}, encoding: {image_msg.encoding}")
                
                # If it's actually raw image data, not H265 compressed
                if 'bgr8' in image_msg.encoding or 'rgb8' in image_msg.encoding:
                    expected_size = image_msg.width * image_msg.height * 3
                    if len(image_msg.data) == expected_size:
                        # Reshape raw data to image
                        image_array = np.frombuffer(image_msg.data, dtype=np.uint8)
                        cv_image = image_array.reshape((image_msg.height, image_msg.width, 3))
                        
                        if 'rgb8' in image_msg.encoding:
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                            
                        resized_image = cv2.resize(cv_image, (1600, 900))
                        return resized_image
            
            # If all methods fail, return blank image
            print(f"Warning: All H265 decoding methods failed, using blank image")
            return np.zeros((900, 1600, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error in H265 decoding: {e}")
            return np.zeros((900, 1600, 3), dtype=np.uint8)
    
    def _extract_object_annotations(self, sync_frame):
        """Extract object detection and tracking annotations"""
        
        # Initialize empty annotations
        annotations = {
            'gt_boxes': np.array([]),
            'gt_names': [],
            'gt_velocity': np.array([]),
            'gt_tokens': [],
            'valid_flag': np.array([])
        }
        
        # Extract from detected_objects_prediction if available
        if 'objects_prediction' in sync_frame:
            objects_msg = sync_frame['objects_prediction']
            
            if hasattr(objects_msg, 'objects'):
                boxes = []
                names = []
                velocities = []
                tokens = []
                flags = []
                
                for obj in objects_msg.objects:
                    # Extract 3D bounding box [x, y, z, w, l, h, yaw]
                    pose = obj.pose.position
                    # Use object dimensions if available, otherwise defaults
                    width = getattr(obj, 'width', 2.0)
                    length = getattr(obj, 'length', 4.5)
                    height = 1.8  # Default height
                    yaw = 0.0  # Extract from orientation if available
                    
                    boxes.append([pose.x, pose.y, pose.z, width, length, height, yaw])
                    
                    # Object class
                    obj_class = getattr(obj, 'Class', 1)  # Default to CAR
                    class_names = {0: 'pedestrian', 1: 'car', 2: 'truck', 
                                  4: 'motorcycle', 5: 'bicycle'}
                    names.append(class_names.get(obj_class, 'car'))
                    
                    # Velocity
                    vel = obj.velocity.linear
                    velocities.append([vel.x, vel.y])
                    
                    # Object token
                    tokens.append(f"obj_{obj.id}")
                    flags.append(True)
                
                if boxes:
                    annotations['gt_boxes'] = np.array(boxes)
                    annotations['gt_names'] = names
                    annotations['gt_velocity'] = np.array(velocities)
                    annotations['gt_tokens'] = tokens
                    annotations['valid_flag'] = np.array(flags)
        
        return annotations
    
    def _generate_can_bus_data(self, sync_frame):
        """Generate 18-dimensional CAN bus data (reuse extract_can_bus.py logic)"""
        
        # Default CAN bus vector
        can_bus = [0.0] * 18
        
        try:
            # Extract from car_state and imu data
            if 'car_state' in sync_frame and 'imu' in sync_frame:
                car_state = sync_frame['car_state']
                imu_data = sync_frame['imu']
                
                # Position (indices 0-2) - use current position for now
                can_bus[0] = car_state.pose.pose.position.x  # Will compute delta later
                can_bus[1] = car_state.pose.pose.position.y
                can_bus[2] = car_state.pose.pose.position.z
                
                # Quaternion rotation (indices 3-6)
                can_bus[3] = imu_data.orientation.w
                can_bus[4] = imu_data.orientation.x
                can_bus[5] = imu_data.orientation.y
                can_bus[6] = imu_data.orientation.z
                
                # Linear acceleration (indices 7-9)
                can_bus[7] = imu_data.linear_acceleration.x
                can_bus[8] = imu_data.linear_acceleration.y
                can_bus[9] = imu_data.linear_acceleration.z
                
                # Angular velocity (indices 10-12)
                can_bus[10] = imu_data.angular_velocity.x
                can_bus[11] = imu_data.angular_velocity.y
                can_bus[12] = imu_data.angular_velocity.z
                
                # Vehicle velocity (indices 13-15)
                can_bus[13] = car_state.twist.twist.linear.x
                can_bus[14] = car_state.twist.twist.linear.y
                can_bus[15] = car_state.twist.twist.linear.z
                
                # Yaw angle (index 16)
                quaternion = (can_bus[4], can_bus[5], can_bus[6], can_bus[3])
                _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
                can_bus[16] = yaw
                
                # Reserved (index 17)
                can_bus[17] = 0.0
                
        except Exception as e:
            print(f"Warning: Could not generate complete CAN bus data: {e}")
        
        return can_bus
    
    def process_single_bag(self):
        """Process a single ROS bag file"""
        
        print(f"Starting extraction for {self.bag_file}")
        print("=" * 60)
        
        # Step 1: Extract synchronized frames
        sync_frames = self.extract_synchronized_frames()
        
        if not sync_frames:
            print("No synchronized frames found!")
            return None
        
        # Step 2: Convert to nuScenes format
        nuscenes_samples = []
        
        for idx, frame in enumerate(sync_frames):
            try:
                sample = self.convert_to_nuscenes_format(frame, idx)
                nuscenes_samples.append(sample)
                
                if idx % 50 == 0:
                    print(f"Processed {idx+1}/{len(sync_frames)} frames")
                    
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
                continue
        
        # Step 3: Save processed data
        output_file = f"{self.output_dir}/samples_data.pkl"
        
        formatted_data = {
            'metadata': {
                'version': 'v1.0-trainval',
                'bag_file': os.path.basename(self.bag_file),
                'total_samples': len(nuscenes_samples),
                'camera_setup': '4-cam',
                'cameras': list(self.camera_topics.keys())
            },
            'samples': nuscenes_samples
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(formatted_data, f)
        
        # Step 4: Save metadata JSON
        metadata_file = f"{self.output_dir}/extraction_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(formatted_data['metadata'], f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Processed {len(nuscenes_samples)} samples")
        print(f"Data saved to: {output_file}")
        print(f"Metadata saved to: {metadata_file}")
        
        return formatted_data


def main():
    parser = argparse.ArgumentParser(description='Extract ITRI ROS bag data for UniAD training')
    parser.add_argument('bag_file', help='Input ROS bag file')
    parser.add_argument('--output_dir', '-o', required=True, 
                       help='Output directory for extracted data')
    parser.add_argument('--semantic_map_dir', 
                       help='Semantic map directory (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bag_file):
        print(f"Error: {args.bag_file} does not exist")
        return 1
    
    # Create extractor
    extractor = ITRIRosBagExtractor(
        args.bag_file, 
        args.output_dir,
        args.semantic_map_dir
    )
    
    # Process the bag file
    result = extractor.process_single_bag()
    
    if result:
        print("\n" + "="*60)
        print("ROS BAG EXTRACTION SUCCESSFUL")
        print("="*60)
        return 0
    else:
        print("\n" + "="*60)
        print("ROS BAG EXTRACTION FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit(main())