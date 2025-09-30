#!/usr/bin/env python3

"""
BEVFormer Integration Module for UniAD with ITRI 6-Camera Setup

This module provides utilities for integrating BEVFormer with the ITRI
4-camera setup extended to 6 cameras with dummy cameras for UniAD compatibility.
Real cameras: CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK  
Dummy cameras: CAM_BACK_LEFT, CAM_BACK_RIGHT (for UniAD 6-camera requirement)

Camera Specifications:
- Image Resolution: 1440x928 pixels
- Real Cameras: Calibrated with actual ITRI hardware parameters
- Intrinsics: Converted from flat arrays to 3x3 matrices
- Extrinsics: Quaternions converted to Euler angles [roll, pitch, yaw]
- Distortion: Zero distortion assumed

Key Features:
1. ITRI hardware-specific camera calibration with real parameters
2. Quaternion to Euler angle conversion for UniAD compatibility  
3. Image preprocessing pipeline for 6-camera configuration
4. Coordinate transformations between camera frames and ego vehicle frame
5. Integration with existing semantic map lane queries
6. BEVFormer configuration and inference utilities
"""

import os
import json
import pickle
import sys


import numpy as np


import torch
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# Import memory bridge functionality
from bev_memory_bridge import create_memory_bridge, BEVMemoryBridge

# NumPy compatibility fix for pickle files created with newer numpy versions
# Maps numpy._core to numpy.core for compatibility with numpy 1.22.4
import numpy.core
sys.modules['numpy._core'] = numpy.core
sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
if hasattr(numpy.core, '_multiarray_umath'):
    sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# def quaternion_to_euler(quaternion: List[float]) -> List[float]:
#     """
#     Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
    
#     Args:
#         quaternion: Quaternion as [w, x, y, z]
        
#     Returns:
#         Euler angles as [roll, pitch, yaw] in radians
#     """
#     w, x, y, z = quaternion
    
#     # Roll (x-axis rotation)
#     sinr_cosp = 2 * (w * x + y * z)
#     cosr_cosp = 1 - 2 * (x * x + y * y)
#     roll = np.arctan2(sinr_cosp, cosr_cosp)
    
#     # Pitch (y-axis rotation)
#     sinp = 2 * (w * y - z * x)
#     if np.abs(sinp) >= 1:
#         pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
#     else:
#         pitch = np.arcsin(sinp)
    
#     # Yaw (z-axis rotation)
#     siny_cosp = 2 * (w * z + x * y)
#     cosy_cosp = 1 - 2 * (y * y + z * z)
#     yaw = np.arctan2(siny_cosp, cosy_cosp)
    
#     return [roll, pitch, yaw]


def intrinsic_array_to_matrix(intrinsic_array: List[float]) -> np.ndarray:
    """
    Convert flat intrinsic array to 3x3 matrix.
    
    Args:
        intrinsic_array: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        
    Returns:
        3x3 intrinsic matrix
    """
    return np.array([
        [intrinsic_array[0], intrinsic_array[1], intrinsic_array[2]],
        [intrinsic_array[3], intrinsic_array[4], intrinsic_array[5]],
        [intrinsic_array[6], intrinsic_array[7], intrinsic_array[8]]
    ], dtype=np.float32)


class CameraCalibration:
    """
    Camera calibration and parameter management for ITRI 6-camera setup.
    
    Manages intrinsic and extrinsic parameters for:
    - CAM_FRONT: Front camera with 100° FOV (1440x928)
    - CAM_FRONT_LEFT: Front-left camera with 100° FOV (1440x928)  
    - CAM_FRONT_RIGHT: Front-right camera with 100° FOV (1440x928)
    - CAM_BACK: Rear camera with 60° FOV (1440x928)
    - CAM_BACK_LEFT: Dummy camera (interpolated from rear camera)
    - CAM_BACK_RIGHT: Dummy camera (interpolated from rear camera)
    
    Uses actual ITRI hardware calibration data with quaternion-to-Euler conversion.
    """
    
    def __init__(self, calibration_config: Optional[Dict] = None):
        """
        Initialize camera calibration parameters.
        
        Args:
            calibration_config: Optional dictionary containing calibration parameters
        """
        # 6-camera setup: 4 real ITRI cameras + 2 dummy cameras for UniAD compatibility
        self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        self.calibration_config = calibration_config or self._default_calibration()
        
    def _default_calibration(self) -> Dict:
        """
        Generate ITRI camera calibration parameters using actual hardware specifications.
        
        Returns:
            Dictionary containing intrinsic and extrinsic parameters for all ITRI cameras
        """
        # ITRI image resolution: 1440x928 for all cameras
        img_height, img_width = 928, 1440
        
        # Zero distortion parameters (as specified)
        zero_distortion = np.zeros(5, dtype=np.float32)
        
        # ITRI Camera Configuration (from actual hardware)
        itri_cameras = {
            'CAM_FRONT': {
                'data_path': '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image/lucid_cameras_x00.gige_100_f_hdr.h265',
                'sensor2ego_translation': [0.005, -0.125, -0.171],
                'sensor2ego_rotation': [-0.525, 0.509, -0.491, -0.495],
                'cam_intrinsic': [657.904403, 0.0, 714.717385597, 0.0, 658.500765392, 463.1351298, 0.0, 0.0, 1.0],
                'is_dummy': False
            },
            'CAM_FRONT_LEFT': {
                'data_path': '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image/lucid_cameras_x01.gige_100_fl_hdr.h265',
                'sensor2ego_translation': [0.213, -0.073, -0.671],
                'sensor2ego_rotation': [0.691, -0.169, 0.162, 0.684],
                'cam_intrinsic': [660.05027892, 0.0, 720.216502034, 0.0, 660.349325206, 467.66874548, 0.0, 0.0, 1.0],
                'is_dummy': False
            },
            'CAM_FRONT_RIGHT': {
                'data_path': '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image/lucid_cameras_x01.gige_100_fr_hdr.h265',
                'sensor2ego_translation': [-0.223, -0.081, -0.654],
                'sensor2ego_rotation': [-0.177, 0.688, -0.682, -0.176],
                'cam_intrinsic': [659.20444086, 0.0, 713.833685513, 0.0, 659.864183739, 455.950517493, 0.0, 0.0, 1.0],
                'is_dummy': False
            },
            'CAM_BACK': {
                'data_path': '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image/lucid_cameras_x00.gige_60_b_hdr.h265',
                'sensor2ego_translation': [-0.011, -0.034, -2.469],
                'sensor2ego_rotation': [0.508, 0.513, -0.486, 0.493],
                'cam_intrinsic': [1033.063670, 0.0, 730.505500, 0.0, 1034.542542, 468.282976, 0.0, 0.0, 1.0],
                'is_dummy': False
            },
            # Dummy cameras (interpolated from real back camera)
            'CAM_BACK_LEFT': {
                'data_path': None,  # No real data
                'sensor2ego_translation': [0, 0, 0],  # Same as back
                'sensor2ego_rotation': [0, 0, 0, 0],  # Rotated ~30° left
                'cam_intrinsic': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Placeholder
                'is_dummy': True
            },
            'CAM_BACK_RIGHT': {
                'data_path': None,  # No real data
                'sensor2ego_translation': [0, 0, 0],  # Same as back
                'sensor2ego_rotation': [0, 0, 0, 0],  # Rotated ~30° left
                'cam_intrinsic': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Placeholder
                'is_dummy': True
            }
        }
        
        calibration = {}
        for cam_name in self.camera_names:
            cam_config = itri_cameras[cam_name]
            
            # Convert flat intrinsic array to 3x3 matrix
            intrinsic_matrix = intrinsic_array_to_matrix(cam_config['cam_intrinsic'])
            
            # Convert quaternion to Euler angles
            quaternion = cam_config['sensor2ego_rotation']
            roll, pitch, yaw = quaternion_to_euler(quaternion)
            
            # Combine translation and Euler angles for extrinsic
            translation = cam_config['sensor2ego_translation']
            extrinsic = translation + [roll, pitch, yaw]  # [x, y, z, roll, pitch, yaw]
            
            calibration[cam_name] = {
                'intrinsic': intrinsic_matrix,
                'distortion': zero_distortion.copy(),
                'extrinsic': extrinsic,
                'image_size': (img_width, img_height),
                'data_path': cam_config['data_path'],
                'is_dummy': cam_config['is_dummy']
            }
            
        return calibration
    
    def get_camera_intrinsic(self, camera_name: str) -> np.ndarray:
        """Get intrinsic matrix for specified camera."""
        return self.calibration_config[camera_name]['intrinsic']
    
    def get_camera_extrinsic(self, camera_name: str) -> List[float]:
        """Get extrinsic parameters for specified camera."""
        return self.calibration_config[camera_name]['extrinsic']
    
    def get_lidar2img_transforms(self) -> Dict[str, np.ndarray]:
        """
        Generate lidar2img transformation matrices for all cameras.
        
        Returns:
            Dictionary mapping camera names to 4x4 transformation matrices
        """
        transforms = {}
        
        for cam_name in self.camera_names:
            # Get extrinsic parameters [x, y, z, roll, pitch, yaw]
            extrinsic = self.get_camera_extrinsic(cam_name)
            x, y, z, roll, pitch, yaw = extrinsic
            
            # Create rotation matrix from Euler angles
            cos_r, sin_r = np.cos(roll), np.sin(roll)
            cos_p, sin_p = np.cos(pitch), np.sin(pitch)
            cos_y, sin_y = np.cos(yaw), np.sin(yaw)
            
            # Combined rotation matrix (ZYX convention)
            R = np.array([
                [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
                [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
                [-sin_p,      cos_p*sin_r,                      cos_p*cos_r]
            ], dtype=np.float32)
            
            # Translation vector
            t = np.array([x, y, z], dtype=np.float32)
            
            # Create 4x4 extrinsic matrix [R|t]
            extrinsic_matrix = np.eye(4, dtype=np.float32)
            extrinsic_matrix[:3, :3] = R
            extrinsic_matrix[:3, 3] = t
            
            # Get intrinsic matrix and expand to 4x4
            intrinsic = self.get_camera_intrinsic(cam_name)
            intrinsic_4x4 = np.eye(4, dtype=np.float32)
            intrinsic_4x4[:3, :3] = intrinsic
            
            # Combined lidar2img transformation
            lidar2img = intrinsic_4x4 @ np.linalg.inv(extrinsic_matrix)
            transforms[cam_name] = lidar2img
            
        return transforms
    
    def save_calibration(self, filepath: str):
        """Save calibration parameters to file."""
        with open(filepath, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_config = {}
            for cam_name, params in self.calibration_config.items():
                serializable_config[cam_name] = {
                    'intrinsic': params['intrinsic'].tolist(),
                    'distortion': params['distortion'].tolist(),
                    'extrinsic': params['extrinsic'],
                    'image_size': params['image_size']
                }
            json.dump(serializable_config, f, indent=2)
        logger.info(f"Camera calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """Load calibration parameters from file."""
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            
        # Convert lists back to numpy arrays
        for cam_name, params in loaded_config.items():
            params['intrinsic'] = np.array(params['intrinsic'], dtype=np.float32)
            params['distortion'] = np.array(params['distortion'], dtype=np.float32)
            
        self.calibration_config = loaded_config
        logger.info(f"Camera calibration loaded from {filepath}")


class ImagePreprocessor:
    """
    Image preprocessing pipeline for ITRI 6-camera setup (4 real + 2 dummy).
    
    Handles:
    - Image loading and resizing for real cameras
    - Generation of dummy images for padded cameras
    - Normalization and formatting for BEVFormer input
    - Multi-camera batch preparation (6 cameras total)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (480, 800), 
                 mean: List[float] = [0.485, 0.456, 0.406],
                 std: List[float] = [0.229, 0.224, 0.225]):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            mean: Normalization mean values (RGB)
            std: Normalization std values (RGB)
        """
        self.target_size = target_size
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Preprocessed image as numpy array (C, H, W)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize with ImageNet statistics
        image = (image - self.mean) / self.std
        
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def generate_dummy_image(self) -> np.ndarray:
        """
        Generate a dummy image filled with normalization mean values.
        
        Returns:
            Dummy image as numpy array (C, H, W)
        """
        # Create image filled with mean values (creates a neutral gray image)
        dummy_image = np.full((self.target_size[0], self.target_size[1], 3), 
                             self.mean.reshape(1, 1, 3), dtype=np.float32)
        
        # Convert to CHW format
        dummy_image = np.transpose(dummy_image, (2, 0, 1))
        
        return dummy_image
    
    def prepare_multi_camera_batch(self, image_paths: Dict[str, str]) -> torch.Tensor:
        """
        Prepare batch of images from multiple cameras (6 total: 4 real + 2 dummy).
        
        Args:
            image_paths: Dictionary mapping camera names to image file paths (only for real cameras)
            
        Returns:
            Batch tensor of shape (1, 6, C, H, W)
        """
        # Define camera order: real cameras first, then dummy cameras
        camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        real_cameras = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK']
        dummy_cameras = ['CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        images = []
        
        for cam_name in camera_names:
            if cam_name in real_cameras:
                # Load and process real camera image
                if cam_name not in image_paths:
                    raise ValueError(f"Missing image path for real camera: {cam_name}")
                    
                img_path = image_paths[cam_name]
                processed_img = self.load_and_preprocess_image(img_path)
                images.append(processed_img)
                
            elif cam_name in dummy_cameras:
                # Generate dummy image
                dummy_img = self.generate_dummy_image()
                images.append(dummy_img)
                
        # Stack images: (6, C, H, W)
        batch_images = np.stack(images, axis=0)
        
        # Add batch dimension: (1, 6, C, H, W)
        batch_images = np.expand_dims(batch_images, axis=0)
        
        # Convert to PyTorch tensor
        return torch.from_numpy(batch_images)


class CoordinateTransformer:
    """
    Coordinate transformation utilities for BEVFormer integration.
    
    Handles transformations between:
    - Camera coordinate frames
    - Ego vehicle coordinate frame
    - BEV (Bird's Eye View) coordinate frame
    """
    
    def __init__(self, pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]):
        """
        Initialize coordinate transformer.
        
        Args:
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.pc_range = pc_range
        
    def camera_to_ego(self, points: np.ndarray, camera_extrinsic: List[float]) -> np.ndarray:
        """
        Transform points from camera coordinates to ego vehicle coordinates.
        
        Args:
            points: Points in camera coordinates (N, 3)
            camera_extrinsic: Camera extrinsic parameters [x, y, z, roll, pitch, yaw]
            
        Returns:
            Points in ego coordinates (N, 3)
        """
        x, y, z, roll, pitch, yaw = camera_extrinsic
        
        # Create rotation matrix
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        R = np.array([
            [cos_y*cos_p, cos_y*sin_p*sin_r - sin_y*cos_r, cos_y*sin_p*cos_r + sin_y*sin_r],
            [sin_y*cos_p, sin_y*sin_p*sin_r + cos_y*cos_r, sin_y*sin_p*cos_r - cos_y*sin_r],
            [-sin_p,      cos_p*sin_r,                      cos_p*cos_r]
        ], dtype=np.float32)
        
        # Transform points
        ego_points = (R @ points.T).T + np.array([x, y, z])
        
        return ego_points
    
    def filter_points_by_range(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter points within the specified PC range.
        
        Args:
            points: Points to filter (N, 3)
            
        Returns:
            Tuple of (filtered_points, valid_mask)
        """
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        
        valid_mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        )
        
        filtered_points = points[valid_mask]
        return filtered_points, valid_mask


class BEVFormerDataProcessor:
    """
    Enhanced data processor for integrating ITRI data with BEVFormer and memory bridge.
    
    Combines camera calibration, image preprocessing, coordinate transformations,
    and memory bridging to prepare complete data for UniAD inference.
    """
    
    def __init__(self, calibration_config: Optional[Dict] = None,
                 pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 device: str = 'cuda'):
        """
        Initialize enhanced BEVFormer data processor.
        
        Args:
            calibration_config: Camera calibration configuration
            pc_range: Point cloud range for BEV representation
            device: Device for tensor operations
        """
        self.camera_calibration = CameraCalibration(calibration_config)
        self.image_preprocessor = ImagePreprocessor()
        self.coordinate_transformer = CoordinateTransformer(pc_range)
        self.device = device
        
        # Initialize memory bridge
        bridge_config = {
            'bev_h': 200,
            'bev_w': 200,
            'embed_dims': 256,
            'pc_range': pc_range
        }
        self.memory_bridge = create_memory_bridge(bridge_config, device)
        
    def load_canbus_data(self, canbus_path: str) -> Dict[str, np.ndarray]:
        """
        Load CAN bus data from pickle file with numpy compatibility handling.
        
        Args:
            canbus_path: Path to CAN bus pickle file
            
        Returns:
            Dictionary containing 'can_bus' and 'timestamps' arrays
        """
        logger.info(f"Loading CAN bus data from: {canbus_path}")
        
        try:
            with open(canbus_path, 'rb') as f:
                canbus_data = pickle.load(f)
                
            logger.info(f"Loaded CAN bus data successfully:")
            logger.info(f"  - Frames: {canbus_data['can_bus'].shape[0]}")
            logger.info(f"  - Dimensions: {canbus_data['can_bus'].shape[1]}")
            logger.info(f"  - Data type: {canbus_data['can_bus'].dtype}")
            
            return canbus_data
            
        except Exception as e:
            logger.error(f"Failed to load CAN bus data: {e}")
            logger.error(f"File path: {canbus_path}")
            raise
    
    def prepare_sample_data(self, 
                           image_paths: Dict[str, str], 
                           canbus_data: Dict[str, np.ndarray],
                           frame_index: int = 0) -> Dict:
        """
        Prepare a complete data sample for BEVFormer inference.
        
        Args:
            image_paths: Dictionary mapping camera names to image file paths
            canbus_data: CAN bus data dictionary
            frame_index: Index of frame to use from CAN bus data
            
        Returns:
            Dictionary containing all data needed for BEVFormer
        """
        # Prepare multi-camera images
        img_batch = self.image_preprocessor.prepare_multi_camera_batch(image_paths)
        
        # Get CAN bus data for current frame
        can_bus_frame = canbus_data['can_bus'][frame_index]
        timestamp = canbus_data['timestamps'][frame_index]
        
        # Get camera transformation matrices
        lidar2img_transforms = self.camera_calibration.get_lidar2img_transforms()
        
        # Prepare metadata
        img_metas = [{
            'can_bus': can_bus_frame,
            'lidar2img': [lidar2img_transforms[cam] for cam in self.camera_calibration.camera_names],
            'timestamp': timestamp,
            'img_shape': [(self.image_preprocessor.target_size[0], 
                          self.image_preprocessor.target_size[1], 3)] * 4,
            'pad_shape': [(self.image_preprocessor.target_size[0], 
                          self.image_preprocessor.target_size[1], 3)] * 4,
            'scale_factor': [1.0, 1.0, 1.0, 1.0],
        }]
        
        return {
            'img': img_batch,
            'img_metas': img_metas,
            'can_bus': can_bus_frame,
            'timestamp': timestamp
        }
    
    def run_bevformer_inference(
        self,
        bevformer_model,
        image_paths: Dict[str, str],
        canbus_data: Dict[str, np.ndarray],
        frame_index: int = 0
    ) -> torch.Tensor:
        """
        Run BEVFormer inference on ITRI 4-camera + CAN bus data.
        
        Args:
            bevformer_model: BEVFormer model instance
            image_paths: Dictionary mapping camera names to image paths
            canbus_data: CAN bus data dictionary
            frame_index: Frame index to process
            
        Returns:
            BEV feature tensor [B, C, H, W]
        """
        # Prepare input data
        sample_data = self.prepare_sample_data(image_paths, canbus_data, frame_index)
        
        # Move tensors to device
        img_tensor = sample_data['img'].to(self.device)
        img_metas = sample_data['img_metas']
        
        # Run BEVFormer encoder - extract multi-level features
        with torch.no_grad():
            img_feats = bevformer_model.extract_img_feat(
                img=img_tensor
            )
            
            # Convert multi-level features to BEV embeddings
            bev_embed, bev_pos = bevformer_model.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats,
                img_metas=img_metas
            )
            
        logger.info(f"BEVFormer inference completed - BEV shape: {bev_embed.shape}")
        return bev_embed
    
    def integrate_with_semantic_map(
        self,
        bev_embed: torch.Tensor,
        lane_query: torch.Tensor,
        lane_query_pos: torch.Tensor,
        img_metas: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Integrate BEV features with semantic map lane queries using memory bridge.
        
        Args:
            bev_embed: BEV feature tensor from BEVFormer
            lane_query: Lane queries from semantic map conversion
            lane_query_pos: Lane query positional encodings
            img_metas: Image metadata
            
        Returns:
            Complete outs_seg dictionary for MotionFormer
        """
        # Use memory bridge to create complete args_tuple
        outs_seg = self.memory_bridge.bridge_bev_to_motion(
            bev_embed, lane_query, lane_query_pos, img_metas
        )
        
        logger.info("BEV features integrated with semantic map queries")
        return outs_seg
    
    def process_complete_frame(
        self,
        bevformer_model,
        image_paths: Dict[str, str],
        canbus_data: Dict[str, np.ndarray],
        lane_query: torch.Tensor,
        lane_query_pos: torch.Tensor,
        frame_index: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process complete frame: 4-camera images + CAN bus → BEV → Memory Bridge.
        
        Args:
            bevformer_model: BEVFormer model instance
            image_paths: Dictionary of camera image paths
            canbus_data: CAN bus data
            lane_query: Lane queries from semantic map
            lane_query_pos: Lane query positional encodings
            frame_index: Frame index to process
            
        Returns:
            Tuple of (bev_embed, outs_seg)
        """
        # Step 1: Run BEVFormer inference
        bev_embed = self.run_bevformer_inference(
            bevformer_model, image_paths, canbus_data, frame_index
        )
        
        # Step 2: Prepare image metadata
        sample_data = self.prepare_sample_data(image_paths, canbus_data, frame_index)
        img_metas = sample_data['img_metas']
        
        # Step 3: Integrate with semantic map via memory bridge
        outs_seg = self.integrate_with_semantic_map(
            bev_embed, lane_query, lane_query_pos, img_metas
        )
        
        logger.info(f"Complete frame processing finished - Frame {frame_index}")
        return bev_embed, outs_seg
    
    def validate_integration(self) -> bool:
        """
        Validate that the enhanced integration is working correctly.
        
        Returns:
            True if integration is valid
        """
        try:
            # Test memory bridge
            test_bev = torch.randn(1, 256, 200, 200, device=self.device)
            test_lane_query = torch.randn(1, 100, 256, device=self.device) 
            test_lane_pos = torch.randn(1, 100, 256, device=self.device)
            
            # Test integration
            result = self.integrate_with_semantic_map(
                test_bev, test_lane_query, test_lane_pos
            )
            
            # Validate result structure
            if 'args_tuple' not in result:
                logger.error("Missing args_tuple in integration result")
                return False
                
            args_tuple = result['args_tuple']
            if len(args_tuple) != 7:
                logger.error(f"Invalid args_tuple length: {len(args_tuple)}")
                return False
                
            # Check memory features are populated
            if any(args_tuple[i] is None for i in [0, 1, 2]):
                logger.error("Memory features still None after integration")
                return False
                
            logger.info("Enhanced BEVFormer integration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Integration validation failed: {e}")
            return False
    
    def save_configuration(self, config_dir: str):
        """Save all configuration files to specified directory."""
        os.makedirs(config_dir, exist_ok=True)
        
        # Save camera calibration
        calib_path = os.path.join(config_dir, 'camera_calibration.json')
        self.camera_calibration.save_calibration(calib_path)
        
        # Save processor configuration
        config_path = os.path.join(config_dir, 'bevformer_config.json')
        config = {
            'pc_range': self.coordinate_transformer.pc_range,
            'target_image_size': self.image_preprocessor.target_size,
            'normalization_mean': self.image_preprocessor.mean.tolist(),
            'normalization_std': self.image_preprocessor.std.tolist(),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Configuration saved to {config_dir}")
    
    @classmethod
    def load_configuration(cls, config_dir: str) -> 'BEVFormerDataProcessor':
        """Load processor from saved configuration."""
        # Load camera calibration
        calib_path = os.path.join(config_dir, 'camera_calibration.json')
        camera_calib = CameraCalibration()
        camera_calib.load_calibration(calib_path)
        
        # Load processor configuration
        config_path = os.path.join(config_dir, 'bevformer_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        processor = cls(
            calibration_config=camera_calib.calibration_config,
            pc_range=config['pc_range']
        )
        
        # Update preprocessor settings
        processor.image_preprocessor.target_size = tuple(config['target_image_size'])
        processor.image_preprocessor.mean = np.array(config['normalization_mean'])
        processor.image_preprocessor.std = np.array(config['normalization_std'])
        
        logger.info(f"Configuration loaded from {config_dir}")
        return processor


def create_default_setup():
    """Create default BEVFormer integration setup for ITRI data."""
    processor = BEVFormerDataProcessor()
    
    # Save default configuration
    config_dir = '/home/bryan/Desktop/Allen/UniAD/semantic-map/config'
    processor.save_configuration(config_dir)
    
    logger.info("Default BEVFormer integration setup created")
    return processor


if __name__ == "__main__":
    # Example usage
    processor = create_default_setup()
    
    # Example: Load and process a sample
    data_dir = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/images"
    
    # Example image paths for frame 1 (using actual directory structure and file names)
    image_paths = {
        'CAM_FRONT': f"{data_dir}/front_100/f100_142305_001.jpg",
        'CAM_FRONT_LEFT': f"{data_dir}/frontleft_100/fl100_142305_001.jpg", 
        'CAM_FRONT_RIGHT': f"{data_dir}/frontright_100/fr100_142305_001.jpg",
        'CAM_BACK': f"{data_dir}/back_60/b60_142305_001.jpg"
    }
    
    # Load CAN bus data
    canbus_path = "/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus/2025-08-06-14-23-05_0_can_bus.pkl"

    try:
        canbus_data = processor.load_canbus_data(canbus_path)
        
        # Check if image files exist before processing
        missing_files = []
        for cam, path in image_paths.items():
            if not os.path.exists(path):
                missing_files.append(f"{cam}: {path}")
                
        if missing_files:
            logger.warning(f"Missing image files: {missing_files}")
        else:
            sample_data = processor.prepare_sample_data(image_paths, canbus_data, frame_index=0)
            logger.info(f"Sample data prepared successfully")
            logger.info(f"Image batch shape: {sample_data['img'].shape}")
            logger.info(f"CAN bus data shape: {sample_data['can_bus'].shape}")
            
    except Exception as e:
        logger.error(f"Error processing sample data: {e}")
        
    logger.info("ITRI BEVFormer integration with actual hardware parameters complete")
