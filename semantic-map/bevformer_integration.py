#!/usr/bin/env python3

"""
BEVFormer Integration Module for UniAD with ITRI 4-Camera Setup

This module provides utilities for integrating BEVFormer with the ITRI
4-camera setup (front_100deg, front_left_100deg, front_right_100deg, back_60deg)
and existing CAN bus data.

Key Features:
1. Camera calibration and intrinsic/extrinsic parameter management
2. Image preprocessing pipeline for 4-camera configuration
3. Coordinate transformations between camera frames and ego vehicle frame
4. Integration with existing semantic map lane queries
5. BEVFormer configuration and inference utilities
"""

import os
import json
import pickle
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

# Import memory bridge functionality
from bev_memory_bridge import create_memory_bridge, BEVMemoryBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraCalibration:
    """
    Camera calibration and parameter management for ITRI 4-camera setup.
    
    Manages intrinsic and extrinsic parameters for:
    - front_100deg: Front camera with 100° FOV
    - front_left_100deg: Front-left camera with 100° FOV  
    - front_right_100deg: Front-right camera with 100° FOV
    - back_60deg: Rear camera with 60° FOV
    """
    
    def __init__(self, calibration_config: Optional[Dict] = None):
        """
        Initialize camera calibration parameters.
        
        Args:
            calibration_config: Optional dictionary containing calibration parameters
        """
        self.camera_names = ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg']
        self.calibration_config = calibration_config or self._default_calibration()
        
    def _default_calibration(self) -> Dict:
        """
        Generate default calibration parameters based on typical automotive camera setup.
        
        Returns:
            Dictionary containing intrinsic and extrinsic parameters for all cameras
        """
        # Default image resolution (adjust based on your actual images)
        img_height, img_width = 720, 1280
        
        # Default intrinsic parameters (focal length, principal point)
        # These should be calibrated with actual camera hardware
        fx, fy = img_width * 0.7, img_height * 0.7  # Approximate focal length
        cx, cy = img_width / 2, img_height / 2      # Principal point at center
        
        default_intrinsic = np.array([
            [fx,  0, cx],
            [ 0, fy, cy], 
            [ 0,  0,  1]
        ], dtype=np.float32)
        
        # Default distortion parameters (assumes minimal distortion)
        default_distortion = np.zeros(5, dtype=np.float32)
        
        # Extrinsic parameters: [x, y, z, roll, pitch, yaw] relative to ego vehicle center
        # Positions in meters, rotations in radians
        extrinsics = {
            'front_100deg': [2.0, 0.0, 1.8, 0.0, 0.0, 0.0],           # Front center
            'front_left_100deg': [1.8, -0.8, 1.8, 0.0, 0.0, -np.pi/4], # Front-left, 45° left
            'front_right_100deg': [1.8, 0.8, 1.8, 0.0, 0.0, np.pi/4],  # Front-right, 45° right  
            'back_60deg': [-1.5, 0.0, 1.8, 0.0, 0.0, np.pi]           # Rear center, 180° rotation
        }
        
        calibration = {}
        for cam_name in self.camera_names:
            calibration[cam_name] = {
                'intrinsic': default_intrinsic.copy(),
                'distortion': default_distortion.copy(),
                'extrinsic': extrinsics[cam_name],
                'image_size': (img_width, img_height)
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
    Image preprocessing pipeline for ITRI 4-camera setup.
    
    Handles:
    - Image loading and resizing
    - Normalization and formatting for BEVFormer input
    - Multi-camera batch preparation
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
    
    def prepare_multi_camera_batch(self, image_paths: Dict[str, str]) -> torch.Tensor:
        """
        Prepare batch of images from multiple cameras.
        
        Args:
            image_paths: Dictionary mapping camera names to image file paths
            
        Returns:
            Batch tensor of shape (1, num_cams, C, H, W)
        """
        camera_names = ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg']
        images = []
        
        for cam_name in camera_names:
            if cam_name not in image_paths:
                raise ValueError(f"Missing image path for camera: {cam_name}")
                
            img_path = image_paths[cam_name]
            processed_img = self.load_and_preprocess_image(img_path)
            images.append(processed_img)
            
        # Stack images: (num_cams, C, H, W)
        batch_images = np.stack(images, axis=0)
        
        # Add batch dimension: (1, num_cams, C, H, W)
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
        Load CAN bus data from pickle file.
        
        Args:
            canbus_path: Path to CAN bus pickle file
            
        Returns:
            Dictionary containing 'can_bus' and 'timestamps' arrays
        """
        with open(canbus_path, 'rb') as f:
            canbus_data = pickle.load(f)
            
        logger.info(f"Loaded CAN bus data: {canbus_data['can_bus'].shape[0]} frames, "
                   f"{canbus_data['can_bus'].shape[1]} dimensions")
        
        return canbus_data
    
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
    data_dir = "/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic"
    
    # Example image paths for frame 1
    image_paths = {
        'front_100deg': f"{data_dir}/photo_extracted/front_100deg/f100_142305_1.jpg",
        'front_left_100deg': f"{data_dir}/photo_extracted/front_left_100deg/fl100_142305_1.jpg", 
        'front_right_100deg': f"{data_dir}/photo_extracted/front_right_100deg/fr100_142305_1.jpg",
        'back_60deg': f"{data_dir}/photo_extracted/back_60deg/b60_142305_1.jpg"
    }
    
    # Load CAN bus data
    canbus_path = f"{data_dir}/canbus/2025-08-06-14-23-05_0_can_bus.pkl"
    
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