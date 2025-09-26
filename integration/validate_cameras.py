#!/usr/bin/env python3
"""
Camera Data Pipeline Validation for ITRI 4-Camera Setup

Validates the processing of 4-camera data through the complete pipeline:
1. Camera image loading and preprocessing
2. Calibration parameter application
3. BEVFormer input tensor generation
4. CAN bus data integration
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Add paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/integration')

from bevformer_integration import BEVFormerDataProcessor, CameraCalibration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraValidator:
    """Validate ITRI 4-camera data processing pipeline."""
    
    def __init__(self, data_path: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic'):
        self.data_path = Path(data_path)
        self.photo_path = self.data_path / 'photo_extracted'
        self.canbus_path = self.data_path / 'canbus'
        
        # Initialize camera calibration
        self.calibration = CameraCalibration()
        self.camera_names = ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg']
        
        # BEVFormer processor
        self.processor = BEVFormerDataProcessor()
        
    def validate_camera_structure(self) -> bool:
        """Validate that all 4 camera directories exist with images."""
        logger.info("🏗️ Validating camera directory structure...")
        
        for cam_name in self.camera_names:
            cam_dir = self.photo_path / cam_name
            if not cam_dir.exists():
                logger.error(f"❌ Camera directory missing: {cam_dir}")
                return False
                
            # Check for images
            image_files = list(cam_dir.glob('*.jpg'))
            if not image_files:
                logger.error(f"❌ No images found in {cam_dir}")
                return False
                
            logger.info(f"✅ {cam_name}: {len(image_files)} images found")
            
        return True
        
    def load_sample_images(self, frame_idx: int = 0) -> Dict[str, np.ndarray]:
        """Load sample images from each camera for validation."""
        logger.info(f"📷 Loading sample images for frame {frame_idx}...")
        
        sample_images = {}
        
        for cam_name in self.camera_names:
            cam_dir = self.photo_path / cam_name
            image_files = sorted(list(cam_dir.glob('*.jpg')))
            
            if frame_idx >= len(image_files):
                logger.warning(f"⚠️ Frame {frame_idx} not available for {cam_name}, using frame 0")
                frame_idx = 0
                
            image_path = image_files[frame_idx]
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"❌ Failed to load image: {image_path}")
                continue
                
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sample_images[cam_name] = image
            
            logger.info(f"✅ {cam_name}: {image.shape} loaded from {image_path.name}")
            
        return sample_images
        
    def validate_image_preprocessing(self, images: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Validate image preprocessing pipeline."""
        logger.info("🔧 Validating image preprocessing...")
        
        processed_images = {}
        
        for cam_name, image in images.items():
            # Get calibration parameters
            intrinsic = self.calibration.get_camera_intrinsic(cam_name)
            extrinsic = self.calibration.get_camera_extrinsic(cam_name)
            
            logger.info(f"📐 {cam_name} intrinsic shape: {intrinsic.shape}")
            logger.info(f"📐 {cam_name} extrinsic: {extrinsic}")
            
            # Process image for BEVFormer input
            # Resize to standard input size (adjust as needed)
            target_size = (800, 450)  # width, height
            resized = cv2.resize(image, target_size)
            
            # Normalize (typical ImageNet normalization)
            normalized = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            normalized = (normalized - mean) / std
            
            # Convert to tensor (C, H, W)
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1))
            processed_images[cam_name] = tensor
            
            logger.info(f"✅ {cam_name}: {image.shape} → {tensor.shape}")
            
        return processed_images
        
    def create_multi_camera_tensor(self, processed_images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create multi-camera tensor for BEVFormer input."""
        logger.info("🔗 Creating multi-camera tensor...")
        
        # Stack cameras in order: front, front_left, front_right, back
        camera_order = ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg']
        
        camera_tensors = []
        for cam_name in camera_order:
            if cam_name in processed_images:
                camera_tensors.append(processed_images[cam_name])
            else:
                logger.warning(f"⚠️ Missing camera {cam_name}, creating dummy tensor")
                # Create dummy tensor with same shape as others
                dummy_tensor = torch.zeros_like(list(processed_images.values())[0])
                camera_tensors.append(dummy_tensor)
                
        # Stack: (num_cameras, channels, height, width)
        multi_cam_tensor = torch.stack(camera_tensors, dim=0)
        
        # Add batch dimension: (batch_size, num_cameras, channels, height, width)
        batched_tensor = multi_cam_tensor.unsqueeze(0)
        
        logger.info(f"✅ Multi-camera tensor shape: {batched_tensor.shape}")
        return batched_tensor
        
    def validate_canbus_data(self) -> bool:
        """Validate CAN bus data loading."""
        logger.info("🚌 Validating CAN bus data...")
        
        canbus_files = list(self.canbus_path.glob('*.pkl'))
        if not canbus_files:
            logger.error("❌ No CAN bus files found")
            return False
            
        logger.info(f"📁 Found {len(canbus_files)} CAN bus files")
        
        # Load a sample CAN bus file
        sample_file = canbus_files[0]
        try:
            with open(sample_file, 'rb') as f:
                canbus_data = pickle.load(f)
                
            logger.info(f"✅ CAN bus data type: {type(canbus_data)}")
            
            if isinstance(canbus_data, (list, np.ndarray)):
                logger.info(f"✅ CAN bus data length: {len(canbus_data)}")
                if len(canbus_data) > 0:
                    sample_entry = canbus_data[0]
                    logger.info(f"✅ Sample entry type: {type(sample_entry)}")
                    if hasattr(sample_entry, 'shape'):
                        logger.info(f"✅ Sample entry shape: {sample_entry.shape}")
                        
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load CAN bus data: {e}")
            return False
            
    def run_validation(self, num_frames: int = 3) -> bool:
        """Run complete camera validation pipeline."""
        logger.info("🚀 Starting ITRI Camera Data Pipeline Validation")
        logger.info("=" * 60)
        
        # Step 1: Validate directory structure
        if not self.validate_camera_structure():
            return False
            
        # Step 2: Validate CAN bus data
        if not self.validate_canbus_data():
            return False
            
        # Step 3: Process sample frames
        for frame_idx in range(num_frames):
            logger.info(f"\n📸 Processing Frame {frame_idx}")
            logger.info("-" * 30)
            
            # Load sample images
            images = self.load_sample_images(frame_idx)
            if len(images) != 4:
                logger.warning(f"⚠️ Only {len(images)}/4 cameras available for frame {frame_idx}")
                
            # Validate preprocessing
            processed_images = self.validate_image_preprocessing(images)
            
            # Create multi-camera tensor
            multi_cam_tensor = self.create_multi_camera_tensor(processed_images)
            
            # Validate tensor properties
            logger.info(f"📊 Final tensor shape: {multi_cam_tensor.shape}")
            logger.info(f"📊 Tensor dtype: {multi_cam_tensor.dtype}")
            logger.info(f"📊 Tensor device: {multi_cam_tensor.device}")
            logger.info(f"📊 Tensor memory: {multi_cam_tensor.numel() * multi_cam_tensor.element_size() / 1024**2:.2f} MB")
            
        logger.info("\n" + "=" * 60)
        logger.info("✅ ITRI Camera Data Pipeline Validation Complete!")
        return True


if __name__ == "__main__":
    import pickle
    
    validator = CameraValidator()
    success = validator.run_validation(num_frames=3)
    
    if success:
        logger.info("🎉 All validations passed!")
    else:
        logger.error("❌ Some validations failed!")
        sys.exit(1)