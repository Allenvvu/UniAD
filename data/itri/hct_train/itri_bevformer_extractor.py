#!/usr/bin/env python3
"""
ITRI BEVFormer Feature Extractor

Extracts BEV features from ITRI 4-camera data using pre-trained BEVFormer model.
Implements 4→6 camera padding with dummy cameras and image alignment for training.

Key Features:
1. 4→6 camera padding with attention masking
2. Image alignment for cameras with different frame counts  
3. CAN bus data integration
4. Pre-computed BEV feature generation for efficient training

Author: UniAD Integration Project
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
from tqdm import tqdm
import glob
from timestamp_extract import load_master_timestamps, find_closest_timestamp

# Add UniAD paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/projects/mmdet3d_plugin')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')

# Import enhanced BEVFormer integration
from bevformer_integration_6cam import BEVFormerDataProcessor

# Import BEVFormer model components
from mmdet3d.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint

# Import UniAD plugin modules to register models
try:
    import projects.mmdet3d_plugin.uniad
except ImportError:
    logger.warning("Could not import UniAD plugin modules")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealBEVFormerLoader:
    """
    Real BEVFormer model loader for authentic feature extraction.
    Uses the enhanced integration from bevformer_integration_6cam.py.
    """
    
    def __init__(self, device='cuda'):
        """
        Initialize real BEVFormer with enhanced data processor.
        
        Args:
            device: Device for model inference ('cuda' or 'cpu')
        """
        self.device = device
        self.bev_h, self.bev_w = 200, 200
        self.feature_dim = 256
        
        # Initialize enhanced BEVFormer data processor
        logger.info("Initializing BEVFormer data processor...")
        self.bevformer_processor = BEVFormerDataProcessor(device=device)
        
        # Load real BEVFormer model
        self.bevformer_model = self._load_real_bevformer()
        
        logger.info(f"RealBEVFormerLoader initialized on device: {device}")
        
    def _load_real_bevformer(self):
        """
        Load pre-trained BEVFormer model using ITRI 6-camera configuration.
        
        Returns:
            Loaded BEVFormer model
        """
        try:
            # Load ITRI 6-camera config
            config_path = '/home/bryan/Desktop/Allen/UniAD/semantic-map/config_itri_6cam.py'
            cfg = Config.fromfile(config_path)
            
            # Build model
            model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
            
            # Load pre-trained weights (UniAD checkpoint with full architecture)
            checkpoint_path = '/home/bryan/Desktop/Allen/UniAD/ckpts/uniad_base_track_map.pth'
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading BEVFormer checkpoint: {checkpoint_path}")
                load_checkpoint(model, checkpoint_path, map_location=self.device)
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                logger.info("Using randomly initialized BEVFormer model")
            
            model.to(self.device)
            model.eval()
            
            logger.info("BEVFormer model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load BEVFormer model: {e}")
            raise
    
    def extract_feat(self, image_paths, canbus_data, frame_index=0):
        """
        Extract real BEV features using BEVFormer model.
        
        Args:
            image_paths: Dict mapping camera names to image file paths
            canbus_data: CAN bus data dictionary
            frame_index: Frame index to process
            
        Returns:
            BEV features tensor [1, H*W, C] = [1, 40000, 256]
        """
        try:
            # Debug: Log input data info
            logger.info(f"Processing frame {frame_index}")
            logger.info(f"Image paths: {list(image_paths.keys())}")
            logger.info(f"CAN bus data shape: {canbus_data['can_bus'].shape}")
            
            # Run BEVFormer inference using enhanced processor
            bev_embed = self.bevformer_processor.run_bevformer_inference(
                self.bevformer_model, image_paths, canbus_data, frame_index
            )
            
            # Debug: Log intermediate results
            logger.info(f"Raw BEV embed shape: {bev_embed.shape}")
            logger.info(f"BEV embed stats: min={bev_embed.min():.6f}, max={bev_embed.max():.6f}, mean={bev_embed.mean():.6f}")
            
            # Ensure correct output format [1, H*W, C]
            if bev_embed.dim() == 4:  # [B, C, H, W]
                batch_size, channels, height, width = bev_embed.shape
                bev_features = bev_embed.view(batch_size, channels, height * width)
                bev_features = bev_features.permute(0, 2, 1)  # [B, H*W, C]
            elif bev_embed.dim() == 3:  # Already [B, H*W, C]
                bev_features = bev_embed
            else:
                raise ValueError(f"Unexpected BEV tensor shape: {bev_embed.shape}")
            
            logger.info(f"Final BEV features: shape={bev_features.shape}, min={bev_features.min():.6f}, max={bev_features.max():.6f}")
            return bev_features
            
        except Exception as e:
            logger.error(f"Failed to extract BEV features: {e}")
            raise
    
    def forward(self, image_paths, canbus_data, frame_index=0):
        """Forward pass compatibility method."""
        return self.extract_feat(image_paths, canbus_data, frame_index)


class ITRI_CameraConfig:
    """
    ITRI 4-camera + 2-dummy camera configuration for UniAD compatibility.
    Manages calibration parameters and coordinate transformations.
    """
    
    def __init__(self):
        """Initialize camera configuration with ITRI calibration parameters."""
        
        # ITRI 4-camera setup (real cameras)
        self.real_cameras = {
            'CAM_FRONT': {
                'data_path': 'lucid_cameras_x00.gige_100_f_hdr.h265',
                'sensor2ego_translation': [0.005, -0.125, -0.171],
                'sensor2ego_rotation': [-0.525, 0.509, -0.491, -0.495],
                'cam_intrinsic': [657.904403, 0.0, 714.717385597, 0.0, 658.500765392, 463.1351298, 0.0, 0.0, 1.0],
                'is_dummy': False,
                'image_count': 124
            },
            'CAM_FRONT_LEFT': {
                'data_path': 'lucid_cameras_x01.gige_100_fl_hdr.h265', 
                'sensor2ego_translation': [0.213, -0.073, -0.671],
                'sensor2ego_rotation': [0.691, -0.169, 0.162, 0.684],
                'cam_intrinsic': [660.05027892, 0.0, 720.216502034, 0.0, 660.349325206, 467.66874548, 0.0, 0.0, 1.0],
                'is_dummy': False,
                'image_count': 124
            },
            'CAM_FRONT_RIGHT': {
                'data_path': 'lucid_cameras_x01.gige_100_fr_hdr.h265',
                'sensor2ego_translation': [-0.223, -0.081, -0.654],
                'sensor2ego_rotation': [-0.177, 0.688, -0.682, -0.176],
                'cam_intrinsic': [659.20444086, 0.0, 713.833685513, 0.0, 659.864183739, 455.950517493, 0.0, 0.0, 1.0],
                'is_dummy': False,
                'image_count': 110,  # Needs alignment
                'alignment_factor': 124/110  # α = 1.127
            },
            'CAM_BACK': {
                'data_path': 'lucid_cameras_x00.gige_60_b_hdr.h265',
                'sensor2ego_translation': [-0.011, -0.034, -2.469],
                'sensor2ego_rotation': [0.508, 0.513, -0.486, 0.493],
                'cam_intrinsic': [1033.063670, 0.0, 730.505500, 0.0, 1034.542542, 468.282976, 0.0, 0.0, 1.0],
                'is_dummy': False,
                'image_count': 124
            }
        }
        
        # Dummy cameras for 6-camera compatibility
        self.dummy_cameras = {
            'CAM_BACK_LEFT': {
                'data_path': None,
                'sensor2ego_translation': [-0.011, -0.034, -2.469],  # Same as back
                'sensor2ego_rotation': [0.608, 0.413, -0.386, 0.593],  # Rotated ~30° left
                'cam_intrinsic': [1033.063670, 0.0, 730.505500, 0.0, 1034.542542, 468.282976, 0.0, 0.0, 1.0],
                'is_dummy': True
            },
            'CAM_BACK_RIGHT': {
                'data_path': None, 
                'sensor2ego_translation': [-0.011, -0.034, -2.469],  # Same as back
                'sensor2ego_rotation': [0.408, 0.613, -0.586, 0.393],  # Rotated ~30° right
                'cam_intrinsic': [1033.063670, 0.0, 730.505500, 0.0, 1034.542542, 468.282976, 0.0, 0.0, 1.0],
                'is_dummy': True
            }
        }
        
        # Combined camera configuration
        self.all_cameras = {**self.real_cameras, **self.dummy_cameras}
        
        # Camera order for BEVFormer (must match expected order)
        self.camera_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 
                           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        
        # Reference camera for frame alignment
        self.reference_camera = 'CAM_FRONT'
        self.total_frames = 124
        
    def get_camera_config(self, camera_name: str) -> Dict:
        """Get configuration for specific camera."""
        return self.all_cameras.get(camera_name, {})
        
    def is_dummy_camera(self, camera_name: str) -> bool:
        """Check if camera is dummy."""
        return self.all_cameras.get(camera_name, {}).get('is_dummy', False)
        
    def needs_alignment(self, camera_name: str) -> bool:
        """Check if camera needs index alignment."""
        return 'alignment_factor' in self.all_cameras.get(camera_name, {})


class ImageAligner:
    """
    Handles image index alignment for cameras with different frame counts.
    Implements the alignment strategy from image_allignment.md
    """
    
    def __init__(self, camera_config: ITRI_CameraConfig):
        self.config = camera_config
        
    def align_index(self, camera_name: str, original_index: int) -> int:
        """
        Align image index for camera to reference timeline.
        
        Args:
            camera_name: Name of camera
            original_index: Original frame index (0-based)
            
        Returns:
            Aligned frame index
        """
        camera_info = self.config.get_camera_config(camera_name)
        
        if not self.config.needs_alignment(camera_name):
            return original_index
            
        # Apply scaling factor and round to nearest integer
        alignment_factor = camera_info['alignment_factor']
        aligned_index = round(original_index * alignment_factor)
        
        # Ensure within bounds
        max_index = camera_info.get('image_count', 110) - 1
        aligned_index = min(aligned_index, max_index)
        
        return aligned_index
    
    def get_aligned_filename(self, camera_name: str, frame_index: int) -> str:
        """
        Get aligned filename for camera at given frame index.
        
        Args:
            camera_name: Name of camera
            frame_index: Frame index in reference timeline (0-based)
            
        Returns:
            Aligned filename
        """
        camera_info = self.config.get_camera_config(camera_name)
        data_path = camera_info['data_path']
        
        if self.config.needs_alignment(camera_name):
            # For frontright_100, map reference frame to actual frame
            actual_index = round(frame_index / camera_info['alignment_factor'])
            actual_index = min(actual_index, camera_info['image_count'] - 1)
        else:
            actual_index = frame_index
            
        # New dataset uses timestamp-named images; filename resolution is handled in extractor
        filename = f"frame_{actual_index+1:03d}.jpg"
            
        return filename


class BEVFormerFeatureExtractor:
    """
    Main BEV feature extraction class using pre-trained BEVFormer model.
    Handles 4→6 camera padding, image alignment, and feature generation.
    """
    
    def __init__(self, 
                 checkpoint_path: str = '/home/bryan/Desktop/Allen/UniAD/ckpts/uniad_base_track_map.pth',
                 config_path: str = None,
                 device: str = 'cuda:0',
                 timestamp_file: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl',
                 canbus_dir: Optional[str] = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus',
                 canbus_file: Optional[str] = None,
                 tolerance: float = 0.5,
                 images_root: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image'):
        """
        Initialize BEVFormer feature extractor.
        
        Args:
            checkpoint_path: Path to pre-trained BEVFormer checkpoint
            config_path: Path to model config (will use default if None)
            device: Device to run model on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        self.camera_config = ITRI_CameraConfig()
        self.image_aligner = ImageAligner(self.camera_config)
        
        # Initialize real BEVFormer loader
        self.model = RealBEVFormerLoader(device=str(self.device))

        # Load global master timestamps
        self.master_timestamps, self.master_hz = load_master_timestamps(timestamp_file)
        logger.info(f"Loaded master timestamps: {len(self.master_timestamps)} @ {self.master_hz} Hz")

        # CAN bus configuration
        self.canbus_dir = canbus_dir
        self.canbus_file = canbus_file
        self.match_tolerance = tolerance
        self.images_root = images_root
        
        logger.info(f"BEVFormer extractor initialized on {self.device}")
        
    # Model loading is now handled by RealBEVFormerLoader
        
    def load_camera_images(self, 
                          frame_index: int, 
                          images_root: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/images') -> Dict[str, np.ndarray]:
        """
        Load synchronized camera images for given frame index.
        
        Args:
            frame_index: Frame index (0-based)
            images_root: Root directory containing camera images
            
        Returns:
            Dictionary of camera images
        """
        images = {}
        
        for camera_name in self.camera_config.camera_order:
            if self.camera_config.is_dummy_camera(camera_name):
                # Create black dummy image (900x1600x3)
                images[camera_name] = np.zeros((900, 1600, 3), dtype=np.uint8)
            else:
                # Load real camera image
                camera_info = self.camera_config.get_camera_config(camera_name)
                data_path = camera_info['data_path']
                
                # Get aligned filename
                filename = self.image_aligner.get_aligned_filename(camera_name, frame_index)
                image_path = os.path.join(images_root, data_path, filename)
                
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        images[camera_name] = image
                    else:
                        logger.warning(f"Failed to load image: {image_path}")
                        images[camera_name] = np.zeros((900, 1600, 3), dtype=np.uint8)
                else:
                    logger.warning(f"Image not found: {image_path}")
                    images[camera_name] = np.zeros((900, 1600, 3), dtype=np.uint8)
                    
        return images
        
    def load_canbus_data(self, 
                        frame_index: Optional[int] = None,
                        canbus_path: Optional[str] = None) -> Dict:
        """
        Load CAN bus data dictionary for BEVFormerDataProcessor.
        
        Args:
            frame_index: Frame index (not used, loads complete dataset)
            canbus_path: Path to CAN bus pickle file
            
        Returns:
            Dictionary with 'can_bus' and 'timestamps' keys
        """
        try:
            # Determine source(s)
            if canbus_path is None:
                if self.canbus_file is not None:
                    canbus_path = self.canbus_file
                elif self.canbus_dir is not None:
                    # Merge all *_can_bus.pkl files from dir
                    canbus_files = sorted(glob.glob(os.path.join(self.canbus_dir, '*_can_bus.pkl')))
                    if not canbus_files:
                        raise FileNotFoundError(f"No CAN bus files found in {self.canbus_dir}")
                    logger.info(f"Merging {len(canbus_files)} CAN bus files from {self.canbus_dir}")
                    can_list = []
                    ts_list = []
                    for fpath in canbus_files:
                        with open(fpath, 'rb') as f:
                            d = pickle.load(f)
                        can_list.append(d['can_bus'])
                        ts_list.append(d['timestamps'])
                    can_all = np.concatenate(can_list, axis=0)
                    ts_all = np.concatenate(ts_list, axis=0)
                else:
                    raise ValueError("No CAN bus source provided")
            
            # Load single file if provided
            if canbus_path is not None:
                logger.info(f"Loading CAN bus data from: {canbus_path}")
                with open(canbus_path, 'rb') as f:
                    d = pickle.load(f)
                can_all = d['can_bus']
                ts_all = d['timestamps']

            logger.info(f"Loaded raw CAN bus: {can_all.shape}, timestamps: {ts_all.shape}")

            # If CAN bus timestamps already align with master, bypass re-alignment
            ts_all = np.asarray(ts_all, dtype=float)
            master = np.asarray(self.master_timestamps, dtype=float)
            if len(ts_all) == len(master) and np.max(np.abs(ts_all - master)) <= self.match_tolerance:
                aligned_can = can_all
                aligned_ts = ts_all
                logger.info("CAN bus already aligned to master timeline; skipping re-alignment")
            else:
                # Align CAN bus rows to master timestamps
                aligned_rows = []
                aligned_ts = []
                for idx, tgt in enumerate(master):
                    closest_time, closest_idx = find_closest_timestamp(float(tgt), ts_all, self.match_tolerance)
                    if closest_time is None:
                        aligned_rows.append(np.zeros((18,), dtype=float))
                        aligned_ts.append(float(tgt))
                    else:
                        aligned_rows.append(can_all[closest_idx])
                        aligned_ts.append(float(closest_time))
                aligned_can = np.vstack(aligned_rows)
            self._cached_canbus_data = {
                'can_bus': aligned_can,
                'timestamps': np.asarray(aligned_ts, dtype=float),
                'master_timestamps': np.asarray(self.master_timestamps, dtype=float),
                'tolerance': float(self.match_tolerance)
            }

            logger.info(f"Aligned CAN bus to master timeline: {aligned_can.shape}")
            return self._cached_canbus_data
                
        except Exception as e:
            logger.error(f"Failed to load CAN bus data: {e}")
            # Return default data structure compatible with BEVFormerDataProcessor
            default_canbus = np.zeros((124, 18), dtype=np.float32)
            default_timestamps = np.arange(124, dtype=np.float64)
            return {
                'can_bus': default_canbus,
                'timestamps': default_timestamps
            }
            
    def prepare_model_input(self, 
                           images: Dict[str, np.ndarray], 
                           canbus_data: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Prepare model input tensors from images and CAN bus data.
        
        Args:
            images: Dictionary of camera images
            canbus_data: 18-dimensional CAN bus array
            
        Returns:
            Model input dictionary
        """
        # Prepare camera features
        camera_features = []
        camera_masks = []
        target_size = (224, 224)  # Resize all images to this size
        
        for camera_name in self.camera_config.camera_order:
            image = images[camera_name]
            
            # Resize image to target size
            image_resized = cv2.resize(image, target_size)
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_resized).float().permute(2, 0, 1) / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
            
            if self.camera_config.is_dummy_camera(camera_name):
                # Dummy camera: create attention mask to zero out
                mask = torch.ones(1, target_size[0], target_size[1], 
                                dtype=torch.bool, device=self.device)
            else:
                # Real camera: no masking
                mask = torch.zeros(1, target_size[0], target_size[1], 
                                 dtype=torch.bool, device=self.device)
            
            camera_features.append(image_tensor)
            camera_masks.append(mask)
            
        # Stack camera inputs
        camera_features = torch.stack(camera_features, dim=1)  # [1, 6, 3, 224, 224]
        camera_masks = torch.stack(camera_masks, dim=1)        # [1, 6, 224, 224]
        
        # Prepare CAN bus data
        canbus_tensor = torch.from_numpy(canbus_data).float().unsqueeze(0).to(self.device)  # [1, 18]
        
        return {
            'camera_features': camera_features,
            'camera_masks': camera_masks,
            'canbus_data': canbus_tensor
        }
        
    def extract_bev_features(self, image_paths: Dict[str, str], canbus_data: Dict, frame_index: int) -> torch.Tensor:
        """
        Extract BEV features using real BEVFormer model via RealBEVFormerLoader.
        
        Args:
            image_paths: Dictionary mapping camera names to image file paths
            canbus_data: CAN bus data dictionary
            frame_index: Frame index to process
            
        Returns:
            BEV features tensor [1, 40000, 256]
        """
        try:
            # Use RealBEVFormerLoader to extract authentic BEV features
            bev_features = self.model.extract_feat(
                image_paths=image_paths,
                canbus_data=canbus_data,
                frame_index=frame_index
            )
            
            # Ensure correct output shape [1, 40000, 256]
            expected_shape = (1, 40000, 256)
            if bev_features.shape != expected_shape:
                logger.warning(f"BEV feature shape mismatch: got {bev_features.shape}, expected {expected_shape}")
                
                # Reshape if needed
                if bev_features.numel() == 40000 * 256:
                    bev_features = bev_features.view(expected_shape)
                else:
                    logger.error(f"Cannot reshape tensor with {bev_features.numel()} elements to {expected_shape}")
                    raise ValueError(f"Invalid BEV feature tensor size: {bev_features.shape}")
                    
            return bev_features
                
        except Exception as e:
            logger.error(f"BEV feature extraction failed: {e}")
            # Return dummy features for debugging
            return torch.zeros(1, 40000, 256, device=self.device)
            
    def process_frame(self, frame_index: int) -> Tuple[torch.Tensor, Dict]:
        """
        Process single frame to extract BEV features.
        
        Args:
            frame_index: Frame index to process
            
        Returns:
            BEV features and metadata
        """
        # Build image paths for the frame using timestamp-based filenames
        image_paths = {}
        ts = float(self.master_timestamps[frame_index]) if frame_index < len(self.master_timestamps) else None
        ts_ns = int(round(ts * 1e9)) if ts is not None else None
        for cam_name, cam_config in self.camera_config.real_cameras.items():
            if ts_ns is None:
                continue
            subdir = cam_config['data_path']
            image_path = os.path.join(self.images_root, subdir, f"{ts_ns}.jpg")
            image_paths[cam_name] = image_path
        
        # If any required image is missing, skip inference and output zeros
        missing_images = [cam for cam, p in image_paths.items() if not os.path.exists(p)]
        if len(missing_images) > 0 or len(image_paths) == 0:
            bev_features = torch.zeros(1, 40000, 256, device=self.device)
            skipped_inference = True
        else:
            # Load CAN bus data (aligned to master timeline)
            canbus_data = self.load_canbus_data(frame_index)
            
            # Extract BEV features using new interface
            bev_features = self.extract_bev_features(image_paths, canbus_data, frame_index)
            skipped_inference = False
        
        # Prepare metadata
        metadata = {
            'frame_index': frame_index,
            'timestamp': float(self.master_timestamps[frame_index]) if frame_index < len(self.master_timestamps) else None,
            'missing_images': missing_images,
            'skipped_inference': skipped_inference,
            'bev_shape': list(bev_features.shape),
            'dummy_cameras': list(self.camera_config.dummy_cameras.keys()),
            'extraction_time': datetime.now().isoformat(),
            'image_paths': image_paths
        }
        
        return bev_features, metadata
        
    def extract_all_features(self, 
                           output_dir: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bev_features',
                           start_frame: int = 0,
                           end_frame: Optional[int] = None) -> Dict:
        """
        Extract BEV features for all frames.
        
        Args:
            output_dir: Directory to save extracted features
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all frames)
            
        Returns:
            Extraction summary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if end_frame is None:
            end_frame = self.camera_config.total_frames
            
        extraction_log = {
            'total_frames': end_frame - start_frame,
            'processed_frames': 0,
            'failed_frames': [],
            'output_directory': output_dir,
            'start_time': datetime.now().isoformat(),
            'camera_config': {
                'real_cameras': list(self.camera_config.real_cameras.keys()),
                'dummy_cameras': list(self.camera_config.dummy_cameras.keys()),
                'alignment_strategy': 'none'
            }
        }
        
        logger.info(f"Starting BEV feature extraction: frames {start_frame}-{end_frame-1}")
        
        for frame_idx in tqdm(range(start_frame, end_frame), desc="Extracting BEV features"):
            try:
                # Process frame
                bev_features, metadata = self.process_frame(frame_idx)
                
                # Save BEV features
                feature_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_bev.pt")
                torch.save(bev_features.cpu(), feature_path)
                
                # Save metadata
                meta_path = os.path.join(output_dir, f"frame_{frame_idx:06d}_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                extraction_log['processed_frames'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {e}")
                extraction_log['failed_frames'].append({
                    'frame_index': frame_idx,
                    'error': str(e)
                })
                
        extraction_log['end_time'] = datetime.now().isoformat()
        
        # Save extraction log
        log_path = os.path.join(output_dir, 'extraction_log.json')
        with open(log_path, 'w') as f:
            json.dump(extraction_log, f, indent=2)
            
        logger.info(f"BEV extraction completed: {extraction_log['processed_frames']}/{extraction_log['total_frames']} frames")
        
        return extraction_log


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ITRI BEVFormer Feature Extractor')
    parser.add_argument('--start-frame', type=int, default=0, help='Starting frame index')
    parser.add_argument('--end-frame', type=int, default=None, help='Ending frame index')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bev_features',
                       help='Output directory for BEV features')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/bryan/Desktop/Allen/UniAD/ckpts/uniad_base_track_map.pth',
                       help='UniAD checkpoint path')
    parser.add_argument('--timestamps', type=str,
                        default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/timestamps/continuous_timestamps.pkl',
                        help='Path to global master timestamps pickle')
    parser.add_argument('--images-root', type=str,
                        default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/image',
                        help='Root directory containing camera image subdirectories')
    parser.add_argument('--canbus-dir', type=str,
                        default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus',
                        help='Directory containing *_can_bus.pkl files')
    parser.add_argument('--canbus-file', type=str, default=None,
                        help='Optional single CAN bus file to use instead of directory')
    parser.add_argument('--tolerance', type=float, default=0.5,
                        help='Timestamp matching tolerance for CAN bus alignment (seconds)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = BEVFormerFeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=args.device,
        timestamp_file=args.timestamps,
        canbus_dir=args.canbus_dir,
        canbus_file=args.canbus_file,
        tolerance=args.tolerance,
        images_root=args.images_root
    )
    
    # Extract features
    extraction_log = extractor.extract_all_features(
        output_dir=args.output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
    
    print(f"\nExtraction Summary:")
    print(f"Total frames processed: {extraction_log['processed_frames']}")
    print(f"Failed frames: {len(extraction_log['failed_frames'])}")
    print(f"Output directory: {extraction_log['output_directory']}")
    
    if extraction_log['failed_frames']:
        print(f"Failed frames: {[f['frame_index'] for f in extraction_log['failed_frames']]}")


if __name__ == '__main__':
    main()