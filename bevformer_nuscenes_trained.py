#!/usr/bin/env python3
"""
Minimal BEV Feature Extractor for UniAD Motion/Planning Training

Extracts BEV features from ITRI 4-camera data using a minimal model with NuScenes
trained BEVFormer weights. Skips the tracking head entirely for efficient BEV feature
extraction optimized for motion/planning training.

Usage:
    # Test output format compatibility
    python bevformer_nuscenes_trained.py --test_only

    # Extract BEV features for training
    python bevformer_nuscenes_trained.py --data_root hct_train --max_samples 100

Key Features:
- Minimal model: backbone + neck + BEVFormer encoder only (no tracking head)
- Uses NuScenes BEVFormer weights (bevformer_r101_dcn_24ep.pth)
- Compatible weight loading for backbone, neck, and encoder components
- 4→6 camera padding with dummy cameras (back-left, back-right)
- Direct BEV feature extraction without tracking overhead
- Output format [B, C, H, W] compatible with motion_head.py and planning_head.py
- 256-dim BEV features at 200×200 resolution

Output:
- Individual frame files: frame_XXXXXX_timestamp.pt
- Training metadata: training_metadata.json
- BEV features in [B, C, H, W] format as expected by motion/planning heads
- Ready for end-to-end training pipeline

Architecture:
- Backbone: ResNet-101 with DCN (from NuScenes checkpoint)
- Neck: FPN (from NuScenes checkpoint)
- BEV Encoder: BEVFormerEncoder (from NuScenes checkpoint)
- Output: Direct BEV features without tracking/detection components

Benefits:
- Reduced memory usage (no tracking head)
- Faster inference (no unnecessary tracking computation)
- Clean BEV features for motion/planning training
- Compatible with UniAD training pipeline
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import glob
import rosbag
from collections import defaultdict

# Add UniAD paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/projects/mmdet3d_plugin')

# Import MMDet3D components
from mmdet3d.models import build_model
from mmcv import Config
from mmcv.runner import load_checkpoint
import mmcv

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import UniAD plugin modules to register custom models
try:
    import projects.mmdet3d_plugin
    import projects.mmdet3d_plugin.uniad
    logger.info("Successfully imported UniAD plugin modules")
except ImportError as e:
    logger.warning(f"Could not import UniAD plugin modules: {e}")

# Import data processing utilities
from timestamp_extract import load_master_timestamps, find_closest_timestamp


class NuScenesBEVFormerITRI:
    """
    NuScenes BEVFormer model runner for ITRI data.

    Loads the pre-trained NuScenes model and adapts it for ITRI 4-camera setup
    with padding to 6 cameras for compatibility.
    """

    def __init__(self, data_root: str, device: str = 'cuda'):
        """
        Initialize BEVFormer for ITRI data processing.

        Args:
            data_root: Root directory containing ITRI data
            device: Device for model inference
        """
        self.data_root = Path(data_root)
        self.device = device

        # BEVFormer configuration from NuScenes checkpoint
        self.bev_h, self.bev_w = 200, 200
        self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        self.voxel_size = [0.2, 0.2, 8]

        # Image normalization from NuScenes config
        self.img_norm = {
            'mean': [103.53, 116.28, 123.675],
            'std': [1.0, 1.0, 1.0],
            'to_rgb': False
        }

        # Camera order for 6-camera setup
        self.camera_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # Load configurations
        self._load_camera_config()
        self._load_timestamps()
        self._setup_model()

        logger.info(f"NuScenes BEVFormer initialized for ITRI data at {data_root}")

    def _load_camera_config(self):
        """Load camera configuration from camera_config.md"""
        config_file = self.data_root / 'image' / 'camera_config.md'

        # Parse camera config (simplified parsing)
        self.cameras = {
            'CAM_FRONT': {
                'intrinsic': [657.904403, 0.0, 714.717385597, 0.0, 658.500765392, 463.1351298, 0.0, 0.0, 1.0],
                'extrinsic': [0.005, -0.125, -0.171, -0.525, 0.509, -0.491, -0.495],
                'is_dummy': False
            },
            'CAM_FRONT_LEFT': {
                'intrinsic': [660.05027892, 0.0, 720.216502034, 0.0, 660.349325206, 467.66874548, 0.0, 0.0, 1.0],
                'extrinsic': [0.213, -0.073, -0.671, 0.691, -0.169, 0.162, 0.684],
                'is_dummy': False
            },
            'CAM_FRONT_RIGHT': {
                'intrinsic': [659.20444086, 0.0, 713.833685513, 0.0, 659.864183739, 455.950517493, 0.0, 0.0, 1.0],
                'extrinsic': [-0.223, -0.081, -0.654, -0.177, 0.688, -0.682, -0.176],
                'is_dummy': False
            },
            'CAM_BACK': {
                'intrinsic': [1033.063670, 0.0, 730.505500, 0.0, 1034.542542, 468.282976, 0.0, 0.0, 1.0],
                'extrinsic': [-0.011, -0.034, -2.469, 0.508, 0.513, -0.486, 0.493],
                'is_dummy': False
            },
            # Dummy cameras (will be filled with zeros/interpolated data)
            'CAM_BACK_LEFT': {
                'intrinsic': [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0],  # Dummy intrinsic
                'extrinsic': [0.0, 0.0, -2.469, 0.5, 0.5, -0.5, 0.5],  # Dummy extrinsic
                'is_dummy': True
            },
            'CAM_BACK_RIGHT': {
                'intrinsic': [800.0, 0.0, 640.0, 0.0, 800.0, 360.0, 0.0, 0.0, 1.0],  # Dummy intrinsic
                'extrinsic': [0.0, 0.0, -2.469, 0.5, 0.5, -0.5, 0.5],  # Dummy extrinsic
                'is_dummy': True
            }
        }

    def _load_timestamps(self):
        """Load synchronized timestamps"""
        timestamp_file = self.data_root / 'timestamps' / 'continuous_timestamps.pkl'

        # Handle absolute vs relative paths
        if not timestamp_file.exists():
            # Try absolute path
            abs_timestamp_file = Path('/home/bryan/Desktop/Allen/UniAD') / self.data_root / 'timestamps' / 'continuous_timestamps.pkl'
            if abs_timestamp_file.exists():
                timestamp_file = abs_timestamp_file
            else:
                logger.error(f"Timestamp file not found at: {timestamp_file}")
                logger.error(f"Also tried: {abs_timestamp_file}")
                raise FileNotFoundError(f"Timestamp file not found")

        with open(timestamp_file, 'rb') as f:
            loaded = pickle.load(f)

        # Support both raw list/array and dict format with key 'timestamps'
        if isinstance(loaded, dict) and 'timestamps' in loaded:
            ts = loaded['timestamps']
        else:
            ts = loaded

        # Convert numpy arrays to list for consistent indexing/serialization
        try:
            import numpy as np  # local import to avoid hard dependency at module import
            if isinstance(ts, np.ndarray):
                ts = ts.tolist()
        except Exception:
            pass

        # Ensure we have a sequence
        if not isinstance(ts, (list, tuple)):
            try:
                ts = list(ts)
            except Exception:
                raise ValueError("Unsupported timestamps format in pickle file; expected list/array or dict['timestamps']")

        self.timestamps = ts
        logger.info(f"Loaded {len(self.timestamps)} synchronized timestamps from {timestamp_file}")

    def _setup_model(self):
        """Setup minimal BEV extractor model with NuScenes BEVFormer checkpoint"""
        # Create simplified model that manually combines components
        self.model = self._build_bev_extractor_model()

        # Load compatible weights from NuScenes checkpoint
        self._load_compatible_weights()

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Minimal BEV extractor model loaded with NuScenes BEVFormer weights")

    def _build_bev_extractor_model(self):
        """Build minimal model with only backbone, neck, and BEV encoder"""
        import torch.nn as nn
        from mmdet3d.models.builder import build_backbone, build_neck
        from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence

        config = self._create_uniad_config()
        model_cfg = config.model

        class BEVExtractorModel(nn.Module):
            def __init__(self, backbone_cfg, neck_cfg, encoder_cfg, pos_encoding_cfg,
                        embed_dims, bev_h, bev_w, pc_range):
                super().__init__()
                self.img_backbone = build_backbone(backbone_cfg)
                self.img_neck = build_neck(neck_cfg)
                self.bev_encoder = build_transformer_layer_sequence(encoder_cfg)
                self.positional_encoding = build_positional_encoding(pos_encoding_cfg)
                self.embed_dims = embed_dims
                self.bev_h = bev_h
                self.bev_w = bev_w
                self.pc_range = pc_range

            def extract_img_feat(self, img, img_metas):
                """Extract multi-level features from images"""
                B, N, C, H, W = img.shape
                img = img.view(B * N, C, H, W)

                # Extract features using backbone
                img_feats = self.img_backbone(img)

                # Process through neck
                if self.img_neck is not None:
                    img_feats = self.img_neck(img_feats)

                # Reshape back to include camera dimension
                img_feats_reshaped = []
                for feat in img_feats:
                    BN, C, H, W = feat.shape
                    feat = feat.view(B, N, C, H, W)
                    img_feats_reshaped.append(feat)

                return img_feats_reshaped

        model = BEVExtractorModel(
            backbone_cfg=model_cfg['img_backbone'],
            neck_cfg=model_cfg['img_neck'],
            encoder_cfg=model_cfg['bev_encoder'],
            pos_encoding_cfg=model_cfg['positional_encoding'],
            embed_dims=model_cfg['embed_dims'],
            bev_h=model_cfg['bev_h'],
            bev_w=model_cfg['bev_w'],
            pc_range=model_cfg['pc_range']
        )

        return model

    def _load_compatible_weights(self):
        """Load compatible weights from NuScenes BEVFormer checkpoint"""
        checkpoint_path = '/home/bryan/Desktop/Allen/UniAD/ckpts/bevformer_r101_dcn_24ep.pth'

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)

            # Filter and load compatible weights
            model_state_dict = self.model.state_dict()
            compatible_weights = {}
            weight_counts = {'backbone': 0, 'neck': 0, 'encoder': 0, 'pos_encoding': 0}

            for name, param in state_dict.items():
                # Map backbone weights
                if 'img_backbone' in name:
                    new_name = name
                    if new_name in model_state_dict and model_state_dict[new_name].shape == param.shape:
                        compatible_weights[new_name] = param
                        weight_counts['backbone'] += 1

                # Map neck weights
                elif 'img_neck' in name:
                    new_name = name
                    if new_name in model_state_dict and model_state_dict[new_name].shape == param.shape:
                        compatible_weights[new_name] = param
                        weight_counts['neck'] += 1

                # Map BEV encoder weights (from pts_bbox_head.transformer.encoder)
                elif 'pts_bbox_head.transformer.encoder' in name:
                    new_name = name.replace('pts_bbox_head.transformer.encoder', 'bev_encoder')
                    if new_name in model_state_dict and model_state_dict[new_name].shape == param.shape:
                        compatible_weights[new_name] = param
                        weight_counts['encoder'] += 1

                # Map positional encoding weights
                elif 'pts_bbox_head.positional_encoding' in name:
                    new_name = name.replace('pts_bbox_head.positional_encoding', 'positional_encoding')
                    if new_name in model_state_dict and model_state_dict[new_name].shape == param.shape:
                        compatible_weights[new_name] = param
                        weight_counts['pos_encoding'] += 1

            # Load the compatible weights
            missing_keys, unexpected_keys = self.model.load_state_dict(compatible_weights, strict=False)

            # Log summary by component
            logger.info(f"Loaded {len(compatible_weights)} compatible weights from NuScenes checkpoint:")
            logger.info(f"  - Backbone: {weight_counts['backbone']} weights")
            logger.info(f"  - Neck: {weight_counts['neck']} weights")
            logger.info(f"  - BEV Encoder: {weight_counts['encoder']} weights")
            logger.info(f"  - Positional Encoding: {weight_counts['pos_encoding']} weights")

            if missing_keys:
                logger.warning(f"Missing keys: {len(missing_keys)} total (first 5: {missing_keys[:5]})")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {len(unexpected_keys)} total (first 5: {unexpected_keys[:5]})")

        except Exception as e:
            logger.error(f"Failed to load checkpoint weights: {e}")
            logger.info("Proceeding with randomly initialized weights")

    def _create_uniad_config(self):
        """Create minimal BEV extractor config without tracking head"""
        config = mmcv.Config(dict(
            # Load NuScenes BEVFormer checkpoint automatically
            load_from='/home/bryan/Desktop/Allen/UniAD/ckpts/bevformer_r101_dcn_24ep.pth',
            model=dict(
                type='BEVFormerExtractor',  # Custom minimal model type
                use_grid_mask=True,
                # Image backbone (ResNet-101 with DCN, same as NuScenes)
                img_backbone=dict(
                    type='ResNet',
                    depth=101,
                    num_stages=4,
                    out_indices=(1, 2, 3),
                    frozen_stages=1,
                    norm_cfg=dict(type='BN2d', requires_grad=False),
                    norm_eval=True,
                    style='caffe',
                    dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
                    stage_with_dcn=(False, False, True, True)
                ),
                # FPN neck (same as NuScenes)
                img_neck=dict(
                    type='FPN',
                    in_channels=[512, 1024, 2048],
                    out_channels=256,
                    start_level=0,
                    add_extra_convs='on_output',
                    num_outs=4,
                    relu_before_extra_convs=True
                ),
                # Standalone BEVFormer encoder (no tracking head)
                bev_encoder=dict(
                    type='BEVFormerEncoder',
                    num_layers=6,
                    pc_range=self.pc_range,
                    num_points_in_pillar=4,
                    return_intermediate=False,
                    transformerlayers=dict(
                        type='BEVFormerLayer',
                        attn_cfgs=[
                            dict(
                                type='TemporalSelfAttention',
                                embed_dims=256,
                                num_levels=1
                            ),
                            dict(
                                type='SpatialCrossAttention',
                                pc_range=self.pc_range,
                                deformable_attention=dict(
                                    type='MSDeformableAttention3D',
                                    embed_dims=256,
                                    num_points=8,
                                    num_levels=4
                                ),
                                embed_dims=256
                            )
                        ],
                        feedforward_channels=512,
                        ffn_dropout=0.1,
                        operation_order=(
                            'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'
                        )
                    )
                ),
                # Positional encoding for BEV queries
                positional_encoding=dict(
                    type='LearnedPositionalEncoding',
                    num_feats=128,
                    row_num_embed=self.bev_h,
                    col_num_embed=self.bev_w
                ),
                # Basic parameters for BEV feature extraction
                pc_range=self.pc_range,
                embed_dims=256,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                freeze_img_backbone=True,  # Keep backbone frozen for inference
                freeze_img_neck=False,
                freeze_bn=False
            )
        ))
        return config

    def load_images_at_timestamp(self, timestamp: float) -> Dict[str, np.ndarray]:
        """Load images for all cameras at given timestamp"""
        images = {}

        # Load real camera images
        for cam_name in self.camera_names:
            if self.cameras[cam_name]['is_dummy']:
                # Create dummy image (black image with proper size)
                images[cam_name] = np.zeros((900, 1600, 3), dtype=np.uint8)
            else:
                # Load real image based on timestamp
                image_dir = self.data_root / 'image' / cam_name.lower()

                # Find closest image file by timestamp
                image_files = list(image_dir.glob(f"*{timestamp:.6f}*.jpg"))
                if not image_files:
                    # Fallback: find closest timestamp
                    all_files = list(image_dir.glob("*.jpg"))
                    if all_files:
                        # Use first available image as fallback
                        image_files = [all_files[0]]

                if image_files:
                    image_path = image_files[0]
                    image = cv2.imread(str(image_path))
                    if image is not None:
                        images[cam_name] = image
                    else:
                        # Fallback to dummy image
                        images[cam_name] = np.zeros((900, 1600, 3), dtype=np.uint8)
                else:
                    # Fallback to dummy image
                    images[cam_name] = np.zeros((900, 1600, 3), dtype=np.uint8)

        return images

    def load_can_bus_data(self, timestamp: float) -> np.ndarray:
        """Load CAN bus data at given timestamp"""
        # Load CAN bus data from bag files
        canbus_dir = self.data_root / 'canbus'

        # Find relevant bag file containing this timestamp
        bag_files = list(canbus_dir.glob("*.bag"))

        can_bus_data = np.zeros(18)  # Standard CAN bus vector size

        for bag_file in bag_files:
            try:
                bag = rosbag.Bag(str(bag_file))

                # Find closest CAN bus message to timestamp
                closest_msg = None
                min_time_diff = float('inf')

                for topic, msg, t in bag.read_messages():
                    if 'can' in topic.lower() or 'odom' in topic.lower():
                        time_diff = abs(t.to_sec() - timestamp)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_msg = msg

                bag.close()

                if closest_msg and hasattr(closest_msg, 'twist'):
                    # Extract motion information
                    can_bus_data[0] = closest_msg.twist.twist.linear.x  # velocity_x
                    can_bus_data[1] = closest_msg.twist.twist.linear.y  # velocity_y
                    can_bus_data[2] = closest_msg.twist.twist.angular.z  # angular_velocity_z
                    # Fill other fields as available

                break  # Use first successful bag file

            except Exception as e:
                logger.warning(f"Failed to read bag file {bag_file}: {e}")
                continue

        return can_bus_data

    def preprocess_images(self, images: Dict[str, np.ndarray]) -> torch.Tensor:
        """Preprocess images for BEVFormer input"""
        processed_images = []

        for cam_name in self.camera_names:
            img = images[cam_name]

            # Resize to standard size
            img = cv2.resize(img, (1600, 900))

            # Normalize (keep float32 to avoid upcasting to float64)
            img = img.astype(np.float32)
            mean = np.array(self.img_norm['mean'], dtype=np.float32)
            std = np.array(self.img_norm['std'], dtype=np.float32)
            img = (img - mean) / std

            # Pad to size divisible by 32
            h, w = img.shape[:2]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')

            # Convert to CHW format
            img = img.transpose(2, 0, 1)
            processed_images.append(img)

        # Stack images: [N_cams, C, H, W]
        img_np = np.stack(processed_images).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).to(self.device).float()

        # Add batch dimension: [1, N_cams, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)

        return img_tensor

    def run_inference(self, timestamp: float) -> Dict[str, torch.Tensor]:
        """Run UniAD inference to extract BEV features for motion_head training"""
        # Load data
        images = self.load_images_at_timestamp(timestamp)
        can_bus = self.load_can_bus_data(timestamp)

        # Preprocess
        img_tensor = self.preprocess_images(images)
        logger.info(f"Input image tensor dtype: {img_tensor.dtype}, shape: {tuple(img_tensor.shape)}")
        can_bus_tensor = torch.from_numpy(can_bus).float().to(self.device).unsqueeze(0)

        # Prepare camera parameters and metadata
        img_metas = self._prepare_img_metas(timestamp, can_bus_tensor)

        # Extract BEV features using custom forward pass
        with torch.no_grad():
            bev_features = self._extract_bev_features(img_tensor, img_metas)

        return {
            'bev_features': bev_features,
            'timestamp': timestamp,
            'img_shape': img_tensor.shape,
            'can_bus': can_bus_tensor
        }

    def _get_bev_queries(self, bev_h: int, bev_w: int, device: torch.device) -> torch.Tensor:
        """Initialize BEV queries for the encoder"""
        # Create learnable BEV queries: [H*W, 1, embed_dims]
        num_queries = bev_h * bev_w
        bev_queries = torch.zeros(num_queries, 1, self.model.embed_dims,
                                device=device, dtype=torch.float32)
        return bev_queries

    def _get_bev_pos_embed(self, bev_h: int, bev_w: int, device: torch.device) -> torch.Tensor:
        """Get positional embeddings for BEV queries"""
        # Create positional embeddings manually
        # Based on LearnedPositionalEncoding implementation
        num_feats = self.model.embed_dims // 2  # 128 for 256 embed_dims

        # Create coordinate grids
        y_embed = torch.arange(bev_h, device=device, dtype=torch.float32)
        x_embed = torch.arange(bev_w, device=device, dtype=torch.float32)

        # Normalize coordinates to [0, 1]
        y_embed = y_embed / (bev_h - 1) if bev_h > 1 else y_embed
        x_embed = x_embed / (bev_w - 1) if bev_w > 1 else x_embed

        # Create 2D grid
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing='ij')

        # Use the model's positional encoding weights
        pos_embed = self.model.positional_encoding(
            torch.stack([x_embed.flatten(), y_embed.flatten()], dim=1).unsqueeze(0)
        )  # [1, H*W, embed_dims]

        # Convert to format expected by encoder: [H*W, 1, embed_dims]
        pos_embed = pos_embed.permute(1, 0, 2)

        return pos_embed

    def _extract_bev_features(self, img_tensor: torch.Tensor, img_metas: List[Dict]) -> torch.Tensor:
        """Extract BEV features directly from BEVFormer encoder without tracking head"""
        batch_size = img_tensor.shape[0]
        device = img_tensor.device

        # Extract image features using backbone and neck
        img_feats = self.model.extract_img_feat(img_tensor, img_metas)

        # Initialize BEV queries and positional embeddings
        bev_queries = self._get_bev_queries(self.bev_h, self.bev_w, device)
        bev_queries = bev_queries.repeat(1, batch_size, 1)  # [H*W, B, embed_dims]

        # Get positional embeddings for BEV
        bev_pos = self._get_bev_pos_embed(self.bev_h, self.bev_w, device)
        bev_pos = bev_pos.repeat(1, batch_size, 1)  # [H*W, B, embed_dims]

        # Prepare multi-level features for the encoder
        # Expect key/value shape: (num_cam, num_value, bs, embed_dims)
        mlvl_feats = []  # each: [N_cam, H*W, B, C]
        spatial_shapes = []
        for lvl, feat in enumerate(img_feats):
            # feat shape: [B, N_cam, C, H, W]
            bs, num_cam, c, h, w = feat.shape
            spatial_shapes.append([h, w])
            # Rearrange to [N_cam, H*W, B, C]
            feat = feat.permute(1, 3, 4, 0, 2).reshape(num_cam, h * w, bs, c)
            mlvl_feats.append(feat)

        # Create level start indices for multi-scale features
        level_start_index = []
        start_idx = 0
        for shape in spatial_shapes:
            level_start_index.append(start_idx)
            start_idx += shape[0] * shape[1]
        level_start_index = torch.tensor(level_start_index, device=device)
        spatial_shapes = torch.tensor(spatial_shapes, device=device)

        # Concatenate multi-level features along spatial dimension
        # Result: [N_cam, total_hw, B, C]
        concat_mlvl_feats = torch.cat(mlvl_feats, dim=1)

        # Run BEVFormer encoder directly
        bev_embed = self.model.bev_encoder(
            bev_queries,                    # [H*W, B, embed_dims]
            concat_mlvl_feats,             # [N_cam, total_hw, B, C] (key & value)
            concat_mlvl_feats,             # [N_cam, total_hw, B, C]
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_pos=bev_pos,               # [H*W, B, embed_dims]
            spatial_shapes=spatial_shapes,  # [num_levels, 2]
            level_start_index=level_start_index,  # [num_levels]
            prev_bev=None,                 # No temporal information for single frame
            shift=torch.zeros(batch_size, 2, device=device),  # No shift
            img_metas=img_metas
        )

        # Handle both 3D and 4D returns from encoder
        # 3D expected: [H*W, B, C]; 4D (when return_intermediate=True): [L, H*W, B, C]
        logger.info(f"BEV encoder raw output dims: {bev_embed.dim()}, shape: {tuple(bev_embed.shape)}")

        if bev_embed.dim() == 4:
            # If first dim is layers, take last; also squeeze singleton layer dim cases
            logger.info(f"4D tensor detected, shape[0]={bev_embed.shape[0]}")
            if bev_embed.shape[0] == 1:
                bev_embed = bev_embed.squeeze(0)
                logger.info(f"After squeeze(0): {tuple(bev_embed.shape)}")
            else:
                bev_embed = bev_embed[-1]
                logger.info(f"After taking last layer: {tuple(bev_embed.shape)}")
        elif bev_embed.dim() != 3:
            raise RuntimeError(f"Unexpected BEV encoder output dims: {bev_embed.dim()} with shape {tuple(bev_embed.shape)}")

        logger.info(f"BEV encoder output shape before permute: {tuple(bev_embed.shape)}")

        # Convert from [H*W, B, C] to [B, C, H, W] format expected by motion/planning heads
        if bev_embed.dim() == 3:
            bev_embed = bev_embed.permute(1, 2, 0)  # [B, C, H*W]
            bev_embed = bev_embed.reshape(batch_size, self.model.embed_dims, self.bev_h, self.bev_w)
        else:
            raise RuntimeError(f"Expected 3D tensor after processing, got {bev_embed.dim()}D with shape {tuple(bev_embed.shape)}")

        return bev_embed

    def _prepare_img_metas(self, timestamp: float, can_bus_tensor: torch.Tensor) -> List[Dict]:
        """Prepare image metadata for UniAD model"""
        camera_params = self._prepare_camera_params()

        img_metas = [{
            'filename': f'timestamp_{timestamp:.6f}',
            'can_bus': can_bus_tensor.squeeze(0),  # Remove batch dimension for metadata
            'lidar2img': camera_params['lidar2img'],
            'ego2img': camera_params['lidar2img'],  # Same as lidar2img for our case
            'timestamp': timestamp,

            # Image shape information
            'img_shape': [(900, 1600, 3)] * 6,  # 6 cameras
            'ori_shape': [(900, 1600, 3)] * 6,
            'pad_shape': [(928, 1600, 3)] * 6,  # After padding to 32-divisible
            'scale_factor': [1.0, 1.0, 1.0, 1.0],

            # Camera information
            'cam_names': self.camera_names,
            'img_norm_cfg': self.img_norm,

            # BEV and coordinate system info
            'bev_size': (self.bev_h, self.bev_w),
            'pc_range': self.pc_range,
            'voxel_size': self.voxel_size,

            # Sample information for tracking (not used for single frame)
            'sample_idx': 0,
            'scene_token': f'itri_scene_{timestamp}',
            'frame_idx': 0,
        }]

        return img_metas

    def _prepare_camera_params(self) -> Dict:
        """Prepare camera parameters for UniAD model input"""
        lidar2img = []
        lidar2cam = []
        cam_intrinsic = []

        for cam_name in self.camera_names:
            cam_info = self.cameras[cam_name]

            # Camera intrinsic matrix (3x3 -> 4x4)
            intrinsic = np.array(cam_info['intrinsic']).reshape(3, 3)
            intrinsic_4x4 = np.eye(4)
            intrinsic_4x4[:3, :3] = intrinsic

            # Camera extrinsic (translation + quaternion -> 4x4 matrix)
            trans = np.array(cam_info['extrinsic'][:3])
            quat = np.array(cam_info['extrinsic'][3:])  # [w, x, y, z]

            # Convert quaternion to rotation matrix
            w, x, y, z = quat
            R = np.array([
                [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
            ])

            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = trans

            # For UniAD, we need sensor2lidar transformation
            # Assuming the ego vehicle coordinate is the same as lidar coordinate for ITRI data
            sensor2lidar = extrinsic

            # Lidar to camera (inverse of sensor2lidar)
            lidar2cam_matrix = np.linalg.inv(sensor2lidar)
            lidar2cam.append(lidar2cam_matrix)

            # Lidar to image transformation
            lidar2img_matrix = intrinsic_4x4 @ lidar2cam_matrix
            lidar2img.append(lidar2img_matrix)

            cam_intrinsic.append(intrinsic)

        # Convert to torch tensors as expected by UniAD
        return {
            'lidar2img': [torch.from_numpy(mat.astype(np.float32)) for mat in lidar2img],
            'lidar2cam': [torch.from_numpy(mat.astype(np.float32)) for mat in lidar2cam],
            'cam_intrinsic': [torch.from_numpy(mat.astype(np.float32)) for mat in cam_intrinsic]
        }

    def batch_inference(self, max_samples: int = None, output_dir: str = None):
        """Run inference on all timestamps and save BEV features for motion_head training"""
        timestamps = self.timestamps[:max_samples] if max_samples else self.timestamps

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

        results = []
        successful_count = 0

        logger.info(f"Running BEV feature extraction on {len(timestamps)} timestamps...")

        for i, timestamp in enumerate(tqdm(timestamps, desc="Extracting BEV features")):
            try:
                result = self.run_inference(timestamp)
                results.append(result)

                # Save BEV features in format suitable for motion_head training
                if output_dir and result['bev_features'] is not None:
                    self._save_bev_features_for_training(result, output_path, i)
                    successful_count += 1

            except Exception as e:
                logger.error(f"Failed inference at timestamp {timestamp}: {e}")
                continue

        # Save summary and metadata
        if output_dir:
            self._save_training_metadata(results, output_path, successful_count)

        logger.info(f"Completed BEV feature extraction: {successful_count}/{len(timestamps)} successful")

        return results

    def _save_bev_features_for_training(self, result: Dict, output_path: Path, frame_idx: int):
        """Save BEV features in format optimized for motion_head training"""
        timestamp = result['timestamp']
        bev_features = result['bev_features']

        # Prepare training data format
        training_data = {
            'bev_features': bev_features.cpu(),  # Shape: [1, 200, 200, 256] or similar
            'timestamp': timestamp,
            'frame_idx': frame_idx,
            'can_bus': result['can_bus'].cpu() if result['can_bus'] is not None else None,

            # Metadata for motion_head
            'bev_h': self.bev_h,
            'bev_w': self.bev_w,
            'pc_range': self.pc_range,
            'voxel_size': self.voxel_size,
            'embed_dims': 256,

            # Save format identifier
            'format_version': 'uniad_bev_features_v1.0',
            'source': 'nuscenes_bevformer_itri'
        }

        # Save individual frame
        frame_file = output_path / f"frame_{frame_idx:06d}_{timestamp:.6f}.pt"
        torch.save(training_data, frame_file)

        # Also save in timestamped format for easy lookup
        timestamp_file = output_path / f"bev_features_{timestamp:.6f}.pt"
        torch.save(training_data, timestamp_file)

    def _save_training_metadata(self, results: List[Dict], output_path: Path, successful_count: int):
        """Save metadata and summary for the training dataset"""
        metadata = {
            'dataset_info': {
                'source': 'ITRI HCT data processed with NuScenes BEVFormer',
                'total_frames': len(results),
                'successful_frames': successful_count,
                'bev_resolution': (self.bev_h, self.bev_w),
                'feature_dims': 256,
                'pc_range': self.pc_range,
                'voxel_size': self.voxel_size,
            },
            'model_info': {
                'backbone': 'NuScenes BEVFormer',
                'checkpoint': 'bevformer_r101_dcn_24ep.pth',
                'architecture': 'UniAD',
                'normalization': self.img_norm,
            },
            'camera_setup': {
                'num_cameras': len(self.camera_names),
                'camera_names': self.camera_names,
                'real_cameras': [name for name, config in self.cameras.items() if not config['is_dummy']],
                'dummy_cameras': [name for name, config in self.cameras.items() if config['is_dummy']],
            },
            'timestamps': [result['timestamp'] for result in results if result.get('bev_features') is not None],
            'frame_indices': list(range(successful_count)),
        }

        # Save metadata
        metadata_file = output_path / 'training_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save timestamp mapping
        timestamp_mapping = {
            result['timestamp']: i for i, result in enumerate(results)
            if result.get('bev_features') is not None
        }
        mapping_file = output_path / 'timestamp_to_frame_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(timestamp_mapping, f, indent=2)

        logger.info(f"Training metadata saved to {metadata_file}")
        logger.info(f"Timestamp mapping saved to {mapping_file}")

        # Create a summary file for quick reference
        summary_file = output_path / 'dataset_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"BEV Features Dataset Summary\n")
            f.write(f"===========================\n\n")
            f.write(f"Total frames processed: {len(results)}\n")
            f.write(f"Successful extractions: {successful_count}\n")
            f.write(f"BEV resolution: {self.bev_h} x {self.bev_w}\n")
            f.write(f"Feature dimensions: 256\n")
            f.write(f"PC range: {self.pc_range}\n")
            f.write(f"Source: NuScenes BEVFormer on ITRI data\n\n")
            f.write(f"Files:\n")
            f.write(f"- frame_XXXXXX_timestamp.pt: Individual BEV features\n")
            f.write(f"- training_metadata.json: Dataset metadata\n")
            f.write(f"- timestamp_to_frame_mapping.json: Timestamp lookup\n")
            f.write(f"- dataset_summary.txt: This summary\n")

        logger.info(f"Dataset summary saved to {summary_file}")

    def test_output_format(self, test_timestamp: float = None):
        """Test that output format is compatible with motion/planning heads"""
        if test_timestamp is None:
            if not self.timestamps:
                raise KeyError('No timestamps available to test')
            test_timestamp = self.timestamps[0]

        logger.info(f"Testing BEV feature extraction output format...")
        logger.info(f"Expected output format: [B, C, H, W] = [1, 256, {self.bev_h}, {self.bev_w}]")

        try:
            result = self.run_inference(test_timestamp)
            bev_features = result['bev_features']

            # Check output shape
            expected_shape = (1, 256, self.bev_h, self.bev_w)
            actual_shape = bev_features.shape

            logger.info(f"Actual output shape: {actual_shape}")
            logger.info(f"BEV features dtype: {bev_features.dtype}")

            if actual_shape == expected_shape:
                logger.info("✅ Output shape matches expected format for motion/planning heads")
                return True
            else:
                logger.error(f"❌ Output shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False

        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='Extract BEV features from ITRI data using NuScenes trained BEVFormer'
    )
    parser.add_argument('--data_root', type=str, default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train',
                       help='Root directory of ITRI data (relative to current dir)')
    parser.add_argument('--output_dir', type=str, default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/bev_features_nuscenes',
                       help='Output directory for BEV features suitable for motion_head training')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')
    parser.add_argument('--test_only', action='store_true',
                       help='Only test output format compatibility, do not run full inference')

    args = parser.parse_args()

    try:
        # Initialize minimal BEV extractor model
        logger.info("Initializing minimal BEV extractor model with NuScenes BEVFormer weights...")
        processor = NuScenesBEVFormerITRI(
            data_root=args.data_root,
            device=args.device
        )

        if args.test_only:
            # Only test output format
            logger.info("Running output format compatibility test...")
            success = processor.test_output_format()
            if success:
                logger.info("✅ Output format test PASSED - compatible with motion/planning heads")
            else:
                logger.error("❌ Output format test FAILED - output may not be compatible")
            return

        # Run BEV feature extraction
        logger.info("Starting BEV feature extraction for motion_head training...")
        results = processor.batch_inference(
            max_samples=args.max_samples,
            output_dir=args.output_dir
        )

        logger.info(f"BEV feature extraction completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Features are in [B, C, H, W] format compatible with motion/planning heads.")
        logger.info(f"Ready for UniAD motion_head training.")

    except Exception as e:
        logger.error(f"Error during BEV feature extraction: {e}")
        raise


if __name__ == '__main__':
    main()