#!/usr/bin/env python3
"""
ITRI UniAD Integration Interface (6-Camera Padded Version)

Complete end-to-end pipeline that integrates all three ITRI modules with 6-camera setup:
1. Semantic Map → Lane Queries
2. TrackFormer → Track Queries  
3. BEVFormer → BEV Features + Memory Bridge (4 real cameras + 2 dummy cameras)

This creates a unified interface for running UniAD inference with ITRI data, 
using padded 6-camera input for compatibility with pre-trained UniAD model.

Author: Generated for UniAD Integration Project
"""

import os
import sys
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Add semantic-map to Python path
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

# Import our modules
from bevformer_integration_6cam import BEVFormerDataProcessor
from motionformer_integration import create_integrator
from itri_to_lane_query import create_converter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ITRIUniADPipeline:
    """
    Complete ITRI to UniAD integration pipeline.
    
    Orchestrates all three major components:
    - BEVFormer processing (4-camera + CAN bus → BEV features)  
    - Semantic map conversion (ITRI map → lane queries)
    - Track query integration (ROS bags → track queries)
    """
    
    def __init__(
        self,
        config: Dict = None,
        device: str = 'cuda'
    ):
        """
        Initialize ITRI UniAD pipeline.
        
        Args:
            config: Configuration dictionary
            device: Device for tensor operations
        """
        self.device = device
        self.config = config or self._default_config()
        
        # Initialize components
        self._setup_components()
        
        # Cache for loaded data
        self._data_cache = {}
        
    def _default_config(self) -> Dict:
        """Generate default configuration."""
        return {
            'data_root': '/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic',
            'semantic_map_root': '/home/bryan/Desktop/Allen/UniAD/semantic-map/data/itri',
#           'track_queries_path': '/home/bryan/Desktop/Allen/UniAD/track/track_queries/complete_track_queries.pt',
            'track_queries_path': '/home/bryan/Desktop/Allen/UniAD/track/track_queries/complete_track_queries_reduced.pt',
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            'bev_h': 200,
            'bev_w': 200,
            'embed_dims': 256,
            'num_cameras': 4,
            'camera_names': ['front_100deg', 'front_left_100deg', 'front_right_100deg', 'back_60deg']
        }
    
    def _setup_components(self):
        """Initialize all pipeline components."""
        logger.info("Setting up ITRI UniAD pipeline components...")
        
        # 1. BEVFormer data processor with memory bridge
        self.bev_processor = BEVFormerDataProcessor(
            pc_range=self.config['pc_range'],
            device=self.device
        )
        
        # 2. Semantic map converter
        converter_config = {
            'device': self.device,
            'pc_range': self.config['pc_range']
        }
        self.semantic_converter = create_converter(converter_config)
        
        # 3. MotionFormer integrator
        self.motion_integrator = create_integrator(
            converter_config=converter_config,
            device=self.device
        )
        
        logger.info("All pipeline components initialized successfully")
    
    def load_itri_data(
        self,
        frame_index: int = 0,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Load complete ITRI data for specified frame.
        
        Args:
            frame_index: Frame index to load
            use_cache: Whether to cache loaded data
            
        Returns:
            Dictionary containing all ITRI data
        """
        cache_key = f"frame_{frame_index}"
        
        if use_cache and cache_key in self._data_cache:
            logger.info(f"Using cached data for frame {frame_index}")
            return self._data_cache[cache_key]
        
        logger.info(f"Loading ITRI data for frame {frame_index}")
        
        # Load 4-camera images
        image_paths = self._get_image_paths(frame_index)
        
        # Load CAN bus data
        canbus_data = self._load_canbus_data()
        
        # Load track queries
        track_data = self._load_track_queries()
        
        # Prepare ego pose (from CAN bus data)
        ego_pose = self._extract_ego_pose(canbus_data, frame_index)
        
        itri_data = {
            'image_paths': image_paths,
            'canbus_data': canbus_data,
            'track_queries': track_data,
            'ego_pose': ego_pose,
            'frame_index': frame_index,
            'timestamp': canbus_data.get('timestamps', [frame_index])[frame_index]
        }
        
        if use_cache:
            self._data_cache[cache_key] = itri_data
            
        logger.info(f"ITRI data loaded successfully for frame {frame_index}")
        return itri_data
    
    def _get_image_paths(self, frame_index: int) -> Dict[str, str]:
        """Get image file paths for specified frame."""
        image_paths = {}
        
        for cam_name in self.config['camera_names']:
            # Construct image filename based on ITRI naming convention
            if cam_name == 'front_100deg':
                filename = f"f100_142305_{frame_index + 1}.jpg"
            elif cam_name == 'front_left_100deg':
                filename = f"fl100_142305_{frame_index + 1}.jpg"
            elif cam_name == 'front_right_100deg':
                filename = f"fr100_142305_{frame_index + 1}.jpg"
            elif cam_name == 'back_60deg':
                filename = f"b60_142305_{frame_index + 1}.jpg"
                
            image_path = os.path.join(
                self.config['data_root'], 'photo_extracted', cam_name, filename
            )
            
            if os.path.exists(image_path):
                image_paths[cam_name] = image_path
            else:
                logger.warning(f"Image not found: {image_path}")
                
        return image_paths
    
    def _load_canbus_data(self) -> Dict[str, np.ndarray]:
        """Load CAN bus data from pickle files."""
        canbus_dir = os.path.join(self.config['data_root'], 'canbus')
        
        # Look for the first available canbus file
        canbus_files = [f for f in os.listdir(canbus_dir) if f.endswith('.pkl')]
        
        if not canbus_files:
            raise FileNotFoundError(f"No CAN bus files found in {canbus_dir}")
            
        canbus_path = os.path.join(canbus_dir, canbus_files[0])
        return self.bev_processor.load_canbus_data(canbus_path)
    
    def _load_track_queries(self) -> Dict[str, torch.Tensor]:
        """Load pre-computed track queries."""
        track_path = self.config['track_queries_path']
        
        if not os.path.exists(track_path):
            logger.warning(f"Track queries not found: {track_path}")
            return self._create_empty_track_queries()
            
        track_data = torch.load(track_path, map_location=self.device)
        logger.info(f"Loaded track queries: {track_data['track_query_embeddings'].shape}")
        
        # Ensure all tensors are on the correct device
        for key, value in track_data.items():
            if torch.is_tensor(value):
                track_data[key] = value.to(self.device)
            elif key.endswith('_bbox_results') and isinstance(value, list):
                # Handle complex bbox results structures
                for i, bbox_level in enumerate(value):
                    if isinstance(bbox_level, list):
                        for j, bbox_item in enumerate(bbox_level):
                            if hasattr(bbox_item, 'to'):  # LiDARInstance3DBoxes or similar
                                track_data[key][i][j] = bbox_item.to(self.device)
                            elif torch.is_tensor(bbox_item):
                                track_data[key][i][j] = bbox_item.to(self.device)
        
        return track_data
    
    def _create_empty_track_queries(self) -> Dict[str, torch.Tensor]:
        """Create empty track queries when data is not available."""
        return {
            'track_query_embeddings': torch.zeros(1, 1, 256, device=self.device),
            'track_bbox_results': [],
            'sdc_embedding': torch.zeros(256, device=self.device),
            'sdc_track_bbox_results': []
        }
    
    def _extract_ego_pose(
        self, 
        canbus_data: Dict[str, np.ndarray], 
        frame_index: int
    ) -> Tuple[float, float, float]:
        """Extract ego pose from CAN bus data."""
        can_bus_frame = canbus_data['can_bus'][frame_index]
        
        # Extract x, y, yaw from CAN bus data
        # CAN bus format: [x, y, z, ..., yaw] (18 dimensions total)
        x, y = can_bus_frame[0], can_bus_frame[1]
        yaw = can_bus_frame[-1]  # Last element is typically yaw
        
        return (float(x), float(y), float(yaw))
    
    def run_complete_inference(
        self,
        uniad_model,
        frame_index: int = 0
    ) -> Dict[str, Any]:
        """
        Run complete end-to-end UniAD inference with ITRI data.
        
        Args:
            uniad_model: Loaded UniAD model
            frame_index: Frame index to process
            
        Returns:
            Complete inference results
        """
        logger.info(f"Starting complete UniAD inference for frame {frame_index}")
        
        # Step 1: Load ITRI data
        itri_data = self.load_itri_data(frame_index)
        
        # Step 2: Process BEVFormer (4-camera + CAN bus → BEV features)
        bev_embed = self.bev_processor.run_bevformer_inference(
            uniad_model,  # Use complete UniAD model for feature extraction
            itri_data['image_paths'],
            itri_data['canbus_data'],
            frame_index
        )
        
        # Step 3: Convert semantic map to lane queries
        lane_query, lane_query_pos = self.semantic_converter.convert(
            self.config['semantic_map_root'],
            itri_data['ego_pose']
        )
        
        # Step 4: Integrate BEV features with semantic map via memory bridge
        sample_data = self.bev_processor.prepare_sample_data(
            itri_data['image_paths'],
            itri_data['canbus_data'],
            frame_index
        )
        
        outs_seg = self.bev_processor.integrate_with_semantic_map(
            bev_embed,
            lane_query,
            lane_query_pos,
            sample_data['img_metas']
        )
        
        # Step 5: Prepare track queries
        outs_track = itri_data['track_queries']
        
        # Step 6: Run MotionFormer inference
        logger.info(f"Pre-motion head - GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"BEV embed shape: {bev_embed.shape}")
        logger.info(f"Track query shape: {outs_track['track_query_embeddings'].shape}")
        logger.info(f"Lane query shape from args_tuple[3]: {outs_seg['args_tuple'][3].shape}")
        
        with torch.no_grad():
            motion_outputs = uniad_model.motion_head.forward_test(
                bev_embed=bev_embed,
                outs_track=outs_track,
                outs_seg=outs_seg
            )
            
            # Handle different return formats from motion head
            if isinstance(motion_outputs, tuple):
                if len(motion_outputs) == 2:
                    motion_results, outs_motion = motion_outputs
                elif len(motion_outputs) == 3:
                    motion_results, outs_motion, _ = motion_outputs  # Ignore third value
                else:
                    logger.warning(f"Unexpected motion head output format: {len(motion_outputs)} values")
                    motion_results, outs_motion = motion_outputs[0], motion_outputs[1]
            else:
                # Single value return (outs_motion only)
                motion_results = None
                outs_motion = motion_outputs

            if outs_motion is not None and isinstance(outs_motion, dict):
                # Ensure track_query has a temporal dimension for n_future
                track_query = outs_motion.get('track_query')
                track_query_pos = outs_motion.get('track_query_pos')

                if track_query is not None and track_query.ndim == 3:  # [B, Q, C]
                    track_query = track_query.unsqueeze(1)  # [B, 1, Q, C]
                if track_query_pos is not None and track_query_pos.ndim == 3:
                    track_query_pos = track_query_pos.unsqueeze(1)  # [B, 1, Q, C]

                outs_motion['track_query'] = track_query
                outs_motion['track_query_pos'] = track_query_pos
        
        # DEBUG LOGGING (ITRI integration debug - comprehensive outs_motion validation):
        print(f"DEBUG: Motion head completed - analyzing outs_motion structure")
        print(f"DEBUG: outs_motion type: {type(outs_motion)}")
        if outs_motion is not None and isinstance(outs_motion, dict):
            print(f"DEBUG: outs_motion keys: {list(outs_motion.keys())}")
            print(f"DEBUG: outs_motion shapes: {[(k, v.shape if torch.is_tensor(v) else type(v)) for k, v in outs_motion.items()]}")
            
            # Check for planning head requirements
            planning_required = ['sdc_traj_query', 'sdc_track_query', 'bev_pos']
            for key in planning_required:
                if key in outs_motion:
                    if torch.is_tensor(outs_motion[key]):
                        print(f"DEBUG: {key} shape: {outs_motion[key].shape}")
                    else:
                        print(f"DEBUG: {key} type: {type(outs_motion[key])}")
                else:
                    print(f"ERROR: Missing required key for planning head: {key}")
                    
        else:
            print(f"ERROR: outs_motion is None or not a dict: {outs_motion}")
        
        # Step 7: Run OccFormer if available
        outs_occ = {}
        if hasattr(uniad_model, 'occ_head') and uniad_model.occ_head is not None:
            print(f"DEBUG: Starting OccFormer with comprehensive input validation")
            
            # DEBUG LOGGING (ITRI integration debug - comprehensive OccFormer input analysis):
            print(f"DEBUG: OccFormer inputs analysis:")
            print(f"  - bev_embed type: {type(bev_embed)}, shape: {bev_embed.shape if torch.is_tensor(bev_embed) else 'Not tensor'}")
            print(f"  - outs_motion type: {type(outs_motion)}")
            
            if isinstance(outs_motion, dict):
                print(f"  - outs_motion keys: {list(outs_motion.keys())}")
                for key, value in outs_motion.items():
                    if value is None:
                        print(f"    ❌ {key}: None (POTENTIAL ISSUE!)")
                    elif torch.is_tensor(value):
                        print(f"    ✅ {key}: tensor {value.shape}")
                    else:
                        print(f"    ✅ {key}: {type(value)}")
            else:
                print(f"  ❌ outs_motion is not a dict: {outs_motion}")
            
            # Check for specific OccFormer requirements
            occ_required_keys = ['traj_query', 'track_query', 'track_query_pos']  # Common OccFormer inputs
            missing_keys = []
            none_keys = []
            
            for key in occ_required_keys:
                if key not in outs_motion:
                    missing_keys.append(key)
                elif outs_motion[key] is None:
                    none_keys.append(key)
            
            if missing_keys:
                print(f"  ❌ Missing OccFormer keys: {missing_keys}")
            if none_keys:
                print(f"  ❌ None values in OccFormer keys: {none_keys}")
            
            print(f"DEBUG: About to call OccFormer forward_test")
            
            try:
                # PATCHED CODE (ITRI integration fix - provide proper ground truth tensors for OccFormer):
                # OccFormer expects ground truth tensors, create appropriate dummy tensors for inference
                device = bev_embed.device
                batch_size = bev_embed.shape[0] if len(bev_embed.shape) > 2 else 1
                
                # Create dummy ground truth tensors matching expected OccFormer format
                # Based on get_occ_labels method - expects extra dimension for inference mode (lines 469-471)
                # The method does gt_x = gt_x[0] when not training, so we need [1, B, T, ...] format
                gt_segmentation = torch.zeros(1, batch_size, 6, 200, 200, device=device)  # [1, B, T, H, W] -> gets [0] -> [B, T, H, W] 
                gt_instance = torch.zeros(1, batch_size, 6, 200, 200, device=device)      # [1, B, T, H, W] -> gets [0] -> [B, T, H, W]
                gt_img_is_valid = torch.ones(1, batch_size, 7, device=device).bool()      # [1, B, T] -> gets [0] -> [B, T] (receptive_field + n_future)
                
                print(f"DEBUG: Created OccFormer ground truth tensors:")
                print(f"  - gt_segmentation: {gt_segmentation.shape}")
                print(f"  - gt_instance: {gt_instance.shape}")  
                print(f"  - gt_img_is_valid: {gt_img_is_valid.shape}")
                
                outs_occ = uniad_model.occ_head.forward_test(
                    bev_embed, outs_motion,
                    gt_segmentation=gt_segmentation,
                    gt_instance=gt_instance,
                    gt_img_is_valid=gt_img_is_valid
                )
                print(f"DEBUG: OccFormer completed successfully")
            except Exception as e:
                print(f"ERROR in OccFormer: {e}")
                print(f"ERROR: Exception type: {type(e)}")
                import traceback
                print(f"ERROR: Full traceback:")
                traceback.print_exc()
                raise
        
        # Step 8: Run Planner if available
        planning_results = {}
        if hasattr(uniad_model, 'planning_head') and uniad_model.planning_head is not None:
            print(f"DEBUG: Starting PlanningHead with validation")
            
            # Validate required keys for planning head
            planning_required = ['sdc_traj_query', 'sdc_track_query', 'bev_pos']
            for key in planning_required:
                if key not in outs_motion:
                    raise KeyError(f"Missing required key for planning head: {key}")
                if outs_motion[key] is None:
                    raise ValueError(f"None value for required planning key: {key}")
            
            try:
                planning_results = uniad_model.planning_head.forward_test(
                    bev_embed, outs_motion, outs_occ
                )
                print(f"DEBUG: PlanningHead completed successfully")
            except Exception as e:
                print(f"ERROR in PlanningHead: {e}")
                print(f"DEBUG: outs_motion structure:")
                for k, v in outs_motion.items():
                    print(f"  {k}: {v.shape if torch.is_tensor(v) else type(v)}")
                raise
        
        # Compile results
        results = {
            'frame_index': frame_index,
            'timestamp': itri_data['timestamp'],
            'ego_pose': itri_data['ego_pose'],
            'bev_embed': bev_embed,
            'motion_results': motion_results,
            'outs_motion': outs_motion,
            'outs_occ': outs_occ,
            'planning_results': planning_results,
            'input_data': {
                'image_paths': itri_data['image_paths'],
                'num_tracks': outs_track['track_query_embeddings'].shape[1] if 'track_query_embeddings' in outs_track else 0
            }
        }
        
        logger.info(f"Complete UniAD inference finished for frame {frame_index}")
        return results
    
    def run_batch_inference(
        self,
        uniad_model,
        frame_indices: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple frames.
        
        Args:
            uniad_model: Loaded UniAD model
            frame_indices: List of frame indices to process
            
        Returns:
            List of inference results
        """
        batch_results = []
        
        for frame_idx in frame_indices:
            try:
                result = self.run_complete_inference(uniad_model, frame_idx)
                batch_results.append(result)
                logger.info(f"Frame {frame_idx} processed successfully")
                
            except Exception as e:
                logger.error(f"Failed to process frame {frame_idx}: {e}")
                batch_results.append({
                    'frame_index': frame_idx,
                    'error': str(e),
                    'success': False
                })
        
        logger.info(f"Batch inference completed - {len(batch_results)} frames processed")
        return batch_results
    
    def validate_pipeline(self) -> bool:
        """
        Validate that the complete pipeline is working correctly.
        
        Returns:
            True if pipeline is valid
        """
        try:
            logger.info("Validating ITRI UniAD pipeline...")
            
            # Test data loading
            test_data = self.load_itri_data(frame_index=0)
            
            # Validate image paths
            missing_images = []
            for cam, path in test_data['image_paths'].items():
                if not os.path.exists(path):
                    missing_images.append(f"{cam}: {path}")
            
            if missing_images:
                logger.warning(f"Missing images: {missing_images}")
            
            # Test BEVFormer integration
            if not self.bev_processor.validate_integration():
                logger.error("BEVFormer integration validation failed")
                return False
            
            # Test semantic map conversion
            try:
                lane_query, lane_query_pos = self.semantic_converter.convert(
                    self.config['semantic_map_root'],
                    test_data['ego_pose']
                )
                logger.info(f"Semantic map validation passed - Lane queries: {lane_query.shape}")
                
            except Exception as e:
                logger.error(f"Semantic map validation failed: {e}")
                return False
            
            # Test track queries
            if test_data['track_queries']['track_query_embeddings'].numel() > 0:
                logger.info("Track queries validation passed")
            else:
                logger.warning("No track queries available - using empty queries")
            
            logger.info("✅ ITRI UniAD pipeline validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return False
    
    def clear_cache(self):
        """Clear data cache."""
        self._data_cache.clear()
        logger.info("Pipeline data cache cleared")
    
    def save_results(
        self,
        results: Dict[str, Any],
        output_dir: str
    ):
        """Save inference results to file."""
        os.makedirs(output_dir, exist_ok=True)
        
        frame_idx = results['frame_index']
        output_path = os.path.join(output_dir, f"uniad_results_frame_{frame_idx}.json")
        
        # Convert tensors to lists for JSON serialization
        serializable_results = self._make_serializable(results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert tensors and numpy arrays to serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def create_itri_pipeline(
    config: Dict = None,
    device: str = 'cuda'
) -> ITRIUniADPipeline:
    """
    Factory function to create ITRI UniAD pipeline.
    
    Args:
        config: Configuration dictionary
        device: Device for operations
        
    Returns:
        Configured ITRIUniADPipeline instance
    """
    return ITRIUniADPipeline(config, device)


def main():
    """Example usage of the ITRI UniAD pipeline."""
    print("ITRI UniAD Integration Pipeline")
    print("=" * 50)
    
    # Create pipeline
    pipeline = create_itri_pipeline()
    
    # Validate pipeline
    if pipeline.validate_pipeline():
        print("✅ Pipeline validation successful")
    else:
        print("❌ Pipeline validation failed")
        return
    
    print("Pipeline ready for inference!")
    
    # Example: Load and validate frame 0
    try:
        test_data = pipeline.load_itri_data(0)
        print(f"✅ Frame 0 data loaded successfully")
        print(f"   - Images: {len(test_data['image_paths'])} cameras")
        print(f"   - CAN bus: {test_data['canbus_data']['can_bus'].shape}")
        print(f"   - Ego pose: {test_data['ego_pose']}")
        
    except Exception as e:
        print(f"❌ Failed to load frame 0: {e}")


if __name__ == "__main__":
    main()