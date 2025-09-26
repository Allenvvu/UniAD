#!/usr/bin/env python3
"""
ITRI UniAD Integration Interface

Complete end-to-end pipeline that integrates all three ITRI modules:
1. Semantic Map → Lane Queries
2. TrackFormer → Track Queries
3. BEVFormer → BEV Features + Memory Bridge

This creates a unified interface for running UniAD inference with ITRI data.

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
from bevformer_integration import BEVFormerDataProcessor
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
            'track_queries_path': '/home/bryan/Desktop/Allen/UniAD/track/track_queries/complete_track_queries.pt',
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
        with torch.no_grad():
            motion_results, outs_motion = uniad_model.motion_head.forward_test(
                bev_embed=bev_embed,
                outs_track=outs_track,
                outs_seg=outs_seg
            )
        
        # Step 7: Run OccFormer if available
        outs_occ = {}
        if hasattr(uniad_model, 'occ_head') and uniad_model.occ_head is not None:
            outs_occ = uniad_model.occ_head.forward_test(
                bev_embed, outs_motion
            )
        
        # Step 8: Run Planner if available
        planning_results = {}
        if hasattr(uniad_model, 'planning_head') and uniad_model.planning_head is not None:
            planning_results = uniad_model.planning_head.forward_test(
                bev_embed, outs_motion, outs_occ
            )
        
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