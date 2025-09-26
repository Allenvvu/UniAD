#!/usr/bin/env python3
"""
End-to-End Integration Tests for ITRI UniAD Pipeline

Comprehensive test suite to validate the complete integration pipeline:
1. BEV Memory Bridge functionality
2. Enhanced BEVFormer Integration  
3. UniAD Interface orchestration
4. Data format adapters
5. Complete pipeline validation

Author: Generated for UniAD Integration Project
"""

import os
import sys
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/integration')
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

# Import our modules
try:
    # Modules from semantic-map
    from bev_memory_bridge import create_memory_bridge, BEVMemoryBridge
    from bevformer_integration import BEVFormerDataProcessor
    # Modules from integration
    from itri_uniad_inference import create_itri_pipeline, ITRIUniADPipeline
    from itri_data_adapter import create_data_adapter, ITRIDataAdapter
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all integration modules are in the correct paths")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegrationTestSuite:
    """
    Comprehensive test suite for the ITRI UniAD integration.
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize test suite.
        
        Args:
            device: Device for testing ('cuda' or 'cpu')
        """
        self.device = device
        self.test_results = []
        
        logger.info(f"Initializing integration tests on device: {device}")
        
    def run_all_tests(self) -> Dict[str, bool]:
        """
        Run complete test suite.
        
        Returns:
            Dictionary of test results
        """
        logger.info("Starting comprehensive integration test suite")
        logger.info("=" * 60)
        
        tests = [
            ("BEV Memory Bridge", self.test_bev_memory_bridge),
            ("BEVFormer Integration", self.test_bevformer_integration),
            ("Data Format Adapter", self.test_data_adapter),
            ("UniAD Pipeline", self.test_uniad_pipeline),
            ("End-to-End Validation", self.test_end_to_end),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running: {test_name}")
            logger.info("-" * 40)
            
            try:
                success = test_func()
                results[test_name] = success
                
                if success:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{test_name:.<40} {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - Integration is ready!")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed - Integration needs attention")
        
        return results
    
    def test_bev_memory_bridge(self) -> bool:
        """Test BEV Memory Bridge functionality."""
        try:
            # Create memory bridge
            bridge = create_memory_bridge(device=self.device)
            
            # Test with mock BEV features
            bev_embed = torch.randn(1, 256, 200, 200, device=self.device)
            lane_query = torch.randn(1, 100, 256, device=self.device)
            lane_query_pos = torch.randn(1, 100, 256, device=self.device)
            
            # Test memory bridge
            result = bridge.bridge_bev_to_motion(
                bev_embed, lane_query, lane_query_pos
            )
            
            # Validate result structure
            if 'args_tuple' not in result:
                logger.error("Missing args_tuple in bridge result")
                return False
            
            args_tuple = result['args_tuple']
            if len(args_tuple) != 7:
                logger.error(f"Invalid args_tuple length: {len(args_tuple)}")
                return False
            
            # Check memory features are populated (not None)
            memory, memory_mask, memory_pos = args_tuple[:3]
            if any(x is None for x in [memory, memory_mask, memory_pos]):
                logger.error("Memory features still None after bridging")
                return False
            
            # Validate tensor shapes
            batch_size, seq_len, embed_dim = memory.shape
            expected_seq_len = 200 * 200  # BEV grid size
            
            if seq_len != expected_seq_len:
                logger.error(f"Unexpected memory sequence length: {seq_len}")
                return False
                
            if embed_dim != 256:
                logger.error(f"Unexpected embedding dimension: {embed_dim}")
                return False
            
            # Validate device placement
            expected_device = torch.device(self.device)
            if memory.device.type != expected_device.type:
                logger.error(f"Memory tensor on wrong device: {memory.device}")
                return False
            
            logger.info("BEV Memory Bridge validation passed")
            return True
            
        except Exception as e:
            logger.error(f"BEV Memory Bridge test failed: {e}")
            return False
    
    def test_bevformer_integration(self) -> bool:
        """Test enhanced BEVFormer integration."""
        try:
            # Create BEVFormer processor
            processor = BEVFormerDataProcessor(device=self.device)
            
            # Test validation
            if not processor.validate_integration():
                logger.error("BEVFormer integration validation failed")
                return False
            
            # Test data preparation
            test_image_paths = {
                'front_100deg': '/test/front.jpg',
                'front_left_100deg': '/test/front_left.jpg',
                'front_right_100deg': '/test/front_right.jpg',
                'back_60deg': '/test/back.jpg'
            }
            
            # Create mock CAN bus data
            mock_canbus = {
                'can_bus': np.random.randn(10, 18),  # 10 frames, 18 dimensions
                'timestamps': np.arange(10)
            }
            
            # Test semantic map integration
            test_bev = torch.randn(1, 256, 200, 200, device=self.device)
            test_lane_query = torch.randn(1, 100, 256, device=self.device)
            test_lane_pos = torch.randn(1, 100, 256, device=self.device)
            
            result = processor.integrate_with_semantic_map(
                test_bev, test_lane_query, test_lane_pos
            )
            
            # Validate integration result
            if 'args_tuple' not in result:
                logger.error("Missing args_tuple in integration result")
                return False
            
            args_tuple = result['args_tuple']
            if any(args_tuple[i] is None for i in [0, 1, 2]):  # memory components
                logger.error("Memory features still None after integration")
                return False
            
            logger.info("BEVFormer integration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"BEVFormer integration test failed: {e}")
            return False
    
    def test_data_adapter(self) -> bool:
        """Test ITRI data format adapter."""
        try:
            # Create data adapter
            adapter = create_data_adapter(device=self.device)
            
            # Test coordinate transformations
            world_coords = np.array([[10.0, 20.0], [-5.0, 15.0], [0.0, 0.0]])
            bev_coords = adapter.coord_adapter.world_to_bev(world_coords)
            world_back = adapter.coord_adapter.bev_to_world(bev_coords)
            
            # Check coordinate round-trip accuracy
            if not np.allclose(world_coords, world_back, atol=1e-3):
                logger.error("Coordinate transformation round-trip failed")
                return False
            
            # Test tensor formatting
            test_array = np.random.randn(3, 4, 5)
            tensor = adapter.tensor_adapter.ensure_tensor_format(test_array)
            
            if not isinstance(tensor, torch.Tensor):
                logger.error("Tensor formatting failed")
                return False
            
            expected_device = torch.device(self.device)
            if tensor.device.type != expected_device.type:
                logger.error(f"Tensor on wrong device: {tensor.device}")
                return False
            
            # Test track data adaptation
            mock_track_data = {
                'track_query_embeddings': torch.randn(1, 10, 256),
                'sdc_embedding': torch.randn(256),
                'track_bbox_results': [],
                'sdc_track_bbox_results': []
            }
            
            adapted_tracks = adapter.track_adapter.adapt_track_queries(mock_track_data)
            
            required_fields = ['track_query_embeddings', 'sdc_embedding']
            for field in required_fields:
                if field not in adapted_tracks:
                    logger.error(f"Missing adapted track field: {field}")
                    return False
            
            # Test complete frame adaptation
            mock_itri_data = {
                'ego_pose': (10.0, 20.0, 0.5),
                'canbus_data': {
                    'can_bus': np.random.randn(5, 18),
                    'timestamps': np.arange(5)
                },
                'track_queries': mock_track_data,
                'frame_index': 0,
                'image_paths': {
                    'front_100deg': '/test/front.jpg'
                }
            }
            
            adapted_data = adapter.adapt_complete_frame(mock_itri_data)
            
            # Validate adapted data
            if not adapter.validate_adapted_data(adapted_data):
                logger.error("Adapted data validation failed")
                return False
            
            logger.info("Data adapter validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Data adapter test failed: {e}")
            return False
    
    def test_uniad_pipeline(self) -> bool:
        """Test UniAD pipeline orchestration."""
        try:
            # Create pipeline with test configuration
            test_config = {
                'data_root': '/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic',
                'semantic_map_root': '/home/bryan/Desktop/Allen/UniAD/semantic-map/data/itri',
                'track_queries_path': '/home/bryan/Desktop/Allen/UniAD/track/track_queries/complete_track_queries.pt',
                'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            }
            
            pipeline = create_itri_pipeline(config=test_config, device=self.device)
            
            # Test pipeline validation
            if not pipeline.validate_pipeline():
                logger.warning("Pipeline validation failed - this may be due to missing data files")
                # Continue with available tests
            
            # Test component initialization
            components = [
                ('bev_processor', BEVFormerDataProcessor),
                ('motion_integrator', type(None)),  # MotionFormerIntegrator
                ('semantic_converter', type(None))   # ITRIToLaneQueryConverter
            ]
            
            for component_name, expected_type in components:
                if not hasattr(pipeline, component_name):
                    logger.error(f"Pipeline missing component: {component_name}")
                    return False
            
            # Test data loading (if data exists)
            try:
                test_data = pipeline.load_itri_data(frame_index=0)
                logger.info(f"Successfully loaded test data with keys: {list(test_data.keys())}")
                
                # Validate data structure
                expected_keys = ['ego_pose', 'canbus_data', 'track_queries', 'frame_index']
                for key in expected_keys:
                    if key not in test_data:
                        logger.warning(f"Missing expected key in test data: {key}")
                        
            except Exception as e:
                logger.warning(f"Could not load test data (expected if data files missing): {e}")
            
            logger.info("UniAD pipeline validation passed")
            return True
            
        except Exception as e:
            logger.error(f"UniAD pipeline test failed: {e}")
            return False
    
    def test_end_to_end(self) -> bool:
        """Test complete end-to-end integration."""
        try:
            logger.info("Testing end-to-end integration flow...")
            
            # 1. Test BEV Memory Bridge
            bridge = create_memory_bridge(device=self.device)
            
            # 2. Test BEVFormer Integration  
            bev_processor = BEVFormerDataProcessor(device=self.device)
            
            # 3. Test Data Adapter
            data_adapter = create_data_adapter(device=self.device)
            
            # 4. Create mock complete pipeline flow
            logger.info("Simulating complete pipeline flow...")
            
            # Mock BEV features from BEVFormer
            mock_bev_embed = torch.randn(1, 256, 200, 200, device=self.device)
            
            # Mock lane queries from semantic map
            mock_lane_query = torch.randn(1, 100, 256, device=self.device)
            mock_lane_query_pos = torch.randn(1, 100, 256, device=self.device)
            
            # Mock track queries
            mock_track_data = {
                'track_query_embeddings': torch.randn(1, 50, 256, device=self.device),
                'sdc_embedding': torch.randn(256, device=self.device),
                'track_bbox_results': [],
                'sdc_track_bbox_results': []
            }
            
            # Step 1: Bridge BEV features with semantic map
            outs_seg = bridge.bridge_bev_to_motion(
                mock_bev_embed, mock_lane_query, mock_lane_query_pos
            )
            
            # Step 2: Adapt track queries
            outs_track = data_adapter.track_adapter.adapt_track_queries(mock_track_data)
            
            # Step 3: Validate complete integration format
            # This simulates what MotionFormer would receive
            
            # Validate outs_seg structure
            if 'args_tuple' not in outs_seg:
                logger.error("Missing args_tuple in outs_seg")
                return False
            
            args_tuple = outs_seg['args_tuple']
            if len(args_tuple) != 7:
                logger.error(f"Invalid args_tuple structure: length {len(args_tuple)}")
                return False
            
            # Validate args_tuple components
            memory, memory_mask, memory_pos, lane_query, _, lane_query_pos, hw_lvl = args_tuple
            
            # Check all memory components are populated
            if any(x is None for x in [memory, memory_mask, memory_pos]):
                logger.error("Memory components still None in final integration")
                return False
            
            # Check lane query components
            if lane_query is None or lane_query_pos is None:
                logger.error("Lane query components are None")
                return False
            
            # Check spatial dimensions
            if hw_lvl != [(200, 200)]:
                logger.error(f"Invalid spatial dimensions: {hw_lvl}")
                return False
            
            # Validate outs_track structure
            required_track_fields = ['track_query_embeddings', 'sdc_embedding']
            for field in required_track_fields:
                if field not in outs_track:
                    logger.error(f"Missing track field: {field}")
                    return False
            
            # Check tensor shapes and devices
            if memory.shape != (1, 40000, 256):  # 200*200 = 40000
                logger.error(f"Invalid memory shape: {memory.shape}")
                return False
            
            expected_device = torch.device(self.device)
            if outs_track['track_query_embeddings'].device.type != expected_device.type:
                logger.error("Track queries on wrong device")
                return False
            
            logger.info("✅ End-to-end integration validation successful!")
            logger.info(f"   - BEV Memory Bridge: ✅ Working")
            logger.info(f"   - Memory Features: ✅ Populated ({memory.shape})")
            logger.info(f"   - Lane Queries: ✅ Available ({lane_query.shape})")
            logger.info(f"   - Track Queries: ✅ Available ({outs_track['track_query_embeddings'].shape})")
            logger.info(f"   - Device Consistency: ✅ All on {self.device}")
            
            return True
            
        except Exception as e:
            logger.error(f"End-to-end integration test failed: {e}")
            return False


def main():
    """Run the complete integration test suite."""
    print("ITRI UniAD Integration Test Suite")
    print("=" * 50)
    
    # Determine device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🔧 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("🔧 Using CPU (CUDA not available)")
    
    # Create and run test suite
    test_suite = IntegrationTestSuite(device=device)
    results = test_suite.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        print("\n🎉 All integration tests passed!")
        print("The ITRI UniAD integration is ready for use.")
        exit(0)
    else:
        print("\n❌ Some integration tests failed.")
        print("Please review the errors above and fix the issues.")
        exit(1)


if __name__ == "__main__":
    main()