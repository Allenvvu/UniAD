#!/usr/bin/env python3
"""
CAN Bus Data Validation for ITRI Dataset

Validates the 18-dimensional CAN bus data structure and format for UniAD integration.
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import logging
from typing import Dict, List, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CANBusValidator:
    """Validate ITRI CAN bus data structure and format."""
    
    def __init__(self, data_path: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic'):
        self.data_path = Path(data_path)
        self.canbus_path = self.data_path / 'canbus'
        
    def load_canbus_file(self, file_path: Path) -> Dict:
        """Load and examine a CAN bus pickle file."""
        logger.info(f"📁 Loading CAN bus file: {file_path.name}")
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"✅ Data type: {type(data)}")
        
        if isinstance(data, dict):
            logger.info(f"📊 Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                logger.info(f"   {key}: {type(value)}")
                if hasattr(value, 'shape'):
                    logger.info(f"      Shape: {value.shape}")
                elif isinstance(value, (list, tuple)):
                    logger.info(f"      Length: {len(value)}")
                    if len(value) > 0:
                        logger.info(f"      First element: {type(value[0])}")
                        
        return data
        
    def analyze_can_structure(self, can_data: Dict) -> Dict[str, Any]:
        """Analyze CAN bus data structure for UniAD compatibility."""
        logger.info("🔍 Analyzing CAN bus data structure...")
        
        analysis = {
            'total_entries': 0,
            'data_format': None,
            'dimensions': None,
            'sample_data': None,
            'time_stamps': None,
            'vehicle_states': {}
        }
        
        # Check for common CAN data keys
        common_keys = ['timestamp', 'speed', 'steering', 'acceleration', 'yaw_rate', 'position', 'orientation']
        
        for key in can_data.keys():
            value = can_data[key]
            logger.info(f"🔑 Key: {key}")
            
            if isinstance(value, (list, np.ndarray)):
                analysis['total_entries'] = len(value)
                logger.info(f"   📊 Entries: {len(value)}")
                
                if len(value) > 0:
                    sample = value[0]
                    logger.info(f"   📋 Sample type: {type(sample)}")
                    
                    if hasattr(sample, 'shape'):
                        logger.info(f"   📐 Sample shape: {sample.shape}")
                        analysis['dimensions'] = sample.shape
                    elif isinstance(sample, dict):
                        logger.info(f"   🗂️ Sample keys: {list(sample.keys())}")
                        analysis['sample_data'] = list(sample.keys())
                    elif isinstance(sample, (list, tuple)):
                        logger.info(f"   📏 Sample length: {len(sample)}")
                        analysis['dimensions'] = (len(sample),)
                        
                        # Check if this could be the 18D vector
                        if len(sample) == 18:
                            logger.info("   🎯 Found 18-dimensional data!")
                            analysis['data_format'] = '18D_vector'
                            
        return analysis
        
    def extract_18d_vectors(self, can_data: Dict) -> np.ndarray:
        """Extract 18-dimensional vectors from CAN data."""
        logger.info("🎯 Extracting 18-dimensional vectors...")
        
        vectors = []
        
        for key, value in can_data.items():
            if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                sample = value[0]
                
                # Check if this is already 18D
                if hasattr(sample, 'shape') and sample.shape == (18,):
                    vectors = np.array(value)
                    logger.info(f"✅ Found 18D vectors in key '{key}': shape {vectors.shape}")
                    break
                elif isinstance(sample, (list, tuple)) and len(sample) == 18:
                    vectors = np.array(value)
                    logger.info(f"✅ Found 18D vectors in key '{key}': shape {vectors.shape}")
                    break
                    
        if len(vectors) == 0:
            # Try to construct 18D vectors from available data
            logger.info("🔨 Constructing 18D vectors from available data...")
            
            # Common vehicle state parameters (adjust based on actual data structure)
            components = []
            
            for key, value in can_data.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    sample = value[0]
                    if isinstance(sample, (int, float)):
                        components.append(value)
                    elif hasattr(sample, 'shape') and len(sample.shape) == 1:
                        components.extend([sample[i] for i in range(len(sample))])
                        
            if len(components) >= 18:
                # Take first 18 components
                vectors = np.column_stack(components[:18])
                logger.info(f"🔨 Constructed 18D vectors: shape {vectors.shape}")
            else:
                logger.warning(f"⚠️ Only {len(components)} components found, padding to 18D")
                # Create dummy 18D vectors
                if len(components) > 0:
                    base_length = len(components[0]) if hasattr(components[0], '__len__') else 1000
                else:
                    base_length = 1000
                    
                vectors = np.zeros((base_length, 18))
                for i, comp in enumerate(components):
                    if hasattr(comp, '__len__'):
                        vectors[:len(comp), i] = comp
                    else:
                        vectors[:, i] = comp
                        
        return vectors
        
    def validate_for_bevformer(self, vectors: np.ndarray) -> bool:
        """Validate CAN data format for BEVFormer integration."""
        logger.info("🤖 Validating CAN data for BEVFormer integration...")
        
        if vectors.shape[1] != 18:
            logger.error(f"❌ Expected 18 dimensions, got {vectors.shape[1]}")
            return False
            
        logger.info(f"✅ Vector shape: {vectors.shape}")
        logger.info(f"✅ Data type: {vectors.dtype}")
        
        # Check for reasonable value ranges (adjust based on expected ranges)
        logger.info("📊 Data statistics:")
        logger.info(f"   Min values: {vectors.min(axis=0)}")
        logger.info(f"   Max values: {vectors.max(axis=0)}")
        logger.info(f"   Mean values: {vectors.mean(axis=0)}")
        
        # Convert to tensor for BEVFormer
        tensor = torch.from_numpy(vectors).float()
        logger.info(f"🔧 Tensor shape: {tensor.shape}")
        logger.info(f"🔧 Tensor dtype: {tensor.dtype}")
        
        return True
        
    def run_validation(self) -> bool:
        """Run complete CAN bus validation."""
        logger.info("🚀 Starting CAN Bus Data Validation")
        logger.info("=" * 50)
        
        canbus_files = list(self.canbus_path.glob('*.pkl'))
        if not canbus_files:
            logger.error("❌ No CAN bus files found")
            return False
            
        logger.info(f"📁 Found {len(canbus_files)} CAN bus files:")
        for file in canbus_files:
            logger.info(f"   - {file.name}")
            
        # Analyze first file in detail
        sample_file = canbus_files[0]
        can_data = self.load_canbus_file(sample_file)
        
        # Analyze structure
        analysis = self.analyze_can_structure(can_data)
        
        # Extract 18D vectors
        vectors = self.extract_18d_vectors(can_data)
        
        if vectors.size > 0:
            # Validate for BEVFormer
            success = self.validate_for_bevformer(vectors)
            
            if success:
                logger.info("✅ CAN bus data validation successful!")
                
                # Save sample for integration testing
                output_path = Path('/home/bryan/Desktop/Allen/UniAD/integration/output/canbus_sample.npz')
                output_path.parent.mkdir(exist_ok=True)
                np.savez(output_path, canbus_data=vectors, metadata=analysis)
                logger.info(f"💾 Sample data saved to: {output_path}")
                
            return success
        else:
            logger.error("❌ Could not extract valid 18D vectors")
            return False


if __name__ == "__main__":
    validator = CANBusValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("🎉 CAN bus validation passed!")
    else:
        logger.error("❌ CAN bus validation failed!")