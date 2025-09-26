#!/usr/bin/env python3
"""
Multi-bag inference script for processing multiple HCT bag files
"""

import sys
import os
import argparse
from pathlib import Path
sys.path.append('/home/bryan/Desktop/Allen/UniAD')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/integration')

from phase5_uniad_inference import Phase5Inference, DatasetDiscovery, FrameInfo
from config.device_config import DeviceConfig, get_device_config, configure_from_args
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiBagDatasetDiscovery(DatasetDiscovery):
    """Extended dataset discovery for multiple bag files"""
    
    def __init__(self, bag_name: str, base_path: str = '/home/bryan/Desktop/Allen/UniAD'):
        super().__init__(base_path)
        self.bag_name = bag_name
        
        # Set bag-specific paths and time windows
        self.setup_bag_config(bag_name)
        
    def setup_bag_config(self, bag_name: str):
        """Configure paths and time windows for specific bag files"""
        self.can_bus_file = self.base_path / f'data/itri/2025-08-06-hct_logistic/canbus/{bag_name}_can_bus.pkl'
        
        # Define temporal windows for each bag (these are estimates - need to be refined)
        bag_configs = {
            '2025-08-06-14-23-05_0': {
                'start': 1754461385.584,  # 14:23:05.584
                'end': 1754461441.087     # 14:24:01.087
            },
            '2025-08-06-14-24-01_1': {
                'start': 1754461441.0,    # 14:24:01.0
                'end': 1754461495.0       # 14:24:55.0 (estimate)
            },
            '2025-08-06-14-24-55_2': {
                'start': 1754461495.0,    # 14:24:55.0  
                'end': 1754461549.0       # 14:25:49.0 (estimate)
            }
        }
        
        if bag_name in bag_configs:
            config = bag_configs[bag_name]
            self.start_timestamp = config['start']
            self.end_timestamp = config['end']
            logger.info(f"Configured {bag_name}: {self.start_timestamp} → {self.end_timestamp}")
        else:
            raise ValueError(f"Unknown bag name: {bag_name}. Available: {list(bag_configs.keys())}")

def run_bag_inference(bag_name: str, device_config: DeviceConfig):
    """Run inference on a specific bag file"""
    logger.info(f"🚀 Starting inference for bag: {bag_name}")
    logger.info("=" * 60)
    
    try:
        # Create bag-specific dataset discovery
        discovery = MultiBagDatasetDiscovery(bag_name)
        
        # Initialize Phase5 inference with bag-specific discovery
        phase5 = Phase5Inference(device_config=device_config, enable_metrics=True)
        phase5.dataset_discovery = discovery
        
        # Setup the inference pipeline
        phase5.setup()
        
        logger.info(f"📊 Dataset Discovery: {len(phase5.aligned_frames)} synchronized frames found for {bag_name}")
        
        if len(phase5.aligned_frames) == 0:
            logger.warning(f"❌ No synchronized frames found for {bag_name}")
            return None
            
        # Run full dataset inference
        output_dir = f"output/hct_{bag_name.replace('-', '_').replace(':', '_')}"
        logger.info(f"🗃️ Running inference on {len(phase5.aligned_frames)} frames...")
        logger.info(f"📁 Output directory: {output_dir}")
        
        results = phase5.run_full_dataset_inference(
            batch_size=8,
            output_dir=output_dir
        )
        
        logger.info(f"✅ {bag_name} inference completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed to process {bag_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Multi-bag UniAD Inference')
    parser.add_argument('--bags', nargs='+', 
                       default=['2025-08-06-14-24-01_1', '2025-08-06-14-24-55_2'],
                       help='Bag names to process')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto',
                       help='Device configuration')
    
    args = parser.parse_args()
    
    # Configure device
    device_config = DeviceConfig(args.device)
    
    print("🎯 Multi-Bag UniAD Inference")
    print(f"📱 Device: {device_config.model_device}")
    print(f"📦 Bags to process: {args.bags}")
    print("=" * 60)
    
    results = {}
    
    for bag_name in args.bags:
        print(f"\n🔄 Processing {bag_name}...")
        
        result = run_bag_inference(bag_name, device_config)
        results[bag_name] = result
        
        if result:
            print(f"✅ {bag_name}: {result['dataset_info']['successful_frames']} frames processed")
        else:
            print(f"❌ {bag_name}: Processing failed")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Multi-bag inference complete!")
    print("📊 Results Summary:")
    
    total_frames = 0
    successful_bags = 0
    
    for bag_name, result in results.items():
        if result and result['dataset_info']['successful_frames'] > 0:
            frames = result['dataset_info']['successful_frames']
            total_frames += frames
            successful_bags += 1
            success_rate = result['dataset_info']['success_rate'] * 100
            print(f"   {bag_name}: {frames} frames ({success_rate:.1f}% success)")
        else:
            print(f"   {bag_name}: Failed")
    
    print(f"\n📈 Total: {total_frames} frames across {successful_bags}/{len(args.bags)} bags")

if __name__ == '__main__':
    main()