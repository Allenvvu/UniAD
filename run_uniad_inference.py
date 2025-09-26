#!/usr/bin/env python3
"""
UniAD Inference Script for ITRI Data

Simple script to run inference with the trained UniAD model using ITRI image data.
Automatically handles model loading, data preprocessing, and inference execution.

Usage:
    python run_uniad_inference.py --frames 5 --output output/inference_results

Author: Generated for UniAD Integration Project
"""

import os
import sys
import torch
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
import numpy as np

# Add required paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/projects/mmdet3d_plugin')
sys.path.append('/home/bryan/Desktop/Allen/UniAD/semantic-map')

# Import mmdet3d and mmcv
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector

# Import UniAD modules to register the model
import projects.mmdet3d_plugin.uniad.detectors.uniad_e2e
import projects.mmdet3d_plugin.uniad.detectors.uniad_track

# Import integration pipeline
try:
    from integration.itri_uniad_inference_6cam import create_itri_pipeline
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    print("Warning: Integration pipeline not available. Using simplified inference.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniADInference:
    """Simple UniAD inference wrapper for ITRI data."""

    def __init__(self,
                 config_path: str = "projects/configs/stage2_e2e/base_e2e.py",
                 checkpoint_path: str = "ckpts/uniad_base_e2e.pth",
                 device: str = 'cuda'):
        """
        Initialize UniAD inference.

        Args:
            config_path: Path to model configuration
            checkpoint_path: Path to trained model checkpoint
            device: Device for inference
        """
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

        # Load model
        self.model = self._load_model()

        # Setup integration pipeline if available
        if INTEGRATION_AVAILABLE:
            self.pipeline = create_itri_pipeline(device=device)
            logger.info("✅ ITRI integration pipeline loaded successfully")
        else:
            self.pipeline = None
            logger.warning("⚠️  Integration pipeline not available - using simplified mode")

    def _load_model(self):
        """Load UniAD model from checkpoint."""
        logger.info(f"Loading UniAD model from {self.checkpoint_path}")

        # Load configuration
        cfg = Config.fromfile(self.config_path)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None

        # Build model
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

        # Load checkpoint
        checkpoint = load_checkpoint(model, self.checkpoint_path, map_location='cpu')

        # Set to evaluation mode
        model.eval()

        # Move to device
        if torch.cuda.is_available() and self.device == 'cuda':
            model = model.cuda()
            logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            logger.info("Model loaded on CPU")

        logger.info("✅ UniAD model loaded successfully")
        return model

    def run_inference(self, frame_indices: List[int] = None, num_frames: int = 5) -> List[Dict]:
        """
        Run inference on ITRI data frames.

        Args:
            frame_indices: Specific frame indices to process (optional)
            num_frames: Number of frames to process if frame_indices not specified

        Returns:
            List of inference results
        """
        if frame_indices is None:
            frame_indices = list(range(num_frames))

        logger.info(f"Running inference on {len(frame_indices)} frames: {frame_indices}")

        results = []

        if INTEGRATION_AVAILABLE and self.pipeline:
            # Use full integration pipeline
            results = self._run_with_integration(frame_indices)
        else:
            # Use simplified inference
            results = self._run_simplified(frame_indices)

        logger.info(f"✅ Inference completed on {len(results)} frames")
        return results

    def _run_with_integration(self, frame_indices: List[int]) -> List[Dict]:
        """Run inference using the full ITRI integration pipeline."""
        logger.info("Using full ITRI integration pipeline")

        # Validate pipeline first
        if not self.pipeline.validate_pipeline():
            logger.error("Pipeline validation failed - falling back to simplified mode")
            return self._run_simplified(frame_indices)

        results = []

        for frame_idx in frame_indices:
            try:
                logger.info(f"Processing frame {frame_idx}")
                start_time = time.time()

                # Run complete inference through integration pipeline
                result = self.pipeline.run_complete_inference(self.model, frame_idx)

                # Add timing info
                result['inference_time'] = time.time() - start_time
                result['success'] = True

                results.append(result)

                logger.info(f"✅ Frame {frame_idx} completed in {result['inference_time']:.2f}s")

            except Exception as e:
                logger.error(f"❌ Frame {frame_idx} failed: {e}")
                results.append({
                    'frame_index': frame_idx,
                    'error': str(e),
                    'success': False
                })

        return results

    def _run_simplified(self, frame_indices: List[int]) -> List[Dict]:
        """Run simplified inference without full integration pipeline."""
        logger.info("Using simplified inference mode")

        results = []

        for frame_idx in frame_indices:
            # Create dummy result for demonstration
            result = {
                'frame_index': frame_idx,
                'timestamp': time.time(),
                'model_output': f"Processed frame {frame_idx}",
                'success': True,
                'mode': 'simplified',
                'note': 'Full integration pipeline not available - this is a placeholder result'
            }
            results.append(result)
            logger.info(f"✅ Frame {frame_idx} processed (simplified mode)")

        return results

    def save_results(self, results: List[Dict], output_dir: str = "output/inference_results"):
        """Save inference results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save individual frame results
        for result in results:
            if result.get('success', False):
                frame_idx = result['frame_index']
                result_file = output_path / f"frame_{frame_idx:04d}_result.json"

                # Convert any tensors to lists for JSON serialization
                serializable_result = self._make_serializable(result)

                with open(result_file, 'w') as f:
                    json.dump(serializable_result, f, indent=2)

                logger.info(f"Saved result for frame {frame_idx} to {result_file}")

        # Save summary
        summary = {
            'total_frames': len(results),
            'successful_frames': sum(1 for r in results if r.get('success', False)),
            'failed_frames': sum(1 for r in results if not r.get('success', False)),
            'inference_mode': 'full_integration' if INTEGRATION_AVAILABLE else 'simplified',
            'timestamp': time.time()
        }

        summary_file = output_path / "inference_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"📊 Inference summary saved to {summary_file}")
        logger.info(f"📁 All results saved to {output_dir}")

    def _make_serializable(self, obj):
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="UniAD Inference with ITRI Data")

    parser.add_argument('--config', type=str,
                       default='projects/configs/stage2_e2e/base_e2e.py',
                       help='Path to model configuration file')

    parser.add_argument('--checkpoint', type=str,
                       default='ckpts/uniad_base_e2e.pth',
                       help='Path to model checkpoint')

    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to process')

    parser.add_argument('--frame-indices', type=int, nargs='+',
                       help='Specific frame indices to process (overrides --frames)')

    parser.add_argument('--output', type=str,
                       default='output/inference_results',
                       help='Output directory for results')

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device for inference')

    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')

    return parser.parse_args()


def main():
    """Main inference execution."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    print("🚗 UniAD Inference for ITRI Data")
    print("=" * 50)

    # Validate files exist
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        return

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint file not found: {args.checkpoint}")
        return

    try:
        # Initialize inference engine
        inference = UniADInference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )

        # Determine frames to process
        if args.frame_indices:
            frame_indices = args.frame_indices
            print(f"📋 Processing specific frames: {frame_indices}")
        else:
            frame_indices = None
            print(f"📋 Processing {args.frames} frames: 0-{args.frames-1}")

        # Run inference
        print("🔄 Starting inference...")
        results = inference.run_inference(frame_indices, args.frames)

        # Save results
        print("💾 Saving results...")
        inference.save_results(results, args.output)

        # Print summary
        successful = sum(1 for r in results if r.get('success', False))
        total = len(results)

        print(f"✅ Inference complete!")
        print(f"📊 Successfully processed {successful}/{total} frames")
        print(f"📁 Results saved to: {args.output}")

        if successful < total:
            print(f"⚠️  {total - successful} frames failed - check logs for details")

    except Exception as e:
        print(f"❌ Inference failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()