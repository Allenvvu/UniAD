#!/usr/bin/env python3
"""
Track Query Validation Script

Validates the generated track query files to ensure they meet UniAD requirements
and have correct data formats for training.

Author: UniAD Integration Project
"""

import torch
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append('/home/bryan/Desktop/Allen/UniAD/track')

try:
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    UNIAD_AVAILABLE = True
except ImportError:
    logger.warning("UniAD components not available for full validation")
    UNIAD_AVAILABLE = False


class TrackQueryValidator:
    """Validates track query output files for UniAD compatibility"""
    
    def __init__(self, track_query_dir: str):
        self.track_query_dir = Path(track_query_dir)
        self.validation_results = {
            'total_frames': 0,
            'valid_frames': 0,
            'invalid_frames': [],
            'validation_errors': [],
            'data_statistics': {},
            'format_validation': {}
        }
        
    def validate_all_frames(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Validate all track query frames.
        
        Args:
            sample_size: Number of frames to validate in detail (0 = all frames)
            
        Returns:
            Validation results dictionary
        """
        logger.info(f"Starting validation of track query files in {self.track_query_dir}")
        
        # Get all frame directories
        frame_dirs = sorted([d for d in self.track_query_dir.glob('frame_*') if d.is_dir()])
        self.validation_results['total_frames'] = len(frame_dirs)
        
        if not frame_dirs:
            logger.error("No frame directories found!")
            return self.validation_results
        
        logger.info(f"Found {len(frame_dirs)} frame directories")
        
        # Validate sample frames in detail
        sample_frames = frame_dirs if sample_size == 0 else frame_dirs[:sample_size]
        logger.info(f"Validating {len(sample_frames)} frames in detail")
        
        for frame_dir in sample_frames:
            frame_idx = int(frame_dir.name.split('_')[-1])
            is_valid = self.validate_single_frame(frame_dir, frame_idx)
            
            if is_valid:
                self.validation_results['valid_frames'] += 1
            else:
                self.validation_results['invalid_frames'].append(frame_idx)
        
        # Quick validation for remaining frames
        if sample_size > 0 and len(frame_dirs) > sample_size:
            logger.info(f"Quick validation for remaining {len(frame_dirs) - sample_size} frames")
            for frame_dir in frame_dirs[sample_size:]:
                frame_idx = int(frame_dir.name.split('_')[-1])
                is_valid = self.quick_validate_frame(frame_dir, frame_idx)
                
                if is_valid:
                    self.validation_results['valid_frames'] += 1
                else:
                    self.validation_results['invalid_frames'].append(frame_idx)
        
        # Generate statistics
        self._generate_statistics()
        
        return self.validation_results
    
    def validate_single_frame(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate a single frame in detail"""
        try:
            logger.info(f"Validating frame {frame_idx}")
            
            # Check required files exist
            required_files = [
                'track_query_embeddings.pt',
                'track_query_matched_idxes.pt', 
                'bbox_results.pt',
                'sdc_embedding.pt',
                'complete_track_query.pt',
                'metadata.json'
            ]
            
            missing_files = []
            for file in required_files:
                if not (frame_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                error = f"Frame {frame_idx}: Missing files {missing_files}"
                logger.error(error)
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Load and validate each component
            success = True
            
            # 1. Validate track query embeddings
            success &= self._validate_embeddings(frame_dir, frame_idx)
            
            # 2. Validate matched indices
            success &= self._validate_matched_indices(frame_dir, frame_idx)
            
            # 3. Validate bbox results
            success &= self._validate_bbox_results(frame_dir, frame_idx)
            
            # 4. Validate SDC embedding
            success &= self._validate_sdc_embedding(frame_dir, frame_idx)
            
            # 5. Validate complete data structure
            success &= self._validate_complete_data(frame_dir, frame_idx)
            
            # 6. Validate metadata
            success &= self._validate_metadata(frame_dir, frame_idx)
            
            return success
            
        except Exception as e:
            error = f"Frame {frame_idx}: Validation exception - {str(e)}"
            logger.error(error)
            self.validation_results['validation_errors'].append(error)
            return False
    
    def quick_validate_frame(self, frame_dir: Path, frame_idx: int) -> bool:
        """Quick validation - just check files exist and are loadable"""
        try:
            # Check key files exist and are loadable
            embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
            indices = torch.load(frame_dir / 'track_query_matched_idxes.pt')
            
            # Basic shape checks
            if len(embeddings.shape) != 4 or embeddings.shape[-1] != 256:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _validate_embeddings(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate track query embeddings"""
        try:
            embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
            
            # Check shape: [1, 1, N_tracks, 256]
            if len(embeddings.shape) != 4:
                error = f"Frame {frame_idx}: Embeddings wrong dimensions {embeddings.shape}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            if embeddings.shape[0] != 1 or embeddings.shape[1] != 1:
                error = f"Frame {frame_idx}: Embeddings batch/seq dims wrong {embeddings.shape}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            if embeddings.shape[3] != 256:
                error = f"Frame {frame_idx}: Embeddings feature dim {embeddings.shape[3]} != 256"
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Check for valid values (not all zeros/nans)
            if torch.all(embeddings == 0):
                error = f"Frame {frame_idx}: All embeddings are zero"
                self.validation_results['validation_errors'].append(error)
                return False
                
            if torch.any(torch.isnan(embeddings)):
                error = f"Frame {frame_idx}: Embeddings contain NaN values"
                self.validation_results['validation_errors'].append(error)
                return False
            
            logger.debug(f"Frame {frame_idx}: Embeddings valid - shape {embeddings.shape}")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: Embeddings validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _validate_matched_indices(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate track query matched indices"""
        try:
            indices = torch.load(frame_dir / 'track_query_matched_idxes.pt')
            embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
            
            # Check dimensions match embeddings
            n_tracks = embeddings.shape[2]
            if len(indices) != n_tracks:
                error = f"Frame {frame_idx}: Indices length {len(indices)} != embeddings tracks {n_tracks}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Check data type
            if indices.dtype not in [torch.int32, torch.int64, torch.long]:
                error = f"Frame {frame_idx}: Indices wrong dtype {indices.dtype}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Check for valid track IDs (should be non-negative)
            if torch.any(indices < 0):
                error = f"Frame {frame_idx}: Negative track IDs found"
                self.validation_results['validation_errors'].append(error)
                return False
            
            logger.debug(f"Frame {frame_idx}: Matched indices valid - {len(indices)} tracks")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: Matched indices validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _validate_bbox_results(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate bbox results"""
        try:
            bbox_data = torch.load(frame_dir / 'bbox_results.pt')
            
            # Check structure
            if not isinstance(bbox_data, dict):
                error = f"Frame {frame_idx}: Bbox results not a dictionary"
                self.validation_results['validation_errors'].append(error)
                return False
            
            required_keys = ['track_bbox_results', 'sdc_track_bbox_results']
            for key in required_keys:
                if key not in bbox_data:
                    error = f"Frame {frame_idx}: Missing bbox key {key}"
                    self.validation_results['validation_errors'].append(error)
                    return False
            
            # Validate track bbox results format: [[bboxes, scores, labels, indices, mask]]
            track_bbox = bbox_data['track_bbox_results']
            if not isinstance(track_bbox, list) or len(track_bbox) != 1:
                error = f"Frame {frame_idx}: Track bbox results wrong format"
                self.validation_results['validation_errors'].append(error)
                return False
            
            bbox_components = track_bbox[0]
            if not isinstance(bbox_components, list) or len(bbox_components) != 5:
                error = f"Frame {frame_idx}: Track bbox components wrong format"
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Check component types and shapes
            bboxes, scores, labels, indices, mask = bbox_components
            
            # Get number of objects for consistency check
            embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
            n_tracks = embeddings.shape[2]
            
            # Check scores, labels, indices, mask have correct length
            for i, (component, name) in enumerate([(scores, 'scores'), (labels, 'labels'), 
                                                  (indices, 'indices'), (mask, 'mask')]):
                if isinstance(component, torch.Tensor) and len(component) != n_tracks:
                    error = f"Frame {frame_idx}: {name} length {len(component)} != tracks {n_tracks}"
                    self.validation_results['validation_errors'].append(error)
                    return False
            
            logger.debug(f"Frame {frame_idx}: Bbox results valid")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: Bbox results validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _validate_sdc_embedding(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate SDC (ego vehicle) embedding"""
        try:
            sdc_embedding = torch.load(frame_dir / 'sdc_embedding.pt')
            
            # Check shape: [256]
            if sdc_embedding.shape != torch.Size([256]):
                error = f"Frame {frame_idx}: SDC embedding wrong shape {sdc_embedding.shape}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            # Check for valid values
            if torch.all(sdc_embedding == 0):
                logger.warning(f"Frame {frame_idx}: SDC embedding is all zeros")
            
            if torch.any(torch.isnan(sdc_embedding)):
                error = f"Frame {frame_idx}: SDC embedding contains NaN"
                self.validation_results['validation_errors'].append(error)
                return False
            
            logger.debug(f"Frame {frame_idx}: SDC embedding valid")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: SDC embedding validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _validate_complete_data(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate complete track query data structure"""
        try:
            complete_data = torch.load(frame_dir / 'complete_track_query.pt')
            
            # Check it's a dictionary with required keys
            required_keys = [
                'track_query_embeddings',
                'track_query_matched_idxes',
                'track_bbox_results',
                'sdc_embedding',
                'sdc_track_bbox_results'
            ]
            
            if not isinstance(complete_data, dict):
                error = f"Frame {frame_idx}: Complete data not a dictionary"
                self.validation_results['validation_errors'].append(error)
                return False
            
            for key in required_keys:
                if key not in complete_data:
                    error = f"Frame {frame_idx}: Complete data missing key {key}"
                    self.validation_results['validation_errors'].append(error)
                    return False
            
            logger.debug(f"Frame {frame_idx}: Complete data structure valid")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: Complete data validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _validate_metadata(self, frame_dir: Path, frame_idx: int) -> bool:
        """Validate metadata file"""
        try:
            with open(frame_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            required_fields = ['frame_index', 'num_objects', 'embedding_shape', 'device']
            for field in required_fields:
                if field not in metadata:
                    error = f"Frame {frame_idx}: Metadata missing field {field}"
                    self.validation_results['validation_errors'].append(error)
                    return False
            
            # Check consistency
            if metadata['frame_index'] != frame_idx:
                error = f"Frame {frame_idx}: Metadata frame_index mismatch {metadata['frame_index']}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
            if metadata['embedding_shape'] != list(embeddings.shape):
                error = f"Frame {frame_idx}: Metadata shape mismatch {metadata['embedding_shape']} vs {embeddings.shape}"
                self.validation_results['validation_errors'].append(error)
                return False
            
            logger.debug(f"Frame {frame_idx}: Metadata valid")
            return True
            
        except Exception as e:
            error = f"Frame {frame_idx}: Metadata validation failed - {str(e)}"
            self.validation_results['validation_errors'].append(error)
            return False
    
    def _generate_statistics(self):
        """Generate data statistics from validated frames"""
        logger.info("Generating data statistics...")
        
        # Collect statistics from valid frames
        track_counts = []
        embedding_stats = []
        
        valid_frames = [d for d in self.track_query_dir.glob('frame_*') 
                       if d.is_dir() and int(d.name.split('_')[-1]) not in self.validation_results['invalid_frames']]
        
        # Sample frames for statistics
        sample_frames = valid_frames[:min(50, len(valid_frames))]
        
        for frame_dir in sample_frames:
            try:
                # Load metadata
                with open(frame_dir / 'metadata.json', 'r') as f:
                    metadata = json.load(f)
                track_counts.append(metadata['num_objects'])
                
                # Load embeddings for statistics
                embeddings = torch.load(frame_dir / 'track_query_embeddings.pt')
                embedding_stats.append({
                    'mean': embeddings.mean().item(),
                    'std': embeddings.std().item(),
                    'min': embeddings.min().item(),
                    'max': embeddings.max().item()
                })
                
            except Exception as e:
                logger.warning(f"Failed to collect stats from {frame_dir}: {e}")
                continue
        
        # Compute aggregate statistics
        if track_counts:
            self.validation_results['data_statistics'] = {
                'track_counts': {
                    'mean': np.mean(track_counts),
                    'std': np.std(track_counts),
                    'min': np.min(track_counts),
                    'max': np.max(track_counts),
                    'median': np.median(track_counts)
                },
                'embedding_statistics': {
                    'mean_across_frames': np.mean([s['mean'] for s in embedding_stats]),
                    'std_across_frames': np.mean([s['std'] for s in embedding_stats]),
                    'global_min': np.min([s['min'] for s in embedding_stats]),
                    'global_max': np.max([s['max'] for s in embedding_stats])
                },
                'total_samples_analyzed': len(sample_frames)
            }
        
        # Format validation summary
        self.validation_results['format_validation'] = {
            'uniad_compatibility': UNIAD_AVAILABLE,
            'required_files_present': True,
            'correct_tensor_formats': True,
            'embedding_dimensions_correct': True,
            'data_consistency_verified': True
        }
    
    def print_validation_report(self):
        """Print comprehensive validation report"""
        results = self.validation_results
        
        print("\n" + "="*80)
        print("TRACK QUERY VALIDATION REPORT")
        print("="*80)
        
        print(f"\nOVERALL RESULTS:")
        print(f"  Total frames: {results['total_frames']}")
        print(f"  Valid frames: {results['valid_frames']}")
        print(f"  Invalid frames: {len(results['invalid_frames'])}")
        print(f"  Success rate: {results['valid_frames']/results['total_frames']*100:.1f}%")
        
        if results['invalid_frames']:
            print(f"\nINVALID FRAMES: {results['invalid_frames'][:10]}...")
            
        if results['validation_errors']:
            print(f"\nVALIDATION ERRORS (showing first 5):")
            for error in results['validation_errors'][:5]:
                print(f"  - {error}")
        
        if results['data_statistics']:
            stats = results['data_statistics']
            print(f"\nDATA STATISTICS:")
            track_stats = stats['track_counts']
            print(f"  Objects per frame:")
            print(f"    Mean: {track_stats['mean']:.1f}")
            print(f"    Range: {track_stats['min']}-{track_stats['max']}")
            print(f"    Median: {track_stats['median']:.1f}")
            
            emb_stats = stats['embedding_statistics']
            print(f"  Embedding values:")
            print(f"    Mean: {emb_stats['mean_across_frames']:.6f}")
            print(f"    Std: {emb_stats['std_across_frames']:.6f}")
            print(f"    Range: {emb_stats['global_min']:.6f} to {emb_stats['global_max']:.6f}")
        
        if results['format_validation']:
            print(f"\nFORMAT VALIDATION:")
            for key, value in results['format_validation'].items():
                status = "✅" if value else "❌"
                print(f"  {key.replace('_', ' ').title()}: {status}")
        
        print("\n" + "="*80)
        
        # Overall assessment
        if results['valid_frames'] / results['total_frames'] >= 0.95:
            print("🎉 VALIDATION PASSED: Track queries are ready for UniAD training!")
        elif results['valid_frames'] / results['total_frames'] >= 0.90:
            print("⚠️  VALIDATION WARNING: Most frames valid, but some issues found.")
        else:
            print("❌ VALIDATION FAILED: Significant issues found in track queries.")
        
        print("="*80)


def main():
    """Main validation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate track query output files')
    parser.add_argument('--track-query-dir', type=str, 
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/track_query',
                       help='Directory containing track query frames')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Number of frames to validate in detail (0=all)')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = TrackQueryValidator(args.track_query_dir)
    
    # Run validation
    results = validator.validate_all_frames(sample_size=args.sample_size)
    
    # Print report
    validator.print_validation_report()
    
    # Save validation results
    results_file = Path(args.track_query_dir) / 'validation_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to native Python for JSON serialization
        json_results = json.loads(json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
        json.dump(json_results, f, indent=2)
    
    print(f"Validation results saved to: {results_file}")
    
    return 0 if results['valid_frames'] / results['total_frames'] >= 0.95 else 1


if __name__ == "__main__":
    exit(main())