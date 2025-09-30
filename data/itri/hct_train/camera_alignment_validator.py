#!/usr/bin/env python3
"""
Camera Alignment Validator

Validates image index mapping and alignment for ITRI cameras with different frame counts.
Ensures proper synchronization between cameras for BEV feature extraction.

Author: UniAD Integration Project
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json

# Import the extractor components
from itri_bevformer_extractor import ITRI_CameraConfig, ImageAligner


class CameraAlignmentValidator:
    """Validates camera alignment and synchronization."""
    
    def __init__(self, images_root: str = '/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/images'):
        self.images_root = Path(images_root)
        self.camera_config = ITRI_CameraConfig()
        self.image_aligner = ImageAligner(self.camera_config)
        
    def scan_camera_images(self) -> Dict[str, List[str]]:
        """Scan all camera directories and return image lists."""
        camera_images = {}
        
        for camera_name, camera_info in self.camera_config.real_cameras.items():
            data_path = camera_info['data_path']
            camera_dir = self.images_root / data_path
            
            if camera_dir.exists():
                # Get all JPG files and sort them
                image_files = sorted(glob.glob(str(camera_dir / "*.jpg")))
                camera_images[camera_name] = image_files
                print(f"{camera_name} ({data_path}): {len(image_files)} images")
            else:
                print(f"Warning: {camera_dir} not found")
                camera_images[camera_name] = []
                
        return camera_images
        
    def validate_alignment_mapping(self) -> Dict:
        """Validate alignment mapping for cameras that need it."""
        validation_results = {
            'alignment_validation': {},
            'reference_camera': self.camera_config.reference_camera,
            'total_reference_frames': self.camera_config.total_frames
        }
        
        print(f"\n=== Camera Alignment Validation ===")
        print(f"Reference camera: {self.camera_config.reference_camera} ({self.camera_config.total_frames} frames)")
        
        for camera_name, camera_info in self.camera_config.real_cameras.items():
            camera_validation = {
                'needs_alignment': self.camera_config.needs_alignment(camera_name),
                'image_count': camera_info.get('image_count', 0),
                'alignment_factor': camera_info.get('alignment_factor', 1.0),
                'sample_mappings': []
            }
            
            if self.camera_config.needs_alignment(camera_name):
                print(f"\n{camera_name}: Needs alignment")
                print(f"  - Images: {camera_info['image_count']}")
                print(f"  - Alignment factor (α): {camera_info['alignment_factor']:.3f}")
                
                # Test sample mappings
                sample_frames = [0, 10, 50, 100, 123]  # Reference frame indices
                print(f"  - Sample mappings (ref → actual):")
                
                for ref_frame in sample_frames:
                    if ref_frame < self.camera_config.total_frames:
                        actual_frame = round(ref_frame / camera_info['alignment_factor'])
                        actual_frame = min(actual_frame, camera_info['image_count'] - 1)
                        
                        mapping = {
                            'reference_frame': ref_frame,
                            'actual_frame': actual_frame,
                            'filename': self.image_aligner.get_aligned_filename(camera_name, ref_frame)
                        }
                        camera_validation['sample_mappings'].append(mapping)
                        print(f"    {ref_frame} → {actual_frame} ({mapping['filename']})")
            else:
                print(f"\n{camera_name}: No alignment needed ({camera_info['image_count']} frames)")
                
            validation_results['alignment_validation'][camera_name] = camera_validation
            
        return validation_results
        
    def validate_file_existence(self, max_frames: int = 10) -> Dict:
        """Validate that aligned files actually exist."""
        existence_validation = {
            'checked_frames': max_frames,
            'missing_files': [],
            'existing_files': {},
            'success_rate': {}
        }
        
        print(f"\n=== File Existence Validation (first {max_frames} frames) ===")
        
        for camera_name in self.camera_config.real_cameras.keys():
            missing_count = 0
            existing_count = 0
            missing_files = []
            
            for frame_idx in range(max_frames):
                filename = self.image_aligner.get_aligned_filename(camera_name, frame_idx)
                camera_info = self.camera_config.get_camera_config(camera_name)
                data_path = camera_info['data_path']
                file_path = self.images_root / data_path / filename
                
                if file_path.exists():
                    existing_count += 1
                else:
                    missing_count += 1
                    missing_files.append({
                        'frame_index': frame_idx,
                        'filename': filename,
                        'expected_path': str(file_path)
                    })
                    
            success_rate = existing_count / max_frames * 100
            print(f"{camera_name}: {existing_count}/{max_frames} files exist ({success_rate:.1f}%)")
            
            if missing_files:
                print(f"  Missing files: {[f['filename'] for f in missing_files[:5]]}")
                if len(missing_files) > 5:
                    print(f"  ... and {len(missing_files)-5} more")
                    
            existence_validation['missing_files'].extend(missing_files)
            existence_validation['existing_files'][camera_name] = existing_count
            existence_validation['success_rate'][camera_name] = success_rate
            
        return existence_validation
        
    def generate_alignment_visualization(self, output_path: str = None):
        """Generate visualization of alignment mapping."""
        if output_path is None:
            output_path = self.images_root.parent / 'camera_alignment_plot.png'
            
        # Find camera that needs alignment
        aligned_camera = None
        for camera_name, camera_info in self.camera_config.real_cameras.items():
            if self.camera_config.needs_alignment(camera_name):
                aligned_camera = camera_name
                break
                
        if aligned_camera is None:
            print("No camera needs alignment - skipping visualization")
            return
            
        camera_info = self.camera_config.get_camera_config(aligned_camera)
        alignment_factor = camera_info['alignment_factor']
        
        # Generate mapping data
        reference_frames = list(range(0, self.camera_config.total_frames, 5))  # Every 5th frame
        actual_frames = [round(ref / alignment_factor) for ref in reference_frames]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(reference_frames, actual_frames, alpha=0.7, s=50)
        plt.plot([0, self.camera_config.total_frames], [0, self.camera_config.total_frames/alignment_factor], 
                'r--', alpha=0.5, label=f'Perfect mapping (α={alignment_factor:.3f})')
        
        plt.xlabel(f'Reference Frame Index ({self.camera_config.reference_camera})')
        plt.ylabel(f'Actual Frame Index ({aligned_camera})')
        plt.title(f'Camera Frame Alignment Mapping\n{aligned_camera} alignment to {self.camera_config.reference_camera}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add statistics
        max_ref = max(reference_frames)
        max_actual = max(actual_frames)
        plt.text(0.02, 0.98, f'Reference frames: 0-{self.camera_config.total_frames-1}\n'
                              f'Actual frames: 0-{camera_info["image_count"]-1}\n'
                              f'Alignment factor: {alignment_factor:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Alignment visualization saved: {output_path}")
        
    def run_full_validation(self, output_file: str = None) -> Dict:
        """Run complete validation suite."""
        if output_file is None:
            output_file = self.images_root.parent / 'camera_alignment_validation.json'
            
        print("=== ITRI Camera Alignment Validation ===")
        
        # Scan camera images
        camera_images = self.scan_camera_images()
        
        # Validate alignment mapping
        alignment_results = self.validate_alignment_mapping()
        
        # Validate file existence
        existence_results = self.validate_file_existence()
        
        # Generate visualization
        self.generate_alignment_visualization()
        
        # Compile full results
        full_results = {
            'validation_timestamp': str(Path.cwd()),
            'images_root': str(self.images_root),
            'camera_scan': {
                camera: len(files) for camera, files in camera_images.items()
            },
            'alignment_validation': alignment_results,
            'file_existence': existence_results,
            'summary': {
                'total_reference_frames': self.camera_config.total_frames,
                'cameras_needing_alignment': sum(1 for cam in self.camera_config.real_cameras.keys() 
                                               if self.camera_config.needs_alignment(cam)),
                'overall_file_success_rate': sum(existence_results['success_rate'].values()) / 
                                           len(existence_results['success_rate']) if existence_results['success_rate'] else 0
            }
        }
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
            
        print(f"\n=== Validation Summary ===")
        print(f"Total reference frames: {full_results['summary']['total_reference_frames']}")
        print(f"Cameras needing alignment: {full_results['summary']['cameras_needing_alignment']}")
        print(f"Overall file success rate: {full_results['summary']['overall_file_success_rate']:.1f}%")
        print(f"Full results saved: {output_file}")
        
        return full_results


def main():
    """Main validation execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ITRI Camera Alignment Validator')
    parser.add_argument('--images-root', type=str, 
                       default='/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/images',
                       help='Root directory containing camera images')
    parser.add_argument('--max-frames', type=int, default=10,
                       help='Maximum frames to check for file existence')
    parser.add_argument('--output-file', type=str, default=None,
                       help='Output file for validation results')
    
    args = parser.parse_args()
    
    # Run validation
    validator = CameraAlignmentValidator(args.images_root)
    results = validator.run_full_validation(args.output_file)
    
    return results


if __name__ == '__main__':
    main()