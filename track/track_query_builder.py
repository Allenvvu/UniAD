#!/usr/bin/env python3
"""
Track Query Format Builder for MotionFormer

This module converts transformed ITRI tracking data to the exact format 
required by MotionFormer, including track_bbox_results, track_query_embeddings,
and track_query_matched_idxes.

Based on trackq_plan.md and using reference implementations from:
- semantic-map/motionformer_integration.py:186-201 (track_bbox_results format)
- projects/mmdet3d_plugin/uniad/detectors/uniad_track.py (track query specifications)

## Notes: using global timestamps for time allignment.
# can be found: data/itri/hct_train/timestamps/continuous_timestamps.pkl
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Add data directory to path for SDC functions
data_dir = project_root / 'data' / 'itri' / 'hct_train'
sys.path.append(str(data_dir))

try:
    # Try to import UniAD components if available
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    UNIAD_AVAILABLE = True
except ImportError:
    print("UniAD components not available, using mock implementations")
    UNIAD_AVAILABLE = False

# Import SDC embedding functions
try:
    from sdc_embedding_extract import (
        create_sdc_embedding_from_canbus,
        create_sdc_track_bbox_results
    )
    SDC_FUNCTIONS_AVAILABLE = True
    print("SDC embedding functions imported successfully")
except ImportError as e:
    print(f"SDC embedding functions not available: {e}")
    create_sdc_embedding_from_canbus = None
    create_sdc_track_bbox_results = None
    SDC_FUNCTIONS_AVAILABLE = False


@dataclass 
class TrackQueryData:
    """Container for MotionFormer track query inputs"""
    track_query_embeddings: torch.Tensor      # [1, 1, N_tracks, 256]
    track_query_matched_idxes: torch.Tensor   # [N_tracks] 
    track_bbox_results: List[List[torch.Tensor]]  # [[bboxes, scores, labels, bbox_index, mask]]
    sdc_embedding: torch.Tensor               # [256]
    sdc_track_bbox_results: List[List[torch.Tensor]]  # Same format for ego vehicle
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for MotionFormer input"""
        return {
            'track_query_embeddings': self.track_query_embeddings,
            'track_query_matched_idxes': self.track_query_matched_idxes,
            'track_bbox_results': self.track_bbox_results,
            'sdc_embedding': self.sdc_embedding,
            'sdc_track_bbox_results': self.sdc_track_bbox_results
        }


class MockLiDARInstance3DBoxes:
    """Mock implementation when UniAD is not available"""
    def __init__(self, tensor_data):
        self.tensor = tensor_data
        
    def to(self, device):
        print("Error: UniAD LiDARInstance3DBoxes not available, using MockLiDARInstance3DBoxes.", file=sys.stderr)
        return MockLiDARInstance3DBoxes(self.tensor.to(device))


class ITRITrackQueryBuilder:
    """
    Track query builder for converting ITRI data to MotionFormer format.
    
    Uses reference implementations from semantic-map/motionformer_integration.py
    and follows the specifications from trackq_plan.md.
    """
    
    def __init__(self, device: str = 'cpu', embedding_dim: int = 256):
        """
        Initialize track query builder.
        
        Args:
            device: PyTorch device ('cpu' or 'cuda')
            embedding_dim: Dimension of track query embeddings (256 for UniAD)
        """
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Reference implementation from motionformer_integration.py
        self.LiDARBoxes = LiDARInstance3DBoxes if UNIAD_AVAILABLE else MockLiDARInstance3DBoxes
        
        print(f"Initialized TrackQueryBuilder:")
        print(f"  Device: {self.device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  UniAD available: {UNIAD_AVAILABLE}")
        
    def generate_track_embeddings(self, transformed_objects: List[Dict]) -> torch.Tensor:
        """
        Generate 256D track query embeddings from geometric and semantic features.
        
        Based on motionformer_integration.py:212-240 approach for synthetic embeddings.
        
        Args:
            transformed_objects: List of transformed tracking objects
            
        Returns:
            Tensor of shape [1, 1, N_tracks, 256]
        """
        if not transformed_objects:
            print("Error: No transformed objects found, returning empty embeddings.", file=sys.stderr)
            return torch.zeros(1, 1, 0, self.embedding_dim, device=self.device)
            
        embeddings = []
        
        for obj in transformed_objects:
            # Extract geometric features
            position_features = torch.tensor([
                obj['bev_x'], obj['bev_y'], obj['bev_z']], dtype=torch.float32)
            
            velocity_features = torch.tensor([
                obj['vx'], obj['vy'], obj['vz']], dtype=torch.float32)
            
            dimension_features = torch.tensor([
                obj['length'], obj['width'], obj['height']], dtype=torch.float32)
            
            # Orientation and angular velocity
            orientation_features = torch.tensor([
                np.sin(obj['yaw']), np.cos(obj['yaw']), obj['angular_velocity']], dtype=torch.float32)
            
            # Semantic features
            class_one_hot = torch.zeros(10, dtype=torch.float32)  # 10 UniAD classes
            class_one_hot[obj['class_id']] = 1.0
            
            # Confidence and temporal features  
            temporal_features = torch.tensor([
                obj['confidence'], 
                obj['original_tracked_period'],
                obj['timestamp'] % 1000  # Normalized timestamp
            ], dtype=torch.float32)
            
            # Combine all features (3+3+3+3+10+3 = 25 dimensions)
            base_features = torch.cat([
                position_features,      # 3D
                velocity_features,      # 3D  
                dimension_features,     # 3D
                orientation_features,   # 3D
                class_one_hot,         # 10D
                temporal_features      # 3D
            ])
            
            # Expand to 256D using learned-like projection
            # Use multiple linear projections to create rich representation
            expanded_embedding = self._expand_features_to_embedding(base_features)
            embeddings.append(expanded_embedding)
            
        # Stack and reshape to [1, 1, N_tracks, 256]
        embeddings_tensor = torch.stack(embeddings).unsqueeze(0).unsqueeze(0)
        embeddings_tensor = embeddings_tensor.to(self.device)
        
        return embeddings_tensor
        
    def _expand_features_to_embedding(self, base_features: torch.Tensor) -> torch.Tensor:
        """
        Expand 25D base features to 256D embedding using projection patterns.
        
        This simulates learned embeddings by creating rich feature representations.
        """
        # Create multiple projections of base features
        projections = []
        
        # Direct features
        projections.append(base_features)  # 25D
        
        # Polynomial features (squares)
        poly_features = base_features ** 2
        projections.append(poly_features[:20])  # 20D (subset)
        
        # Trigonometric features 
        trig_features = torch.cat([
            torch.sin(base_features[:10] * np.pi),
            torch.cos(base_features[:10] * np.pi)
        ])  # 20D
        projections.append(trig_features)
        
        # Cross-product features (selected combinations)
        cross_features = []
        for i in range(0, min(15, len(base_features)), 3):
            for j in range(i+1, min(i+4, len(base_features))):
                if len(cross_features) < 30:
                    cross_features.append(base_features[i] * base_features[j])
        projections.append(torch.tensor(cross_features[:30]))  # 30D
        
        # Random-like projections (deterministic based on features)
        random_seed_features = []
        for i in range(len(base_features)):
            # Create pseudo-random features based on input values
            seed_val = abs(base_features[i].item()) * 1000
            for k in range(6):  # 6 features per input
                pseudo_random = np.sin(seed_val * (k + 1) * 7.13) * 0.1
                random_seed_features.append(pseudo_random)
        projections.append(torch.tensor(random_seed_features[:150], dtype=torch.float32))  # 150D
        
        # Concatenate all projections
        full_features = torch.cat(projections)
        
        # Truncate or pad to exactly 256D
        if len(full_features) > self.embedding_dim:
            embedding = full_features[:self.embedding_dim]
        else:
            # Pad with zeros if needed
            padding = torch.zeros(self.embedding_dim - len(full_features))
            embedding = torch.cat([full_features, padding])
            
        # Normalize
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        
        return embedding
        
    def create_track_bbox_results(self, transformed_objects: List[Dict]) -> List[List[torch.Tensor]]:
        """
        Create track_bbox_results in the format expected by MotionFormer.
        
        Based on semantic-map/motionformer_integration.py:186-201 reference implementation.
        
        Format: [[bboxes, scores, labels, bbox_index, mask]]
        """
        if not transformed_objects:
            # Return empty results
            print("Error: No transformed objects found, returning empty bbox results.", file=sys.stderr)
            empty_boxes = torch.zeros(0, 9, device=self.device)  # 9DOF boxes
            empty_scores = torch.zeros(0, device=self.device)
            empty_labels = torch.zeros(0, dtype=torch.long, device=self.device)
            empty_indices = torch.zeros(0, dtype=torch.long, device=self.device)
            empty_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            
            if UNIAD_AVAILABLE:
                bbox_3d = LiDARInstance3DBoxes(empty_boxes)
            else:
                bbox_3d = MockLiDARInstance3DBoxes(empty_boxes)
                
            return [[bbox_3d, empty_scores, empty_labels, empty_indices, empty_mask]]
        
        # Extract 9DOF bounding boxes: [x, y, z, w, l, h, yaw, vx, vy]
        bboxes_list = []
        scores_list = []
        labels_list = []
        indices_list = []
        
        for i, obj in enumerate(transformed_objects):
            # 7DOF bbox format (standard for LiDARInstance3DBoxes)
            bbox_7dof = torch.tensor([
                obj['bev_x'],     # x
                obj['bev_y'],     # y  
                obj['bev_z'],     # z
                obj['width'],     # w
                obj['length'],    # l
                obj['height'],    # h
                obj['yaw']        # yaw
            ], dtype=torch.float32, device=self.device)
            
            bboxes_list.append(bbox_7dof)
            scores_list.append(obj['confidence'])
            labels_list.append(obj['class_id'])
            indices_list.append(i)  # Sequential indices
            
        # Convert to tensors
        track_bboxes_tensor = torch.stack(bboxes_list)
        track_scores_tensor = torch.tensor(scores_list, dtype=torch.float32, device=self.device)
        track_labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=self.device)
        track_indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=self.device)
        track_mask_tensor = torch.ones(len(transformed_objects), dtype=torch.bool, device=self.device)
        
        # Create LiDARInstance3DBoxes (following motionformer_integration.py:186-201)
        if UNIAD_AVAILABLE:
            bbox_3d = LiDARInstance3DBoxes(track_bboxes_tensor)
        else:
            bbox_3d = MockLiDARInstance3DBoxes(track_bboxes_tensor)
            
        # Return in the expected format
        bbox_results = [
            bbox_3d,                # LiDARInstance3DBoxes
            track_scores_tensor,    # scores  
            track_labels_tensor,    # labels
            track_indices_tensor,   # bbox_index
            track_mask_tensor       # mask
        ]
        
        return [bbox_results]
        
    def create_track_query_matched_idxes(self, transformed_objects: List[Dict]) -> torch.Tensor:
        """
        Create track_query_matched_idxes for MotionFormer.
        
        Based on projects/mmdet3d_plugin/uniad/detectors/uniad_track.py:480
        
        Args:
            transformed_objects: List of transformed objects
            
        Returns:
            Tensor of matched indices [N_tracks]
        """
        if not transformed_objects:
            return torch.zeros(0, dtype=torch.long, device=self.device)
            
        # For ITRI data, we use object IDs as matched indices
        # This maintains consistency with tracking across frames
        matched_idxes = []
        for obj in transformed_objects:
            matched_idxes.append(obj['id'])
            
        return torch.tensor(matched_idxes, dtype=torch.long, device=self.device)

    def create_fallback_sdc_embedding(self) -> torch.Tensor:
        """
        Create fallback SDC embedding when canbus data is not available.

        Returns 256D embedding representing the ego vehicle state.
        """
        # Create synthetic ego vehicle embedding
        sdc_embedding = torch.randn(self.embedding_dim, device=self.device) * 0.1
        sdc_embedding = torch.nn.functional.normalize(sdc_embedding, p=2, dim=0)

        return sdc_embedding

    def create_fallback_sdc_track_bbox_results(self) -> List[List[torch.Tensor]]:
        """
        Create fallback ego vehicle track bbox results when canbus data is not available.

        For ITRI data, we assume ego vehicle is at origin with default properties.
        """
        # Ego vehicle at origin (BEV center) - 7DOF format
        ego_bbox = torch.tensor([
            0.0, 0.0, 0.0,         # BEV center position
            1.8, 4.5, 1.5,        # Standard car dimensions (W, L, H)
            0.0                    # Yaw (facing forward)
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        ego_score = torch.tensor([1.0], dtype=torch.float32, device=self.device)
        ego_label = torch.tensor([0], dtype=torch.long, device=self.device)  # Car class
        ego_index = torch.tensor([0], dtype=torch.long, device=self.device)
        ego_mask = torch.tensor([True], dtype=torch.bool, device=self.device)

        if UNIAD_AVAILABLE:
            ego_bbox_3d = LiDARInstance3DBoxes(ego_bbox)
        else:
            ego_bbox_3d = MockLiDARInstance3DBoxes(ego_bbox)

        return [[ego_bbox_3d, ego_score, ego_label, ego_index, ego_mask]]
        

    def build_track_queries(self,
                           transformed_objects: List[Dict],
                           sdc_embedding: torch.Tensor = None,
                           sdc_track_bbox_results: List[List[torch.Tensor]] = None) -> TrackQueryData:
        """
        Build complete track query data structure for MotionFormer.

        Args:
            transformed_objects: List of transformed tracking objects
            sdc_embedding: Pre-computed SDC embedding (from sdc_embedding_extract.py)
            sdc_track_bbox_results: Pre-computed SDC bbox results (from sdc_embedding_extract.py)

        Returns:
            TrackQueryData containing all required MotionFormer inputs
        """
        print(f"Building track queries for {len(transformed_objects)} objects...")

        # Generate all required components
        track_embeddings = self.generate_track_embeddings(transformed_objects)
        track_bbox_results = self.create_track_bbox_results(transformed_objects)
        track_matched_idxes = self.create_track_query_matched_idxes(transformed_objects)

        # Use provided SDC data or create fallback
        if sdc_embedding is None:
            print("Warning: No SDC embedding provided, using fallback")
            sdc_embedding = self.create_fallback_sdc_embedding()
        else:
            # Ensure SDC embedding is on correct device
            sdc_embedding = sdc_embedding.to(self.device)

        if sdc_track_bbox_results is None:
            print("Warning: No SDC bbox results provided, using fallback")
            sdc_bbox_results = self.create_fallback_sdc_track_bbox_results()
        else:
            sdc_bbox_results = sdc_track_bbox_results

        # Create track query data structure
        track_query_data = TrackQueryData(
            track_query_embeddings=track_embeddings,
            track_query_matched_idxes=track_matched_idxes,
            track_bbox_results=track_bbox_results,
            sdc_embedding=sdc_embedding,
            sdc_track_bbox_results=sdc_bbox_results
        )

        print(f"Track query components created:")
        print(f"  Embeddings shape: {track_embeddings.shape}")
        print(f"  Matched indices: {len(track_matched_idxes)} objects")
        print(f"  Bbox results: {len(track_bbox_results[0][0].tensor if hasattr(track_bbox_results[0][0], 'tensor') else track_bbox_results[0][0])} boxes")
        print(f"  SDC embedding shape: {sdc_embedding.shape}")
        print(f"  Using canbus-derived SDC data: {sdc_embedding is not None and sdc_track_bbox_results is not None}")

        return track_query_data
        
    def save_track_queries(self, track_query_data: TrackQueryData, output_dir: str = "track_queries") -> Dict[str, str]:
        """
        Save track query data for MotionFormer integration.
        
        Args:
            track_query_data: Generated track query data
            output_dir: Directory to save track queries
            
        Returns:
            Dictionary of saved file paths
        """
        # Create output directory in track folder
        track_dir = Path(__file__).parent
        output_path = track_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        saved_files = {}
        
        # Save embeddings
        embeddings_file = output_path / "track_query_embeddings.pt"
        torch.save(track_query_data.track_query_embeddings, embeddings_file)
        saved_files['embeddings'] = str(embeddings_file)
        
        # Save matched indices
        indices_file = output_path / "track_query_matched_idxes.pt"
        torch.save(track_query_data.track_query_matched_idxes, indices_file)
        saved_files['matched_indices'] = str(indices_file)
        
        # Save SDC embedding
        sdc_file = output_path / "sdc_embedding.pt"
        torch.save(track_query_data.sdc_embedding, sdc_file)
        saved_files['sdc_embedding'] = str(sdc_file)
        
        # Save complete track query data (for MotionFormer integration)
        complete_file = output_path / "complete_track_queries.pt"
        torch.save(track_query_data.to_dict(), complete_file)
        saved_files['complete_data'] = str(complete_file)
        
        # Save metadata
        metadata = {
            "num_objects": len(track_query_data.track_query_matched_idxes),
            "embedding_dim": track_query_data.track_query_embeddings.shape[-1],
            "device": str(track_query_data.track_query_embeddings.device),
            "bbox_format": "7DOF [x,y,z,w,l,h,yaw]",
            "coordinate_system": "BEV",
            "reference_implementations": [
                "semantic-map/motionformer_integration.py:186-201",
                "projects/mmdet3d_plugin/uniad/detectors/uniad_track.py:480"
            ]
        }
        
        metadata_file = output_path / "track_queries_metadata.json" 
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = str(metadata_file)
        
        print(f"\nTrack queries saved to: {output_path}")
        for file_type, file_path in saved_files.items():
            print(f"  {file_type}: {Path(file_path).name}")
            
        return saved_files


def main():
    """Example usage of ITRITrackQueryBuilder"""
    
    # Initialize builder
    builder = ITRITrackQueryBuilder(device='cpu')
    
    # Load transformed objects
    transformed_data_path = Path("transformed_data/uniad_format_objects.json")
    if not transformed_data_path.exists():
        print(f"Error: Transformed data not found at {transformed_data_path}")
        return
        
    print(f"Loading transformed data from: {transformed_data_path}")
    with open(transformed_data_path, 'r') as f:
        transformed_objects = json.load(f)
        
    print(f"Loaded {len(transformed_objects)} transformed objects")
    
    # Build track queries (without SDC data in standalone mode)
    track_query_data = builder.build_track_queries(transformed_objects)

    # Save track queries
    saved_files = builder.save_track_queries(track_query_data)
    
    print(f"\n=== TRACK QUERY BUILDER SUCCESS ===")
    print(f"Generated complete MotionFormer-compatible track queries!")
    
    return builder, track_query_data


if __name__ == "__main__":
    main()