#!/usr/bin/env python3
"""
BEV Memory Bridge for UniAD Integration

This module extracts memory features from BEVFormer outputs and bridges them
to the MotionFormer args_tuple format, replacing the None placeholders in
the semantic map integration.

Author: Generated for UniAD Integration Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BEVMemoryExtractor:
    """
    Extracts memory features from BEVFormer outputs for MotionFormer integration.
    
    This class handles the critical bridge between BEVFormer's BEV features and
    the memory components required by MotionFormer in the args_tuple.
    """
    
    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        embed_dims: int = 256,
        pc_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        device: str = 'cuda'
    ):
        """
        Initialize BEV memory extractor.
        
        Args:
            bev_h: BEV height dimension
            bev_w: BEV width dimension  
            embed_dims: Embedding dimension
            pc_range: Point cloud range for coordinate conversion
            device: Device for tensor operations
        """
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.device = device
        
        # Initialize positional encoding generator
        self.pos_encoder = BEVPositionalEncoding(
            bev_h, bev_w, embed_dims, pc_range, device
        )
        
    def extract_memory_features(
        self,
        bev_embed: torch.Tensor,
        img_metas: List[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract memory, memory_mask, and memory_pos from BEV features.
        
        Args:
            bev_embed: BEV feature tensor [B, C, H, W]
            img_metas: Image metadata for coordinate alignment
            
        Returns:
            Tuple of (memory, memory_mask, memory_pos)
        """
        # Handle both 4D and 3D BEV embedding formats
        if len(bev_embed.shape) == 4:
            # 4D format: [B, C, H, W]
            batch_size, channels, height, width = bev_embed.shape
            # Flatten BEV features for memory format: [B, H*W, C]  
            memory = bev_embed.permute(0, 2, 3, 1).contiguous()
            memory = memory.view(batch_size, height * width, channels)
        elif len(bev_embed.shape) == 3:
            # 3D format (already flattened): [B, H*W, C]
            batch_size, spatial_dim, channels = bev_embed.shape
            memory = bev_embed
            height, width = self.bev_h, self.bev_w  # Use configured BEV dimensions
            
            # Verify spatial dimension matches expected BEV grid
            if spatial_dim != height * width:
                logger.warning(f"BEV spatial dimension mismatch: got {spatial_dim}, expected {height * width}")
        else:
            raise ValueError(f"Unexpected BEV embed shape: {bev_embed.shape}. Expected 3D or 4D tensor.")
        
        # Generate memory mask (all positions valid for now)
        memory_mask = torch.zeros(
            batch_size, height * width, 
            dtype=torch.bool, device=self.device
        )
        
        # Generate positional encodings
        memory_pos = self.pos_encoder.generate_positional_encoding(batch_size)
        
        logger.debug(f"Extracted memory features - Memory: {memory.shape}, "
                    f"Mask: {memory_mask.shape}, Pos: {memory_pos.shape}")
        
        return memory, memory_mask, memory_pos
    
    def create_enhanced_args_tuple(
        self,
        bev_embed: torch.Tensor,
        lane_query: torch.Tensor,
        lane_query_pos: torch.Tensor,
        img_metas: List[Dict] = None
    ) -> List[Any]:
        """
        Create complete args_tuple with BEV memory features.
        
        Args:
            bev_embed: BEV feature tensor from BEVFormer
            lane_query: Lane queries from semantic map conversion
            lane_query_pos: Lane query positional encodings
            img_metas: Image metadata
            
        Returns:
            Complete args_tuple with populated memory features
        """
        # Extract memory features from BEV
        memory, memory_mask, memory_pos = self.extract_memory_features(
            bev_embed, img_metas
        )
        
        # Create spatial dimensions info
        hw_lvl = [(self.bev_h, self.bev_w)]
        
        # Ensure lane queries are on the correct device
        lane_query = lane_query.to(self.device)
        lane_query_pos = lane_query_pos.to(self.device)
        
        # Build complete args_tuple
        # Format: [memory, memory_mask, memory_pos, lane_query, None, lane_query_pos, hw_lvl]
        args_tuple = [
            memory,         # BEV memory features
            memory_mask,    # Memory attention mask
            memory_pos,     # Memory positional encodings
            lane_query,     # Lane queries from semantic map
            None,          # Unused slot
            lane_query_pos, # Lane positional encodings
            hw_lvl         # Spatial dimensions
        ]
        
        logger.info(f"Created enhanced args_tuple with BEV memory features")
        return args_tuple
    
    def validate_memory_features(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        memory_pos: torch.Tensor
    ) -> bool:
        """
        Validate extracted memory features for compatibility.
        
        Args:
            memory: Memory feature tensor
            memory_mask: Memory attention mask
            memory_pos: Memory positional encodings
            
        Returns:
            True if features are valid
        """
        try:
            # Check tensor shapes
            batch_size, seq_len, embed_dim = memory.shape
            
            if memory_mask.shape != (batch_size, seq_len):
                logger.error(f"Memory mask shape mismatch: {memory_mask.shape}")
                return False
                
            if memory_pos.shape != (batch_size, seq_len, embed_dim):
                logger.error(f"Memory pos shape mismatch: {memory_pos.shape}")
                return False
                
            # Check expected dimensions
            if seq_len != self.bev_h * self.bev_w:
                logger.error(f"Sequence length mismatch: {seq_len} vs {self.bev_h * self.bev_w}")
                return False
                
            if embed_dim != self.embed_dims:
                logger.error(f"Embedding dimension mismatch: {embed_dim} vs {self.embed_dims}")
                return False
                
            # Check device placement
            if (memory.device != memory_mask.device or 
                memory.device != memory_pos.device):
                logger.error("Memory features on different devices")
                return False
                
            logger.info("Memory features validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Memory features validation failed: {e}")
            return False


class BEVPositionalEncoding:
    """
    Generates positional encodings for BEV memory features.
    """
    
    def __init__(
        self,
        bev_h: int,
        bev_w: int, 
        embed_dims: int,
        pc_range: List[float],
        device: str = 'cuda'
    ):
        """
        Initialize positional encoding generator.
        
        Args:
            bev_h: BEV height dimension
            bev_w: BEV width dimension
            embed_dims: Embedding dimension
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
            device: Device for computations
        """
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dims = embed_dims
        self.pc_range = pc_range
        self.device = device
        
    def generate_positional_encoding(self, batch_size: int) -> torch.Tensor:
        """
        Generate 2D positional encodings for BEV grid.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Positional encoding tensor [B, H*W, C]
        """
        # Generate 2D grid coordinates
        x_coords = torch.linspace(0, 1, self.bev_w, device=self.device)
        y_coords = torch.linspace(0, 1, self.bev_h, device=self.device)
        
        # Create meshgrid
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Flatten coordinates: [H*W, 2]
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # Generate sinusoidal encodings
        pos_encoding = self._generate_sinusoidal_encoding(coords)
        
        # Expand for batch dimension: [B, H*W, C]
        pos_encoding = pos_encoding.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        return pos_encoding
    
    def _generate_sinusoidal_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Generate sinusoidal positional encodings from coordinates.
        
        Args:
            coords: 2D coordinates [N, 2]
            
        Returns:
            Sinusoidal encodings [N, embed_dims]
        """
        num_pos = coords.shape[0]
        
        # Create frequency bands  
        num_pos_feats = self.embed_dims // 4  # Divide by 4 since we have x/y and sin/cos
        temperature = 10000
        
        # Generate frequency multipliers
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=self.device
        )
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
        
        # Apply sinusoidal encoding to x and y coordinates
        pos_x = coords[:, 0:1] / dim_t  # [N, num_pos_feats]
        pos_y = coords[:, 1:2] / dim_t  # [N, num_pos_feats]
        
        # Interleave sin and cos
        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=2).flatten(1)
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=2).flatten(1)
        
        # Concatenate x and y encodings to get exact embed_dims
        pos_encoding = torch.cat([pos_x, pos_y], dim=1)  # [N, 2*num_pos_feats*2]
        
        # Ensure exact embedding dimension
        if pos_encoding.shape[1] > self.embed_dims:
            pos_encoding = pos_encoding[:, :self.embed_dims]
        elif pos_encoding.shape[1] < self.embed_dims:
            # Pad with zeros if needed
            padding = torch.zeros(
                num_pos, self.embed_dims - pos_encoding.shape[1], 
                device=self.device
            )
            pos_encoding = torch.cat([pos_encoding, padding], dim=1)
            
        return pos_encoding


class BEVMemoryBridge:
    """
    Main bridge interface for integrating BEV memory with MotionFormer.
    """
    
    def __init__(
        self,
        config: Dict = None,
        device: str = 'cuda'
    ):
        """
        Initialize BEV memory bridge.
        
        Args:
            config: Configuration dictionary
            device: Device for tensor operations
        """
        self.device = device
        
        # Default configuration
        default_config = {
            'bev_h': 200,
            'bev_w': 200, 
            'embed_dims': 256,
            'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize memory extractor
        self.memory_extractor = BEVMemoryExtractor(
            bev_h=self.config['bev_h'],
            bev_w=self.config['bev_w'],
            embed_dims=self.config['embed_dims'],
            pc_range=self.config['pc_range'],
            device=device
        )
        
    def bridge_bev_to_motion(
        self,
        bev_embed: torch.Tensor,
        lane_query: torch.Tensor,
        lane_query_pos: torch.Tensor,
        img_metas: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Bridge BEV features to MotionFormer format.
        
        Args:
            bev_embed: BEV feature tensor from BEVFormer
            lane_query: Lane queries from semantic map
            lane_query_pos: Lane query positional encodings
            img_metas: Image metadata
            
        Returns:
            Complete outs_seg dictionary for MotionFormer
        """
        # Create enhanced args_tuple with memory features
        args_tuple = self.memory_extractor.create_enhanced_args_tuple(
            bev_embed, lane_query, lane_query_pos, img_metas
        )
        
        # Validate memory features
        memory, memory_mask, memory_pos = args_tuple[:3]
        if not self.memory_extractor.validate_memory_features(
            memory, memory_mask, memory_pos
        ):
            raise ValueError("Invalid memory features generated")
        
        # Return complete outs_seg dictionary
        return {
            'args_tuple': args_tuple,
            'bev_embed': bev_embed,
            'outputs_classes': None,
            'outputs_coords': None, 
            'enc_outputs_class': None,
            'enc_outputs_coord': None,
            'reference': None
        }


def create_memory_bridge(
    config: Dict = None,
    device: str = 'cuda'
) -> BEVMemoryBridge:
    """
    Factory function to create BEV memory bridge.
    
    Args:
        config: Configuration dictionary
        device: Device for operations
        
    Returns:
        Configured BEVMemoryBridge instance
    """
    return BEVMemoryBridge(config, device)


if __name__ == "__main__":
    # Test the memory bridge
    print("Testing BEV Memory Bridge...")
    
    try:
        bridge = create_memory_bridge()
        print("✅ BEV Memory Bridge created successfully")
    except Exception as e:
        print(f"❌ BEV Memory Bridge creation failed: {e}")