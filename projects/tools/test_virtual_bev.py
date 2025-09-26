#!/usr/bin/env python3
#---------------------------------------------------------------------------------#
# Test Virtual BEV Implementation
# Quick test script to verify the virtual BEV approach works with dummy data
#---------------------------------------------------------------------------------#

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmdet3d_plugin.uniad.dense_heads.virtual_bev_module import VirtualBEVModule


def test_virtual_bev_module():
    """Test the VirtualBEVModule with dummy embeddings."""
    print("Testing VirtualBEVModule...")

    # Initialize module
    virtual_bev = VirtualBEVModule(
        embed_dim=256,
        bev_h=200,
        bev_w=200,
        use_learnable_bev=True,
        use_spatial_encoding=True
    )

    # Create dummy embeddings similar to ITRI data
    batch_size = 1
    embed_dim = 256

    # SDC embeddings [B, D] or [D]
    sdc_embeddings = torch.randn(batch_size, embed_dim)
    print(f"SDC embeddings shape: {sdc_embeddings.shape}")

    # Track queries [B, N_track, D] or [N_track, D]
    num_tracks = 5
    track_queries = torch.randn(batch_size, num_tracks, embed_dim)
    print(f"Track queries shape: {track_queries.shape}")

    # Map queries [B, N_map, D] or [N_map, D]
    num_maps = 10
    map_queries = torch.randn(batch_size, num_maps, embed_dim)
    print(f"Map queries shape: {map_queries.shape}")

    # Test forward pass
    print("\nTesting forward pass...")
    virtual_bev_features = virtual_bev(sdc_embeddings, track_queries, map_queries, batch_size)
    print(f"Virtual BEV features shape: {virtual_bev_features.shape}")

    expected_shape = (200 * 200, batch_size, embed_dim)
    assert virtual_bev_features.shape == expected_shape, f"Expected {expected_shape}, got {virtual_bev_features.shape}"
    print("✓ Virtual BEV features have correct shape")

    # Test dummy outs_track creation
    print("\nTesting dummy outs_track creation...")
    outs_track = virtual_bev.create_dummy_outs_track(sdc_embeddings, track_queries, batch_size)

    required_keys = ['bev_embed', 'bev_pos', 'track_query_embeddings', 'sdc_embedding', 'track_bbox_results']
    for key in required_keys:
        assert key in outs_track, f"Missing key: {key}"
    print("✓ outs_track has required keys")

    # Test dummy outs_seg creation
    print("\nTesting dummy outs_seg creation...")
    outs_seg = virtual_bev.create_dummy_outs_seg(map_queries, batch_size)

    assert 'args_tuple' in outs_seg, "Missing args_tuple in outs_seg"
    args_tuple = outs_seg['args_tuple']
    assert len(args_tuple) == 7, f"Expected 7 elements in args_tuple, got {len(args_tuple)}"
    print("✓ outs_seg has correct structure")

    print("\n✅ All tests passed!")
    return True


def test_tensor_compatibility():
    """Test different tensor formats for compatibility."""
    print("\nTesting tensor format compatibility...")

    virtual_bev = VirtualBEVModule(embed_dim=256, bev_h=200, bev_w=200)
    embed_dim = 256

    # Test 1: Single embeddings (1D tensors)
    print("Test 1: Single embeddings (1D tensors)")
    sdc_embeddings = torch.randn(embed_dim)
    track_queries = torch.randn(3, embed_dim)
    map_queries = torch.randn(5, embed_dim)

    result = virtual_bev(sdc_embeddings, track_queries, map_queries, batch_size=1)
    assert result.shape == (40000, 1, 256), f"Unexpected shape: {result.shape}"
    print("✓ 1D tensor inputs work")

    # Test 2: Batch embeddings (2D/3D tensors)
    print("Test 2: Batch embeddings (2D/3D tensors)")
    sdc_embeddings = torch.randn(2, embed_dim)
    track_queries = torch.randn(2, 3, embed_dim)
    map_queries = torch.randn(2, 5, embed_dim)

    result = virtual_bev(sdc_embeddings, track_queries, map_queries, batch_size=2)
    assert result.shape == (40000, 2, 256), f"Unexpected shape: {result.shape}"
    print("✓ Batch tensor inputs work")

    # Test 3: Missing embeddings (None values)
    print("Test 3: Missing embeddings (None values)")
    result = virtual_bev(None, None, None, batch_size=1)
    assert result.shape == (40000, 1, 256), f"Unexpected shape: {result.shape}"
    print("✓ None inputs work (using defaults)")

    print("✅ Tensor compatibility tests passed!")


def test_integration_format():
    """Test the format expected by UniAD integration."""
    print("\nTesting UniAD integration format...")

    virtual_bev = VirtualBEVModule(embed_dim=256, bev_h=200, bev_w=200)

    # Create embeddings in expected ITRI format
    sdc_embeddings = torch.randn(256)  # Single embedding

    # Track queries as dict (similar to loaded data)
    track_queries_dict = {
        'queries': torch.randn(3, 256),
        'query_pos': torch.randn(3, 256),
        'reference_points': torch.randn(3, 3)
    }

    # Map queries as dict
    map_queries_dict = {
        'lane_queries': torch.randn(5, 256),
        'lane_positions': torch.randn(5, 2),
        'lane_masks': torch.ones(5, dtype=torch.bool)
    }

    # Test with dict inputs (extract the main queries)
    track_queries = track_queries_dict['queries']
    map_queries = map_queries_dict['lane_queries']

    result = virtual_bev(sdc_embeddings, track_queries, map_queries, batch_size=1)
    print(f"Integration test result shape: {result.shape}")

    # Test dummy structure creation
    outs_track = virtual_bev.create_dummy_outs_track(sdc_embeddings, track_queries, batch_size=1)
    outs_seg = virtual_bev.create_dummy_outs_seg(map_queries, batch_size=1)

    print("✓ Integration format test passed")

    return outs_track, outs_seg, result


def main():
    """Run all tests."""
    print("=" * 60)
    print("Virtual BEV Implementation Test Suite")
    print("=" * 60)

    try:
        # Test 1: Basic functionality
        test_virtual_bev_module()

        # Test 2: Tensor compatibility
        test_tensor_compatibility()

        # Test 3: Integration format
        outs_track, outs_seg, bev_features = test_integration_format()

        # Summary
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print(f"Virtual BEV features shape: {bev_features.shape}")
        print(f"outs_track keys: {list(outs_track.keys())}")
        print(f"outs_seg keys: {list(outs_seg.keys())}")
        print("\nThe virtual BEV implementation is ready for training!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)