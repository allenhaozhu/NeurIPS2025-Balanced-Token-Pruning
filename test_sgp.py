#!/usr/bin/env python3
"""
Quick test to verify SGP implementation works correctly
"""

import sys
sys.path.insert(0, '/home/user/NeurIPS2025-Balanced-Token-Pruning/llava')

import torch
from pruning_methods import SpatialGroupedPruner

def test_spatial_grouping():
    """Test that spatial grouping creates correct groups"""
    print("="*60)
    print("Testing Spatial Grouped Pruning")
    print("="*60)

    pruner = SpatialGroupedPruner(num_groups=4, grid_size=24, grouping='spatial')

    # Test group creation
    print("\n1. Testing group creation...")
    groups = pruner.create_spatial_groups(576)

    assert len(groups) == 4, f"Expected 4 groups, got {len(groups)}"

    for i, group in enumerate(groups):
        print(f"   Group {i}: {len(group)} tokens")
        assert len(group) == 144, f"Expected 144 tokens per group, got {len(group)}"

    print("   ✓ Group creation correct!")

    # Test group indices are correct
    print("\n2. Testing spatial layout...")
    group0_indices = groups[0].numpy()

    # Group 0 should be top-left (rows 0-11, cols 0-11)
    expected_first = 0  # row 0, col 0
    expected_last = 11 * 24 + 11  # row 11, col 11 = 275

    assert group0_indices[0] == expected_first, f"Expected {expected_first}, got {group0_indices[0]}"
    assert group0_indices[-1] == expected_last, f"Expected {expected_last}, got {group0_indices[-1]}"

    print(f"   Group 0 first index: {group0_indices[0]} ✓")
    print(f"   Group 0 last index: {group0_indices[-1]} ✓")
    print("   ✓ Spatial layout correct!")

    # Test pruning
    print("\n3. Testing pruning function...")

    # Create dummy hidden states
    batch_size = 1
    seq_len = 611 + 50  # 35 system + 576 image + 50 text
    d_model = 4096

    hidden_states = torch.randn(batch_size, seq_len, d_model)

    # Create dummy attention weights
    num_heads = 32
    attention_weights = torch.randn(batch_size, num_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)

    # Test pruning to 288 tokens
    k = 288
    indices = pruner.prune(
        hidden_states,
        k,
        attention_weights=attention_weights,
        image_start=35,
        image_end=611
    )

    assert len(indices) == k, f"Expected {k} indices, got {len(indices)}"
    assert indices.min() >= 0, f"Indices should be >= 0, got {indices.min()}"
    assert indices.max() < 576, f"Indices should be < 576, got {indices.max()}"

    print(f"   Selected {len(indices)} tokens ✓")
    print(f"   Index range: [{indices.min()}, {indices.max()}] ✓")

    # Verify each group contributed
    tokens_per_group = k // 4
    group_contributions = []
    for group in groups:
        # Count how many selected tokens are from this group
        count = sum(1 for idx in indices if idx in group)
        group_contributions.append(count)

    print(f"\n   Tokens per group: {group_contributions}")
    for i, count in enumerate(group_contributions):
        assert count == tokens_per_group, f"Group {i} should have {tokens_per_group}, got {count}"

    print("   ✓ All groups contribute equally!")

    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60)
    print("\nSGP implementation is working correctly!")
    print("You can now test it with:")
    print("  python test_pruning_methods.py --method sgp --benchmark pope")


if __name__ == '__main__':
    test_spatial_grouping()
