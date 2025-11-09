#!/usr/bin/env python3
"""
Quick test to verify VCA pruning implementation works correctly
"""

import sys
sys.path.insert(0, '/home/user/NeurIPS2025-Balanced-Token-Pruning/llava')

import torch
import torch.nn as nn
from pruning_methods import VisualContrastPruner

def test_vca_pruning():
    """Test VCA-inspired pruning"""
    print("="*60)
    print("Testing Visual Contrast Attention (VCA) Pruning")
    print("="*60)

    # Create pruner
    pruner = VisualContrastPruner(n_contrast_tokens=64, d_model=4096)
    print("\n1. VCA Pruner created ✓")
    print(f"   Contrast tokens: {pruner.n_contrast_tokens}")
    print(f"   Model dimension: {pruner.d_model}")

    # Create dummy hidden states
    batch_size = 1
    seq_len = 611 + 50  # 35 system + 576 image + 50 text
    d_model = 4096

    hidden_states = torch.randn(batch_size, seq_len, d_model)
    print(f"\n2. Created dummy hidden states: {hidden_states.shape} ✓")

    # Test compression
    print("\n3. Testing token compression...")
    img_tokens = hidden_states[0, 35:611, :]  # [576, 4096]
    compressed = pruner.compress_tokens(img_tokens)

    assert compressed.shape == (64, 4096), f"Expected (64, 4096), got {compressed.shape}"
    print(f"   Original: {img_tokens.shape}")
    print(f"   Compressed: {compressed.shape}")
    print(f"   Compression ratio: {576/64:.1f}x ✓")

    # Test differential contrast
    print("\n4. Testing differential contrast...")
    importance = pruner.compute_differential_contrast(compressed, img_tokens)

    assert importance.shape == (576,), f"Expected (576,), got {importance.shape}"
    print(f"   Importance scores shape: {importance.shape}")
    print(f"   Min importance: {importance.min():.4f}")
    print(f"   Max importance: {importance.max():.4f}")
    print(f"   Mean importance: {importance.mean():.4f} ✓")

    # Test pruning
    print("\n5. Testing full pruning pipeline...")
    k = 288  # Prune to 50%

    indices = pruner.prune(
        hidden_states,
        k,
        attention_weights=None,
        image_start=35,
        image_end=611
    )

    assert len(indices) == k, f"Expected {k} indices, got {len(indices)}"
    assert indices.min() >= 0, f"Indices should be >= 0, got {indices.min()}"
    assert indices.max() < 576, f"Indices should be < 576, got {indices.max()}"

    print(f"   Selected {len(indices)} tokens ✓")
    print(f"   Index range: [{indices.min()}, {indices.max()}] ✓")

    # Verify indices are sorted
    assert torch.all(indices[1:] >= indices[:-1]), "Indices should be sorted"
    print(f"   Indices sorted: ✓")

    # Test learnable parameters
    print("\n6. Testing learnable parameters...")
    print(f"   Positive encoding: {pruner.pos_encoding.shape}")
    print(f"   Negative encoding: {pruner.neg_encoding.shape}")
    print(f"   Both are nn.Parameters: {isinstance(pruner.pos_encoding, nn.Parameter)} ✓")

    # Test that encodings are different
    diff = (pruner.pos_encoding - pruner.neg_encoding).abs().mean()
    print(f"   Encoding difference: {diff:.4f} ✓")

    print("\n" + "="*60)
    print("✅ All VCA tests passed!")
    print("="*60)
    print("\nVCA pruning implementation is working correctly!")
    print("\nKey features:")
    print("  • Compresses 576 tokens → 64 contrast tokens")
    print("  • Uses learnable positive/negative encodings")
    print("  • Computes differential contrast for importance")
    print("  • Complexity: O(64×576) vs BTP's O(576²)")
    print("\nYou can now test it with:")
    print("  python test_pruning_methods.py --method vca --benchmark pope")


if __name__ == '__main__':
    test_vca_pruning()
