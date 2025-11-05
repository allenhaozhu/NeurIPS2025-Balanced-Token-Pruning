#!/bin/bash

# Quick Start: Testing Different Pruning Methods
# ================================================

echo "================================"
echo "Pruning Methods Quick Test"
echo "================================"
echo ""
echo "This script will test all 4 pruning methods on POPE benchmark"
echo "(POPE is the fastest benchmark for quick testing)"
echo ""

# Setup
echo "Step 1: Setting up environment..."
cd /home/user/NeurIPS2025-Balanced-Token-Pruning

# Copy files to transformers (modify path as needed)
TRANSFORMERS_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/llama"

echo "Step 2: Copying modified files to transformers..."
echo "Target: $TRANSFORMERS_PATH"

if [ -d "$TRANSFORMERS_PATH" ]; then
    cp llava/modeling_llama.py "$TRANSFORMERS_PATH/"
    cp llava/pruning_methods.py "$TRANSFORMERS_PATH/"
    echo "✓ Files copied successfully"
else
    echo "✗ Error: Transformers path not found!"
    echo "Please update TRANSFORMERS_PATH in this script"
    exit 1
fi

echo ""
echo "Step 3: Running tests..."
echo ""

# Test BTP (baseline)
echo ">>> Testing BTP (Baseline)..."
cd llava
python test_pruning_methods.py --method btp --benchmark pope --gpu 0

echo ""
echo ">>> Testing RPD (4× faster diversity)..."
python test_pruning_methods.py --method rpd --benchmark pope --gpu 0

echo ""
echo ">>> Testing CAM (Cross-attention)..."
python test_pruning_methods.py --method cam --benchmark pope --gpu 0

echo ""
echo ">>> Testing HFP (Hybrid)..."
python test_pruning_methods.py --method hfp --benchmark pope --gpu 0

echo ""
echo "================================"
echo "All tests completed!"
echo "================================"
echo ""
echo "Results saved to: ./logs/"
echo ""
echo "To view results:"
echo "  ls -lh ./logs/"
echo ""
echo "To run more comprehensive tests:"
echo "  python test_pruning_methods.py --method all --benchmark all"
echo ""
