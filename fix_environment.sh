#!/bin/bash
# Fix PyTorch/Transformers compatibility issue

echo "============================================================"
echo "Fixing PyTorch/Transformers Compatibility"
echo "============================================================"

# Check current versions
echo ""
echo "Current versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: Not found"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')" 2>/dev/null || echo "Transformers: Not found"

echo ""
echo "Applying fix..."
echo ""

# Option 1: Update transformers (recommended)
echo "Option 1: Updating transformers to latest version..."
pip install --upgrade transformers

echo ""
echo "New versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo ""
echo "============================================================"
echo "✅ Fix complete! Try running your benchmark again."
echo "============================================================"
