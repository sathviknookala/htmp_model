#!/bin/bash
# Simple CLI Submission Script
set -e

# Add kaggle to PATH
export PATH="$HOME/Library/Python/3.11/bin:$PATH"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          KAGGLE SUBMISSION - eshaanganguly                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Upload dataset
echo "[1/3] Uploading models to Kaggle..."
cd models

# Check if dataset exists
if kaggle datasets list -m -s "eshaanganguly/hull-tactical-models" | grep -q "hull-tactical-models"; then
    echo "  → Updating existing dataset..."
    kaggle datasets version -p . -m "Ensemble models - $(date '+%Y-%m-%d %H:%M')" --dir-mode zip
    echo "  ✓ Dataset updated"
else
    echo "  → Creating new dataset..."
    kaggle datasets create -p . --dir-mode zip
    echo "  ✓ Dataset created"
fi

cd ..

# Wait for processing
echo ""
echo "[2/3] Waiting for dataset to process..."
sleep 15
echo "  ✓ Dataset should be ready"

# Step 2: Push kernel
echo ""
echo "[3/3] Pushing kernel to Kaggle..."
kaggle kernels push
echo "  ✓ Kernel pushed"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  SUBMISSION IN PROGRESS                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Your kernel is now running on Kaggle."
echo ""
echo "View status:"
echo "  kaggle kernels status eshaanganguly/hull-tactical-submission"
echo ""
echo "View kernel:"
echo "  https://www.kaggle.com/code/eshaanganguly/hull-tactical-submission"
echo ""
echo "The kernel will take 30-60 minutes to complete."
echo "Once done, go to the kernel page and click 'Submit to Competition'"
echo ""
echo "To monitor progress, run:"
echo "  watch -n 30 'export PATH=\"\$HOME/Library/Python/3.11/bin:\$PATH\" && kaggle kernels status eshaanganguly/hull-tactical-submission'"
echo ""
echo "Good luck! 🚀"

