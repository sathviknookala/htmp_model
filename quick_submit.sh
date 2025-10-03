#!/bin/bash
# Quick Submission Script
# Automates the entire submission process

set -e  # Exit on error

echo "=========================================="
echo "HULL TACTICAL - QUICK SUBMISSION"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
KAGGLE_USERNAME=${KAGGLE_USERNAME:-"your-username"}
DATASET_NAME="hull-tactical-models"
KERNEL_NAME="hull-tactical-submission"
COMPETITION="hull-tactical-market-prediction"

# Function to print colored messages
print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Kaggle username is set
if [ "$KAGGLE_USERNAME" == "your-username" ]; then
    print_error "Please set your Kaggle username:"
    echo "export KAGGLE_USERNAME='your-actual-username'"
    exit 1
fi

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    print_error "Kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# Step 1: Train models (optional)
print_step "1. Train Models (optional)"
read -p "Do you want to train new models? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Training models..."
    python train_and_prepare_for_upload.py
    print_info "✓ Training complete"
else
    print_info "Skipping training, using existing models"
fi

# Step 2: Update metadata
print_step "2. Update Dataset Metadata"
print_info "Updating dataset metadata with username: $KAGGLE_USERNAME"

# Update dataset metadata
sed -i.bak "s/your-username/$KAGGLE_USERNAME/g" models/dataset-metadata.json
rm -f models/dataset-metadata.json.bak

# Update kernel metadata
if [ -f "kernel-metadata.json" ]; then
    sed -i.bak "s/your-username/$KAGGLE_USERNAME/g" kernel-metadata.json
    rm -f kernel-metadata.json.bak
fi

print_info "✓ Metadata updated"

# Step 3: Upload/Update dataset
print_step "3. Upload Models to Kaggle"
cd models

# Check if dataset exists
DATASET_EXISTS=$(kaggle datasets list -m -s "$KAGGLE_USERNAME/$DATASET_NAME" | wc -l)

if [ $DATASET_EXISTS -gt 1 ]; then
    print_info "Updating existing dataset..."
    kaggle datasets version -p . -m "Updated models $(date '+%Y-%m-%d %H:%M')"
else
    print_info "Creating new dataset..."
    kaggle datasets create -p .
fi

cd ..
print_info "✓ Dataset uploaded"

# Step 4: Wait for dataset processing
print_step "4. Waiting for Dataset Processing"
print_info "Waiting for dataset to be ready..."
sleep 5

for i in {1..12}; do
    STATUS=$(kaggle datasets status "$KAGGLE_USERNAME/$DATASET_NAME" 2>/dev/null || echo "processing")
    if [[ $STATUS == *"complete"* ]]; then
        print_info "✓ Dataset ready"
        break
    fi
    print_info "Still processing... ($i/12)"
    sleep 10
done

# Step 5: Update submission script
print_step "5. Update Submission Script"
print_info "Updating dataset path in submission script..."

# Backup original
cp kaggle_submission_ensemble.py kaggle_submission_ensemble.py.bak

# Update path
sed -i.tmp "s|/kaggle/input/hull-tactical-models|/kaggle/input/$KAGGLE_USERNAME-$DATASET_NAME|g" kaggle_submission_ensemble.py
rm -f kaggle_submission_ensemble.py.tmp

print_info "✓ Submission script updated"

# Step 6: Test locally (optional)
print_step "6. Test Locally (optional)"
read -p "Do you want to test locally first? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running local test..."
    timeout 30 python kaggle_submission_ensemble.py || print_info "Test completed (or timeout)"
fi

# Step 7: Create/Update kernel
print_step "7. Submit Kernel to Kaggle"

# Check if kernel metadata exists
if [ ! -f "kernel-metadata.json" ]; then
    print_info "Creating kernel metadata..."
    cat > kernel-metadata.json << EOF
{
  "id": "$KAGGLE_USERNAME/$KERNEL_NAME",
  "title": "Hull Tactical Submission",
  "code_file": "kaggle_submission_ensemble.py",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "$KAGGLE_USERNAME/$DATASET_NAME",
    "kaggle-competitions/$COMPETITION"
  ],
  "competition_sources": [
    "$COMPETITION"
  ],
  "kernel_sources": []
}
EOF
fi

print_info "Pushing kernel to Kaggle..."
kaggle kernels push

print_info "✓ Kernel submitted"

# Step 8: Monitor kernel
print_step "8. Monitor Kernel Execution"
print_info "Waiting for kernel to start..."
sleep 10

for i in {1..36}; do
    STATUS=$(kaggle kernels status "$KAGGLE_USERNAME/$KERNEL_NAME" 2>/dev/null || echo "running")
    if [[ $STATUS == *"complete"* ]]; then
        print_info "✓ Kernel completed successfully"
        break
    elif [[ $STATUS == *"error"* ]]; then
        print_error "Kernel failed. Check logs at:"
        echo "https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_NAME"
        exit 1
    fi
    print_info "Kernel running... ($i/36)"
    sleep 30
done

# Step 9: Submit to competition
print_step "9. Submit to Competition"
print_info "Submitting to competition..."

# Note: Actual submission depends on kernel output
# May need to download output and submit manually
print_info "To complete submission:"
echo "1. Go to https://www.kaggle.com/code/$KAGGLE_USERNAME/$KERNEL_NAME"
echo "2. Check kernel output for errors"
echo "3. Click 'Submit to Competition' button"
echo "4. Or download output and run:"
echo "   kaggle competitions submit -c $COMPETITION -f submission.csv -m 'Ensemble submission'"

# Cleanup
print_step "10. Cleanup"
if [ -f "kaggle_submission_ensemble.py.bak" ]; then
    print_info "Restore original file with: mv kaggle_submission_ensemble.py.bak kaggle_submission_ensemble.py"
fi

echo ""
echo "=========================================="
echo "SUBMISSION PROCESS COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check kernel logs on Kaggle"
echo "2. Submit to competition from kernel output"
echo "3. Monitor leaderboard for results"
echo ""
echo "Good luck!"

