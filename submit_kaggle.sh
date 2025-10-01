#!/bin/bash
# Kaggle Command Line Submission Script
# For Hull Tactical Market Prediction Competition

echo "================================================"
echo "Kaggle Command Line Submission"
echo "================================================"

# Check if kaggle CLI is installed
if ! command -v kaggle &> /dev/null; then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check for API credentials
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "❌ Kaggle API credentials not found!"
    echo ""
    echo "To set up Kaggle API:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New Token'"
    echo "4. Save kaggle.json to ~/.kaggle/"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

echo "✅ Kaggle CLI found"
echo "✅ API credentials found"
echo ""

# Competition name
COMPETITION="hull-tactical-market-prediction"

# Check competition status
echo "Checking competition status..."
kaggle competitions list | grep -i hull || echo "Competition not found in list"
echo ""

# For this forecasting competition, you need to submit via Kaggle Notebooks
# But here's how you would do it for a regular competition:

echo "================================================"
echo "IMPORTANT: This is a FORECASTING competition"
echo "================================================"
echo ""
echo "For forecasting competitions, you must:"
echo "1. Create a Kaggle Notebook"
echo "2. Upload your models as a dataset"
echo "3. Submit the notebook (not a CSV file)"
echo ""
echo "However, here's what you CAN do from command line:"
echo ""

# You can push/pull notebooks via API
echo "Available Kaggle API commands:"
echo ""
echo "# Download competition data"
echo "kaggle competitions download -c $COMPETITION"
echo ""
echo "# List your notebooks"
echo "kaggle kernels list --user YOUR_USERNAME"
echo ""
echo "# Push a notebook"
echo "kaggle kernels push -p /path/to/notebook"
echo ""
echo "# Check submission status"
echo "kaggle competitions submissions -c $COMPETITION"
echo ""

# Create a kernel metadata file for easier notebook pushing
cat > kernel-metadata.json << EOF
{
  "id": "YOUR_USERNAME/hull-tactical-submission",
  "title": "Hull Tactical Submission",
  "code_file": "kaggle_submission.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": ["YOUR_USERNAME/hull-models"],
  "competition_sources": ["hull-tactical-market-prediction"],
  "kernel_sources": []
}
EOF

echo "✅ Created kernel-metadata.json template"
echo ""
echo "Next steps:"
echo "1. Edit kernel-metadata.json with your username"
echo "2. Upload models as dataset using: kaggle datasets create -p models/"
echo "3. Push notebook using: kaggle kernels push"
echo ""


