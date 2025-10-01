# Quick Command Line Reference

## TL;DR - Command Line Submission

Yes, you can use command line! Here are the key commands:

### One-Time Setup

```bash
# 1. Install Kaggle CLI
pip install kaggle

# 2. Get API token from https://www.kaggle.com/settings/account
#    Download kaggle.json and run:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Verify it works
kaggle competitions list | grep -i hull
```

### Upload Your Models

```bash
# Create dataset metadata (first time only)
cat > models/dataset-metadata.json << 'EOF'
{
  "title": "Hull Tactical Models",
  "id": "YOUR_USERNAME/hull-tactical-models",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

# Edit YOUR_USERNAME in the file above, then:

# Upload models (first time)
kaggle datasets create -p models/

# OR update existing dataset
kaggle datasets version -p models/ -m "Updated models"
```

### Push Your Notebook

```bash
# Create notebook metadata (first time only)
cat > kernel-metadata.json << 'EOF'
{
  "id": "YOUR_USERNAME/hull-tactical-submission",
  "title": "Hull Tactical Submission",
  "code_file": "kaggle_submission.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": ["YOUR_USERNAME/hull-tactical-models"],
  "competition_sources": ["hull-tactical-market-prediction"]
}
EOF

# Edit YOUR_USERNAME in the file above, then:

# Push notebook to Kaggle
kaggle kernels push
```

### Submit (Web Required)

For forecasting competitions, you MUST complete submission via web:

1. Go to: `https://www.kaggle.com/code/YOUR_USERNAME/hull-tactical-submission`
2. Click "Save & Run All"
3. After completion, click "Submit to Competition"

### Check Status

```bash
# See your submissions
kaggle competitions submissions -c hull-tactical-market-prediction
```

## Interactive Helper

For a guided experience:

```bash
python3 setup_kaggle_api.py
```

This will walk you through all the steps interactively.

## That's It!

The key limitation is that **forecasting competitions require web UI for final submission**, but everything else (setup, uploads, updates) can be done from command line.

