# Command Line Submission Guide

Yes! You can interact with Kaggle from the command line using the **Kaggle API**.

## Quick Start

```bash
# Run the interactive setup script
python3 setup_kaggle_api.py
```

This will guide you through:
1. Installing Kaggle CLI
2. Setting up API credentials
3. Uploading your models
4. Creating and pushing notebooks
5. Checking submission status

## Manual Command Line Workflow

### 1. Install Kaggle CLI

```bash
pip install kaggle
```

### 2. Setup API Credentials

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New Token" under API section
3. Move the downloaded `kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download Competition Data

```bash
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip
```

### 4. Upload Models as Dataset

First, create `models/dataset-metadata.json`:

```json
{
  "title": "Hull Tactical Models",
  "id": "YOUR_USERNAME/hull-tactical-models",
  "licenses": [{"name": "CC0-1.0"}]
}
```

Then upload:

```bash
# First time
kaggle datasets create -p models/

# Or to update existing dataset
kaggle datasets version -p models/ -m "Updated models"
```

### 5. Push Submission Notebook

Create `kernel-metadata.json`:

```json
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
```

Push notebook:

```bash
kaggle kernels push
```

### 6. Submit the Notebook

**Important**: For forecasting competitions, you must submit via web UI:

1. Go to your notebook on Kaggle
2. Click "Save Version" → "Save & Run All"
3. After it completes, click "Submit to Competition"

### 7. Check Submission Status

```bash
kaggle competitions submissions -c hull-tactical-market-prediction
```

## Useful Kaggle CLI Commands

### Competitions

```bash
# List competitions
kaggle competitions list

# Get competition info
kaggle competitions list -c hull-tactical-market-prediction

# Download data
kaggle competitions download -c hull-tactical-market-prediction

# List your submissions
kaggle competitions submissions -c hull-tactical-market-prediction

# Download sample submission
kaggle competitions download -c hull-tactical-market-prediction -f sample_submission.csv
```

### Datasets

```bash
# List your datasets
kaggle datasets list --user YOUR_USERNAME

# Create new dataset
kaggle datasets create -p /path/to/dataset

# Update dataset
kaggle datasets version -p /path/to/dataset -m "Update message"

# Download dataset
kaggle datasets download -d YOUR_USERNAME/dataset-name
```

### Notebooks (Kernels)

```bash
# List your notebooks
kaggle kernels list --user YOUR_USERNAME

# Push/create notebook
kaggle kernels push -p /path/to/notebook

# Check notebook status
kaggle kernels status YOUR_USERNAME/notebook-name

# Pull notebook
kaggle kernels pull YOUR_USERNAME/notebook-name -p ./output
```

## Important Notes for This Competition

### This is a Forecasting Competition

Unlike standard competitions where you submit a CSV file:

1. **Submission Format**: Notebooks only (not CSV files)
2. **Evaluation**: Live forecasting - your notebook runs periodically
3. **Time Limits**: Must complete within 8-9 hours
4. **Internet**: Must be disabled
5. **Pre-trained Models**: Upload models as dataset

### Command Line Limitations

From command line, you CAN:
- ✅ Download competition data
- ✅ Upload models as dataset
- ✅ Push notebooks
- ✅ Check submission status

From command line, you CANNOT:
- ❌ Submit notebook to competition (must use web UI)
- ❌ Run notebook (happens on Kaggle servers)

### Workflow Summary

```bash
# 1. Setup (one time)
pip install kaggle
# Add credentials to ~/.kaggle/kaggle.json

# 2. Upload models
cd /path/to/htmp_model
kaggle datasets create -p models/

# 3. Push notebook
kaggle kernels push

# 4. Go to Kaggle website and submit notebook

# 5. Check status
kaggle competitions submissions -c hull-tactical-market-prediction
```

## Automated Script

Use the provided script for an interactive experience:

```bash
python3 setup_kaggle_api.py
```

This handles:
- Installing Kaggle CLI
- Credential verification
- Dataset creation
- Notebook metadata generation
- Submission instructions

## Alternative: Direct Web Upload

If command line is too complex:

1. Go to competition page
2. Click "Code" → "New Notebook"  
3. Click "Add Data" → "Upload Dataset" (upload models/)
4. Copy `kaggle_submission.py` code into notebook
5. Update `MODEL_DIR` path
6. Save & Run All
7. Submit to Competition

## Troubleshooting

### "403 Forbidden" Error

You haven't accepted competition rules. Go to competition page and click "I Understand and Accept".

### "Dataset not found"

Make sure:
- Dataset name in kernel-metadata.json matches uploaded dataset
- Dataset is public or you own it
- Dataset finished processing

### "Notebook failed to run"

Check:
- Model files are in dataset
- Paths in code match dataset location
- All dependencies in requirements.txt
- Notebook completes within time limit

## Files Created

- `setup_kaggle_api.py` - Interactive setup script
- `submit_kaggle.sh` - Bash script with commands
- `kernel-metadata.json` - Notebook metadata (generated)
- `models/dataset-metadata.json` - Dataset metadata (generated)

## Next Steps

1. Run: `python3 setup_kaggle_api.py`
2. Follow the interactive prompts
3. Upload models as dataset
4. Push notebook
5. Submit via Kaggle website
6. Monitor leaderboard!

---

**Key Takeaway**: The Kaggle CLI helps with setup and uploads, but for forecasting competitions, the final submission must be done through the web interface.

