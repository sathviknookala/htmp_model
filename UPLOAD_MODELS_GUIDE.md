# How to Upload Your Trained Models to Kaggle

Your models are trained and ready in the `models/` directory. Here are two ways to upload them:

## Method 1: Web Interface (Easiest - Recommended)

### Step 1: Prepare Files
Your files are ready in `models/` directory:
- gbm_model.txt (226.3 KB)
- gbm_model.txt.metadata
- xgb_model.json (502.1 KB)
- xgb_model.json.metadata
- feature_engineer.pkl
- feature_selector.pkl
- allocation_strategy.pkl
- model_metadata.pkl
- ensemble.pkl
- dataset-metadata.json

### Step 2: Create Kaggle Dataset
1. Go to https://www.kaggle.com/datasets
2. Click "New Dataset" button
3. Click "Upload" and select all files from your `models/` folder
4. Fill in dataset details:
   - **Title**: Hull Tactical Models
   - **Subtitle**: Ensemble models for market prediction
   - **Description**: LightGBM + XGBoost ensemble with feature selection
5. Click "Create"

### Step 3: Note Your Dataset Path
After creation, your dataset URL will be:
```
https://www.kaggle.com/datasets/YOUR-USERNAME/hull-tactical-models
```

Your dataset path for code is:
```
/kaggle/input/hull-tactical-models
```

Keep this path - you'll need it for the submission script!

## Method 2: Kaggle CLI (For Advanced Users)

### Step 1: Install Kaggle CLI
```bash
pip install kaggle
```

### Step 2: Configure API Key
1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`
4. Set permissions:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Upload Dataset
```bash
cd models
kaggle datasets create
```

Or to update existing dataset:
```bash
kaggle datasets version -p . -m "Updated with ensemble models"
```

## Next Steps After Upload

Once your models are uploaded, proceed to:

1. **Update submission script** with your dataset path
2. **Test locally** to ensure everything works
3. **Submit to competition**

See `KAGGLE_SUBMISSION_GUIDE.md` for detailed next steps.

---

## Troubleshooting

**Dataset upload fails:**
- Check file size limits (500MB max per dataset)
- Ensure all files are in the models/ directory
- Try web interface if CLI has issues

**Can't find dataset after upload:**
- Go to https://www.kaggle.com/YOUR-USERNAME/datasets
- Find "hull-tactical-models"
- Copy the dataset path

**Files missing:**
- Ensure you selected ALL files in models/ directory
- Re-upload if any are missing

---

Ready to continue? After uploading, let me know and I'll help you update the submission script!

