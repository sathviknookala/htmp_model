# Kaggle Submission Guide

Complete step-by-step guide to train models, upload to Kaggle, and submit.

## Step 1: Train Models

Choose one of two approaches:

### Option A: Quick Training (Existing Models)
Use your existing trained models:
```bash
# Skip to Step 2
```

### Option B: Train New Ensemble (Recommended)
Train all improved models:

```bash
# Install dependencies if needed
pip install xgboost catboost

# Train models
python train_and_prepare_for_upload.py
```

This will:
- Select top 100 features
- Train LightGBM, XGBoost, CatBoost
- Save all models to `models/` directory
- Create `dataset-metadata.json` for Kaggle

**Expected time**: 10-15 minutes

## Step 2: Upload Models to Kaggle

### 2.1: Update Dataset Metadata

Edit `models/dataset-metadata.json` and replace `your-username` with your Kaggle username:

```json
{
  "title": "Hull Tactical Models",
  "id": "your-username/hull-tactical-models",
  ...
}
```

### 2.2: Create or Update Kaggle Dataset

**First time (create new dataset):**
```bash
cd models
kaggle datasets create
```

**Updating existing dataset:**
```bash
kaggle datasets version -p models -m "Updated with ensemble models"
```

### 2.3: Wait for Processing

Kaggle will process your dataset. Check status:
```bash
kaggle datasets status your-username/hull-tactical-models
```

Wait until status shows "complete" before proceeding.

## Step 3: Create Submission Notebook

### 3.1: Choose Submission Script

Two options:

**Option A: Simple (existing)**
```bash
# Use kaggle_submission.py
# Single LightGBM model with linear allocation
```

**Option B: Ensemble (recommended)**
```bash
# Use kaggle_submission_ensemble.py
# Multiple models with confidence-based allocation
```

### 3.2: Update Dataset Path

Edit your chosen submission script and update the dataset path:

```python
MODEL_DIR = Path('/kaggle/input/your-username-hull-tactical-models')
#                              ^^^^^^^^^^^^^^ Update this
```

Replace with your actual Kaggle dataset path.

### 3.3: Create Kernel Metadata

Create/update `kernel-metadata.json`:

```json
{
  "id": "your-username/hull-tactical-submission",
  "title": "Hull Tactical Submission",
  "code_file": "kaggle_submission_ensemble.py",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "your-username/hull-tactical-models",
    "kaggle-competitions/hull-tactical-market-prediction"
  ],
  "competition_sources": [
    "hull-tactical-market-prediction"
  ],
  "kernel_sources": []
}
```

Update `your-username` throughout.

## Step 4: Test Locally (Optional but Recommended)

Test your submission script locally:

```bash
python kaggle_submission_ensemble.py
```

This will:
- Load models from `models/` directory
- Run local predictions
- Show sample outputs

Check for any errors before uploading.

## Step 5: Submit to Kaggle

### 5.1: Push Submission Script

```bash
kaggle kernels push
```

### 5.2: Check Kernel Status

```bash
kaggle kernels status your-username/hull-tactical-submission
```

Wait for kernel to finish running.

### 5.3: Submit to Competition

Once kernel completes successfully:

```bash
kaggle competitions submit -c hull-tactical-market-prediction \
  -f submission.csv \
  -m "Ensemble model with confidence-based allocation"
```

Or use the web interface:
1. Go to your kernel on Kaggle
2. Click "Submit to Competition"
3. Add a description
4. Submit

## Step 6: Monitor Results

Check your submission:
1. Go to competition leaderboard
2. View your submission score
3. Check for any errors in logs

## Troubleshooting

### Models Not Found
```
Error: Model file not found
```
**Solution**: Check MODEL_DIR path in submission script matches your Kaggle dataset

### Import Errors
```
ModuleNotFoundError: No module named 'xgboost'
```
**Solution**: Add to kernel metadata or use single-model submission

### Feature Mismatch
```
KeyError: 'feature_name'
```
**Solution**: Ensure same features used in training and submission

### Memory Issues
```
MemoryError or OOM
```
**Solution**: 
- Reduce historical_data from 500 to 100 rows
- Use fewer models in ensemble
- Use single model submission

### Timeout
```
Kernel timeout after 9 hours
```
**Solution**:
- Optimize feature engineering
- Reduce model complexity
- Use simpler allocation strategy

## Quick Commands Cheat Sheet

```bash
# 1. Train models
python train_and_prepare_for_upload.py

# 2. Upload models
cd models
kaggle datasets create  # or version
cd ..

# 3. Test locally
python kaggle_submission_ensemble.py

# 4. Submit
kaggle kernels push

# 5. Check status
kaggle kernels status your-username/hull-tactical-submission

# 6. Submit to competition
kaggle competitions submit -c hull-tactical-market-prediction \
  -f submission.csv -m "Updated ensemble"
```

## File Checklist

Before submitting, ensure you have:

- [ ] `models/gbm_model.txt` - LightGBM model
- [ ] `models/gbm_model.txt.metadata` - Model metadata
- [ ] `models/feature_engineer.pkl` - Feature engineering
- [ ] `models/allocation_strategy.pkl` - Allocation strategy
- [ ] `models/model_metadata.pkl` - Configuration
- [ ] `models/dataset-metadata.json` - Updated with username
- [ ] `kaggle_submission_ensemble.py` - Updated with dataset path
- [ ] `kernel-metadata.json` - Updated with username

Optional (for ensemble):
- [ ] `models/xgb_model.json` - XGBoost
- [ ] `models/catboost_model.cbm` - CatBoost
- [ ] `models/ensemble.pkl` - Ensemble config

## Performance Expectations

### Single Model
- Correlation: 0.15-0.20
- Rank: Top 50%

### Ensemble
- Correlation: 0.20-0.25
- Rank: Top 30%

## Tips for Better Performance

1. **Feature Selection**: Use top 50-100 features only
2. **Ensemble**: Combine 3+ models for stability
3. **Confidence**: Use confidence-based allocation
4. **Rolling Window**: Keep 200-500 recent samples
5. **Testing**: Always test locally first

## Getting Help

If you encounter issues:

1. Check logs in Kaggle kernel output
2. Review `IMPROVEMENTS_README.md` for detailed docs
3. Test with `kaggle_submission.py` (simple version)
4. Check competition discussion forum

## Next Steps After Submission

1. Monitor leaderboard position
2. Analyze submission logs
3. Iterate on features/models
4. Try different allocation strategies
5. Experiment with model weights

## Advanced: Hyperparameter Tuning for Final Submission

For best results before final deadline:

```bash
# Enable tuning in train script
# Edit train_and_prepare_for_upload.py:
USE_HYPERPARAMETER_TUNING = True  # Set to True

# This will take 1-2 hours but finds better parameters
python train_and_prepare_for_upload.py

# Then follow normal upload/submit process
```

Good luck!

