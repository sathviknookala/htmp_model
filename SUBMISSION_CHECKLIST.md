# Kaggle Submission Checklist

Use this checklist to ensure everything is ready before submitting.

## Pre-Submission Checklist

### 1. Models
- [ ] Models are trained and saved in `models/` directory
- [ ] `models/gbm_model.txt` exists (primary model)
- [ ] `models/feature_engineer.pkl` exists
- [ ] `models/allocation_strategy.pkl` exists
- [ ] Optional: `models/xgb_model.json` and `models/catboost_model.cbm` for ensemble

### 2. Configuration Files
- [ ] `models/dataset-metadata.json` exists
- [ ] Username updated in `dataset-metadata.json` (replace `your-username`)
- [ ] `kernel-metadata.json` exists (if using kernel push)
- [ ] Username updated in `kernel-metadata.json`

### 3. Submission Script
- [ ] Choose script: `kaggle_submission.py` (simple) OR `kaggle_submission_ensemble.py` (advanced)
- [ ] Dataset path updated in submission script
- [ ] Test locally: `python kaggle_submission_ensemble.py` runs without errors

### 4. Kaggle Setup
- [ ] Kaggle CLI installed: `pip install kaggle`
- [ ] Kaggle API credentials configured (~/.kaggle/kaggle.json)
- [ ] Can run: `kaggle competitions list` successfully

## Quick Start Options

### Option A: Automated Script (Easiest)
```bash
# Set your username
export KAGGLE_USERNAME='your-actual-username'

# Run automated submission
./quick_submit.sh
```

### Option B: Manual Steps (More Control)

#### Step 1: Train Models
```bash
python train_and_prepare_for_upload.py
```
**Expected time**: 10-15 minutes

#### Step 2: Update Metadata
```bash
# Edit models/dataset-metadata.json
# Replace "your-username" with your actual Kaggle username
```

#### Step 3: Upload Models
```bash
cd models
kaggle datasets create     # First time
# OR
kaggle datasets version -p . -m "Updated models"  # Update existing
cd ..
```

#### Step 4: Wait for Processing
```bash
kaggle datasets status your-username/hull-tactical-models
```
Wait until status is "complete"

#### Step 5: Update Submission Script
```bash
# Edit kaggle_submission_ensemble.py
# Update MODEL_DIR path to match your dataset
```

#### Step 6: Test Locally
```bash
python kaggle_submission_ensemble.py
```

#### Step 7: Create Kernel Metadata
```bash
# Edit kernel-metadata.json
# Update all "your-username" references
```

#### Step 8: Submit Kernel
```bash
kaggle kernels push
```

#### Step 9: Monitor and Submit
```bash
# Check kernel status
kaggle kernels status your-username/hull-tactical-submission

# Once complete, submit from Kaggle web interface
```

### Option C: Use Existing Models (Fastest)

If you already have trained models:

```bash
# 1. Skip training, go directly to upload
cd models
kaggle datasets version -p . -m "Updated"
cd ..

# 2. Update and test submission script
# Edit kaggle_submission.py
python kaggle_submission.py

# 3. Submit
kaggle kernels push
```

## Common Issues and Solutions

### Issue: Models Not Found
**Error**: `FileNotFoundError: gbm_model.txt`
**Solution**: 
```bash
python train_and_prepare_for_upload.py
```

### Issue: Kaggle CLI Not Working
**Error**: `kaggle: command not found`
**Solution**:
```bash
pip install kaggle
# Configure API key from https://www.kaggle.com/settings
```

### Issue: Dataset Path Wrong
**Error**: `Dataset not found at /kaggle/input/...`
**Solution**: Update path in submission script to match your dataset name

### Issue: Import Errors
**Error**: `ModuleNotFoundError: No module named 'xgboost'`
**Solution**: Either install dependencies or use simple submission:
```python
# Use kaggle_submission.py instead of kaggle_submission_ensemble.py
```

### Issue: Memory Error
**Error**: `MemoryError` or OOM
**Solution**: Reduce historical data in submission script:
```python
self.historical_data = train_df.tail(100)  # Reduce from 500
```

### Issue: Timeout
**Error**: Kernel timeout
**Solution**: Simplify model or use single model submission

## Verification Steps

Before final submission, verify:

1. **Local Test Passes**
   ```bash
   python kaggle_submission_ensemble.py
   # Should run without errors
   ```

2. **Models Load Successfully**
   ```python
   # Check console output for:
   # ✓ LightGBM loaded
   # ✓ Feature engineer loaded
   # ✓ Predictor ready!
   ```

3. **Predictions in Valid Range**
   ```python
   # Predictions should be in [0, 2]
   # Check console output
   ```

4. **Dataset Accessible**
   ```bash
   kaggle datasets files your-username/hull-tactical-models
   # Should list all model files
   ```

## Performance Expectations

### Single Model (kaggle_submission.py)
- **Correlation**: 0.15-0.20
- **Rank**: Top 50%
- **Speed**: Fast inference

### Ensemble (kaggle_submission_ensemble.py)
- **Correlation**: 0.20-0.25
- **Rank**: Top 30%
- **Speed**: Moderate (3+ models)

## Time Estimates

| Task | Time Required |
|------|---------------|
| Train single model | 5 minutes |
| Train ensemble | 15 minutes |
| Upload dataset | 2-5 minutes |
| Dataset processing | 1-3 minutes |
| Kernel execution | 20-60 minutes |
| **Total (single)** | **~30 minutes** |
| **Total (ensemble)** | **~1 hour** |

## Final Check

Before hitting submit:

- [ ] All models uploaded and accessible
- [ ] Submission script tested locally
- [ ] Kernel metadata configured correctly
- [ ] Expected performance metrics understood
- [ ] Backup of original submission script saved

## Submission Command

Final submission (after kernel completes):

```bash
# From Kaggle web interface:
# 1. Go to your kernel
# 2. Click "Submit to Competition"
# 3. Add message: "Ensemble model with confidence-based allocation"
# 4. Submit

# Or via CLI (if output file available):
kaggle competitions submit -c hull-tactical-market-prediction \
  -f submission.csv \
  -m "Ensemble model with confidence-based allocation"
```

## Post-Submission

After submitting:

1. **Check Leaderboard**
   - Public score
   - Ranking
   - Percentile

2. **Review Logs**
   - Kernel execution logs
   - Any warnings or errors
   - Prediction statistics

3. **Iterate**
   - Analyze what worked
   - Try different models
   - Adjust allocation strategy
   - Resubmit improved version

## Quick Reference

```bash
# Set username
export KAGGLE_USERNAME='your-username'

# Train
python train_and_prepare_for_upload.py

# Upload
cd models && kaggle datasets version -p . && cd ..

# Test
python kaggle_submission_ensemble.py

# Submit
kaggle kernels push

# Monitor
kaggle kernels status $KAGGLE_USERNAME/hull-tactical-submission
```

## Get Help

If stuck:
1. Check `KAGGLE_SUBMISSION_GUIDE.md` for detailed steps
2. Review `IMPROVEMENTS_README.md` for model documentation
3. Check Kaggle competition discussion forum
4. Review kernel logs for specific errors

Good luck with your submission!

