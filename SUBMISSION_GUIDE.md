# Complete Submission Guide

## Yes, You Can Use Command Line!

I've created **3 ways** to submit from command line:

### 🚀 Option 1: Interactive Script (Easiest)

```bash
python3 setup_kaggle_api.py
```

This interactive script will:
- Check/install Kaggle CLI
- Verify API credentials
- Help you upload models
- Create notebook metadata
- Show submission instructions
- Check submission status

### 📝 Option 2: Bash Script

```bash
bash submit_kaggle.sh
```

Provides a quick overview and generates templates.

### ⌨️ Option 3: Manual Commands (Most Control)

See `QUICK_COMMANDS.md` for copy-paste commands.

## What You Can Do From Command Line

### ✅ Fully Automated

- Install Kaggle CLI
- Download competition data
- Upload models as dataset
- Update models
- Push/update notebook code
- Check submission status
- List competitions
- View leaderboard

### ⚠️ Requires Web Browser (One Click)

- Final submission (must click "Submit to Competition" on Kaggle)

This is a Kaggle limitation for forecasting competitions - they require final submission through web UI for security/verification.

## Complete Workflow

### Step 1: Install & Setup (One Time)

```bash
# Install
pip install kaggle

# Get credentials from https://www.kaggle.com/settings/account
# Save to ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**OR** use the interactive script:

```bash
python3 setup_kaggle_api.py
# Choose option 1-3 to setup
```

### Step 2: Upload Models

```bash
# Using interactive script
python3 setup_kaggle_api.py
# Choose option 2

# OR manually
kaggle datasets create -p models/
```

### Step 3: Push Notebook

```bash
# Using interactive script  
python3 setup_kaggle_api.py
# Choose option 3

# OR manually
kaggle kernels push
```

### Step 4: Submit (Web)

Go to your notebook URL and click "Submit to Competition"

```
https://www.kaggle.com/code/YOUR_USERNAME/hull-tactical-submission
```

### Step 5: Monitor

```bash
# Check status
kaggle competitions submissions -c hull-tactical-market-prediction

# OR use interactive script
python3 setup_kaggle_api.py
# Choose option 4
```

## Files I Created for You

| File | Purpose |
|------|---------|
| `setup_kaggle_api.py` | Interactive CLI tool (recommended) |
| `submit_kaggle.sh` | Bash script with commands |
| `QUICK_COMMANDS.md` | Quick reference for copy-paste |
| `COMMAND_LINE_SUBMISSION.md` | Detailed documentation |
| `SUBMISSION_GUIDE.md` | This file |

## Common Commands

```bash
# See all available commands
kaggle --help

# Competition commands
kaggle competitions --help
kaggle competitions list
kaggle competitions download -c hull-tactical-market-prediction
kaggle competitions submissions -c hull-tactical-market-prediction

# Dataset commands
kaggle datasets --help
kaggle datasets list --user YOUR_USERNAME
kaggle datasets create -p models/
kaggle datasets version -p models/ -m "Updated"

# Notebook commands
kaggle kernels --help
kaggle kernels list --user YOUR_USERNAME
kaggle kernels push
kaggle kernels status YOUR_USERNAME/notebook-name
```

## Troubleshooting

### "Kaggle CLI not found"

```bash
pip install kaggle
# OR
python3 -m pip install kaggle
```

### "401 Unauthorized"

Your credentials are wrong or missing:

```bash
# Check file exists
ls -la ~/.kaggle/kaggle.json

# Check permissions (should be 600)
chmod 600 ~/.kaggle/kaggle.json
```

### "403 Forbidden"

You haven't accepted competition rules:
1. Go to https://www.kaggle.com/competitions/hull-tactical-market-prediction
2. Click "I Understand and Accept"

### "Dataset not found"

Make sure dataset names match in `kernel-metadata.json`:

```json
"dataset_sources": ["YOUR_USERNAME/hull-tactical-models"]
```

### "Kernel push failed"

Check `kernel-metadata.json` format and ensure all referenced datasets exist.

## Quick Start (Copy-Paste Ready)

```bash
# 1. Setup (one time)
pip install kaggle
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Test it works
kaggle competitions list | grep -i hull

# 3. Upload models (edit YOUR_USERNAME first)
cd /path/to/htmp_model
# Edit models/dataset-metadata.json with your username
kaggle datasets create -p models/

# 4. Push notebook (edit YOUR_USERNAME first)
# Edit kernel-metadata.json with your username
kaggle kernels push

# 5. Go to web and submit
# https://www.kaggle.com/code/YOUR_USERNAME/hull-tactical-submission

# 6. Check status
kaggle competitions submissions -c hull-tactical-market-prediction
```

## Best Practice Workflow

```bash
# After training new models:
cd /path/to/htmp_model

# Update models on Kaggle
kaggle datasets version -p models/ -m "Improved features v2"

# Push updated code
kaggle kernels push

# Go to web and resubmit
```

## Summary

| Task | Command Line | Web Required |
|------|-------------|--------------|
| Setup API | ✅ Yes | ⚠️ Get token from web |
| Download data | ✅ Yes | ❌ No |
| Upload models | ✅ Yes | ❌ No |
| Push notebook | ✅ Yes | ❌ No |
| **Final submission** | ❌ No | ✅ **Yes** (one click) |
| Check status | ✅ Yes | ❌ No |
| View leaderboard | ✅ Yes | ❌ No |

**Bottom Line**: Everything except the final "Submit to Competition" button can be done from command line.

## Resources

- **Kaggle API Docs**: https://github.com/Kaggle/kaggle-api
- **Your Datasets**: https://www.kaggle.com/YOUR_USERNAME/datasets
- **Your Notebooks**: https://www.kaggle.com/YOUR_USERNAME/code
- **Competition**: https://www.kaggle.com/competitions/hull-tactical-market-prediction

## Next Step

Run this now:

```bash
python3 setup_kaggle_api.py
```

It will guide you through everything! 🚀

