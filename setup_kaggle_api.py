#!/usr/bin/env python3
"""
Setup and use Kaggle API for command-line submission
"""
import os
import sys
import json
import subprocess
from pathlib import Path


def check_kaggle_installed():
    """Check if kaggle CLI is installed"""
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        print("✅ Kaggle CLI is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Kaggle CLI not found")
        return False


def install_kaggle():
    """Install kaggle CLI"""
    print("\nInstalling Kaggle CLI...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
        print("✅ Kaggle CLI installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Kaggle CLI")
        return False


def check_credentials():
    """Check if Kaggle API credentials exist"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_json.exists():
        print(f"✅ Kaggle credentials found at {kaggle_json}")
        # Check permissions
        if oct(os.stat(kaggle_json).st_mode)[-3:] != '600':
            print("⚠️  Fixing credentials permissions...")
            os.chmod(kaggle_json, 0o600)
            print("✅ Permissions fixed")
        return True
    else:
        print(f"❌ Kaggle credentials not found at {kaggle_json}")
        return False


def setup_credentials():
    """Guide user through credential setup"""
    print("\n" + "="*60)
    print("KAGGLE API CREDENTIALS SETUP")
    print("="*60)
    print("\nTo get your Kaggle API credentials:")
    print("\n1. Go to: https://www.kaggle.com/settings/account")
    print("2. Scroll down to the 'API' section")
    print("3. Click 'Create New Token'")
    print("4. This will download 'kaggle.json'")
    print("\n5. Move the file to ~/.kaggle/:")
    
    kaggle_dir = Path.home() / '.kaggle'
    print(f"\n   mkdir -p {kaggle_dir}")
    print(f"   mv ~/Downloads/kaggle.json {kaggle_dir}/")
    print(f"   chmod 600 {kaggle_dir}/kaggle.json")
    
    print("\n6. Then run this script again")
    print("="*60 + "\n")


def download_competition_data(competition='hull-tactical-market-prediction'):
    """Download competition data"""
    print(f"\nDownloading data for {competition}...")
    try:
        subprocess.run(['kaggle', 'competitions', 'download', '-c', competition], check=True)
        print("✅ Data downloaded")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to download data")
        print("   Make sure you've accepted the competition rules on Kaggle website")
        return False


def create_dataset_from_models():
    """Create a Kaggle dataset from models directory"""
    print("\nCreating dataset from models...")
    
    # Create dataset metadata
    dataset_metadata = {
        "title": "Hull Tactical Models",
        "id": "YOUR_USERNAME/hull-tactical-models",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    models_dir = Path('models')
    metadata_path = models_dir / 'dataset-metadata.json'
    
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    
    print("✅ Created dataset-metadata.json")
    print("\nTo upload dataset:")
    print(f"1. Edit {metadata_path} and replace YOUR_USERNAME")
    print(f"2. Run: kaggle datasets create -p {models_dir}")
    print("   OR")
    print(f"3. Run: kaggle datasets version -p {models_dir} -m 'Updated models'")


def create_notebook_metadata():
    """Create notebook metadata for Kaggle Notebooks API"""
    
    metadata = {
        "id": "YOUR_USERNAME/hull-tactical-submission",
        "title": "Hull Tactical - Submission",
        "code_file": "kaggle_submission.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": ["YOUR_USERNAME/hull-tactical-models"],
        "competition_sources": ["hull-tactical-market-prediction"],
        "kernel_sources": []
    }
    
    with open('kernel-metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Created kernel-metadata.json")
    print("\nTo push notebook:")
    print("1. Edit kernel-metadata.json and replace YOUR_USERNAME")
    print("2. Run: kaggle kernels push")
    print("3. Go to Kaggle and submit the notebook")


def list_submissions(competition='hull-tactical-market-prediction'):
    """List your submissions"""
    print(f"\nYour submissions for {competition}:")
    try:
        subprocess.run(['kaggle', 'competitions', 'submissions', '-c', competition])
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to list submissions")
        return False


def main():
    """Main setup and submission workflow"""
    print("\n" + "="*60)
    print("KAGGLE API SETUP AND SUBMISSION")
    print("="*60)
    
    # Step 1: Check/install Kaggle CLI
    if not check_kaggle_installed():
        if input("\nInstall Kaggle CLI? (y/n): ").lower() == 'y':
            if not install_kaggle():
                sys.exit(1)
        else:
            print("\nPlease install manually: pip install kaggle")
            sys.exit(1)
    
    # Step 2: Check credentials
    if not check_credentials():
        setup_credentials()
        sys.exit(0)
    
    # Step 3: Main menu
    while True:
        print("\n" + "="*60)
        print("WHAT WOULD YOU LIKE TO DO?")
        print("="*60)
        print("\n1. Download competition data")
        print("2. Create dataset from models (for upload)")
        print("3. Create notebook metadata (for submission)")
        print("4. List your submissions")
        print("5. Show submission instructions")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            download_competition_data()
        elif choice == '2':
            create_dataset_from_models()
        elif choice == '3':
            create_notebook_metadata()
        elif choice == '4':
            list_submissions()
        elif choice == '5':
            show_instructions()
        elif choice == '6':
            break
        else:
            print("Invalid choice")


def show_instructions():
    """Show detailed submission instructions"""
    print("\n" + "="*60)
    print("SUBMISSION INSTRUCTIONS FOR FORECASTING COMPETITION")
    print("="*60)
    
    print("""
This is a FORECASTING competition, which requires a special submission process:

📋 STEP-BY-STEP SUBMISSION:

1. UPLOAD YOUR MODELS AS A DATASET:
   ├─ Run option 2 from this menu
   ├─ Edit models/dataset-metadata.json with your Kaggle username
   ├─ Run: kaggle datasets create -p models/
   └─ Note the dataset name (e.g., YOUR_USERNAME/hull-tactical-models)

2. CREATE SUBMISSION NOTEBOOK:
   ├─ Run option 3 from this menu
   ├─ Edit kernel-metadata.json with your username and dataset name
   ├─ Run: kaggle kernels push
   └─ This uploads your kaggle_submission.py as a notebook

3. SUBMIT THE NOTEBOOK:
   ├─ Go to: https://www.kaggle.com/code/YOUR_USERNAME/hull-tactical-submission
   ├─ Click "Edit" to open the notebook
   ├─ Add your dataset (models) to the notebook's data sources
   ├─ Click "Save Version" → "Save & Run All"
   ├─ Wait for it to complete
   └─ Click "Submit to Competition"

4. MONITOR SUBMISSION:
   ├─ Run option 4 from this menu
   └─ Or check: https://www.kaggle.com/competitions/hull-tactical-market-prediction/submissions

⚙️  ALTERNATIVE (Manual Web Upload):

1. Go to: https://www.kaggle.com/competitions/hull-tactical-market-prediction
2. Click "Code" → "New Notebook"
3. Upload models as dataset (Add Data → Upload)
4. Copy code from kaggle_submission.py into notebook
5. Adjust MODEL_DIR path to point to your uploaded dataset
6. Save & Run All
7. Submit to competition

📝 KEY POINTS:

- This is NOT a CSV submission - it's a notebook submission
- Your notebook must run within time limits (8-9 hours)
- Internet must be disabled in notebook settings
- Models must be pre-trained (no training in submission notebook)
- Your predict() function will be called by the evaluation system

✅ YOUR FILES ARE READY:
   ├─ kaggle_submission.py (submission code)
   ├─ models/ (trained models)
   └─ requirements.txt (dependencies)

""")


if __name__ == '__main__':
    main()

