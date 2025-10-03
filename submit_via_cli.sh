#!/bin/bash
# Complete Command-Line Submission Script
# Automates everything: upload models, create kernel, submit

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          KAGGLE CLI SUBMISSION - eshaanganguly               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check Kaggle CLI
echo -e "${GREEN}[1/5] Checking Kaggle CLI...${NC}"
if ! command -v kaggle &> /dev/null; then
    echo -e "${YELLOW}Kaggle CLI not found in PATH. Trying python module...${NC}"
    alias kaggle="python3 -m kaggle"
    if ! python3 -m kaggle --version &> /dev/null; then
        echo -e "${RED}Error: Kaggle not installed.${NC}"
        echo "Install with: python3 -m pip install kaggle"
        exit 1
    fi
    # Make alias permanent for this session
    shopt -s expand_aliases
    echo -e "${GREEN}✓ Using: python3 -m kaggle${NC}"
else
    echo -e "${GREEN}✓ Kaggle CLI found${NC}"
fi

# Step 2: Check API credentials
echo ""
echo -e "${GREEN}[2/5] Checking Kaggle API credentials...${NC}"
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Kaggle API credentials not found!${NC}"
    echo ""
    echo "To set up:"
    echo "1. Go to: https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Save the downloaded kaggle.json to ~/.kaggle/"
    echo ""
    echo "Quick commands:"
    echo "  mkdir -p ~/.kaggle"
    echo "  mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API credentials found${NC}"

# Test credentials
if ! python3 -m kaggle competitions list &> /dev/null; then
    echo -e "${RED}Error: API credentials invalid${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API credentials valid${NC}"

# Step 3: Upload dataset
echo ""
echo -e "${GREEN}[3/5] Uploading models to Kaggle...${NC}"
cd models

# Check if dataset exists
DATASET_EXISTS=$(python3 -m kaggle datasets list -m -s "eshaanganguly/hull-tactical-models" 2>/dev/null | wc -l)

if [ $DATASET_EXISTS -gt 1 ]; then
    echo -e "${YELLOW}Dataset exists. Creating new version...${NC}"
    python3 -m kaggle datasets version -p . -m "Ensemble models - $(date '+%Y-%m-%d %H:%M')" --dir-mode zip
    echo -e "${GREEN}✓ Dataset updated${NC}"
else
    echo -e "${YELLOW}Creating new dataset...${NC}"
    python3 -m kaggle datasets create -p . --dir-mode zip
    echo -e "${GREEN}✓ Dataset created${NC}"
fi

cd ..

# Wait for dataset to be ready
echo ""
echo -e "${YELLOW}Waiting for dataset to process...${NC}"
sleep 10

# Check dataset status
for i in {1..12}; do
    STATUS=$(python3 -m kaggle datasets status eshaanganguly/hull-tactical-models 2>&1 || echo "processing")
    if echo "$STATUS" | grep -q "ready"; then
        echo -e "${GREEN}✓ Dataset ready!${NC}"
        break
    fi
    echo -e "${YELLOW}  Still processing... ($i/12)${NC}"
    sleep 10
done

# Step 4: Push kernel
echo ""
echo -e "${GREEN}[4/5] Pushing submission kernel to Kaggle...${NC}"

# Check if kernel metadata exists
if [ ! -f "kernel-metadata.json" ]; then
    echo -e "${RED}Error: kernel-metadata.json not found${NC}"
    exit 1
fi

echo "Pushing kernel..."
python3 -m kaggle kernels push

echo -e "${GREEN}✓ Kernel pushed${NC}"

# Step 5: Monitor kernel execution
echo ""
echo -e "${GREEN}[5/5] Monitoring kernel execution...${NC}"
echo -e "${YELLOW}This will take 30-60 minutes...${NC}"
echo ""

sleep 15

for i in {1..120}; do
    STATUS=$(python3 -m kaggle kernels status eshaanganguly/hull-tactical-submission 2>&1 || echo "running")
    
    if echo "$STATUS" | grep -q "complete"; then
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}✓ Kernel completed successfully!${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        break
    elif echo "$STATUS" | grep -q "error"; then
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${RED}✗ Kernel failed!${NC}"
        echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        echo "View logs at:"
        echo "https://www.kaggle.com/code/eshaanganguly/hull-tactical-submission"
        exit 1
    fi
    
    # Show progress every 5 minutes
    if [ $((i % 10)) -eq 0 ]; then
        MINUTES=$((i / 2))
        echo -e "${YELLOW}  Still running... ($MINUTES minutes elapsed)${NC}"
    fi
    
    sleep 30
done

# Final submission
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    SUBMISSION COMPLETE                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Models uploaded${NC}"
echo -e "${GREEN}✓ Kernel executed${NC}"
echo ""
echo "Next step: Submit to competition"
echo ""
echo "View your kernel:"
echo "https://www.kaggle.com/code/eshaanganguly/hull-tactical-submission"
echo ""
echo "To submit from the web:"
echo "1. Go to the kernel URL above"
echo "2. Click 'Submit to Competition'"
echo ""
echo "Or submit via CLI (if output file available):"
echo "  kaggle competitions submit -c hull-tactical-market-prediction \\"
echo "    -f submission.csv -m 'Ensemble submission'"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Good luck! 🚀"

