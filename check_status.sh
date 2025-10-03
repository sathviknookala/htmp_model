#!/bin/bash
# Quick status check script

export PATH="$HOME/Library/Python/3.11/bin:$PATH"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              KAGGLE KERNEL STATUS CHECK                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

STATUS=$(kaggle kernels status eshaanganguly/hull-tactical-ensemble 2>&1)
echo "Status: $STATUS"
echo ""

if echo "$STATUS" | grep -q "RUNNING"; then
    echo "✓ Kernel is running..."
    echo ""
    echo "View progress:"
    echo "  https://www.kaggle.com/code/eshaanganguly/hull-tactical-ensemble"
elif echo "$STATUS" | grep -q "COMPLETE"; then
    echo "✓✓ Kernel completed!"
    echo ""
    echo "Next step: Submit to competition"
    echo "  1. Go to: https://www.kaggle.com/code/eshaanganguly/hull-tactical-ensemble"
    echo "  2. Click 'Submit to Competition'"
elif echo "$STATUS" | grep -q "ERROR\|FAILED"; then
    echo "✗ Kernel failed"
    echo ""
    echo "Check logs at:"
    echo "  https://www.kaggle.com/code/eshaanganguly/hull-tactical-ensemble"
else
    echo "Status: $STATUS"
fi

echo ""
echo "To monitor continuously:"
echo "  watch -n 30 ./check_status.sh"

