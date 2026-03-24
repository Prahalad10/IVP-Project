#!/bin/bash
# Complete HGVFI pipeline: Train, Benchmark, and Analyze

set -e

echo "=========================================="
echo "HGVFI - Complete Pipeline"
echo "=========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

# Step 1: Training
echo "=========================================="
echo "STEP 1: Training HGVFI Model"
echo "=========================================="
echo ""
python3 train.py
echo ""

# Step 2: Benchmarking
echo "=========================================="
echo "STEP 2: Benchmarking Against Baselines"
echo "=========================================="
echo ""
python3 benchmark.py
echo ""

# Step 3: Analysis
echo "=========================================="
echo "STEP 3: Analyzing Results"
echo "=========================================="
echo ""
python3 analyze_results.py
echo ""

# Summary
echo "=========================================="
echo "✓ PIPELINE COMPLETE"
echo "=========================================="
echo ""
echo "Generated Files:"
echo "  • hgvfi_model.pt (trained weights)"
echo "  • training_history.json (metrics per epoch)"
echo "  • benchmark_results.json (comparison results)"
echo ""
echo "Next Steps:"
echo "  1. Review results above"
echo "  2. Read README.md for detailed analysis"
echo "  3. Run 'python3 hgvfi.py' for quick evaluation"
echo ""
