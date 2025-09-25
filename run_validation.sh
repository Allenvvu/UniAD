#!/bin/bash
# UniAD Inference Validation Script - 100 samples

echo "=========================================="
echo "UniAD Inference Validation (100 samples)"
echo "=========================================="

# Check if in correct directory
if [ ! -f "l2g_inference_config.py" ]; then
    echo "❌ ERROR: l2g_inference_config.py not found"
    echo "Please run this script from the UniAD project root directory"
    exit 1
fi

if [ ! -f "ckpts/uniad_base_e2e.pth" ]; then
    echo "❌ ERROR: ckpts/uniad_base_e2e.pth not found"
    echo "Please ensure the checkpoint file is present"
    exit 1
fi

# Create output directory
mkdir -p validation_output

# Set environment variables
export PYTHONPATH=.

echo "🚀 Starting validation inference on 100 samples..."
echo "This should take approximately 2-5 minutes..."
echo ""

# Run inference with limited samples
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29502 \
    test_100_samples.py \
    l2g_inference_config.py \
    ckpts/uniad_base_e2e.pth \
    --out validation_output/results_100_samples.pkl \
    --launcher pytorch \
    --max-samples 100

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ VALIDATION SUCCESSFUL!"
    echo "=========================================="
    echo "📊 Results saved to: validation_output/results_100_samples.pkl"
    echo ""
    echo "📈 Next steps:"
    echo "1. Check the results file exists and has reasonable size"
    echo "2. If validation looks good, run full inference on all 2675 samples"
    echo ""
    echo "To run full inference:"
    echo "PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29500 \\"
    echo "    tools/test.py l2g_inference_config.py ckpts/uniad_base_e2e.pth \\"
    echo "    --out output/l2g_results.pkl --launcher pytorch"
    echo "=========================================="

    # Show file info
    if [ -f "validation_output/results_100_samples.pkl" ]; then
        file_size=$(du -h validation_output/results_100_samples.pkl | cut -f1)
        echo "📁 Results file size: $file_size"
    fi

else
    echo ""
    echo "=========================================="
    echo "❌ VALIDATION FAILED"
    echo "=========================================="
    echo "Please check the error messages above and fix any issues"
    echo "Common issues:"
    echo "1. GPU memory issues - try reducing batch size"
    echo "2. Missing dependencies - check environment setup"
    echo "3. Corrupted checkpoint file"
    exit 1
fi