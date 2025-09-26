# UniAD Inference Guide

Quick guide for running inference with the trained UniAD model on ITRI data.

## Prerequisites

- Trained model checkpoint: `ckpts/uniad_base_e2e.pth` ✅
- ITRI image data: `data/itri/hct_train/image/` ✅
- PyTorch and CUDA environment ✅

## Quick Start

### 1. Basic Inference (5 frames)
```bash
python run_uniad_inference.py
```

### 2. Process Specific Number of Frames
```bash
python run_uniad_inference.py --frames 10
```

### 3. Process Specific Frame Indices
```bash
python run_uniad_inference.py --frame-indices 0 5 10 15
```

### 4. Save to Custom Output Directory
```bash
python run_uniad_inference.py --frames 3 --output my_results/
```

### 5. CPU-only Inference
```bash
python run_uniad_inference.py --device cpu
```

## Advanced Usage

### Using Custom Configuration
```bash
python run_uniad_inference.py \
    --config projects/configs/itri_inference.py \
    --checkpoint ckpts/uniad_base_e2e.pth \
    --frames 5
```

### Full Integration Pipeline (if available)
The script automatically uses the full ITRI integration pipeline if available, which includes:
- BEVFormer feature extraction
- Semantic map conversion
- Track query processing
- Motion prediction
- Occupancy forecasting
- Path planning

### Simplified Mode
If the integration pipeline is not available, the script falls back to simplified inference mode.

## Output Structure

Results are saved to the specified output directory:
```
output/inference_results/
├── frame_0000_result.json    # Individual frame results
├── frame_0001_result.json
├── frame_0002_result.json
├── ...
└── inference_summary.json    # Overall summary
```

## Results Format

Each frame result contains:
```json
{
  "frame_index": 0,
  "timestamp": 1693123456.789,
  "ego_pose": [x, y, yaw],
  "motion_results": {...},
  "outs_occ": {...},
  "planning_results": {...},
  "inference_time": 0.85,
  "success": true
}
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   python run_uniad_inference.py --device cpu
   ```

2. **Integration Pipeline Missing**
   - Script automatically falls back to simplified mode
   - Check that semantic-map and integration modules are properly installed

3. **Checkpoint Not Found**
   ```bash
   # Verify checkpoint exists
   ls -la ckpts/uniad_base_e2e.pth
   ```

4. **ITRI Data Format Issues**
   - Ensure images are in correct format
   - Check that CAN bus data and track queries are available

### Enable Verbose Logging
```bash
python run_uniad_inference.py --verbose
```

## Performance Notes

- **GPU Memory**: ~1-2GB for inference
- **Speed**: ~1-2 seconds per frame on RTX 3090
- **Batch Processing**: Processes frames sequentially for stability

## Alternative Methods

### 1. Using Existing Integration Scripts
```bash
cd integration/
python phase5_uniad_inference.py --frames 5
```

### 2. Using Standard MMDetection3D Test Script
```bash
python tools/test.py \
    projects/configs/stage2_e2e/base_e2e.py \
    ckpts/uniad_base_e2e.pth \
    --eval bbox
```

### 3. Using Custom ITRI Configuration
```bash
python tools/test.py \
    projects/configs/itri_inference.py \
    ckpts/uniad_base_e2e.pth
```

## Model Components

The trained `uniad_base_e2e.pth` includes:
- **BEVFormer**: Multi-camera to BEV features
- **MotionFormer**: Motion prediction
- **OccFormer**: Occupancy prediction
- **PlanningHead**: Path planning

## Data Requirements

For full functionality, ensure you have:
- ✅ Multi-camera images (4 cameras)
- ✅ CAN bus data (vehicle pose/state)
- ✅ Track queries (object detection results)
- ✅ Semantic map data (lane information)

## Next Steps

1. **Validate Results**: Check output quality and consistency
2. **Visualization**: Add visualization tools for motion/planning outputs
3. **Performance Tuning**: Optimize for real-time inference
4. **Integration**: Connect with downstream autonomous driving systems