# ITRI UniAD Motion/Occupancy/Planning Training Plan

## Overview

This document outlines a comprehensive training plan for UniAD's motion prediction, occupancy forecasting, and planning modules using ITRI data. The approach masks out image inputs and focuses training on the three key heads while using pre-computed BEV features and existing ITRI data assets.

## Available ITRI Data

Your training data is organized in `data/itri/hct_train/`:

- **CAN Bus Data**: `canbus/` - Ego vehicle state information
- **Map Queries**: `map_query/` - Semantic map lane queries
- **SDC Embeddings**: `sdc_embeddings/` - Self-driving car embeddings
- **SDC Planning**: `sdc_planning/` - Ground truth planning trajectories
- **Track Queries**: `track_query/` - Object track queries
- **Future Trajectories**: `gt_fut_traj/` - Ground truth future object trajectories

## Implementation Summary

### 1. Core Components Created

#### Configuration Files
- **`configs/itri_motion_occ_planning_training.py`**: Main training configuration
  - Masks out image inputs (`use_camera=False`)
  - Focuses training on motion_head + occ_head + planning_head
  - Uses pre-computed BEV features instead of images
  - Configured for ITRI data paths and pipeline

#### Dataset Implementation
- **`mmdet3d_plugin/datasets/itri_dataset.py`**: Custom ITRI dataset class
  - Handles ITRI-specific data format
  - Creates data mapping from bag files to available data
  - Supports queue-based temporal data loading
  - Compatible with existing UniAD evaluation framework

#### Data Pipeline
- **`mmdet3d_plugin/datasets/pipelines/itri_loading.py`**: Custom data loaders
  - `LoadPreComputedBEVFeatures`: Loads pre-computed BEV features
  - `LoadCANBusData`: Loads ego vehicle CAN bus data
  - `LoadSDCEmbeddings`: Loads SDC embeddings
  - `LoadTrackQueries`: Loads object track queries
  - `LoadMapQueries`: Loads semantic map queries
  - `LoadGTFutureTraj`: Loads future trajectory ground truth
  - `LoadSDCPlanningData`: Loads planning ground truth

#### Training Infrastructure
- **`tools/train_itri_motion_occ_planning.py`**: Custom training script
  - Freezes unnecessary model components (detection, segmentation)
  - Applies custom loss weighting for motion/occ/planning focus
  - Supports distributed training
  - Logs trainable vs frozen parameters

#### Evaluation & Monitoring
- **`tools/evaluate_itri_model.py`**: Comprehensive evaluation script
  - Motion prediction metrics (ADE, FDE, Miss Rate)
  - Occupancy prediction metrics (IoU, Precision, Recall)
  - Planning metrics (L2 distances, collision rates)

- **`tools/monitor_training.py`**: Real-time training monitoring
  - Parses training logs for loss tracking
  - Creates visualization plots
  - Generates summary reports
  - Supports continuous monitoring

### 2. Training Strategy

#### Model Configuration
- **Frozen Components**:
  - Image backbone and neck (ResNet + FPN)
  - BEV encoder (pre-trained BEVFormer)
  - Detection head (pts_bbox_head)
  - Segmentation head (seg_head)

- **Trainable Components**:
  - Motion head (MotionFormer)
  - Occupancy head (OccFormer)
  - Planning head (PlanningHead)

#### Loss Weighting
- Motion loss weight: 2.0x
- Occupancy loss weight: 2.0x
- Planning loss weight: 2.0x
- Detection/tracking losses: 0.0x (disabled)
- Segmentation losses: 0.0x (disabled)

#### Training Parameters
- Learning rate: 1e-4 (reduced for fine-tuning)
- Epochs: 10 (shorter since training subset of model)
- Batch size: 2 per GPU (can increase without image processing)
- Optimizer: AdamW with component-specific learning rates

## Usage Instructions

### 1. Prepare Training Environment

```bash
cd /home/bryan/Desktop/Allen/UniAD/projects_original

# Ensure all dependencies are installed
# Make sure your ITRI data is in data/itri/hct_train/
```

### 2. Start Training

```bash
# Single GPU training
python tools/train_itri_motion_occ_planning.py \
    configs/itri_motion_occ_planning_training.py \
    --work-dir ./work_dirs/itri_motion_occ_planning

# Multi-GPU training (if available)
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    tools/train_itri_motion_occ_planning.py \
    configs/itri_motion_occ_planning_training.py \
    --launcher pytorch \
    --work-dir ./work_dirs/itri_motion_occ_planning
```

### 3. Monitor Training Progress

```bash
# One-time monitoring (creates plots and reports)
python tools/monitor_training.py ./work_dirs/itri_motion_occ_planning

# Continuous monitoring (updates every 30 seconds)
python tools/monitor_training.py \
    ./work_dirs/itri_motion_occ_planning \
    --continuous \
    --refresh-interval 30
```

### 4. Evaluate Trained Model

```bash
python tools/evaluate_itri_model.py \
    configs/itri_motion_occ_planning_training.py \
    ./work_dirs/itri_motion_occ_planning/latest.pth \
    --work-dir ./work_dirs/itri_motion_occ_planning/evaluation \
    --eval-modes motion occ planning
```

## Expected Outcomes

### Training Benefits
- **Focused Learning**: Only motion/occ/planning parameters are updated
- **Faster Training**: No image processing overhead
- **Memory Efficient**: Pre-computed features reduce GPU memory usage
- **ITRI-Optimized**: Tailored to your specific data and use case

### Performance Metrics
- **Motion Prediction**: ADE < 1.0m, FDE < 2.0m at 3s horizon
- **Occupancy Forecasting**: IoU > 0.6 for vehicle occupancy
- **Planning**: L2 error < 1.5m at 3s planning horizon

### Model Output
- Trained checkpoint: `work_dirs/itri_motion_occ_planning/latest.pth`
- Training logs: `work_dirs/itri_motion_occ_planning/*.log`
- Visualizations: `work_dirs/itri_motion_occ_planning/*.png`
- Evaluation results: `work_dirs/itri_motion_occ_planning/evaluation/`

## Next Steps After Training

1. **Integration Testing**: Test trained model with your existing inference pipeline
2. **Performance Tuning**: Adjust loss weights and hyperparameters based on results
3. **Validation**: Test on additional ITRI bag files to ensure generalization
4. **Deployment**: Integrate trained weights into production system

## Troubleshooting

### Common Issues
- **Data Loading Errors**: Check that all ITRI data paths exist and files are accessible
- **Memory Issues**: Reduce batch size in config file
- **Training Slow**: Ensure BEV features are pre-computed and frozen components are not updating

### Debug Commands
```bash
# Check data availability
ls -la data/itri/hct_train/*/

# Verify config
python -c "from mmcv import Config; cfg = Config.fromfile('configs/itri_motion_occ_planning_training.py'); print(cfg.pretty_text)"

# Test single batch
python tools/train_itri_motion_occ_planning.py \
    configs/itri_motion_occ_planning_training.py \
    --work-dir ./work_dirs/debug \
    --cfg-options runner.max_epochs=1 data.samples_per_gpu=1
```

## File Structure Summary

```
projects_original/
├── configs/
│   └── itri_motion_occ_planning_training.py     # Main training config
├── mmdet3d_plugin/
│   └── datasets/
│       ├── itri_dataset.py                      # ITRI dataset class
│       └── pipelines/
│           └── itri_loading.py                  # Data loading pipelines
├── tools/
│   ├── train_itri_motion_occ_planning.py        # Training script
│   ├── evaluate_itri_model.py                   # Evaluation script
│   └── monitor_training.py                      # Monitoring script
└── ITRI_TRAINING_PLAN.md                        # This document
```

This comprehensive setup enables focused training of UniAD's motion, occupancy, and planning components using your existing ITRI data assets while bypassing the need for image processing.