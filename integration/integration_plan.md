# UniAD-ITRI Inferencing Plan (stage 1-4)

## Overview
Integrate UniAD's multi-task autonomous driving framework with ITRI's 4-camera setup and semantic map data for end-to-end inference.

## Phase 2: UniAD Pre-trained Model Inference with ITRI Data

### Infrastructure Status ✅
Current State: Production-Ready Components
- UniAD Models: 
  - uniad_base_e2e.pth (996MB)
  - uniad_base_track_map.pth (869MB)
- Pipeline: phase5_uniad_inference.py
- Data: 9 ROS bags, 4-camera setup, CAN bus
- Systems: GPU/CPU config, metrics collection

### Implementation Steps

#### 1. Model Loading & Validation ✅
- Load Stage 2 model (uniad_base_e2e.pth)
- Adapt config for ITRI 4-cam setup
- Verify: architecture, GPU memory (~0.7GB)
- Check tensor shapes and device placement

Key Technical Achievements ✅
- Pre-trained nuScenes model successfully adapted to ITRI 4-camera setup
- Memory bridge functioning correctly with BEV feature extraction
- Attention mechanisms working with correct spatial dimensions (200×200 BEV grid)
- Tensor compatibility validated throughout the pipeline


#### 2. Data Pipeline Setup ✅
- 4-camera system integration:
  - front_100deg, front_left_100deg
  - front_right_100deg, back_60deg
- CAN bus: 18D data from .pkl
- Map: ITRI polylines → lane queries
- Tracks: 16,582 preprocessed objects

Validation Status ✅
- Camera Processing: 2,313 images validated
- CAN Bus: 18D × 10 files integrated
- Map Conversion: lanes (1, 300, 256)
- Track Objects: 16,582 loaded
- Tensors: All shapes validated

Pipeline ready for inference


#### 3. Single Frame Testing ✅
- Complete pipeline execution:
  BEVFormer → MotionFormer → OccFormer → Planner
- Validate BEV features (40k points × 256D)
- Check all output formats

#### 4. Batch Processing ✅
- Test: 5+ sequential frames
- Monitor: temporal consistency, memory
- Verify: tensor stability

Stage 4 Validation Results                   
- Single Frame: Successful
- Batch Processing (8 frames): Successful
- Pipeline Stability: All frames processed without errors
- Memory Consistency: 0.72GB per frame maintained
- Tensor Shape Stability: Consistent across all frames
- Temporal Processing: Sequential frame processing working
- GPU Memory Management: No memory leaks detected
- Attention Mechanism: 24M elements processed per frame
- BEV Features: 40k×256D stable across batch
- Device Placement: All tensors on CUDA correctly


#### 5. Inferencing
continued in integration_plan_2.py

### Deliverables

#### Expected Outcomes
- End-to-end inference pipeline
- Performance metrics baseline
- Domain gap analysis
- Production deployment readiness

#### Success Metrics
- GPU memory: <1GB
- Inference speed: <1s/frame
- Batch processing: 5+ frames
- Tensor validation: shapes & values
- Documentation: complete metrics

#### Risk Management
- Automatic GPU/CPU fallback
- Tensor shape validation
- Memory monitoring (OOM prevention)
- CPU processing fallback

### Data Flow
```
ITRI 4-Camera Images → BEVFormer → BEV Features
ITRI Semantic Map → Lane Queries ↘
ITRI Track Data → Track Queries → MotionFormer → Motion Output
                                              ↓
                              OccFormer → Occupancy Output
                                              ↓
                              Planner → Planning Trajectories
```
