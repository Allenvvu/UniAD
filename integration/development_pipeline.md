# UniAD Development Pipeline: From Integration to Production

## Current Integration: **INFERENCE ONLY** 🔍

The current integration is now **fully operational** and purely **inference-focused** using pre-trained UniAD models. Here's the scope:

### What We're Doing (Inference):
- Using **pre-trained weights** from nuScenes dataset
- Adapting **ITRI data format** → UniAD expected format
- Creating **real-time prediction pipeline** for ITRI autonomous driving data
- No model weight updates, training, or fine-tuning

### Current Status: FULLY OPERATIONAL
- ✅ **Model Initialization**: UniAD model loads successfully on both CPU and GPU
- ✅ **Single Frame Inference**: Complete pipeline working (BEVFormer → MotionFormer → OccFormer → PlanningHead)
- ✅ **Batch Inference**: 5/5 frames processed successfully
- ✅ **GPU Performance**: ~0.7GB memory usage, efficient execution
- ✅ **Pipeline Stability**: All tensor shapes consistent, no dimension mismatches

#### Working Components
- **Model Loading**: UniAD weights load correctly
- **Data Pipeline**: ITRI → UniAD format conversion
- **BEV Processing**: Memory bridge and spatial features
- **Motion Prediction**: Track query processing and trajectory generation
- **Occupancy Forecasting**: BEV-based occupancy maps
- **Path Planning**: Navigation command integration

#### Performance Metrics
- **GPU Memory**: ~0.7GB peak usage
- **Processing Speed**: Real-time capable on RTX 3090
- **Batch Stability**: Multi-frame inference reliable
- **Tensor Consistency**: All shapes validated throughout pipeline

### Why Inference First:
- **Proof of Concept**: Validate UniAD works with ITRI data
- **Domain Gap Assessment**: See how well nuScenes-trained models generalize
- **Pipeline Validation**: Ensure all data conversion modules work correctly

## Complete Development Pipeline After Integration 🚀

### **PHASE 1: INFERENCE INTEGRATION** ✅ **COMPLETE**
**Goal**: Get UniAD running with ITRI data using pre-trained weights
- ✅ Semantic map → lane queries (16,582 objects)
- ✅ Track queries in MotionFormer format
- ✅ 4-camera + CAN bus data processing
- ✅ BEV Memory Bridge - args_tuple None placeholders resolved
- ✅ Camera compatibility - dual-path strategy (4-cam + 6-cam)
- ✅ UniAD model loading with pre-trained weights
- ✅ All integration tests passed (5/5)
- ✅ End-to-end pipeline functional and stable
- ✅ Batch processing working (tensor mismatch issues resolved)
- ✅ Performance targets met (GPU inference fast and memory efficient)
- ✅ Results generation (motion, occupancy, planning)
- **Output**: Working inference pipeline with ITRI data

### **PHASE 2: VALIDATION & PERFORMANCE ANALYSIS** 
**Goal**: Assess model performance and domain gap
- Performance evaluation on ITRI scenarios vs nuScenes training
- Identify failure cases and domain adaptation needs  
- Benchmark accuracy, latency, and resource usage
- **Output**: Performance analysis report, failure case documentation

### **PHASE 3: DATA PREPARATION FOR TRAINING**
**Goal**: Prepare ITRI dataset for model training/fine-tuning
- Convert ITRI data → nuScenes format for training pipeline
- Create training/validation splits from ITRI ROS bags
- Generate ground truth labels for motion, occupancy, planning
- **Output**: ITRI training dataset in nuScenes format

### **PHASE 4: DOMAIN ADAPTATION TRAINING**
**Goal**: Fine-tune or retrain UniAD on ITRI data
- **Stage 1**: Track/Map modules fine-tuning (50GB GPU, 2 days, 8x A100)
- **Stage 2**: Full pipeline training (17GB GPU, 4 days, 8x A100)
- Compare transfer learning vs full retraining strategies
- **Output**: ITRI-optimized UniAD models

### **PHASE 5: PRODUCTION OPTIMIZATION**
**Goal**: Production-ready deployment
- Model compression and optimization for real-time performance
- Integration with ITRI vehicle systems
- Continuous monitoring and model updates
- **Output**: Production-ready autonomous driving system

## Training Requirements (Phase 4 Details)

### GPU Requirements:
- **Stage 1 Training**: ~50GB GPU memory, 2 days, 8x A100 GPUs recommended
  - Memory optimization: Change `queue_length=5` to `3` → ~30GB (V100 compatible)
- **Stage 2 Training**: ~17GB GPU memory, 4 days, 8x A100 GPUs
  - BEV encoder frozen, focus on task-specific queries
  - Compatible with V100/3090 GPUs

### Training Commands:
```bash
# Stage 1: Track/Map training
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/base_track_map.py 8

# Stage 2: Full pipeline training  
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/base_e2e.py 8

# Evaluation
./tools/uniad_dist_eval.sh ./projects/configs/stage1_track_map/base_track_map.py /path/to/ckpt.pth 8
```

## Key Milestones & Success Metrics

### Phase 1 Success Criteria:
- ✅ End-to-end inference pipeline setup
- ✅ Memory bridge args_tuple resolution
- ✅ Camera compatibility (dual-path strategy)
- ✅ Integration tests passed (5/5)
- ✅ Batch processing and performance targets met

### Phase 2 Success Criteria:
- Performance benchmarks on ITRI test scenarios
- Domain gap quantification vs nuScenes performance
- Identification of critical failure modes

### Phase 4 Success Criteria:
- Improved performance on ITRI scenarios vs pre-trained models
- Reduced domain gap and failure rates
- Model convergence within expected training time

**Total Timeline**: ~4-6 months from current state to production deployment