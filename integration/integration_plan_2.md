# Stage 5: Large-Scale Dataset Inference & Evaluation

## Overview
Phased approach to scale up UniAD inference: start with 10-frame validation, then expand to full dataset processing (2,313 images, 10 sequences) with comprehensive evaluation and visualization.

## Implementation Phases

### Phase 5.1: Initial 10-Frame Pipeline Test 🧪
**Goal:** Validate pipeline with 10 frames (4 cameras each = 40 images total) as foundation

#### Test Configuration
- Select representative 10-frame sequence from one CAN bus file
- 4-camera setup per frame: front_100deg, front_left_100deg, front_right_100deg, back_60deg
- Single sequence temporal processing to validate continuity
- Baseline performance measurement

#### Pipeline Validation
- Complete UniAD pipeline execution: BEVFormer → MotionFormer → OccFormer → PlanningHead
- Tensor shape stability across 10 frames


### Phase 5.2: Large-Scale Processing ✨
**Goal:** Process complete ITRI dataset efficiently (after 10-frame validation)

#### Batch Configuration
- Optimize for RTX 3090 (16-32 frames/batch based on Phase 5.1 results)
- Memory management optimization
- Temporal sequence handling across all 10 CAN bus sequences

#### Pipeline Execution
- Process remaining 9 CAN bus sequences  
- Handle full 2,313 camera images
- Error recovery system

#### Performance Monitoring
- Real-time throughput tracking
- Memory usage profiling
- Per-module timing analysis

### Phase 5.3: Evaluation Framework 📊
**Goal:** Quantitative assessment starting with 10-frame baseline

#### Phase 5.3a: Initial Evaluation (10 frames)
- **Motion Analysis**: Trajectory consistency across 10 frames, velocity profiles validation
- **Occupancy Assessment**: BEV map quality for single sequence, temporal coherence check  
- **Planning Validation**: Navigation command execution, path smoothness analysis
- **Baseline Metrics**: Establish performance benchmarks for scaling

#### Phase 5.3b: Full Dataset Evaluation (after large-scale processing)
- **Comprehensive Motion Analysis**: Multi-sequence trajectory comparison, stability metrics
- **Extended Occupancy Assessment**: Cross-sequence BEV quality, spatial accuracy validation
- **Production Planning Validation**: Real-time performance across all sequences
- **Metrics Output**: Per-sequence JSON reports, aggregate statistics, baseline comparisons

### Phase 5.4: Visualization Suite 🎨  
**Goal:** Progressive visual output generation

#### Phase 5.4a: Initial Visualization (10 frames)
- **Tools Integration**: Use `Created/generate_scene_samples.py` as reference for ITRI adaptation
- **Basic Outputs**: 10-frame sequence visualization following NuScenes style
- **Focus Areas**: BEV occupancy heatmaps, 4-camera overlay predictions, motion trajectories
- **Image Output Priority**: Static image generation first (video generation in later stage)

#### Phase 5.4b: Full Dataset Visualization  
- **Comprehensive Gallery**: Complete visual outputs for all sequences
- **Advanced Visualizations**: Planning paths, statistical dashboards
- **Interactive Elements**: Frame-by-frame progression, side-by-side comparisons


### Phase 5.5: Deployment Validation 🚀
**Goal:** Progressive production readiness verification

#### Phase 5.5a: Initial Deployment Check (10 frames)
- **Baseline Performance**: Single-sequence processing capability
- **Memory Stability**: Monitor for leaks during 10-frame processing
- **Latency Baseline**: Per-frame processing time measurement
- **Resource Profile**: GPU/CPU utilization analysis

#### Phase 5.5b: Full Deployment Validation (after large-scale processing)  
- **Sustained Processing**: Multi-sequence, extended processing capability
- **Memory Leak Detection**: Long-term stability validation
- **Production Latency**: Real-time performance across full dataset

