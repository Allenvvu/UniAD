# UniAD Project Modifications Summary

## Major Areas of Development

Summary: Edit the modules to train UniAD with ITRI tracking and map dataset. Using 'projects/mmdet3d_plugin/uniad/dense_heads/virtual_bev_module.py' to create virtual BEV from sdc_embeddings, track_queries, and map_queries.

### 1. Core Model
**Modified Existing Files in projects/:**
- `projects/mmdet3d_plugin/uniad/detectors/uniad_e2e.py` - (261 lines changed)
    - Lines 12-16   Added VirtualBEVModule and DataContainer imports
    - Lines 33-43   Added use_virtual_bev flag and virtual_bev_config dict with spatial settings
    - Lines 177-402 Rewrite forward_train ITRI datacontainer input
        - Line 251-318 BEVformation for ITRI
        - Line 348-410 Forward motion_head
- `projects/mmdet3d_plugin/uniad/dense_heads/motion_head.py` - (426 lines changed)
    - Lines 113-158 Normalized tensor dimension for track_query_embeddings
    - Lines 163-353 Normalize sdc_embedding, helper functions
        - _unwrap_bbox(): Unwrap nested lists
        - _ensure_list_of_lists(): Format standardization
        - _to_tensor_like(): tensor conversion
        - _to_boxes_obj(): LiDARInstance3DBoxes generation
        - _to_5tuple(): Bbox result tuple formatting
    - Lines 384-504 forward_test() rewrite with comprehensive tensor normalization and SDC embedding handling
- `projects/mmdet3d_plugin/datasets/__init__.py`
    - Line 2  Added import for ITRIDataset
    - Line 7  Registered ITRIDataset in __all__ exports
- `projects/mmdet3d_plugin/datasets/pipelines/__init__.py`
    - Lines 7-10  Added imports for 8 ITRI loading components
    - Lines 17-19 Added ITRI components to __all__ exports 

**New Core Modules:**
- `projects/mmdet3d_plugin/uniad/dense_heads/virtual_bev_module.py`
    - Lines 13-64   VirtualBEVModule class definition with init parameters
    - Lines 21-34   __init__ with embed_dim=256, bev_h/w=200, learnable_bev, spatial_encoding flags
    - Lines 36-64   Learnable BEV parameters and projection layers (sdc/track/map projectors)
    - Lines 66-89   create_spatial_encoding() - Creates spatial position encoding for BEV grid
    - Lines 90-187  forward() - Main forward pass generating BEV from embeddings
    - Lines 283-402 create_outs_track() - Creates real tracking outputs from embeddings
- `projects/mmdet3d_plugin/datasets/itri_dataset.py` (569 lines)
    - ITRI data configuration setup
- `projects/mmdet3d_plugin/datasets/pipelines/itri_loading.py` (540 lines)
    - ITRI data loading file


### 2. Training & Evaluation
**Training Infrastructure:**
- `projects/tools/train_itri_motion_occ_planning.py` (470 lines)
    - Lines 1-91    Imports, argument parsing, and setup functions
    - Lines 38-91   parse_args() - Command line argument parser
    - Lines 92-107  freeze_model_components() - Freezes specified model layers
    - Lines 108-137 setup_model_for_itri_training() - Configures model for ITRI data
    - Lines 138-176 custom_loss_weighting() - Applies task-specific loss weights
    - Lines 177-341 main() - Main training loop setup, data loading, model building
    - Lines 342-470 train_model_itri() - Custom training function with ITRI-specific handling
- `projects/configs/itri_motion_occ_planning_training.py` (300 lines)
    - Lines 1-32    Base config import and class definitions
    - Lines 33-45   Data paths and input modality (use_camera=False, use_external=True)
    - Lines 47-72   Virtual BEV model config (embed_dim=256, bev_h/w=200)
    - Lines 74-88   ITRI dataset type and annotation files
    - Lines 80-180  train_pipeline - 8 ITRI loading transforms
    - Lines 182-230 test_pipeline - Validation data pipeline
    - Lines 232-260 data config - train/val dataloaders with queue_length=3
    - Lines 262-300 Optimizer, lr_scheduler, runner config (24 epochs)


### 3. Track Folder
**Track Processing Infrastructure:**
- `track/track_query_builder.py` (515 lines)
    - Builds track query embeddings from tracking detections
    - Handles query construction for motion prediction module
    - Manages object identity and temporal consistency


### 4. Semantic-Map Folder
**Most files were developed for the earlier ITRI 4-camera BEV-generation pipeline**
- `bevformer_integration_6cam.py`
    - BEVFormer integration for ITRI 6-camera setup (4 real + 2 dummy cameras)
    - hardware calibration with 1440x928 resolution
    - Integration with BEV memory bridge
    - Used by: `data/itri/hct_train/itri_bevformer_extractor.py`

- `motionformer_integration.py`
    - Written to bridge the track query and map query to MotionFormer
    - Referenced in `track/track_query_builder.py`

- `bev_memory_bridge.py`
    - Extract memory features from BEVFormer outputs for MotionFormer
    - Bridges BEVFormer BEV features to MotionFormer args_tuple format
    - Used by:`bevformer_integration_6cam.py`

- `coordinate_transform.py`
    - Purpose: Coordinate transformation utilities for world/ego/BEV conversions
    - Key Functions:
    - `transform_points_to_ego()` - World coords → Ego coords with rotation/translation
    - `transform_points_to_world()` - Ego coords → World coords (inverse transform)
    - `filter_points_by_bev_range()` - Filter points within BEV range
    - `normalize_coordinates()` - Normalize to [-1, 1] range
        - Semantic map Z-coordinate normalization to Z_MEAN=43.92 
    - Used by:`data/itri/hct_train/gt_lane_extraction.py`

- `geometric_utils.py`
    - Purpose: Geometric operations for polyline processing
    - Key Functions:
        - `sample_polyline_uniform()` - Uniform sampling along polylines with scipy interpolation
        - `compute_direction_features()` - Direction vector computation
        - `compute_curvature_features()` - Curvature calculation using cross product
        - `points_to_relative_coords()` - Absolute → Relative coordinate conversion
        - `compute_polyline_length()` - Total polyline length
        - `smooth_polyline()` - Gaussian smoothing with ndimage
        - `compute_lane_width()` - Average width between lane boundaries
        - `compute_heading_angle()` - Heading angles at each point
        - `resample_polyline_by_count()` - Resample to target point count
    - Used by: `data/itri/hct_train/gt_lane_extraction.py`

- `config_itri_6cam.py`
    - Purpose: UniAD configuration for ITRI 6-camera setup
    - Used by: `data/itri/hct_train/itri_bevformer_extractor.py`


### 5. Data Extraction Pipeline (data/itri/hct_train/)
**Most files were developed for the earlier ITRI 4-camera BEV-generation pipeline. We planned to use trained bevformer module from UniAD**

- `timestamp_extract.py`
    - Purpose: Extract and generate master timestamps from ROS bag files
        - `extract_bag_duration()` - Get start/end times from bag
        - `load_master_timestamps()` - Load continuous timestamps
        - `find_closest_timestamp()` - Temporal alignment with tolerance
    - Output: `timestamps/continuous_timestamps.pkl` (master time reference)
    - Used by: ALL other extraction scripts (canbus, track_query, gt_sdc, gt_fut_traj)

- `canbus_extraction.py`
    - Purpose: Extract 18-D CAN bus vectors from ROS bags
        - Reads `/car_state` (position, quaternion)
        - Reads `/imu/data` (linear accel, angular velocity)
        - Reads `/filter/velocity` (velocity)
    - Output: `canbus/{bag_name}_can_bus.pkl` (5,627 frames for 0.bag)
    - Imports: `timestamp_extract`, `tf.transformations`

- `track_query_extract.py`
    - Purpose: Convert ROS detected objects to MotionFormer track queries
    - Uses ITRITrackQueryBuilder from `track/track_query_builder.py`
    - Integrates SDC embedding from sdc_embedding_extract
    - Output: `track_query/*.pt` (549/562 frames successful, 13 failed)
    - Imports: `track_query_builder`, `timestamp_extract`, `sdc_embedding_extract`

- `sdc_embedding_extract.py`
    - Purpose: Create SDC (ego vehicle) embeddings from CAN bus data
    - `create_sdc_embedding_from_canbus()` - Generate SDC query embedding
    - `create_sdc_track_bbox_results()` - Create ego bbox results
    - Integrates with track_query generation
    - Used by: `track_query_extract.py`

- `gt_sdc_extraction.py`
    - Purpose: Generate ground truth SDC trajectories for planning
    - Output: `gt_sdc/*.pt` (5,616 frames - ground truth ego trajectories)
    - Imports: `timestamp_extract`, `canbus_extraction`
    - Data: Future 6 seconds at 2Hz sampling

- `gt_fut_traj_extract.py`
    - Purpose: Extract ground truth future trajectories for all tracked objects
    - Output format: (num_objects, predict_steps, 2) with masks
    - Output: `gt_fut_traj/*.pt` (per-frame future trajectory data)

- `gt_lane_extraction.py`
    - Purpose: Convert semantic map polylines to lane ground truth
        - gt_lane_labels - Lane class labels
        - gt_lane_bboxes - Bounding boxes [x_min, y_min, x_max, y_max]
        - gt_lane_masks - Binary segmentation masks [H, W]
    - Imports: `coordinate_transform`, `geometric_utils` from semantic-map
    - Output: `gt_lane/*.pt`

- `itri_bevformer_extractor.py`  **PRIMARY BEV FEATURE GENERATOR**
    - Purpose: Extract BEV features using pre-trained BEVFormer model
    - Uses `bevformer_integration_6cam.py` from semantic-map
    - Model: Loads UniAD checkpoint (`ckpts/uniad_base_track_map.pth`)
    - Output: `bev_features/*.pt` (123 frames at ~2Hz)
    - Imports: `timestamp_extract`, `bevformer_integration_6cam`, mmdet3d model builders


**Data Pipeline Flow:**
1. **ROS Bags** → `hct_train_bag_to_jpg.py` → **Images**
2. **ROS Bags** → `timestamp_extract.py` → **Master Timestamps**
3. **ROS Bags** + **Timestamps** → `canbus_extraction.py` → **CAN Bus Data**
4. **Images** + **CAN Bus** → `itri_bevformer_extractor.py` → **BEV Features**
5. **ROS Bags** + **Timestamps** → `track_query_extract.py` → **Track Queries**
6. **CAN Bus** + **Timestamps** → `gt_sdc_extraction.py` → **GT SDC Trajectories**
7. **ROS Bags** + **Timestamps** → `gt_fut_traj_extract.py` → **GT Future Trajectories**
8. **Semantic Map** + **Timestamps** → `gt_lane_extraction.py` → **GT Lane Annotations**

###