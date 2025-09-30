### Inference running function
```bash
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=29500 \
      tools/test.py l2g_inference_config.py ckpts/uniad_base_e2e.pth \
      --out output/l2g_results.pkl --launcher pytorch
```

## Configuration Files

- `l2g_inference_config.py`
  - Main configuration file for L2G dataset inference
  - Passed as argument to `tools/test.py` during inference execution
- `simple_inference_config.py`
  - Simplified alternative configuration for quick testing
  - Not used in the current inference pipeline
  - Omits full dataset specifications

---

## Utility Scripts

- `utility/convert_l2g_to_uniad.py`
  - Convert L2G camera data format to UniAD-compatible nuScenes format
    - `compute_lidar_to_global_transform()` - Calculates coordinate transformations
    - `create_uniad_sample_info()` - Structures data in UniAD-expected format
    - `create_temporal_sequence()` - Creates temporal sequences with `queue_length=3`
  - Input: L2G camera config, CAN bus data, timestamps
  - Output: Pickle file with UniAD-compatible sample information
- `utility/combine_l2g_data.py`
  - Combine multiple L2G scene files into single annotation file
  - Run after converting individual scenes to combine them for inference
- `utility/add_missing_cameras.py`
  - Creates dummy camera entries with zero intrinsics
  - Uses identity transformations and `/dev/null` paths

---

## Path & Data Fixing Utilities

- `utility/fix_dataset_metadata.py`
  - Add required metadata structure to the L2G dataset
- `utility/fix_infos_pickle_paths.py`
  - Fix image paths in pickle files under `data/infos/`


---

## projects/ Folder

- `projects/mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py`
  - **Lines 347-373**: Added handling for list-based data structures (L2G converted data format)
  - **Lines 402-407**: Added velocity handling for L2G format
  - **Lines 501-514**: Added empty tensor handling for edge cases with no valid masks
  - **Lines 558-565**: Added identity transformation when no lidar data is available
  - **Lines 713-720**: Added list-based data structure handling in `occ_get_detection_ann_info`
  - **Lines 746-756**: Added velocity handling for L2G format in occupancy detection

### 2. **`projects/mmdet3d_plugin/uniad/dense_heads/panseg_head.py`**
   - Contains modifications for handling cases


---

## tools/ Folder

- `tools/autofix_missing_lines.py`
  - Custom script to fix missing line references in traffic_light records for nuScenes expansion maps
  - Finds nearest nodes to create line records when traffic_light line_token is missing

- `tools/validate_map_json.py`
  - Validator for nuScenes expansion map JSON cross-layer token references
  - Checks that all tokens referenced by non-geometric layers exist in their geometric layers
  - Validates polygon, line, node, lane, stop_line, and traffic_light references

- `tools/semantic_map_to_nuscenes_expansion.py`
  - Converts HCT/semantic_map JSON files to nuScenes expansion-style map JSON
  - Includes affine transformation to align semantic_map coordinates into target 2D map frame
  - Handles geometry tables (polygon, line, node) and semantic layers (drivable_area, ped_crossing, etc.)


## data/ Folder Structure

```
data/
├── infos/                              # Processed annotation files
│   └── nuscenes_infos_temporal_val.pkl # Combined temporal annotations for inference
├── l2g/                                # L2G converted scene data
│   ├── 2025-08-06-*_*_l2g_converted_infos.pkl  # Individual scene pickle files
│   └── backup/                         # Backup of original converted files
├── nuscenes/                           # NuScenes-compatible data structure
│   ├── image/                          # Camera image data
│   │   ├── lucid_cameras_x00.gige_100_f_hdr.h265/   # Front camera
│   │   ├── lucid_cameras_x00.gige_60_b_hdr.h265/    # Back camera
│   │   ├── lucid_cameras_x01.gige_100_fl_hdr.h265/  # Front-left camera
│   │   └── lucid_cameras_x01.gige_100_fr_hdr.h265/  # Front-right camera
│   ├── maps/                           # Map data
│   │   ├── basemap/                    # PNG map images
│   │   ├── expansion/                  # JSON expansion map layers
│   │   └── prediction/                 # Prediction scene definitions
│   ├── semantic_map/                   # ITRI semantic map data (HCT format)
│   ├── itri_map/                       # Converted ITRI expansion maps
│   └── v1.0-trainval/                  # NuScenes metadata tables
│       ├── sample.json                 # Sample records
│       ├── sample_data.json            # Sample data records
│       ├── ego_pose.json               # Ego vehicle pose
│       ├── calibrated_sensor.json      # Camera calibrations
│       ├── instance.json               # Object instances
│       ├── sample_annotation.json      # Annotations
│       ├── category.json               # Object categories
│       ├── attribute.json              # Object attributes
│       ├── log.json                    # Log metadata
│       └── map.json                    # Map metadata
└── others/                             # Auxiliary data
    └── motion_anchor_infos_mode6.pkl   # Motion prediction anchor information
```

### Data Storage Logic

1. **L2G Scene Data** (`data/l2g/`)
   - Individual scene files converted from L2G format
   - Each file contains temporal sequences for one driving scene
   - Naming: `{timestamp}_{scene_id}_l2g_converted_infos.pkl`
   - Backup copies preserved for safety

2. **Combined Annotations** (`data/infos/`)
   - `nuscenes_infos_temporal_val.pkl`: Merged data from all L2G scenes
   - Used directly by UniAD inference pipeline
   - Contains temporal sequences with queue_length=3

3. **Image Storage** (`data/nuscenes/image/`)
   - Organized by camera sensor name
   - Images named by nanosecond timestamps
   - Format: `{timestamp_ns}.jpg`
   - Supports 4 cameras (front, back, front-left, front-right)

4. **NuScenes Metadata** (`data/nuscenes/v1.0-trainval/`)
   - Standard nuScenes JSON tables
   - Links samples, sensors, poses, and annotations
   - Required for dataset compatibility

5. **Map Data** (`data/nuscenes/maps/`)
   - **basemap/**: Static PNG images for visualization
   - **expansion/**: Vector map layers (lanes, dividers, crossings)
   - **semantic_map/**: Original ITRI HCT format
   - **itri_map/**: Converted expansion maps for ITRI scenes

6. **Motion Anchors** (`data/others/`)
   - Pre-computed motion prediction anchors
   - Used by motion head for trajectory prediction
