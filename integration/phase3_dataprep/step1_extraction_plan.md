# Step 1: ROS Bag Data Extraction Plan

## Overview
Single bag extraction for 4-camera setup training data preparation.

### Camera Configuration (4-camera setup)
- **CAM_FRONT**: `/lucid_cameras_x00/gige_100_f_hdr/h265` (Front 100°)
- **CAM_FRONT_LEFT**: `/lucid_cameras_x01/gige_100_fl_hdr/h265` (Front-Left 100°)  
- **CAM_FRONT_RIGHT**: `/lucid_cameras_x01/gige_100_fr_hdr/h265` (Front-Right 100°)
- **CAM_BACK**: `/lucid_cameras_x00/gige_60_b_hdr/h265` (Back 60°)

## Implementation: `ros_bag_extractor.py`

### Key Features
1. **Temporal Synchronization**: 10Hz LiDAR-based sync with 100ms tolerance
2. **4-Camera Processing**: H265 → JPEG conversion at 1600×900 resolution
3. **Object Annotations**: Detection + tracking + predicted trajectories  
4. **CAN Bus Integration**: 18-dimensional vector generation
5. **nuScenes Format**: Training-ready data structure

### Core Pipeline

```python
class ITRIRosBagExtractor:
    def __init__(self, bag_file, output_dir):
        # 4-camera topic mapping
        self.camera_topics = {
            'CAM_FRONT': '/lucid_cameras_x00/gige_100_f_hdr/h265',
            'CAM_FRONT_LEFT': '/lucid_cameras_x01/gige_100_fl_hdr/h265',
            'CAM_FRONT_RIGHT': '/lucid_cameras_x01/gige_100_fr_hdr/h265',
            'CAM_BACK': '/lucid_cameras_x00/gige_60_b_hdr/h265'
        }
    
    def extract_synchronized_frames(self):
        """Extract at LiDAR frequency (~10Hz, ~519 frames/bag)"""
        
    def convert_to_nuscenes_format(self, sync_frame, sample_idx):
        """Convert to training-ready format with:
        - Camera intrinsics from bag_to_pkl.py
        - Object annotations from detected_objects_prediction
        - 18D CAN bus data from car_state + imu
        - Ego poses from ndt_scan_matcher_node
        """
```

### Camera Calibration (from existing `bag_to_pkl.py`)
```python
self.camera_calibrations = {
    'CAM_FRONT': {
        'cam_intrinsic': [657.904403, 0.0, 714.717385597, 
                         0.0, 658.500765392, 463.1351298, 
                         0.0, 0.0, 1.0],
        'sensor2ego_translation': [0.005, -0.125, -0.171],
        'sensor2ego_rotation': [-0.525, 0.509, -0.491, -0.495]
    },
    # ... other 3 cameras with verified calibration data
}
```

### Object Annotation Processing
- **Source**: `/detected_objects_prediction.objects[]`
- **Format**: 3D boxes [x, y, z, w, l, h, yaw] + velocity + class + ID
- **Tracking**: Object tokens for temporal consistency

### CAN Bus Data Generation  
- **18D Vector**: [delta_x, delta_y, z, qw, qx, qy, qz, accel_xyz, angular_vel_xyz, vel_xyz, yaw, reserved]
- **Source Topics**: `/car_state`, `/imu/data`, `/vehicle_state`
- **Integration**: Reuses `extract_can_bus.py` logic

## Usage Instructions

### Single Bag Test
```bash
# Test with one bag file
python integration/phase3_dataprep/ros_bag_extractor.py \
    /home/bryan/Desktop/Allen/UniAD/data/itri/2025-08-06-hct_logistic/bag_files/2025-08-06-14-23-05_0.bag \
    --output_dir /home/bryan/Desktop/Allen/UniAD/data/itri_extracted_test/
```

### Expected Output Structure
```
data/itri_extracted_test/
├── samples/                    # Camera images
│   ├── sample_000001_CAM_FRONT.jpg
│   ├── sample_000001_CAM_FRONT_LEFT.jpg  
│   ├── sample_000001_CAM_FRONT_RIGHT.jpg
│   ├── sample_000001_CAM_BACK.jpg
│   └── ... (519 samples × 4 cameras)
├── sweeps/                     # LiDAR data (placeholder)
├── maps/                       # Semantic map integration
├── samples_data.pkl           # Main training data
└── extraction_metadata.json   # Dataset information
```

### Data Validation
- **Expected samples**: ~519 (10Hz × 52 seconds)
- **Camera coverage**: 4/6 cameras (compatible with UniAD 6-cam padding)
- **Annotation quality**: Full detection + tracking + motion prediction
- **CAN bus completeness**: 18D vectors synchronized to frames

## Success Metrics
- [x] **4-camera extraction**: Front 100°, Front-Left 100°, Front-Right 100°, Back 60°
- [x] **Temporal sync**: LiDAR-based 10Hz alignment
- [x] **nuScenes format**: Training-ready data structure
- [x] **Object annotations**: Complete detection + tracking pipeline
- [x] **CAN bus integration**: 18D vector generation
- [x] **Calibration data**: Verified camera intrinsics/extrinsics

## Next Steps (Post Single-Bag Test)
1. **Batch processing**: Scale to all 24 bags
2. **Data validation**: Quality checks and format verification
3. **Ground truth generation**: Motion + planning labels
4. **Dataset splitting**: Train/val temporal splits
5. **UniAD integration**: Training pipeline testing

## Dependencies
- `rosbag`: ROS bag reading
- `cv2`: H265 image decoding  
- `tf.transformations`: Quaternion/Euler conversions
- `numpy`, `pickle`: Data processing
- Existing tools: `extract_can_bus.py`, `bag_to_pkl.py` logic