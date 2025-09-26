# ROS Bag Data Extraction Plan for MotionFormer

## Overview
This document outlines the targeted data extraction strategy for converting ITRI tracking data to UniAD MotionFormer input format, focusing only on essential topics and data fields.

## Todo
Phase 1 completed


## Available ITRI Data Sources

### Primary Topics (from ITRI_TRACKING_DATA_REFERENCE.md)
- **`/detected_objects`** - Main fused tracking results (519 messages per bag) ✅ **PRIMARY TARGET**
- `/detected_objects_prediction` - Motion prediction outputs (519 messages)
- `/front_detection_objects` - Front sensor detections (519 messages) 
- `/left_detection_objects` - Left sensor detections (519 messages)
- `/right_detection_objects` - Right sensor detections (519 messages) 
- `/top_detection_objects` - Top sensor detections (519 messages)
- `/center_track/.../vision_objects` - CenterTrack vision tracking (415 messages)

### Data Frequency & Coverage
- **Frequency**: ~10Hz (519 messages in 51.9 seconds)
- **Files**: 10 ROS bag files (2025-08-06-14-*.bag)
- **Objects**: 57 tracked objects with persistent IDs
- **Coverage**: 360-degree multi-sensor fusion

## MotionFormer Requirements Analysis

### Core Data Structures Needed
1. **`track_query_embeddings`**: `[1, 1, N_tracks, 256]` - Learned 256D features
2. **`track_bbox_results`**: List format `[[bboxes, scores, labels, bbox_index, mask]]`
   - `bboxes`: LiDARInstance3DBoxes `[x, y, z, w, l, h, yaw, vx, vy]` 
   - `scores`: Detection confidence scores
   - `labels`: Object class IDs (0-9 for UniAD)
   - `bbox_index`: Sequential indices
   - `mask`: Boolean validity mask
3. **`track_query_matched_idxes`**: `[N_tracks]` - Track ID mapping
4. **`sdc_embedding`**: `[256]` - Ego vehicle embedding
5. **`sdc_track_bbox_results`**: Same format as above for ego vehicle

### Critical vs Optional Fields

**CRITICAL (Motion prediction fails without):**
- `track_query_embeddings` - Core learned features
- `track_bbox_results[0][0]` (bboxes) - Required for coordinate transforms
- `track_bbox_results[0][2]` (labels) - Required for class-specific motion

**IMPORTANT (Used in training/validation):**
- `track_query_matched_idxes` - GT matching for loss computation
- `track_bbox_results[0][1]` (scores) - Confidence weighting

**OPTIONAL (Can be synthesized):**
- `track_bbox_results[0][3]` (bbox_index) - Sequential indices
- `track_bbox_results[0][4]` (mask) - Can be all True

## Targeted Extraction Strategy

### Extract ONLY `/detected_objects` Topic
**Rationale**: 
- Contains complete fused tracking results with persistent IDs
- All essential fields available: pose, velocity, dimensions, tracking duration
- Higher quality than individual sensor topics (already fused)
- Sufficient data for MotionFormer conversion

### Skip These Topics (Redundant/Lower Quality):
- Individual sensor topics (`/front_detection_objects`, etc.) - Already fused
- Raw vision tracking - Lower quality than main fused results
- Prediction topics - Not needed for input conversion

### Key ITRI Fields to Extract

From `/detected_objects` messages:
```python
essential_fields = {
    'id': int,                    # Persistent tracking ID
    'label': str,                 # Object classification  
    'score': float,               # Detection confidence
    'trackedPeriod': float,       # Tracking duration (seconds)
    'pose': {
        'position': {'x': float, 'y': float, 'z': float},
        'orientation': {'x': float, 'y': float, 'z': float, 'w': float}
    },
    'dimensions': {'x': float, 'y': float, 'z': float},
    'velocity': {
        'linear': {'x': float, 'y': float, 'z': float},
        'angular': {'z': float}
    },
    'variance': {'x': float, 'y': float, 'z': float}  # Position uncertainty
}
```

## Implementation Plan

### Phase 1: ROS Bag Reader (Week 1)
1. **Create `ros_bag_reader.py`**
   - Target only `/detected_objects` topic
   - Parse complete object structure
   - Handle 10 sequential bag files
   - Build temporal sequences using timestamps

2. **Data Structure Extraction**
   ```python
   extracted_data = {
       'timestamp': rospy.Time,
       'objects': [
           {
               'id': 46,
               'label': 'motorbike', 
               'score': 0.0,
               'tracked_period': 14.442,
               'position': (47.05, 5.57, -0.96),
               'orientation_quat': (0.0, 0.0, -0.76, 0.65),
               'dimensions': (0.49, 2.37, 1.83),
               'velocity': (-0.001, -0.006, 0.006),
               'angular_velocity': -3.8e-08
           }
       ]
   }
   ```

3. **Temporal Sequence Building**
   - Group objects by persistent ID across frames
   - Build multi-frame trajectories for motion prediction
   - Handle object appearance/disappearance

### Phase 2: Coordinate Transformation (Week 1-2)
1. **Use existing utilities**:
   - `find_yaw.py` for quaternion→yaw conversion
   - `coordinate_alignment.py` framework
   - BEV coordinate mapping `[-51.2, 51.2]` range

2. **Transform to UniAD format**:
   ```python
   # base_link → BEV coordinates
   bev_x = (x - pc_range[0]) / (pc_range[3] - pc_range[0]) * bev_size[1]
   bev_y = (y - pc_range[1]) / (pc_range[4] - pc_range[1]) * bev_size[0]
   
   # Quaternion → yaw angle
   yaw = tf.transformations.euler_from_quaternion(qx, qy, qz, qw)[2]
   ```

### Phase 3: Track Query Format Generation (Week 2)
1. **Use reference implementation**: `semantic-map/motionformer_integration.py:186-201`
2. **Generate required formats**:
   - LiDARInstance3DBoxes for bboxes
   - Confidence scoring using tracking duration proxy
   - Class mapping ITRI→UniAD labels

## File Organization

### Implementation Files:
- `track/ros_bag_reader.py` - Main extraction implementation
- `track/coordinate_converter.py` - Coordinate system transformations  
- `track/track_query_builder.py` - MotionFormer format generation
- `track/progress_tracker.md` - Implementation status

### Output Data:
- `track/extracted_data/` - Processed ITRI tracking sequences
- `track/converted_data/` - UniAD-compatible track queries
- `track/validation/` - Format validation and testing

This focused extraction strategy ensures we collect only the essential data needed for MotionFormer while avoiding redundant processing of lower-quality sensor-specific topics.


