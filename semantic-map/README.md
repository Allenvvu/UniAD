# ITRI Semantic Map to UniAD Integration

This package converts ITRI semantic map polylines to the lane query format required by UniAD's MotionFormer, enabling trajectory prediction without retraining MapFormer.

## Overview

The package provides a complete pipeline to:
1. Load ITRI semantic map data (roadlines, crossings, markers)
2. Transform polylines to ego vehicle coordinates  
3. Generate lane feature queries and positional encodings
4. Integrate seamlessly with MotionFormer for trajectory prediction

## Key Features

- **Direct Conversion**: Convert ITRI polylines directly to lane queries
- **No Retraining**: Use existing MotionFormer weights without MapFormer
- **Rich Features**: Geometric and semantic features from polyline data
- **Efficient Caching**: Cache map conversions for real-time performance
- **Flexible Integration**: Easy integration with existing tracking pipelines

## Installation

The package requires PyTorch and standard scientific Python libraries:

```bash
pip install torch numpy scipy
```

## Quick Start

### Basic Conversion

```python
from semantic_map import create_converter

# Configure converter
config = {
    'embed_dim': 256,
    'max_queries': 300, 
    'sample_distance': 1.0,
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
}

# Create converter
converter = create_converter(config)

# Convert ITRI data to lane queries
ego_pose = (100.0, 50.0, 0.5)  # x, y, yaw
lane_query, lane_query_pos = converter.convert('./itri', ego_pose)

print(f"Lane queries shape: {lane_query.shape}")  # [1, 300, 256]
```

### Full MotionFormer Integration

```python
from semantic_map import create_integrator

# Create integrator
integrator = create_integrator(device='cuda')

# Your tracking data
tracked_objects = [
    {
        'bbox': [10.0, 5.0, 0.0, 4.5, 2.0, 1.8, 0.1],
        'score': 0.95,
        'label': 0,
        'track_id': 1
    }
]

# Prepare inputs for MotionFormer
motion_inputs = integrator.prepare_motion_inputs(
    itri_data_path='./itri',
    ego_pose=(100.0, 50.0, 0.5),
    tracked_objects=tracked_objects,
    bev_features=your_bev_features
)

# Run MotionFormer
traj_results, motion_outputs = integrator.run_motion_prediction(
    motion_head=your_motion_head,
    itri_data_path='./itri',
    ego_pose=(100.0, 50.0, 0.5),
    tracked_objects=tracked_objects,
    bev_features=your_bev_features
)
```

## Data Format

### ITRI Input Format

Place your ITRI JSON files in the `semantic-map/itri/` folder:

```
semantic-map/itri/
├── roadlines.json      # Primary lane boundaries
├── pedestrian_crossing.json  # Crosswalk data
├── roadmarkers.json    # Additional lane markers
├── lanes_info.json     # Lane connectivity (optional)
└── ...
```

### Required JSON Structure

**roadlines.json:**
```json
{
  "roadlines": [
    {
      "id": 1,
      "points": [
        {
          "point_id": 1,
          "type": 7,
          "x": 10.587,
          "y": -19.489, 
          "z": -9.013
        }
      ]
    }
  ]
}
```

**pedestrian_crossing.json:**
```json
{
  "non_accessible": [
    {
      "id": 3,
      "points": [
        {"x": -9.081, "y": -14.675, "z": -8.735}
      ]
    }
  ]
}
```

### Output Format

The converter generates lane queries compatible with MotionFormer:

- `lane_query`: `[1, max_queries, embed_dim]` - Lane feature embeddings
- `lane_query_pos`: `[1, max_queries, embed_dim]` - Positional encodings

## Architecture

### Core Components

1. **ITRIToLaneQueryConverter**: Main conversion class
   - Loads ITRI JSON data
   - Transforms coordinates to ego frame
   - Generates feature embeddings
   - Creates positional encodings

2. **MotionFormerIntegrator**: Integration interface
   - Prepares MotionFormer inputs
   - Handles tracking data conversion
   - Manages caching for performance
   - Provides unified inference interface

3. **Geometric Utilities**: Polyline processing
   - Uniform sampling along polylines
   - Direction and curvature computation
   - Coordinate transformations

### Data Flow

```
ITRI JSON Files → Polyline Loading → Coordinate Transform → 
Feature Extraction → Lane Queries → MotionFormer → Trajectories
```

## Configuration

### Converter Configuration

```python
config = {
    'embed_dim': 256,        # Feature embedding dimension
    'max_queries': 300,      # Maximum number of lane queries
    'sample_distance': 1.0,  # Polyline sampling distance (meters)
    'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]  # BEV range
}
```

### Lane Type Mapping

ITRI line types are mapped to UniAD classes:

```python
type_mapping = {
    7: 0,   # Divider lines → divider class
    # Add custom mappings based on your ITRI data
}
```

Classes:
- **0**: Divider (lane boundaries, road dividers)
- **1**: Crossing (pedestrian crossings)  
- **2**: Contour (road boundaries, markers)

## Performance Optimization

### Caching

Enable caching for repeated map queries:

```python
# Cache map conversions for the same location
motion_inputs = integrator.prepare_motion_inputs(
    itri_data_path='./itri',
    ego_pose=ego_pose,
    tracked_objects=tracked_objects,
    bev_features=bev_features,
    use_cache=True  # Enable caching
)

# Clear cache when needed
integrator.clear_cache()
```

### Batch Processing

Process multiple frames efficiently:

```python
trajectory = [(100.0, 50.0, 0.5), (102.0, 51.0, 0.52), ...]

for ego_pose in trajectory:
    motion_inputs = integrator.prepare_motion_inputs(
        itri_data_path='./itri',
        ego_pose=ego_pose,
        tracked_objects=tracked_objects,
        bev_features=bev_features,
        use_cache=True  # Reuse map data
    )
```

## Examples

Run the provided examples:

```bash
# All examples
python example_usage.py --example all

# Specific examples
python example_usage.py --example basic
python example_usage.py --example integration
python example_usage.py --example batch
python example_usage.py --example performance
```

## Integration with UniAD

### Replace MapFormer Head

Instead of using MapFormer, directly inject lane queries:

```python
# In your UniAD inference pipeline
from semantic_map import create_integrator

integrator = create_integrator()

# Replace this:
# outs_seg = map_head.forward_test(bev_features, ...)

# With this:
motion_inputs = integrator.prepare_motion_inputs(
    itri_data_path='./itri',
    ego_pose=current_ego_pose,
    tracked_objects=current_detections,
    bev_features=bev_features
)

# Continue with MotionFormer
traj_results, outs_motion = motion_head.forward_test(
    bev_embed=motion_inputs['bev_embed'],
    outs_track=motion_inputs['outs_track'],
    outs_seg=motion_inputs['outs_seg']  # Contains lane queries from ITRI
)
```

### Complete Pipeline

```python
# Your existing pipeline
camera_images = get_camera_images()
bev_features = bevformer_encoder(camera_images)
tracked_objects = detection_tracker(bev_features)

# Add ITRI integration
current_ego_pose = get_current_ego_pose()
motion_inputs = integrator.prepare_motion_inputs(
    itri_data_path='./itri',
    ego_pose=current_ego_pose,
    tracked_objects=tracked_objects,
    bev_features=bev_features
)

# Motion prediction
trajectories, motion_outputs = motion_head.forward_test(
    bev_embed=motion_inputs['bev_embed'],
    outs_track=motion_inputs['outs_track'],
    outs_seg=motion_inputs['outs_seg']
)

# Continue with occupancy and planning
occ_results = occ_head.forward_test(...)
plan_results = planning_head.forward_test(...)
```

## Troubleshooting

### Common Issues

1. **Missing ITRI files**: Ensure JSON files are in `semantic-map/itri/`
2. **Empty lane queries**: Check ego pose is within map bounds
3. **Memory issues**: Reduce `max_queries` or use smaller `embed_dim`
4. **Performance issues**: Enable caching and reduce `sample_distance`

### Debugging

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Check conversion results
lane_query, lane_query_pos = converter.convert(itri_path, ego_pose)
non_zero_queries = (lane_query.norm(dim=-1) > 0).sum()
print(f"Generated {non_zero_queries} non-zero lane queries")
```

### Validation

Use the built-in validation:

```python
from semantic_map.motionformer_integration import validate_inputs

is_valid = validate_inputs(itri_data_path, ego_pose, tracked_objects)
if not is_valid:
    print("Fix input data before proceeding")
```

## Contributing

To extend the package:

1. **Add new ITRI data types**: Modify `_load_itri_data()` in `itri_to_lane_query.py`
2. **Custom feature extraction**: Extend `_create_lane_embedding()` method
3. **New coordinate systems**: Add functions to `coordinate_transform.py`
4. **Integration improvements**: Enhance `motionformer_integration.py`

## License

This package is generated for the UniAD integration project. Please follow your project's licensing requirements.