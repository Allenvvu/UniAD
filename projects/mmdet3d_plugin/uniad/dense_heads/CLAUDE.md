### File Introductions
  1. motion_head.py:
    - Handles motion prediction for vehicles
    - Predicts future trajectories of objects
    - Includes trajectory classification and regression
  2. occ_head.py:
    - Occupancy prediction head
    - Predicts 3D occupancy grid for scene understanding
    - Helps in understanding scene layout and free/occupied spaces
  3. panseg_head.py:
    - Panoptic segmentation head
    - Combines semantic and instance segmentation
    - Identifies and classifies objects in the scene
  4. planning_head.py:
    - Generates autonomous driving trajectories
    - Includes collision optimization
    - Responsible for path planning for the self-driving vehicle
  5. track_head.py:
    - Object tracking head
    - Tracks objects across different frames
    - Maintains object identities and their temporal information
