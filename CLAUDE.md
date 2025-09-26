# UniAD Integration Project

## Project Overview
Autonomous driving project integrating UniAD into existing ITRI pipeline with 4-camera setup and semantic map data.
Currently preparing to train the Uniad prediction + planning model, including MotionFormer + Occuformer + Planner. 
chekout train/training_prep/end2end_training_plan.md for more details

## UniAD Architecture Components
- **BEVFormer**: Multi-camera to Bird's Eye View features + CAN bus
- **OccFormer**: Occupancy prediction using BEV features  
- **MotionFormer**: Motion prediction with lane context and track queries
- **Planner**: Path planning module


