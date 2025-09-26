# Key Files and Components Required

## 1. Core Model Files

- **mmdet3d_plugin/uniad/detectors/uniad_e2e.py**  
  Main UniAD model implementation  

- **mmdet3d_plugin/uniad/dense_heads/motion_head.py**  
  Motion prediction head  

- **mmdet3d_plugin/uniad/dense_heads/occ_head.py**  
  Occupancy prediction head  

- **mmdet3d_plugin/uniad/dense_heads/planning_head.py**  
  Path planning head  

- **configs/stage2_e2e/base_e2e.py**  
  Base configuration template  

---

## 2. Data Pipeline Files

- **mmdet3d_plugin/datasets/nuscenes_e2e_dataset.py**  
  Dataset implementation (needs ITRI adaptation)  

- **mmdet3d_plugin/datasets/pipelines/loading.py**  
  Data loading pipelines  

- **mmdet3d_plugin/datasets/pipelines/occflow_label.py**  
  Occupancy label generation  

- **mmdet3d_plugin/uniad/apis/train.py**  
  Training API  

---

## 3. Loss Functions

- **mmdet3d_plugin/losses/traj_loss.py**  
  Motion trajectory loss  

- **mmdet3d_plugin/losses/planning_loss.py**  
  Planning trajectory loss  

- **mmdet3d_plugin/losses/occflow_loss.py**  
  Occupancy flow loss  
