#!/usr/bin/env python3
import pickle
import numpy as np
import torch

# Load the pickle file
with open('output/l2g_results.pkl', 'rb') as f:
    data = pickle.load(f)

print('='*80)
print('L2G RESULTS DATA CONTENTS')
print('='*80)

print('\n=== BBOX RESULTS (showing first 3 entries) ===')
for i in range(min(3, len(data['bbox_results']))):
    entry = data['bbox_results'][i]
    print(f'\nEntry {i} (token: {entry["token"]}):')
    print(f'  Command: {entry["command"]}')
    print(f'  SDC boxes shape: {entry["sdc_boxes_3d"].tensor.shape}')
    print(f'  SDC scores: {entry["sdc_scores_3d"]}')
    print(f'  Detection boxes shape: {entry["boxes_3d_det"].tensor.shape}')
    print(f'  Trajectory shape: {entry["traj"].shape}')
    print(f'  Planning trajectory shape: {entry["planning_traj"].shape}')
    print(f'  Planning trajectory values:\n{entry["planning_traj"]}')

print('\n=== OCCUPANCY RESULTS ===')
for key, value in data['occ_results_computed'].items():
    print(f'{key}: {value}')

print('\n=== PLANNING RESULTS ===')
for key, value in data['planning_results_computed'].items():
    print(f'{key}: {value}')