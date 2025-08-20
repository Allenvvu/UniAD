#!/usr/bin/env python3

import pickle
import sys
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

def convert_results_format():
    """
    Convert results_10.pkl to format expected by visualizer
    """
    
    # Load results_10.pkl
    with open('output/results_10.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print(f"Original format: {type(results)}, length: {len(results)}")
    
    # Convert to expected format
    bbox_results = []
    for result in results:
        # Convert to expected format matching original results.pkl structure
        bbox_result = {
            'token': result['token'],
            'track_bbox_results': result.get('track_bbox_results', []),
            'boxes_3d': result['boxes_3d'],
            'scores_3d': result['scores_3d'], 
            'labels_3d': result['labels_3d'],
            'track_scores': result.get('track_scores', result['scores_3d']),
            'track_ids': result.get('track_ids', None),
            'sdc_boxes_3d': result.get('sdc_boxes_3d', None),
            'sdc_scores_3d': result.get('sdc_scores_3d', None),
            'sdc_track_scores': result.get('sdc_track_scores', None),
            'sdc_track_bbox_results': result.get('sdc_track_bbox_results', []),
            'boxes_3d_det': result.get('boxes_3d_det', result['boxes_3d']),
            'scores_3d_det': result.get('scores_3d_det', result['scores_3d']),
            'labels_3d_det': result.get('labels_3d_det', result['labels_3d']),
            'traj_0': result.get('traj_0', None),
            'traj_scores_0': result.get('traj_scores_0', None),
            'traj_1': result.get('traj_1', None),
            'traj_scores_1': result.get('traj_scores_1', None),
            'traj': result.get('traj', None),
            'traj_scores': result.get('traj_scores', None),
            'ret_iou': result.get('ret_iou', {}),
            'planning_traj': result['planning']['result_planning']['sdc_traj'] if 'planning' in result and 'result_planning' in result['planning'] else None,
            'planning_traj_gt': result['planning']['planning_gt']['sdc_planning'][0] if 'planning' in result and 'planning_gt' in result['planning'] and result['planning']['planning_gt']['sdc_planning'] else None,
            'command': result['planning']['planning_gt']['command'][0] if 'planning' in result and 'planning_gt' in result['planning'] and result['planning']['planning_gt']['command'] else None,
        }
        bbox_results.append(bbox_result)
    
    # Create the expected format matching results.pkl structure
    converted_results = {
        'bbox_results': bbox_results,
        'occ_results_computed': True,
        'planning_results_computed': True
    }
    
    # Save converted results
    with open('output/results_10_converted.pkl', 'wb') as f:
        pickle.dump(converted_results, f)
    
    print(f"Converted to format with {len(bbox_results)} bbox_results")
    print("Saved to output/results_10_converted.pkl")

if __name__ == '__main__':
    convert_results_format()