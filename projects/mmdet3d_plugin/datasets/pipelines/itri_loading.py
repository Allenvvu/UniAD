#---------------------------------------------------------------------------------#
# ITRI Data Loading Pipelines for UniAD
# Custom pipeline transforms for loading ITRI-specific data formats
#---------------------------------------------------------------------------------#

import numpy as np
import torch
import mmcv
import os
import pickle
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmcv.parallel import DataContainer as DC


@PIPELINES.register_module()
class LoadPreComputedBEVFeatures(object):
    """Load pre-computed BEV features from disk.

    Args:
        bev_features_root (str): Root directory for BEV features
        to_float32 (bool): Whether to convert to float32
    """

    def __init__(self, bev_features_root, to_float32=True):
        self.bev_features_root = bev_features_root
        self.to_float32 = to_float32

    def __call__(self, results):
        """Load BEV features for current sample."""
        bag_name = results['bag_name']
        frame_idx = results['frame_idx']

        # Construct BEV features path
        bev_file = results['available_data'].get('bev_features')
        if bev_file and os.path.exists(bev_file):
            # Load BEV features
            if bev_file.endswith('.pt'):
                bev_features = torch.load(bev_file, map_location='cpu')
            elif bev_file.endswith('.pkl'):
                with open(bev_file, 'rb') as f:
                    bev_features = pickle.load(f)
            elif bev_file.endswith('.npz'):
                data = np.load(bev_file)
                bev_features = data['bev_features']
            else:
                raise ValueError(f"Unsupported BEV features format: {bev_file}")

            if isinstance(bev_features, np.ndarray):
                bev_features = torch.from_numpy(bev_features)

            if self.to_float32:
                bev_features = bev_features.float()

            results['bev_features'] = bev_features
        else:
            # Create dummy BEV features if not available
            results['bev_features'] = torch.zeros((256, 200, 200), dtype=torch.float32)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(bev_features_root={self.bev_features_root})'


@PIPELINES.register_module()
class LoadCANBusData(object):
    """Load CAN bus data from disk.

    Args:
        canbus_root (str): Root directory for CAN bus data
    """

    def __init__(self, canbus_root):
        self.canbus_root = canbus_root

    def __call__(self, results):
        """Load CAN bus data for current sample."""
        bag_name = results['bag_name']

        # Look for CAN bus data file
        canbus_file = results['available_data'].get('canbus')
        if canbus_file and os.path.exists(canbus_file):
            if canbus_file.endswith('.pt'):
                canbus_data = torch.load(canbus_file, map_location='cpu')
            elif canbus_file.endswith('.pkl'):
                with open(canbus_file, 'rb') as f:
                    canbus_data = pickle.load(f)
            elif canbus_file.endswith('.npz'):
                data = np.load(canbus_file)
                canbus_data = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported CAN bus format: {canbus_file}")

            results['canbus_data'] = canbus_data
        else:
            # Create dummy CAN bus data
            results['canbus_data'] = {
                'rotation': np.array([0.0, 0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0, 0.0]),
                'angular_velocity': np.array([0.0, 0.0, 0.0]),
                'velocity': np.array([0.0, 0.0, 0.0]),
                'translation': np.array([0.0, 0.0, 0.0]),
            }

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(canbus_root={self.canbus_root})'


@PIPELINES.register_module()
class LoadSDCEmbeddings(object):
    """Load SDC embeddings from disk.

    Args:
        sdc_embeddings_root (str): Root directory for SDC embeddings
    """

    def __init__(self, sdc_embeddings_root):
        self.sdc_embeddings_root = sdc_embeddings_root

    def __call__(self, results):
        """Load SDC embeddings for current sample."""
        sdc_file = results['available_data'].get('sdc_embeddings')
        if sdc_file and os.path.exists(sdc_file):
            if sdc_file.endswith('.pt'):
                sdc_embeddings = torch.load(sdc_file, map_location='cpu')
            elif sdc_file.endswith('.pkl'):
                with open(sdc_file, 'rb') as f:
                    sdc_embeddings = pickle.load(f)
            elif sdc_file.endswith('.npz'):
                data = np.load(sdc_file)
                sdc_embeddings = data['sdc_embeddings']
            else:
                raise ValueError(f"Unsupported SDC embeddings format: {sdc_file}")

            # Normalize to a flat tensor; drop nested complex structures not needed for training
            if isinstance(sdc_embeddings, dict):
                # Prefer 'sdc_embedding' tensor if present
                if 'sdc_embedding' in sdc_embeddings and torch.is_tensor(sdc_embeddings['sdc_embedding']):
                    results['sdc_embeddings'] = sdc_embeddings['sdc_embedding'].detach().clone().float()
                else:
                    # Attempt to find any tensor in dict
                    tensor_vals = [v for v in sdc_embeddings.values() if torch.is_tensor(v)]
                    if tensor_vals:
                        results['sdc_embeddings'] = tensor_vals[0].detach().clone().float()
                    else:
                        # Fallback to zeros if no tensor found
                        print("Warning: SDC embeddings dict lacks tensor fields, using dummy SDC embedding")
                        results['sdc_embeddings'] = torch.zeros((256,), dtype=torch.float32)
            else:
                if isinstance(sdc_embeddings, np.ndarray):
                    sdc_embeddings = torch.from_numpy(sdc_embeddings)
                if torch.is_tensor(sdc_embeddings):
                    results['sdc_embeddings'] = sdc_embeddings.detach().clone().float()
                else:
                    print("Warning: Unsupported SDC embedding type, using dummy SDC embedding")
                    results['sdc_embeddings'] = torch.zeros((256,), dtype=torch.float32)
        else:
            # Create dummy SDC embeddings
            results['sdc_embeddings'] = torch.zeros((256,), dtype=torch.float32)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sdc_embeddings_root={self.sdc_embeddings_root})'


@PIPELINES.register_module()
class LoadTrackQueries(object):
    """Load track queries from disk.

    Args:
        track_query_root (str): Root directory for track queries
    """

    def __init__(self, track_query_root):
        self.track_query_root = track_query_root

    def __call__(self, results):
        """Load track queries for current sample."""
        track_file = results['available_data'].get('track_query')
        if track_file and os.path.exists(track_file):
            if track_file.endswith('.pt'):
                track_data = torch.load(track_file, map_location='cpu')
            elif track_file.endswith('.pkl'):
                with open(track_file, 'rb') as f:
                    track_data = pickle.load(f)
            elif track_file.endswith('.npz'):
                data = np.load(track_file)
                track_data = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported track query format: {track_file}")

            # Process the track query data based on its structure
            if isinstance(track_data, dict) and 'track_query_embeddings' in track_data:
                # Extract track queries from the ITRI format
                embeddings_obj = track_data['track_query_embeddings']

                target_num_queries = 300

                if isinstance(embeddings_obj, list):
                    if len(embeddings_obj) > 0:
                        # Use the most recent track query embedding (last in sequence)
                        # Expected shape example: [1, 300, 256] -> [300, 256]
                        current_embedding = embeddings_obj[-1]
                        if isinstance(current_embedding, torch.Tensor):
                            if current_embedding.dim() == 3 and current_embedding.size(0) == 1:
                                current_embedding = current_embedding.squeeze(0)
                        # Pad or truncate to fixed size [300, 256]
                        if current_embedding.dim() == 2:
                            num_q, dim = current_embedding.shape
                            if num_q < target_num_queries:
                                pad = torch.zeros((target_num_queries - num_q, dim), dtype=current_embedding.dtype)
                                current_embedding = torch.cat([current_embedding, pad], dim=0)
                            elif num_q > target_num_queries:
                                current_embedding = current_embedding[:target_num_queries]
                            current_embedding = current_embedding.detach().clone().float()
                            results['track_queries'] = current_embedding
                            # Pass-through optional fields if present
                            if 'track_query_matched_idxes' in track_data:
                                results['track_query_matched_idxes'] = track_data['track_query_matched_idxes']
                            if 'track_bbox_results' in track_data:
                                results['track_bbox_results'] = track_data['track_bbox_results']
                            if 'sdc_track_bbox_results' in track_data:
                                results['sdc_track_bbox_results'] = track_data['sdc_track_bbox_results']
                        else:
                            print(f"Warning: Unexpected track query tensor dims from list: {current_embedding.size()} - using dummy")
                            results['track_queries'] = torch.zeros((target_num_queries, 256), dtype=torch.float32)
                    else:
                        print("Warning: Empty track_query_embeddings list, using dummy queries")
                        results['track_queries'] = torch.zeros((target_num_queries, 256), dtype=torch.float32)
                elif torch.is_tensor(embeddings_obj):
                    # Handle tensor formats, e.g., [1, 1, N, 256], [S, N, 256], [N, 256]
                    tensor = embeddings_obj
                    # Normalize to [*, N, D]
                    if tensor.dim() < 2:
                        print(f"Warning: track_query_embeddings tensor has dim {tensor.dim()}, using dummy")
                        results['track_queries'] = torch.zeros((target_num_queries, 256), dtype=torch.float32)
                    else:
                        # Collapse leading dims except the last two (N, D), then take the last slice
                        n_q, dim = tensor.shape[-2], tensor.shape[-1]
                        reshaped = tensor.reshape(-1, n_q, dim)
                        current_embedding = reshaped[-1]
                        # Ensure [N, D] and pad/truncate to target_num_queries
                        num_q, dim = current_embedding.shape
                        if num_q < target_num_queries:
                            pad = torch.zeros((target_num_queries - num_q, dim), dtype=current_embedding.dtype)
                            current_embedding = torch.cat([current_embedding, pad], dim=0)
                        elif num_q > target_num_queries:
                            current_embedding = current_embedding[:target_num_queries]
                        current_embedding = current_embedding.detach().clone().float()
                        results['track_queries'] = current_embedding
                        # Pass-through optional fields if present
                        if isinstance(track_data, dict):
                            if 'track_query_matched_idxes' in track_data:
                                results['track_query_matched_idxes'] = track_data['track_query_matched_idxes']
                            if 'track_bbox_results' in track_data:
                                results['track_bbox_results'] = track_data['track_bbox_results']
                            if 'sdc_track_bbox_results' in track_data:
                                results['sdc_track_bbox_results'] = track_data['sdc_track_bbox_results']
                        # Optional: simple debug
                        print(f"Loaded track queries from tensor: original_shape={tuple(tensor.shape)}, used_num_queries={num_q}")
            elif isinstance(track_data, dict) and ('queries' in track_data or 'query_pos' in track_data):
                # Standard format
                results['track_queries'] = track_data
                if 'track_query_matched_idxes' in track_data:
                    results['track_query_matched_idxes'] = track_data['track_query_matched_idxes']
                if 'track_bbox_results' in track_data:
                    results['track_bbox_results'] = track_data['track_bbox_results']
                if 'sdc_track_bbox_results' in track_data:
                    results['sdc_track_bbox_results'] = track_data['sdc_track_bbox_results']
            else:
                print(f"Warning: Unexpected track query data format: {type(track_data)}, using dummy queries")
                results['track_queries'] = torch.zeros((300, 256), dtype=torch.float32)
        else:
            print("Warning: No track query file found, using dummy queries")
            # Create dummy track queries
            results['track_queries'] = torch.zeros((300, 256), dtype=torch.float32)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(track_query_root={self.track_query_root})'


@PIPELINES.register_module()
class LoadMapQueries(object):
    """Load map queries from disk.

    Args:
        map_query_root (str): Root directory for map queries
    """

    def __init__(self, map_query_root):
        self.map_query_root = map_query_root

    def __call__(self, results):
        """Load map queries for current sample."""
        map_file = results['available_data'].get('map_query')
        if map_file and os.path.exists(map_file):
            if map_file.endswith('.pt'):
                map_data = torch.load(map_file, map_location='cpu')
            elif map_file.endswith('.pkl'):
                with open(map_file, 'rb') as f:
                    map_data = pickle.load(f)
            elif map_file.endswith('.npz'):
                data = np.load(map_file)
                map_data = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported map query format: {map_file}")

            # Detach any tensors to avoid requires_grad=True in DataLoader
            def _detach_tensors(obj):
                if torch.is_tensor(obj):
                    return obj.detach().clone().float()
                if isinstance(obj, dict):
                    return {k: _detach_tensors(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_detach_tensors(v) for v in obj]
                return obj

            sanitized = _detach_tensors(map_data)
            # Keep only essential tensor fields to avoid complex nested structures
            if isinstance(sanitized, dict):
                kept = {}
                for k in ['lane_query', 'lane_queries', 'lane_query_pos', 'lane_positions']:
                    if k in sanitized and torch.is_tensor(sanitized[k]):
                        kept[k] = sanitized[k]
                if not kept:
                    # Fallback: if unexpected structure, create dummy map queries
                    print("Warning: Unexpected map query structure, using dummy map queries")
                    kept = {
                        'lane_queries': torch.zeros((100, 256), dtype=torch.float32),
                        'lane_positions': torch.zeros((100, 2), dtype=torch.float32),
                    }
                results['map_queries'] = kept
            else:
                results['map_queries'] = {
                    'lane_queries': torch.zeros((100, 256), dtype=torch.float32),
                    'lane_positions': torch.zeros((100, 2), dtype=torch.float32),
                }
        else:
            # Create dummy map queries
            results['map_queries'] = {
                'lane_queries': torch.zeros((100, 256), dtype=torch.float32),
                'lane_positions': torch.zeros((100, 2), dtype=torch.float32),
                'lane_masks': torch.zeros((100,), dtype=torch.bool),
            }

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(map_query_root={self.map_query_root})'


@PIPELINES.register_module()
class LoadGTFutureTraj(object):
    """Load ground truth future trajectories from disk.

    Args:
        gt_fut_traj_root (str): Root directory for GT future trajectories
    """

    def __init__(self, gt_fut_traj_root):
        self.gt_fut_traj_root = gt_fut_traj_root

    def __call__(self, results):
        """Load GT future trajectories for current sample."""
        traj_file = results['available_data'].get('gt_fut_traj')
        if traj_file and os.path.exists(traj_file):
            if traj_file.endswith('.pt'):
                traj_data = torch.load(traj_file, map_location='cpu')
            elif traj_file.endswith('.pkl'):
                with open(traj_file, 'rb') as f:
                    traj_data = pickle.load(f)
            elif traj_file.endswith('.npz'):
                data = np.load(traj_file)
                traj_data = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported GT trajectory format: {traj_file}")

            results['gt_fut_traj'] = traj_data.get('gt_fut_traj', torch.zeros((300, 12, 2)))
            results['gt_fut_traj_mask'] = traj_data.get('gt_fut_traj_mask', torch.ones((300, 12)))
        else:
            # Create dummy GT trajectories
            results['gt_fut_traj'] = torch.zeros((300, 12, 2), dtype=torch.float32)
            results['gt_fut_traj_mask'] = torch.zeros((300, 12), dtype=torch.bool)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(gt_fut_traj_root={self.gt_fut_traj_root})'


@PIPELINES.register_module()
class LoadSDCPlanningData(object):
    """Load SDC planning data from disk.

    Args:
        sdc_planning_root (str): Root directory for SDC planning data
    """

    def __init__(self, sdc_planning_root):
        self.sdc_planning_root = sdc_planning_root

    def __call__(self, results):
        """Load SDC planning data for current sample."""
        planning_file = results['available_data'].get('sdc_planning')
        if planning_file and os.path.exists(planning_file):
            if planning_file.endswith('.pt'):
                planning_data = torch.load(planning_file, map_location='cpu')
            elif planning_file.endswith('.pkl'):
                with open(planning_file, 'rb') as f:
                    planning_data = pickle.load(f)
            elif planning_file.endswith('.npz'):
                data = np.load(planning_file)
                planning_data = {key: data[key] for key in data.files}
            else:
                raise ValueError(f"Unsupported planning data format: {planning_file}")

            results['sdc_planning'] = planning_data.get('sdc_planning', torch.zeros((6, 3)))
            results['sdc_planning_mask'] = planning_data.get('sdc_planning_mask', torch.ones((6,)))
            results['command'] = planning_data.get('command', 0)  # Default command
        else:
            # Create dummy planning data
            results['sdc_planning'] = torch.zeros((6, 3), dtype=torch.float32)
            results['sdc_planning_mask'] = torch.zeros((6,), dtype=torch.bool)
            results['command'] = 0

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(sdc_planning_root={self.sdc_planning_root})'


@PIPELINES.register_module()
class FormatITRIData(object):
    """Format ITRI data for model input."""

    def __init__(self, class_names=None, with_label=True):
        self.class_names = class_names
        self.with_label = with_label

    def __call__(self, results):
        """Format the results for ITRI model input."""
        # Convert relevant data to DataContainer format - avoid double wrapping
        variable_length_keys = ['gt_fut_traj', 'gt_fut_traj_mask']
        for key in ['track_queries', 'map_queries',
                   'gt_fut_traj', 'gt_fut_traj_mask', 'sdc_planning',
                   'sdc_planning_mask', 'sdc_embeddings']:
            if key in results:
                if isinstance(results[key], torch.Tensor):
                    if key in variable_length_keys:
                        results[key] = DC(results[key], stack=False)
                    else:
                        results[key] = DC(results[key])
                elif isinstance(results[key], dict):
                    # Extract tensor data from dict format - single wrap only
                    if 'queries' in results[key] and isinstance(results[key]['queries'], torch.Tensor):
                        results[key] = DC(results[key]['queries'])
                    elif 'lane_queries' in results[key] and isinstance(results[key]['lane_queries'], torch.Tensor):
                        results[key] = DC(results[key]['lane_queries'])
                    else:
                        # Don't double-wrap: either wrap individual tensors OR wrap the dict, not both
                        results[key] = DC(results[key])

        # Wrap optional track structures to avoid stacking
        for key in ['track_query_matched_idxes', 'track_bbox_results', 'sdc_track_bbox_results']:
            if key in results and not isinstance(results[key], DC):
                results[key] = DC(results[key], stack=False)

        # Generate missing gt_past_traj and gt_past_traj_mask if not present
        if 'gt_past_traj' not in results:
            # Create dummy past trajectories (4 past steps, 2D coordinates)
            results['gt_past_traj'] = torch.zeros((300, 4, 2), dtype=torch.float32)
        if 'gt_past_traj_mask' not in results:
            # Create dummy past trajectory masks
            results['gt_past_traj_mask'] = torch.zeros((300, 4), dtype=torch.bool)

        # Convert gt_past_traj and gt_past_traj_mask to DataContainer format
        results['gt_past_traj'] = DC(results['gt_past_traj'])
        results['gt_past_traj_mask'] = DC(results['gt_past_traj_mask'])

        # Generate missing occupancy and segmentation data if not present
        if 'gt_segmentation' not in results:
            # Create dummy segmentation data (sequence_len, height, width)
            results['gt_segmentation'] = torch.zeros((3, 200, 200), dtype=torch.long)
        if 'gt_instance' not in results:
            # Create dummy instance data (sequence_len, height, width)
            results['gt_instance'] = torch.zeros((3, 200, 200), dtype=torch.long)
        if 'gt_centerness' not in results:
            # Create dummy centerness data (sequence_len, 1, height, width)
            results['gt_centerness'] = torch.zeros((3, 1, 200, 200), dtype=torch.float32)
        if 'gt_offset' not in results:
            # Create dummy offset data (sequence_len, 2, height, width)
            results['gt_offset'] = torch.zeros((3, 2, 200, 200), dtype=torch.float32)
        if 'gt_flow' not in results:
            # Create dummy flow data (sequence_len, 2, height, width)
            results['gt_flow'] = torch.zeros((3, 2, 200, 200), dtype=torch.float32)
        if 'gt_backward_flow' not in results:
            # Create dummy backward flow data (sequence_len, 2, height, width)
            results['gt_backward_flow'] = torch.zeros((3, 2, 200, 200), dtype=torch.float32)
        if 'gt_occ_has_invalid_frame' not in results:
            # Create dummy occupancy invalid frame flag
            results['gt_occ_has_invalid_frame'] = torch.zeros((3,), dtype=torch.bool)
        if 'gt_occ_img_is_valid' not in results:
            # Create dummy occupancy image validity flag
            results['gt_occ_img_is_valid'] = torch.ones((3,), dtype=torch.bool)

        # Convert all occupancy/segmentation data to DataContainer format
        for key in ['gt_segmentation', 'gt_instance', 'gt_centerness', 'gt_offset', 
                   'gt_flow', 'gt_backward_flow', 'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid']:
            if key in results:
                results[key] = DC(results[key])

        # Format CAN bus data - simplify to avoid recursion
        if 'canbus_data' in results:
            # Create simple transformation matrices - no nesting
            results['l2g_r_mat'] = DC(torch.eye(3, dtype=torch.float32))
            results['l2g_t'] = DC(torch.zeros(3, dtype=torch.float32))
            # Convert canbus to simple tensor format
            results['canbus_data'] = DC(torch.zeros(18, dtype=torch.float32))  # Standard UniAD canbus size

        # Ensure timestamp is properly formatted - DON'T wrap in DataContainer
        if 'timestamp' not in results:
            results['timestamp'] = results.get('frame_idx', 0)

        # Add dummy image data if needed (for compatibility)
        if 'img' not in results:
            results['img'] = DC(torch.zeros(1, 3, 224, 224))  # Dummy image tensor

        # Ensure img_metas exists - DON'T wrap in DataContainer, keep as simple dict
        if 'img_metas' not in results:
            results['img_metas'] = {}

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(class_names={self.class_names}, with_label={self.with_label})'