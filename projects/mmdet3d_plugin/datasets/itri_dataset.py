#---------------------------------------------------------------------------------#
# ITRI Dataset for UniAD Motion/Occupancy/Planning Training
# Adapted from NuScenesE2EDataset for ITRI-specific data format
#---------------------------------------------------------------------------------#

import copy
import numpy as np
import torch
import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from os import path as osp
import pickle
import os
from mmcv.parallel import DataContainer as DC


@DATASETS.register_module()
class ITRIDataset(NuScenesDataset):
    """ITRI Dataset for Motion/Occupancy/Planning Training.

    This dataset loads pre-computed data for training motion, occupancy,
    and planning heads without image processing.
    """

    def __init__(self,
                 queue_length=3,
                 bev_size=(200, 200),
                 patch_size=(102.4, 102.4),
                 canvas_size=(200, 200),
                 predict_steps=12,
                 planning_steps=6,
                 past_steps=4,
                 fut_steps=4,
                 use_nonlinear_optimizer=False,
                 eval_mod=None,

                 # ITRI-specific data paths
                 bev_features_root=None,
                 canbus_root=None,
                 sdc_embeddings_root=None,
                 track_query_root=None,
                 map_query_root=None,
                 gt_fut_traj_root=None,
                 sdc_planning_root=None,

                 # Occ dataset params
                 occ_receptive_field=3,
                 occ_n_future=4,
                 occ_filter_invalid_sample=False,

                 file_client_args=dict(backend='disk'),
                 *args,
                 **kwargs):

        # Store ITRI data paths
        self.bev_features_root = bev_features_root
        self.canbus_root = canbus_root
        self.sdc_embeddings_root = sdc_embeddings_root
        self.track_query_root = track_query_root
        self.map_query_root = map_query_root
        self.gt_fut_traj_root = gt_fut_traj_root
        self.sdc_planning_root = sdc_planning_root

        # File client setup
        self.file_client_args = file_client_args
        self.file_client = mmcv.FileClient(**file_client_args)

        # Store parameters
        self.queue_length = queue_length
        self.bev_size = bev_size
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.use_nonlinear_optimizer = use_nonlinear_optimizer
        self.eval_mod = eval_mod

        # Occ parameters
        self.occ_receptive_field = occ_receptive_field
        self.occ_n_future = occ_n_future
        self.occ_filter_invalid_sample = occ_filter_invalid_sample

        # Initialize parent class first to get data_root
        super().__init__(*args, **kwargs)

        # Create mapping from bag files to data indices AFTER parent init
        self._create_data_mapping()
        self._load_timestamps()

    def _create_data_mapping(self):
        """Create mapping between bag files and available data files."""
        self.data_mapping = {}

        # Get list of available bag files
        if os.path.exists(self.data_root):
            bag_files = []
            if os.path.exists(os.path.join(self.data_root, "bag")):
                for f in os.listdir(os.path.join(self.data_root, "bag")):
                    if f.endswith('.bag'):
                        bag_name = f.replace('.bag', '')
                        bag_files.append(bag_name)

            # For each bag file, check what data is available
            for bag_name in bag_files:
                self.data_mapping[bag_name] = {
                    'canbus': self._check_canbus_data(bag_name),
                    'sdc_embeddings': self._check_timestamp_data('sdc_embeddings'),
                    'track_query': self._check_timestamp_data('track_query'),
                    'map_query': self._check_timestamp_data('map_query'),
                    'gt_fut_traj': self._check_timestamp_data('gt_fut_traj'),
                    'sdc_planning': self._check_timestamp_data('sdc_planning'),
                    'gt_sdc': self._check_timestamp_data('gt_sdc'),
                    'timestamps': self._check_timestamps_data(),
                }

    def _check_canbus_data(self, bag_name):
        """Check for canbus data for specific bag."""
        canbus_path = os.path.join(self.data_root, 'canbus')
        if not os.path.exists(canbus_path):
            return None
        
        canbus_file = os.path.join(canbus_path, f"{bag_name}_can_bus.pkl")
        return canbus_file if os.path.exists(canbus_file) else None

    def _check_timestamp_data(self, data_type):
        """Check for timestamp-based data files."""
        data_path = os.path.join(self.data_root, data_type)
        if not os.path.exists(data_path):
            return []
        
        # Get all .pt files and sort by timestamp
        data_files = []
        try:
            for f in os.listdir(data_path):
                if f.endswith('.pt'):
                    # Extract timestamp from filename
                    timestamp_str = f.replace('.pt', '')
                    if timestamp_str.isdigit():
                        data_files.append(os.path.join(data_path, f))
        except OSError:
            pass
        
        # Sort by timestamp
        return sorted(data_files, key=lambda x: int(os.path.basename(x).replace('.pt', '')))

    def _check_timestamps_data(self):
        """Check for timestamps data."""
        timestamps_path = os.path.join(self.data_root, 'timestamps')
        if not os.path.exists(timestamps_path):
            return None
        
        timestamps_file = os.path.join(timestamps_path, 'continuous_timestamps.pkl')
        return timestamps_file if os.path.exists(timestamps_file) else None


    def _load_timestamps(self):
        """Load continuous timestamps for validation."""
        self.continuous_timestamps = {}
        timestamp_file = os.path.join(self.data_root, 'timestamps', 'continuous_timestamps.pkl')

        if os.path.exists(timestamp_file):
            try:
                with open(timestamp_file, 'rb') as f:
                    timestamp_data = pickle.load(f)
                    # Handle the new timestamp structure
                    if isinstance(timestamp_data, dict) and 'timestamps' in timestamp_data:
                        # Extract timestamps from the dict structure
                        self.continuous_timestamps = timestamp_data['timestamps']
                    else:
                        # Direct array/list of timestamps
                        self.continuous_timestamps = timestamp_data
            except Exception as e:
                print(f"Warning: Could not load timestamps: {e}")
                self.continuous_timestamps = []

    def _validate_timestamp(self, timestamp, bag_name):
        """Validate timestamp against continuous timestamps."""
        if not self.continuous_timestamps:
            return True

        # Check if timestamp exists in the valid range (100ms tolerance)
        return any(abs(ts - timestamp) < 100000 for ts in self.continuous_timestamps)

    def load_annotations(self, ann_file):
        """Load ITRI annotations.

        For ITRI data, we create synthetic annotations based on available data files.
        """
        # Handle both file path strings and file handles
        if ann_file is None:
            # No annotation file provided, create synthetic annotations
            return self._create_synthetic_annotations()
        elif hasattr(ann_file, 'read'):  # It's a file handle
            ann_file_path = ann_file.name
        else:  # It's a string path
            ann_file_path = ann_file
            
        if ann_file_path and os.path.exists(ann_file_path):
            # Load existing annotation file if available
            with open(ann_file_path, 'rb') as f:
                data = pickle.load(f)
            infos = data.get('infos', [])
            # Update data paths to match new structure
            return self._update_annotation_paths(infos)
        else:
            # Create synthetic annotations from available data
            return self._create_synthetic_annotations()

    def _update_annotation_paths(self, infos):
        """Update annotation paths to match new data structure."""
        updated_infos = []

        # If data_mapping doesn't exist yet, create a simple mapping
        if not hasattr(self, 'data_mapping'):
            self._create_data_mapping()

        for info in infos:
            # Extract bag name from scene_token
            bag_name = info.get('scene_token', 'unknown')
            timestamp = info.get('timestamp', 0)

            # Create new available_data structure
            available_data = {}

            # Map canbus data
            if bag_name in self.data_mapping:
                canbus_file = self.data_mapping[bag_name].get('canbus')
                available_data['canbus'] = canbus_file

            # Check if info has old-style itri_data paths
            itri_data = info.get('itri_data', {})

            # Map timestamp-based data
            for data_type in ['sdc_embeddings', 'track_query', 'map_query', 'gt_fut_traj', 'sdc_planning', 'gt_sdc']:
                matching_file = None

                # First, try to extract timestamp from old annotation paths
                if data_type in itri_data:
                    old_path = itri_data[data_type]
                    if old_path:
                        # Extract timestamp from old path (e.g., "data/itri/hct_train/sdc_embeddings/bag_name/1754461385583160.pt")
                        old_timestamp = self._extract_timestamp_from_filename(old_path)
                        if old_timestamp > 0:
                            # Look for file with this timestamp in new structure
                            new_path = os.path.join(self.data_root, data_type, f"{old_timestamp}.pt")
                            if os.path.exists(new_path):
                                matching_file = new_path
                        else:
                            # For files like track_queries.pt, use the annotation timestamp
                            # Try both milliseconds and microseconds format
                            for ts_candidate in [timestamp, timestamp * 1000]:
                                new_path = os.path.join(self.data_root, data_type, f"{ts_candidate}.pt")
                                if os.path.exists(new_path):
                                    matching_file = new_path
                                    break

                # If no match from old paths, try timestamp matching
                if not matching_file:
                    # Check if the data directory exists
                    data_dir = os.path.join(self.data_root, data_type)
                    if os.path.exists(data_dir):
                        # Get all files and find closest timestamp match
                        target_ts = timestamp * 1000  # Convert to microseconds
                        best_match = None
                        min_diff = float('inf')

                        for filename in os.listdir(data_dir):
                            if filename.endswith('.pt'):
                                file_ts = self._extract_timestamp_from_filename(filename)
                                if file_ts > 0:
                                    diff = abs(file_ts - target_ts)
                                    if diff < min_diff and diff < 1000:  # 1ms tolerance
                                        min_diff = diff
                                        best_match = os.path.join(data_dir, filename)

                        if best_match:
                            matching_file = best_match

                available_data[data_type] = matching_file

            # Add bag_name to info
            info['bag_name'] = bag_name
            info['available_data'] = available_data

            updated_infos.append(info)

        return updated_infos

    def _create_synthetic_annotations(self):
        """Create synthetic annotations from available ITRI data."""
        data_infos = []

        # Get all available timestamps from any data type
        all_timestamps = set()
        for bag_name, data_dict in self.data_mapping.items():
            for data_type, files in data_dict.items():
                if data_type == 'canbus' and files:  # canbus is a single file per bag
                    continue
                elif data_type == 'timestamps' and files:  # timestamps is a single file
                    continue
                elif files:  # timestamp-based data
                    for file_path in files:
                        timestamp = self._extract_timestamp_from_filename(file_path)
                        if timestamp > 0:
                            all_timestamps.add(timestamp)

        # Sort timestamps
        sorted_timestamps = sorted(all_timestamps)

        # Create data info for each timestamp
        for i, timestamp in enumerate(sorted_timestamps):
            if self._validate_timestamp(timestamp, None):
                # Find which bag this timestamp belongs to (if any)
                bag_name = self._find_bag_for_timestamp(timestamp)
                data_info = self._create_data_info_for_timestamp(bag_name, i, timestamp, len(data_infos))
                data_infos.append(data_info)

        return sorted(data_infos, key=lambda x: x['timestamp'])

    def _find_bag_for_timestamp(self, timestamp):
        """Find which bag a timestamp belongs to based on canbus data."""
        for bag_name, data_dict in self.data_mapping.items():
            if data_dict.get('canbus'):
                # For now, we'll use a simple heuristic - assign timestamps to bags
                # In practice, you might want to check canbus timestamp ranges
                return bag_name
        return None

    def _create_data_info_for_timestamp(self, bag_name, frame_idx, timestamp, sample_idx):
        """Create data info entry for a specific timestamp."""
        # Map data files for this timestamp
        available_data = {}
        
        # Add canbus data if available
        if bag_name and bag_name in self.data_mapping:
            canbus_file = self.data_mapping[bag_name].get('canbus')
            available_data['canbus'] = canbus_file
        
        # Add timestamp-based data
        for data_type in ['sdc_embeddings', 'track_query', 'map_query', 'gt_fut_traj', 'sdc_planning', 'gt_sdc']:
            files = self.data_mapping.get(bag_name, {}).get(data_type, [])
            if files:
                # Find the file with matching timestamp
                matching_file = None
                for file_path in files:
                    file_timestamp = self._extract_timestamp_from_filename(file_path)
                    # Convert annotation timestamp to microseconds if needed
                    target_timestamp = timestamp
                    if timestamp < 1e12:  # If timestamp is in milliseconds, convert to microseconds
                        target_timestamp = timestamp * 1000
                    
                    if abs(file_timestamp - target_timestamp) < 1000000:  # 2 second tolerance
                        matching_file = file_path
                        break
                available_data[data_type] = matching_file
            else:
                available_data[data_type] = None

        return {
            'bag_name': bag_name,
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'scene_token': bag_name or 'unknown',
            'sample_idx': sample_idx,
            'available_data': available_data,
        }


    def _extract_timestamp_from_filename(self, filename):
        """Extract timestamp from filename (format: 1754461385583160.pt)."""
        try:
            # Extract timestamp from filename (e.g., "1754461385583160.pt")
            basename = os.path.basename(filename)
            timestamp_str = basename.split('.')[0]

            # Check if it's a timestamp-like string (numeric)
            if timestamp_str.isdigit():
                return int(timestamp_str)

            # Handle other possible formats (e.g., bag_name_track_queries.pt)
            # Look for numeric patterns in the filename
            import re
            numbers = re.findall(r'\d{13,}', basename)  # Look for 13+ digit numbers (timestamps)
            if numbers:
                return int(numbers[0])

            return 0
        except:
            return 0

    def prepare_train_data(self, index):
        """Prepare training data for given index."""
        # Ensure we have enough frames for queue
        if index < self.queue_length - 1:
            return None

        # Check if all frames are from same scene
        current_scene = self.data_infos[index]['scene_token']
        for i in range(index - self.queue_length + 1, index + 1):
            if self.data_infos[i]['scene_token'] != current_scene:
                return None

        # Prepare data queue
        data_queue = []
        for i in range(index - self.queue_length + 1, index + 1):
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None

            # Apply pipeline transforms
            example = self.pipeline(input_dict)
            if example is None:
                return None

            data_queue.append(example)

        return self.union2one(data_queue)

    def get_data_info(self, index):
        """Get data info for specific index."""
        info = self.data_infos[index]

        input_dict = dict(
            bag_name=info['bag_name'],
            frame_idx=info['frame_idx'],
            timestamp=info['timestamp'],
            scene_token=info['scene_token'],
            sample_idx=info['sample_idx'],
            available_data=info['available_data'],
        )

        # Add camera and lidar info for compatibility (empty since we're not using images)
        input_dict['img_info'] = dict(
            cams=dict(),
            filename=[],
        )
        input_dict['lidar_info'] = dict(filename='')

        # Add basic img_metas for compatibility - simplified to avoid recursion
        input_dict['img_metas'] = dict(
            sample_idx=info['sample_idx'],
            timestamp=info['timestamp'],
            scene_token=info['scene_token'],
            img_shape=(0, 0, 3),
            pad_shape=(0, 0, 3),
            scale_factor=1.0,
        )

        return input_dict

    def union2one(self, data_queue):
        """Union multiple frames into one sample."""
        imgs = []
        img_metas = []
        timestamps = []
        l2g_r_mats = []
        l2g_ts = []

        # Collect data from all frames in queue
        for sample in data_queue:
            if 'img' in sample:
                # Extract tensor from DataContainer if needed
                img_data = sample['img']
                if hasattr(img_data, 'data'):
                    imgs.append(img_data.data)
                else:
                    imgs.append(img_data)
            
                # Handle img_metas - completely flatten to avoid any nesting
            if 'img_metas' in sample:
                # Always extract to simple dict with essential fields only
                meta = sample['img_metas']
                if isinstance(meta, list):
                    meta = meta[0] if meta else {}

                # Create minimal meta with only essential fields
                simple_meta = {
                    'sample_idx': meta.get('sample_idx', 0),
                    'timestamp': meta.get('timestamp', 0),
                }
                img_metas.append(simple_meta)

            if 'timestamp' in sample:
                timestamp_data = sample['timestamp']
                if hasattr(timestamp_data, 'data'):
                    timestamps.append(timestamp_data.data)
                else:
                    timestamps.append(timestamp_data)
            
            if 'l2g_r_mat' in sample:
                l2g_r_mat_data = sample['l2g_r_mat']
                if hasattr(l2g_r_mat_data, 'data'):
                    l2g_r_mats.append(l2g_r_mat_data.data)
                else:
                    l2g_r_mats.append(l2g_r_mat_data)
            
            if 'l2g_t' in sample:
                l2g_t_data = sample['l2g_t']
                if hasattr(l2g_t_data, 'data'):
                    l2g_ts.append(l2g_t_data.data)
                else:
                    l2g_ts.append(l2g_t_data)

        # Create unified sample - flatten img_metas to avoid nested sequences
        unified_sample = {
            'img_metas': img_metas[-1] if img_metas else {},  # Use only current frame meta
            'timestamp': timestamps[-1] if timestamps else 0,  # Use only current timestamp
        }

        if imgs:
            unified_sample['img'] = torch.stack(imgs, dim=0)
        if l2g_r_mats:
            unified_sample['l2g_r_mat'] = torch.stack(l2g_r_mats, dim=0)
        if l2g_ts:
            unified_sample['l2g_t'] = torch.stack(l2g_ts, dim=0)

        # Copy other data from the last (current) frame and PRESERVE DataContainer
        current_sample = data_queue[-1]
        for key in current_sample:
            if key not in unified_sample:
                data = current_sample[key]
                # Preserve DataContainer so collate can avoid stacking variable-length tensors
                if hasattr(data, 'data'):
                    unified_sample[key] = data
                else:
                    # Avoid copying complex nested structures that might cause recursion
                    if key in ['available_data', 'bag_name', 'scene_token']:
                        # Skip metadata that could contain circular references
                        continue
                    unified_sample[key] = data

        return unified_sample

    def __len__(self):
        return len(self.data_infos)

    def evaluate(self, results, *args, **kwargs):
        """Evaluate function for ITRI dataset."""
        # Implement evaluation logic based on eval_mod
        eval_results = {}

        if 'motion' in self.eval_mod:
            eval_results.update(self._evaluate_motion(results))
        if 'occ' in self.eval_mod:
            eval_results.update(self._evaluate_occupancy(results))
        if 'planning' in self.eval_mod:
            eval_results.update(self._evaluate_planning(results))

        return eval_results

    def _evaluate_motion(self, results):
        """Evaluate motion prediction results."""
        # Placeholder for motion evaluation
        return {'motion_ade': 0.0, 'motion_fde': 0.0}

    def _evaluate_occupancy(self, results):
        """Evaluate occupancy prediction results."""
        # Placeholder for occupancy evaluation
        return {'occ_iou': 0.0}

    def _evaluate_planning(self, results):
        """Evaluate planning results."""
        # Placeholder for planning evaluation
        return {'planning_l2': 0.0, 'planning_collision': 0.0}