#!/usr/bin/env python3

import pickle

def fix_dataset_metadata():
    """Add required metadata to our L2G dataset file"""

    # Load our current dataset
    dataset_file = "data/infos/nuscenes_infos_temporal_val.pkl"
    with open(dataset_file, 'rb') as f:
        data = pickle.load(f)

    print(f"Current data keys: {data.keys()}")
    print(f"Number of samples: {len(data['infos'])}")

    # Add required metadata structure
    # This mimics the nuScenes dataset format that UniAD expects
    data['metadata'] = {
        'version': 'v1.0-trainval',  # Standard nuScenes version
        'use_camera': True,
        'use_lidar': False,
        'use_radar': False,
        'use_map': False,
        'use_external': True,
    }

    # Save the fixed dataset
    print(f"Adding metadata: {data['metadata']}")
    with open(dataset_file, 'wb') as f:
        pickle.dump(data, f)

    print(f"Fixed dataset saved to {dataset_file}")

if __name__ == "__main__":
    fix_dataset_metadata()