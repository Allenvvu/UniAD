#!/usr/bin/env python3

import pickle
import glob
import os

def combine_l2g_datasets():
    """Combine all individual L2G converted scene files into a single annotation file"""

    # Find all converted L2G data files
    data_files = sorted(glob.glob("data/*_l2g_converted_infos.pkl"))
    print(f"Found {len(data_files)} L2G scene files to combine")

    combined_data = {'infos': []}

    # Load and combine all scene data
    for data_file in data_files:
        print(f"Loading {data_file}...")
        with open(data_file, 'rb') as f:
            scene_data = pickle.load(f)

        # Add all samples from this scene to combined dataset
        if 'infos' in scene_data:
            combined_data['infos'].extend(scene_data['infos'])
            print(f"  Added {len(scene_data['infos'])} samples")
        elif isinstance(scene_data, list):
            combined_data['infos'].extend(scene_data)
            print(f"  Added {len(scene_data)} samples")
        else:
            print(f"  Warning: Unexpected data format in {data_file}")

    # Save combined dataset
    output_file = "data/infos/nuscenes_infos_temporal_val.pkl"
    print(f"\nSaving combined dataset to {output_file}")
    print(f"Total samples: {len(combined_data['infos'])}")

    with open(output_file, 'wb') as f:
        pickle.dump(combined_data, f)

    print("Dataset combination complete!")
    return output_file

if __name__ == "__main__":
    combine_l2g_datasets()