#!/usr/bin/env python3

import os
import argparse
import pickle
from typing import Dict, Any

import numpy as np
import torch


def load_canbus_pkl(pkl_path: str) -> Dict[str, Any]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def to_torch_tensor(array: Any) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array
    if isinstance(array, np.ndarray):
        # Preserve float64 for timestamps to avoid precision loss
        if array.dtype == np.float64:
            return torch.from_numpy(np.ascontiguousarray(array)).to(torch.float64)
        # Ensure contiguous and float32 for other numeric arrays
        elif array.dtype.kind in {"f", "i", "u"}:
            return torch.from_numpy(np.ascontiguousarray(array)).to(torch.float32)
        return torch.from_numpy(np.ascontiguousarray(array))
    # Fallback: try constructing tensor directly
    return torch.tensor(array)


def convert_single_file(pkl_path: str, pt_path: str) -> None:
    data = load_canbus_pkl(pkl_path)

    # Expecting keys: 'can_bus' (N, 18) and 'timestamps' (N,)
    torch_data: Dict[str, torch.Tensor] = {}
    for key, value in data.items():
        torch_data[key] = to_torch_tensor(value)

    # Basic validation
    if 'can_bus' in torch_data:
        if torch_data['can_bus'].ndim != 2 or torch_data['can_bus'].size(-1) != 18:
            print(f"Warning: {os.path.basename(pkl_path)} has unexpected can_bus shape {tuple(torch_data['can_bus'].shape)} (expected [N, 18])")

    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    torch.save(torch_data, pt_path)
    print(f"Saved: {pt_path}")


def extract_yaw_and_timestamps(pkl_path: str):
    data = load_canbus_pkl(pkl_path)
    if 'can_bus' not in data:
        raise KeyError(f"'can_bus' not found in {os.path.basename(pkl_path)}")
    can_bus = data['can_bus']
    # Accept numpy array or tensor
    if isinstance(can_bus, torch.Tensor):
        yaw = can_bus[:, 16].detach().cpu().numpy()
    else:
        yaw = np.asarray(can_bus)[:, 16]
    timestamps = None
    if 'timestamps' in data:
        timestamps = np.asarray(data['timestamps'])
    return yaw, timestamps


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CAN bus .pkl files to .pt (PyTorch) format.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/home/bryan/Desktop/Allen/UniAD/data/itri/hct_train/canbus",
        help="Directory containing .pkl CAN bus files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pt files if present",
    )
    parser.add_argument(
        "--yaw",
        action="store_true",
        help="If set, output yaw values (one per line) instead of converting to .pt",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    pkl_files = [
        os.path.join(input_dir, name)
        for name in sorted(os.listdir(input_dir))
        if name.endswith(".pkl")
    ]

    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return

    print(f"Found {len(pkl_files)} .pkl file(s) in {input_dir}")

    if args.yaw:
        # Output only yaw values for all files, one per line
        for pkl_path in pkl_files:
            try:
                yaw, _ = extract_yaw_and_timestamps(pkl_path)
                for y in yaw:
                    # Print yaw only, as requested
                    print(float(y))
            except Exception as exc:
                print(f"Failed to extract yaw from {os.path.basename(pkl_path)}: {exc}")
        return

    for pkl_path in pkl_files:
        base, _ = os.path.splitext(pkl_path)
        pt_path = f"{base}.pt"
        if os.path.exists(pt_path) and not args.overwrite:
            print(f"Skip (exists): {os.path.basename(pt_path)}")
            continue
        try:
            convert_single_file(pkl_path, pt_path)
        except Exception as exc:
            print(f"Failed to convert {os.path.basename(pkl_path)}: {exc}")


if __name__ == "__main__":
    main()




