#!/usr/bin/env python

import argparse
import os
import warnings
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='Test UniAD with limited samples')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results_10.pkl', help='output result file')
    parser.add_argument('--max-samples', type=int, default=10, help='maximum number of samples to test')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    
    # Build dataset
    dataset = build_dataset(cfg.data.test)
    print(f"Original dataset size: {len(dataset)}")
    
    # Limit dataset to max_samples
    if args.max_samples and args.max_samples < len(dataset):
        # Create a subset of the dataset
        original_infos = dataset.data_infos
        dataset.data_infos = original_infos[:args.max_samples]
        print(f'Limited dataset to {len(dataset.data_infos)} samples')
    
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)
    
    # Build model
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    
    # Run single GPU test function to avoid distributed processing issues
    print(f"Starting inference on {len(dataset)} samples...")
    results = single_gpu_test(model, data_loader, show=False)
    
    # Save results
    print(f"Saving results to {args.out}")
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    mmcv.dump(results, args.out)
    print(f'Results saved to {args.out}')
    print(f'Completed inference on {len(results)} samples')

if __name__ == '__main__':
    main()