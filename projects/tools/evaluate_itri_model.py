#!/usr/bin/env python3
#---------------------------------------------------------------------------------#
# ITRI Model Evaluation Script
# Evaluation script for trained motion/occupancy/planning heads
#---------------------------------------------------------------------------------#

import argparse
import os
import os.path as osp
import pickle
import tempfile
import numpy as np
import torch
import mmcv
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet3d.apis import single_gpu_test, multi_gpu_test
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger

import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate ITRI trained model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--eval-modes',
        nargs='+',
        default=['motion', 'occ', 'planning'],
        help='evaluation modes')
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


class ITRIEvaluator:
    """Evaluator for ITRI motion/occupancy/planning tasks."""

    def __init__(self, eval_modes=['motion', 'occ', 'planning']):
        self.eval_modes = eval_modes
        self.results = {}

    def evaluate_motion(self, outputs, dataset):
        """Evaluate motion prediction results."""
        print("Evaluating motion prediction...")

        motion_results = {
            'ade': [],  # Average Displacement Error
            'fde': [],  # Final Displacement Error
            'mr': [],   # Miss Rate
        }

        for i, output in enumerate(outputs):
            if 'motion_pred' in output and 'motion_gt' in output:
                pred_traj = output['motion_pred']  # [N, T, 2]
                gt_traj = output['motion_gt']      # [N, T, 2]
                mask = output.get('motion_mask', torch.ones_like(gt_traj[..., 0]))

                # Calculate ADE (Average Displacement Error)
                ade = self._calculate_ade(pred_traj, gt_traj, mask)
                motion_results['ade'].append(ade)

                # Calculate FDE (Final Displacement Error)
                fde = self._calculate_fde(pred_traj, gt_traj, mask)
                motion_results['fde'].append(fde)

                # Calculate Miss Rate (>2m at final timestep)
                mr = self._calculate_miss_rate(pred_traj, gt_traj, mask, threshold=2.0)
                motion_results['mr'].append(mr)

        # Aggregate results
        final_results = {}
        for metric, values in motion_results.items():
            if values:
                final_results[f'motion_{metric}'] = np.mean(values)
                final_results[f'motion_{metric}_std'] = np.std(values)

        return final_results

    def evaluate_occupancy(self, outputs, dataset):
        """Evaluate occupancy prediction results."""
        print("Evaluating occupancy prediction...")

        occ_results = {
            'iou': [],
            'precision': [],
            'recall': [],
        }

        for i, output in enumerate(outputs):
            if 'occ_pred' in output and 'occ_gt' in output:
                pred_occ = output['occ_pred']  # [H, W, T]
                gt_occ = output['occ_gt']      # [H, W, T]

                # Calculate IoU
                iou = self._calculate_occupancy_iou(pred_occ, gt_occ)
                occ_results['iou'].append(iou)

                # Calculate precision and recall
                precision, recall = self._calculate_occupancy_precision_recall(pred_occ, gt_occ)
                occ_results['precision'].append(precision)
                occ_results['recall'].append(recall)

        # Aggregate results
        final_results = {}
        for metric, values in occ_results.items():
            if values:
                final_results[f'occ_{metric}'] = np.mean(values)
                final_results[f'occ_{metric}_std'] = np.std(values)

        return final_results

    def evaluate_planning(self, outputs, dataset):
        """Evaluate planning results."""
        print("Evaluating planning...")

        planning_results = {
            'l2_1s': [],
            'l2_2s': [],
            'l2_3s': [],
            'collision_rate': [],
        }

        for i, output in enumerate(outputs):
            if 'planning_pred' in output and 'planning_gt' in output:
                pred_plan = output['planning_pred']  # [T, 3] (x, y, yaw)
                gt_plan = output['planning_gt']      # [T, 3]

                # Calculate L2 distances at different time horizons
                l2_1s = self._calculate_planning_l2(pred_plan, gt_plan, time_idx=1)
                l2_2s = self._calculate_planning_l2(pred_plan, gt_plan, time_idx=2)
                l2_3s = self._calculate_planning_l2(pred_plan, gt_plan, time_idx=3)

                planning_results['l2_1s'].append(l2_1s)
                planning_results['l2_2s'].append(l2_2s)
                planning_results['l2_3s'].append(l2_3s)

                # Calculate collision rate (simplified)
                collision = self._check_collision(pred_plan, output.get('obstacles', None))
                planning_results['collision_rate'].append(collision)

        # Aggregate results
        final_results = {}
        for metric, values in planning_results.items():
            if values:
                final_results[f'planning_{metric}'] = np.mean(values)
                final_results[f'planning_{metric}_std'] = np.std(values)

        return final_results

    def _calculate_ade(self, pred, gt, mask):
        """Calculate Average Displacement Error."""
        if len(pred.shape) == 3 and len(gt.shape) == 3:
            # pred: [N, T, 2], gt: [N, T, 2], mask: [N, T]
            distances = torch.norm(pred - gt, dim=-1)  # [N, T]
            masked_distances = distances * mask
            ade = masked_distances.sum() / mask.sum()
            return ade.item()
        return 0.0

    def _calculate_fde(self, pred, gt, mask):
        """Calculate Final Displacement Error."""
        if len(pred.shape) == 3 and len(gt.shape) == 3:
            # Use the last valid timestep for each trajectory
            final_distances = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)  # [N]
            final_mask = mask[:, -1]  # [N]
            fde = (final_distances * final_mask).sum() / final_mask.sum()
            return fde.item()
        return 0.0

    def _calculate_miss_rate(self, pred, gt, mask, threshold=2.0):
        """Calculate miss rate (percentage of predictions > threshold at final timestep)."""
        if len(pred.shape) == 3 and len(gt.shape) == 3:
            final_distances = torch.norm(pred[:, -1] - gt[:, -1], dim=-1)  # [N]
            final_mask = mask[:, -1]  # [N]
            misses = (final_distances > threshold) * final_mask
            miss_rate = misses.sum() / final_mask.sum()
            return miss_rate.item()
        return 0.0

    def _calculate_occupancy_iou(self, pred, gt):
        """Calculate IoU for occupancy prediction."""
        if pred.shape != gt.shape:
            return 0.0

        pred_binary = (pred > 0.5).float()
        gt_binary = gt.float()

        intersection = (pred_binary * gt_binary).sum()
        union = (pred_binary + gt_binary - pred_binary * gt_binary).sum()

        if union > 0:
            return (intersection / union).item()
        return 0.0

    def _calculate_occupancy_precision_recall(self, pred, gt):
        """Calculate precision and recall for occupancy prediction."""
        if pred.shape != gt.shape:
            return 0.0, 0.0

        pred_binary = (pred > 0.5).float()
        gt_binary = gt.float()

        tp = (pred_binary * gt_binary).sum()
        fp = (pred_binary * (1 - gt_binary)).sum()
        fn = ((1 - pred_binary) * gt_binary).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return precision.item(), recall.item()

    def _calculate_planning_l2(self, pred, gt, time_idx):
        """Calculate L2 distance for planning at specific time index."""
        if pred.shape[0] > time_idx and gt.shape[0] > time_idx:
            distance = torch.norm(pred[time_idx, :2] - gt[time_idx, :2])
            return distance.item()
        return 0.0

    def _check_collision(self, plan, obstacles):
        """Simple collision checking (placeholder)."""
        # This is a simplified collision check
        # In practice, you'd want more sophisticated collision detection
        return 0.0  # No collision for now

    def evaluate(self, outputs, dataset):
        """Main evaluation function."""
        results = {}

        if 'motion' in self.eval_modes:
            motion_results = self.evaluate_motion(outputs, dataset)
            results.update(motion_results)

        if 'occ' in self.eval_modes:
            occ_results = self.evaluate_occupancy(outputs, dataset)
            results.update(occ_results)

        if 'planning' in self.eval_modes:
            planning_results = self.evaluate_planning(outputs, dataset)
            results.update(planning_results)

        return results

    def print_results(self, results):
        """Print evaluation results in a formatted table."""
        table = PrettyTable()
        table.field_names = ['Metric', 'Value', 'Std']

        for metric, value in results.items():
            if '_std' not in metric:
                std_key = f'{metric}_std'
                std_value = results.get(std_key, 0.0)
                table.add_row([metric, f'{value:.4f}', f'{std_value:.4f}'])

        print(table)

    def save_results(self, results, output_path):
        """Save results to file."""
        with open(output_path, 'w') as f:
            for metric, value in results.items():
                f.write(f'{metric}: {value:.6f}\n')

        print(f"Results saved to {output_path}")


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Import modules from plugin
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # Set random seeds
    if cfg.get('seed', None) is not None:
        mmcv.utils.set_random_seed(cfg.seed, deterministic=True)

    # Set up work_dir
    if args.work_dir is not None:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        json_file = osp.join(args.work_dir, 'eval_results.json')
    else:
        json_file = None

    # Build the dataset
    dataset = build_dataset(cfg.data.test)

    # Build the model and load checkpoint
    cfg.model.pretrained = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # Fuse conv and bn
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # Set up distributed testing
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, False)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                args.gpu_collect)

    # Initialize evaluator
    evaluator = ITRIEvaluator(eval_modes=args.eval_modes)

    # Evaluate results
    results = evaluator.evaluate(outputs, dataset)

    # Print results
    evaluator.print_results(results)

    # Save results
    if args.work_dir:
        result_file = osp.join(args.work_dir, 'evaluation_results.txt')
        evaluator.save_results(results, result_file)

    if args.out:
        print(f'Saving results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()