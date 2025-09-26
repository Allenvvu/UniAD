#!/usr/bin/env python3
#---------------------------------------------------------------------------------#
# ITRI Training Monitoring Script
# Monitor and visualize training progress for motion/occupancy/planning heads
#---------------------------------------------------------------------------------#

import argparse
import os
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import time


class TrainingMonitor:
    """Monitor training progress and create visualizations."""

    def __init__(self, work_dir, refresh_interval=30):
        self.work_dir = work_dir
        self.refresh_interval = refresh_interval
        self.loss_history = {}
        self.eval_history = {}

    def parse_log_file(self, log_file):
        """Parse training log file to extract loss and evaluation metrics."""
        losses = {
            'epoch': [],
            'iter': [],
            'motion_loss': [],
            'occ_loss': [],
            'planning_loss': [],
            'total_loss': [],
        }

        evals = {
            'epoch': [],
            'motion_ade': [],
            'motion_fde': [],
            'occ_iou': [],
            'planning_l2_1s': [],
            'planning_l2_2s': [],
            'planning_l2_3s': [],
        }

        if not os.path.exists(log_file):
            return losses, evals

        with open(log_file, 'r') as f:
            for line in f:
                # Parse training losses
                if 'Epoch' in line and 'iter' in line and 'loss' in line:
                    loss_data = self._parse_training_line(line)
                    if loss_data:
                        for key, value in loss_data.items():
                            if key in losses:
                                losses[key].append(value)

                # Parse evaluation results
                if 'Eval' in line or 'validation' in line.lower():
                    eval_data = self._parse_eval_line(line)
                    if eval_data:
                        for key, value in eval_data.items():
                            if key in evals:
                                evals[key].append(value)

        return losses, evals

    def _parse_training_line(self, line):
        """Parse a training log line to extract loss values."""
        data = {}

        # Extract epoch
        epoch_match = re.search(r'Epoch\s*\[(\d+)\]', line)
        if epoch_match:
            data['epoch'] = int(epoch_match.group(1))

        # Extract iteration
        iter_match = re.search(r'iter\s*:\s*(\d+)', line)
        if iter_match:
            data['iter'] = int(iter_match.group(1))

        # Extract various losses
        loss_patterns = {
            'motion_loss': r'motion[_\s]*loss[:\s]+([0-9.]+)',
            'occ_loss': r'occ[_\s]*loss[:\s]+([0-9.]+)',
            'planning_loss': r'plan(?:ning)?[_\s]*loss[:\s]+([0-9.]+)',
            'total_loss': r'(?:total[_\s]*)?loss[:\s]+([0-9.]+)',
        }

        for loss_name, pattern in loss_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                data[loss_name] = float(match.group(1))

        return data if len(data) > 2 else None  # Only return if we have more than just epoch/iter

    def _parse_eval_line(self, line):
        """Parse an evaluation log line to extract metrics."""
        data = {}

        # Extract epoch
        epoch_match = re.search(r'Epoch\s*\[(\d+)\]', line)
        if epoch_match:
            data['epoch'] = int(epoch_match.group(1))

        # Extract evaluation metrics
        eval_patterns = {
            'motion_ade': r'motion[_\s]*ade[:\s]+([0-9.]+)',
            'motion_fde': r'motion[_\s]*fde[:\s]+([0-9.]+)',
            'occ_iou': r'occ[_\s]*iou[:\s]+([0-9.]+)',
            'planning_l2_1s': r'planning[_\s]*l2[_\s]*1s[:\s]+([0-9.]+)',
            'planning_l2_2s': r'planning[_\s]*l2[_\s]*2s[:\s]+([0-9.]+)',
            'planning_l2_3s': r'planning[_\s]*l2[_\s]*3s[:\s]+([0-9.]+)',
        }

        for metric_name, pattern in eval_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                data[metric_name] = float(match.group(1))

        return data if len(data) > 1 else None  # Only return if we have metrics

    def create_loss_plots(self, losses, save_path):
        """Create loss visualization plots."""
        if not any(losses.values()):
            print("No loss data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Loss Progress', fontsize=16)

        # Total loss plot
        if losses['total_loss']:
            axes[0, 0].plot(losses['iter'], losses['total_loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Total Loss')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)

        # Motion loss plot
        if losses['motion_loss']:
            axes[0, 1].plot(losses['iter'][:len(losses['motion_loss'])],
                           losses['motion_loss'], 'r-', linewidth=2)
            axes[0, 1].set_title('Motion Loss')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Occupancy loss plot
        if losses['occ_loss']:
            axes[1, 0].plot(losses['iter'][:len(losses['occ_loss'])],
                           losses['occ_loss'], 'g-', linewidth=2)
            axes[1, 0].set_title('Occupancy Loss')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)

        # Planning loss plot
        if losses['planning_loss']:
            axes[1, 1].plot(losses['iter'][:len(losses['planning_loss'])],
                           losses['planning_loss'], 'm-', linewidth=2)
            axes[1, 1].set_title('Planning Loss')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_eval_plots(self, evals, save_path):
        """Create evaluation metrics visualization plots."""
        if not any(evals.values()):
            print("No evaluation data available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Evaluation Metrics Progress', fontsize=16)

        # Motion metrics
        if evals['motion_ade'] and evals['motion_fde']:
            epochs = evals['epoch'][:len(evals['motion_ade'])]
            axes[0, 0].plot(epochs, evals['motion_ade'], 'b-o', label='ADE', linewidth=2)
            axes[0, 0].plot(epochs[:len(evals['motion_fde'])], evals['motion_fde'], 'r-o', label='FDE', linewidth=2)
            axes[0, 0].set_title('Motion Prediction Metrics')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Error (m)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Occupancy IoU
        if evals['occ_iou']:
            epochs = evals['epoch'][:len(evals['occ_iou'])]
            axes[0, 1].plot(epochs, evals['occ_iou'], 'g-o', linewidth=2)
            axes[0, 1].set_title('Occupancy IoU')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('IoU')
            axes[0, 1].grid(True, alpha=0.3)

        # Planning L2 errors
        if evals['planning_l2_1s']:
            epochs = evals['epoch'][:len(evals['planning_l2_1s'])]
            if evals['planning_l2_1s']:
                axes[1, 0].plot(epochs, evals['planning_l2_1s'], 'c-o', label='1s', linewidth=2)
            if evals['planning_l2_2s']:
                axes[1, 0].plot(epochs[:len(evals['planning_l2_2s'])], evals['planning_l2_2s'], 'm-o', label='2s', linewidth=2)
            if evals['planning_l2_3s']:
                axes[1, 0].plot(epochs[:len(evals['planning_l2_3s'])], evals['planning_l2_3s'], 'y-o', label='3s', linewidth=2)
            axes[1, 0].set_title('Planning L2 Error')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('L2 Error (m)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Summary metrics
        if evals['motion_ade'] and evals['occ_iou'] and evals['planning_l2_1s']:
            # Create a combined performance score
            motion_score = 1.0 / (1.0 + np.array(evals['motion_ade']))
            occ_score = np.array(evals['occ_iou'])
            planning_score = 1.0 / (1.0 + np.array(evals['planning_l2_1s']))

            min_len = min(len(motion_score), len(occ_score), len(planning_score))
            combined_score = (motion_score[:min_len] + occ_score[:min_len] + planning_score[:min_len]) / 3

            epochs = evals['epoch'][:min_len]
            axes[1, 1].plot(epochs, combined_score, 'k-o', linewidth=2)
            axes[1, 1].set_title('Combined Performance Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_summary_report(self, losses, evals, save_path):
        """Create a summary report of training progress."""
        with open(save_path, 'w') as f:
            f.write("ITRI UniAD Training Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Training progress
            f.write("Training Progress:\n")
            f.write("-" * 20 + "\n")
            if losses['total_loss']:
                f.write(f"Total iterations: {len(losses['total_loss'])}\n")
                f.write(f"Current total loss: {losses['total_loss'][-1]:.6f}\n")
                f.write(f"Best total loss: {min(losses['total_loss']):.6f}\n")

            if losses['motion_loss']:
                f.write(f"Current motion loss: {losses['motion_loss'][-1]:.6f}\n")
                f.write(f"Best motion loss: {min(losses['motion_loss']):.6f}\n")

            if losses['occ_loss']:
                f.write(f"Current occ loss: {losses['occ_loss'][-1]:.6f}\n")
                f.write(f"Best occ loss: {min(losses['occ_loss']):.6f}\n")

            if losses['planning_loss']:
                f.write(f"Current planning loss: {losses['planning_loss'][-1]:.6f}\n")
                f.write(f"Best planning loss: {min(losses['planning_loss']):.6f}\n")

            # Evaluation results
            f.write(f"\nEvaluation Results:\n")
            f.write("-" * 20 + "\n")
            if evals['motion_ade']:
                f.write(f"Best Motion ADE: {min(evals['motion_ade']):.6f}m\n")
            if evals['motion_fde']:
                f.write(f"Best Motion FDE: {min(evals['motion_fde']):.6f}m\n")
            if evals['occ_iou']:
                f.write(f"Best Occupancy IoU: {max(evals['occ_iou']):.6f}\n")
            if evals['planning_l2_1s']:
                f.write(f"Best Planning L2 (1s): {min(evals['planning_l2_1s']):.6f}m\n")
            if evals['planning_l2_2s']:
                f.write(f"Best Planning L2 (2s): {min(evals['planning_l2_2s']):.6f}m\n")
            if evals['planning_l2_3s']:
                f.write(f"Best Planning L2 (3s): {min(evals['planning_l2_3s']):.6f}m\n")

    def monitor_training(self, continuous=False):
        """Monitor training progress."""
        print(f"Monitoring training in: {self.work_dir}")

        while True:
            # Find the latest log file
            log_files = [f for f in os.listdir(self.work_dir) if f.endswith('.log')]
            if not log_files:
                print("No log files found. Waiting...")
                if not continuous:
                    break
                time.sleep(self.refresh_interval)
                continue

            latest_log = max(log_files, key=lambda f: os.path.getctime(os.path.join(self.work_dir, f)))
            log_path = os.path.join(self.work_dir, latest_log)

            # Parse the log file
            losses, evals = self.parse_log_file(log_path)

            # Create visualizations
            if losses['total_loss']:
                loss_plot_path = os.path.join(self.work_dir, 'training_losses.png')
                self.create_loss_plots(losses, loss_plot_path)
                print(f"Loss plots saved to: {loss_plot_path}")

            if any(evals.values()):
                eval_plot_path = os.path.join(self.work_dir, 'evaluation_metrics.png')
                self.create_eval_plots(evals, eval_plot_path)
                print(f"Evaluation plots saved to: {eval_plot_path}")

            # Create summary report
            summary_path = os.path.join(self.work_dir, 'training_summary.txt')
            self.create_summary_report(losses, evals, summary_path)
            print(f"Summary report saved to: {summary_path}")

            if not continuous:
                break

            print(f"Monitoring... (refreshing every {self.refresh_interval}s)")
            time.sleep(self.refresh_interval)


def parse_args():
    parser = argparse.ArgumentParser(description='Monitor ITRI training progress')
    parser.add_argument('work_dir', help='training work directory')
    parser.add_argument('--continuous', action='store_true',
                       help='continuously monitor training')
    parser.add_argument('--refresh-interval', type=int, default=30,
                       help='refresh interval in seconds for continuous monitoring')
    return parser.parse_args()


def main():
    args = parse_args()

    monitor = TrainingMonitor(args.work_dir, args.refresh_interval)
    monitor.monitor_training(continuous=args.continuous)


if __name__ == '__main__':
    main()