#!/usr/bin/env python3
#---------------------------------------------------------------------------------#
# ITRI Motion/Occupancy/Planning Training Script
# Training script for UniAD motion, occupancy, and planning heads using ITRI data
#---------------------------------------------------------------------------------#

import sys
import os
# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import copy
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, IterBasedRunner,
                         OptimizerHook, Fp16OptimizerHook,
                         build_optimizer, build_runner, EvalHook, DistEvalHook, HOOKS)
from mmcv.utils import build_from_cfg
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Train ITRI Motion/Occ/Planning model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
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
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def freeze_model_components(model, freeze_list):
    """Freeze specified model components."""
    for name, param in model.named_parameters():
        should_freeze = False
        for freeze_component in freeze_list:
            if freeze_component in name:
                should_freeze = True
                break

        if should_freeze:
            param.requires_grad = False
            print(f"Frozen parameter: {name}")
        else:
            print(f"Trainable parameter: {name}")


def setup_model_for_itri_training(model, cfg):
    """Setup model for ITRI training - freeze unnecessary components."""

    # Components to freeze (everything except motion, occ, planning heads)
    freeze_components = [
        'img_backbone',
        'img_neck',
        'pts_bbox_head',  # Detection/tracking head
        'seg_head',       # Segmentation head
    ]

    # Optional: also freeze BEV encoder if specified in config
    if getattr(cfg.model, 'freeze_bev_encoder', False):
        freeze_components.append('bev_encoder')

    # Freeze specified components
    freeze_model_components(model, freeze_components)

    # Verify that motion, occ, planning heads are trainable
    trainable_heads = ['motion_head', 'occ_head', 'planning_head']
    for head_name in trainable_heads:
        if hasattr(model, head_name):
            head = getattr(model, head_name)
            for param in head.parameters():
                param.requires_grad = True
            print(f"Ensured {head_name} is trainable")

    return model


def custom_loss_weighting(losses, cfg):
    """Apply custom loss weighting for ITRI training."""
    weighted_losses = {}

    # Get loss weights from config
    loss_weights = getattr(cfg, 'loss_weights', {})

    # Default weights for ITRI training (focus on motion/occ/planning)
    default_weights = {
        'motion_loss_weight': 2.0,
        'occ_loss_weight': 2.0,
        'planning_loss_weight': 2.0,
        'detection_loss_weight': 0.0,  # Disable detection losses
        'tracking_loss_weight': 0.0,   # Disable tracking losses
        'seg_loss_weight': 0.0,        # Disable segmentation losses
    }

    # Merge config weights with defaults
    weights = {**default_weights, **loss_weights}

    # Apply weights to losses
    for loss_name, loss_value in losses.items():
        if 'motion' in loss_name.lower():
            weighted_losses[loss_name] = loss_value * weights['motion_loss_weight']
        elif 'occ' in loss_name.lower():
            weighted_losses[loss_name] = loss_value * weights['occ_loss_weight']
        elif 'planning' in loss_name.lower() or 'plan' in loss_name.lower():
            weighted_losses[loss_name] = loss_value * weights['planning_loss_weight']
        elif any(key in loss_name.lower() for key in ['det', 'track', 'bbox', 'cls']):
            weighted_losses[loss_name] = loss_value * weights['detection_loss_weight']
        elif 'seg' in loss_name.lower():
            weighted_losses[loss_name] = loss_value * weights['seg_loss_weight']
        else:
            # Keep original weight for unrecognized losses
            weighted_losses[loss_name] = loss_value

    return weighted_losses


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            import sys
            
            # Add current directory to Python path for plugin imports
            current_dir = os.getcwd()
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # Set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # Update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # Use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.auto_scale_lr:
        # Apply auto scale lr
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            warnings.warn('Can not find "auto_scale_lr" or '
                          '"auto_scale_lr.enable" or '
                          '"auto_scale_lr.base_batch_size" in your'
                          ' configuration file. Please update your'
                          ' configs according to the new sample configuration file.')

    if args.gpus is not None:
        cfg.gpu_ids = range(1 if args.gpus is None else args.gpus)
    else:
        cfg.gpu_ids = range(1)

    # Init distributed env first
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # Re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # Dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # Init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # Init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # Log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # Log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # Set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        # Use mmcv's set_random_seed if available, otherwise use torch's
        if hasattr(mmcv.utils, 'set_random_seed'):
            mmcv.utils.set_random_seed(args.seed, deterministic=args.deterministic)
        else:
            import random
            import numpy as np
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                torch.cuda.manual_seed_all(args.seed)
    cfg.seed = args.seed
    meta['seed'] = args.seed

    # Build model
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # Setup model for ITRI training (freeze components)
    model = setup_model_for_itri_training(model, cfg)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Trainable parameters: {trainable_params:,} / {total_params:,} '
                f'({100 * trainable_params / total_params:.2f}%)')

    # Build datasets
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # Save UniAD version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            version=__version__,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Use custom training with loss weighting
    train_model_itri(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


def train_model_itri(model,
                     datasets,
                     cfg,
                     distributed=False,
                     validate=False,
                     timestamp=None,
                     meta=None):
    """Custom training function for ITRI with loss weighting."""

    logger = get_root_logger(cfg.log_level)

    # Prepare data loaders
    dataset = datasets[0]
    if validate:
        val_dataset = datasets[1] if len(datasets) > 1 else None

    if distributed:
        from mmcv.parallel import MMDistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model = model.cuda()
        
    # Ensure all model parameters are on the correct device
    device = next(model.parameters()).device
    logger.info(f'Model moved to device: {device}')
    
    # Verify all parameters are on the same device
    for name, param in model.named_parameters():
        if param.device != device:
            logger.warning(f'Parameter {name} is on {param.device}, expected {device}')
            param.data = param.data.to(device)

    # Build data loader
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,  # dict(type="DistributedGroupSampler")
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,  # dict(type="DistributedSampler")
        )
    ]

    # Build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # Build runner
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # An ugly walkaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # FP16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # Register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # Register eval hooks
    if validate:
        val_dataloader = build_dataloader(
            val_dataset if val_dataset else dataset,
            1,  # samples_per_gpu for validation
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False,
            seed=cfg.seed,
            shuffler_sampler=cfg.data.nonshuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        )
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # User-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    # Load checkpoint if specified
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # Start training
    runner.run(data_loaders, cfg.workflow)


if __name__ == '__main__':
    main()