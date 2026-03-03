""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examplesf
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2023 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import sys
# Add parent directory to path to import get_cls_map
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
# Also add mambavision directory to path for relative imports
mambavision_dir = os.path.dirname(os.path.abspath(__file__))
if mambavision_dir not in sys.path:
    sys.path.insert(0, mambavision_dir)
get_cls_map = None
get_classification_map = None
list_to_colormap = None
try:
    from get_cls_map import get_cls_map, get_classification_map, list_to_colormap
except ImportError:
    # Fallback if get_cls_map is not in parent directory
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("get_cls_map", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "get_cls_map.py"))
        get_cls_map_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(get_cls_map_module)
        get_cls_map = get_cls_map_module.get_cls_map
        get_classification_map = get_cls_map_module.get_classification_map
        list_to_colormap = get_cls_map_module.list_to_colormap
    except Exception as e:
        # Logger not available at import time, will log later
        pass

from timm.data import ImageDataset, create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm import utils
from timm.loss import JsdCrossEntropy, BinaryCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy,\
    LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler
from scheduler.scheduler_factory import create_scheduler
import shutil
# Import utils.datasets - handle both relative and absolute imports
try:
    from utils.datasets import imagenet_lmdb_dataset, get_hsi_loaders
except ImportError:
    # Fallback: try importing from mambavision.utils if running from parent directory
    from mambavision.utils.datasets import imagenet_lmdb_dataset, get_hsi_loaders
from tensorboard import TensorboardLogger
from models.mamba_vision import *

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('--data_dir', metavar='DIR',
                    help='path to dataset')
group.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (imagenet, hsi, or default ImageFolder/ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--tag', default='exp', type=str, metavar='TAG')
parser.add_argument('--hsi-bands', type=int, default=15, help='Target number of HSI bands after PCA (default: 15)')
parser.add_argument('--hsi-patch', type=int, default=15, help='Spatial patch size for HSI cubes (default: 15)')
parser.add_argument('--hsi-test-ratio', type=float, default=0.99,
                    help='Fraction of data used as test/val (1-train). E.g. 0.98 = 2%% train (Table 9 ablation); 0.90 = 10%% train (default: 0.99)')
parser.add_argument('--hsi-dataset', type=str, default='Indian_pines',
                    help='HSI dataset: Indian_pines, salinas, PaviaU, WHU_Hi_HongHu, etc. (default: Indian_pines)')
parser.add_argument('--use-kan', action='store_true', default=False, help='Use EfficientKAN instead of MLP blocks (default: False)')
parser.add_argument('--use-prompt', action='store_true', default=False, help='Enable prompt integration (default: False)')
parser.add_argument('--task-classes', type=int, default=6, help='Number of task classes for prompts (default: 6)')
parser.add_argument('--prompt-inject-levels', type=int, nargs='+', default=[1, 2], help='Levels to inject prompts (default: [1, 2])')
parser.add_argument('--prompt-mode', type=str, default='full', choices=['full', 'visual_only', 'text_only'],
                    help='Prompt mode for ablation: full (both visual and text), visual_only, or text_only (default: full)')
parser.add_argument('--load-prompt-weights', action='store_true', default=False, help='Load pretrained prompt weights from checkpoints directory (default: False)')
# Stage ablation arguments
parser.add_argument('--abl-stages', type=str, default='all', choices=['all', '012', '01', '0'],
                    help='Stage ablation: all (4 stages), 012 (3 stages), 01 (2 stages), 0 (1 stage) (default: all)')
# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='gc_vit_tiny', type=str, metavar='MODEL',
                    help='Name of model to train (default: "gc_vit_tiny"')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--loadcheckpoint', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image patch size (default: None => model default)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=0.875, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='torch.jit.script the full model')
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                    help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
group.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8, use opt default)')
group.add_argument('--opt-betas', default=[0.9, 0.999], type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')
group.add_argument('--clip-grad', type=float, default=5.0, metavar='NORM',
                    help='Clip gradient norm (default: 5.0, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr-ep', action='store_true', default=False,
                        help='using the epoch-based scheduler')
group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=1.0, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
group.add_argument('--min-lr', type=float, default=5e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (5e-6)')
group.add_argument('--epochs', type=int, default=310, metavar='N',
                    help='number of epochs to train (default: 310)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default="rand-m9-mstd0.5-inc1", metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                    help='Random erase prob (default: 0.25)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.8,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
group.add_argument('--cutmix', type=float, default=1.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
group.add_argument('--attn-drop-rate', type=float, default=0.0, metavar='PCT',
                    help='Drop of the attention, gaussian std')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=1, metavar='N',
                    help='number of checkpoints to keep (default: 3)')
group.add_argument('-j', '--workers', type=int, default=8, metavar='N',
                    help='how many training processes to use (default: 8)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
group.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--log_dir', default='./log_dir/', type=str,
                    help='where to store tensorboard')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument("--data_len", default=1281167, type=int,help='size of the dataset')

group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
group.add_argument('--validate_only', action='store_true', default=False,
                    help='run model validation only')

group.add_argument('--no_saver', action='store_true', default=False,
                    help='Save checkpoints')

# Early convergence warning parameters
group = parser.add_argument_group('Early convergence warning parameters')
group.add_argument('--early-conv-threshold', type=float, default=0.99,
                    help='Accuracy threshold for early convergence warning (default: 0.99)')
group.add_argument('--early-conv-min-batch', type=float, default=0.5,
                    help='Minimum batch progress before checking early convergence (default: 0.5)')
group.add_argument('--early-conv-warning', action='store_true', default=True,
                    help='Enable early convergence warning (default: True)')
group.add_argument('--ampere_sparsity', action='store_true', default=False,
                    help='Save checkpoints')
group.add_argument('--lmdb_dataset', action='store_true', default=False,
                    help='use lmdb dataset')
group.add_argument('--bfloat', action='store_true', default=False,
                    help='use bfloat datatype')
group.add_argument('--mesa',  type=float, default=0.0,
                    help='use memory efficient sharpness optimization, enabled if >0.0')
group.add_argument('--mesa-start-ratio',  type=float, default=0.25,
                    help='when to start MESA, ratio to total training time, def 0.25')

kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda()

def kdloss(y, teacher_scores):
    T = 3
    p = torch.nn.functional.log_softmax(y/T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores/T, dim=1)
    l_kl = 50.0*kl_loss(p, q)
    return l_kl

def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    args.prefetcher = not args.no_prefetcher
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        # torch.cuda.set_device(rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d. Local rank %d'
                     % (args.rank, args.world_size, args.local_rank))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")

    utils.random_seed(args.seed, args.rank)

    # Load HSI data first to get num_classes
    if args.dataset == 'hsi':
        print("Load and preprocess hyperspectral dataset name")
        print("Loading hyperspectral data...")

        train_loader, eval_loader, num_classes, dataset_name = get_hsi_loaders(args)
        args.num_classes = num_classes

        print(f"Dataset: {dataset_name}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Test samples: {len(eval_loader.dataset)}")
        print(f"Total samples: {len(train_loader.dataset) + len(eval_loader.dataset)}")
        print(f"PCA components: {args.hsi_bands}")
        print(f"HSI dataset loaded: {num_classes} classes")
        if args.use_kan:
            _logger.warning("--use-kan is disabled for HSI training to prevent numerical instability.")
        args.use_kan = False

    if args.dataset == 'hsi':
        # Create MambaVision model directly for hyperspectral data
        drop_path_rate = args.drop_path if args.drop_path is not None else 0.0
        
        # Stage ablation: modify depths based on abl_stages
        depths = None
        if args.abl_stages == '012':
            # Remove last stage (keep levels 0, 1, 2)
            if args.model == 'mamba_vision_T':
                depths = [1, 3, 8]  # Original: [1, 3, 8, 4]
            elif args.model == 'mamba_vision_S':
                depths = [3, 3, 7]  # Original: [3, 3, 7, 5]
            elif args.model == 'mamba_vision_B':
                depths = [3, 3, 10]  # Original: [3, 3, 10, 5]
        elif args.abl_stages == '01':
            # Remove last two stages (keep levels 0, 1)
            if args.model == 'mamba_vision_T':
                depths = [1, 3]  # Original: [1, 3, 8, 4]
            elif args.model == 'mamba_vision_S':
                depths = [3, 3]  # Original: [3, 3, 7, 5]
            elif args.model == 'mamba_vision_B':
                depths = [3, 3]  # Original: [3, 3, 10, 5]
        elif args.abl_stages == '0':
            # Only first stage
            if args.model == 'mamba_vision_T':
                depths = [1]  # Original: [1, 3, 8, 4]
            elif args.model == 'mamba_vision_S':
                depths = [3]  # Original: [3, 3, 7, 5]
            elif args.model == 'mamba_vision_B':
                depths = [3]  # Original: [3, 3, 10, 5]
        # else: depths = None (use model default)
        
        # Get prompt_mode (default to 'full' if not using prompts)
        prompt_mode = args.prompt_mode if args.use_prompt else 'full'
        
        # Debug: Print depths being used
        if depths is not None:
            print(f"Using custom depths for stage ablation: {depths}")
            print(f"Number of stages: {len(depths)}")
        
        if args.model == 'mamba_vision_T':
            model = mamba_vision_T(
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                use_hsi_adaptor=True,
                hsi_bands=args.hsi_bands,
                drop_path_rate=drop_path_rate,
                use_kan=args.use_kan,
                use_prompt=args.use_prompt,
                task_classes=args.task_classes,
                prompt_inject_levels=args.prompt_inject_levels,
                prompt_mode=prompt_mode,
                depths=depths,  # Pass custom depths for stage ablation
            )
        elif args.model == 'mamba_vision_S':
            model = mamba_vision_S(
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                use_hsi_adaptor=True,
                hsi_bands=args.hsi_bands,
                drop_path_rate=drop_path_rate,
                use_kan=args.use_kan,
                use_prompt=args.use_prompt,
                task_classes=args.task_classes,
                prompt_inject_levels=args.prompt_inject_levels,
                prompt_mode=prompt_mode,
                depths=depths,  # Pass custom depths for stage ablation
            )
        elif args.model == 'mamba_vision_B':
            model = mamba_vision_B(
                pretrained=args.pretrained,
                num_classes=args.num_classes,
                use_hsi_adaptor=True,
                hsi_bands=args.hsi_bands,
                drop_path_rate=drop_path_rate,
                use_kan=args.use_kan,
                use_prompt=args.use_prompt,
                task_classes=args.task_classes,
                prompt_inject_levels=args.prompt_inject_levels,
                prompt_mode=prompt_mode,
                depths=depths,  # Pass custom depths for stage ablation
            )
        else:
            raise ValueError(f"Unknown model for HSI: {args.model}")

        # Print model efficiency analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model efficiency analysis:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")

        # Load prompt weights if requested
        if args.use_prompt and args.load_prompt_weights:
            print("\nLoading pretrained prompt weights from checkpoints directory...")
            model.load_prompt_weights(verbose=True)

    else:
        model = create_model(
            args.model,
            pretrained=args.pretrained,
            num_classes=args.num_classes,
            global_pool=args.gp,
            bn_momentum=args.bn_momentum,
            bn_eps=args.bn_eps,
            scriptable=args.torchscript,
            checkpoint_path=args.initial_checkpoint,
            attn_drop_rate=args.attn_drop_rate,
            drop_rate=args.drop_rate,
            drop_path_rate=args.drop_path,
            use_kan=args.use_kan)

    if args.bfloat:
        args.dtype = torch.bfloat16
    else:
        args.dtype = torch.float16

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.dataset == 'hsi':
        # For HSI data, use HSI-specific configuration
        data_config = {
            'input_size': (args.hsi_bands, args.hsi_patch, args.hsi_patch),
            'interpolation': 'bicubic',
            'mean': (0.0,) * args.hsi_bands,  # No normalization for HSI
            'std': (1.0,) * args.hsi_bands,   # No normalization for HSI
            'crop_pct': 1.0,
            'crop_mode': 'center'
        }
        if args.local_rank == 0:
            print("HSI data processing configuration:")
            for key, value in data_config.items():
                print(f"\t{key}: {value}")
    else:
        data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    print("filter_bias_and_bn")
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    class DummyAutocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    amp_autocast = DummyAutocast  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        # Use new torch.amp.autocast API to avoid deprecation warning
        if hasattr(torch.amp, 'autocast'):
            # PyTorch 2.0+ - use new API
            def amp_autocast_wrapper(**kwargs):
                return torch.amp.autocast('cuda', **kwargs)
            amp_autocast = amp_autocast_wrapper
        else:
            # Fallback for older PyTorch versions
            amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None

    if not os.path.isfile(args.resume):
        args.resume = ""

    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    if args.loadcheckpoint:
        _logger.info(f"Loading checkpoint {args.loadcheckpoint}, checking for existing parameters if their shape match")
        new_model_weights = torch.load(args.loadcheckpoint)["state_dict"]
        current_model = model.state_dict()

        new_state_dict = OrderedDict()
        for k in current_model.keys():
            if k in new_model_weights.keys():
                if new_model_weights[k].size() == current_model[k].size():
                    print(f"loading weights {k} {new_model_weights[k].size()}")
                    new_state_dict[k] = new_model_weights[k]

        model.load_state_dict(new_state_dict, strict=False)
        model_ema.module.load_state_dict(new_state_dict, strict=False)

    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    if args.dataset == 'hsi':
        # HSI loaders already created above
        dataset_train, dataset_eval = None, None
    elif args.lmdb_dataset:
        train_dir = os.path.join(args.data_dir, 'train')

        if 'lmdb' in args.data_dir:
            dataset_train = imagenet_lmdb_dataset(
                train_dir, transform=None)
        else:
            if not os.path.exists(train_dir):
                _logger.error('Training folder does not exist at: {}'.format(train_dir))
                exit(1)
            dataset_train = ImageDataset(train_dir)

        eval_dir = os.path.join(args.data_dir, 'val')
        if 'lmdb' in args.data_dir:
            dataset_eval = imagenet_lmdb_dataset(
                eval_dir, transform=None)
        else:
            if not os.path.isdir(eval_dir):
                eval_dir = os.path.join(args.data_dir, 'validation')
                if not os.path.isdir(eval_dir):
                    _logger.error('Validation folder does not exist at: {}'.format(eval_dir))
                    exit(1)
            dataset_eval = ImageDataset(eval_dir)
    else:
        dataset_train = create_dataset(
            args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size,
            repeats=args.epoch_repeats)

        dataset_eval = create_dataset(
            args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
            class_map=args.class_map,
            download=args.dataset_download,
            batch_size=args.batch_size)

    collate_fn = None
    mixup_fn = None
    mixup_active = (args.dataset != 'hsi') and (args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None)
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    if args.dataset == 'hsi':
        # Use HSI loaders directly
        loader_train = train_loader
        loader_eval = eval_loader
    else:
        train_interpolation = args.train_interpolation
        if args.no_aug or not train_interpolation:
            train_interpolation = data_config['interpolation']
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            batch_size=args.batch_size,
            is_training=True,
            use_prefetcher=args.prefetcher,
            no_aug=args.no_aug,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_repeats=args.aug_repeats,
            num_aug_splits=num_aug_splits,
            interpolation=train_interpolation,
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            collate_fn=collate_fn,
            pin_memory=args.pin_mem,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            worker_seeding=args.worker_seeding,
        )
        loader_eval = create_loader(
            dataset_eval,
            input_size=data_config['input_size'],
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            use_prefetcher=args.prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'],
            std=data_config['std'],
            num_workers=args.workers,
            distributed=args.distributed,
            crop_pct=data_config['crop_pct'],
            pin_memory=args.pin_mem,
        )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    higher_is_better = eval_metric != 'loss'
    saver = None
    output_dir = None
    if args.rank == 0:
        log_dir = args.log_dir  + '_' + args.tag
        os.makedirs(log_dir, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=log_dir)
    else:
        log_writer = None

    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
            args.experiment = exp_name

        output_dir = utils.get_outdir(args.output if args.output else f'./output/train/{args.tag}/', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=1)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        if 1: #args.copy_code
            # copy .py files
            files = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if
                     os.path.splitext(f)[1] == '.py']
            for f in files:
                if "/code_copy/" in f: continue
                new_path = output_dir + "/code_copy/" + f
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.copyfile(f, new_path)

    if args.validate_only:
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
        print(eval_metrics)
        exit()

    test_acc_track=[]
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            saver = saver if not args.no_saver else None
            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)
            # print(train_metrics.keys())

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)
            if log_writer is not None:
                # structured TB logging
                log_writer.update(loss=train_metrics['loss'], head="train", step=epoch)
                log_writer.update(loss=eval_metrics['loss'], head="val", step=epoch)
                log_writer.update(top1=eval_metrics['top1'], head="val", step=epoch)
                log_writer.update(top5=eval_metrics['top5'], head="val", step=epoch)
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                log_writer.update(lr=lr, head="train", step=epoch)

            test_acc_track.append(eval_metrics['top1'])
            stopif = True if len(test_acc_track)>1 and test_acc_track[-1]<1.0 else False

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if log_writer is not None:
                # also log EMA metrics under val_ema
                log_writer.update(top1=eval_metrics['top1'], head="val_ema", step=epoch)
                log_writer.update(loss=eval_metrics['loss'], head="val_ema", step=epoch)

            if lr_scheduler is not None and args.lr_ep:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, None if eval_metrics is None else eval_metrics[eval_metric])

            if output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            # compute improvement and print clear epoch summary
            current_metric = None if eval_metrics is None else eval_metrics[eval_metric]
            prev_best = best_metric
            improved = False
            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=current_metric)
                if prev_best is None and best_metric is not None:
                    improved = True
                elif prev_best is not None and current_metric is not None:
                    if higher_is_better:
                        improved = current_metric > prev_best
                    else:
                        improved = current_metric < prev_best

            # Pretty per-epoch summary
            if args.local_rank == 0:
                metric_name = 'Acc@1' if eval_metric == 'top1' else eval_metric
                best_str = 'None' if best_metric is None else f"{best_metric:.4f} (epoch {best_epoch})"
                last_str = 'None' if current_metric is None else f"{current_metric:.4f}"
                delta_str = ''
                if improved and prev_best is not None and current_metric is not None:
                    sign = '+' if (higher_is_better) else '-'
                    delta = abs(current_metric - prev_best)
                    delta_str = f" {sign}{delta:.4f}"
                status = 'NEW BEST' if improved else 'no improvement'
                _logger.info(
                    f"Epoch {epoch:03d} Summary | Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {eval_metrics['loss']:.4f} | Val Top1: {eval_metrics['top1']:.4f} | "
                    f"Val Top5: {eval_metrics['top5']:.4f} | {metric_name} last: {last_str} | "
                    f"best: {best_str} | {status}{delta_str}")

                if log_writer is not None and current_metric is not None and best_metric is not None:
                    log_writer.update(is_best=1.0 if improved else 0.0, head="val", step=epoch)
                    if eval_metric == 'top1':
                        log_writer.update(best_top1=best_metric, head="val", step=epoch)
                    elif eval_metric == 'loss':
                        log_writer.update(best_loss=best_metric, head="val", step=epoch)


            if not np.isfinite(eval_metrics['loss']) or stopif:
                # if got None then exit
                if args.local_rank == 0:
                    _logger.info("Nan in loss, exit")
                    _logger.error("Nan in loss, exit")

                    input, target = next(iter(loader_eval))
                    input = input.cuda()

                    with torch.autograd.detect_anomaly():
                        with amp_autocast(dtype=args.dtype):
                            output = model(input)

                    print(output)

                exit(1)
                return 0

    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    # Generate classification report and maps for HSI datasets
    if args.dataset == 'hsi' and args.local_rank == 0 and output_dir is not None:
        _logger.info('Generating comprehensive classification report and maps...')

        # Load best model if available
        if best_epoch is not None and saver is not None:
            best_checkpoint_path = os.path.join(output_dir, f'model_best.pth.tar')
            if os.path.exists(best_checkpoint_path):
                _logger.info(f'Loading best model from epoch {best_epoch} for final evaluation...')
                # Use weights_only=False since checkpoints contain argparse.Namespace and other metadata
                checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
                # Handle DDP wrapped models
                model_to_load = model.module if hasattr(model, 'module') else model
                if 'state_dict' in checkpoint:
                    model_to_load.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model' in checkpoint:
                    model_to_load.load_state_dict(checkpoint['model'], strict=False)
                model.eval()

        # Collect predictions from validation set
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args,
                               amp_autocast=amp_autocast, collect_predictions=True)

        if 'predictions' in eval_metrics and 'targets' in eval_metrics:
            y_pred = eval_metrics['predictions']
            y_true = eval_metrics['targets']

            # Generate classification report
            classification, oa, confusion, each_acc, aa, kappa = acc_reports(
                y_true, y_pred, num_classes=args.num_classes)

            # Save classification report
            report_file = os.path.join(output_dir, 'classification_report.txt')
            with open(report_file, 'w') as f:
                f.write('='*80 + '\n')
                f.write('Classification Report\n')
                f.write('='*80 + '\n\n')
                f.write(f'Model: {args.model}\n')
                f.write(f'Dataset: {args.dataset}\n')
                f.write(f'Best Epoch: {best_epoch}\n')
                f.write(f'Best Metric: {best_metric:.4f}\n\n')
                f.write('Performance Metrics:\n')
                f.write(f'Overall Accuracy (OA): {oa:.2f}%\n')
                f.write(f'Average Accuracy (AA): {aa:.2f}%\n')
                f.write(f'Kappa Score: {kappa:.2f}%\n\n')
                f.write('Per-Class Accuracies:\n')
                for i, acc in enumerate(each_acc):
                    f.write(f'Class_{i}: {acc:.2f}%\n')
                f.write('\n' + '='*80 + '\n')
                f.write('Detailed Classification Report:\n')
                f.write('='*80 + '\n')
                f.write(classification)
                f.write('\n' + '='*80 + '\n')
                f.write('Confusion Matrix:\n')
                f.write('='*80 + '\n')
                f.write(str(confusion))
                f.write('\n')

            _logger.info(f'Classification report saved to: {report_file}')
            _logger.info(f'Overall Accuracy: {oa:.2f}% | Average Accuracy: {aa:.2f}% | Kappa: {kappa:.2f}%')

            # Generate classification maps for HSI
            try:
                # Load full HSI dataset for map generation
                # Note: get_hsi_loaders is already imported at the top, don't re-import it
                from utils.datasets import pad_with_zeros, create_image_cubes, apply_pca, HyperspectralDataset

                # Get data path
                data_root = args.data_dir or getattr(args, 'data', None) or os.path.join(os.path.dirname(__file__), 'Data')
                data_root = os.path.abspath(data_root)
                if not os.path.isdir(data_root):
                    # default to mambavision Data folder
                    mambavision_root = os.path.abspath(os.path.dirname(__file__))
                    data_root = os.path.join(mambavision_root, 'Data')

                # Discover HSI files
                from utils.datasets import _discover_hsi_in_dir
                preferred_dataset = getattr(args, 'hsi_dataset', None)
                dp, lp, dkey, lkey = _discover_hsi_in_dir(data_root, preferred_dataset)

                # Load full dataset
                X = sio.loadmat(dp)[dkey]
                y_all = sio.loadmat(lp)[lkey]
                if y_all.ndim > 2:
                    y_all = y_all.squeeze()

                # Apply PCA if needed
                target_bands = getattr(args, 'hsi_bands', 15)
                if X.shape[2] != target_bands:
                    X = apply_pca(X, num_components=target_bands)

                # Create patches for all pixels
                window = getattr(args, 'hsi_patch', 15)
                patches_all, labels_all = create_image_cubes(X, y_all, window_size=window, remove_zero_labels=True)

                # Convert to [N, C, H, W]
                patches_all = np.transpose(patches_all, (0, 3, 1, 2))

                # Create dataset and loader for all data
                all_set = HyperspectralDataset(patches_all, labels_all, per_sample_norm=True)
                all_loader = torch.utils.data.DataLoader(
                    all_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=min(4, os.cpu_count() or 2),
                    drop_last=False,
                )

                # Generate classification maps using get_cls_map.py
                # Use the same device as the model
                device = next(model.parameters()).device
                # Convert y_all back to original format (add 1 to convert from 0-based to 1-based)
                y_all_original = y_all.copy()
                generate_classification_maps_with_prompts(model, all_loader, y_all_original, args, output_dir, device)

            except Exception as e:
                _logger.warning(f'Could not generate classification maps: {e}')
                import traceback
                traceback.print_exc()


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    num_iters = len(loader)
    display_first = True

    if args.ampere_sparsity:
        model.enforce_mask()

    for batch_idx, (input, target) in enumerate(loader):

        if lr_scheduler is not None and not args.lr_ep:
            lr_scheduler.step_update(num_updates=(epoch * num_iters) + batch_idx + 1)

        if (batch_idx == 0) or (batch_idx % 50 == 0):
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)
        if not args.prefetcher or args.dataset == 'hsi':
            input, target = input.cuda(), target.cuda()
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast(dtype=args.dtype):
            # For HSI classification with prompts, task_id should be based on degradation type
            # If not provided, use classification label or default to 0
            # For now, we'll use default task_id=0 (can be made configurable)
            if args.use_prompt and args.dataset == 'hsi':
                # Option 1: Use classification label as task_id (if task aligns with class)
                # task_id = target.clone()  # Uncomment if tasks align with classes
                # Option 2: Use default task_id (Gaussian noise = 0)
                task_id = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                output = model(input, task_id=task_id)
            else:
                output = model(input)
            loss = loss_fn(output, target)

            if args.mesa>0.0:
                if epoch/args.epochs > args.mesa_start_ratio:
                    with torch.no_grad():
                        ema_output = model_ema.module(input).data.detach()
                    kd = kdloss(output, ema_output)
                    loss += args.mesa * kd

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))

            # Early convergence warning mechanism (based on loss)
            if args.early_conv_warning:
                batch_progress = batch_idx / len(loader)
                current_loss = loss.item()

                # Check if loss is very low (indicating potential overfitting)
                if (current_loss < 0.01 and batch_progress > args.early_conv_min_batch):
                    print(f"\n⚠️  EARLY CONVERGENCE WARNING ⚠️")
                    print(f"   Current loss: {current_loss:.6f}")
                    print(f"   Loss threshold: 0.01")
                    print(f"   Batch progress: {batch_progress:.2%}")
                    print(f"   Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(loader)}")
                    print(f"   Consider:")
                    print(f"   - Reducing learning rate")
                    print(f"   - Increasing regularization (dropout, weight decay)")
                    print(f"   - Checking for overfitting")
                    print(f"   - Early stopping if validation accuracy plateaus")
                    print("=" * 60)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            if args.ampere_sparsity:
                model.enforce_mask(grad=True)

            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)

            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:

            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

                # Early convergence warning mechanism (based on loss) for distributed training
                if args.early_conv_warning and args.local_rank == 0:
                    batch_progress = batch_idx / len(loader)
                    current_loss = reduced_loss.item()

                    # Check if loss is very low (indicating potential overfitting)
                    if (current_loss < 0.01 and batch_progress > args.early_conv_min_batch):
                        print(f"\n⚠️  EARLY CONVERGENCE WARNING ⚠️")
                        print(f"   Current loss: {current_loss:.6f}")
                        print(f"   Loss threshold: 0.01")
                        print(f"   Batch progress: {batch_progress:.2%}")
                        print(f"   Epoch: {epoch+1}, Batch: {batch_idx+1}/{len(loader)}")
                        print(f"   Consider:")
                        print(f"   - Reducing learning rate")
                        print(f"   - Increasing regularization (dropout, weight decay)")
                        print(f"   - Checking for overfitting")
                        print(f"   - Early stopping if validation accuracy plateaus")
                        print("=" * 60)

            if args.local_rank == 0:
                _logger.info(
                    'Epoch: {} [{:>4d}/{}]  '
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})  '
                    'Time: {batch_time.val:.2f}s  '
                    'LR: {lr:.2e}'.format(
                        epoch,
                        batch_idx, len(loader),
                        loss=losses_m,
                        batch_time=batch_time_m,
                        lr=lr))

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None and args.lr_ep:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def AA_andEachClassAccuracy(confusion_matrix):
    """Calculate average accuracy and per-class accuracy from confusion matrix."""
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test, num_classes=None):
    """Generate comprehensive classification report similar to enhanced_training.py."""
    # Determine class names based on number of classes
    if num_classes is None:
        num_classes = max(y_test.max(), y_pred_test.max()) + 1

    # Generate generic class names if needed
    target_names = [f'Class_{i}' for i in range(num_classes)]

    classification = classification_report(y_test, y_pred_test, digits=4,
                                         target_names=target_names, zero_division=0)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100


def generate_classification_maps_with_prompts(model, all_loader, y_all, args, output_dir, device):
    """Wrapper around get_cls_map to handle prompts and custom output directory."""
    if get_cls_map is None:
        _logger.warning("get_cls_map not available, skipping classification map generation")
        return None

    # Create a wrapper model that handles prompts
    use_prompt = args.use_prompt
    dataset = args.dataset

    class ModelWrapper(nn.Module):
        def __init__(self, base_model, use_prompt=False, dataset='hsi'):
            super().__init__()
            self.base_model = base_model
            self.use_prompt = use_prompt
            self.dataset = dataset

        def forward(self, x):
            if self.use_prompt and self.dataset == 'hsi':
                task_id = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
                output = self.base_model(x, task_id=task_id)
            else:
                output = self.base_model(x)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output

    # Wrap model to handle prompts
    wrapped_model = ModelWrapper(model, use_prompt=use_prompt, dataset=dataset)
    wrapped_model.eval()

    # Change to output directory temporarily to save maps there
    original_cwd = os.getcwd()
    maps_dir = os.path.join(output_dir, 'classification_maps')
    os.makedirs(maps_dir, exist_ok=True)

    try:
        # Change to maps directory so get_cls_map saves there
        os.chdir(maps_dir)

        # Call get_cls_map from the imported module
        cls_labels = get_cls_map(wrapped_model, device, all_loader, y_all)

        # Rename files to more generic names
        if os.path.exists('prediction_hong.png'):
            os.rename('prediction_hong.png', 'prediction_map.png')
        if os.path.exists('ground_truth-hong.png'):
            os.rename('ground_truth-hong.png', 'ground_truth_map.png')
        if os.path.exists('difference_map_hong.png'):
            os.rename('difference_map_hong.png', 'difference_map.png')

        _logger.info(f'Classification maps saved to: {maps_dir}')
        return cls_labels
    finally:
        os.chdir(original_cwd)


def validate(model, loader, loss_fn, args, amp_autocast=suppress, log_suffix='', collect_predictions=False):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    # For collecting predictions and targets for classification report
    all_predictions = []
    all_targets = []

    model.eval()

    if args.ampere_sparsity:
        model.enforce_mask()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher or args.dataset == 'hsi':
                input = input.cuda()
                target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast(dtype=args.dtype):
                # For HSI classification with prompts, provide task_id
                if args.use_prompt and args.dataset == 'hsi':
                    task_id = torch.zeros(input.size(0), dtype=torch.long, device=input.device)
                    output = model(input, task_id=task_id)
                else:
                    output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            # Collect predictions and targets for classification report
            if collect_predictions:
                pred = output.argmax(dim=1).cpu().numpy()
                targ = target.cpu().numpy()
                all_predictions.extend(pred)
                all_targets.extend(targ)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Val' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Loss: {loss.val:.4f} ({loss.avg:.4f})  '
                    'Acc: {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                        log_name, batch_idx, last_idx,
                        loss=losses_m, top1=top1_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
    if collect_predictions:
        metrics['predictions'] = np.array(all_predictions)
        metrics['targets'] = np.array(all_targets)
    return metrics

if __name__ == '__main__':
    main()
