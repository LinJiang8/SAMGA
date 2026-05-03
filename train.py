import os
import argparse
import logging
from datetime import datetime
import json
import random
import time
import sys
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from module.dataset import EEGPreImageDataset
from module.eeg_encoder.atm.atm import ATMS
from module.eeg_encoder.model import EEGNet, EEGProject, TSConv, EEGTransformer
from module.loss import ContrastiveLoss, mmd_rbf
from module.util import retrieve_all
from module.projector import *
from module.eeg_augmentation import RandomTimeShift, RandomGaussianNoise, RandomChannelDropout, RandomSmooth


class SubjectAwareLayerMixer(nn.Module):
    def __init__(
        self,
        layer_ids,
        num_subjects: int,
        prior_center: int = 28,
        prior_strength: float = 1.0,
        temperature: float = 1.0,
        subject_dropout: float = 0.3,
    ):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.num_layers = len(self.layer_ids)
        self.temperature = temperature
        self.subject_dropout = subject_dropout

        layer_ids_tensor = torch.tensor(self.layer_ids, dtype=torch.float32)
        self.register_buffer("layer_ids_tensor", layer_ids_tensor)

        if len(self.layer_ids) > 1:
            sorted_ids = sorted(self.layer_ids)
            diffs = [sorted_ids[i + 1] - sorted_ids[i] for i in range(len(sorted_ids) - 1)]
            positive_diffs = [d for d in diffs if d > 0]
            step = float(min(positive_diffs)) if len(positive_diffs) > 0 else 1.0
        else:
            step = 1.0

        dist = torch.abs(layer_ids_tensor - float(prior_center)) / step
        init_logits = -prior_strength * dist

        self.global_logits = nn.Parameter(init_logits.clone())
        self.subject_bias = nn.Embedding(num_subjects, self.num_layers)
        nn.init.zeros_(self.subject_bias.weight)

    def forward(self, subject_ids: torch.Tensor = None, force_global: bool = False):
        if subject_ids is None:
            logits = self.global_logits.unsqueeze(0)
        else:
            bsz = subject_ids.shape[0]
            logits = self.global_logits.unsqueeze(0).expand(bsz, -1)
            if not force_global:
                bias = self.subject_bias(subject_ids.long())
                if self.training and self.subject_dropout > 0:
                    keep_mask = (
                        torch.rand(bsz, 1, device=subject_ids.device) > self.subject_dropout
                    ).float()
                    bias = bias * keep_mask
                logits = logits + bias
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return weights

    def get_global_weights(self):
        return torch.softmax(self.global_logits / self.temperature, dim=-1)


# Set the random seed. If not provided, generate a new one based on the current time.
def seed_everything(seed: int = None):
    if seed is None:
        seed = int(time.time()) % (2**32 - 1)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[seed_everything] All seeds set to: {seed}")
    return seed


def prepare_multilayer_feature_dir(src_dir, layer_ids, cache_root, log_fn=print):
    """
    Converts folders containing per-layer files like:
      image_train_layer20.npy, ..., image_test_layer36.npy
    into a temporary directory with dataset-compatible files:
      image_train.npy, image_test.npy
    Each output has shape [Nobj, Nimg, K, D].
    """
    os.makedirs(cache_root, exist_ok=True)
    cache_name = "stacked_" + "_".join(map(str, layer_ids))
    dst_dir = os.path.join(cache_root, cache_name)
    os.makedirs(dst_dir, exist_ok=True)

    def build_split(split):
        dst_file = os.path.join(dst_dir, f"image_{split}.npy")
        if os.path.exists(dst_file):
            log_fn(f"Use cached multilayer feature: {dst_file}")
            arr = np.load(dst_file, mmap_mode='r')
            log_fn(f"Cached multilayer {split} shape: {arr.shape}")
            return

        layer_arrays = []
        for lid in layer_ids:
            candidate = os.path.join(src_dir, f"image_{split}_layer{lid}.npy")
            if not os.path.exists(candidate):
                raise FileNotFoundError(
                    f"Missing multilayer feature file: {candidate}. "
                    f"Your folder must contain files like image_{split}_layer20.npy"
                )
            arr = np.load(candidate)
            layer_arrays.append(arr)
            log_fn(f"Loaded {candidate}, shape={arr.shape}")

        base_shape = layer_arrays[0].shape
        for lid, arr in zip(layer_ids, layer_arrays):
            if arr.shape != base_shape:
                raise RuntimeError(
                    f"Layer feature shape mismatch at layer {lid}: "
                    f"expected {base_shape}, got {arr.shape}"
                )

        stacked = np.stack(layer_arrays, axis=2)  # [Nobj, Nimg, K, D]
        np.save(dst_file, stacked)
        log_fn(f"Saved multilayer {split} feature to {dst_file}, shape={stacked.shape}")

    build_split("train")
    build_split("test")
    return dst_dir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0', type=str, help='training device')
    parser.add_argument('--num_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--output_dir', default='./result', type=str)
    parser.add_argument('--output_name', default=None, type=str)
    parser.add_argument('--train_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--test_subject_ids', default=[8], nargs='+', type=int)
    parser.add_argument('--data_average', action='store_true')
    parser.add_argument('--data_random', action='store_true')
    parser.add_argument('--init_temperature', default=0.07, type=float)
    parser.add_argument('--t_learnable', action='store_true')
    parser.add_argument('--softplus', action='store_true')
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--img_l2norm', action='store_true')
    parser.add_argument('--text_l2norm', action='store_true')
    parser.add_argument('--eeg_l2norm', action='store_true')
    parser.add_argument('--eeg_data_dir', default='./things_eeg/data/preprocessed_eeg', type=str, help='where your EEG data are')
    parser.add_argument("--selected_channels", default=[], nargs='*', type=str, help="selected EEG channels, empty means all channels")
    parser.add_argument('--time_window', type=int, default=[0, 250], nargs=2, help='time window for EEG data, in sample points')
    parser.add_argument('--eeg_aug', action='store_true')
    parser.add_argument('--eeg_aug_type', type=str, choices=['noise', 'time_shift', 'channel_dropout', 'smooth'], default='noise', help='eeg augmentation type')
    parser.add_argument('--eeg_encoder_type', type=str, choices=['ATM', 'EEGNet', 'EEGProject', 'TSConv', 'EEGTransformer', 'MultiBandPrior'], default='EEGProject')
    parser.add_argument('--image_aug', action='store_true')
    parser.add_argument('--image_test_aug', action='store_true')
    parser.add_argument('--eeg_test_aug', action='store_true')
    parser.add_argument('--frozen_eeg_prior', action='store_true', help='whether to use frozen eeg prior')

    parser.add_argument('--projector', type=str, choices=['direct', 'linear', 'mlp'], default='direct')
    parser.add_argument('--feature_dim', type=int, default=512, help='dont work when direct')
    parser.add_argument('--eeg_feature_dim', type=int, default=1024, help='raw EEG encoder output dim before projector')
    parser.add_argument('--image_mid_dim', type=int, default=1024, help='image intermediate dim before final projector')

    parser.add_argument('--image_feature_dir', default='./data/things_eeg/image_feature/RN50', type=str, help='where your image feature are')
    parser.add_argument('--aug_image_feature_dirs', default=[], nargs='+', type=str, help='where your augmentation image feature are')
    parser.add_argument('--text_feature_dir', default='./data/things_eeg/text_feature/BLIP2', type=str, help='where your text feature are')

    parser.add_argument('--save_weights', action='store_true', help='whether to save model weights')
    parser.add_argument('--stage1_epochs', default=20, type=int, help='epochs for stage-1 alignment')
    parser.add_argument('--stage2_learning_rate', default=5e-5, type=float, help='learning rate after stage-1')
    parser.add_argument('--stage1_mmd_start', default=0.9, type=float, help='initial MMD weight in stage-1')
    parser.add_argument('--stage1_mmd_end', default=0.2, type=float, help='final MMD weight at the end of stage-1')
    parser.add_argument('--early_stop_patience', default=10, type=int, help='early stop if test Top-1 does not improve for N epochs after stage-1; <=0 disables')
    parser.add_argument('--early_stop_min_delta', default=0.0, type=float, help='minimum Top-1 improvement in percentage points to reset early stopping')
    parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility')
    parser.add_argument('--use_multilayer_router', action='store_true')
    parser.add_argument('--layer_ids', default=[20, 24, 28, 32, 36], nargs='+', type=int)
    parser.add_argument('--router_temperature', type=float, default=1.0)
    parser.add_argument('--router_subject_dropout', type=float, default=0.3)
    parser.add_argument('--router_layer_dropout', type=float, default=0.0)
    parser.add_argument('--router_eval_mode', type=str, choices=['global', 'subject'], default='global')
    parser.add_argument('--layer_prior_center', type=int, default=28)
    parser.add_argument('--layer_prior_strength', type=float, default=1.0)

    args = parser.parse_args()
    seed = seed_everything(seed=args.seed)

    if args.output_name is not None:
        log_dir = os.path.join(args.output_dir, f"{datetime.now().strftime(r'%Y%m%d-%H%M%S')}-{args.output_name}")
    else:
        log_dir = os.path.join(args.output_dir, datetime.now().strftime(r'%Y%m%d-%H%M%S'))

    log_dir_suffix = '-'.join(log_dir.split('-')[2:])
    if os.path.exists(args.output_dir):
        for existing_dir in os.listdir(args.output_dir):
            if existing_dir.endswith(log_dir_suffix):
                if os.path.exists(os.path.join(args.output_dir, existing_dir, 'result.csv')):
                    print(f"Experiment with the same name '{log_dir_suffix}' already exists. Exiting to avoid overwriting.")
                    sys.exit(0)
                else:
                    shutil.rmtree(os.path.join(args.output_dir, existing_dir))
                    print(f"Removed incomplete experiment directory '{existing_dir}' to avoid conflicts.")

    writer = SummaryWriter(log_dir=log_dir)

    args_dict = vars(args)
    with open(os.path.join(writer.log_dir, 'train_config.json'), 'w') as f:
        json.dump(args_dict, f, indent=4)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        encoding='utf-8',
        filename=f'{writer.log_dir}/train.log',
        filemode='w'
    )

    def log(s):
        logging.info(s)
        print(s)

    log('Input arguments:')
    for key, val in vars(args).items():
        log(f'{key:22} {val}')

    with open(os.path.join(args.output_dir, 'last_run.txt'), 'w') as f:
        f.write(writer.log_dir)

    print('')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f'Using device: {device}')

    def get_stage1_weights(epoch, stage1_epochs, mmd_start, mmd_end):
        if stage1_epochs <= 1:
            mmd_w = mmd_end
        else:
            progress = (epoch - 1) / (stage1_epochs - 1)
            progress = max(0.0, min(1.0, progress))
            mmd_w = mmd_start + (mmd_end - mmd_start) * progress
        contrast_w = 1.0 - mmd_w
        return mmd_w, contrast_w

    print('\n>>> Loading Train Data <<<')
    if args.eeg_aug:
        if args.eeg_aug_type == 'noise':
            eeg_transform = RandomGaussianNoise(std=0.001)
        elif args.eeg_aug_type == 'time_shift':
            eeg_transform = RandomTimeShift(max_shift=5)
        elif args.eeg_aug_type == 'channel_dropout':
            eeg_transform = RandomChannelDropout(drop_prob=0.1)
        elif args.eeg_aug_type == 'smooth':
            eeg_transform = RandomSmooth(kernel_size=5, smooth_prob=0.3)
        else:
            raise ValueError(f'Unsupported eeg_aug_type: {args.eeg_aug_type}')
    else:
        eeg_transform = None

    effective_image_feature_dir = args.image_feature_dir
    if args.use_multilayer_router:
        effective_image_feature_dir = prepare_multilayer_feature_dir(
            src_dir=args.image_feature_dir,
            layer_ids=args.layer_ids,
            cache_root=os.path.join(writer.log_dir, 'multilayer_cache'),
            log_fn=log,
        )
        log(f'Effective multilayer image_feature_dir: {effective_image_feature_dir}')

    aug_img_type = ['GaussianBlur', 'GaussianNoise', 'LowResolution', 'Mosaic']
    aug_image_feature_dirs = [os.path.join(effective_image_feature_dir, aug_type) for aug_type in aug_img_type]

    train_dataset = EEGPreImageDataset(
        args.train_subject_ids,
        args.eeg_data_dir,
        args.selected_channels,
        args.time_window,
        effective_image_feature_dir,
        args.text_feature_dir,
        args.image_aug,
        aug_image_feature_dirs,
        args.data_average,
        args.data_random,
        eeg_transform,
        True,
        args.image_test_aug,
        args.eeg_test_aug,
        args.frozen_eeg_prior,
    )

    eeg_sample_points = train_dataset.num_sample_points
    log(f'EEG sample points: {eeg_sample_points}')
    image_feature_dim = train_dataset.image_features.shape[-1]
    text_feature_dim = train_dataset.text_features.shape[-1] if hasattr(train_dataset, 'text_features') else image_feature_dim
    eeg_feature_dim = args.eeg_feature_dim
    log(f'image raw feature dimension: {image_feature_dim}')
    log(f'text raw feature dimension: {text_feature_dim}')
    log(f'eeg raw feature dimension: {eeg_feature_dim}')
    log(f'image intermediate dimension: {args.image_mid_dim}')
    log(f'final alignment dimension: {args.feature_dim}')
    log(f'train image_features shape: {tuple(train_dataset.image_features.shape)}')

    log(f'data length: {len(train_dataset)}')
    channels_num = train_dataset.channels_num
    log(f'number of channels: {channels_num}')

    sample = train_dataset[0]
    log(f'sample eeg shape: {tuple(sample[0].shape)}')
    log(f'sample image_feature shape: {tuple(sample[1].shape)}')
    log(f'sample text_feature shape: {tuple(sample[2].shape)}')

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    print('\n>>> Loading Test Data <<<')
    test_dataset = EEGPreImageDataset(
        args.test_subject_ids,
        args.eeg_data_dir,
        args.selected_channels,
        args.time_window,
        effective_image_feature_dir,
        args.text_feature_dir,
        args.image_aug,
        aug_image_feature_dirs,
        True,
        False,
        eeg_transform,
        False,
        args.image_test_aug,
        args.eeg_test_aug,
        args.frozen_eeg_prior,
    )
    log(f'test image_features shape: {tuple(test_dataset.image_features.shape)}')
    test_sample = test_dataset[0]
    log(f'test sample image_feature shape: {tuple(test_sample[1].shape)}')
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    args_dict = vars(args)
    keys_needed = ['eeg_encoder_type', 'eeg_data_dir', 'image_feature_dir']
    inference_config = {k: args_dict[k] for k in keys_needed}
    inference_config['effective_image_feature_dir'] = effective_image_feature_dir
    inference_config['eeg_sample_points'] = eeg_sample_points
    inference_config['feature_dim'] = eeg_feature_dim
    with open(os.path.join(writer.log_dir, 'evaluate_config.json'), 'w') as f:
        json.dump(inference_config, f, indent=4)

    if args.eeg_encoder_type == 'ATM':
        model = ATMS(feature_dim=eeg_feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGNet':
        model = EEGNet(feature_dim=eeg_feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGProject':
        model = EEGProject(feature_dim=eeg_feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'TSConv':
        model = TSConv(feature_dim=eeg_feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    elif args.eeg_encoder_type == 'EEGTransformer':
        model = EEGTransformer(feature_dim=eeg_feature_dim, eeg_sample_points=eeg_sample_points, channels_num=channels_num)
    else:
        raise ValueError(f'Unsupported eeg_encoder_type in this script: {args.eeg_encoder_type}')

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_m = num_params / 1e6
    log(f'EEG Encoder trainable parameters: {num_params_m:.2f}M')
    log(str(model))

    use_multilayer_router = args.use_multilayer_router 
    if use_multilayer_router:
        if train_dataset.image_features.ndim != 4:
            raise RuntimeError(
                f'Router enabled but dataset image_features should be [Nobj, Nimg, K, D]. '
                f'Got {tuple(train_dataset.image_features.shape)}'
            )
        if train_dataset.image_features.shape[2] != len(args.layer_ids):
            raise RuntimeError(
                f'Router enabled but K={train_dataset.image_features.shape[2]} does not match layer_ids={args.layer_ids}'
            )
        log(f'use_multilayer_router: True, layer_ids={args.layer_ids}')
    else:
        log('use_multilayer_router: False')

    if args.projector == 'direct':
        eeg_projector = ProjectorDirect().to(device)
        img_pre_projector = ProjectorDirect().to(device)
        text_projector = ProjectorDirect().to(device)
        if use_multilayer_router:
            img_projectors = nn.ModuleList([ProjectorDirect().to(device) for _ in range(len(args.layer_ids))])
        else:
            img_projector = ProjectorDirect().to(device)
    elif args.projector == 'linear':
        eeg_projector = ProjectorLinear(eeg_feature_dim, args.feature_dim).to(device)
        img_pre_projector = ProjectorLinear(image_feature_dim, args.image_mid_dim).to(device)
        text_projector = ProjectorLinear(text_feature_dim, args.feature_dim).to(device)
        if use_multilayer_router:
            img_projectors = nn.ModuleList([
                ProjectorLinear(args.image_mid_dim, args.feature_dim).to(device)
                for _ in range(len(args.layer_ids))
            ])
        else:
            img_projector = ProjectorLinear(args.image_mid_dim, args.feature_dim).to(device)
    elif args.projector == 'mlp':
        eeg_projector = ProjectorMLP(eeg_feature_dim, args.feature_dim).to(device)
        img_pre_projector = ProjectorMLP(image_feature_dim, args.image_mid_dim).to(device)
        text_projector = ProjectorMLP(text_feature_dim, args.feature_dim).to(device)
        if use_multilayer_router:
            img_projectors = nn.ModuleList([
                ProjectorMLP(args.image_mid_dim, args.feature_dim).to(device)
                for _ in range(len(args.layer_ids))
            ])
        else:
            img_projector = ProjectorMLP(args.image_mid_dim, args.feature_dim).to(device)
    else:
        raise ValueError(f'Unsupported projector type: {args.projector}')

    share_encoder = ShareEncoder(args.feature_dim, args.feature_dim).to(device)

    if use_multilayer_router:
        num_subjects = max(args.train_subject_ids + args.test_subject_ids) + 1
        layer_router = SubjectAwareLayerMixer(
            layer_ids=args.layer_ids,
            num_subjects=num_subjects,
            prior_center=args.layer_prior_center,
            prior_strength=args.layer_prior_strength,
            temperature=args.router_temperature,
            subject_dropout=args.router_subject_dropout,
        ).to(device)
    else:
        layer_router = None

    projector_param_count = (
        sum(p.numel() for p in eeg_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in img_pre_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in text_projector.parameters() if p.requires_grad)
        + sum(p.numel() for p in share_encoder.parameters() if p.requires_grad)
    )
    if use_multilayer_router:
        projector_param_count += sum(p.numel() for p in img_projectors.parameters() if p.requires_grad)
        projector_param_count += sum(p.numel() for p in layer_router.parameters() if p.requires_grad)
    else:
        projector_param_count += sum(p.numel() for p in img_projector.parameters() if p.requires_grad)
    log(f'Projector trainable parameters: {projector_param_count / 1e6:.2f}M')

    criterion = ContrastiveLoss(
        args.init_temperature,
        args.alpha,
        args.beta,
        args.eeg_l2norm,
        args.img_l2norm,
        args.text_l2norm,
        args.t_learnable,
        args.softplus,
    ).to(device)
    log(str(criterion))

    def build_optimizer(lr, include_share_encoder=True):
        trainable_parameters = (
            list(model.parameters())
            + list(eeg_projector.parameters())
            + list(img_pre_projector.parameters())
            + list(text_projector.parameters())
        )
        if include_share_encoder:
            trainable_parameters += list(share_encoder.parameters())
        if use_multilayer_router:
            trainable_parameters += list(img_projectors.parameters())
            trainable_parameters += list(layer_router.parameters())
        else:
            trainable_parameters += list(img_projector.parameters())
        if args.t_learnable:
            trainable_parameters.extend([p for p in criterion.parameters() if p.requires_grad])
        return optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    optimizer = build_optimizer(args.learning_rate, include_share_encoder=True)

    def apply_layer_dropout(weights, drop_prob):
        if drop_prob <= 0:
            return weights
        keep = (torch.rand_like(weights) > drop_prob).float()
        dead_row = keep.sum(dim=-1, keepdim=True) == 0
        keep = torch.where(dead_row, torch.ones_like(keep), keep)
        weights = weights * keep
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights

    def build_image_teacher(image_feature_batch, subject_id_batch, training=True):
        if not use_multilayer_router:
            image_feature_batch = img_pre_projector(image_feature_batch)
            image_feature_batch = img_projector(image_feature_batch)
            if image_feature_batch.ndim == 3:
                image_feature_batch = image_feature_batch.mean(dim=1)
            elif image_feature_batch.ndim != 2:
                raise RuntimeError(f'Unexpected image feature shape before share encoder: {tuple(image_feature_batch.shape)}')
            image_feature_batch = share_encoder(image_feature_batch)
            return image_feature_batch

        if image_feature_batch.ndim != 3:
            raise RuntimeError(f'Router enabled but batch image_feature should be [B, K, D], got {tuple(image_feature_batch.shape)}')
        if image_feature_batch.shape[1] != len(args.layer_ids):
            raise RuntimeError(f'Batch K={image_feature_batch.shape[1]} does not match layer_ids={args.layer_ids}')

        projected_layers = []
        for i in range(image_feature_batch.shape[1]):
            feat_i = image_feature_batch[:, i, :]
            feat_i = img_pre_projector(feat_i)
            feat_i = img_projectors[i](feat_i)
            projected_layers.append(feat_i)
        projected_layers = torch.stack(projected_layers, dim=1)

        force_global = (not training) and (args.router_eval_mode == 'global')
        layer_weights = layer_router(subject_id_batch, force_global=force_global)
        if training and args.router_layer_dropout > 0:
            layer_weights = apply_layer_dropout(layer_weights, args.router_layer_dropout)

        mixed = torch.sum(projected_layers * layer_weights.unsqueeze(-1), dim=1)
        mixed = share_encoder(mixed)
        return mixed

    # Training process
    model.train()
    eeg_projector.train()
    img_pre_projector.train()
    text_projector.train()
    share_encoder.train()
    if use_multilayer_router:
        img_projectors.train()
        layer_router.train()
    else:
        img_projector.train()

    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_test_loss = float('inf')
    best_test_epoch = 0
    early_stop_counter = 0
    share_encoder_frozen = False
    stage2_switched = False

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        eeg_projector.train()
        img_pre_projector.train()
        text_projector.train()
        if use_multilayer_router:
            img_projectors.train()
            layer_router.train()
        else:
            img_projector.train()
        if not share_encoder_frozen:
            share_encoder.train()
        else:
            share_encoder.eval()

        if (not share_encoder_frozen) and (epoch > args.stage1_epochs):
            for param in share_encoder.parameters():
                param.requires_grad = False
            share_encoder_frozen = True
            log(f'Freeze share_encoder at epoch {epoch}')

        if (not stage2_switched) and (epoch > args.stage1_epochs):
            optimizer = build_optimizer(args.stage2_learning_rate, include_share_encoder=False)
            stage2_switched = True
            log(f'Switch optimizer at epoch {epoch}, stage2 lr = {args.stage2_learning_rate}')

        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs} [Train]'):
            eeg_batch = batch[0].to(device)
            subject_id_batch = batch[3].to(device)
            image_feature_batch = batch[1].to(device)
            text_feature_batch = batch[2].to(device)

            optimizer.zero_grad()
            if args.eeg_encoder_type in ['ATM']:
                eeg_feature_batch = model(eeg_batch, subject_id_batch)
            else:
                eeg_feature_batch = model(eeg_batch)

            eeg_feature_batch = eeg_projector(eeg_feature_batch)
            eeg_feature_batch = share_encoder(eeg_feature_batch)

            image_feature_batch = build_image_teacher(image_feature_batch, subject_id_batch, training=True)
            text_feature_batch = text_projector(text_feature_batch)

            if epoch <= args.stage1_epochs:
                mmd_w, contrast_w = get_stage1_weights(epoch, args.stage1_epochs, args.stage1_mmd_start, args.stage1_mmd_end)
                loss = mmd_w * mmd_rbf(eeg_feature_batch, image_feature_batch) + contrast_w * criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
            else:
                loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        if epoch <= args.stage1_epochs:
            mmd_w, contrast_w = get_stage1_weights(epoch, args.stage1_epochs, args.stage1_mmd_start, args.stage1_mmd_end)
            writer.add_scalar('LossWeight/mmd', mmd_w, epoch)
            writer.add_scalar('LossWeight/contrastive', contrast_w, epoch)
            log(f'Epoch [{epoch}/{args.num_epochs}] Train Loss: {avg_loss:.4f} | stage1 mmd={mmd_w:.3f} contrast={contrast_w:.3f}')
        else:
            log(f'Epoch [{epoch}/{args.num_epochs}] Train Loss: {avg_loss:.4f} | stage2 contrastive only')

        if use_multilayer_router:
            with torch.no_grad():
                gw = layer_router.get_global_weights().detach().cpu().numpy()
            log(f'Global layer weights {args.layer_ids}: {[round(float(x), 4) for x in gw]}')
            for lid, w in zip(args.layer_ids, gw):
                writer.add_scalar(f'Router/global_weight_L{lid}', float(w), epoch)

        if args.save_weights:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'eeg_projector_state_dict': eeg_projector.state_dict(),
                'img_pre_projector_state_dict': img_pre_projector.state_dict(),
                'text_projector_state_dict': text_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'share_enc_state_dict': share_encoder.state_dict(),
                'loss': avg_loss,
            }
            if use_multilayer_router:
                state['img_projectors_state_dict'] = img_projectors.state_dict()
                state['layer_router_state_dict'] = layer_router.state_dict()
                state['layer_ids'] = args.layer_ids
            else:
                state['img_projector_state_dict'] = img_projector.state_dict()
            torch.save(state, f"{writer.log_dir}/checkpoint_last.pth")

        model.eval()
        eeg_projector.eval()
        img_pre_projector.eval()
        text_projector.eval()
        share_encoder.eval()
        if use_multilayer_router:
            img_projectors.eval()
            layer_router.eval()
        else:
            img_projector.eval()

        total_test_loss = 0.0
        eeg_feature_list = []
        image_feature_list = []

        with torch.no_grad():
            for batch in test_dataloader:
                eeg_batch = batch[0].to(device)
                subject_id_batch = batch[3].to(device)
                image_feature_batch = batch[1].to(device)
                text_feature_batch = batch[2].to(device)

                if args.eeg_encoder_type in ['ATM']:
                    eeg_feature_batch = model(eeg_batch, subject_id_batch)
                else:
                    eeg_feature_batch = model(eeg_batch)

                eeg_feature_batch = eeg_projector(eeg_feature_batch)
                eeg_feature_batch = share_encoder(eeg_feature_batch)

                image_feature_batch = build_image_teacher(image_feature_batch, subject_id_batch, training=False)
                text_feature_batch = text_projector(text_feature_batch)

                loss = criterion(eeg_feature_batch, image_feature_batch, text_feature_batch)
                total_test_loss += loss.item()

                eeg_feature_list.append(eeg_feature_batch.cpu().numpy())
                image_feature_list.append(image_feature_batch.cpu().numpy())

        avg_test_loss = total_test_loss / len(test_dataloader)
        writer.add_scalar('Loss/test', avg_test_loss, epoch)

        eeg_feature_all = np.concatenate(eeg_feature_list, axis=0)
        image_feature_all = np.concatenate(image_feature_list, axis=0)
        top5_count, top1_count, total = retrieve_all(eeg_feature_all, image_feature_all, args.data_average)
        top5_acc = top5_count / total * 100
        top1_acc = top1_count / total * 100
        log(f'top5 acc {top5_acc:.2f}%\ttop1 acc {top1_acc:.2f}%\tTest Loss: {avg_test_loss:.4f}')

        is_better = False
        if top1_acc > best_top1_acc + args.early_stop_min_delta:
            is_better = True
        elif avg_test_loss < best_test_loss and top1_acc == best_top1_acc:
            is_better = True

        if is_better:
            best_test_loss = avg_test_loss
            best_top5_acc = top5_acc
            best_top1_acc = top1_acc
            best_test_epoch = epoch
            if args.save_weights:
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'eeg_projector_state_dict': eeg_projector.state_dict(),
                    'img_pre_projector_state_dict': img_pre_projector.state_dict(),
                    'text_projector_state_dict': text_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'share_enc_state_dict': share_encoder.state_dict(),
                    'loss': avg_test_loss,
                }
                if use_multilayer_router:
                    state['img_projectors_state_dict'] = img_projectors.state_dict()
                    state['layer_router_state_dict'] = layer_router.state_dict()
                    state['layer_ids'] = args.layer_ids
                else:
                    state['img_projector_state_dict'] = img_projector.state_dict()
                torch.save(state, f"{writer.log_dir}/checkpoint_test_best.pth")

            early_stop_counter = 0
        else:
            if epoch > args.stage1_epochs:
                early_stop_counter += 1
                if args.early_stop_patience > 0:
                    log(f'EarlyStop counter: {early_stop_counter}/{args.early_stop_patience} | best epoch: {best_test_epoch}')

        if args.early_stop_patience > 0 and epoch > args.stage1_epochs and early_stop_counter >= args.early_stop_patience:
            log(f'Early stopping at epoch {epoch}: no Top-1 improvement for {args.early_stop_patience} epochs after stage-1. Best epoch = {best_test_epoch}.')
            break

    result_dict = {}
    result_dict['top1 acc'] = f'{top1_acc:.2f}'
    result_dict['top5 acc'] = f'{top5_acc:.2f}'
    result_dict['best top1 acc'] = f'{best_top1_acc:.2f}'
    result_dict['best top5 acc'] = f'{best_top5_acc:.2f}'
    result_dict['best test loss'] = f'{best_test_loss:.4f}'
    result_dict['best epoch'] = best_test_epoch
    df = pd.DataFrame(result_dict, index=[0])
    df.to_csv(os.path.join(log_dir, 'result.csv'), index=False)

    log(f'best test loss: {best_test_loss:.4f} top5 acc: {best_top5_acc:.2f} top1 acc: {best_top1_acc:.2f} at epoch {best_test_epoch}')
