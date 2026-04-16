import copy
import platform
import random
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from mmcv.utils.parrots_wrapper import DataLoader, PoolDataLoader
from torch.utils.data import DistributedSampler, WeightedRandomSampler

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    hard_limit = rlimit[1]
    soft_limit = min(4096, hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')


def _build_weighted_random_sampler(dataset, sampler_cfg):
    sampler_cfg = copy.deepcopy(sampler_cfg)
    sampler_type = sampler_cfg.pop('type', None)
    if sampler_type != 'WeightedRandomSampler':
        raise ValueError(f'Unsupported sampler type: {sampler_type}')

    weights = sampler_cfg.pop('weights', None)
    weights_file = sampler_cfg.pop('weights_file', None)
    if weights is None:
        if weights_file is None:
            raise ValueError('`weights` or `weights_file` must be provided for '
                             '`WeightedRandomSampler`.')
        weights = np.load(weights_file)

    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    dataset_len = len(dataset)
    if weights.size != dataset_len:
        if hasattr(dataset, 'dataset') and hasattr(dataset, 'times') and \
                weights.size == len(dataset.dataset):
            weights = np.tile(weights, int(dataset.times))
        elif dataset_len % weights.size == 0:
            weights = np.tile(weights, dataset_len // weights.size)
        else:
            raise ValueError(
                f'Weights length mismatch: got {weights.size}, '
                f'but dataset length is {dataset_len}.')

    num_samples = sampler_cfg.pop('num_samples', dataset_len)
    replacement = sampler_cfg.pop('replacement', True)
    if sampler_cfg:
        raise ValueError(
            f'Unsupported WeightedRandomSampler args: '
            f'{list(sampler_cfg.keys())}')

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=replacement)


def _concat_dataset(cfg, default_args=None):
    """Build :obj:`ConcatDataset by."""
    from .dataset_wrappers import ConcatDataset
    img_dir = cfg['img_dir']
    ann_dir = cfg.get('ann_dir', None)
    split = cfg.get('split', None)
    num_img_dir = len(img_dir) if isinstance(img_dir, (list, tuple)) else 1
    if ann_dir is not None:
        num_ann_dir = len(ann_dir) if isinstance(ann_dir, (list, tuple)) else 1
    else:
        num_ann_dir = 0
    if split is not None:
        num_split = len(split) if isinstance(split, (list, tuple)) else 1
    else:
        num_split = 0
    if num_img_dir > 1:
        assert num_img_dir == num_ann_dir or num_ann_dir == 0
        assert num_img_dir == num_split or num_split == 0
    else:
        assert num_split == num_ann_dir or num_ann_dir <= 1
    num_dset = max(num_split, num_img_dir)

    datasets = []
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        if isinstance(img_dir, (list, tuple)):
            data_cfg['img_dir'] = img_dir[i]
        if isinstance(ann_dir, (list, tuple)):
            data_cfg['ann_dir'] = ann_dir[i]
        if isinstance(split, (list, tuple)):
            data_cfg['split'] = split[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    """Build datasets."""
    from .dataset_wrappers import ConcatDataset, RepeatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg.get('img_dir'), (list, tuple)) or isinstance(
            cfg.get('split', None), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     drop_last=False,
                     pin_memory=True,
                     dataloader_type='PoolDataLoader',
                     sampler_cfg=None,
                     **kwargs):
    """Build PyTorch DataLoader.

    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.

    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        seed (int | None): Seed to be used. Default: None.
        drop_last (bool): Whether to drop the last incomplete batch in epoch.
            Default: False
        pin_memory (bool): Whether to use pin_memory in DataLoader.
            Default: True
        dataloader_type (str): Type of dataloader. Default: 'PoolDataLoader'
        kwargs: any keyword argument to be used to initialize DataLoader

    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    if dist:
        if sampler_cfg is not None:
            raise NotImplementedError(
                '`sampler_cfg` is currently only supported in non-distributed '
                'training.')
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        if sampler_cfg is not None:
            sampler = _build_weighted_random_sampler(dataset, sampler_cfg)
            shuffle = False
            print(
                f'[WeightedRandomSampler] enabled: dataset_len={len(dataset)}, '
                f'sampler={type(sampler).__name__}, num_samples={sampler.num_samples}, '
                f'replacement={sampler.replacement}')
        else:
            sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    assert dataloader_type in (
        'DataLoader',
        'PoolDataLoader'), f'unsupported dataloader {dataloader_type}'

    if dataloader_type == 'PoolDataLoader':
        dataloader = PoolDataLoader
    elif dataloader_type == 'DataLoader':
        dataloader = DataLoader

    data_loader = dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
