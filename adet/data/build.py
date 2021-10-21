# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import logging
import operator
import pickle

import numpy as np
import torch.utils.data
from detectron2.data.build import (print_instances_class_histogram,
                                   trivial_batch_collator,
                                   worker_init_reset_seed)
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from detectron2.data.common import MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import check_metadata_consistency
from detectron2.data.samplers import (InferenceSampler,
                                      RepeatFactorTrainingSampler,
                                      TrainingSampler)
from detectron2.utils.comm import get_world_size
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored

from .common import VideoAspectRatioGroupedDataset, VideoDataset

__all__ = ['build_youtubevis_train_loader']


def build_batch_data_loader(dataset,
                            sampler,
                            total_batch_size,
                            *,
                            aspect_ratio_grouping=False,
                            num_workers=0):
    """Build a batched dataloader for training.

    Args:
        dataset (torch.utils.data.Dataset): map-style PyTorch dataset. Can be indexed.
        sampler (torch.utils.data.sampler.Sampler): a sampler that produces indices
        total_batch_size (int): total batch size across GPUs.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), 'Total batch size ({}) must be divisible by the number of gpus ({}).'.format(
        total_batch_size, world_size)

    batch_size = total_batch_size // world_size
    if aspect_ratio_grouping:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return VideoAspectRatioGroupedDataset(data_loader, batch_size)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size,
            drop_last=True)  # drop_last so the batch always have the same size
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=trivial_batch_collator,
            worker_init_fn=worker_init_reset_seed,
        )


def get_youtubevis_dataset_dicts(dataset_names, filter_empty=True):
    assert len(dataset_names)
    """
    only youtubevis supported now
    """
    dataset_dicts, img_ids, vids_annos = [
        DatasetCatalog.get(dataset_name) for dataset_name in dataset_names
    ][0]
    """
    DatasetCatalog get a triplet return: (dataset_dicts, img_ids, vids_annos)
    """
    for dataset_name, dicts in zip(dataset_names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty".format(dataset_name)

    def valid(anns):
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:
                return True
        return False

    has_instances = 'annotations' in dataset_dicts[0]

    if filter_empty and has_instances:
        num_before = len(dataset_dicts)
        valid_idx = []
        for idx in range(len(dataset_dicts)):
            if valid(dataset_dicts[idx]['annotations']):
                valid_idx.append(idx)
        dataset_dicts = [dataset_dicts[idx] for idx in valid_idx]
        img_ids = [img_ids[idx] for idx in valid_idx]
        num_after = len(dataset_dicts)
        logger = logging.getLogger(__name__)
        logger.info(
            'Removed {} images with no usable annotations. {} images left.'.
            format(num_before - num_after, num_after))

    if has_instances:
        try:
            class_names = MetadataCatalog.get(dataset_names[0]).thing_classes
            check_metadata_consistency('thing_classes', dataset_names)
            print_instances_class_histogram(dataset_dicts, class_names)
        except AttributeError:  # class names are not available for this dataset
            pass
    return dataset_dicts, img_ids, vids_annos


def build_youtubevis_train_loader(cfg, mapper=None):
    dataset_dicts, img_ids, vids_annos = get_youtubevis_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
    )
    dataset = VideoDataset(
        img_ids,
        dataset_dicts,
        vids_annos,
        sample_region=cfg.MODEL.CROSSVIS.CROSSOVER.SAMPLE_REGION)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info('Using training sampler {}'.format(sampler_name))

    # TODO avoid if-else?
    if sampler_name == 'TrainingSampler':
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == 'RepeatFactorTrainingSampler':
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD)
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError('Unknown training sampler: {}'.format(sampler_name))
    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
