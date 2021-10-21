# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import datetime
import io
import json
import logging
import os

import numpy as np
import pycocotools.mask as mask_util
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, PolygonMasks
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ['load_youtubevis_json', 'register_youtubevis_video']


def load_youtubevis_json(json_file,
                         image_root,
                         dataset_name=None,
                         extra_annotation_keys=None):
    # lazy import YTVOS
    from pycocotools.ytvos import YTVOS

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        youtubevis_api = YTVOS(json_file)
    if timer.seconds() > 1:
        logger.info('Loading {} takes {:.2f} seconds.'.format(
            json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(youtubevis_api.getCatIds())
        cats = youtubevis_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [
            c['name'] for c in sorted(cats, key=lambda x: x['id'])
        ]
        meta.thing_classes = thing_classes

        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if 'youtubevis' not in dataset_name:
                logger.warning("""
                    Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                    """)
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # sort video indices for reproducible results
    vid_ids = sorted(youtubevis_api.vids.keys())
    vid_infos = youtubevis_api.loadVids(vid_ids)  # get video infos

    img_ids = []  # get image ids
    for idx, vid_info in enumerate(vid_infos):
        for frame_id in range(len(vid_info['file_names'])):
            img_ids.append((idx, frame_id))

    anns = [youtubevis_api.vidToAnns[vid_id]
            for vid_id in vid_ids]  # get all video annotations

    vids_annos = list(zip(vid_infos, anns))

    logger.info('Loaded {} images in YOUTUBEVIS format from {}'.format(
        len(img_ids), json_file))

    dataset_dicts = []

    num_instances_without_valid_segmentation = 0

    for _, (vid, frame_id) in enumerate(img_ids):
        record = {}
        record['file_name'] = os.path.join(
            image_root, vid_infos[vid]['file_names'][frame_id])
        record['height'] = vid_infos[vid]['height']
        record['width'] = vid_infos[vid]['width']
        record['video_id'] = vid

        objs = []
        for anno in anns[vid]:
            assert anno['video_id'] == record['video_id'] + 1

            assert anno.get(
                'ignore',
                0) == 0, '"ignore" in COCO json file is not supported.'

            obj = dict(
                iscrowd=anno['iscrowd'],  # iscrowd
                id=anno['id'],  # instance id
                bbox=anno['bboxes'][frame_id],  # bbox
                category_id=anno['category_id']  # category
            )

            if obj['bbox'] is None:
                continue
                # if bbox is None, skip this object for this frame.

            segm = anno['segmentations'][frame_id]

            if isinstance(segm, dict):
                if isinstance(segm['counts'], list):
                    segm = mask_util.frPyObjects(segm, *segm['size'])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [
                        poly for poly in segm
                        if len(poly) % 2 == 0 and len(poly) >= 6
                    ]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue  # ignore this instance
                obj['segmentation'] = segm

            obj['bbox_mode'] = BoxMode.XYWH_ABS
            if id_map:
                obj['category_id'] = id_map[obj['category_id']]
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)

    return dataset_dicts, img_ids, vids_annos


def register_youtubevis_video(name, metadata, json_file, image_root):
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(
        name, lambda: load_youtubevis_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type='youtubevis',
                                  **metadata)
