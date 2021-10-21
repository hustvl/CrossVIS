# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import pickle
import random

import numpy as np
import torch.utils.data as data

__all__ = ['VideoDataset', 'VideoAspectRatioGroupedDataset']


class VideoDataset(data.Dataset):
    def __init__(self,
                 img_ids,
                 lst,
                 vids_annos,
                 copy: bool = True,
                 serialize: bool = True,
                 sample_region=-1):
        self._img_ids = img_ids
        self._lst = lst
        self._serialize = serialize
        self._vids_annos = vids_annos
        self.sample_region = sample_region

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            logger = logging.getLogger(__name__)
            logger.info(
                'Serializing {} elements to byte tensors and concatenating them all ...'
                .format(len(self._lst)))
            self._lst = [_serialize(x) for x in self._lst]
            self._addr = np.asarray([len(x) for x in self._lst],
                                    dtype=np.int64)
            self._addr = np.cumsum(self._addr)
            self._lst = np.concatenate(self._lst)
            logger.info('Serialized dataset takes {:.2f} MiB'.format(
                len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._img_ids)

    def sample_ref(self, video_id, frame_id):
        annos = self._vids_annos[video_id][0]
        valid_imgs = []
        for i in range(annos['length']):
            if i == frame_id:
                continue
            elif (video_id, i) in self._img_ids:
                valid_imgs.append((video_id, i))
        assert len(valid_imgs) > 0
        ref = random.choice(valid_imgs)
        return self._img_ids.index(ref)

    def sample_ref_region(self, video_id, frame_id):
        annos = self._vids_annos[video_id][0]
        valid_imgs = []
        for i in range(annos['length']):
            if i == frame_id:
                continue
            elif abs(i - frame_id) > self.sample_region and self.sample_region:
                continue
            elif (video_id, i) in self._img_ids:
                valid_imgs.append((video_id, i))

        if len(valid_imgs):
            ref = random.choice(valid_imgs)
        else:
            ref = (video_id, frame_id)
        return self._img_ids.index(ref)

    def __getitem__(self, idx):
        """sample image and reference image."""
        if self.sample_region > 0:
            ref_idx = self.sample_ref_region(*self._img_ids[idx])
        else:
            ref_idx = self.sample_ref(*self._img_ids[idx])

        if self._serialize:
            start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
            end_addr = self._addr[idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            data = pickle.loads(bytes)

            start_addr = 0 if ref_idx == 0 else self._addr[ref_idx - 1].item()
            end_addr = self._addr[ref_idx].item()
            bytes = memoryview(self._lst[start_addr:end_addr])
            ref_data = pickle.loads(bytes)
        else:
            data = self._lst[idx]
            ref_data = self._lst[ref_idx]

        return data, ref_data


class VideoAspectRatioGroupedDataset(data.IterableDataset):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self._buckets = [[] for _ in range(2)]

    def __iter__(self):
        for d in self.dataset:
            w, h = d[0]['width'], d[0]['height']
            bucket_id = 0 if w > h else 1
            bucket = self._buckets[bucket_id]
            bucket.append(d)
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]
