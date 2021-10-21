import argparse
import copy
import json
import os.path as osp

import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.modeling import build_model
from detectron2.utils.visualizer import ColorMode
from pycocotools import mask as maskUtils
from tqdm import tqdm

from adet.checkpoint import AdetCheckpointer
from adet.config import get_cfg


class StaticImageDemo:
    def __init__(
        self,
        cfg,
        instance_mode=ColorMode.IMAGE,
    ):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else '__unused')
        self.cpu_device = torch.device('cpu')
        self.instance_mode = instance_mode
        self.predictor = StaticImagePredictor(cfg)

    def __call__(self, image):
        predictions = self.predictor(image)
        return predictions


class StaticImagePredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        AdetCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
            cfg.INPUT.MAX_SIZE_TEST)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    def __call__(self, input_image):
        image = copy.deepcopy(input_image)
        with torch.no_grad():
            if self.input_format == 'RGB':
                image = image[:, :, ::-1]
            height = image.shape[0]
            width = image.shape[1]
            image = self.aug.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))

            input = {'image': image, 'height': height, 'width': width}
            prediction = self.model.forward([input])

            return prediction


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config-file',
                        metavar='FILE',
                        help='path to config file')
    parser.add_argument('--json-file',
                        metavar='FILE',
                        help='path to json file')

    parser.add_argument(
        '--opts',
        help='Modify config options using the command-line',
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def compute_scores(pred_inst, assigned_insts):
    denominator = 0
    numerator = 0
    numerators = []

    def innerp(x, y):
        ret = torch.mm(
            x, y.T).item() / torch.norm(x).item() / torch.norm(y).item()
        return ret

    def area_of(left_top, right_bottom):
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]

    def iou_of(boxes0, boxes1, eps=1e-5):
        boxes0 = boxes0[:, None, :4]
        boxes1 = boxes1[None, :, :4]

        overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = area_of(overlap_left_top, overlap_right_bottom)
        area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
        ret = overlap_area / (area0 + area1 - overlap_area + eps)
        return ret.item()

    def distance_func(inst1, inst2):
        return innerp(inst1['reid_feat'], inst2['reid_feat']) + iou_of(
            inst1['pred_box'], inst2['pred_box'])

    for inst in assigned_insts:
        if inst['pred_class'] != pred_inst['pred_class']:
            continue
        denominator += 1

        numerators.append(distance_func(pred_inst, inst))

    numerator = sum(numerators)

    if not denominator:
        return -1.
    else:
        return numerator / denominator


if __name__ == '__main__':
    args = parse_args()
    print('Command Line Args:', args)
    cfg = setup(args)
    demo = StaticImageDemo(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    cpu_device = torch.device('cpu')

    data = json.load(open(args.json_file, 'r'))
    results = []
    for video in tqdm(data['videos']):
        memory = dict()
        images = [
            read_image(osp.join('datasets/youtubevis/valid', image_name),
                       'BGR') for image_name in video['file_names']
        ]
        video_name = video['file_names'][0].split('/')[0]

        for index in range(len(images)):
            predictions = demo(images[index])[0]['instances']
            predictions_num = len(predictions)
            pred_insts = []
            if predictions_num == 0:
                continue
            for pred_ind in range(predictions_num):
                pred_insts.append(
                    dict(assigned=(-1, 0),
                         index=index,
                         reid_feat=predictions._fields['reid_feat'][pred_ind][
                             None, ...],
                         pred_mask=predictions._fields['pred_masks'][pred_ind],
                         pred_box=predictions._fields['pred_boxes']
                         [pred_ind].tensor,
                         pred_score=predictions._fields['scores']
                         [pred_ind].detach().cpu().numpy().item(),
                         pred_class=predictions._fields['pred_classes']
                         [pred_ind].detach().cpu().numpy().item()))

            counts = len(memory)
            scores_ = []

            for pred_inst in pred_insts:
                scores, ids = [], list(memory.keys())
                for k in ids:
                    scores.append(compute_scores(pred_inst, memory[k][-1:]))
                scores_.append(scores)
                if not scores or max(scores) < 0.4:
                    pred_inst['assigned'] = (-1, 0)
                else:
                    pred_inst['assigned'] = (ids[scores.index(max(scores))],
                                             max(scores))
            scores = np.array(scores_)

            max_assigned_score = np.ones(len(memory)) * -1.
            for pred_inst in pred_insts:
                if pred_inst['assigned'][0] == -1:
                    memory[len(memory)] = [pred_inst]
                else:
                    if pred_inst['assigned'][1] > max_assigned_score[
                            pred_inst['assigned'][0]]:
                        if max_assigned_score[pred_inst['assigned'][0]] < 0:
                            memory[pred_inst['assigned'][0]].append(pred_inst)
                        else:
                            memory[pred_inst['assigned'][0]].pop(-1)
                            memory[pred_inst['assigned'][0]].append(pred_inst)
                        max_assigned_score[pred_inst['assigned']
                                           [0]] = pred_inst['assigned'][1]

        for assigned_inst_id in memory.keys():
            objs = dict(video_id=video['id'],
                        score=sum([
                            pred_inst['pred_score']
                            for pred_inst in memory[assigned_inst_id]
                        ]) / len(memory[assigned_inst_id]),
                        category_id=memory[assigned_inst_id][0]['pred_class'] +
                        1,
                        segmentations=[])

            for frame in range(len(video['file_names'])):
                count = 0
                mask_ = None
                for inst in memory[assigned_inst_id]:
                    if inst['index'] == frame:
                        if mask_ is not None:
                            mask_ += inst['pred_mask']
                            count += 1
                        else:
                            mask_ = inst['pred_mask']
                            count = 1.
                if mask_ is not None:
                    mask_ /= count
                    mask_ = mask_.detach().cpu().numpy()
                    mask_ = np.array(mask_ > 0.5, dtype=np.uint8)
                    mask_ = maskUtils.encode(
                        np.array(mask_[:, :, np.newaxis],
                                 order='F',
                                 dtype='uint8'))[0]
                    mask_['counts'] = mask_['counts'].decode()
                objs['segmentations'].append(mask_)
            results.append(objs)
    json.dump(results, open('results.json', 'w'))
