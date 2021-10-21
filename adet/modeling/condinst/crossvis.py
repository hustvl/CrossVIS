import logging
import math
import random

import torch
import torch.nn.functional as F
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import ImageList
from detectron2.structures.instances import Instances
from detectron2.structures.masks import PolygonMasks, polygons_to_bitmask
from fvcore.nn import sigmoid_focal_loss_jit
from skimage import color
from torch import nn

from adet.utils.comm import aligned_bilinear

from .dynamic_mask_head import build_dynamic_mask_head
from .mask_branch import build_mask_branch

__all__ = ['CrossVIS']

logger = logging.getLogger(__name__)


def unfold_wo_center(x, kernel_size, dilation):
    assert x.dim() == 4
    assert kernel_size % 2 == 1

    # using SAME padding
    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x,
                          kernel_size=kernel_size,
                          padding=padding,
                          dilation=dilation)

    unfolded_x = unfolded_x.reshape(x.size(0), x.size(1), -1, x.size(2),
                                    x.size(3))

    # remove the center pixels
    size = kernel_size**2
    unfolded_x = torch.cat(
        (unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]),
        dim=2)

    return unfolded_x


def get_images_color_similarity(images, image_masks, kernel_size, dilation):
    assert images.dim() == 4
    assert images.size(0) == 1

    unfolded_images = unfold_wo_center(images,
                                       kernel_size=kernel_size,
                                       dilation=dilation)

    diff = images[:, :, None] - unfolded_images
    similarity = torch.exp(-torch.norm(diff, dim=1) * 0.5)

    unfolded_weights = unfold_wo_center(image_masks[None, None],
                                        kernel_size=kernel_size,
                                        dilation=dilation)
    unfolded_weights = torch.max(unfolded_weights, dim=1)[0]

    return similarity * unfolded_weights


@META_ARCH_REGISTRY.register()
class CrossVIS(nn.Module):
    """Main class for CrossVIS architectures (see
    https://arxiv.org/abs/2003.05664)."""
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(
            cfg, self.backbone.output_shape())
        self.mask_head = build_dynamic_mask_head(cfg)
        self.mask_branch = build_mask_branch(cfg, self.backbone.output_shape())

        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE

        self.max_proposals = cfg.MODEL.CONDINST.MAX_PROPOSALS
        self.topk_proposals_per_im = cfg.MODEL.CONDINST.TOPK_PROPOSALS_PER_IM

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH

        # build top module
        in_channels = self.proposal_generator.in_channels_to_top_module

        self.controller = nn.Conv2d(in_channels,
                                    self.mask_head.num_gen_params,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        torch.nn.init.normal_(self.controller.weight, std=0.01)
        torch.nn.init.constant_(self.controller.bias, 0)

        self.nID = 3774
        self.cls = nn.Linear(256, self.nID, bias=True)
        torch.nn.init.normal_(self.cls.weight, std=0.01)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls.bias, bias_value)

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(
            3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(
            3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        if self.training:
            return self.forward_train(batched_inputs)
        else:
            return self.forward_test(batched_inputs)

    def forward_train(self, batched_inputs):
        assert self.training
        original_images_0 = [
            x[0]['image'].to(self.device) for x in batched_inputs
        ]
        original_images_1 = [
            x[1]['image'].to(self.device) for x in batched_inputs
        ]

        # normalize images
        images_norm_0 = [self.normalizer(x) for x in original_images_0]
        images_norm_0 = ImageList.from_tensors(images_norm_0,
                                               self.backbone.size_divisibility)

        images_norm_1 = [self.normalizer(x) for x in original_images_1]
        images_norm_1 = ImageList.from_tensors(images_norm_1,
                                               self.backbone.size_divisibility)

        features_0_origin = self.backbone(images_norm_0.tensor)
        features_1_origin = self.backbone(images_norm_1.tensor)
        features_0, features_1 = dict(), dict()
        # backbone lr 0.1x
        for k in features_0_origin.keys():
            features_0[k] = features_0_origin[k] * 0.1 + features_0_origin[
                k].detach() * 0.9
            features_1[k] = features_1_origin[k] * 0.1 + features_1_origin[
                k].detach() * 0.9

        if 'instances' in batched_inputs[0][0]:
            gt_instances_0 = [
                x[0]['instances'].to(self.device) for x in batched_inputs
            ]
            self.add_bitmasks(gt_instances_0, images_norm_0.tensor.size(-2),
                              images_norm_0.tensor.size(-1))

            gt_instances_1 = [
                x[1]['instances'].to(self.device) for x in batched_inputs
            ]
            self.add_bitmasks(gt_instances_1, images_norm_1.tensor.size(-2),
                              images_norm_1.tensor.size(-1))

            count_0, count_1 = 0, 0

            for inst_0, inst_1 in zip(gt_instances_0, gt_instances_1):
                zero_to_one = inst_0.rel_ids > -1
                one_to_zero = inst_1.rel_ids > -1

                inst_0.rel_ids += inst_0.rel_ids.new_ones(
                    inst_0.rel_ids.size()) * count_1 * zero_to_one
                inst_1.rel_ids += inst_1.rel_ids.new_ones(
                    inst_1.rel_ids.size()) * count_0 * one_to_zero

                count_0 += len(inst_0)
                count_1 += len(inst_1)
        else:
            raise BaseException('no ground truth for training')

        mask_feats_0, sem_losses_0 = self.mask_branch(features_0,
                                                      gt_instances_0)
        mask_feats_1, sem_losses_1 = self.mask_branch(features_1,
                                                      gt_instances_1)

        proposals_0, proposal_losses_0 = self.proposal_generator(
            images_norm_0, features_0, gt_instances_0, self.controller)
        proposals_1, proposal_losses_1 = self.proposal_generator(
            images_norm_1, features_1, gt_instances_1, self.controller)

        mask_losses_0 = self._forward_mask_heads_train(proposals_0,
                                                       mask_feats_0,
                                                       gt_instances_0)
        mask_losses_1 = self._forward_mask_heads_train(proposals_1,
                                                       mask_feats_1,
                                                       gt_instances_1)

        loss_cross_over = self.cross_over(proposals_0, proposals_1,
                                          mask_feats_0, mask_feats_1,
                                          gt_instances_0, gt_instances_1)

        reid_feats_0 = proposals_0['instances'].reid_feats
        reid_feats_1 = proposals_1['instances'].reid_feats
        reid_feats = torch.cat([reid_feats_0, reid_feats_1], dim=0)

        gt_ids_0 = torch.cat([per_im.id for per_im in gt_instances_0], dim=0)
        ids_0 = gt_ids_0[proposals_0['instances'].gt_inds]
        proposals_0['instances'].set('ids', ids_0)

        gt_ids_1 = torch.cat([per_im.id for per_im in gt_instances_1], dim=0)
        ids_1 = gt_ids_1[proposals_1['instances'].gt_inds]
        proposals_1['instances'].set('ids', ids_1)

        reid_target = torch.cat([ids_0, ids_1], dim=0) - 1

        reid_feats = self.emb_scale * F.normalize(reid_feats)

        reid_output = self.cls(reid_feats)
        reid_target_one_hot = reid_output.new_zeros(
            reid_feats.size(0),
            self.nID).scatter_(1,
                               reid_target.long().view(-1, 1), 1)
        loss_reid = sigmoid_focal_loss_jit(
            reid_output,
            reid_target_one_hot,
            alpha=0.25,
            gamma=2.,
            reduction='sum') / reid_output.size(0)

        losses, losses_0, losses_1 = {}, {}, {}
        losses.update(sem_losses_0)
        losses.update(proposal_losses_0)
        losses.update(mask_losses_0)

        losses.update(sem_losses_1)
        losses.update(proposal_losses_1)
        losses.update(mask_losses_1)

        for key in losses_0:
            losses[key] = (losses_0[key] + losses_1[key]) / 2.

        losses.update({'loss_cross_over': loss_cross_over})
        losses.update({'loss_reid': loss_reid})

        return losses

    def cross_over(self, proposals_0, proposals_1, mask_feats_0, mask_feats_1,
                   gt_instances_0, gt_instances_1):

        pred_instances_0 = proposals_0['instances']
        pred_instances_1 = proposals_1['instances']

        if 0 <= self.max_proposals < len(pred_instances_0):
            inds = torch.randperm(len(pred_instances_0),
                                  device=mask_feats_0.device).long()
            logger.info('clipping proposals from {} to {}'.format(
                len(pred_instances_0), self.max_proposals))
            pred_instances_0 = pred_instances_0[inds[:self.max_proposals]]

        if 0 <= self.max_proposals < len(pred_instances_1):
            inds = torch.randperm(len(pred_instances_1),
                                  device=mask_feats_1.device).long()
            logger.info('clipping proposals from {} to {}'.format(
                len(pred_instances_1), self.max_proposals))
            pred_instances_1 = pred_instances_1[inds[:self.max_proposals]]

        # compute relative gt_inds
        rel_ids_0 = torch.cat([per_im.rel_ids for per_im in gt_instances_0])
        rel_ids_0[torch.where(
            (rel_ids_0[:,
                       None] == pred_instances_1.gt_inds[None, :]).float().sum(
                           dim=1) == 0)] = -1
        gt_inds_0 = pred_instances_0.gt_inds
        inst_rel_ids = rel_ids_0[gt_inds_0].long()
        pred_instances_0.set('inst_rel_ids', inst_rel_ids)

        # compute relative im_inds
        im_inds = []
        for i in range(len(gt_instances_1)):
            for _ in range(len(gt_instances_1[i])):
                im_inds.append(i)
        im_inds = inst_rel_ids.new_tensor(im_inds)
        inst_im_inds = im_inds[inst_rel_ids]
        pred_instances_0.set('inst_im_inds', inst_im_inds)

        pred_inst_0 = pred_instances_0[torch.where(
            pred_instances_0.inst_rel_ids > -1)]

        inst_0 = Instances((0, 0))
        inst_0.set('gt_inds', pred_inst_0.inst_rel_ids)
        inst_0.set('im_inds', pred_inst_0.inst_im_inds)
        inst_0.set('mask_head_params', pred_inst_0.mask_head_params)

        # random sample locations and fpn_levels
        locations_0 = rel_ids_0.new_zeros((len(inst_0), 2))
        fpn_levels_0 = rel_ids_0.new_zeros(len(inst_0))

        for i in range(len(inst_0)):
            locations_pool = pred_instances_1.locations[torch.where(
                pred_instances_1.gt_inds == inst_0.gt_inds[i])]
            fpn_levels_pool = pred_instances_1.fpn_levels[torch.where(
                pred_instances_1.gt_inds == inst_0.gt_inds[i])]
            index = random.randint(0, locations_pool.size(0) - 1)
            locations_0[i, :] = locations_pool[index, :]
            fpn_levels_0[i] = fpn_levels_pool[index]

        inst_0.set('locations', locations_0)
        inst_0.set('fpn_levels', fpn_levels_0.long())

        # compute relative gt_inds
        rel_ids_1 = torch.cat([per_im.rel_ids for per_im in gt_instances_1])
        rel_ids_1[torch.where(
            (rel_ids_1[:,
                       None] == pred_instances_0.gt_inds[None, :]).float().sum(
                           dim=1) == 0)] = -1
        gt_inds_1 = pred_instances_1.gt_inds
        inst_rel_ids = rel_ids_1[gt_inds_1].long()
        pred_instances_1.set('inst_rel_ids', inst_rel_ids)

        # compute relative im_inds
        im_inds = []
        for i in range(len(gt_instances_0)):
            for _ in range(len(gt_instances_0[i])):
                im_inds.append(i)
        im_inds = inst_rel_ids.new_tensor(im_inds)
        inst_im_inds = im_inds[inst_rel_ids]
        pred_instances_1.set('inst_im_inds', inst_im_inds)

        pred_inst_1 = pred_instances_1[torch.where(
            pred_instances_1.inst_rel_ids > -1)]

        inst_1 = Instances((0, 0))
        inst_1.set('gt_inds', pred_inst_1.inst_rel_ids)
        inst_1.set('im_inds', pred_inst_1.inst_im_inds)
        inst_1.set('mask_head_params', pred_inst_1.mask_head_params)

        # random sample locations and fpn_levels
        locations_1 = rel_ids_1.new_zeros((len(inst_1), 2))
        fpn_levels_1 = rel_ids_1.new_zeros(len(inst_1))

        for i in range(len(inst_1)):
            locations_pool = pred_instances_0.locations[torch.where(
                pred_instances_0.gt_inds == inst_1.gt_inds[i])]
            fpn_levels_pool = pred_instances_0.fpn_levels[torch.where(
                pred_instances_0.gt_inds == inst_1.gt_inds[i])]
            index = random.randint(0, locations_pool.size(0) - 1)
            locations_1[i, :] = locations_pool[index]
            fpn_levels_1[i] = fpn_levels_pool[index]

        inst_1.set('locations', locations_1)
        inst_1.set('fpn_levels', fpn_levels_1.long())

        loss_mask_cross_over_0 = self.mask_head(mask_feats_0,
                                                self.mask_branch.out_stride,
                                                inst_1,
                                                gt_instances_0)['loss_mask']

        loss_mask_cross_over_1 = self.mask_head(mask_feats_1,
                                                self.mask_branch.out_stride,
                                                inst_0,
                                                gt_instances_1)['loss_mask']

        return (loss_mask_cross_over_0 + loss_mask_cross_over_1) * .5

    def forward_test(self, batched_inputs):
        original_images = [x['image'].to(self.device) for x in batched_inputs]

        # normalize images
        images_norm = [self.normalizer(x) for x in original_images]
        images_norm = ImageList.from_tensors(images_norm,
                                             self.backbone.size_divisibility)

        features = self.backbone(images_norm.tensor)

        if 'instances' in batched_inputs[0]:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
            if self.boxinst_enabled:
                original_image_masks = [
                    torch.ones_like(x[0], dtype=torch.float32)
                    for x in original_images
                ]

                for i in range(len(original_image_masks)):
                    im_h = batched_inputs[i]['height']
                    pixels_removed = int(self.bottom_pixels_removed *
                                         float(original_images[i].size(1)) /
                                         float(im_h))
                    if pixels_removed > 0:
                        original_image_masks[i][-pixels_removed:, :] = 0

                original_images = ImageList.from_tensors(
                    original_images, self.backbone.size_divisibility)
                original_image_masks = ImageList.from_tensors(
                    original_image_masks,
                    self.backbone.size_divisibility,
                    pad_value=0.0)
                self.add_bitmasks_from_boxes(gt_instances,
                                             original_images.tensor,
                                             original_image_masks.tensor,
                                             original_images.tensor.size(-2),
                                             original_images.tensor.size(-1))
            else:
                self.add_bitmasks(gt_instances, images_norm.tensor.size(-2),
                                  images_norm.tensor.size(-1))
        else:
            gt_instances = None

        mask_feats, sem_losses = self.mask_branch(features, gt_instances)

        proposals, proposal_losses = self.proposal_generator(
            images_norm, features, gt_instances, self.controller)

        if self.training:
            mask_losses = self._forward_mask_heads_train(
                proposals, mask_feats, gt_instances)

            losses = {}
            losses.update(sem_losses)
            losses.update(proposal_losses)
            losses.update(mask_losses)
            return losses
        else:
            pred_instances_w_masks = self._forward_mask_heads_test(
                proposals, mask_feats)

            padded_im_h, padded_im_w = images_norm.tensor.size()[-2:]
            processed_results = []
            for im_id, (input_per_image, image_size) in enumerate(
                    zip(batched_inputs, images_norm.image_sizes)):
                height = input_per_image.get('height', image_size[0])
                width = input_per_image.get('width', image_size[1])

                instances_per_im = pred_instances_w_masks[
                    pred_instances_w_masks.im_inds == im_id]
                instances_per_im = self.postprocess(instances_per_im, height,
                                                    width, padded_im_h,
                                                    padded_im_w)

                processed_results.append({'instances': instances_per_im})

            return processed_results

    def _forward_mask_heads_train(self, proposals, mask_feats, gt_instances):
        # prepare the inputs for mask heads
        pred_instances = proposals['instances']

        assert (self.max_proposals == -1) or \
            (self.topk_proposals_per_im == -1), \
            'MAX_PROPOSALS and TOPK_PROPOSALS_PER_IM ' \
            'cannot be used at the same time.'
        if self.max_proposals != -1:
            if self.max_proposals < len(pred_instances):
                inds = torch.randperm(len(pred_instances),
                                      device=mask_feats.device).long()
                logger.info('clipping proposals from {} to {}'.format(
                    len(pred_instances), self.max_proposals))
                pred_instances = pred_instances[inds[:self.max_proposals]]
        elif self.topk_proposals_per_im != -1:
            num_images = len(gt_instances)

            kept_instances = []
            for im_id in range(num_images):
                instances_per_im = pred_instances[pred_instances.im_inds ==
                                                  im_id]
                if len(instances_per_im) == 0:
                    kept_instances.append(instances_per_im)
                    continue

                unique_gt_inds = instances_per_im.gt_inds.unique()
                num_instances_per_gt = max(
                    int(self.topk_proposals_per_im / len(unique_gt_inds)), 1)

                for gt_ind in unique_gt_inds:
                    instances_per_gt = instances_per_im[
                        instances_per_im.gt_inds == gt_ind]

                    if len(instances_per_gt) > num_instances_per_gt:
                        scores = instances_per_gt.logits_pred.sigmoid().max(
                            dim=1)[0]
                        ctrness_pred = instances_per_gt.ctrness_pred.sigmoid()
                        inds = (scores * ctrness_pred).topk(
                            k=num_instances_per_gt, dim=0)[1]
                        instances_per_gt = instances_per_gt[inds]

                    kept_instances.append(instances_per_gt)

            pred_instances = Instances.cat(kept_instances)

        pred_instances.mask_head_params = pred_instances.top_feats

        loss_mask = self.mask_head(mask_feats, self.mask_branch.out_stride,
                                   pred_instances, gt_instances)

        return loss_mask

    def _forward_mask_heads_test(self, proposals, mask_feats):
        # prepare the inputs for mask heads
        for im_id, per_im in enumerate(proposals):
            per_im.im_inds = per_im.locations.new_ones(
                len(per_im), dtype=torch.long) * im_id
        pred_instances = Instances.cat(proposals)
        pred_instances.mask_head_params = pred_instances.top_feat

        pred_instances_w_masks = self.mask_head(mask_feats,
                                                self.mask_branch.out_stride,
                                                pred_instances)

        return pred_instances_w_masks

    def add_bitmasks(self, instances, im_h, im_w):
        for per_im_gt_inst in instances:
            if not per_im_gt_inst.has('gt_masks'):
                continue
            start = int(self.mask_out_stride // 2)
            if isinstance(per_im_gt_inst.get('gt_masks'), PolygonMasks):
                polygons = per_im_gt_inst.get('gt_masks').polygons
                per_im_bitmasks = []
                per_im_bitmasks_full = []
                for per_polygons in polygons:
                    bitmask = polygons_to_bitmask(per_polygons, im_h, im_w)
                    bitmask = torch.from_numpy(bitmask).to(self.device).float()
                    start = int(self.mask_out_stride // 2)
                    bitmask_full = bitmask.clone()
                    bitmask = bitmask[start::self.mask_out_stride,
                                      start::self.mask_out_stride]

                    assert bitmask.size(0) * self.mask_out_stride == im_h
                    assert bitmask.size(1) * self.mask_out_stride == im_w

                    per_im_bitmasks.append(bitmask)
                    per_im_bitmasks_full.append(bitmask_full)

                per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks,
                                                         dim=0)
                per_im_gt_inst.gt_bitmasks_full = torch.stack(
                    per_im_bitmasks_full, dim=0)
            else:  # RLE format bitmask
                bitmasks = per_im_gt_inst.get('gt_masks').tensor
                h, w = bitmasks.size()[1:]
                # pad to new size
                bitmasks_full = F.pad(bitmasks, (0, im_w - w, 0, im_h - h),
                                      'constant', 0)
                bitmasks = bitmasks_full[:, start::self.mask_out_stride,
                                         start::self.mask_out_stride]
                per_im_gt_inst.gt_bitmasks = bitmasks
                per_im_gt_inst.gt_bitmasks_full = bitmasks_full

    def add_bitmasks_from_boxes(self, instances, images, image_masks, im_h,
                                im_w):
        stride = self.mask_out_stride
        start = int(stride // 2)

        assert images.size(2) % stride == 0
        assert images.size(3) % stride == 0

        downsampled_images = F.avg_pool2d(images.float(),
                                          kernel_size=stride,
                                          stride=stride,
                                          padding=0)[:, [2, 1, 0]]
        image_masks = image_masks[:, start::stride, start::stride]

        for im_i, per_im_gt_inst in enumerate(instances):
            images_lab = color.rgb2lab(downsampled_images[im_i].byte().permute(
                1, 2, 0).cpu().numpy())
            images_lab = torch.as_tensor(images_lab,
                                         device=downsampled_images.device,
                                         dtype=torch.float32)
            images_lab = images_lab.permute(2, 0, 1)[None]
            images_color_similarity = get_images_color_similarity(
                images_lab, image_masks[im_i], self.pairwise_size,
                self.pairwise_dilation)

            per_im_boxes = per_im_gt_inst.gt_boxes.tensor
            per_im_bitmasks = []
            per_im_bitmasks_full = []
            for per_box in per_im_boxes:
                bitmask_full = torch.zeros(
                    (im_h, im_w)).to(self.device).float()
                bitmask_full[int(per_box[1]):int(per_box[3] + 1),
                             int(per_box[0]):int(per_box[2] + 1)] = 1.0

                bitmask = bitmask_full[start::stride, start::stride]

                assert bitmask.size(0) * stride == im_h
                assert bitmask.size(1) * stride == im_w

                per_im_bitmasks.append(bitmask)
                per_im_bitmasks_full.append(bitmask_full)

            per_im_gt_inst.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
            per_im_gt_inst.gt_bitmasks_full = torch.stack(per_im_bitmasks_full,
                                                          dim=0)
            per_im_gt_inst.image_color_similarity = torch.cat(
                [images_color_similarity for _ in range(len(per_im_gt_inst))],
                dim=0)

    def postprocess(self,
                    results,
                    output_height,
                    output_width,
                    padded_im_h,
                    padded_im_w,
                    mask_threshold=0.5):
        """Resize the output instances.

        The input images are often resized when entering an object detector.
        As a result, we often need the outputs of the detector in a different
        resolution from its inputs.
        This function will resize the raw outputs of an R-CNN detector
        to produce outputs according to the desired output resolution.
        Args:
            results (Instances): the raw outputs from the detector.
                `results.image_size` contains the input image resolution.
                This object might be modified in-place.
            output_height, output_width: the desired output resolution.
        Returns:
            Instances: the resized output from the model,
            based on the output resolution
        """
        scale_x, scale_y = (output_width / results.image_size[1],
                            output_height / results.image_size[0])
        resized_im_h, resized_im_w = results.image_size
        results = Instances((output_height, output_width),
                            **results.get_fields())

        if results.has('pred_boxes'):
            output_boxes = results.pred_boxes
        elif results.has('proposal_boxes'):
            output_boxes = results.proposal_boxes

        output_boxes.scale(scale_x, scale_y)
        output_boxes.clip(results.image_size)

        results = results[output_boxes.nonempty()]

        if results.has('pred_global_masks'):
            mask_h, mask_w = results.pred_global_masks.size()[-2:]
            factor_h = padded_im_h // mask_h
            factor_w = padded_im_w // mask_w
            assert factor_h == factor_w
            factor = factor_h
            pred_global_masks = aligned_bilinear(results.pred_global_masks,
                                                 factor)
            pred_global_masks = pred_global_masks[:, :, :resized_im_h, :
                                                  resized_im_w]
            pred_global_masks = F.interpolate(pred_global_masks,
                                              size=(output_height,
                                                    output_width),
                                              mode='bilinear',
                                              align_corners=False)
            pred_global_masks = pred_global_masks[:, 0, :, :]
            results.pred_masks = (pred_global_masks > mask_threshold).float()

        return results
