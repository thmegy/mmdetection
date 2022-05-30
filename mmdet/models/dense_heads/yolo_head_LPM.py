import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .yolo_head import YOLOV3Head


@HEADS.register_module()
class YOLOV3HeadLPM(YOLOV3Head):
    """YOLOV3Head supplemented by Loss Prediction Module.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 out_channels=(1024, 512, 256),
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_xy=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_wh=dict(type='MSELoss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal', std=0.01,
                     override=dict(name='convs_pred')),
                 margin=1.,
                 weight=1.):
        super(YOLOV3HeadLPM, self).__init__(num_classes, in_channels, out_channels,
                                         anchor_generator, bbox_coder, featmap_strides,
                                         one_hot_smoother, conv_cfg, norm_cfg, act_cfg,
                                         loss_cls, loss_conf, loss_xy, loss_wh,
                                         train_cfg, test_cfg, init_cfg)

        self.fc1_neck = nn.Linear(512, 128)
        self.fc2_neck = nn.Linear(256, 128)
        self.fc3_neck = nn.Linear(128, 128)
        self.fc1_backbone = nn.Linear(256, 128)
        self.fc2_backbone = nn.Linear(512, 128)
        self.fc3_backbone = nn.Linear(1024, 128)
        self.fc_out = nn.Linear(6*128, 1)

        # parameters for loss calculation
        self.margin = margin
        self.weight = weight



    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple(tuple[Tensor], tuple[Tensor])): Features from the upstream network, each is
                a 4D-tensor. (Output of neck (N=3), output of backbone (N=3)).

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
            Tensor: loss prediction for each batch image
        """
        feats_neck = feats[0]
        feats_backbone = feats[1]
        assert len(feats_neck) == self.num_levels

        # target module
        pred_maps = []
        for i in range(self.num_levels):
            x = feats_neck[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        # loss prediction module
        out1_neck = feats_neck[0].mean(dim=(-2,-1))
        out1_neck = F.relu(self.fc1_neck(out1_neck))

        out2_neck = feats_neck[1].mean(dim=(-2,-1))
        out2_neck = F.relu(self.fc2_neck(out2_neck))

        out3_neck = feats_neck[2].mean(dim=(-2,-1))
        out3_neck = F.relu(self.fc3_neck(out3_neck))

        out1_backbone = feats_backbone[0].mean(dim=(-2,-1))
        out1_backbone = F.relu(self.fc1_backbone(out1_backbone))

        out2_backbone = feats_backbone[1].mean(dim=(-2,-1))
        out2_backbone = F.relu(self.fc2_backbone(out2_backbone))

        out3_backbone = feats_backbone[2].mean(dim=(-2,-1))
        out3_backbone = F.relu(self.fc3_backbone(out3_backbone))

        out = torch.cat((out1_neck, out2_neck, out3_neck,
                         out1_backbone, out2_backbone, out3_backbone), dim=1)
        out = self.fc_out(out).squeeze()

        return tuple(pred_maps), out



    @force_fp32(apply_to=('pred_maps', 'loss_prediction'))
    def loss(self,
             pred_maps,
             loss_prediction,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            loss_prediction (Tensor): loss prediction, shape (N)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device
        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [mlvl_anchors for _ in range(num_imgs)]

        responsible_flag_list = []
        for img_id in range(len(img_metas)):
            responsible_flag_list.append(
                self.prior_generator.responsible_flags(featmap_sizes,
                                                       gt_bboxes[img_id],
                                                       device))

        target_maps_list, neg_maps_list = self.get_targets(
            anchor_list, responsible_flag_list, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_xy, losses_wh = multi_apply(
            self.loss_single, pred_maps, target_maps_list, neg_maps_list)

        # compute losses per event: sum over bboxes, then sum over levels (small, medium, large)
        losses_cls = sum([_loss.sum(dim=(1,2)) for _loss in losses_cls])
        losses_conf = sum([_loss.sum(dim=-1) for _loss in losses_conf])
        losses_xy = sum([_loss.sum(dim=(1,2)) for _loss in losses_xy])
        losses_wh = sum([_loss.sum(dim=(1,2)) for _loss in losses_wh])

        # compute target loss --> sum
        loss_target = losses_cls + losses_conf + losses_xy + losses_wh

        # compute loss-prediction-module loss
        loss_pred_pairs = loss_prediction.reshape(-1,2)
        loss_target_pairs = loss_target.reshape(-1,2)
        
        ones = torch.ones(loss_target_pairs.shape[0], dtype=torch.double).to(device)
        pm = torch.where(loss_target_pairs[:,0] > loss_target_pairs[:,1], ones, -1.)
        
        loss_module = -1 * pm * (loss_pred_pairs[:,0] - loss_pred_pairs[:,1]) + self.margin
        loss_module = torch.where(loss_module > 0, loss_module, 0.)

        loss_module = loss_module * self.weight
        
        return dict(
            loss_target=loss_target,
            loss_module=loss_module
        )



    @force_fp32(apply_to=('pred_maps', 'loss_prediction'))
    def get_bboxes(self,
                   pred_maps,
                   loss_prediction,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   **kwargs):
        """Transform network output for a batch into bbox predictions. It has
        been accelerated since PR #5991.

        Args:
            pred_maps (tuple[Tensor]): Raw predictions for a batch of images.
            loss_prediction (Tensor): Loss prediction for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """

        if 'active_learning' in kwargs and kwargs['active_learning']:            
            return loss_prediction

        scale_factors = [img_meta['scale_factor'] for img_meta in img_metas]
        num_imgs = len(img_metas)
        cfg = self.test_cfg if cfg is None else cfg

        assert len(pred_maps) == self.num_levels
        featmap_sizes = [pred_map.shape[-2:] for pred_map in pred_maps]

        mlvl_anchors = self.prior_generator.grid_priors(
            featmap_sizes, device=pred_maps[0].device)

        flatten_preds = []
        flatten_strides = []
        for pred, stride in zip(pred_maps, self.featmap_strides):
            pred = pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                    self.num_attrib)
            pred[..., :2].sigmoid_()
            flatten_preds.append(pred)
            flatten_strides.append(
                pred.new_tensor(stride).expand(pred.size(1)))

        flatten_preds = torch.cat(flatten_preds, dim=1)
        flatten_bbox_preds = flatten_preds[..., :4]
        flatten_objectness = flatten_preds[..., 4].sigmoid()
        flatten_cls_scores = flatten_preds[..., 5:].sigmoid()
            
        flatten_anchors = torch.cat(mlvl_anchors)
        flatten_strides = torch.cat(flatten_strides)
        flatten_bboxes = self.bbox_coder.decode(flatten_anchors,
                                                flatten_bbox_preds,
                                                flatten_strides.unsqueeze(-1))

        
        if with_nms and (flatten_objectness.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            flatten_bboxes /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        padding = flatten_bboxes.new_zeros(num_imgs, flatten_bboxes.shape[1],
                                           1)
        flatten_cls_scores = torch.cat([flatten_cls_scores, padding], dim=-1)

        det_results = []
        for (bboxes, scores, objectness) in zip(flatten_bboxes,
                                                flatten_cls_scores,
                                                flatten_objectness):
            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            if conf_thr > 0:
                conf_inds = objectness >= conf_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            det_bboxes, det_labels = multiclass_nms(
                bboxes,
                scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=objectness)
            det_results.append(tuple([det_bboxes, det_labels]))
        return det_results
    
