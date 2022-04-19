import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss


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
        super(YOLOV3Head, self).__init__(num_classes, in_channels, out_channels,
                                         anchor_generator, bbox_coder, featmap_strides,
                                         one_hot_smoother, conv_cfg, norm_cfg, act_cfg,
                                         loss_cls, loss_conf, loss_xy, loss_wh,
                                         train_cfg, test_cfg, init_cfg)

        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(3*128, 1)

        # parameters for loss calculation
        self.margin = margin
        self.weight = weight



    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels

        # target module
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            x = self.convs_bridge[i](x)
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        # loss prediction module
        out1 = feats[0].mean(dim=(-2,-1))
        out1 = F.relu(self.fc1(out1))

        out2 = feats[1].mean(dim=(-2,-1))
        out2 = F.relu(self.fc2(out2))

        out3 = feats[2].mean(dim=(-2,-1))
        out3 = F.relu(self.fc3(out3))

        out = torch.cat((out1, out2, out3), dim=1)
        out = self.fc_out(out).squeeze()

        return tuple(pred_maps), out




    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             output,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            output (tuple[ list[Tensor], loss_prediction]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W); loss prediction, shape (N)
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

        pred_maps = output[0]
        loss_prediction = output[1]
        
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
