import torch

from ..builder import DETECTORS
from .yolo import YOLOV3


@DETECTORS.register_module()
class YOLOV3LPM(YOLOV3):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOV3LPM, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x_backbone = self.backbone(img)
        x_neck = self.neck(x_backbone)
        return x_neck, x_backbone
