from ..registry import DETECTORS
from .rbox_single_stage import RBoxSingleStageDetector


@DETECTORS.register_module
class QuadRetinaNet(RBoxSingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 rbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(QuadRetinaNet, self).__init__(backbone, neck, rbox_head, train_cfg,
                                        test_cfg, pretrained)
