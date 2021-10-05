import torch
import torch.nn as nn
import torchvision.models as models
import math
import cv2
import os
import shutil
import numpy as np

from models.fpn import FPN, LastLevelP6P7
from models.heads import CLSHead, REGHead #, MultiHead
from models.anchors import Anchors
from models.losses import IntegratedLoss, RegressLoss #, KLLoss

from utils.nms_wrapper import nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes, rbox_2_quad
from utils.utils import hyp_parse


class RetinaNet(nn.Module):
    def __init__(self,backbone='res50',hyps=None):
        super(RetinaNet, self).__init__()
        self.num_refining = int(hyps['num_refining'])
        self.num_classes  = int(hyps['num_classes']) + 1
        self.anchor_generator = Anchors(
            ratios = np.array([0.2, 0.5, 1, 2.0, 5.0]),
            scales = np.array([ 2**0, 2 ** 0.333, 2 ** 0.666]),
            # rotations = np.array([0, 60, 120])
            )
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        
        self.fpn = FPN(
            in_channels_list=self.fpn_in_channels,
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256),
            use_asff = False
        )
        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,      
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5   # xywha
        )     
        if self.num_refining > 0:
            self.ref_heads = nn.ModuleList(
                [REGHead(
                    in_channels=256,
                    feat_channels=256,
                    num_stacked=2,
                    num_anchors=self.num_anchors,
                    num_regress=5,
                ) for _ in range(self.num_refining)]
            )
            self.loss_ref = RegressLoss(func='smooth')   
        self.loss = IntegratedLoss(func='bi')
        self.box_coder = BoxCoder()

    def init_backbone(self, backbone):
        if backbone == 'res34':
            self.backbone = models.resnet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
        elif backbone == 'res50':
            self.backbone = models.resnet50(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        elif backbone == 'res101':
            self.backbone = models.resnet101(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048] 
        elif backbone == 'res152':
            self.backbone = models.resnet152(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048] 
        elif backbone == 'resnext50':
            self.backbone = models.resnext50_32x4d(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError
        del self.backbone.avgpool
        del self.backbone.fc

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        #c_i shape: bs,C,H,W
        return [c3, c4, c5]

    def forward(self, ims, gt_boxes=None, test_conf=None,process=None):
        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(ims)   # (bs, num_all_achors, 5)
        anchors_list.append(original_anchors)
        features = self.fpn(self.ims_2_features(ims))
        if self.num_refining > 0:
            for i in range(self.num_refining):
                bbox_pred = torch.cat([self.ref_heads[i](feature) for feature in features], dim=1)
                refined_anchors = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()
                anchors_list.append(refined_anchors)
                offsets_list.append(bbox_pred)
        cls_score = torch.cat([self.cls_head(feature) for feature in features], dim=1)
        bbox_pred = torch.cat([self.reg_head(feature) for feature in features], dim=1)
        bboxes = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()

        if self.training:
            losses = dict()
            if self.num_refining > 0:
                ref_losses = []
                for i in range(self.num_refining):
                    ref_losses.append(self.loss_ref(
                        offsets_list[i], anchors_list[i], gt_boxes, 
                        iou_thres=(0.3 + i * 0.1),
                        )
                    )
                losses['loss_ref'] = 0.5*torch.stack(ref_losses).mean(dim=0, keepdim=True) 

            losses['loss_cls'], losses['loss_reg'] = self.loss(cls_score, bbox_pred, anchors_list[-1], gt_boxes, \
                                                               iou_thres=0.5)
            return losses

        else:   # eval() mode
            return self.decoder(ims, anchors_list[-1], cls_score, bbox_pred, test_conf=test_conf)


    def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=0.5, nms_thresh=0.1, test_conf=None):
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        # NMS
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()





if __name__ == '__main__':
    pass
