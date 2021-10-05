import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square, xyxy2xywh_a, rbox_2_quad
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps
from utils.point_justify import pointsJf


# cuda_overlaps
class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func = 'smooth'):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss
        elif func == 'bi':
            self.criteron = boundary_invariant_loss
            
    def forward(self, classifications, regressions, anchors, annotations,iou_thres=0.5):
        
        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        all_pred_boxes = self.box_coder.decode(anchors, regressions, mode='xywht')
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            pred_boxes = all_pred_boxes[j, :, :]
            if bbox_annotation.shape[0] == 0:
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses.append(torch.tensor(0).float().cuda())
                continue
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )

            centers = xyxy2xywh_a(anchors[j, :, :])[:,:2].contiguous()
            polygons = torch.from_numpy(rbox_2_quad(bbox_annotation[:, :-1])).cuda()
            inside = torch.full([anchors[j, :, :].shape[0], bbox_annotation[:, :-1].shape[0]], 0.).cuda().float()
            pointsJf(centers,polygons,inside)
            
            ious = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                (inside * indicator).cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(ious):
                ious = torch.from_numpy(ious).cuda()
            
            iou_max, iou_argmax = torch.max(ious, dim=1)
           
            positive_indices = torch.ge(iou_max, iou_thres)

            max_gt, argmax_gt = ious.max(0) 
            if (max_gt < iou_thres).any():
                positive_indices[argmax_gt[max_gt < iou_thres]]=1
              
            # cls loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda()
            cls_targets[torch.lt(iou_max, iou_thres - 0.1), :] = 0
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1
            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(cls_targets * torch.log(classification+1e-6) + (1.0 - cls_targets) * torch.log(1.0 - classification+1e-6))
            cls_loss = focal_weight * bin_cross_entropy 
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            # reg loss
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                if self.criteron == boundary_invariant_loss:
                    preds = pred_boxes[positive_indices, :]
                    reg_loss = self.criteron(preds, gt_boxes)
                else:
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)

                if not torch.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
            else:
                reg_losses.append(torch.tensor(0).float().cuda())
        loss_cls = 5*torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = 2*torch.stack(reg_losses).mean(dim=0, keepdim=True)
        return loss_cls, loss_reg

    


class RegressLoss(nn.Module):
    def __init__(self, func='smooth'):
        super(RegressLoss, self).__init__()
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss
        elif func == 'bi':
            self.criteron = boundary_invariant_loss
            
    def forward(self, regressions, anchors, annotations, iou_thres=0.5):
        losses = []
        batch_size = regressions.shape[0]
        all_pred_boxes = self.box_coder.decode(anchors, regressions, mode='xywht')
        for j in range(batch_size):
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            pred_boxes = all_pred_boxes[j, :, :]
            if bbox_annotation.shape[0] == 0:
                losses.append(torch.tensor(0).float().cuda())
                continue
            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            overlaps = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                indicator.cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(overlaps):
                overlaps = torch.from_numpy(overlaps).cuda()

            iou_max, iou_argmax = torch.max(overlaps, dim=1)
            positive_indices = torch.ge(iou_max, iou_thres)
            # MaxIoU assigner
            max_gt, argmax_gt = overlaps.max(0) 
            if (max_gt < iou_thres).any():
                positive_indices[argmax_gt[max_gt < iou_thres]]=1

            assigned_annotations = bbox_annotation[iou_argmax, :]
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                targets = self.box_coder.encode(all_rois, gt_boxes)
                if self.criteron == boundary_invariant_loss:
                    preds = pred_boxes[positive_indices, :]
                    loss = self.criteron(preds, gt_boxes)
                else:
                    loss = self.criteron(regression[positive_indices, :], targets)
                losses.append(loss)
            else:
                losses.append(torch.tensor(0).float().cuda())
        return torch.stack(losses).mean(dim=0, keepdim=True)




def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    if  weight is  None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.unsqueeze(1)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()


def wh_iou(input, target):
    inter = torch.min(input[:, 0] ,target[:, 0]) * torch.min(input[:, 1] ,target[:, 1])
    union = input[:, 0] * input[:, 1]  + target[:, 0] * target[:, 1] - inter
    areac = torch.max(input[:, 0] ,target[:, 0]) * torch.max(input[:, 1] ,target[:, 1])
    hiou_loss = - torch.log( inter / (union + 1e-6) + 1e-6) + (areac - union) / (areac + 1e-6)
    return hiou_loss


def rotation_mapping(input, target):
    temp_ratios_w = torch.abs(input[:, 0] / (target[:, 0] + 1e-6))
    temp_thetas = input[:, -1] - target[:, -1] 
    ratios_w = torch.where(temp_ratios_w > torch.ones_like(temp_ratios_w),  1 / (temp_ratios_w + 1e-6), temp_ratios_w)
    dtheta = torch.where(temp_thetas > torch.zeros_like(temp_thetas), temp_thetas, -temp_thetas)
    delta_theta = torch.where(((dtheta % 180) > torch.zeros_like(dtheta)) & ((dtheta % 180) < (90 * torch.ones_like(dtheta))), \
        dtheta % 180, 180 - (dtheta % 180))
    rotation_metric = 1 / (1 + 1e-6 + ratios_w * torch.cos(delta_theta*3.1415926/180)) - 0.5
    return rotation_metric

def shape_mapping(input, target):
    loss = torch.min(
            torch.stack(
                [wh_iou(input[:, [1,0]], target[:, :2]), 
                 wh_iou(input[:, [0,1]], target[:, :2])], 1
            ), 1
    )[0]
    return loss    

def emd_loss(input, target):
    target_plus  = torch.cat([target[:, [1,0]], (target[:, -1] + 90).unsqueeze(1)], -1)
    target_minus = torch.cat([target[:, [1,0]], (target[:, -1] - 90).unsqueeze(1)], -1)
    loss = torch.min(
            torch.stack(
                [10*rotation_mapping(input, target_plus) + shape_mapping(input, target_plus),
                 10*rotation_mapping(input, target) + shape_mapping(input, target),
                 10*rotation_mapping(input, target_minus) + shape_mapping(input, target_minus)], 1
            ), 1
    )[0]
    return loss


def boundary_invariant_loss(inputs, targets):
    # inputs, targets: xyxya
    # delta_inputs, delta_targets: xywha 
    inputs_cx  = (inputs[:, 2]  + inputs[:, 0]) * 0.5
    inputs_cy  = (inputs[:, 3]  + inputs[:, 1]) * 0.5
    targets_cx  = (targets[:, 2]  + targets[:, 0]) * 0.5
    targets_cy  = (targets[:, 3]  + targets[:, 1]) * 0.5
    inputs_wht  = torch.stack([inputs[:, 2] -  inputs[:, 0],  inputs[:, 3] -  inputs[:, 1], inputs[:, 4]], 1)
    targets_wht = torch.stack([targets[:, 2] - targets[:, 0], targets[:, 3] - targets[:, 1], targets[:, 4]], 1)
    # center-metric
    temp_box_ratio = targets_wht[:, 0] / (targets_wht[:, 1] + 1e-6)
    box_ratios = torch.where(temp_box_ratio > torch.ones_like(temp_box_ratio), temp_box_ratio, 1 / (temp_box_ratio + 1e-6))
    center_dist = (inputs_cx - targets_cx)**2 + (inputs_cy - targets_cy)**2
    diagonal = (targets_wht[:, 0]**2 + targets_wht[:, 1]**2) * 0.25 
    center_metric = box_ratios * center_dist / (diagonal + 1e-6)
    # geometry-metric
    geometry_metric = emd_loss(inputs_wht, targets_wht)
    loss = 1*center_metric + 1*geometry_metric 
    # print(center_metric)
    # print(geometry_metric)
    # print('---')
    return loss.mean()
