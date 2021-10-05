import math
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ..registry import LOSSES
from .utils import weighted_loss

def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


## for obb loss
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
    dtheta = torch.where(temp_thetas > torch.zeros_like(temp_thetas), temp_thetas, -temp_thetas) % math.pi
    delta_theta = torch.where((dtheta > torch.zeros_like(dtheta)) & (dtheta < (math.pi * 0.5 * torch.ones_like(dtheta))), \
        dtheta, math.pi - dtheta)
    rotation_metric = 1 / (1 + 1e-6 + ratios_w * torch.cos(delta_theta)) - 0.5
    return rotation_metric

def shape_mapping(input, target):
    loss = torch.min(
            torch.stack(
                [wh_iou(input[:, [1,0]], target[:, :2]), 
                 wh_iou(input[:, [0,1]], target[:, :2])], 1
            ), 1
    )[0]
    return loss    

def hungarian_shape(input, target):
    target_plus  = torch.cat([target[:, [1,0]], (target[:, -1] + math.pi * 0.5).unsqueeze(1)], -1)
    loss = torch.min(
            torch.stack(
                [10*rotation_mapping(input, target_plus) + 0.1 * shape_mapping(input, target_plus),
                 10*rotation_mapping(input, target) + 0.1 * shape_mapping(input, target)]
            , 1 ), 1
    )[0]
    return loss


##### main losses
def hungarian_loss_obb(inputs, targets):
    # center-metric
    temp_box_ratio = targets[:, 2] / (targets[:, 3] + 1e-6)
    box_ratios = torch.where(temp_box_ratio > torch.ones_like(temp_box_ratio), temp_box_ratio, 1 / (temp_box_ratio + 1e-6))
    smoothl1 = True
    if smoothl1:
        center_dist = smooth_l1_loss(inputs[:, :2], targets[:, :2]).sum(1) 
    else:
        center_dist = (inputs[:, 0] - targets[:, 0])**2 + (inputs[:, 1] - targets[:, 1])**2
    diagonal = (targets[:, 2]**2 + targets[:, 3]**2) 
    center_metric = box_ratios * 0.25 * center_dist / (diagonal + 1e-6)
    # geometry-metric
    geometry_metric = hungarian_shape(inputs[:, 2:], targets[:, 2:])
    loss = center_metric + geometry_metric 
    return loss


def hungarian_loss_quad(inputs, targets):
    quad_inputs  = inputs.reshape(-1, 4, 2)
    quad_targets = targets.reshape(-1, 4, 2)
    losses = torch.stack(
        [smooth_l1_loss(quad_inputs, quad_targets[:, i, :].unsqueeze(1).repeat(1, 4, 1)).sum(2) \
            for i in range(4)] , 1
            )
    indices = [linear_sum_assignment(loss.cpu().detach().numpy()) for loss in losses]
    match_loss = []
    for cnt, (row_ind,col_ind) in enumerate(indices):
        match_loss.append(losses[cnt, row_ind, col_ind])
    return torch.stack(match_loss).sum(1)
#####

@LOSSES.register_module
class HungarianLoss(nn.Module):

    def __init__(self, beta=1.0, reduction='mean', form='obb', loss_weight=1.0):
        super(HungarianLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.form = form

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        valid_idx = weight.nonzero()[:,0].unique()
        if len(valid_idx) == 0:
            return torch.tensor(0).float().cuda()
        else:
            if self.form == 'obb':
                loss = hungarian_loss_obb(pred[valid_idx], target[valid_idx].float()) * self.loss_weight
            elif self.form == 'quad':
                loss = hungarian_loss_quad(pred[valid_idx], target[valid_idx].float()) * self.loss_weight
            else:
                raise NotImplementedError
            return loss.sum() / avg_factor
