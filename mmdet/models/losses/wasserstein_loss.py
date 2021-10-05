import torch
import numpy as np
import scipy.linalg
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Function
from .utils import weighted_loss
from ...core.bbox.transforms_rbox import poly2rbox_torch

from ..registry import LOSSES

grads = {}

def get_param_from_rect(box):
    cx, cy, w, h, theta = torch.chunk(box, 5)
    theta = theta/180*3.1415926
    mean = box[:2]
    cov = torch.cat([
        torch.pow(w, 2) * torch.pow(torch.cos(theta),2) + torch.pow(h,2) * torch.pow(torch.sin(theta),2),
        (torch.pow(w, 2) - torch.pow(h, 2)) * torch.sin(theta) * torch.cos(theta),
        (torch.pow(w, 2) - torch.pow(h, 2)) * torch.sin(theta) * torch.cos(theta),
        torch.pow(w, 2) * torch.pow(torch.sin(theta),2) + torch.pow(h, 2) * torch.pow(torch.cos(theta),2)
    ]).reshape(2,2)*0.25
    # import ipdb;ipdb.set_trace()
    # if box.requires_grad:
        # mean.register_hook(save_grad('mean'))
        # cov.register_hook(save_grad('cov'))
    # print(grads['mean'], grads['cov'])
    return mean, cov


class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply

# @weighted_loss
def wasserstein_dist(pbox, tbox):
    m1, cov1 = get_param_from_rect(pbox)
    m2, cov2 = get_param_from_rect(tbox)
    item1 = torch.norm(m1-m2, 2)**2
    item2 = torch.trace(cov1 + cov2 - \
        2 * sqrtm( torch.mm(torch.mm(sqrtm(cov1), cov2), sqrtm(cov1)) ))
    dist = torch.sqrt(torch.clamp(item1 + item2 + 1e-8, 0., 1e6))
    l_gwd = 1.0 - (1.0 / (dist + 2))
    return l_gwd


@LOSSES.register_module
class WassersteinLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(WassersteinLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if    reduction_override else self.reduction)
        wdists = []
        valid_idx = weight.nonzero()[:,0].unique()
        if len(valid_idx) == 0:
            return torch.tensor(0).float().cuda()
            
        if pred.size(1) == 5:   # obb
            valid_pred = pred[valid_idx]
            valid_target = target[valid_idx].float()
        elif pred.size(1) == 8:   # quad2obb
            valid_pred = poly2rbox_torch(pred[valid_idx])
            valid_target = poly2rbox_torch(target[valid_idx]).float()
        else:
            raise NotImplementedError

        for pbox, tbox in zip(valid_pred, valid_target):
            wdist = wasserstein_dist(pbox, tbox)
            wdists.append(wdist)
        if len(wdists)==0:
            return torch.tensor(0).float().cuda()
        return torch.stack(wdists, 0).sum() * self.loss_weight / avg_factor 
