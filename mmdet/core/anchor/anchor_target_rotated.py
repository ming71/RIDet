import torch

from ..bbox import (PseudoSampler, build_assigner, build_sampler,
                    poly2rbox_torch, rbox2delta, rbox2poly_torch,
                    rbox2rect_torch, rect2rbox, quad2delta)
from ..utils import multi_apply
from .anchor_target import images_to_levels, unmap


def anchor_target_rotated(anchor_list,
                          valid_flag_list,
                          gt_bboxes_list,
                          img_metas,
                          target_means,
                          target_stds,
                          cfg,
                          bbox_preds=None,
                          gt_bboxes_ignore_list=None,
                          gt_labels_list=None,
                          label_channels=1,
                          sampling=True,
                          unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    anchor_target_type = cfg.get('anchor_target_type', 'hbb_obb_bbox_overlap')
    if anchor_target_type in ['obb_quad_rbox_overlap', 'hbb_quad_rbox_overlap']:
        bbox_preds = torch.cat([x.permute(0, 2,3,1).reshape(num_imgs, -1, 8 ) \
                            for x in bbox_preds], 1)  
    else:
        bbox_preds = torch.cat([x.permute(0, 2,3,1).reshape(num_imgs, -1, 5 ) \
                            for x in bbox_preds], 1)  
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]

    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list) = multi_apply(
         anchor_target_single_warpper,
         anchor_list,
         valid_flag_list,
         gt_bboxes_list,
         gt_bboxes_ignore_list,
         gt_labels_list,
         img_metas,
         bbox_preds,
         target_means=target_means,
         target_stds=target_stds,
         cfg=cfg,
         label_channels=label_channels,
         sampling=sampling,
         unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = images_to_levels(all_labels, num_level_anchors)
    label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg)


def anchor_target_single_warpper(*args, **kwargs):
    cfg = kwargs.get('cfg')
    anchor_target_type = cfg.get('anchor_target_type', 'hbb_obb_bbox_overlap')
    if anchor_target_type == 'hbb_obb_bbox_overlap':
        return anchor_target_single_hbb_obb_bbox(*args, **kwargs)
    elif anchor_target_type == 'hbb_obb_rbox_overlap':
        return anchor_target_single_hbb_obb_rbox(*args, **kwargs)
    elif anchor_target_type == 'obb_obb_rbox_overlap':
        return anchor_target_single_obb_obb_rbox(*args, **kwargs)
    elif anchor_target_type == 'obb_quad_rbox_overlap':
        return anchor_target_single_obb_quad_rbox(*args, **kwargs)
    elif anchor_target_type == 'hbb_quad_rbox_overlap':
        return anchor_target_single_hbb_quad_rbox(*args, **kwargs)
    else:
        raise NotImplementedError

def anchor_target_single_hbb_obb_bbox(flat_anchors,
                                      valid_flags,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      img_meta,
                                      bbox_preds,
                                      target_means,
                                      target_stds,
                                      cfg,
                                      label_channels=1,
                                      sampling=True,
                                      unmap_outputs=True):
    gt_bboxes_hbb = None
    if gt_bboxes is not None:
        gt_bboxes_hbb = rbox2rect_torch(gt_bboxes)
    gt_bboxes_ignore_hbb = None
    if gt_bboxes_ignore is not None:
        gt_bboxes_ignore_hbb = rbox2rect_torch(gt_bboxes_ignore)

    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :].double()
    rbox_anchors = rect2rbox(anchors)

    if sampling:
        bbox_assigner = build_assigner(cfg.assigner)
        bbox_sampler = build_sampler(cfg.sampler)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes_hbb, gt_bboxes_ignore_hbb,
                                             gt_labels)
        sampling_result = bbox_sampler.sample(assign_result, rbox_anchors, gt_bboxes,
                                              gt_labels)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes_hbb,
                                             gt_bboxes_ignore_hbb, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, rbox_anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(rbox_anchors)
    bbox_weights = torch.zeros_like(rbox_anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_bbox_targets = rbox2delta(sampling_result.pos_bboxes,
                                    sampling_result.pos_gt_bboxes,
                                    target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_target_single_hbb_obb_rbox(flat_anchors,
                                      valid_flags,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      img_meta,
                                      bbox_preds,
                                      target_means,
                                      target_stds,
                                      cfg,
                                      label_channels=1,
                                      sampling=True,
                                      unmap_outputs=True):
    inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :].double()
    anchors = rect2rbox(anchors)

    if sampling:
        bbox_assigner = build_assigner(cfg.assigner)
        bbox_sampler = build_sampler(cfg.sampler)
        assign_result = bbox_assigner.assign(anchors, gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)
        sampling_result = bbox_sampler.sample(assign_result, anchors, gt_bboxes,
                                              gt_labels)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        bbox_preds = bbox_preds[inside_flags]
        if 'Das' in cfg.assigner.type:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,bbox_preds,
                                                 gt_bboxes_ignore, gt_labels)
                                                 
        else:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        if cfg.get('calc_offset') == False:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        else:
            pos_bbox_targets = rbox2delta(sampling_result.pos_bboxes,
                                        sampling_result.pos_gt_bboxes,
                                        target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_target_single_obb_obb_rbox(flat_anchors,
                                      valid_flags,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      img_meta,
                                      bbox_preds,
                                      target_means,
                                      target_stds,
                                      cfg,
                                      label_channels=1,
                                      sampling=True,
                                      unmap_outputs=True):
    """
    flat_anchors and gt_bboxes are all rboxes
    """
    """
    DEBUG ONLY CODE
    """
    if cfg.get('anchor_inside_type', 'border') == 'center':
        anchor_inside_flags_func = rotated_anchor_center_inside_flags
    else:
        anchor_inside_flags_func = rotated_anchor_inside_flags

    inside_flags = anchor_inside_flags_func(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :].double()

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        # assign_result = bbox_assigner.assign(anchors, gt_bboxes,
        #                                      gt_bboxes_ignore, gt_labels)
        bbox_preds = bbox_preds[inside_flags]
        if 'Das' in cfg.assigner.type:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,bbox_preds,
                                                 gt_bboxes_ignore, gt_labels)
        else:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        if cfg.get('calc_offset') == False:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        else:
            pos_bbox_targets = rbox2delta(sampling_result.pos_bboxes,
                                        sampling_result.pos_gt_bboxes,
                                        target_means, target_stds)
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_target_single_obb_quad_rbox(flat_anchors,
                                      valid_flags,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      img_meta,
                                      bbox_preds,
                                      target_means,
                                      target_stds,
                                      cfg,
                                      label_channels=1,
                                      sampling=True,
                                      unmap_outputs=True):
    """
    flat_anchors and gt_bboxes are all rboxes
    """
    """
    DEBUG ONLY CODE
    """
    if cfg.get('anchor_inside_type', 'border') == 'center':
        anchor_inside_flags_func = rotated_anchor_center_inside_flags
    else:
        anchor_inside_flags_func = rotated_anchor_inside_flags

    inside_flags = anchor_inside_flags_func(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :].double()

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        bbox_preds = bbox_preds[inside_flags]
        if 'Das' in cfg.assigner.type:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,bbox_preds,
                                                 gt_bboxes_ignore, gt_labels)
        else:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(bbox_preds, dtype=torch.float64)
    bbox_weights = torch.zeros_like(bbox_preds, dtype=torch.float64)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    if len(pos_inds) > 0:
        pos_quads = rbox2poly_torch(sampling_result.pos_bboxes)
        pos_gt_quads = rbox2poly_torch(sampling_result.pos_gt_bboxes)
        if cfg.get('calc_offset') == False:
            pos_bbox_targets = pos_gt_quads
        else:
            pos_bbox_targets = quad2delta(pos_quads,    # proposals_quad
                                        sampling_result.pos_bboxes,  # proposals_rbox
                                        pos_gt_quads    # gt_quad
                                        )
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)



def anchor_target_single_hbb_quad_rbox(flat_anchors,
                                      valid_flags,
                                      gt_bboxes,
                                      gt_bboxes_ignore,
                                      gt_labels,
                                      img_meta,
                                      bbox_preds,
                                      target_means,
                                      target_stds,
                                      cfg,
                                      label_channels=1,
                                      sampling=True,
                                      unmap_outputs=True):
    """
    flat_anchors and gt_bboxes are all rboxes
    """
    """
    DEBUG ONLY CODE
    """

    if cfg.get('anchor_inside_type', 'border') == 'center':
        anchor_inside_flags_func = rotated_anchor_center_inside_flags
    else:
        anchor_inside_flags_func = rotated_anchor_inside_flags

    inside_flags = anchor_inside_flags_func(flat_anchors, valid_flags,
                                            img_meta['img_shape'][:2],
                                            cfg.allowed_border)
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :].double()
    anchors = rect2rbox(anchors)

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        bbox_preds = bbox_preds[inside_flags]
        if 'Das' in cfg.assigner.type:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,bbox_preds,
                                                 gt_bboxes_ignore, gt_labels)
        else:
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    bbox_targets = torch.zeros_like(bbox_preds, dtype=torch.float64)
    bbox_weights = torch.zeros_like(bbox_preds, dtype=torch.float64)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds

    if len(pos_inds) > 0:
        pos_quads = rbox2poly_torch(sampling_result.pos_bboxes)
        pos_gt_quads = rbox2poly_torch(sampling_result.pos_gt_bboxes)
        if cfg.get('calc_offset') == False:
            pos_bbox_targets = pos_gt_quads
        else:
            pos_bbox_targets = quad2delta(pos_quads,    # proposals_quad
                                        sampling_result.pos_bboxes,  # proposals_rbox
                                        pos_gt_quads,    # gt_quad
                                        )
        bbox_targets[pos_inds, :] = pos_bbox_targets
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = unmap(labels, num_total_anchors, inside_flags)
        label_weights = unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds)


def anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def rotated_anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                                allowed_border=0):
    """
    flat_anchors.shape=[N,5].x,y,w,h,a
    """
    img_h, img_w = img_shape[:2]
    poly_anchors = rbox2poly_torch(flat_anchors)
    min_xs, max_xs, min_ys, max_ys = poly_anchors[:, ::2].min(dim=1).values,\
        poly_anchors[:, ::2].max(dim=1).values,\
        poly_anchors[:, 1::2].min(dim=1).values,\
        poly_anchors[:, 1::2].max(dim=1).values
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (min_xs >= -allowed_border).type(torch.uint8) & \
            (min_ys >= -allowed_border).type(torch.uint8) & \
            (max_xs < img_w + allowed_border).type(torch.uint8) & \
            (max_ys < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def rotated_anchor_center_inside_flags(flat_anchors, valid_flags, img_shape,
                                       allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
            (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags
