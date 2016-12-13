# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from utils.cython_bbox import bbox_overlaps
import pdb

DEBUG = False

def forwardAnImage(all_rois, gt_boxes, rois_per_image, _num_classes):
        # Proposal ROIs (x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        #all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        #gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois

        all_rois = np.vstack((all_rois, gt_boxes[:, :-1]))

        fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        if DEBUG:
            print 'all_rois.shape: {}'.format(all_rois.shape)
            print 'gt_boxes.shape: {}'.format(gt_boxes.shape)
        
        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, _num_classes)
        if DEBUG:
            print 'labels.shape:', labels.shape
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())

        blobs = {'labels':labels.reshape((-1,1)), 'rois':rois, 'bbox_targets':bbox_targets, 'bbox_inside_weights':bbox_inside_weights}
        return blobs

def proposal_target_layer(rpn_rois, gt_boxes,_num_classes):
	"""
	Assign object detection proposals to ground-truth targets. Produces proposal
	classification labels and bounding-box regression targets.
	"""

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        # bottom[0] : rpn_data.shape = (rpn num of all images, 5), line [batchnum, x1, y1, x2, y2]
        # bottom[1] : 'gt_boxes'.shape = (gtboxes num of all images, 6), line=[batchid,x1,y1,x2,y2,label]
        # bottom[2]: 'im_info'
        # top[0] : rois.shape = (roi_num, 4 + 1) per line [batchnum,x1,y1,x2,y2]
        # top[1] : labels.shape = (roi_num, 1) per line [score]
        # bbox_targets
        #top[2].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        #top[3].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        #top[4].reshape(1, self._num_classes * 4)
    
        all_gtboxes = gt_boxes
        all_rois = rpn_rois
        ims_num = len(set(gt_boxes[:,0])) # gt_boxes's num is less than rpn_proposal
        rois_per_image = cfg.TRAIN.BATCH_SIZE / ims_num

        if DEBUG:
            print 'all_gtboxes.shape=',all_gtboxes.shape
            print 'all_rois.shape=',all_rois.shape
        rois = []
        labels = []    
        bbox_targets = []
        bbox_inside_weights = []
        #bbox_outside_weights = []
        for i in range(ims_num):
            gbboxes_inds = np.where(all_gtboxes[:,0] == i)[0]
            all_rois_inds = np.where(all_rois[:,0] == i)[0]
            blobs = forwardAnImage(all_rois[all_rois_inds,1:],all_gtboxes[gbboxes_inds,1:],rois_per_image, _num_classes)
            rois.append(np.hstack( (np.ones((blobs['rois'].shape[0],1)) * i,blobs['rois']) ))
            labels.append(blobs['labels'])
            bbox_targets.append(blobs['bbox_targets'])
            bbox_inside_weights.append(blobs['bbox_inside_weights'])
            #bbox_outside_weights.append(blobs['bbox_outside_weights'])
        

        # sampled rois
        rois = np.vstack(rois).astype(np.float32)

        # classification labels
        labels = np.vstack(labels).astype(np.float32)

        # bbox_targets
        bbox_targets = np.vstack(bbox_targets).astype(np.float32)

        # bbox_inside_weights
        bbox_inside_weights = np.vstack(bbox_inside_weights).astype(np.float32)

        # bbox_outside_weights
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

        return rois,labels,bbox_targets,bbox_inside_weights,bbox_outside_weights

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)
    max_overlaps = overlaps.max(axis=1)
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = int(min(bg_rois_per_this_image, bg_inds.size))
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois, gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights
