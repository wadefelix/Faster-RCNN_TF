# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform

DEBUG = False 

class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        self._anchors = generate_anchors(cfg.TRAIN.RPN_BASE_SIZE, cfg.TRAIN.RPN_ASPECTS, cfg.TRAIN.RPN_SCALES)
        self._num_anchors = self._anchors.shape[0]

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        layer_params = yaml.load(self.param_str_)
        self._feat_stride = layer_params['feat_stride']

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        ims_num = bottom[0].data.shape[0]
        # labels
        top[0].reshape(ims_num, 1, A * height, width)
        # bbox_targets
        top[1].reshape(ims_num, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(ims_num, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(ims_num, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        # bottom[0] : rpn_cls_score.shape = (batchsize also imsperbatch, ..., H, W)
        # bottom[1] : gbboxes.shape = (gtboxes num of all images, 6), line=[batchid, x1,y1,x2,y2,label]
        # bottom[2] : im_info.shape = (batchsize also imsperbatch, 3) , per iminfo = [height,width,scale]
        # bottom[4] : data. REMOVED
        # top[0] : labels.shape = (batchsize also imsperbatch, 1, A * height, width))  # A = self._num_anchors
        # top[1] : bbox_targets.shape = (gtboxes num of all images,  A * 4, height, width)
        # top[2] : bbox_inside_weights.shape = (gtboxes num of all images,  A * 4, height, width)
        # top[3] : bbox_outside_weights.shape = (gtboxes num of all images,  A * 4, height, width)

        ims_num = bottom[0].data.shape[0]

        # map of shape (..., H, W)
        height, width = bottom[0].data.shape[-2:]
        # GT boxes (batchid, x1, y1, x2, y2, label)
        gt_boxes = bottom[1].data
        # im_info. the input images have the same size, so
        im_info = bottom[2].data[0, :]

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)
            print 'rpn: gt_boxes.shape', gt_boxes.shape
            print 'rpn: gt_boxes', gt_boxes

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < im_info[1] + self._allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + self._allowed_border)    # height
        )[0]

        if DEBUG:
            print 'total_anchors', total_anchors
            print 'inds_inside', len(inds_inside)

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        if DEBUG:
            print 'anchors.shape', anchors.shape

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside), ims_num), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes[:,1:5], dtype=np.float))

        argmax_overlaps = np.empty((len(inds_inside), ims_num),dtype=np.int32)
        max_overlaps = np.empty((len(inds_inside), ims_num),dtype=np.float)
        #gt_argmax_overlaps = np.zeros((len(inds_inside), ims_num))
        for i in range(ims_num):
            gts_of_im = np.where(gt_boxes[:,0]==i)[0]
            sub_ov = overlaps[:,gts_of_im]
            argmax_overlaps[:,i] = sub_ov.argmax(axis=1).ravel()
            max_overlaps[:,i] = sub_ov[np.arange(len(inds_inside)), argmax_overlaps[:,i]].ravel()
            sub_gt_argmax_overlaps = sub_ov.argmax(axis=0)
            labels[sub_gt_argmax_overlaps,i] = 1
            argmax_overlaps[:,i] = gts_of_im[argmax_overlaps[:,i]]

        #if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        #    # assign bg labels first so that positive labels can clobber them
        #    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        #if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        #    # assign bg labels last so that negative labels can clobber positives
        #    labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
        fg_inds_anchor, fg_inds_img = np.where(labels == 1)
        if len(fg_inds_anchor) > num_fg:
            disable_inds = npr.choice(
                np.arange(len(fg_inds_anchor)), size=(len(fg_inds_anchor) - num_fg), replace=False)
            labels[fg_inds_anchor[disable_inds],fg_inds_img[disable_inds]] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        bg_inds_anchor,bg_inds_img = np.where(labels == 0)
        if len(bg_inds_anchor) > num_bg:
            disable_inds = npr.choice(
                np.arange(len(bg_inds_anchor)), size=(len(bg_inds_anchor) - num_bg), replace=False)
            labels[bg_inds_anchor[disable_inds],bg_inds_img[disable_inds]] = -1

        bbox_targets = np.zeros((len(inds_inside), ims_num, 4), dtype=np.float32)
        for i in range(ims_num):
            bbox_targets[:,i,:] = _compute_targets(anchors, gt_boxes[argmax_overlaps[:,i], :])

        bbox_inside_weights = np.zeros((len(inds_inside), ims_num, 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), ims_num, 4), dtype=np.float32)
        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        if DEBUG:
            self._sums += bbox_targets[labels == 1, :].sum(axis=0)
            self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
            self._counts += np.sum(labels == 1)
            means = self._sums / self._counts
            stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
            print 'means:'
            print means
            print 'stdevs:'
            print stds

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        if DEBUG:
            if gt_boxes.shape[0] != 0:
                print 'rpn: max max_overlap', np.max(max_overlaps)
            else:
                print 'rpn: max max_overlap', 0
            print 'rpn: num_positive', np.sum(labels == 1)
            print 'rpn: num_negative', np.sum(labels == 0)
            self._fg_sum += np.sum(labels == 1)
            self._bg_sum += np.sum(labels == 0)
            self._count += 1
            print 'rpn: num_positive avg', self._fg_sum / self._count
            print 'rpn: num_negative avg', self._bg_sum / self._count

        # labels
        # (total_anchor, ims_num,) total_anchor=K*A,,k = width*height
        # TO (batchsize also imsperbatch, 1, A * height, width))
        labels = labels.reshape((height, width, A, ims_num)).transpose(3, 2, 0, 1)
        labels = labels.reshape((ims_num, 1, A * height, width))
        top[0].reshape(*labels.shape)
        top[0].data[...] = labels

        # bbox_targets
        # bbox_targets = bbox_targets.reshape((height, width, A * 4, ims_num)).transpose(3, 2, 0, 1)
        # the number 4 in transpose's param is not same as it in reshape.
        bbox_targets = bbox_targets.reshape((height, width, A, ims_num, 4)).transpose(3, 2, 4 , 0, 1).reshape(ims_num,A*4,height,width)
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape((height, width, A , ims_num, 4)).transpose(3, 2, 4, 0, 1).reshape(ims_num,A*4,height,width)
        assert bbox_inside_weights.shape[2] == height, "bbox_inside_weights.shape[2]={}, height={}".format(bbox_inside_weights.shape[2], height)
        assert bbox_inside_weights.shape[3] == width, "bbox_inside_weights.shape[3]={}, width={}".format(bbox_inside_weights.shape[3], width)
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape((height, width, A, ims_num, 4)).transpose(3, 2, 4, 0, 1).reshape(ims_num,A*4,height,width)
        assert bbox_outside_weights.shape[2] == height
        assert bbox_outside_weights.shape[3] == width
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 6

    return bbox_transform(ex_rois, gt_rois[:, 1:5]).astype(np.float32, copy=False)

