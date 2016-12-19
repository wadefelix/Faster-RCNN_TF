import tensorflow as tf
from networks.network import Network

# define

n_classes = 21
_feat_stride = [16, ]
anchor_scales = [8, 16, 32]


class resnet_base(Network):
    def __init__(self):
        pass

    def residual_block(self, input, output, input_depth, output_depth, projection=False, trainable=True):
        #(self.feed(input)
        # .conv(1, 1, input_depth, 1, 1, name='{}_branch2a'.format(output), trainable=trainable, bn=True, relu=True)
        # .conv(3, 3, input_depth, 1, 1, name='{}_branch2b'.format(output), trainable=trainable, bn=True, relu=True)
        # .conv(1, 1, output_depth, 1, 1, name='{}_branch2c'.format(output), trainable=trainable, bn=True, relu=False))
        if input.startswith("res"):
            input = "{}_relu".format(input)
        (self.feed(input)
        .conv(1, 1, input_depth, 1, 1, biased=False, relu=False, name='{}_branch2a'.format(output))
        .batch_normalization(relu=True, name='bn{}_branch2a'.format(output[3:]), is_training=False)
        .conv(3, 3, input_depth, 1, 1, biased=False, relu=False, name='{}_branch2b'.format(output))
        .batch_normalization(relu=True, name='bn{}_branch2b'.format(output[3:]), is_training=False)
        .conv(1, 1, output_depth, 1, 1, biased=False, relu=False, name='{}_branch2c'.format(output))
        .batch_normalization(relu=False, name='bn{}_branch2c'.format(output[3:]), is_training=False))

        if projection:
            # Option B: Projection shortcut
            #(self.feed(input)
            # .conv(1, 1, output_depth, 1, 1, name='{}_branch1'.format(output), trainable=trainable, bn=True,
            #       relu=False))
            (self.feed(input)
             .conv(1, 1, output_depth, 1, 1, biased=False, relu=False, name='{}_branch1'.format(output))
             .batch_normalization(name='bn{}_branch1'.format(output[3:]), is_training=False, relu=False))
            (self.feed('{}_branch1'.format(output), '{}_branch2c'.format(output))
             .add(name=output, relu=False)
             .relu(name='{}_relu'.format(output)))
        else:
            # Option A: Zero-padding
            (self.feed(input, '{}_branch2c'.format(output))
             #.add(name=output, relu=True)
             .add(name=output, relu=False)
             .relu(name='{}_relu'.format(output))
             )
        return self


class resnet_train(resnet_base):
    def __init__(self, trainable=True, n=50):
        super(self.__class__, self).__init__()

        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 6])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        #
        (self.feed('data')
         #.conv(7, 7, 64, 2, 2, name='conv1', trainable=False, bn=True, relu=True)
         .conv(7, 7, 64, 2, 2, name='conv1', trainable=False, bn=False, relu=False, biased=False)
         .batch_normalization(relu=True, name='bn_conv1', is_training=False)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1'))
        # (self.feed('pool1')
        #      .conv(7, 7, 64, 2, 2, name='res2a_branch1', trainable=False, bn=True, relu=False))
        # (self.feed('pool1')
        #      .conv(1, 1, 256, 1, 1, name='res2a_branch1', trainable=False, bn=True, relu=False))
        # (self.feed('pool1')
        #      .conv(1, 1, 64, 1, 1, name='res2a_branch2a', trainable=False, bn=True, relu=True)
        #      .conv(3, 3, 64, 1, 1, name='res2a_branch2b', trainable=False, bn=True, relu=True)
        #      .conv(1, 1, 256, 1, 1, name='res2a_branch2c', trainable=False, bn=True, relu=False))
        # (self.feed('res2a_branch1','res2a_branch2c')
        #      .eltwise_add("res2a"))
        (self.residual_block('pool1', 'res2a', 64, 256, projection=True, trainable=False)
         .residual_block('res2a', 'res2b', 64, 256, projection=False, trainable=False)
         .residual_block('res2b', 'res2c', 64, 256, projection=False, trainable=False)
         .residual_block('res2c', 'res3a', 128, 512, projection=True, trainable=True)
         .residual_block('res3a', 'res3b', 128, 512, projection=False, trainable=True)
         .residual_block('res3b', 'res3c', 128, 512, projection=False, trainable=True)
         .residual_block('res3c', 'res3d', 128, 512, projection=False, trainable=True)
         .residual_block('res3d', 'res4a', 256, 1024, projection=True, trainable=True)
         .residual_block('res4a', 'res4b', 256, 1024, projection=False, trainable=True)
         .residual_block('res4b', 'res4c', 256, 1024, projection=False, trainable=True)
         .residual_block('res4c', 'res4d', 256, 1024, projection=False, trainable=True)
         .residual_block('res4d', 'res4e', 256, 1024, projection=False, trainable=True)
         .residual_block('res4e', 'res4f', 256, 1024, projection=False, trainable=True)
         )

        # ========= RPN ============
        (self.feed('res4f')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', relu=True, bn=False)
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score', bn=False))

        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')
         .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred', bn=False))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes')
         .proposal_target_layer(n_classes, name='roi-data'))

        # ========= RCNN ============
        # (self.feed('res4f', 'roi-data')
        #      .roi_pool(7, 7, 1.0/16, name='pool_5')
        #      .fc(4096, name='fc6')
        #      .dropout(0.5, name='drop6')
        #      .fc(4096, name='fc7')
        #      .dropout(0.5, name='drop7')
        #      .fc(n_classes, relu=False, name='cls_score')
        #      .softmax(name='cls_prob'))
        #
        # (self.feed('drop7')
        #      .fc(n_classes*4, relu=False, name='bbox_pred'))

        (self.feed('res4f', 'roi-data')
         .roi_pool(7, 7, 1.0 / 16, name='roi_pool')
         .residual_block('roi_pool', 'res5a', 512, 2048, projection=True, trainable=True)
         .residual_block('res5a', 'res5b', 512, 2048, projection=False, trainable=True)
         .residual_block('res5b', 'res5c', 512, 2048, projection=False, trainable=True)
         #.avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob')
         )
        (self.feed('res5c_relu')
         .fc(n_classes * 4, relu=False, name='bbox_pred'))
