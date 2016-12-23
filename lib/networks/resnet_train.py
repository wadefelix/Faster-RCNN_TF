import tensorflow as tf
from networks.network import Network

# define

n_classes = 21
_feat_stride = [16, ]
anchor_scales = [8, 16, 32]


class resnet_base(Network):
    def __init__(self):
        pass

    def residual_block(self, input, output, input_depth, output_depth, projection=False, trainable=True,
                       padding='SAME'):
        firstconvstride = 1
        if input in ['res5a_branch2a_roipooling','res3d','res2c']:
            firstconvstride = 2
        if input not in ['pool1', 'res5a_branch2a_roipooling']:
            input = "{}_relu".format(input)
        outputkw = output[3:]

        inputForAdd = []

        if projection:
            # Option B: Projection shortcut
            (self.feed(input)
             .conv(1, 1, output_depth, firstconvstride, firstconvstride, biased=False, relu=False, name='{}_branch1'.format(output), padding=padding, bn=False)
             .batch_normalization(name='bn{}_branch1'.format(outputkw), is_training=False, relu=False))
            inputForAdd.append('bn{}_branch1'.format(outputkw))
        else:
            inputForAdd.append(input)

        (self.feed(input)
        .conv(1, 1, input_depth, firstconvstride, firstconvstride, biased=False, relu=False, name='{}_branch2a'.format(output), padding=padding, bn=False)
        .batch_normalization(relu=True, name='bn{}_branch2a'.format(outputkw), is_training=False)
        .conv(3, 3, input_depth, 1, 1, biased=False, relu=False, name='{}_branch2b'.format(output), bn=False)
        .batch_normalization(relu=True, name='bn{}_branch2b'.format(outputkw), is_training=False)
        .conv(1, 1, output_depth, 1, 1, biased=False, relu=False, name='{}_branch2c'.format(output), bn=False)
        .batch_normalization(relu=False, name='bn{}_branch2c'.format(outputkw), is_training=False))

        inputForAdd.append('bn{}_branch2c'.format(outputkw))

        (self.feed(*inputForAdd)
             .add(name=output, relu=False)
             .relu(name='{}_relu'.format(output))
             )
        return self


class resnet_train(resnet_base):
    def __init__(self, trainable=True, n=50):
        #super(self.__class__, self).__init__()

        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 6], name='gt_boxes')
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
         .conv(7, 7, 64, 2, 2, relu=False, name='conv1')
         .batch_normalization(relu=True, name='bn_conv1', is_training=False)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1'))

        (self.residual_block('pool1', 'res2a', 64, 256, projection=True, trainable=False)
         .residual_block('res2a', 'res2b', 64, 256, projection=False, trainable=False)
         .residual_block('res2b', 'res2c', 64, 256, projection=False, trainable=False)
         .residual_block('res2c', 'res3a', 128, 512, projection=True, trainable=True)
         .residual_block('res3a', 'res3b', 128, 512, projection=False, trainable=True, padding='VALID')
         .residual_block('res3b', 'res3c', 128, 512, projection=False, trainable=True)
         .residual_block('res3c', 'res3d', 128, 512, projection=False, trainable=True)
         .residual_block('res3d', 'res4a', 256, 1024, projection=True, trainable=True, padding='VALID')
         .residual_block('res4a', 'res4b', 256, 1024, projection=False, trainable=True)
         .residual_block('res4b', 'res4c', 256, 1024, projection=False, trainable=True)
         .residual_block('res4c', 'res4d', 256, 1024, projection=False, trainable=True)
         .residual_block('res4d', 'res4e', 256, 1024, projection=False, trainable=True)
         .residual_block('res4e', 'res4f', 256, 1024, projection=False, trainable=True)
         )

        # ========= RPN ============
        (self.feed('res4f_relu')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')
         .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred'))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
         .spatial_softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .spatial_reshape_layer(len(anchor_scales) * 3 * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name='rpn_rois'))

        (self.feed('rpn_rois', 'gt_boxes')
         .proposal_target_layer(n_classes, name='roi-data'))

        # ========= RCNN ============
        (self.feed('res4f_relu', 'roi-data')
         .roi_pool(7, 7, 1.0 / 16, name='res5a_branch2a_roipooling')
         .residual_block('res5a_branch2a_roipooling', 'res5a', 512, 2048, projection=True, trainable=True, padding='VALID')
         .residual_block('res5a', 'res5b', 512, 2048, projection=False, trainable=True)
         .residual_block('res5b', 'res5c', 512, 2048, projection=False, trainable=True)
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob')
         )
        (self.feed('res5c_relu')
         .fc(n_classes * 4, relu=False, name='bbox_pred'))
