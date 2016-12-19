import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]


class resnet_test(Network):
    def __init__(self, trainable=True, n = 50):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info})
        self.trainable = trainable
        self.setup()

    def residual_block(self, input, output, input_depth, output_depth, projection=False, trainable=True):
        (self.feed(input)
             .conv(1, 1, input_depth, 1, 1, name='{}_branch2a'.format(output), trainable=trainable, bn=True, relu=True)
             .conv(3, 3, input_depth, 1, 1, name='{}_branch2b'.format(output), trainable=trainable, bn=True, relu=True)
             .conv(1, 1, output_depth, 1, 1, name='{}_branch2c'.format(output), trainable=trainable, bn=True, relu=False))

        if projection:
            # Option B: Projection shortcut
            (self.feed(input)
                 .conv(1, 1, output_depth, 1, 1, name='{}_branch1'.format(output), trainable=trainable, bn=True, relu=False))
            (self.feed('{}_branch1'.format(output), '{}_branch2c'.format(output))
                 .eltwise_add(name=output, relu=True))
        else:
            # Option A: Zero-padding
            (self.feed(input, '{}_branch2c'.format(output))
                 .eltwise_add(name=output, relu=True))
        return self

    def setup(self):
        #
        (self.feed('data')
             .conv(7, 7, 64, 2, 2, name='conv1', trainable=False, bn=True, relu=True)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1'))

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

        #========= RPN ============
        (self.feed('res4f')
             .conv(3,3,512,1,1,name='rpn_conv/3x3',relu=True, bn=False)
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score', bn=False))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred',bn=False))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rois'))

        #========= RCNN ============
        (self.feed('res4f', 'rois')
             .roi_pool(14, 14, 1.0/16, name='roi_pool')
         .residual_block('roi_pool', 'res5a', 512, 2048, projection=True, trainable=True)
         .residual_block('res5a', 'res5b', 512, 2048, projection=False, trainable=True)
         .residual_block('res5b', 'res5c', 512, 2048, projection=False, trainable=True)
         .avg_pool(7, 7, 1, 1, padding='VALID', name='pool5')
         .fc(n_classes, relu=False, name='cls_score')
         .softmax(name='cls_prob')
         )
        (self.feed('pool5')
             .fc(n_classes*4, relu=False, name='bbox_pred'))