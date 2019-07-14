# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import, print_function, division
import tensorflow.contrib.slim as slim
import tensorflow as tf

# from utils.external import mobilenet_v2
# from utils.external.mobilenet import training_scope
# from utils.external.mobilenet_v2 import op
# from utils.external.mobilenet_v2  import ops
from utils.external import mobilenet_v1
from utils.external.mobilenet import training_scope



import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
import numpy as np
from collections import namedtuple

# from nets.network import Network
# from model.config import cfg
#

def separable_conv2d_same(inputs, kernel_size, stride, rate=1, scope=None):
    """Strided 2-D separable convolution with 'SAME' padding.
    Args:
      inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
      kernel_size: An int with the kernel_size of the filters.
      stride: An integer, the output stride.
      rate: An integer, rate for atrous convolution.
      scope: Scope.
    Returns:
      output: A 4-D tensor of size [batch, height_out, width_out, channels] with
        the convolution output.
    """

    # By passing filters=None
    # separable_conv2d produces only a depth-wise convolution layer
    if stride == 1:
        return slim.separable_conv2d(inputs, None, kernel_size,
                                     depth_multiplier=1, stride=1, rate=rate,
                                     padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs,
                        [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.separable_conv2d(inputs, None, kernel_size,
                                     depth_multiplier=1, stride=stride, rate=rate,
                                     padding='VALID', scope=scope)


Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(kernel=3, stride=1, depth=64),
    DepthSepConv(kernel=3, stride=2, depth=128),
    DepthSepConv(kernel=3, stride=1, depth=128),
    DepthSepConv(kernel=3, stride=2, depth=256),
    DepthSepConv(kernel=3, stride=1, depth=256),
    DepthSepConv(kernel=3, stride=2, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    DepthSepConv(kernel=3, stride=1, depth=512),
    # use stride 1 for the 13th layer
    # DepthSepConv(kernel=3, stride=1, depth=512),
    # DepthSepConv(kernel=3, stride=1, depth=512)

    DepthSepConv(kernel=3, stride=1, depth=1024),
    DepthSepConv(kernel=3, stride=1, depth=1024)
]


# Modified mobilenet_v1
def mobilenet_v1_base(inputs,
                      conv_defs,
                      starting_layer=0,
                      min_depth=8,
                      depth_multiplier=1.0,
                      output_stride=None,
                      reuse=None,
                      scope=None):
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse):
        # The current_stride variable keeps track of the output stride of the
        # activations, i.e., the running product of convolution strides up to the
        # current network layer. This allows us to invoke atrous convolution
        # whenever applying the next convolution would result in the activations
        # having output stride larger than the target output_stride.
        current_stride = 1

        # The atrous convolution rate parameter.
        rate = 1

        net = inputs
        for i, conv_def in enumerate(conv_defs):
            end_point_base = 'Conv2d_%d' % (i + starting_layer)

            if output_stride is not None and current_stride == output_stride:
                # If we have reached the target output_stride, then we need to employ
                # atrous convolution with stride=1 and multiply the atrous rate by the
                # current unit's stride for use in subsequent layers.
                layer_stride = 1
                layer_rate = rate
                rate *= conv_def.stride
            else:
                layer_stride = conv_def.stride
                layer_rate = 1
                current_stride *= conv_def.stride

            if isinstance(conv_def, Conv):
                end_point = end_point_base
                net = resnet_utils.conv2d_same(net, depth(conv_def.depth), conv_def.kernel,
                                               stride=conv_def.stride,
                                               scope=end_point)

            elif isinstance(conv_def, DepthSepConv):
                end_point = end_point_base + '_depthwise'

                net = separable_conv2d_same(net, conv_def.kernel,
                                            stride=layer_stride,
                                            rate=layer_rate,
                                            scope=end_point)

                end_point = end_point_base + '_pointwise'

                net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                  stride=1,
                                  scope=end_point)

            else:
                raise ValueError('Unknown convolution type %s for layer %d'
                                 % (conv_def.ltype, i))
        print('depth(conv_def.depth)======', depth(conv_def.depth))
        return net

# # MobileNet options
# __C.MOBILENET = edict()
# __C.MOBILENET.REGU_DEPTH =
# __C.MOBILENET.FIXED_LAYERS = 5
# __C.MOBILENET.WEIGHT_DECAY = 0.00004
# __C.MOBILENET.DEPTH_MULTIPLIER = 1.0
# Modified arg_scope to incorporate configs
def mobilenet_v1_arg_scope(is_training=True,
                           stddev=0.09):
    batch_norm_params = {
        'is_training': False,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
        'trainable': False,
    }

    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    print("__C.MOBILENET.WEIGHT_DECAY = 0.00004 ====== bug==")
    regularizer = tf.contrib.layers.l2_regularizer(0.00004)
    if False:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        trainable=is_training,
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6,
                        normalizer_fn=slim.batch_norm,
                        padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc


#mobilenetv1 to others to call
def mobilenetv1_base(img_batch, is_training=True):
    net_conv = img_batch
    if 5 > 0:
        with slim.arg_scope(mobilenet_v1_arg_scope(is_training=False)):
            net_conv = mobilenet_v1_base(net_conv,
                                         _CONV_DEFS[:5],
                                         starting_layer=0,
                                         depth_multiplier=1.0,
                                         reuse=None,
                                         scope="moblienetv1_base")
    if 5 < 12:
        with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
            net_conv = mobilenet_v1_base(net_conv,
                                         _CONV_DEFS[5:12],
                                         starting_layer=5,
                                         depth_multiplier=1.0,
                                         reuse=None,
                                         scope="moblienetv1_base")

    return net_conv


def mobilenetv1_head(inputs, is_training=True):
    with slim.arg_scope(mobilenet_v1_arg_scope(is_training=is_training)):
        fc7 = mobilenet_v1_base(inputs,
                                _CONV_DEFS[12:],
                                starting_layer=12,
                                depth_multiplier=1.0,
                                reuse=None,
                                scope="moblienetv1_base")
        # average pooling done by reduce_mean
        fc7 = tf.reduce_mean(fc7, axis=[1, 2])
    return fc7
