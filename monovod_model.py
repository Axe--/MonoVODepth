# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from bilinear_sampler import *
from utils import projective_inverse_warp

monovod_parameters = namedtuple('parameters',
                                  'encoder, '
                                  'height, width, '
                                  'batch_size, '
                                  'num_threads, '
                                  'num_epochs, '
                                  'wrap_mode, '
                                  'use_deconv, '
                                  'alpha_image_loss, '
                                  'disp_gradient_loss_weight, '
                                  'lr_loss_weight, '
                                  'pose_loss_weight, '
                                  'full_summary, '
                                  'save_after')
# MIN_DISP = 0.01
MAX_DEPTH = 100
MIN_DEPTH = 0.1

# KITTI: K_02
K = [9.597910e+02, 0.000000e+00, 6.960217e+02,
     0.000000e+00, 9.569251e+02, 2.241806e+02,
     0.000000e+00, 0.000000e+00, 1.000000e+00]

class MonoVODModel(object):
    def __init__(self, params, mode, left_1, right_1, left_2, reuse_variables=False, model_index=0):
        self.params = params
        self.mode = mode
        self.left_1 = left_1
        self.right_1 = right_1
        self.left_2 = left_2
        self.model_collection = ['model_' + str(model_index)]
        self.intrinsics = tf.constant(K, shape=[1,3,3])
        self.intrinsics = tf.tile(self.intrinsics, [self.params.batch_size, 1, 1])

        print(self.left_1, '&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')

        self.reuse_variables = reuse_variables

        # Build the Encoder-Decoder & Estimate Disparity for 1st Left image
        self.encoded_1 = self.build_conv_model(left_1, right_1, reuse_variables)
        self.build_conv_outputs()

        # For Inference, only 1st Left image is required
        if self.mode == 'test':
            return

        # Image pyramid for 2nd Left image
        self.left_2_pyramid = self.scale_pyramid(left_2, 4)

        # Build the Encoder-Decoder for 2nd Left image (shared weights)
        self.encoded_2 = self.build_conv_model(left_1, reuse_variables=True)

        # For Train, add LSTM & Losses
        self.build_convlstm(self.encoded_1, self.encoded_2)
        self.build_losses()
        self.build_summaries()

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
        return smoothness_x + smoothness_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x, num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x, num_layers, 1, 1)
        conv2 = self.conv(conv1, num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:, 3:-1, 3:-1, :]

    def upsample_nn(self, x, scale):
        s = tf.shape(x); h = s[1]; w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * scale, w * scale])

    def upsample(self, x, n_filters, kernel_size, scale, name, bias_init=None, kernel_init=None):
        """
        Applies upsampling on input feature map using Nearest Neighbor interpolation,
        followed by convolutional layer.
        """
        upsampled = self.upsample_nn(x, scale)
        conv = tf.layers.conv2d(upsampled, n_filters, kernel_size, strides=1, padding='SAME', activation=tf.nn.elu,
                                name=name, bias_initializer=bias_init, kernel_initializer=kernel_init)
        return conv

    def build_vgg(self, input, encoder_only=False, reuse_variables=False):
        # set convenience functions
        conv = self.conv
        upconv = self.deconv if self.params.use_deconv else self.upconv

        # In: [384, 768, 3] --> Out: [3, 6, 1024]
        with tf.variable_scope('encoder', reuse=reuse_variables):
            conv1 = self.conv_block(input, 32, 7)   # H/2
            conv2 = self.conv_block(conv1, 64, 5)   # H/4
            conv3 = self.conv_block(conv2, 128, 3)  # H/8
            conv4 = self.conv_block(conv3, 256, 3)  # H/16
            conv5 = self.conv_block(conv4, 512, 3)  # H/32
            conv6 = self.conv_block(conv5, 512, 3)  # H/64
            conv7 = self.conv_block(conv6, 512, 3)  # H/128
            # Encoded
            encoded_vol = conv7

        if encoder_only:
            return encoded_vol

        with tf.variable_scope('skips', reuse=reuse_variables):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('decoder', reuse=reuse_variables):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)  # + MIN_DISP
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)  # + MIN_DISP
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)  # + MIN_DISP
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)  # + MIN_DISP

            print(self.disp3, '///////// disp3 ///////////')
            print(self.disp2, '///////// disp2 ///////////')
            print(iconv2, '///////// iconv2 ///////////')
            print(self.disp1, '///////// disp1 ///////////')
            print(iconv1, '///////// iconv1 ///////////')
            print(upconv1, '///////// upconv1 ///////////')

        return encoded_vol

    def build_vgg_layers(self, input):
        """
        VGG using tf.layers API
        """
        with tf.variable_scope('encoder'):
            self.conv1_1 = tf.layers.conv2d(input, 32, 7, 1, 'SAME', activation=tf.nn.elu, name='conv1_1')
            self.conv1_2 = tf.layers.conv2d(self.conv1_1, 32, 7, 2, 'SAME', activation=tf.nn.elu, name='conv1_2')  # 1/2

            self.conv2_1 = tf.layers.conv2d(self.conv1_2, 64, 5, 1, 'SAME', activation=tf.nn.elu, name='conv2_1')
            self.conv2_2 = tf.layers.conv2d(self.conv2_1, 64, 5, 2, 'SAME', activation=tf.nn.elu, name='conv2_2')  # 1/4

            self.conv3_1 = tf.layers.conv2d(self.conv2_2, 128, 3, 1, 'SAME', activation=tf.nn.elu, name='conv3_1')
            self.conv3_2 = tf.layers.conv2d(self.conv3_1, 128, 3, 2, 'SAME', activation=tf.nn.elu, name='conv3_2') # 1/8

            self.conv4_1 = tf.layers.conv2d(self.conv3_2, 256, 3, 1, 'SAME', activation=tf.nn.elu, name='conv4_1')
            self.conv4_2 = tf.layers.conv2d(self.conv4_1, 256, 3, 2, 'SAME', activation=tf.nn.elu, name='conv4_2') # 1/16

            self.conv5_1 = tf.layers.conv2d(self.conv4_2, 512, 3, 1, 'SAME', activation=tf.nn.elu, name='conv5_1')
            self.conv5_2 = tf.layers.conv2d(self.conv5_1, 512, 3, 2, 'SAME', activation=tf.nn.elu, name='conv5_2') # 1/32

            self.conv6_1 = tf.layers.conv2d(self.conv5_2, 512, 3, 1, 'SAME', activation=tf.nn.elu, name='conv6_1')
            self.conv6_2 = tf.layers.conv2d(self.conv6_1, 512, 3, 2, 'SAME', activation=tf.nn.elu, name='conv6_2') # 1/64

            self.conv7_1 = tf.layers.conv2d(self.conv6_2, 512, 3, 1, 'SAME', activation=tf.nn.elu, name='conv7_1')
            self.conv7_2 = tf.layers.conv2d(self.conv7_1, 512, 3, 2, 'SAME', activation=tf.nn.elu, name='conv7_2') # 1/128

        with tf.variable_scope('skips'):
            self.skip1 = self.conv1_2
            self.skip2 = self.conv2_2
            self.skip3 = self.conv3_2
            self.skip4 = self.conv4_2
            self.skip5 = self.conv5_2
            self.skip6 = self.conv6_2

        with tf.variable_scope('decoder'):
            self.upconv7 = self.upsample(self.conv7_2, n_filters=512, kernel_size=3, scale=2, name='upconv7')  # 1/64
            self.concat7 = tf.concat([self.upconv7, self.skip6], axis=3)
            self.iconv7 = tf.layers.conv2d(self.concat7, 512, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv7')

            self.upconv6 = self.upsample(self.iconv7, n_filters=512, kernel_size=3, scale=2, name='upconv6')  # 1/32
            self.concat6 = tf.concat([self.upconv6, self.skip5], axis=3)
            self.iconv6 = tf.layers.conv2d(self.concat6, 512, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv6')

            self.upconv5 = self.upsample(self.iconv6, n_filters=256, kernel_size=3, scale=2, name='upconv5')  # 1/16
            self.concat5 = tf.concat([self.upconv5, self.skip4], axis=3)
            self.iconv5 = tf.layers.conv2d(self.concat5, 256, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv5')

            self.upconv4 = self.upsample(self.iconv5, n_filters=128, kernel_size=3, scale=2, name='upconv4')  # 1/8
            self.concat4 = tf.concat([self.upconv4, self.skip3], axis=3)
            self.iconv4 = tf.layers.conv2d(self.concat4, 128, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv4')

            self.disp4 = 0.3 * tf.layers.conv2d(self.iconv4, 2, 3, 1, 'SAME', activation=tf.nn.sigmoid, name='disp4')
            up_disp4 = self.upsample_nn(self.disp4, scale=2)

            self.upconv3 = self.upsample(self.iconv4, n_filters=64, kernel_size=3, scale=2, name='upconv3')  # 1/4
            self.concat3 = tf.concat([self.upconv3, self.skip2, up_disp4], axis=3)
            self.iconv3 = tf.layers.conv2d(self.concat3, 64, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv3')

            self.disp3 = 0.3 * tf.layers.conv2d(self.iconv3, 2, 3, 1, 'SAME', activation=tf.nn.sigmoid, name='disp3')
            up_disp3 = self.upsample_nn(self.disp3, scale=2)

            self.upconv2 = self.upsample(self.iconv3, n_filters=32, kernel_size=3, scale=2, name='upconv2')  # 1/2
            self.concat2 = tf.concat([self.upconv2, self.skip1, up_disp3], axis=3)
            self.iconv2 = tf.layers.conv2d(self.concat3, 32, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv2')

            self.disp2 = 0.3 * tf.layers.conv2d(self.iconv2, 2, 3, 1, 'SAME', activation=tf.nn.sigmoid, name='disp2')
            up_disp2 = self.upsample_nn(self.disp2, scale=2)

            self.upconv1 = self.upsample(self.iconv2, n_filters=16, kernel_size=3, scale=2, name='upconv1')  # 1
            self.concat1 = tf.concat([self.upconv1, up_disp2], axis=3)
            self.iconv1 = tf.layers.conv2d(self.concat3, 16, 3, 1, 'SAME', activation=tf.nn.elu, name='iconv1')

            self.disp1 = 0.3 * tf.layers.conv2d(self.iconv1, 2, 3, 1, 'SAME', activation=tf.nn.sigmoid, name='disp1')

            print(self.disp1, '||||||||  LAYERS: disp1 |||||||')

    def build_resnet50(self):
        # set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2)  # H/2  -   64D
            pool1 = self.maxpool(conv1, 3)  # H/4  -   64D
            conv2 = self.resblock(pool1, 64, 3)  # H/8  -  256D
            conv3 = self.resblock(conv2, 128, 4)  # H/16 -  512D
            conv4 = self.resblock(conv3, 256, 6)  # H/32 - 1024D
            conv5 = self.resblock(conv4, 512, 3)  # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4

        # DECODING
        with tf.variable_scope('decoder'):
            upconv6 = upconv(conv5, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = self.get_disp(iconv4)
            udisp4 = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = self.get_disp(iconv3)
            udisp3 = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = self.get_disp(iconv2)
            udisp2 = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = self.get_disp(iconv1)

    def build_conv_model(self, left, right=None, reuse_variables=False):
        """
        Builds the CNN Encoder-Decoder model

        :return: Encoded feature volume from the conv encoder
        """
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('conv_model', reuse=reuse_variables):
                model_input = left

                # If only left image is provided, exclude the decoder
                encoder_only = True if right is None else False

                self.left_1_pyramid = self.scale_pyramid(left, 4)
                if self.mode == 'train' and right is not None:
                    self.right_1_pyramid = self.scale_pyramid(right, 4)

                # build model
                if self.params.encoder == 'vgg':
                    self.build_vgg_layers(model_input)
                    return self.build_vgg(model_input, encoder_only, reuse_variables)
                # elif self.params.encoder == 'resnet50':
                #     self.build_resnet50()
                else:
                    return None

    def build_convlstm(self, encoded_1, encoded_2):
        with tf.variable_scope('lstm', reuse=self.reuse_variables):
            # Concatenate (stack) the encoded representations
            self.encoded_stack = tf.concat([encoded_1, encoded_2], axis=3)   # [batch_size, h, w, 2*c]

            # Add dim, indicating batch size = 1 (time-major : False) & We treat batch_size --> seq_len
            X_convlstm = tf.expand_dims(self.encoded_stack, axis=0)  # LSTM Input : [1, seq_len, h, w, 2*c]

            # ConvLSTM cells
            X_shape = X_convlstm.get_shape().as_list()[2:]
            convlstm_cell_1 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=X_shape,output_channels=256, kernel_shape=[3,3])
            convlstm_cell_2 = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=X_shape, output_channels=256, kernel_shape=[3, 3])

            lstm_cells = tf.contrib.rnn.MultiRNNCell([convlstm_cell_1, convlstm_cell_2])

            # ConvLSTM output
            lstm_outputs, states = tf.nn.dynamic_rnn(lstm_cells, X_convlstm, dtype=tf.float32)  # LSTM Output : [1, seq_len, h, w, n_filters]
            # Remove the dummy batch_size dim
            lstm_outputs = tf.squeeze(lstm_outputs, axis=0)                                     # LSTM Output : [seq_len, h, w, n_filters]

            self.pose_pred = tf.layers.conv2d(lstm_outputs, 6, 3, 1, 'SAME', activation=None)   # Pose: [seq_len, h, w, 6]

            # Average the predicted poses across the h,w for each input in the sequence
            self.pose_pred = tf.reduce_mean(self.pose_pred, axis=[1,2])                         # Pose: [seq_len, 6]


    def build_conv_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est = [tf.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

        # COMPUTE DEPTH
        min_disp = 1 / MAX_DEPTH
        max_disp = 1 / MIN_DEPTH

        def compute_depth(disp):
            scaled_disp = min_disp + (max_disp - min_disp) * disp
            depth = 1 / scaled_disp
            return depth

        # COMPUTE DEPTH
        with tf.variable_scope('depths'):
            self.depth_left_est = [compute_depth(disp_left) for disp_left in self.disp_left_est]
            print('********************DEPTH LEFT EST', self.depth_left_est)

        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est = [self.generate_image_left(self.right_1_pyramid[i], self.disp_left_est[i]) for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_1_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i]) for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness = self.get_disparity_smoothness(self.disp_left_est, self.left_1_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_1_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs(self.left_est[i] - self.left_1_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_1_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM(self.left_est[i], self.left_1_pyramid[i]) for i in range(4)]
            self.ssim_loss_left = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_1_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] +
                                (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left = [self.params.alpha_image_loss * self.ssim_loss_left[i] +
                                (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i] for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # POSE WARPING LOSS
            self.pose_loss = [tf.abs(self.warp_image(i) - self.left_1_pyramid[i]) for i in range(4)]

            # TOTAL LOSS
            self.total_loss = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss \
                              + self.params.lr_loss_weight * self.lr_loss + self.params.pose_loss_weight * self.pose_loss

    def warp_image(self, scale_idx):
        depth = tf.squeeze(self.depth_left_est[scale_idx], axis=3)
        warped_img = projective_inverse_warp(self.left_1_pyramid[scale_idx], depth, self.pose_pred,
                                             self.intrinsics, self.params.batch_size)
        return warped_img

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i),
                                  self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i],
                                  collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i],
                                  collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4,
                                 collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4,
                                 collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_left_' + str(i), self.ssim_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_left_' + str(i), self.l1_left[i], max_outputs=4,
                                     collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4,
                                     collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left', self.left_1, max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right_1, max_outputs=4, collections=self.model_collection)
