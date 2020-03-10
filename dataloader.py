# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

"""Monodepth data loader.
"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf


def string_length_tf(t):
    return tf.py_func(len, [t], [tf.int64])


class MonoVODDataloader(object):

    def __init__(self, data_path, filenames_files, params, mode, n_threads):
        self.data_path = data_path
        self.params = params
        self.mode = mode

        dataset = tf.data.TextLineDataset(filenames_files).map(lambda line: self.parse_input(line),
                                                               n_threads).batch(params.batch_size).repeat(params.num_epochs).prefetch(1)
        self.iterator = dataset.make_initializable_iterator()
        self.init_op = self.iterator.initializer


    def get_initializer(self):
        return self.init_op


    def load_batch(self):
        left_1_batch, right_1_batch, left_2_batch = self.iterator.get_next()

        return left_1_batch, right_1_batch, left_2_batch


    def parse_input(self, line):
        split_line = tf.string_split([line]).values
        #split_line = tf.Print(split_line, [split_line], '++++++++++++++++++++++++')

        left_1_path = tf.string_join([self.data_path, split_line[0]])
        right_1_path = tf.string_join([self.data_path, split_line[1]])
        left_2_path = tf.string_join([self.data_path, split_line[2]])
        right_2_path = tf.string_join([self.data_path, split_line[3]])

        left_1_image_o = self.read_image(left_1_path)
        right_1_image_o = self.read_image(right_1_path)
        left_2_image_o = self.read_image(left_2_path)
        right_2_image_o = self.read_image(right_2_path)

        # randomly flip images
        do_flip = tf.random_uniform([], 0, 1)
        left_1_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_1_image_o), lambda: left_1_image_o)
        right_1_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_1_image_o), lambda: right_1_image_o)
        left_2_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(right_2_image_o), lambda: left_2_image_o)
        # right_2_image = tf.cond(do_flip > 0.5, lambda: tf.image.flip_left_right(left_2_image_o),lambda: right_2_image_o)

        # randomly augment images
        do_augment = tf.random_uniform([], 0, 1)
        left_1_image, right_1_image, left_2_image = tf.cond(do_augment > 0.5,
                                                            lambda: self.augment_images(left_1_image, right_1_image, left_2_image),
                                                            lambda: (left_1_image, right_1_image, left_2_image))

        left_1_image.set_shape([None, None, 3])
        right_1_image.set_shape([None, None, 3])
        left_2_image.set_shape([None, None, 3])

        return left_1_image, right_1_image, left_2_image


    def augment_images(self, left_1_image, right_1_image, left_2_image):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        left_1_image_aug = left_1_image ** random_gamma
        right_1_image_aug = right_1_image ** random_gamma
        left_2_image_aug = left_2_image ** random_gamma

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        left_1_image_aug = left_1_image_aug * random_brightness
        right_1_image_aug = right_1_image_aug * random_brightness
        left_2_image_aug = left_2_image_aug * random_brightness

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white = tf.ones([tf.shape(left_1_image)[0], tf.shape(left_1_image)[1]])
        color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=2)
        left_1_image_aug *= color_image
        right_1_image_aug *= color_image
        left_2_image_aug *= color_image

        # saturate
        left_1_image_aug = tf.clip_by_value(left_1_image_aug, 0, 1)
        right_1_image_aug = tf.clip_by_value(right_1_image_aug, 0, 1)
        left_2_image_aug = tf.clip_by_value(left_2_image_aug, 0, 1)

        return left_1_image_aug, right_1_image_aug, left_2_image_aug


    def read_image(self, image_path):
        path_length = tf.strings.length(image_path)
        file_extension = tf.substr(image_path, path_length - 3, 3)
        file_cond = tf.equal(file_extension, 'jpg')

        image = tf.cond(file_cond, lambda: tf.image.decode_jpeg(tf.read_file(image_path)),
                                   lambda: tf.image.decode_png(tf.read_file(image_path)))

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [self.params.height, self.params.width], tf.image.ResizeMethod.AREA)

        return image
