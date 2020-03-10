"""
Convolution-LSTM Model (DeepVO)

Defines the architecture, outputs and loss
"""
import tensorflow as tf
from collections import namedtuple

deepvo_parameters = namedtuple('parameters', 'height, width, batch_size, num_epochs, full_summary, save_after')

class DeepVOModel():
    def __init__(self, mode, img_1, img_2, pose_gt, reuse_variables, rot_loss_wt=100, model_index=0, use_lstm=True, params=None):
        # Train/Test mode
        self.mode = mode
        self.params = params

        # To use LSTM or FC layers for Pose prediction
        self.use_lstm = use_lstm

        # Collection for model summary
        self.model_collection = ['model_' + str(model_index)]

        # Reuse variables (for Multi-GPU training)
        self.reuse_variables = reuse_variables

        # We need the batch size for tiling the pose prediction Weights
        self.seq_len = tf.shape(img_1)[0]

        # Ground-Truth Pose [0;t] for MSE Loss
        # Add a dummy axis for batch_size
        self.pose_gt = tf.expand_dims(pose_gt, axis=1)    # [seq_len, batch_size=1, 6]

        self.rot_loss_wt = rot_loss_wt

        # Stack images channel-wise
        self.img_stacked = tf.concat([img_1, img_2], axis=3)
        self.img_stacked.set_shape([None, None, None, 6])

        # NOT RELEVANT FOR DeepVO; only for later use with MonoVOD
        #self.is_reuse_cnn = False

        # Build CNN
        self.build_cnn(self.img_stacked)

        # Build LSTM
        if use_lstm:
            self.build_lstm()

        # If Test mode; done
        if self.mode == 'test':
            return

        # Build Loss
        self.build_loss()

        # Log to Tensorboard
        self.build_summaries()


    def build_cnn(self, input, is_reuse_cnn=False):
        """
        Define the CNN
        :param input: input image (B,H,W,C)
        """
        # In: [256, 512, 6] --> Out: [4, 8, 1024]
        with tf.variable_scope('convnet', reuse=self.reuse_variables):
            self.conv1 = tf.layers.conv2d(input, 64, 7, 2, 'SAME', activation=tf.nn.relu)           # 1/2
            self.conv2 = tf.layers.conv2d(self.conv1, 128, 5, 2, 'SAME', activation=tf.nn.relu)     # 1/4
            self.conv3_1 = tf.layers.conv2d(self.conv2, 256, 5, 2, 'SAME', activation=tf.nn.relu)   # 1/8
            self.conv3_2 = tf.layers.conv2d(self.conv3_1, 256, 3, 1, 'SAME', activation=tf.nn.relu)
            self.conv4_1 = tf.layers.conv2d(self.conv3_2, 512, 3, 2, 'SAME', activation=tf.nn.relu) # 1/16
            self.conv4_2 = tf.layers.conv2d(self.conv4_1, 512, 3, 1, 'SAME', activation=tf.nn.relu)
            self.conv5_1 = tf.layers.conv2d(self.conv4_2, 512, 3, 2, 'SAME', activation=tf.nn.relu) # 1/32
            self.conv5_2 = tf.layers.conv2d(self.conv5_1, 512, 3, 1, 'SAME', activation=tf.nn.relu)
            self.conv6 = tf.layers.conv2d(self.conv5_2, 1024, 3, 2, 'SAME', activation=None)        # 1/64

            # Layer input to LSTM
            self.encoded = self.conv6

            if not self.use_lstm:
                self.encoded_flat = tf.layers.flatten(self.encoded)
                self.fc_1 = tf.layers.dense(self.encoded_flat, 1000, tf.nn.relu)
                self.fc_2 = tf.layers.dense(self.fc_1, 1000, tf.nn.relu)
                self.pose_pred = tf.layers.dense(self.fc_2, 6)

                # Add a dummy axis (to comply with LSTM's output format)
                self.pose_pred = tf.expand_dims(self.pose_pred, axis=1)   # Pose: [batch_size, seq_len=1, n_output=6]
                print('Using only CNN!')


    def build_lstm(self):
        with tf.variable_scope('lstm', reuse=self.reuse_variables):
            # Flatten the encoded volume
            self.encoded_flat = tf.layers.flatten(self.encoded)

            # Unstack for LSTM input such that batch_size represents the sequence length
            # => encoded: [batch_size, H*W*C] -> X_: [H*W*C] * batch_size
            # X_lstm = tf.unstack(self.encoded_flat, axis=0)

            # Add a dimension to signify batch size = 1 **************
            X_lstm = tf.expand_dims(self.encoded_flat, axis=0)                              # LSTM Input : [batch_size=1, seq_len, inp_dim]

            # Define two stacked LSTM cells
            n_hidden = 1000
            lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(n_hidden)
            lstm_cell_2 = tf.nn.rnn_cell.LSTMCell(n_hidden)
            lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])

            # LSTM cell output
            lstm_outputs, states = tf.nn.dynamic_rnn(lstm_cells, X_lstm, dtype=tf.float32)  # LSTM Output : [batch_size=1, seq_len, n_hidden]
            lstm_outputs = tf.transpose(lstm_outputs, [1, 0, 2])                            # LSTM Output : [seq_len, batch_size=1, n_hidden]

            # Pose Prediction params (6-DoF)
            n_output = 6
            W = tf.get_variable('W', shape=[n_hidden, n_output], initializer=tf.keras.initializers.he_normal())
            # b = tf.Variable(tf.constant(1, shape=[n_output]))
            b = tf.get_variable('b', shape=[n_output], initializer=tf.constant_initializer(value=1))

            # Tile the weights for pose prediction
            W = tf.tile(tf.expand_dims(W, axis=0), [self.seq_len, 1 , 1])   # W : [seq_len, n_hidden, n_output]

            self.pose_pred = tf.matmul(lstm_outputs, W) + b                 # pose : [seq_len, batch_size=1, n_output=6]


    def build_loss(self):
        with tf.variable_scope('loss', reuse=self.reuse_variables):
            # Translation Loss
            trans_pred = self.pose_pred[:,:,:3]
            trans_gt = self.pose_gt[:,:,:3]
            if self.params.full_summary:
                self.z_pred = tf.reduce_mean(trans_pred[:,:,0])
                self.y_pred = tf.reduce_mean(trans_pred[:,:,1])
                self.x_pred = tf.reduce_mean(trans_pred[:,:,2])

                self.z_gt = tf.reduce_mean(trans_gt[:, :, 0])
                self.y_gt = tf.reduce_mean(trans_gt[:, :, 1])
                self.x_gt = tf.reduce_mean(trans_gt[:, :, 2])

            self.translation_loss = tf.losses.mean_squared_error(trans_gt, trans_pred, reduction=tf.losses.Reduction.MEAN)

            # Rotation Loss
            rot_pred = self.pose_pred[:,:,3:]
            rot_gt = self.pose_gt[:,:,3:]

            self.rotation_loss = tf.losses.mean_squared_error(rot_gt, rot_pred, reduction=tf.losses.Reduction.MEAN)

            # Total Loss
            self.total_loss = self.translation_loss + self.rot_loss_wt * self.rotation_loss


    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            tf.summary.scalar('Rotation_loss', self.rotation_loss, collections=self.model_collection)
            tf.summary.scalar('Translation_loss', self.translation_loss, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.scalar('z_pred', self.z_pred, collections=self.model_collection)
                tf.summary.scalar('y_pred', self.y_pred, collections=self.model_collection)
                tf.summary.scalar('x_pred', self.x_pred, collections=self.model_collection)

                tf.summary.scalar('z_gt', self.z_gt, collections=self.model_collection)
                tf.summary.scalar('y_gt', self.y_gt, collections=self.model_collection)
                tf.summary.scalar('x_gt', self.x_gt, collections=self.model_collection)

            # tf.summary.scalar('Total_loss', self.total_loss)
            # tf.summary.image('left', self.left)


    # Forward pass for MonoVOD
    # def forward():
    #   enc1,depth1 = cnn_forward(im_1, reuse=False)
    #   enc2,depth2 = cnn_forward(im_2, reuse=True)
    #   rel_pose = compute_pose(enc1, enc2)
    #   Proj_inverse_wrap()


# Testing resusability of CNN
# if __name__ == "__main__":
#     im1 = tf.random_uniform([1, 4, 4, 3])
#
#     model = DeepVOModel(im1, im1, None)
#
#     op_1 = model.build_cnn(im1, is_reuse_cnn=False)
#     op_2 = model.build_cnn(im1, is_reuse_cnn=True)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         o1, o2 = sess.run([op_1, op_2])
#
#         print(o1)
#         print('\n\n')
#         print(o2)