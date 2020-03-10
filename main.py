# Train: (Scratch)
# use_gpu 2,3 && python main.py --mode train --model_name kitti_model --data_path /scratch/shikhar/Datasets/KITTI_Odom/dataset/
# --filenames_file /scratch/shikhar/Projects/DeepVO/filenames_files/kitti_filenames.txt
# --log_directory /scratch/shikhar/Projects/DeepVO/train_log/ --batch_size 16 --num_epochs 2 --learning_rate 0.001 --num_gpus 2

# Train: Multi-GPU (Scratch)
# use_gpu 2,3 && python main.py --mode train --model_name kitti_model --data_path /scratch/shikhar/Datasets/KITTI_Odom/dataset/
# --filenames_file /scratch/shikhar/Projects/DeepVO/filenames_files/kitti_filenames.txt
# --log_directory /scratch/shikhar/Projects/DeepVO/train_log/ --batch_size 16 --num_epochs 2 --learning_rate 0.001 --num_gpus 2

# Train: (Fine-Tune)
# python ~/Projects/monodepth/monodepth_main.py --mode train --model_name kitti_rrc_mainlab_corridor_model
#--data_path /media/axe/HD_data/RRC_ZED_dataset/ --filenames_file /home/axe/Projects/monodepth/utils/filenames/rrc_mainlab_corridor.txt
# --log_directory /home/axe/Projects/monodepth/train_log/ --checkpoint_path ~/Projects/monodepth/models/model_kitti
# --batch_size 8 --retrain

# Train: (Fine-Tune || Neon)
# python monodepth_main.py --mode train --model_name kitti_rrc_main_ms_lab --data_path /scratch/shikhar/Datasets/zed_mslab/
# --filenames_file /scratch/shikhar/Projects/monodepth/utils/filenames/rrc_mslab.txt
# --log_directory /scratch/shikhar/Projects/monodepth/train_log --batch_size 16 --retrain
# --checkpoint_path /scratch/shikhar/Projects/monodepth/train_log/kitti_rrc_mainlab_corridor_model/model-90000

# Train: (Fine-Tune || Neon)
# python monodepth_main.py --mode train --model_name kitti_rrc_main_ms_lab --data_path /scratch/shikhar/Datasets/zed_mslab/
# --filenames_file /scratch/shikhar/Projects/monodepth/utils/filenames/rrc_mslab.txt
# --log_directory /scratch/shikhar/Projects/monodepth/train_log
# --checkpoint_path /scratch/shikhar/Projects/monodepth/train_log/kitti_rrc_mainlab_corridor_model/model-120000
# --batch_size 16 --retrain --cycle_over_input

# Test:
# python ~/Projects/monodepth/monodepth_main.py --mode test --data_path /media/axe/HD_data/RRC_ZED_dataset/
# --filenames_file /home/axe/Projects/monodepth/utils/filenames/rrc_corridor_eval.txt --log_directory ~/Projects/monodepth/train_log/
# --checkpoint_path ~/Projects/monodepth/train_log/kitti_rrc_model/model-40000 --output_directory /media/axe/HD_data/Monodepth_output/output_kitti/corridor

# # Test: (Neon)
# python /scratch/shikhar/Projects/monodepth_original/monodepth_main.py --mode test --data_path /scratch/shikhar/Datasets/ZED_RRC_Data_Sampled --filenames_file /scratch/shikhar/Projects/monodepth/data_utils/rrc_sampled_256_512.txt --checkpoint_path /scratch/shikhar/Projects/monodepth_original/train_log_sampled/rrc_scratch_256_512_model/model-3150 --output_directory /scratch/shikhar/Projects/monodepth_original/train_log_sampled/rrc_scratch_256_512_model/results


from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
# import tensorflow.python.debug as tf_debug

from monovod_model import *
from dataloader import *
from utils import average_gradients

parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')

parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='monodepth')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, vgg or resnet50', default='vgg')
parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
parser.add_argument('--input_height',              type=int,   help='input height', default=384)
parser.add_argument('--input_width',               type=int,   help='input width', default=768)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=8)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--alpha_image_loss',          type=float, help='weight between SSIM and L1 in the image loss', default=0.85)
parser.add_argument('--disp_gradient_loss_weight', type=float, help='disparity smoothness weigth', default=0.1)
parser.add_argument('--lr_loss_weight',            type=float, help='left-right consistency weight', default=1.0)
parser.add_argument('--pose_loss_weight',          type=float, help='pose-based reconstruction loss weight', default=1.0)
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a specific checkpoint to load', default='')
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--full_summary',                          help='if set, will keep more data for each summary. Warning: the file can become very large', action='store_true')
parser.add_argument('--use_deconv',                            help='if set, will use transposed convolutions', action='store_true')
parser.add_argument('--wrap_mode',                 type=str,   help='bilinear sampler wrap mode, edge or border', default='border')
parser.add_argument('--save_after',                type=int,   help='save model after every `n` steps', default=10000)
parser.add_argument('--no_lstm',                               help='flag to decide lstm or fc layers for pose regression', action='store_true')
# ZED: b = 0.120 (baseline = 120mm), f = 695

args = parser.parse_args()


def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def train(params, args):
    """Training loop."""

    with tf.Graph().as_default(), tf.device('/cpu:0'):

        global_step = tf.Variable(0, trainable=False)

        # Optimization
        num_training_samples = count_text_lines(args.filenames_file)

        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        initial_learning_rate = args.learning_rate

        boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        values = [initial_learning_rate, initial_learning_rate/ 2, initial_learning_rate/ 4]
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        opt_step = tf.train.AdagradOptimizer(learning_rate)

        print("total number of samples: {}".format(num_training_samples))
        print("total number of steps: {}".format(num_total_steps))

        # Dataloading
        dataloader = MonoVODDataloader(args.data_path, args.filenames_file, params, args.mode, args.num_threads)
        init_op = dataloader.get_initializer()

        left_1_batch, right_1_batch, left_2_batch = dataloader.load_batch()
        print(left_1_batch, '%%%%%%%%%%%%%%%%%%%%%%')

        def verify_dataloader():
            print('Verifying Dataloader...')
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(init_op)

                for _ in range(params.num_epochs):
                    l1, r1, l2 = sess.run([left_1_batch, right_1_batch, left_2_batch])

                    print(np.asarray(l1).shape, np.asarray(r1).shape, np.asarray(l2).shape)
        # verify_dataloader()

        # Training
        def training():
            # Split the batch across GPU's
            left_1_splits  = tf.split(left_1_batch,  args.num_gpus, 0)
            right_1_splits = tf.split(right_1_batch, args.num_gpus, 0)
            left_2_splits = tf.split(left_2_batch, args.num_gpus, 0)

            tower_grads  = []
            tower_losses = []
            reuse_variables = False
            use_lstm = not args.no_lstm

            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(args.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        print(left_1_splits[i], '^^^^^^^^^^^^^^^^^^^^^^^^^^^')
                        model = MonoVODModel(params, args.mode, left_1_splits[i], right_1_splits[i], left_2_splits[i],
                                                                                                     reuse_variables, i)
                        loss = model.total_loss
                        tower_losses.append(loss)

                        reuse_variables = True

                        grads = opt_step.compute_gradients(loss)

                        tower_grads.append(grads)

            grads = average_gradients(tower_grads)

            # BACKPROPAGATION
            apply_gradient_op = opt_step.apply_gradients(grads, global_step=global_step)

            total_loss = tf.reduce_mean(tower_losses)

            tf.summary.scalar('learning_rate', learning_rate, ['model_0'])
            tf.summary.scalar('total_loss', total_loss, ['model_0'])
            summary_op = tf.summary.merge_all('model_0')

            # SESSION
            config = tf.ConfigProto(allow_soft_placement=True)
            sess = tf.Session(config=config)

            # SAVER
            summary_writer = tf.summary.FileWriter(args.log_directory + '/' + args.model_name, sess.graph)
            train_saver = tf.train.Saver()

            # COUNT PARAMS
            total_num_parameters = 0
            for variable in tf.trainable_variables():
                total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            print("number of trainable parameters: {}".format(total_num_parameters))

            # INIT
            sess.run(tf.global_variables_initializer())
            sess.run(init_op)

            # LOAD CHECKPOINT IF SET
            if args.checkpoint_path != '':
                train_saver.restore(sess, args.checkpoint_path.split(".")[0])

                if args.retrain:
                    sess.run(global_step.assign(0))

            # DEBUGGER
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            save_after_n_steps = args.save_after

            # GO!
            print('Training...')
            start_step = global_step.eval(session=sess)
            start_time = time.time()
            for step in range(start_step, num_total_steps):
                before_op_time = time.time()
                _, loss_value = sess.run([apply_gradient_op, total_loss])

                duration = time.time() - before_op_time
                if step and step % 100 == 0:
                    examples_per_sec = params.batch_size / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps / step - 1.0) * time_sofar
                    print_string = 'batch {:>6} | examples/s: {:4.2f} | loss: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    print(print_string.format(step, examples_per_sec, loss_value, time_sofar, training_time_left))
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step=step)

                # Save
                if step and step % save_after_n_steps == 0:
                    train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=step)

            train_saver.save(sess, args.log_directory + '/' + args.model_name + '/model', global_step=num_total_steps)

        training()


# def test(params):
#     """Test function."""
#
#     dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
#     left  = dataloader.left_image_batch
#     right = dataloader.right_image_batch
#
#     model = MonodepthModel(params, args.mode, left, right)
#
#     # SESSION
#     config = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(config=config)
#
#     # SAVER
#     train_saver = tf.train.Saver()
#
#     # INIT
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     coordinator = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)
#
#     # RESTORE
#     if args.checkpoint_path == '':
#         restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
#     else:
#         restore_path = args.checkpoint_path.split(".")[0]
#     train_saver.restore(sess, restore_path)
#
#     num_test_samples = count_text_lines(args.filenames_file)
#
#     print('now testing {} files'.format(num_test_samples))
#     disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#
#     total_time = 0
#
#     for step in range(num_test_samples):
#         start_time = time.time()
#         disp = sess.run(model.disp_left_est[0])
#         end_time = time.time()
#
#         total_time += end_time - start_time
#
#         disparities[step] = disp[0].squeeze()
#         disparities_pp[step] = post_process_disparity(disp.squeeze())
#
#     print('done.')
#     print('Total Forward pass execution time: {} seconds'.format(total_time))
#     print('Per Frame Forward pass execution time: {} seconds'.format(total_time/num_test_samples))
#
#     print('writing disparities.')
#     if args.output_directory == '':
#         output_directory = os.path.dirname(args.checkpoint_path)
#     else:
#         output_directory = args.output_directory
#
#     np.save(output_directory + '/disparities.npy',    disparities)
#     np.save(output_directory + '/disparities_pp.npy', disparities_pp)
#
#     print('done.')

def main(_):

    params = monovod_parameters(
        encoder=args.encoder,
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        wrap_mode=args.wrap_mode,
        use_deconv=args.use_deconv,
        alpha_image_loss=args.alpha_image_loss,
        disp_gradient_loss_weight=args.disp_gradient_loss_weight,
        pose_loss_weight=args.pose_loss_weight,
        lr_loss_weight=args.lr_loss_weight,
        full_summary=args.full_summary,
        save_after=args.save_after
    )

    if args.mode == 'train':
        train(params, args)
    # elif args.mode == 'test':
    #     test(params)

if __name__ == '__main__':
    tf.app.run()