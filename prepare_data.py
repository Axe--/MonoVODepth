"""
Prepares KITTI data for DeepVO; outputs filenames file
Generates .txt file with path to images & pose values in the following format:
img_t.jpg img_t+1.jpg relative_pose_6_dof

# For sequence no. 00-10
Ground-truth pose: KITTI_Odom/dataset/poses/seq_no.txt file
Images: KITTI_Odom/dataset/sequences/seq_no/image_2/*.jpg

Sample Run:
python /scratch/shikhar/Projects/DeepVO/prepare_data.py --dataset_path /scratch/shikhar/Datasets/KITTI_Odom/dataset/
--out_file /scratch/shikhar/Projects/DeepVO/filenames_files/kitti_filenames.txt

python prepare_data.py --dataset_path /scratch/shikhar/Datasets/KITTI_Odom/dataset/
--out_file /scratch/shikhar/Projects/MonoVOD/filenames_files/kitti_seq_00_filenames.txt --num_seq 1
"""
import numpy as np
from glob import glob
import os
import argparse
from utils import mat2euler


parser = argparse.ArgumentParser(description='DeepVO: Prepare Data')

parser.add_argument('--dataset_path', type=str, help='path to ../dataset', required=True)
parser.add_argument('--out_file',     type=str, help='path to output filenames file', required=True)
parser.add_argument('--img_ext',      type=str, help='input image extension {png, jpg}', default='png')
parser.add_argument('--num_seqs',     type=int, help='number sequences to read from 00 to 10', default=11)
#parser.add_argument('--input_height', type=int, help='input height', default=256)

args = parser.parse_args()


def generate_filenames_file(seq_no, image_dir, pose_dir, file_out, img_ext='png'):
    """
    Given the sequence number, writes the image paths and poses to `out_file_path`.
    img_t.jpg img_t+1.jpg relative_pose_6_dof

    The pose vector written in csv format.
    e.g. im1.png im2.png 3.1, 2.3, 1.3, 4.5, 0.1, 1.3

    :param seq_no: sequence no. as per the KITTI Odmotery dataset
    :type seq_no: str
    :param image_dir: path to images directory
    :type image_dir: str
    :param pose_dir: path to poses directory
    :type pose_dir: str
    :param file_out: output file object
    :return: None
    """
    # Read image paths
    left_image_folder = 'image_2'
    right_image_folder = 'image_3'
    left_image_paths = sorted(glob(os.path.join(image_dir, seq_no, left_image_folder, '*.'+ img_ext)))
    right_image_paths = sorted(glob(os.path.join(image_dir, seq_no, right_image_folder, '*.' + img_ext)))

    # Read poses
    # poses_file = os.path.join(pose_dir, seq_no+'.txt')
    # poses = parse_poses_file(poses_file)

    # assert len(left_image_paths) == len(poses), 'No. of Images & Poses mismatch:{} != {}'.format(len(left_image_paths), len(poses))
    assert len(left_image_paths) == len(right_image_paths), \
        'No. of Left & Right images mismatch:{} != {}'.format(len(left_image_paths), len(right_image_paths))

    # Write consecutive image pairs and the relative pose
    total_samples = len(left_image_paths)

    for i in range(total_samples-1):
        # remove the dataset path; only save the relative path
        left_rel_path_1 = left_image_paths[i].replace(dataset_path, "")
        right_rel_path_1 = right_image_paths[i].replace(dataset_path, "")

        left_rel_path_2 = left_image_paths[i+1].replace(dataset_path, "")
        right_rel_path_2 = right_image_paths[i+1].replace(dataset_path, "")

        # Remove the '/' from relative path at the start
        left_rel_path_1 = left_rel_path_1[1:] if left_rel_path_1[0] == '/' else left_rel_path_1
        right_rel_path_1 = right_rel_path_1[1:] if right_rel_path_1[0] == '/' else right_rel_path_1

        left_rel_path_2 = left_rel_path_2[1:] if left_rel_path_2[0] == '/' else left_rel_path_2
        right_rel_path_2 = right_rel_path_2[1:] if right_rel_path_2[0] == '/' else right_rel_path_2

        # Relative pose b/w the two images
        # pose_relative = poses[i+1] - poses[i]
        # pose_relative_csv = list_to_csv_string(pose_relative)

        # Write to File
        file_out.write(left_rel_path_1 + ' ' + right_rel_path_1 + ' ' + left_rel_path_2 + ' ' + right_rel_path_2 + '\n')


def parse_poses_file(pose_file_path):
    """
    Given the poses file containning 3x4 transformation matrix,
    computes corresponding 6-DoF pose.

    :param pose_file_path: path to pose file
    :return: euler angles & translation (6 DoF) poses
    """
    with open(pose_file_path, 'r') as f:
        lines = f.readlines()

        poses = []
        for line in lines:
            transform_matrix = np.asarray([float(x) for x in line.strip().split()])

            pose_vec = compute_pose_vec(transform_matrix)

            poses.append(pose_vec)

    poses = np.asarray(poses)

    return poses


def compute_pose_vec(transform_mat):
    """
    Given the transformation matrix, computes the 6-DoF pose.

    :param transform_mat: 1x12 transformation vector [R | t]
    :return: 6-DoF pose vector
    """

    rotation = np.reshape(transform_mat[[0, 1, 2, 4, 5, 6, 8, 9, 10]], [3, 3])
    euler_angles = list(mat2euler(rotation))
    translation = transform_mat[[3, 7, 11]]

    # Concat Euler Angle & Translation vectors
    pose = euler_angles
    pose.extend(translation)

    pose = np.asarray(pose)

    return pose


def list_to_csv_string(lst):
    s = ''
    for x in lst:
        s += str(x) + ','

    return s[:-1]


# MAIN
dataset_path = args.dataset_path
out_filenames_file = args.out_file
image_ext = args.img_ext
n_seq = args.num_seqs

images_directory = os.path.join(dataset_path, 'sequences')
poses_directory = os.path.join(dataset_path, 'poses')

# Iterate over sequences
sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

print("Generating...")
with open(out_filenames_file, 'w') as f_out:
    for seq in sequences[:n_seq]:
        generate_filenames_file(seq, images_directory, poses_directory, f_out)

        print('Seq: {} Done'.format(seq))

print('Complete!')
