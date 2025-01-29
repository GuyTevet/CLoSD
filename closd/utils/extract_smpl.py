import glob
import os
import sys
import pdb
import os.path as osp
import argparse

sys.path.append(os.getcwd())

# import open3d as o3d
# import open3d.visualization.rendering as rendering
import imageio
from tqdm import tqdm
import joblib
import numpy as np
import torch

from closd.utils.smpllib.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)
import random

from closd.utils.smpllib.smpllib.smpl_mujoco import SMPL_BONE_ORDER_NAMES as joint_names
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from scipy.spatial.transform import Rotation as sRot
import matplotlib.pyplot as plt
from tqdm import tqdm
import math


def main(params):

    data_dir = "closd/data/smpl"
    smpl_parser_n = SMPL_Parser(model_path=data_dir, gender="neutral")
    smpl_parser_m = SMPL_Parser(model_path=data_dir, gender="male")
    smpl_parser_f = SMPL_Parser(model_path=data_dir, gender="female")

    pkl_data = joblib.load(params.record_path)
    mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
    mujoco_2_smpl = [mujoco_joint_names.index(q) for q in joint_names if q in mujoco_joint_names]

    items = list(pkl_data.items())

    npy_dict = {
        'motion': [],
        'text': [],
        'lengths': [],
        'num_samples': len(items),
        'num_repetitions': 1,
    }

    for entry_key, data_seq in tqdm(items):
        gender, beta = data_seq['betas'][0], data_seq['betas'][1:]
        smpl_parser = smpl_parser_n

        pose_quat, trans = data_seq['body_quat'].numpy()[::2], data_seq['trans'].numpy()[::2]
        skeleton_tree = SkeletonTree.from_dict(data_seq['skeleton_tree'])
        offset = skeleton_tree.local_translation[0]
        root_trans_offset = trans - offset.numpy()

        sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=True)

        global_rot = sk_state.global_rotation
        B, J, N = global_rot.shape
        pose_quat = (sRot.from_quat(global_rot.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5])).as_quat().reshape(B, -1, 4)
        B_down = pose_quat.shape[0]
        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat), torch.from_numpy(trans), is_local=False)
        local_rot = new_sk_state.local_rotation
        
        pose_aa = sRot.from_quat(local_rot.reshape(-1, 4).numpy()).as_matrix()[:, :2, :].reshape(B_down, -1, 6)  # 6d representation
        pose_aa = pose_aa[:, mujoco_2_smpl, :]
        roop_pose_pad = np.concatenate([root_trans_offset, np.zeros_like(root_trans_offset)], axis=-1)[:, None]
        motion_rep = np.concatenate([pose_aa, roop_pose_pad], axis=1)[None].transpose(0, 2, 3, 1)
        npy_dict['motion'].append(motion_rep)
        npy_dict['lengths'].append(int(motion_rep.shape[-1]))
        npy_dict['text'].append(entry_key)
    

    npy_dict['lengths'] = np.array(npy_dict['lengths'])
    _moption = np.zeros([len(npy_dict['lengths']), 25, 6, npy_dict['lengths'].max()], dtype=npy_dict['motion'][0].dtype)
    for i, _len in enumerate(npy_dict['lengths']):
        _moption[i, :, :, :_len] = npy_dict['motion'][i]
    npy_dict['motion'] = _moption

    out_path = params.record_path.replace('.pkl', '_smpl.npy')
    np.save(out_path, npy_dict)
    print(f'Saved [{os.path.abspath(out_path)}]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--record_path", type=str, required=True, help='Path to pkl state file')
    params = parser.parse_args()
    main(params)