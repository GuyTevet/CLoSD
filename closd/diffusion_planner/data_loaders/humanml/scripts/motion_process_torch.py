# Source: https://github.com/EricGuo5513/text-to-motion/blob/main/scripts/motion_process.py
import numpy as np 
import torch
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import *
from closd.diffusion_planner.data_loaders.humanml.common.skeleton import Skeleton
from closd.diffusion_planner.data_loaders.humanml.utils.paramUtil import t2m_raw_offsets, t2m_kinematic_chain

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# Default values are from: https://github.com/EricGuo5513/HumanML3D/blob/main/motion_representation.ipynb
def extract_features_t2m(positions, feet_thre=0.002, 
                     n_raw_offsets=t2m_raw_offsets, kinematic_chain=t2m_kinematic_chain, 
                     face_joint_indx=[2, 1, 17, 16], fid_r=[8, 11], fid_l=[7, 10], fix_ik_bug=False):
    # return extract_features(positions, feet_thre, torch.from_numpy(n_raw_offsets), kinematic_chain, face_joint_indx, fid_r, fid_l)
    return extract_features_torch(positions, feet_thre, torch.from_numpy(n_raw_offsets), kinematic_chain, face_joint_indx, fid_r, fid_l, fix_ik_bug=fix_ik_bug)

def extract_features(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l):
    # global_positions = positions.copy()
    global_positions = positions.clone()
    """ Get Foot Contacts """
    #
    feet_l, feet_r = foot_detect(positions, feet_thre, fid_l, fid_r)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx)
    positions = get_rifke(positions, r_rot)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data

def extract_features_torch(positions, feet_thre, n_raw_offsets, kinematic_chain, face_joint_indx, fid_r, fid_l, fix_ik_bug=False):
    # global_positions = positions.copy()
    bs, n_frames, n_joints, n_dim = positions.shape
    global_positions = positions.clone()
    positions = positions.clone()  # avoid in-place operations
    """ Get Foot Contacts """
    #
    feet_l, feet_r = foot_detect_torch(positions, feet_thre, fid_l, fid_r)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params_torch(positions, n_raw_offsets, kinematic_chain, face_joint_indx, fix_ik_bug=fix_ik_bug)
    # assert (cont_6d_params[:,:,0] == quaternion_to_cont6d(r_rot)).all()

    # For translating back to world coordinates - we take frame -2 because last frame is omitted
    recon_data = {
        'r_rot': r_rot[:, -2].clone(),  # [bs, 4]
        'r_pos': positions[:, -2, 0].clone(),  # [bs, 3]
    }

    positions = get_rifke_torch(positions, r_rot)  # local pos relative to root at (0,y,0); rotate positions such that root faces Z+

    '''Root height'''
    root_y = positions[:, :, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = torch.arcsin(r_velocity[:, :, 2:3])
    l_velocity = velocity[:, :, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = torch.cat([r_velocity, l_velocity, root_y[:, :-1]], axis=-1)  # root_y is cur joint, velocity is between current joint to the next one

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, :, 1:].reshape(bs, n_frames, -1)  # omit the root joint

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, :, 1:].reshape(bs, n_frames, -1)  # omit the root joint

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot(torch.repeat_interleave(r_rot[:, :-1, None], global_positions.shape[2], axis=2),
                        global_positions[:, 1:] - global_positions[:, :-1])  # SR: why not simply use positions[:, 1:] - positions[:, :-1]?
    local_vel = local_vel.reshape(bs, n_frames-1, -1)

    data = root_data
    data = torch.cat([data, ric_data[:, :-1]], axis=-1)  # omit position at last frame. so each frame has its own position and velocity to next frame
    data = torch.cat([data, rot_data[:, :-1]], axis=-1)  # omit rotation at last frame
    #     print(data.shape, local_vel.shape)
    data = torch.cat([data, local_vel], axis=-1)
    data = torch.cat([data, feet_l, feet_r], axis=-1)

    # DEBUG start
    # r_rot_reonstructed, r_pos_reconstructed = recover_root_rot_pos(data)
    # glob_r_rot_recon = [qmul(r_rot[:, 0], r_rot_reonstructed[:,frame]) for frame in range(r_rot_reonstructed.shape[1])]
    # glob_r_rot_recon = torch.stack(glob_r_rot_recon, dim=1)
    # assert (glob_r_rot_recon - r_rot[:, :-1]).abs().max() < 0.5
    # DEBUG end

    return data, recon_data

def foot_detect(positions, thres, fid_l, fid_r):
    velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

    feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
    feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
    feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(float)

    feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
    feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
    feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(float)
    return feet_l, feet_r

def foot_detect_torch(positions, thres, fid_l, fid_r):
    velfactor, heightfactor = torch.tensor([thres, thres]).to(positions.device), torch.tensor([3.0, 2.0]).to(positions.device)

    feet_l_x = (positions[:, 1:, fid_l, 0] - positions[:, :-1, fid_l, 0]) ** 2
    feet_l_y = (positions[:, 1:, fid_l, 1] - positions[:, :-1, fid_l, 1]) ** 2
    feet_l_z = (positions[:, 1:, fid_l, 2] - positions[:, :-1, fid_l, 2]) ** 2
    #     feet_l_h = positions[:-1,fid_l,1]
    #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)
    feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

    feet_r_x = (positions[:, 1:, fid_r, 0] - positions[:, :-1, fid_r, 0]) ** 2
    feet_r_y = (positions[:, 1:, fid_r, 1] - positions[:, :-1, fid_r, 1]) ** 2
    feet_r_z = (positions[:, 1:, fid_r, 2] - positions[:, :-1, fid_r, 2]) ** 2
    #     feet_r_h = positions[:-1,fid_r,1]
    #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).float()
    return feet_l, feet_r

def get_rifke(positions, r_rot):
    '''Local pose'''
    positions[..., 0] -= positions[:, 0:1, 0]
    positions[..., 2] -= positions[:, 0:1, 2]
    '''All pose face Z+'''
    positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
    return positions

def get_rifke_torch(positions, r_rot):
    '''Local pose'''
    positions[..., 0] -= positions[..., 0:1, 0].clone()  # must clone in the torch implementation, otherwise there is a weird glitch
    positions[..., 2] -= positions[..., 0:1, 2].clone()  # must clone in the torch implementation, otherwise there is a weird glitch
    '''All pose face Z+'''
    positions = qrot(torch.repeat_interleave(r_rot[:, :, None], positions.shape[2], axis=2), positions)
    return positions

def get_quaternion(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

    '''Fix Quaternion Discontinuity'''
    quat_params = qfix(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    quat_params[1:, 0] = r_velocity
    # (seq_len, joints_num, 4)
    return quat_params, r_velocity, velocity, r_rot

def get_quaternion_torch(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)

    # TODO - translate to torch!
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

    '''Fix Quaternion Discontinuity'''
    quat_params = qfix(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    quat_params[1:, 0] = r_velocity
    # (seq_len, joints_num, 4)
    return quat_params, r_velocity, velocity, r_rot

def get_cont6d_params(positions, n_raw_offsets, kinematic_chain, face_joint_indx):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d_np(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, 0].copy()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot_np(r_rot[1:], velocity)
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot

def get_cont6d_params_torch(positions, n_raw_offsets, kinematic_chain, face_joint_indx, fix_ik_bug):
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (seq_len, joints_num, 4)
    # TODO - translate to torch!
    bs, n_frames, n_joints, n_dim = positions.shape
    # quat_params = skel.inverse_kinematics_np(positions.reshape(-1, n_joints, n_dim).cpu().numpy(), face_joint_indx, smooth_forward=True)
    quat_params = skel.inverse_kinematics_np(positions.reshape(-1, n_joints, n_dim).cpu().numpy(), face_joint_indx, smooth_forward=False, fix_bug=fix_ik_bug)  # TODO - cannot smooth with multi-batch in axis 0
    quat_params = torch.from_numpy(quat_params).reshape(bs, n_frames, n_joints, -1).float().to(positions.device)

    '''Quaternion to continuous 6D'''
    cont_6d_params = quaternion_to_cont6d(quat_params)
    # (seq_len, 4)
    r_rot = quat_params[:, :, 0].contiguous().clone()
    #     print(r_rot[0])
    '''Root Linear Velocity'''
    # (seq_len - 1, 3)
    velocity = (positions[:, 1:, 0] - positions[:, :-1, 0]).clone()  # root joint velocity
    #     print(r_rot.shape, velocity.shape)
    velocity = qrot(r_rot[:, 1:], velocity)  # rotate the root joint velocity according to the direction of the root joint
    '''Root Angular Velocity'''
    # (seq_len - 1, 4)
    r_velocity = qmul(r_rot[:, 1:], qinv(r_rot[:, :-1]))  # root joint angular velocity
    # (seq_len, joints_num, 4)
    return cont_6d_params, r_velocity, velocity, r_rot
