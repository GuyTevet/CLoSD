import numpy as np
import torch
import scipy.interpolate as interpolate
from scipy.spatial.transform import Rotation as sRot
# from T2M_GPT.t2m_utils.motion_process import recover_from_ric, extract_features_t2m, recover_root_rot_pos
# from T2M_GPT.t2m_utils.quaternion import qrot, qinv
from closd.diffusion_planner.data_loaders.humanml.common.quaternion import qrot, qinv
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process import recover_from_ric, recover_root_rot_pos
from closd.diffusion_planner.data_loaders.humanml.scripts.motion_process_torch import extract_features_t2m

from pytorch3d import transforms 
import time

# DEBUG - start
# diff_linear = []
# diff_lin_high = []
# diff_bicubic = []
# DEBUG - end

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]
# [0, 1, 2, 3, 4,  5, 6, 7, 8,  9, 10,11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
mujoco_2_smpl = [0, 1, 5, 9, 2,  6,10, 3, 7, 11,  4, 8,12, 14, 19, 13, 15, 20, 16, 21, 17, 22, 18, 23]


def angle_to_2d_rotmat(angle):
    # angle [bs] (rad)
    # rotmat [bs, 2, 2]
    _angle = angle[:, None]
    return torch.cat([torch.cat([torch.cos(_angle), -torch.sin(_angle)], dim=-1)[:, :, None], 
                      torch.cat([torch.sin(_angle),  torch.cos(_angle)], dim=-1)[:, :, None]], dim=-1)


def area_vectorized(x1, y1, x2, y2, x3, y3):
    return torch.abs((x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))/2.0)

def is_point_in_rectangle_vectorized(points, rect_corners):
    """
    Check if points are inside rectangles using vectorized operations.
    
    Args:
    points: torch.Tensor of shape [bs, 2] representing batched 2D points
    rect_corners: torch.Tensor of shape [bs, 4, 2] representing batched rectangle corners
    
    Returns:
    torch.Tensor of shape [bs] with boolean values indicating if each point is inside its corresponding rectangle
    """
    
    # Unpack points and rectangle corners
    x, y = points[:, 0], points[:, 1]
    x1, y1 = rect_corners[:, 0, 0], rect_corners[:, 0, 1]
    x2, y2 = rect_corners[:, 1, 0], rect_corners[:, 1, 1]
    x3, y3 = rect_corners[:, 2, 0], rect_corners[:, 2, 1]
    x4, y4 = rect_corners[:, 3, 0], rect_corners[:, 3, 1]
    
    # Calculate area of rectangles
    rect_area = area_vectorized(x1, y1, x2, y2, x3, y3) + area_vectorized(x1, y1, x4, y4, x3, y3)
    
    # Calculate areas of triangles formed by the points and rectangle edges
    area1 = area_vectorized(x, y, x1, y1, x2, y2)
    area2 = area_vectorized(x, y, x2, y2, x3, y3)
    area3 = area_vectorized(x, y, x3, y3, x4, y4)
    area4 = area_vectorized(x, y, x4, y4, x1, y1)
    
    # Check if sum of areas of triangles equals rectangle area
    return torch.abs(rect_area - (area1 + area2 + area3 + area4)) < 1e-6

class RepresentationHandler:

    def __init__(self, mean, std, init_humanoid_root_pose, time_prints=True):
        self.mean = mean
        self.std = std
        self.offset_height = 0.92  # FIXME - Hardcoded
        # self.smpl2sim_rot_mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
        self.to_isaac_mat = torch.from_numpy(sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()).cuda().float()
        self.smpl2sim_rot_mat = torch.matmul(self.to_isaac_mat, self.to_isaac_mat)  # weird way to do it
        # self.smpl2sim_rot_mat = self.to_isaac_mat.dot(self.to_isaac_mat)  # weird way to do it
        self.y180_rot = torch.from_numpy(sRot.from_euler('xyz', np.array([0, -np.pi, 0]), degrees=False).as_matrix()).cuda().float()
        self.offset = 0.0  # None # + self.offset_height - body_pos[0, 0, 0, 2].numpy()
        # self.init_humanoid_root_pose = init_humanoid_root_pose #[bs, 3] #  torch.tensor([-0.1443, -0.2548,  0.9356]).cuda()  # hardcoded
        # self.smpl_end_effectors_idx = [8, 10, 22, 23]
        self.smpl_end_effectors_idx = [22, 23]  # rely on hands only
        self.time_prints = time_prints


    def hml_to_pose(self, pose, recon_data, sim_at_hml_idx):
        # hml rep [bs, n_frames_20fps, 263] -> smpl xyz [bs, n_frames_30fps, 24, 3]
        start_time = time.time()
        unnormed_pose = (pose * self.std + self.mean).float()
        hml_xyz = recover_from_ric(unnormed_pose, 22)
        
        r_rot_quat, r_pos = recover_root_rot_pos(unnormed_pose)
        hml_transform_at_sim = {'r_rot': r_rot_quat[:, sim_at_hml_idx], 'r_pos': r_pos[:, sim_at_hml_idx]}
        zeroed_hml_pose = self.align_to_recon_data(hml_xyz, hml_transform_at_sim) 
        aligned_hml_xyz = self.align_to_recon_data(zeroed_hml_pose, recon_data, is_inverse=True)

        pose_20fps = self.smpl_to_sim(self.xyz_to_full_smpl(aligned_hml_xyz))
        pose_30fps = self.fps_convert(pose_20fps, 20, 30, interp_mode='bicubic')
        # assert (self.fps_convert(pose_30fps, 30, 20) - pose_20fps).abs().max() < 0.01  # TEST
        if self.time_prints:
            print('=== hml_to_pose for [{}] envs took [{:.2f}] sec'.format(pose.shape[0], time.time() - start_time))

        return pose_30fps


    def pose_to_hml(self, pose, aux_ponts=None, fix_ik_bug=False):
        # pose [bs, seqlen, 24, 3]
        # aux_ponts [bs, n_points, 3] points to be translated to hml egocentric space
        # print('aux_ponts original: \n', aux_ponts)
        start_time = time.time()
        
        # add a dummy last frame to pose, because last frame is truncated during orig_pose_to_hml
        next_frame = pose[:, [-1]] + (pose[:, [-1]] - pose[:, [-2]])
        pose = torch.cat([pose, next_frame], dim=1)
        
        orig_pose, aux_ponts_hml = self.pose_to_hml_xyz(pose, aux_ponts)
        hml_pose, recon_data = self.orig_pose_to_hml(orig_pose, fix_ik_bug=fix_ik_bug) # add back bs placeholder

        if aux_ponts is not None:
            translated_aux_ponts = self.align_to_recon_data(aux_ponts_hml, recon_data)
            # ensure new code has the same functionality as the old one (delete sometimes in the near future)
            assert (translated_aux_ponts - self.align_to_recon_data_old(aux_ponts_hml, recon_data) < 1e-6).all()  
        else:
            translated_aux_ponts = None
        # print(recon_data)
        normed_hml_pose = (hml_pose - self.mean.to(hml_pose.device)) / self.std.to(hml_pose.device)
        if self.time_prints:
            print('=== pose_to_hml for [{}] envs took [{:.2f}] sec'.format(pose.shape[0], time.time() - start_time))

        return normed_hml_pose, translated_aux_ponts, recon_data

    def align_to_recon_data_old(self, aux_ponts, recon_data):
        # print('aux_ponts before: \n', aux_ponts)
        aux_ponts[..., 0] -= recon_data['r_pos'][..., [0]]
        aux_ponts[..., 2] -= recon_data['r_pos'][..., [2]]
        '''All pose face Z+'''
        aux_ponts = qrot(torch.repeat_interleave(recon_data['r_rot'][:, None], aux_ponts.shape[1], axis=1), aux_ponts)
        # print('aux_ponts after: \n', aux_ponts)
        return aux_ponts
    
    def align_to_recon_data(self, points, recon_data, is_inverse=False):
        # print('aux_ponts before: \n', aux_ponts)
        
        points = points.clone()  # to avoid in-place operations
        r_rot = recon_data['r_rot']
        r_pos = recon_data['r_pos']
        if is_inverse:
            r_rot = qinv(r_rot)
        for _ in range(points.dim()-2):  # expand according to the dimention of points
            r_rot = r_rot.unsqueeze(1)
            r_pos = r_pos.unsqueeze(1)
        
        new_rot_shape = (r_rot.shape[:1] + points.shape[1:-1] + r_rot.shape[-1:])

        if is_inverse:
            # points / rot + pos
            points = qrot(r_rot.expand(new_rot_shape), points)  
            points[..., [0,2]] += r_pos[..., [0,2]]                      
        else:
            # (points - pos) * rot
            points[..., [0,2]] -= r_pos[..., [0,2]]                      
            points = qrot(r_rot.expand(new_rot_shape), points)  # All pose face Z+
            
        # print('points after: \n', points)
        return points
    
    def inverse_align_to_recon_data(self, points, recon_data):
        # recon_data is the transformation from the isaac to the hml space
        # to get back to the isaac space we need to invert the transformation, i.e., multiply by the inverse roataion and add the inverse translation
        
        rot = qinv(recon_data['r_rot'])
        r_pos = recon_data['r_pos']
        for _ in range(points.dim()-2):  # expand rot according to #frames and #joints (if the latter exists)
            rot = rot.unsqueeze(1)
            r_pos = r_pos.unsqueeze(1)
        new_rot_shape = (rot.shape[:1] + points.shape[1:-1] + rot.shape[-1:])
        points = qrot(rot.expand(new_rot_shape), points)

        points[..., [0,2]] += r_pos[..., [0,2]] 

        return points
    
    def pose_to_hml_xyz(self, pose, aux_ponts=None):
        # pose [bs, seqlen, 24, 3]
        # aux_ponts [bs, n_points, 3] points to be translated to hml egocentric space
        pose_20fps = self.fps_convert(pose, 30, 20, interp_mode='bicubic')
        xyz_pose, translated_aux_ponts = self.sim_to_xyz(pose_20fps, aux_ponts)
        smpl_pose, translated_aux_ponts = self.xyz_to_smpl(xyz_pose, translated_aux_ponts)
        orig_pose = self.smpl_to_orig_pose(smpl_pose)
        
        return orig_pose, translated_aux_ponts
    
    def sim_to_xyz(self, ref_rb_pos, aux_ponts=None):
        #inverting json_data -> ref_rb_pos in HumanoidImMCPDemo._compute_task_obs_demo()
        # without scaling and smoothing
        # torch xyz positions [bs, n_frames,  24, 3] -> [bs, n_frames, 24, 3]
        json_data = torch.matmul(ref_rb_pos, self.to_isaac_mat.to(ref_rb_pos.device))
        json_data = json_data[:, :, mujoco_2_smpl]      
        if aux_ponts is not None:
            translated_aux_ponts = torch.matmul(aux_ponts, self.to_isaac_mat)
        else:
            translated_aux_ponts = None
        return json_data, translated_aux_ponts
    
    
    def xyz_to_smpl(self, xyz_pose, aux_ponts=None):
        # inverting smpl_to_sim() - ofseet and rtoate
        # torch xyz positions [bs, n_frames,  24, 3] -> [bs, n_frames, 24, 3]
        smpl_pose = xyz_pose  # .copy()
        smpl_pose[..., 1] -= self.offset
        smpl_pose = torch.matmul(smpl_pose, self.y180_rot.T.to(smpl_pose.device))  # Guy's addition - to avoid the initial 180 deg turn at the begining of the sim
        smpl_pose = torch.matmul(smpl_pose, self.smpl2sim_rot_mat.T.to(smpl_pose.device))
        if aux_ponts is not None:
            translated_aux_ponts = aux_ponts.clone()
            translated_aux_ponts[..., 1] -= self.offset
            translated_aux_ponts = torch.matmul(translated_aux_ponts, self.y180_rot.T)  # Guy's addition - to avoid the initial 180 deg turn at the begining of the sim
            translated_aux_ponts = torch.matmul(translated_aux_ponts, self.smpl2sim_rot_mat.T)
        else:
            translated_aux_ponts = None
        return smpl_pose, translated_aux_ponts
    
    def smpl_to_orig_pose(self, smpl_pose):
        # inverting xyz_to_full_smpl()
        # [bs, seqlen, 24, 3] -> [bs, seqlen, 22, 3]
        return smpl_pose[:, :, :-2]
    
    def orig_pose_to_hml(self, orig_pose, fix_ik_bug=False):
        # inverting recover_from_ric()
        # [bs, seqlen, 22, 3] -> [bs, seqlen, 263]
        return extract_features_t2m(orig_pose, fix_ik_bug=fix_ik_bug)
    
    def xyz_to_full_smpl(self, pred_xyz):
        # adding hands heuristically to HML 22 joints prediction
        # [1, seqlen, 22, 3] -> [seqlen, 24, 3]
        hand_len = 0.08824
        # mdm_jts = pred_xyz.cpu().detach().numpy()  # mdm_jts:  (1, 4, 22, 3)
        mdm_jts = pred_xyz
        
        direction = (mdm_jts[...,  -2, :] - mdm_jts[...,  -4, :])
        left = mdm_jts[...,  -2, :] + direction/torch.linalg.norm(direction, dim=-1, keepdim=True) * hand_len
        direction = (mdm_jts[...,  -1, :] - mdm_jts[...,  -3, :])
        right = mdm_jts[...,  -1, :] + direction/torch.linalg.norm(direction, dim=-1, keepdim=True) * hand_len
        mdm_jts_smpl_24 = torch.cat([mdm_jts, left[...,  None, :], right[..., None, :]], axis = -2)
        
        return mdm_jts_smpl_24  # .squeeze()
    
    def smpl_to_sim(self, smpl_xyz):
        
        sim_xyz = torch.matmul(smpl_xyz, self.smpl2sim_rot_mat)
        sim_xyz = torch.matmul(sim_xyz, self.y180_rot)  # Guy's additin - to avoid the initial 180 deg turn at the begining of the sim
        # FIXME - might be better to avoid this and just start with the same init state
        
        # offset = - self.offset_height - sim_xyz[ 0:1, 0:1, 1]
        # sim_xyz[..., 1] += offset
        
        # add offset
        # if self.offset is None:
        #     # reset offset
        #     self.offset = - self.offset_height - sim_xyz[0, 0, 1]
        sim_xyz[..., 1] += self.offset

        # # Guy's addition - the hml transform poses the character at the origin, hence the xz root position needs to be added back:
        # use coordinate_poses instead
        # sim_xyz[..., 0] += self.init_humanoid_root_pose[:, 0][:, None, None]
        # sim_xyz[..., 2] += self.init_humanoid_root_pose[:, 1][:, None, None]
        
        return sim_xyz
    
    def coordinate_poses(self, pred_poses, cur_pose):

        cur_root = cur_pose[:, 0]

        # substruct predicted root pose - locate the 1st frame at the origin
        pred_poses[..., 0] -= pred_poses[:, 0, 0, 0].clone()[:, None, None]
        pred_poses[..., 2] -= pred_poses[:, 0, 0, 2].clone()[:, None, None]

        # calculate the angle between the first pred pose and the current pose
        first_pred_pose = pred_poses[:, 0].clone()  # SMPL
        ref_pose = cur_pose.clone()  # [..., [0, 2, 1]][:, mujoco_2_smpl]
        # ref_pose[..., 1] *= -1.
        ref_ee_vecs = ref_pose[:, self.smpl_end_effectors_idx] - ref_pose[:, [0]]
        ref_ee_vecs = ref_ee_vecs[..., [0,2]]  # in the xz plane
        pred_ee_vecs = first_pred_pose[:, self.smpl_end_effectors_idx] - first_pred_pose[:, [0]]
        pred_ee_vecs = pred_ee_vecs[..., [0,2]]  # in the xz plane
        inner_product = (pred_ee_vecs * ref_ee_vecs).sum(dim=-1)
        ref_norm = torch.linalg.vector_norm(ref_ee_vecs, dim=-1)
        pred_norm = torch.linalg.vector_norm(pred_ee_vecs, dim=-1)
        cos = inner_product / (ref_norm * pred_norm)
        angle = torch.acos(cos)  #  [n_env, 4] (rad)

        # rotate to coordinate the two poses        
        avg_ang = - angle.mean(dim=-1)[:, None]  # [n_env, 1]
        avg_euler = torch.cat([torch.zeros_like(avg_ang), avg_ang, torch.zeros_like(avg_ang)], dim=-1)  # [n_env, 3]
        avg_quat = transforms.matrix_to_quaternion(transforms.euler_angles_to_matrix(avg_euler, convention='XYZ'))
        pred_poses = qrot(avg_quat[:, None, None].repeat_interleave(pred_poses.shape[1], dim=1).repeat_interleave(pred_poses.shape[2], dim=2), pred_poses)

        # add real root pose
        pred_poses[..., 0] += cur_root[:, 0].clone()[:, None, None]
        pred_poses[..., 2] += cur_root[:, 2].clone()[:, None, None]

        return pred_poses

    
    def fps_convert(self, mdm_jts, src_fps, trg_fps, interp_mode='bicubic'):
        # torch xyz positions [bs, n_frames,  n_joints, 3] -> [bs, n_frames*src_fps//trg_fp,  n_joints, 3]
        bs, n_frames,  n_joints, n_dim = mdm_jts.shape
        fps_ratio = trg_fps / src_fps
        n_frames_out = int(round(n_frames * fps_ratio))
        if interp_mode == 'linear':
            interp_in = mdm_jts.reshape(bs, n_frames, -1).permute(0, 2, 1)
            interp_out = torch.nn.functional.interpolate(
                interp_in, size=n_frames_out, mode=interp_mode)
            return interp_out.permute(0, 2, 1).reshape(bs, -1, n_joints, n_dim)
        else: # bicubic
            assert interp_mode == 'bicubic'
            interp_in = mdm_jts.reshape(bs, n_frames, -1).permute(0, 2, 1).unsqueeze(-1)
            interp_out = torch.nn.functional.interpolate(
                interp_in, size=(n_frames_out,1), mode=interp_mode)
            return interp_out.squeeze(-1).permute(0, 2, 1).reshape(bs, -1, n_joints, n_dim)

    
    def sim_pose_to_ref_pose_old(self, sim_pose):
        # must be a cleaner way to do it (without doing all the way through the hml rep)
        # sim_pose [bs, 24, 3]
        ref_xyz = sim_pose[..., [0, 2, 1]][:, mujoco_2_smpl]
        ref_xyz[..., 1] *= -1.
        return ref_xyz

    def sim_pose_to_ref_pose(self, sim_pose):
        # must be a cleaner way to do it (without doing all the way through the hml rep)
        # sim_pose [bs, 24, 3]
        ref_xyz = sim_pose[..., [0, 2, 1]][..., mujoco_2_smpl, :]
        ref_xyz[..., 1] *= -1.
        return ref_xyz
    
    def ref_pose_to_sim_pose(self, ref_pose):
        sim_pose = ref_pose.clone()
        sim_pose[..., 1] *= -1.
        sim_pose = sim_pose[..., [0, 2, 1]][..., smpl_2_mujoco, :]  # FIXME: double check whether smlpl_2_mujoco is required
        return sim_pose